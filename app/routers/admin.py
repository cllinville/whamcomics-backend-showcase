from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.requests import Request
from sqlalchemy.orm import Session
from uuid import UUID
import csv
import io
import json
import threading
from datetime import datetime, timedelta, timezone

from app import models
from app.config import settings
from app.db import get_db
from app.s3 import download_bytes, upload_bytes, generate_presigned_get_url
from app.services.book_generation import (
    generate_full_book_task,
    rebuild_pdf_from_panels,
    _send_book_ready_email,
    _send_print_order_email,
    _send_print_fulfillment_notification,
    _send_admin_notification_email,
)
import app.services.llamagen_service as llamagen_service
import app.services.lulu_service as lulu_service

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_secret: str | None = Header(default=None)):
    if not x_admin_secret or x_admin_secret != settings.secret_key:
        raise HTTPException(status_code=403, detail="Forbidden")


def _require_admin_secret(secret: str = "", x_admin_secret: str | None = Header(default=None)):
    if (secret or x_admin_secret) != settings.secret_key:
        raise HTTPException(status_code=403, detail="Forbidden")


def _build_llamagen_params(project_id: UUID, db: Session) -> dict:
    """Reconstruct LlamaGen call parameters from DB for a project."""
    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == project_id, models.Character.role == "main")
        .first()
    )
    if not main or not main.photo_keys:
        raise HTTPException(400, "No main character with photo found")

    photo_url = generate_presigned_get_url(main.photo_keys[0], expires_in=86400)
    age = (main.personalization or {}).get("age") or main.age
    gender = (main.personalization or {}).get("character_gender") or (main.traits or {}).get("appearance", {}).get("gender", "")
    app = (main.traits or {}).get("appearance", {})
    notes_parts = []
    if age:
        notes_parts.append(f"{age}-year-old")
    if gender:
        notes_parts.append(gender)
    if app.get("glasses"):
        notes_parts.append(f"wears {app['glasses']} — always include glasses")
    if app.get("hair"):
        notes_parts.append(app["hair"])
    if app.get("skin"):
        notes_parts.append(app["skin"])
    appearance_notes = ", ".join(notes_parts)

    # Build appearance anchor (detailed lock from traits)
    anchor_parts = []
    if app.get("skin"):
        anchor_parts.append(app["skin"] + " skin tone")
    if app.get("hair"):
        anchor_parts.append(app["hair"])
    if app.get("eyes"):
        anchor_parts.append(app["eyes"] + " eyes")
    if app.get("build"):
        anchor_parts.append(app["build"] + " build")
    if app.get("glasses"):
        anchor_parts.append(f"wears {app['glasses']}")
    appearance_anchor = ", ".join(anchor_parts)

    support_chars = (
        db.query(models.Character)
        .filter(models.Character.project_id == project_id, models.Character.role != "main")
        .limit(2)
        .all()
    )
    extra_roles = []
    for sc in support_chars:
        if sc.photo_keys:
            sc_url = generate_presigned_get_url(sc.photo_keys[0], expires_in=86400)
            sc_role: dict = {"name": sc.name, "image": sc_url}
            sc_age = sc.age or (sc.personalization or {}).get("age")
            sc_gender = (sc.personalization or {}).get("character_gender") or (sc.traits or {}).get("appearance", {}).get("gender", "")
            if sc_age:
                sc_role["age"] = int(sc_age)
            if sc_gender:
                sc_role["gender"] = sc_gender
            extra_roles.append(sc_role)

    return {
        "photo_url": photo_url,
        "character_name": main.name,
        "character_age": int(age) if age else None,
        "character_gender": gender,
        "appearance_notes": appearance_anchor or appearance_notes,
        "extra_roles": extra_roles,
    }


# ── Debug: LlamaGen fixPanelNum > 1 structure probe ─────────────────────────
# Temporary investigation endpoint -- fixPanelNum has always been hardcoded to
# 1 in llamagen_service.py, so it's never been confirmed what a batched
# request (fixPanelNum > 1) actually returns: separate per-panel assetUrls,
# or one combined multi-panel page image. That determines whether batching
# panels from the same scene is even usable in a one-image-per-page pipeline.
# Costs real LlamaGen credits per call -- not zero-cost like the OpenAI-only
# story debug endpoints. Remove once the batching question is settled.
@router.post("/debug/llamagen-batch-test", dependencies=[Depends(_require_admin)])
def llamagen_batch_test(project_id: UUID, fix_panel_num: int = 3, generation_id: str = None, size: str = "1024x768", submit_only: bool = False, db: Session = Depends(get_db)):
    import app.services.llamagen_service as llamagen_service

    if submit_only:
        # Fast path: submit and return immediately without polling, so a bad
        # size value shows up as a quick 4xx from the submit call instead of
        # waiting through a full generation (or a Railway/gateway timeout)
        # just to find out the request was rejected.
        params = _build_llamagen_params(project_id, db)
        hero = params["character_name"]
        payload = {
            "prompt": f"[Character: {hero}] standing, comic book illustration.",
            "fixPanelNum": 1,
            "size": size,
            "preset": settings.llamagen_preset,
            "comicRoles": [{"name": hero, "image": params["photo_url"]}],
        }
        import httpx
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.llamagen.ai/v1/comics/generations",
                json=payload,
                headers={"Authorization": f"Bearer {settings.llamagen_api_key}", "Content-Type": "application/json"},
            )
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None
        return {
            "submitted": resp.status_code < 400,
            "status_code": resp.status_code,
            "size": size,
            "response_body": resp_json if resp_json is not None else resp.text,
        }

    if generation_id:
        # Re-fetch and re-host an already-completed generation instead of
        # spending more credits on a new one.
        raw = llamagen_service.debug_poll_raw(generation_id)
        rehosted = []
        for comic in raw.get("comics") or []:
            for panel in comic.get("panels") or []:
                asset_url = panel.get("assetUrl")
                if not asset_url:
                    continue
                try:
                    img_bytes = llamagen_service.download_asset(asset_url)
                    key = f"debug/llamagen-batch-test/{generation_id}/panel-{panel.get('panel')}.png"
                    upload_bytes(key=key, data=img_bytes, content_type="image/png")
                    rehosted.append({"panel": panel.get("panel"), "url": generate_presigned_get_url(key, expires_in=7200)})
                except Exception as e:
                    rehosted.append({"panel": panel.get("panel"), "error": str(e)})
        return {"generation_id": generation_id, "panels": rehosted, "reused_existing": True}

    params = _build_llamagen_params(project_id, db)
    hero = params["character_name"]

    prompt = (
        "⚠️ ABSOLUTE RULE #1 — ZERO TOLERANCE: NO speech bubbles, no dialogue boxes, no text of any kind.\n\n"
        f"MANDATORY — {hero.upper()} MUST LOOK EXACTLY LIKE THIS IN EVERY PANEL: {params['appearance_notes']}. "
        f"Match the reference photo precisely, unchanged across every panel.\n\n"
        f"This is a {fix_panel_num}-panel sequence, all part of ONE continuous scene: {hero} is driving a car "
        f"down a highway with a passenger in the front seat. "
        + " ".join(
            f"Panel {i+1}: {hero} glances at the road, hands on the wheel, driving position unchanged from the previous panel."
            for i in range(fix_panel_num)
        )
        + "\n\nSTYLE: comic book illustration — bold ink linework, cel-shaded coloring, vivid saturated flat colors. "
        "Illustrated comic art, not a photograph."
    )

    payload = {
        "prompt": prompt,
        "fixPanelNum": fix_panel_num,
        "size": size,
        "preset": settings.llamagen_preset,
        "comicRoles": [
            {"name": hero, "image": params["photo_url"]}
        ],
    }

    generation_id = llamagen_service._submit(payload)
    raw = llamagen_service.debug_poll_raw(generation_id)

    # Re-host each panel's assetUrl via our own S3 -- LlamaGen's CDN URLs
    # require the same bearer auth as the API, so they aren't viewable
    # directly without it.
    rehosted = []
    for comic in raw.get("comics") or []:
        for panel in comic.get("panels") or []:
            asset_url = panel.get("assetUrl")
            if not asset_url:
                continue
            try:
                img_bytes = llamagen_service.download_asset(asset_url)
                key = f"debug/llamagen-batch-test/{generation_id}/panel-{panel.get('panel')}.png"
                upload_bytes(key=key, data=img_bytes, content_type="image/png")
                from PIL import Image
                from io import BytesIO
                actual_size = Image.open(BytesIO(img_bytes)).size
                rehosted.append({
                    "panel": panel.get("panel"),
                    "url": generate_presigned_get_url(key, expires_in=7200),
                    "actual_pixel_size": f"{actual_size[0]}x{actual_size[1]}",
                })
            except Exception as e:
                rehosted.append({"panel": panel.get("panel"), "error": str(e)})

    return {"generation_id": generation_id, "fix_panel_num": fix_panel_num, "requested_size": size, "panels": rehosted}


# ── Existing endpoints ────────────────────────────────────────────────────────

_STALE_GENERATION_MINUTES = 30


@router.post("/projects/{project_id}/retry-book", dependencies=[Depends(_require_admin)])
def retry_book_generation(project_id: UUID, force: bool = False, db: Session = Depends(get_db)):
    """Manually retry full book generation for a failed or stuck project.

    `force=true` allows regenerating a project that's already book_ready --
    needed when a book completed successfully but a manual quality review
    still found it unacceptable (e.g. a dropped plot thread) and the fix
    isn't something recompositing the existing panels can address.

    `force=true` also allows regenerating a project stuck in generating_book
    IF it's been that way for over _STALE_GENERATION_MINUTES -- a container
    restart (deploy, crash, OOM) mid-generation kills the background thread
    with no chance to hit the except-block that would otherwise mark it
    failed, leaving the project permanently stuck with no other way to
    retry it. Still refuses if the generation looks recent, so this can't be
    used to fire a second generation on top of one that's actually running.
    """
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status == models.ProjectStatus.generating_book:
        last_update = project.updated_at or project.created_at
        stale = last_update is not None and (datetime.now(timezone.utc) - last_update) > timedelta(minutes=_STALE_GENERATION_MINUTES)
        if not (force and stale):
            if force and last_update:
                mins = (datetime.now(timezone.utc) - last_update).total_seconds() / 60
                raise HTTPException(status_code=409, detail=f"Book generation started {mins:.0f} min ago — still within the normal window, refusing to force a retry that might collide with one actually running. Wait until it's been stuck {_STALE_GENERATION_MINUTES}+ minutes.")
            raise HTTPException(status_code=409, detail="Book generation already in progress")

    if project.status == models.ProjectStatus.book_ready and not force:
        raise HTTPException(status_code=409, detail="Book already generated successfully — pass ?force=true to regenerate anyway")

    project.status = models.ProjectStatus.paid
    db.commit()

    threading.Thread(target=generate_full_book_task, args=(str(project_id),), daemon=True).start()

    return {"ok": True, "project_id": str(project_id), "message": "Book generation queued"}


@router.get("/customers", dependencies=[Depends(_require_admin)])
def export_customers(db: Session = Depends(get_db)):
    """Export all customer + project data for analytics."""
    projects = db.query(models.Project).order_by(models.Project.created_at.desc()).all()

    rows = []
    for p in projects:
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == p.id, models.Character.role == "main")
            .first()
        )
        support = (
            db.query(models.Character)
            .filter(models.Character.project_id == p.id, models.Character.role == "support")
            .first()
        )
        pers = (main.personalization or {}) if main else {}
        traits = ((main.traits or {}).get("appearance", {})) if main else {}
        sp = (support.personalization or {}) if support else {}

        rows.append({
            "project_id": str(p.id),
            "created_at": p.created_at.replace(tzinfo=timezone.utc).isoformat() if p.created_at else "",
            "customer_email": p.customer_email or "",
            "status": p.status.value if p.status else "",
            "product_type": p.product_type or "",
            "theme": p.theme or "",
            "genre": p.genre or "",
            "hero_name": main.name if main else "",
            "character_age": pers.get("age") or (main.age if main else ""),
            "character_gender": pers.get("character_gender") or traits.get("gender", ""),
            "favorite_activities": ", ".join(pers.get("favorite_activities") or []),
            "favorite_animal": pers.get("favorite_animal", ""),
            "personality_traits": ", ".join(pers.get("personality_traits") or []),
            "favorite_place": pers.get("favorite_place", ""),
            "dream": pers.get("dream", ""),
            "special_skill": pers.get("special_skill", ""),
            "catchphrase": pers.get("catchphrase", ""),
            "job": pers.get("job", ""),
            "expertise": pers.get("expertise", ""),
            "support_name": support.name if support else "",
            "support_relationship": sp.get("relationship_to_main", ""),
            "support_nickname_for_hero": sp.get("nickname_for_hero", ""),
            "support_hero_nickname_for_them": sp.get("hero_nickname_for_them", ""),
        })

    return rows


@router.get("/customers.csv")
def export_customers_csv(
    secret: str = "",
    x_admin_secret: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    if (secret or x_admin_secret) != settings.secret_key:
        raise HTTPException(status_code=403, detail="Forbidden")
    """Download all customer data as a CSV file."""
    projects = db.query(models.Project).order_by(models.Project.created_at.desc()).all()

    fieldnames = [
        "created_at", "customer_email", "status", "product_type",
        "theme", "genre", "hero_name", "character_age", "character_gender",
        "favorite_activities", "favorite_animal", "personality_traits",
        "favorite_place", "dream", "special_skill", "catchphrase",
        "job", "expertise",
        "support_name", "support_relationship",
        "support_nickname_for_hero", "support_hero_nickname_for_them",
        "project_id",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for p in projects:
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == p.id, models.Character.role == "main")
            .first()
        )
        support = (
            db.query(models.Character)
            .filter(models.Character.project_id == p.id, models.Character.role == "support")
            .first()
        )
        pers = (main.personalization or {}) if main else {}
        traits = ((main.traits or {}).get("appearance", {})) if main else {}
        sp = (support.personalization or {}) if support else {}

        writer.writerow({
            "project_id": str(p.id),
            "created_at": p.created_at.replace(tzinfo=timezone.utc).isoformat() if p.created_at else "",
            "customer_email": p.customer_email or "",
            "status": p.status.value if p.status else "",
            "product_type": p.product_type or "",
            "theme": p.theme or "",
            "genre": p.genre or "",
            "hero_name": main.name if main else "",
            "character_age": pers.get("age") or (main.age if main else ""),
            "character_gender": pers.get("character_gender") or traits.get("gender", ""),
            "favorite_activities": ", ".join(pers.get("favorite_activities") or []),
            "favorite_animal": pers.get("favorite_animal", ""),
            "personality_traits": ", ".join(pers.get("personality_traits") or []),
            "favorite_place": pers.get("favorite_place", ""),
            "dream": pers.get("dream", ""),
            "special_skill": pers.get("special_skill", ""),
            "catchphrase": pers.get("catchphrase", ""),
            "job": pers.get("job", ""),
            "expertise": pers.get("expertise", ""),
            "support_name": support.name if support else "",
            "support_relationship": sp.get("relationship_to_main", ""),
            "support_nickname_for_hero": sp.get("nickname_for_hero", ""),
            "support_hero_nickname_for_them": sp.get("hero_nickname_for_them", ""),
        })

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=whamcomics-customers.csv"},
    )


# ── Panel review endpoints ─────────────────────────────────────────────────────

@router.get("/projects/{project_id}/characters", dependencies=[Depends(_require_admin)])
def list_character_photos(project_id: UUID, db: Session = Depends(get_db)):
    """Return each character's uploaded photos + cartoon reference as viewable URLs.
    Works for any project regardless of paid/preview status — useful for checking
    what was actually uploaded before generation runs."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    characters = (
        db.query(models.Character)
        .filter(models.Character.project_id == project_id)
        .all()
    )

    out = []
    for c in characters:
        photo_urls = []
        for key in (c.photo_keys or []):
            try:
                photo_urls.append(generate_presigned_get_url(key, expires_in=3600))
            except Exception:
                photo_urls.append(None)

        reference_url = None
        if c.reference_image_key:
            try:
                reference_url = generate_presigned_get_url(c.reference_image_key, expires_in=3600)
            except Exception:
                reference_url = None

        out.append({
            "id": str(c.id),
            "name": c.name,
            "role": c.role,
            "photo_urls": photo_urls,
            "reference_url": reference_url,
        })

    return {"project_id": str(project_id), "characters": out}


@router.get("/projects/{project_id}/prose", dependencies=[Depends(_require_admin)])
def get_project_prose(project_id: UUID, db: Session = Depends(get_db)):
    """Return the persisted prose the panels were adapted from, for diagnosing
    whether a bad panel script was already bad in the prose or introduced
    during panel adaptation. Only exists for books generated after prose
    persistence was added."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    try:
        prose = download_bytes(f"panels/{project_id}/prose.txt").decode("utf-8")
    except Exception as e:
        raise HTTPException(404, f"No persisted prose for this project: {e}")
    return {"project_id": str(project_id), "prose": prose}


@router.get("/projects/{project_id}/panels", dependencies=[Depends(_require_admin)])
def list_panels(project_id: UUID, db: Session = Depends(get_db)):
    """Return all stored panels with presigned image URLs + dialogue/narration."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    try:
        dialogue_map = json.loads(download_bytes(f"panels/{project_id}/dialogue.json"))
    except Exception:
        dialogue_map = {}
    try:
        narration_map = json.loads(download_bytes(f"panels/{project_id}/narration.json"))
    except Exception:
        narration_map = {}
    try:
        prompts_map = json.loads(download_bytes(f"panels/{project_id}/prompts.json"))
    except Exception:
        prompts_map = {}

    panel_count = project.panel_count or 48
    panels = []
    for n in range(1, panel_count + 1):
        try:
            img_url = generate_presigned_get_url(f"panels/{project_id}/panel-{n}.png", expires_in=3600)
        except Exception:
            img_url = None
        panels.append({
            "panel_num": n,
            "image_url": img_url,
            "dialogue": dialogue_map.get(str(n), []),
            "narration": narration_map.get(str(n), ""),
            "prompt": prompts_map.get(str(n), ""),
        })

    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == project_id, models.Character.role == "main")
        .first()
    )

    try:
        cover_url = generate_presigned_get_url(f"panels/{project_id}/cover.png", expires_in=3600)
    except Exception:
        cover_url = None

    print_cover_pdf_url = None
    print_interior_pdf_url = None
    if project.book_cover_pdf_key:
        try:
            print_cover_pdf_url = generate_presigned_get_url(project.book_cover_pdf_key, expires_in=3600)
        except Exception:
            print_cover_pdf_url = None
    if project.book_interior_pdf_key:
        try:
            print_interior_pdf_url = generate_presigned_get_url(project.book_interior_pdf_key, expires_in=3600)
        except Exception:
            print_interior_pdf_url = None

    return {
        "project_id": str(project_id),
        "status": project.status.value,
        "genre": project.genre,
        "hero_name": main.name if main else "",
        "panel_count": panel_count,
        "cover_url": cover_url,
        "panels": panels,
        "product_type": project.product_type or "digital",
        "has_shipping_address": bool(project.shipping_address),
        "print_cover_pdf_url": print_cover_pdf_url,
        "print_interior_pdf_url": print_interior_pdf_url,
        "lulu_print_job_id": project.lulu_print_job_id,
        "lulu_print_job_status": project.lulu_print_job_status,
    }


@router.post("/projects/{project_id}/panels/{panel_num}/regenerate", dependencies=[Depends(_require_admin)])
def regenerate_panel(project_id: UUID, panel_num: int, db: Session = Depends(get_db)):
    """Re-run LlamaGen for a single panel and store the new image. Returns new presigned URL."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Load the stored prompt for this panel
    try:
        prompts_map = json.loads(download_bytes(f"panels/{project_id}/prompts.json"))
    except Exception:
        raise HTTPException(400, "No prompts.json found — was this project generated after the review feature was added?")

    image_prompt = prompts_map.get(str(panel_num), "")
    if not image_prompt:
        raise HTTPException(400, f"No stored prompt for panel {panel_num}")

    # Build LlamaGen params from DB
    lg_params = _build_llamagen_params(project_id, db)

    # Rebuild lg_scene (same adult tag as generation)
    lg_scene = image_prompt
    age = lg_params.get("character_age")
    if age and int(age) >= 18:
        lg_scene = "ALL CHARACTERS IN THIS SCENE ARE ADULTS (18+). No children or teenagers.\n\n" + lg_scene

    # Load dialogue to inject PANEL LAYOUT hint
    try:
        dialogue_map = json.loads(download_bytes(f"panels/{project_id}/dialogue.json"))
        panel_dialogue = dialogue_map.get(str(panel_num), [])
        if len(panel_dialogue) >= 2:
            s0 = (panel_dialogue[0].get("speaker") or "").strip()
            s1 = (panel_dialogue[1].get("speaker") or "").strip()
            if s0 and s1:
                lg_scene += f"\n\nPANEL LAYOUT: {s0} must be positioned on the LEFT side of the panel. {s1} must be positioned on the RIGHT side of the panel."
    except Exception:
        pass

    panel_bytes = llamagen_service.generate_panel_llamagen(
        scene_prompt=lg_scene,
        photo_url=lg_params["photo_url"],
        character_name=lg_params["character_name"],
        character_age=lg_params["character_age"],
        character_gender=lg_params["character_gender"],
        appearance_notes=lg_params["appearance_notes"],
        extra_roles=lg_params["extra_roles"],
    )

    # Store new panel PNG (overwrites old one)
    upload_bytes(
        key=f"panels/{project_id}/panel-{panel_num}.png",
        data=panel_bytes,
        content_type="image/png",
    )

    new_url = generate_presigned_get_url(f"panels/{project_id}/panel-{panel_num}.png", expires_in=3600)
    return {"ok": True, "panel_num": panel_num, "image_url": new_url}


@router.post("/projects/{project_id}/recomposite-book", dependencies=[Depends(_require_admin)])
def recomposite_book(project_id: UUID, db: Session = Depends(get_db)):
    """Rebuild the full PDF from stored panel PNGs without re-running AI."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    try:
        result = rebuild_pdf_from_panels(str(project_id))
    except Exception as e:
        raise HTTPException(500, f"Rebuild failed: {e}")

    # A rebuild that succeeds proves the book is reviewable again — clear a
    # prior failure so Approve isn't blocked, without touching other statuses.
    if project.status == models.ProjectStatus.book_failed:
        project.status = models.ProjectStatus.book_ready
        db.commit()

    return {"ok": True, "project_id": str(project_id), "status": project.status.value, **result}


@router.post("/projects/{project_id}/approve", dependencies=[Depends(_require_admin)])
def approve_and_deliver(project_id: UUID, db: Session = Depends(get_db)):
    """Send the delivery email to the customer. Call this after reviewing all panels."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if project.status not in (models.ProjectStatus.book_ready, models.ProjectStatus.delivered):
        raise HTTPException(400, f"Project status is '{project.status.value}' — must be book_ready or delivered to re-deliver")
    if not project.customer_email:
        raise HTTPException(400, "No customer email on file for this project")

    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == project_id, models.Character.role == "main")
        .first()
    )
    hero_name = main.name if main else "your hero"

    if project.product_type == "print":
        # Customer receipt already sent automatically when generation completed.
        # Approve submits the print job directly to Lulu — guard against ever
        # submitting a second real print job for the same project.
        if project.lulu_print_job_id:
            raise HTTPException(
                400,
                f"Print job already submitted to Lulu (job {project.lulu_print_job_id}, "
                f"status {project.lulu_print_job_status}). Refusing to submit a duplicate order.",
            )
        if not project.book_cover_pdf_key or not project.book_interior_pdf_key:
            raise HTTPException(400, "Print PDFs not found — has the book been generated?")
        if not project.shipping_address:
            raise HTTPException(400, "Shipping address missing — was a shipping address entered at Stripe checkout?")

        cover_url = generate_presigned_get_url(project.book_cover_pdf_key, expires_in=86400 * 7)
        interior_url = generate_presigned_get_url(project.book_interior_pdf_key, expires_in=86400 * 7)

        try:
            lulu_job = lulu_service.create_print_job(
                external_id=str(project.id),
                title=f"{hero_name}'s Comic Book",
                quantity=1,
                cover_url=cover_url,
                interior_url=interior_url,
                shipping=project.shipping_address,
                customer_email=project.customer_email,
            )
        except lulu_service.LuluAPIError as e:
            raise HTTPException(502, f"Failed to submit print job to Lulu: {e}")

        project.lulu_print_job_id = str(lulu_job.get("id"))
        project.lulu_print_job_status = (lulu_job.get("status") or {}).get("name", "CREATED")

        _send_print_fulfillment_notification(
            hero_name, project.customer_email, project.shipping_address,
            project.lulu_print_job_id, project.lulu_print_job_status,
        )
    else:
        if not project.book_pdf_key:
            raise HTTPException(400, "No PDF key found — has the book been generated?")
        download_url = generate_presigned_get_url(project.book_pdf_key, expires_in=86400 * 7)
        _send_book_ready_email(project.customer_email, hero_name, download_url)

    project.status = models.ProjectStatus.delivered
    db.commit()

    return {
        "ok": True,
        "delivered_to": project.customer_email,
        "lulu_print_job_id": project.lulu_print_job_id,
        "lulu_print_job_status": project.lulu_print_job_status,
    }


@router.post("/projects/{project_id}/shipping", dependencies=[Depends(_require_admin)])
async def set_shipping_address(project_id: UUID, request: Request, db: Session = Depends(get_db)):
    """Manually set or update the shipping address for a print order."""
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    body = await request.json()
    project.shipping_address = {
        "name":        body.get("name", ""),
        "line1":       body.get("line1", ""),
        "line2":       body.get("line2", ""),
        "city":        body.get("city", ""),
        "state":       body.get("state", ""),
        "postal_code": body.get("postal_code", ""),
        "country":     body.get("country", "US"),
    }
    db.commit()
    return {"ok": True, "shipping_address": project.shipping_address}


@router.post("/lulu/register-webhook", dependencies=[Depends(_require_admin)])
def register_lulu_webhook():
    """One-time setup: subscribe this backend to Lulu's PRINT_JOB_STATUS_CHANGED webhook."""
    url = f"{settings.base_url}/webhooks/lulu"
    try:
        result = lulu_service.register_webhook(url)
    except lulu_service.LuluAPIError as e:
        raise HTTPException(502, f"Failed to register Lulu webhook: {e}")
    return {"ok": True, "webhook": result}


# ── Admin dashboard ───────────────────────────────────────────────────────────

@router.get("/dashboard")
def dashboard_page(secret: str = "", db: Session = Depends(get_db)):
    if secret != settings.secret_key:
        return HTMLResponse("<h1>403 Forbidden</h1>", status_code=403)

    pending = (
        db.query(models.Project)
        .filter(models.Project.status == models.ProjectStatus.book_ready)
        .order_by(models.Project.created_at.desc())
        .all()
    )
    delivered = (
        db.query(models.Project)
        .filter(models.Project.status == models.ProjectStatus.delivered)
        .order_by(models.Project.created_at.desc())
        .limit(30)
        .all()
    )
    failed = (
        db.query(models.Project)
        .filter(models.Project.status == models.ProjectStatus.book_failed)
        .order_by(models.Project.created_at.desc())
        .limit(20)
        .all()
    )

    def _hero(project):
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == project.id, models.Character.role == "main")
            .first()
        )
        return main.name if main else "Unknown"

    def _row(p, status_label, status_color, action_label, action_href, action_color):
        hero = _hero(p)
        created = p.created_at.strftime("%b %d, %Y %H:%M") if p.created_at else "—"
        return f"""
        <tr>
          <td>{created}</td>
          <td><strong>{hero}</strong></td>
          <td>{p.customer_email or "—"}</td>
          <td>{p.product_type or "digital"}</td>
          <td><span style="color:{status_color};font-weight:600">{status_label}</span></td>
          <td><a href="{action_href}" style="color:{action_color};font-weight:600">{action_label}</a></td>
        </tr>"""

    pending_rows = "".join(
        _row(p, "Awaiting Review", "#ffd700",
             "Review →", f"/admin/review/{p.id}?secret={secret}", "#ffd700")
        for p in pending
    ) or "<tr><td colspan='6' style='color:#666;text-align:center;padding:20px'>No projects pending</td></tr>"

    delivered_rows = "".join(
        _row(p, "Delivered", "#22c55e",
             "View →", f"/admin/review/{p.id}?secret={secret}", "#4ade80")
        for p in delivered
    ) or "<tr><td colspan='6' style='color:#666;text-align:center;padding:20px'>No delivered projects yet</td></tr>"

    failed_rows = "".join(
        _row(p, "Failed", "#f55",
             "Review →", f"/admin/review/{p.id}?secret={secret}", "#f88")
        for p in failed
    ) or "<tr><td colspan='6' style='color:#666;text-align:center;padding:20px'>No failed projects</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhamComics Admin Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 28px; }}
  h1 {{ font-size: 1.5rem; color: #ffd700; margin-bottom: 6px; }}
  .subtitle {{ font-size: 0.85rem; color: #666; margin-bottom: 28px; }}
  h2 {{ font-size: 1rem; color: #ffd700; margin: 28px 0 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ text-align: left; color: #888; font-weight: 500; padding: 8px 12px; border-bottom: 1px solid #333; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #222; vertical-align: middle; }}
  tr:hover td {{ background: #1a1a1a; }}
  a {{ text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .badge-count {{ display: inline-block; background: #ffd700; color: #111; border-radius: 12px;
                  padding: 2px 10px; font-size: 0.8rem; font-weight: 700; margin-left: 8px; }}
</style>
</head>
<body>
<h1>WhamComics Admin Dashboard</h1>
<div class="subtitle">Logged in as admin &nbsp;·&nbsp; <a href="/admin/dashboard?secret={secret}" style="color:#555">Refresh</a></div>

<h2>Pending Review <span class="badge-count">{len(pending)}</span></h2>
<table>
  <thead><tr><th>Created</th><th>Hero</th><th>Customer</th><th>Type</th><th>Status</th><th>Action</th></tr></thead>
  <tbody>{pending_rows}</tbody>
</table>

<h2>Delivered <span class="badge-count" style="background:#22c55e">{len(delivered)}</span></h2>
<table>
  <thead><tr><th>Created</th><th>Hero</th><th>Customer</th><th>Type</th><th>Status</th><th>Action</th></tr></thead>
  <tbody>{delivered_rows}</tbody>
</table>

<h2>Recent Failures</h2>
<table>
  <thead><tr><th>Created</th><th>Hero</th><th>Customer</th><th>Type</th><th>Status</th><th>Action</th></tr></thead>
  <tbody>{failed_rows}</tbody>
</table>
</body>
</html>"""

    return HTMLResponse(html)


# ── HTML review page ───────────────────────────────────────────────────────────

@router.get("/review/{project_id}")
def review_page(project_id: str, secret: str = ""):
    if secret != settings.secret_key:
        return HTMLResponse("<h1>403 Forbidden</h1>", status_code=403)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhamComics Review — {project_id[:8]}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 20px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; color: #ffd700; }}
  .meta {{ font-size: 0.85rem; color: #888; margin-bottom: 20px; }}
  .toolbar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 24px; flex-wrap: wrap; }}
  button {{ padding: 8px 18px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; }}
  .btn-rebuild {{ background: #ffd700; color: #111; }}
  .btn-rebuild:disabled {{ background: #666; color: #999; cursor: not-allowed; }}
  .btn-approve {{ background: #22c55e; color: #fff; }}
  .btn-approve:disabled {{ background: #666; color: #999; cursor: not-allowed; }}
  .btn-retry {{ background: #3a1414; color: #f88; border: 1px solid #822; }}
  .btn-retry:disabled {{ background: #333; color: #777; border-color: #444; cursor: not-allowed; }}
  .status-msg {{ font-size: 0.85rem; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .panel-card {{ background: #1e1e1e; border-radius: 8px; overflow: hidden; border: 2px solid #333; }}
  .panel-card.flagged {{ border-color: #e55; }}
  .panel-card img {{ width: 100%; display: block; cursor: pointer; }}
  .panel-card img.loading {{ opacity: 0.4; }}
  .panel-info {{ padding: 10px 12px; }}
  .panel-num {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
  .panel-dialogue {{ font-size: 0.8rem; color: #ccc; margin-bottom: 8px; min-height: 2em; }}
  .panel-narration {{ font-size: 0.72rem; color: #888; margin-bottom: 8px; font-style: italic; }}
  .btn-regen {{ width: 100%; background: #333; color: #ffd700; border: 1px solid #555; border-radius: 4px; padding: 6px; font-size: 0.82rem; }}
  .btn-regen:hover {{ background: #444; }}
  .btn-regen:disabled {{ background: #222; color: #666; cursor: not-allowed; }}
  .cover-section {{ margin-bottom: 28px; display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
  .cover-section img {{ max-width: 200px; border-radius: 4px; border: 2px solid #555; }}
  #error {{ color: #f55; margin: 10px 0; font-size: 0.9rem; }}
  .section-title {{ font-size: 1rem; color: #ffd700; margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
</style>
</head>
<body>
<h1><a href="/admin/dashboard?secret={secret}" style="color:#ffd700;text-decoration:none;">WhamComics Panel Review</a></h1>
<div class="meta" id="meta">Loading project {project_id[:8]}…</div>
<div id="error"></div>

<div class="toolbar">
  <button class="btn-rebuild" id="btnRebuild" onclick="rebuildPdf()" disabled>Rebuild PDF</button>
  <button class="btn-approve" id="btnApprove" onclick="approveAndDeliver()" disabled>✓ Approve &amp; Send to Customer</button>
  <button class="btn-retry" id="btnRetry" onclick="retryFullGeneration()" disabled>⚠ Full Retry (uses AI credits)</button>
  <span class="status-msg" id="rebuildStatus"></span>
</div>

<div id="shippingSection" style="display:none;background:#1a1a1a;border:1px solid #f55;border-radius:8px;padding:16px;margin-bottom:20px;max-width:480px;">
  <div style="color:#f88;font-weight:600;margin-bottom:12px;">⚠️ Shipping address missing — enter it below before approving</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <input id="sName" placeholder="Full name" style="grid-column:1/-1;padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sLine1" placeholder="Address line 1" style="grid-column:1/-1;padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sLine2" placeholder="Line 2 (optional)" style="grid-column:1/-1;padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sCity" placeholder="City" style="padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sState" placeholder="State" style="padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sZip" placeholder="ZIP / Postal code" style="padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
    <input id="sCountry" placeholder="Country (e.g. US)" value="US" style="padding:7px 10px;background:#111;color:#eee;border:1px solid #444;border-radius:4px;">
  </div>
  <button onclick="saveShipping()" style="margin-top:10px;background:#ffd700;color:#111;border:none;border-radius:6px;padding:8px 18px;font-weight:600;cursor:pointer;">Save Address</button>
  <span id="shippingStatus" style="margin-left:10px;font-size:0.85rem;color:#aaa;"></span>
</div>

<div id="printPreviewSection" style="display:none;background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:20px;"></div>

<div class="cover-section" id="coverSection"></div>
<div class="section-title">Interior Panels</div>
<div class="grid" id="grid"></div>

<script>
const PROJECT_ID = "{project_id}";
const SECRET = "{secret}";
const HEADERS = {{ "x-admin-secret": SECRET }};
let IS_PRINT = false;
let LULU_JOB_ID = null;

async function load() {{
  try {{
    const res = await fetch(`/admin/projects/${{PROJECT_ID}}/panels`, {{ headers: HEADERS }});
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderPage(data);
  }} catch(e) {{
    document.getElementById("error").textContent = "Failed to load: " + e.message;
  }}
}}

function renderPage(data) {{
  document.getElementById("meta").textContent =
    `Project: ${{data.project_id}} | Status: ${{data.status}} | Genre: ${{data.genre}} | Hero: ${{data.hero_name}} | ${{data.panel_count}} panels`;

  if (data.cover_url) {{
    document.getElementById("coverSection").innerHTML =
      `<div><div class="section-title" style="margin-top:0">Cover</div><img src="${{data.cover_url}}" style="max-width:200px"></div>`;
  }}

  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  data.panels.forEach(p => {{
    const dlText = (p.dialogue || []).map(d => `${{d.speaker}}: "${{d.line || d.text || "??"}}"`).join("\\n") || "(no dialogue)";
    const narText = p.narration ? p.narration.slice(0, 100) + (p.narration.length > 100 ? "…" : "") : "";
    const card = document.createElement("div");
    card.className = "panel-card";
    card.id = `card-${{p.panel_num}}`;
    card.innerHTML = `
      <img src="${{p.image_url || ""}}" alt="Panel ${{p.panel_num}}" id="img-${{p.panel_num}}"
           onerror="this.style.background='#333';this.style.minHeight='160px'">
      <div class="panel-info">
        <div class="panel-num">Panel ${{p.panel_num}}</div>
        <div class="panel-dialogue" style="white-space:pre-line">${{dlText}}</div>
        ${{narText ? `<div class="panel-narration">${{narText}}</div>` : ""}}
        <button class="btn-regen" id="btn-${{p.panel_num}}" onclick="regen(${{p.panel_num}})">↺ Regenerate</button>
      </div>`;
    grid.appendChild(card);
  }});
  document.getElementById("btnRebuild").disabled = false;
  document.getElementById("btnApprove").disabled = false;
  document.getElementById("btnRetry").disabled = false;

  IS_PRINT = data.product_type === "print";
  LULU_JOB_ID = data.lulu_print_job_id || null;

  // Show shipping form if print order is missing address
  if (IS_PRINT && !data.has_shipping_address) {{
    document.getElementById("shippingSection").style.display = "block";
  }}

  if (IS_PRINT) {{
    const sec = document.getElementById("printPreviewSection");
    sec.style.display = "block";
    let html = `<div class="section-title" style="margin-top:0">Print Files — review before submitting to Lulu</div>`;

    if (LULU_JOB_ID) {{
      html += `<div style="color:#4ade80;font-weight:600;margin-bottom:12px;">
        ✓ Already submitted to Lulu — Job #${{LULU_JOB_ID}} (status: ${{data.lulu_print_job_status || "unknown"}}).
        Approve is disabled to avoid placing a duplicate order.
      </div>`;
      document.getElementById("btnApprove").disabled = true;
    }}

    if (data.print_cover_pdf_url || data.print_interior_pdf_url) {{
      html += `<div style="display:flex;gap:20px;flex-wrap:wrap;">`;
      if (data.print_cover_pdf_url) {{
        html += `<div>
          <div style="font-size:0.8rem;color:#888;margin-bottom:6px;">Cover (wrap) — <a href="${{data.print_cover_pdf_url}}" target="_blank" style="color:#ffd700;">open full size</a></div>
          <embed src="${{data.print_cover_pdf_url}}" type="application/pdf" style="width:340px;height:220px;border:1px solid #444;background:#fff;">
        </div>`;
      }}
      if (data.print_interior_pdf_url) {{
        html += `<div>
          <div style="font-size:0.8rem;color:#888;margin-bottom:6px;">Interior — <a href="${{data.print_interior_pdf_url}}" target="_blank" style="color:#ffd700;">open full size</a></div>
          <embed src="${{data.print_interior_pdf_url}}" type="application/pdf" style="width:260px;height:400px;border:1px solid #444;background:#fff;">
        </div>`;
      }}
      html += `</div>`;
    }} else {{
      html += `<div style="color:#888;">No print PDFs yet — click "Rebuild PDF" to generate the cover and interior files.</div>`;
    }}
    sec.innerHTML = html;
  }}
}}

async function saveShipping() {{
  const body = {{
    name:        document.getElementById("sName").value,
    line1:       document.getElementById("sLine1").value,
    line2:       document.getElementById("sLine2").value,
    city:        document.getElementById("sCity").value,
    state:       document.getElementById("sState").value,
    postal_code: document.getElementById("sZip").value,
    country:     document.getElementById("sCountry").value,
  }};
  document.getElementById("shippingStatus").textContent = "Saving...";
  try {{
    const res = await fetch(`/admin/projects/${{PROJECT_ID}}/shipping`, {{
      method: "POST", headers: {{ ...HEADERS, "Content-Type": "application/json" }},
      body: JSON.stringify(body)
    }});
    if (!res.ok) throw new Error(await res.text());
    document.getElementById("shippingStatus").textContent = "✓ Address saved";
    document.getElementById("shippingSection").style.borderColor = "#22c55e";
  }} catch(e) {{
    document.getElementById("shippingStatus").textContent = "Failed: " + e.message;
  }}
}}

async function regen(panelNum) {{
  const btn = document.getElementById(`btn-${{panelNum}}`);
  const img = document.getElementById(`img-${{panelNum}}`);
  btn.disabled = true;
  btn.textContent = "Generating…";
  img.classList.add("loading");
  document.getElementById("error").textContent = "";

  try {{
    const res = await fetch(
      `/admin/projects/${{PROJECT_ID}}/panels/${{panelNum}}/regenerate`,
      {{ method: "POST", headers: HEADERS }}
    );
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    img.src = data.image_url;
    img.classList.remove("loading");
    btn.textContent = "↺ Regenerate";
    btn.disabled = false;
  }} catch(e) {{
    document.getElementById("error").textContent = `Panel ${{panelNum}} failed: ${{e.message}}`;
    img.classList.remove("loading");
    btn.textContent = "↺ Regenerate";
    btn.disabled = false;
  }}
}}

async function approveAndDeliver() {{
  if (IS_PRINT && LULU_JOB_ID) {{
    alert(`Already submitted to Lulu as job #${{LULU_JOB_ID}} — refusing to submit a duplicate order.`);
    return;
  }}
  const confirmMsg = IS_PRINT
    ? "Approve & submit to Lulu? This will rebuild the cover + interior PDFs from the current panels (~2 min), then submit a REAL print job to Lulu (check LULU_BASE_URL — sandbox or production) using the shipping address on file. This cannot be undone from here."
    : "Approve & deliver? This will rebuild the PDF from the current panels (~2 min), then send the delivery email.";
  if (!confirm(confirmMsg)) return;
  const btn = document.getElementById("btnApprove");
  const btnRebuild = document.getElementById("btnRebuild");
  const status = document.getElementById("rebuildStatus");
  btn.disabled = true;
  if (btnRebuild) btnRebuild.disabled = true;
  document.getElementById("error").textContent = "";

  try {{
    // Step 1: always rebuild PDF from current panel PNGs before delivering
    status.textContent = "Step 1/2 — Rebuilding PDF from current panels… (~2 minutes)";
    const rebuildRes = await fetch(
      `/admin/projects/${{PROJECT_ID}}/recomposite-book`,
      {{ method: "POST", headers: HEADERS }}
    );
    if (!rebuildRes.ok) throw new Error("PDF rebuild failed: " + await rebuildRes.text());

    // Step 2: send delivery email
    status.textContent = "Step 2/2 — Sending delivery email…";
    const approveRes = await fetch(
      `/admin/projects/${{PROJECT_ID}}/approve`,
      {{ method: "POST", headers: HEADERS }}
    );
    if (!approveRes.ok) throw new Error(await approveRes.text());
    const data = await approveRes.json();
    status.textContent = data.lulu_print_job_id
      ? `✓ Submitted to Lulu — Job #${{data.lulu_print_job_id}} (${{data.lulu_print_job_status || "CREATED"}})`
      : `✓ PDF rebuilt and delivery email sent to ${{data.delivered_to}}`;
    btn.textContent = "✓ Delivered";
    LULU_JOB_ID = data.lulu_print_job_id || LULU_JOB_ID;
  }} catch(e) {{
    document.getElementById("error").textContent = "Delivery failed: " + e.message;
    status.textContent = "";
    btn.disabled = false;
    if (btnRebuild) btnRebuild.disabled = false;
  }}
}}

async function rebuildPdf() {{
  const btn = document.getElementById("btnRebuild");
  const status = document.getElementById("rebuildStatus");
  btn.disabled = true;
  status.textContent = "Rebuilding PDF… (this takes ~2 minutes)";
  document.getElementById("error").textContent = "";

  try {{
    const res = await fetch(
      `/admin/projects/${{PROJECT_ID}}/recomposite-book`,
      {{ method: "POST", headers: HEADERS }}
    );
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    status.textContent = "✓ PDF rebuilt successfully!";
    btn.disabled = false;
  }} catch(e) {{
    document.getElementById("error").textContent = "Rebuild failed: " + e.message;
    status.textContent = "";
    btn.disabled = false;
  }}
}}

async function retryFullGeneration() {{
  if (!confirm("This re-runs FULL AI generation for every panel from scratch and uses AI credits again. Only use this if 'Rebuild PDF' failed because panels are actually missing. Continue?")) return;
  const btn = document.getElementById("btnRetry");
  const status = document.getElementById("rebuildStatus");
  btn.disabled = true;
  status.textContent = "Full regeneration queued — this runs in the background and takes several minutes. Refresh this page later to check progress.";
  document.getElementById("error").textContent = "";

  try {{
    const res = await fetch(
      `/admin/projects/${{PROJECT_ID}}/retry-book?force=true`,
      {{ method: "POST", headers: HEADERS }}
    );
    if (!res.ok) throw new Error(await res.text());
  }} catch(e) {{
    document.getElementById("error").textContent = "Retry failed to start: " + e.message;
    status.textContent = "";
    btn.disabled = false;
  }}
}}

load();
</script>
</body>
</html>"""

    return HTMLResponse(html)


# ── Funnel analytics dashboard ─────────────────────────────────────────────────

@router.get("/funnel")
def funnel_page(secret: str = "", db: Session = Depends(get_db)):
    if secret != settings.secret_key:
        return HTMLResponse("<h1>403 Forbidden</h1>", status_code=403)

    from sqlalchemy import func as sqlfunc

    counts = (
        db.query(models.FunnelEvent.event, sqlfunc.count(models.FunnelEvent.id))
        .group_by(models.FunnelEvent.event)
        .all()
    )
    count_map = {row[0]: row[1] for row in counts}

    recent = (
        db.query(models.FunnelEvent)
        .order_by(models.FunnelEvent.created_at.desc())
        .limit(100)
        .all()
    )

    def fmt_ts(ts):
        return ts.strftime("%Y-%m-%d %H:%M") if ts else "—"

    stages = [
        ("preview_viewed",   "Preview Viewed"),
        ("options_viewed",   "Options Page Viewed"),
        ("checkout_started", "Checkout Started"),
    ]

    stage_rows = ""
    prev_count = None
    for key, label in stages:
        n = count_map.get(key, 0)
        drop = ""
        if prev_count is not None and prev_count > 0:
            pct = int(n / prev_count * 100)
            drop = f'<span style="color:#e74c3c;font-size:12px">({pct}% of previous)</span>'
        stage_rows += f"<tr><td>{label}</td><td><strong>{n}</strong></td><td>{drop}</td></tr>"
        prev_count = n

    event_rows = "".join(
        f"<tr><td>{fmt_ts(ev.created_at)}</td><td>{ev.event}</td>"
        f"<td style='font-size:11px;color:#666'>{str(ev.project_id)[:8] if ev.project_id else '—'}</td></tr>"
        for ev in recent
    )

    html = f"""<!DOCTYPE html>
<html><head><title>WhamComics Funnel</title>
<style>
body{{font-family:Arial,sans-serif;background:#111;color:#eee;padding:32px;}}
h1{{color:#ffd700;}}h2{{color:#aaa;margin-top:32px;}}
table{{border-collapse:collapse;width:100%;max-width:700px;margin-top:12px;}}
th{{background:#222;color:#ffd700;padding:10px 14px;text-align:left;}}
td{{padding:9px 14px;border-bottom:1px solid #333;}}
tr:hover td{{background:#1a1a1a;}}
a{{color:#ffd700;}}
</style></head><body>
<h1>Funnel Analytics</h1>
<a href="/admin?secret={secret}">← Back to dashboard</a>
<h2>Conversion Funnel</h2>
<table><tr><th>Stage</th><th>Count</th><th>vs Previous</th></tr>
{stage_rows}
</table>
<h2>Recent Events (last 100)</h2>
<table><tr><th>Time</th><th>Event</th><th>Project</th></tr>
{event_rows}
</table>
</body></html>"""

    return HTMLResponse(html)
