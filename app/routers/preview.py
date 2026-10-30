from fastapi import APIRouter, Depends, HTTPException
from app.config import settings
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
import base64
import json
import threading
from app.services.ai import (
    GENRE_STYLES,
    generate_preview_pages_ai,
    generate_character_reference_ai,
    generate_image_with_reference,
    build_cartoon_page_prompt,
    build_cover_page_prompt,
    generate_book_title_ai,
    generate_cover_line_ai,
    check_nickname_usage_ai,
    verify_characters_visible_ai,
)
import app.services.fal_service as fal_service
import app.services.llamagen_service as llamagen_service
from app.services.speech_bubbles import add_cover_text
from app.services.comic_layout import compose_comic_page, compose_single_panel_page
from app.s3 import upload_bytes, download_bytes, make_key, generate_presigned_get_url

from ..db import get_db, SessionLocal
from .. import models, schemas


router = APIRouter(prefix="/projects/{project_id}/preview", tags=["preview"])

def normalize_dialogue(dialogue):
    if not dialogue:
        return []

    # If AI returns one string
    if isinstance(dialogue, str):
        if ":" in dialogue:
            speaker, text = dialogue.split(":", 1)
            return [{"speaker": speaker.strip(), "line": text.strip()}]
        return [{"speaker": "Narrator", "line": dialogue.strip()}]

    # If AI returns a list of strings
    if isinstance(dialogue, list):
        normalized = []
        for line in dialogue:
            if isinstance(line, str):
                if ":" in line:
                    speaker, text = line.split(":", 1)
                    normalized.append({
                        "speaker": speaker.strip(),
                        "line": text.strip()
                    })
                else:
                    normalized.append({
                        "speaker": "Narrator",
                        "line": line.strip()
                    })
            elif isinstance(line, dict):
                normalized.append({
                    "speaker": line.get("speaker", "Narrator"),
                    "line": line.get("line") or line.get("text", "")
                })
        return normalized

    return []


def _send_preview_quality_alert_email(project_id: str, hero_name: str, customer_email: str, issues: list) -> None:
    """
    Fires in real time when a preview panel/cover still fails its character-
    visibility check after using up its retry budget -- the customer may be
    looking at this exact preview right now, so the point is to know about a
    likely issue while or just after it happens, not discover it only once
    the customer does. Reuses book_generation.py's Brevo sender rather than
    duplicating it.
    """
    from app.services.book_generation import _brevo_send
    items = "".join(f"<li>{i}</li>" for i in issues)
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h2 style="color:#fff;background:#c0392b;padding:16px;border-radius:8px;">
        ⚠️ Preview may have a visible issue
      </h2>
      <p><strong>Hero:</strong> {hero_name}</p>
      <p><strong>Customer:</strong> {customer_email}</p>
      <p><strong>Project ID:</strong> {project_id}</p>
      <p><strong>What was flagged:</strong></p>
      <ul>{items}</ul>
      <p style="color:#666;font-size:13px;">
        This preview already shipped to the customer (the retry budget was used up) --
        this is a heads-up, not something blocking their experience. Worth a quick look.
      </p>
    </div>
    """
    _brevo_send(settings.owner_email, f"[WhamComics] ⚠️ Preview quality flag — {hero_name}", html)


def _get_project_or_404(db: Session, project_id: str) -> models.Project:
    try:
        pid = UUID(project_id)
    except Exception:
        raise HTTPException(422, "project_id must be a UUID")

    proj = db.get(models.Project, pid)
    if not proj:
        raise HTTPException(404, "Project not found")
    return proj


@router.get("", response_model=List[schemas.PreviewPageOut])
def get_preview(project_id: str, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)

    pages = (
        db.query(models.PreviewPage)
        .filter(models.PreviewPage.project_id == proj.id)
        .order_by(models.PreviewPage.page_number.asc())
        .all()
    )

    response_pages = []

    for page in pages:
        image_url = page.image_url

        if image_url:
            image_url = generate_presigned_get_url(image_url)

        response_pages.append(
            schemas.PreviewPageOut(
                id=page.id,
                project_id=page.project_id,
                story_option_id=page.story_option_id,
                page_number=page.page_number,
                narration=page.narration,
                dialogue=normalize_dialogue(page.dialogue),
                image_prompt=page.image_prompt,
                image_url=image_url,
                created_at=page.created_at,
            )
        )

    return response_pages


def build_appearance_anchor(character: models.Character) -> str:
    if not character:
        return ""
    traits = character.traits or {}
    appearance = traits.get("appearance", {})
    personalization = character.personalization or {}
    # customer-supplied gender overrides AI extraction (extraction can misread e.g. locs as female)
    gender = personalization.get("character_gender") or appearance.get("gender")
    parts = []
    if gender:
        age = appearance.get("age_range", "")
        parts.append(f"{age} {gender}".strip())
    if appearance.get("skin_tone"):
        parts.append(f"{appearance['skin_tone']} skin")
    if appearance.get("hair"):
        parts.append(appearance["hair"])
    if appearance.get("eye_color"):
        parts.append(f"{appearance['eye_color']} eyes")
    if appearance.get("glasses"):
        parts.append(appearance["glasses"])
    if appearance.get("clothing"):
        parts.append(appearance["clothing"])
    return ", ".join(parts)

def build_character_visual_brief(character: models.Character) -> str:
    if not character:
        return "A kid superhero with a friendly comic-book look."

    traits = character.traits or {}
    appearance = traits.get("appearance", {})
    personality = traits.get("personality", [])
    powers = traits.get("powers", [])

    parts = [character.name]

    if appearance.get("skin_tone"):
        parts.append(f"with {appearance['skin_tone']} skin")
    if appearance.get("hair"):
        parts.append(f"and {appearance['hair']}")
    if appearance.get("clothing"):
        parts.append(f"wearing {appearance['clothing']}")
    if personality:
        parts.append(f"with a {' and '.join(personality[:3])} personality")
    if powers:
        parts.append(f"and {'/'.join(powers[:2])} powers")

    return " ".join(parts)

def build_character_lock(character: models.Character) -> str:
    if not character:
        return """
Use the same main character in every image.
Do not redesign the character between pages.
Keep the same face, hair, outfit, age, and proportions in every scene.
"""

    traits = character.traits or {}
    appearance = traits.get("appearance", {})
    personalization = character.personalization or {}

    name = character.name or "the main character"
    skin_tone = appearance.get("skin_tone", "the same skin tone")
    hair = appearance.get("hair", "the same hairstyle")
    clothing = appearance.get("clothing", "the same outfit")
    nose = appearance.get("nose", "")
    glasses = appearance.get("glasses", "")

    char_age = int(personalization.get("age", 0) or 0) or int(character.age or 0)
    is_adult = char_age >= 18
    char_gender = (personalization.get("character_gender") or appearance.get("gender") or "").lower()
    if is_adult:
        if "female" in char_gender or "woman" in char_gender or "girl" in char_gender:
            person_label = "adult woman"
        elif "male" in char_gender or "man" in char_gender or "boy" in char_gender:
            person_label = "adult man"
        else:
            person_label = "adult person"
    else:
        person_label = "child"

    nose_line = f"    - exact same nose: {nose} — do NOT reshape, narrow, or widen it between panels\n" if nose else ""
    glasses_line = f"    - exact same glasses: {glasses} — same frame shape, color, and thickness in every panel\n" if glasses else ""

    return f"""
    This exact same {person_label} must appear in every image.

    Identity lock (EXTREMELY IMPORTANT):
    - same exact face structure in every image
    - same eye shape and eye size
    - same eyebrow shape and thickness
{nose_line}{glasses_line}    - same mouth shape and smile style
    - same ear shape and placement
    - same head shape and proportions

    Hair lock:
    - exact same hair color: {hair}
    - exact same hairstyle in every image
    - do NOT change hair length, volume, or texture

    Skin:
    - exact same skin tone: {skin_tone}

    Reference outfit (everyday clothes from photo — may be overridden by story):
    - {clothing}
    - IMPORTANT: if the SCENE PROMPT describes a different outfit (jersey, leotard, swimsuit, etc.), use ONLY the scene prompt outfit — do NOT layer or combine the reference clothes over the story outfit

    Strict rules:
    - Do NOT redesign the character
    - Do NOT reinterpret the face
    - Do NOT change proportions or age
    - This must look like the SAME {person_label} photographed in different scenes
    - NOT a different drawing of the same idea — the SAME {person_label}

    Consistency priority is MORE IMPORTANT than creativity.
    """


def _generate_preview_background(project_id: str):
    db = SessionLocal()
    try:
        proj = db.get(models.Project, UUID(project_id))
        if not proj:
            return

        option = db.get(models.StoryOption, proj.selected_story_option_id)
        if not option or option.project_id != proj.id:
            proj.status = models.ProjectStatus.draft
            db.commit()
            return

        main = (
            db.query(models.Character)
            .filter(
                models.Character.project_id == proj.id,
                models.Character.role == "main",
            )
            .first()
        )

        if not main or not main.photo_keys:
            proj.status = models.ProjectStatus.draft
            db.commit()
            return

        proj.status = models.ProjectStatus.generating
        db.commit()
        _run_preview_generation(proj, option, main, db)

    except Exception as e:
        print(f"[preview background] generation failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            proj = db.get(models.Project, UUID(project_id))
            if proj:
                proj.status = models.ProjectStatus.draft
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _run_preview_generation(proj, option, main, db):

        hero_name = main.name if main else "Our hero"
        character_lock = build_character_lock(main)
        genre_style = GENRE_STYLES.get(proj.genre or "adventure", GENRE_STYLES["adventure"])

        # Query support chars early so we can pass them to story generation
        all_support_chars = (
            db.query(models.Character)
            .filter(
                models.Character.project_id == proj.id,
                models.Character.role == "support",
            )
            .limit(2)
            .all()
        )
        support_char_info = []
        for sc in all_support_chars:
            sc_appearance = (sc.traits or {}).get("appearance", {})
            sc_info = {"name": sc.name}
            sc_age = sc.age or (sc.personalization or {}).get("age")
            sc_gender = (
                (sc.personalization or {}).get("character_gender")
                or sc_appearance.get("gender", "")
            )
            sc_relationship = (sc.personalization or {}).get("relationship_to_main", "")
            if sc_age:
                sc_info["age"] = int(sc_age)
            if sc_gender:
                sc_info["gender"] = sc_gender
            if sc_relationship:
                sc_info["relationship"] = sc_relationship
            if sc_appearance.get("hair"):
                sc_info["hair"] = sc_appearance["hair"]
            if sc_appearance.get("clothing"):
                sc_info["clothing"] = sc_appearance["clothing"]
            sc_pers = sc.personalization or {}
            if sc_pers.get("nickname_for_hero"):
                sc_info["nickname_for_hero"] = sc_pers["nickname_for_hero"].strip()
            if sc_pers.get("hero_nickname_for_them"):
                sc_info["hero_nickname_for_them"] = sc_pers["hero_nickname_for_them"].strip()
            support_char_info.append(sc_info)

        generated_pages = generate_preview_pages_ai(
            hero_name=hero_name,
            theme=proj.theme,
            logline=option.logline,
            outline=option.outline,
            character_brief=character_lock,
            page_count=3,
            personalization=main.personalization,
            genre=proj.genre or "adventure",
            additional_characters=support_char_info or None,
        )

        if len(generated_pages) != 3:
            raise ValueError(f"Expected 3 panels, got {len(generated_pages)}")

        page_numbers = sorted(item.get("page_number") for item in generated_pages if "page_number" in item)
        if page_numbers != [1, 2, 3]:
            raise ValueError(f"Invalid panel numbers: {page_numbers}")

        generated_pages = check_nickname_usage_ai(
            pages=generated_pages, hero_name=hero_name,
            additional_characters=support_char_info or None, page_count=3,
        )

        db.query(models.PreviewPage).filter(
            models.PreviewPage.project_id == proj.id
        ).delete()

        # Download all uploaded photos once — used as reference for every panel
        raw_photo_bytes_list = [download_bytes(k) for k in main.photo_keys]

        # Always generate a fresh reference per project — never reuse a reference
        # from a different story (e.g. a soccer reference leaking into a fire story)
        cartoon_reference_bytes = generate_character_reference_ai(
            raw_photo_bytes_list[0],
            extra_photos=raw_photo_bytes_list[1:] if len(raw_photo_bytes_list) > 1 else [],
        )
        ref_key = make_key(
            prefix=f"projects/{proj.id}/characters/{main.id}/reference",
            filename="reference.png",
        )
        upload_bytes(ref_key, cartoon_reference_bytes, content_type="image/png")
        main.reference_image_key = ref_key
        db.add(main)
        db.commit()

        sorted_pages = sorted(generated_pages, key=lambda x: x["page_number"])
        _audience_age = (main.personalization or {}).get("age") or 10
        book_title = generate_book_title_ai(hero_name=hero_name, theme=proj.theme, logline=option.logline, audience_age=_audience_age)
        cover_line = generate_cover_line_ai(hero_name=hero_name, logline=option.logline, audience_age=_audience_age)

        # ── Panels first (cover is generated last so it can reference a completed panel) ──
        panels = {}
        use_lora = bool(settings.use_lora and main.fal_lora_url and main.fal_lora_status == "ready")
        use_instantid = bool(not use_lora and settings.use_instantid and raw_photo_bytes_list)
        use_pulid = bool(not use_lora and not use_instantid and settings.use_pulid and raw_photo_bytes_list)
        use_llamagen = bool(not use_lora and not use_instantid and not use_pulid and settings.use_llamagen and main.photo_keys)
        appearance_anchor = build_appearance_anchor(main)
        pulid_reference = raw_photo_bytes_list[0] if raw_photo_bytes_list else None
        # Same TTL bump as book_generation.py's llamagen_photo_url -- LlamaGen's own
        # team reported 403s fetching reference images, and this URL is reused for
        # every panel call for the life of the preview generation.
        _LLAMAGEN_PHOTO_URL_TTL = 86400
        llamagen_photo_url = generate_presigned_get_url(main.photo_keys[0], expires_in=_LLAMAGEN_PHOTO_URL_TTL) if use_llamagen else ""
        llamagen_age = (main.personalization or {}).get("age") or main.age
        llamagen_gender = (main.personalization or {}).get("character_gender") or (main.traits or {}).get("appearance", {}).get("gender", "")
        _lg_appearance = (main.traits or {}).get("appearance", {})
        _lg_notes_parts = []
        if llamagen_age:
            _lg_notes_parts.append(f"{llamagen_age}-year-old")
        if llamagen_gender:
            _lg_notes_parts.append(llamagen_gender)
        if _lg_appearance.get("glasses"):
            _lg_notes_parts.append(f"wears {_lg_appearance['glasses']} — always include glasses")
        if _lg_appearance.get("hair"):
            _lg_notes_parts.append(_lg_appearance["hair"])
        llamagen_appearance_notes = ", ".join(_lg_notes_parts)
        llamagen_extra_roles = []
        if use_llamagen:
            for sc in all_support_chars:
                if sc.photo_keys:
                    sc_url = generate_presigned_get_url(sc.photo_keys[0], expires_in=_LLAMAGEN_PHOTO_URL_TTL)
                    sc_role: dict = {"name": sc.name, "image": sc_url}
                    sc_age = sc.age or int((sc.personalization or {}).get("age", 0) or 0) or None
                    sc_gender = (
                        (sc.personalization or {}).get("character_gender")
                        or (sc.traits or {}).get("appearance", {}).get("gender", "")
                    )
                    if sc_age:
                        sc_role["age"] = int(sc_age)
                    if sc_gender:
                        sc_role["gender"] = sc_gender
                    llamagen_extra_roles.append(sc_role)
        path = "LoRA" if use_lora else ("InstantID" if use_instantid else ("PuLID" if use_pulid else ("LlamaGen" if use_llamagen else "GPT-Image-1")))
        print(f"Image generation path: {path}")

        # Progressive commit: previously, cover/page-1/page-2 were only composited and
        # committed to the DB in one final block after EVERYTHING (all panels, cover,
        # and any retries) finished -- so a customer polling GET /preview saw nothing
        # at all, for the entire generation window, if even one panel needed a slow
        # LlamaGen retry. These helpers let each page get committed the moment its
        # own dependencies are ready, so the customer can see the cover and page 1
        # while page 2 is still retrying, instead of a blank screen the whole time.
        _committed_pages: dict = {}

        def _panel_narration(item: dict, fallback: str = "") -> str:
            """Return AI narration, or a fallback caption if the AI left it empty."""
            n = (item.get("narration") or "").strip()
            if n:
                return n
            # Pull a brief caption from dialogue as a last resort
            d = normalize_dialogue(item.get("dialogue"))
            if d:
                first_line = d[0].get("line", "").strip()
                if first_line:
                    return f'"{first_line}"'
            return fallback

        def _commit_page1():
            if 1 in _committed_pages or 1 not in panels or 2 not in panels:
                return
            panel_a_bytes, item_a = panels[1]
            panel_b_bytes, item_b = panels[2]
            dialogue_a = normalize_dialogue(item_a.get("dialogue"))
            dialogue_b = normalize_dialogue(item_b.get("dialogue"))
            narration_a = _panel_narration(item_a, f"The adventure begins for {hero_name}.")
            narration_b = _panel_narration(item_b)
            spread_bytes = compose_comic_page(
                panel_a_bytes, panel_b_bytes,
                dialogue_a=dialogue_a, dialogue_b=dialogue_b,
                narration_a=narration_a,
                narration_b=narration_b,
                hero_name=hero_name,
            )
            key = make_key(prefix=f"previews/{proj.id}", filename="page-1.png")
            upload_bytes(key=key, data=spread_bytes)
            page = models.PreviewPage(
                project_id=proj.id, story_option_id=option.id, page_number=1,
                narration=f"{narration_a}\n\n{narration_b}",
                dialogue=dialogue_a + dialogue_b,
                image_prompt=item_a["image_prompt"], image_url=key,
            )
            db.add(page)
            db.commit()
            db.refresh(page)
            _committed_pages[1] = page
            print(f"✅ Preview page 1 ready for project {proj.id}")

        def _commit_page2():
            if 2 in _committed_pages or 3 not in panels:
                return
            panel_bytes, item = panels[3]
            dialogue = normalize_dialogue(item.get("dialogue"))
            narration_3 = _panel_narration(item, "What happens next will change everything...")
            single_bytes = compose_single_panel_page(
                panel_bytes,
                dialogue=dialogue,
                narration=narration_3,
                hero_name=hero_name,
            )
            key = make_key(prefix=f"previews/{proj.id}", filename="page-2.png")
            upload_bytes(key=key, data=single_bytes)
            page = models.PreviewPage(
                project_id=proj.id, story_option_id=option.id, page_number=2,
                narration=narration_3,
                dialogue=dialogue,
                image_prompt=item["image_prompt"], image_url=key,
            )
            db.add(page)
            db.commit()
            db.refresh(page)
            _committed_pages[2] = page
            print(f"✅ Preview page 2 ready for project {proj.id}")

        def _commit_cover(raw_cover_bytes):
            if 0 in _committed_pages:
                return
            cover_prompt = f"{hero_name} — {option.logline}"
            upload_bytes(key=f"panels/{proj.id}/cover-raw.png", data=raw_cover_bytes, content_type="image/png")
            cover_supporting_name = support_char_info[0]["name"] if support_char_info else None
            cover_bytes = add_cover_text(
                image_bytes=raw_cover_bytes, title=book_title, hero_name=hero_name,
                cover_line=cover_line, title_color=genre_style["title_color"],
                supporting_name=cover_supporting_name, genre=proj.genre or "adventure",
            )
            cover_key = make_key(prefix=f"previews/{proj.id}", filename="cover.png")
            upload_bytes(key=cover_key, data=cover_bytes)
            cover_page = models.PreviewPage(
                project_id=proj.id, story_option_id=option.id, page_number=0,
                narration=book_title,
                dialogue=[{"speaker": hero_name, "line": cover_line}] if cover_line else [],
                image_prompt=cover_prompt, image_url=cover_key,
            )
            db.add(cover_page)
            db.commit()
            db.refresh(cover_page)
            _committed_pages[0] = cover_page
            print(f"✅ Preview cover ready for project {proj.id}")

        if use_llamagen:
            # Build all panel prompts up front so we can fire cover + all 3 panels in parallel.
            # LlamaGen cover has no dependency on panel bytes (unlike GPT path), so all 4 calls
            # can run simultaneously — cutting wall time from ~8 min sequential to ~2 min parallel.
            # setting_reference_prompts/setting_blocking_prompts mirror the same code-level lock
            # used in book_generation.py's full pipeline -- this preview path never had it, which
            # is exactly why the same driver/passenger seat-swap bug showed up here too.
            setting_reference_prompts: dict[str, str] = {}
            setting_blocking_prompts: dict[tuple, str] = {}
            panel_tasks = []
            for item in sorted_pages:
                panel_num = item["page_number"]
                lg_scene = item["image_prompt"]

                panel_setting = (item.get("setting") or "").strip()
                panel_blocking = (item.get("blocking") or "").strip()
                panel_group = tuple(sorted(str(n).strip().lower() for n in (item.get("characters_present") or [])))
                blocking_key = (panel_setting, panel_group)

                if panel_setting:
                    if panel_setting not in setting_reference_prompts:
                        setting_reference_prompts[panel_setting] = item["image_prompt"]
                    else:
                        lg_scene = (
                            f"SETTING LOCK — THIS PANEL TAKES PLACE AT THE SAME LOCATION AS AN EARLIER PANEL "
                            f"({panel_setting}). The background, environment, colors, and layout MUST match this "
                            f"reference description exactly — do not invent a new or different-looking version of "
                            f"this location:\n\"{setting_reference_prompts[panel_setting]}\"\n\n"
                            f"THIS PANEL'S SPECIFIC MOMENT (same location as above, different action):\n"
                        ) + lg_scene

                if panel_blocking and panel_group:
                    if blocking_key not in setting_blocking_prompts:
                        setting_blocking_prompts[blocking_key] = panel_blocking
                    elif setting_blocking_prompts[blocking_key] != panel_blocking:
                        lg_scene = (
                            f"BLOCKING LOCK — THIS SAME GROUP OF CHARACTERS IS ALREADY POSITIONED AT THIS LOCATION "
                            f"FROM AN EARLIER PANEL. Every character's position (who is driving, front seat, back "
                            f"seat, standing left/right, etc.) MUST match this exactly — do not swap anyone's seat "
                            f"or position: \"{setting_blocking_prompts[blocking_key]}\"\n\n"
                        ) + lg_scene

                if llamagen_age and int(llamagen_age) >= 18:
                    lg_scene = "ALL CHARACTERS IN THIS SCENE ARE ADULTS (18+). No children or teenagers.\n\n" + lg_scene
                # Best friend is intentionally excluded from panel image prompts.
                # Without a photo reference she causes character confusion (LlamaGen uses the hero's
                # photo for her, making them look identical). She appears in story text only.
                # Always inject [Character: Name] tags for every support char with a photo.
                # Never gate on a name-in-text check — the AI may refer to them by role
                # ("her husband") rather than by name, causing the tag to be skipped and
                # LlamaGen to render a generic person instead of using the reference photo.
                for sc_info in support_char_info:
                    if not any(r["name"] == sc_info["name"] for r in llamagen_extra_roles):
                        continue  # no photo uploaded for this support char — skip
                    sc_name = sc_info["name"]
                    # Default to adult when age unknown — same logic as main character
                    sc_is_adult = (sc_info.get("age") is None) or (int(sc_info["age"]) >= 18)
                    sc_age_group = "adult" if sc_is_adult else "child"
                    sc_tag_parts = []
                    if sc_info.get("age"):
                        sc_tag_parts.append(f"{sc_info['age']}-year-old")
                    if sc_info.get("gender"):
                        sc_tag_parts.append(sc_info["gender"])
                    sc_tag_parts.append(sc_age_group)
                    sc_appearance_locks = []
                    if sc_info.get("hair"):
                        sc_appearance_locks.append(sc_info["hair"])
                    if sc_info.get("clothing"):
                        sc_appearance_locks.append(f"wearing {sc_info['clothing']} — same outfit every panel")
                    sc_appearance_str = ", ".join(sc_appearance_locks)
                    sc_tag = f"[Character: {sc_name}] {' '.join(sc_tag_parts)}"
                    if sc_appearance_str:
                        sc_tag += f", {sc_appearance_str}"
                    sc_tag += "."
                    lg_scene = f"{sc_tag}\n\n" + lg_scene
                panel_dialogue = item.get("dialogue") or []
                if panel_num != 3 and len(panel_dialogue) >= 2:
                    s0 = (panel_dialogue[0].get("speaker") or "").strip()
                    s1 = (panel_dialogue[1].get("speaker") or "").strip()
                    if s0 and s1:
                        lg_scene += (
                            f"\n\nPANEL LAYOUT: {s0} is in the left-center area of the panel "
                            f"(NOT at the very left edge — stay at least 25% from the left border). "
                            f"{s1} is in the right-center area of the panel "
                            f"(NOT at the very right edge — stay at least 25% from the right border)."
                        )
                panel_tasks.append((panel_num, lg_scene, item))

            # Build cover support string — photo-backed chars first so LlamaGen prioritizes them
            # if it can only render a limited number of background characters.
            cover_support_parts = []
            # Best friend excluded from cover — no photo means unreliable rendering.
            # Support chars — keep [Character:] tag so their photo reference renders correctly,
            # but depth/size hierarchy is controlled by the cover prompt language
            for sc_info in support_char_info:
                if any(r["name"] == sc_info["name"] for r in llamagen_extra_roles):
                    sc_is_adult = (sc_info.get("age") is None) or (int(sc_info["age"]) >= 18)
                    sc_tag_parts = []
                    if sc_info.get("age"):
                        sc_tag_parts.append(f"{sc_info['age']}-year-old")
                    if sc_info.get("gender"):
                        sc_tag_parts.append(sc_info["gender"])
                    sc_tag_parts.append("adult" if sc_is_adult else "child")
                    if sc_info.get("hair"):
                        sc_tag_parts.append(sc_info["hair"])
                    cover_support_parts.append(f"[Character: {sc_info['name']}] {' '.join(sc_tag_parts)}")
            cover_supporting = ", ".join(cover_support_parts)

            # Collects a note whenever the character-visibility check still
            # fails after its retry budget is used up -- the customer is about
            # to see this preview live, so if it happens, we want to know
            # about it right then, not only from a Railway log line nobody's
            # watching in real time.
            preview_quality_issues: list = []

            def _gen_panel_task(task):
                pnum, scene, data = task
                expected_chars = [str(n).strip() for n in (data.get("characters_present") or []) if str(n).strip()]
                # A single content-filter rejection or transient timeout used
                # to take down the whole preview with no retry -- see the
                # matching helper in book_generation.py for why. Retry once
                # with the identical prompt, then once more with an explicit
                # safety reframing, before giving up.
                last_exc = None
                last_pb = None
                last_missing: list = []
                missing_to_retry: list = []
                for attempt in range(3):
                    scene_prompt = scene
                    if missing_to_retry:
                        # Found in production: retrying with the IDENTICAL prompt after a
                        # missing-character miss reproduced the same miss almost every time
                        # -- the model needs to be told what it actually got wrong, not just
                        # asked to roll the dice again on the same instructions.
                        scene_prompt = (
                            f"⚠️ YOUR PREVIOUS ATTEMPT AT THIS EXACT SCENE FAILED TO SHOW "
                            f"{', '.join(missing_to_retry)} — the image came out as background/environment only "
                            f"with no visible person for {'them' if len(missing_to_retry) > 1 else missing_to_retry[0]}. "
                            f"This time, {', '.join(missing_to_retry)} MUST appear as an actual visible human "
                            f"figure, full body, clearly in the scene — do not simplify the composition down to "
                            f"just the environment.\n\n"
                        ) + scene_prompt
                    if attempt == 2:
                        scene_prompt = (
                            "Render this scene safely and non-graphically -- convey tension, urgency, and physical "
                            "action through body language, framing, and motion cues rather than depicting contact, "
                            "impact, or harm directly. Avoid anything that could be read as violent or graphic.\n\n"
                        ) + scene_prompt
                    try:
                        print(f"[parallel] Generating panel {pnum} (attempt {attempt + 1}/3)...")
                        pb = llamagen_service.generate_panel_llamagen(
                            scene_prompt=scene_prompt,
                            photo_url=llamagen_photo_url,
                            character_name=main.name,
                            character_age=llamagen_age,
                            character_gender=llamagen_gender,
                            appearance_notes=appearance_anchor or llamagen_appearance_notes,
                            extra_roles=llamagen_extra_roles,
                        )
                        last_pb = pb
                        last_missing = []
                        # Ground-truth check: does the image actually show the characters
                        # it's supposed to? Found in production: a correct, detailed prompt
                        # with a valid reference photo still rendered zero visible people in
                        # multiple panels of the same preview -- an image-fidelity miss by
                        # the model, not a prompt or data bug. Checked on every attempt (not
                        # just while retry budget remains) so a still-missing character on
                        # the final attempt gets flagged for the quality-alert email below,
                        # even though we no longer retry for it at that point.
                        if expected_chars:
                            try:
                                missing = verify_characters_visible_ai(pb, expected_chars)
                                if missing:
                                    last_missing = missing
                                    missing_to_retry = missing
                                    if attempt < 2:
                                        print(f"⚠️  Panel {pnum}: expected character(s) {missing} not visible in generated image — retrying")
                                        continue
                                    print(f"⚠️  Panel {pnum}: expected character(s) {missing} still not visible after all retries — shipping anyway, flagging for review")
                            except Exception as ve:
                                print(f"⚠️  Panel {pnum} character-visibility check failed (non-fatal, keeping image): {ve}")
                        break
                    except Exception as e:
                        last_exc = e
                        print(f"⚠️  Panel {pnum} generation failed on attempt {attempt + 1}/3 ({e})")
                if last_missing:
                    preview_quality_issues.append(f"Panel {pnum}: {', '.join(last_missing)} not clearly visible in the generated image.")
                if last_pb is not None:
                    return pnum, last_pb, data
                raise last_exc

            def _gen_cover_task():
                last_cb = None
                last_exc = None
                retry_note = ""
                # Found in production (2026-10-16): generate_cover_llamagen's call was
                # never wrapped in try/except here, unlike _gen_panel_task -- a single
                # LlamaGen timeout on the cover propagated straight through
                # cover_future.result() and crashed the ENTIRE preview, discarding
                # panels that had already succeeded. Matches the panel task's 3-attempt
                # budget and catches generation failures (not just missing-character
                # results) so a slow/timed-out cover gets retried instead of taking
                # everything else down with it.
                for attempt in range(3):
                    print(f"[parallel] Generating cover (attempt {attempt + 1}/3)...")
                    try:
                        cb = llamagen_service.generate_cover_llamagen(
                            hero_name=hero_name,
                            theme=proj.theme,
                            logline=option.logline,
                            photo_url=llamagen_photo_url,
                            character_age=llamagen_age,
                            character_gender=llamagen_gender,
                            appearance_notes=llamagen_appearance_notes,
                            extra_roles=llamagen_extra_roles,
                            supporting_characters=cover_supporting,
                            genre=proj.genre or "",
                            retry_note=retry_note,
                        )
                    except Exception as e:
                        last_exc = e
                        print(f"⚠️  Cover generation failed on attempt {attempt + 1}/3 ({e})")
                        continue
                    last_cb = cb
                    # Same ground-truth visibility check as panels -- the hero is always
                    # expected on the cover, regardless of which support characters make
                    # it in.
                    try:
                        missing = verify_characters_visible_ai(cb, [hero_name])
                        if missing:
                            if attempt < 2:
                                print(f"⚠️  Cover: {hero_name} not visible in generated image — retrying")
                                retry_note = "retry"
                                continue
                            print(f"⚠️  Cover: {hero_name} still not visible after retries — shipping anyway, flagging for review")
                            preview_quality_issues.append(f"Cover: {hero_name} not clearly visible in the generated image.")
                    except Exception as ve:
                        print(f"⚠️  Cover character-visibility check failed (non-fatal, keeping image): {ve}")
                    return last_cb
                if last_cb is not None:
                    return last_cb
                raise last_exc

            import concurrent.futures
            # Found in production (2026-10-20): a panel that exhausts all 3 of its
            # own retry attempts raises, and an unguarded future.result() here used
            # to propagate that straight out -- aborting this loop immediately and
            # skipping cover_future.result() entirely. In the incident that surfaced
            # this, the cover had already succeeded on its retry and panel 3 had
            # already shipped, but panel 1's total failure threw all of that away
            # (except page 2, which happened to commit before the loop reached
            # panel 1's future). Same bug class as the earlier cover-only fix, now
            # closed for the whole collection loop: every future gets a chance to
            # be collected and committed regardless of what else failed, and only
            # once everything's been gathered do we decide the preview is
            # incomplete -- with a clear message naming exactly what never
            # succeeded, instead of a raw exception from one arbitrary generation ID.
            panel_errors: dict[int, Exception] = {}
            raw_cover_bytes = None
            cover_error: Exception | None = None
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                panel_futures = {executor.submit(_gen_panel_task, t): t[0] for t in panel_tasks}
                cover_future = executor.submit(_gen_cover_task)
                for future in concurrent.futures.as_completed(panel_futures):
                    try:
                        pnum, pb, data = future.result()
                    except Exception as e:
                        failed_pnum = panel_futures[future]
                        panel_errors[failed_pnum] = e
                        print(f"⚠️  Panel {failed_pnum} permanently failed after all retries: {e}")
                        continue
                    upload_bytes(key=f"panels/{proj.id}/panel-{pnum}.png", data=pb, content_type="image/png")
                    panels[pnum] = (pb, data)
                    # Commit the moment each page's dependencies are ready, rather than
                    # waiting on the cover and every other panel too.
                    _commit_page1()
                    _commit_page2()
                try:
                    raw_cover_bytes = cover_future.result()
                except Exception as e:
                    cover_error = e
                    print(f"⚠️  Cover permanently failed after all retries: {e}")
            if raw_cover_bytes is not None:
                _commit_cover(raw_cover_bytes)

            if panel_errors or cover_error is not None:
                failed_parts = [f"panel {n}" for n in sorted(panel_errors)] + (["cover"] if cover_error is not None else [])
                raise RuntimeError(f"Preview incomplete -- {', '.join(failed_parts)} never succeeded after all retries")

            if preview_quality_issues:
                try:
                    _send_preview_quality_alert_email(str(proj.id), hero_name, proj.customer_email or "", preview_quality_issues)
                except Exception as ae:
                    print(f"⚠️  Preview quality alert email failed to send (non-fatal): {ae}")

        else:
            # Sequential generation for LoRA / InstantID / PuLID / GPT-Image-1 paths
            for item in sorted_pages:
                panel_num = item["page_number"]
                print(f"Generating panel {panel_num}/4...")
                if use_lora:
                    panel_bytes = fal_service.generate_page_with_lora(
                        scene_prompt=item["image_prompt"],
                        lora_url=main.fal_lora_url,
                        character_appearance=appearance_anchor,
                    )
                elif use_instantid:
                    panel_bytes = fal_service.generate_page_with_instantcharacter(
                        scene_prompt=item["image_prompt"],
                        reference_image_bytes=pulid_reference,
                        character_appearance=appearance_anchor,
                    )
                elif use_pulid:
                    panel_bytes = fal_service.generate_page_with_pulid(
                        scene_prompt=item["image_prompt"],
                        reference_image_bytes=pulid_reference,
                        character_appearance=appearance_anchor,
                    )
                else:
                    prompt = build_cartoon_page_prompt(
                        image_prompt=item["image_prompt"], character_brief=character_lock,
                        genre_style=genre_style,
                    )
                    image_base64 = generate_image_with_reference(
                        prompt=prompt,
                        reference_image_bytes=cartoon_reference_bytes,
                        photo_bytes_list=raw_photo_bytes_list,
                    )
                    panel_bytes = base64.b64decode(image_base64)
                upload_bytes(key=f"panels/{proj.id}/panel-{panel_num}.png", data=panel_bytes, content_type="image/png")
                panels[panel_num] = (panel_bytes, item)

            # ── Cover last (sequential paths) ──
            print("Generating cover page...")
            if use_lora:
                raw_cover_bytes = fal_service.generate_cover_with_lora(
                    hero_name=hero_name, theme=proj.theme, logline=option.logline,
                    lora_url=main.fal_lora_url, character_appearance=appearance_anchor,
                )
            elif use_instantid:
                raw_cover_bytes = fal_service.generate_cover_with_instantcharacter(
                    hero_name=hero_name, theme=proj.theme, logline=option.logline,
                    reference_image_bytes=pulid_reference, character_appearance=appearance_anchor,
                )
            elif use_pulid:
                raw_cover_bytes = fal_service.generate_cover_with_pulid(
                    hero_name=hero_name, theme=proj.theme, logline=option.logline,
                    reference_image_bytes=pulid_reference, character_appearance=appearance_anchor,
                )
            else:
                cover_prompt = build_cover_page_prompt(
                    hero_name=hero_name, theme=proj.theme, logline=option.logline,
                    character_brief=character_lock, genre_style=genre_style,
                )
                cover_base64 = generate_image_with_reference(
                    prompt=cover_prompt,
                    reference_image_bytes=cartoon_reference_bytes,
                    photo_bytes_list=raw_photo_bytes_list,
                    prior_page_bytes=[panels[1][0]],
                )
                raw_cover_bytes = base64.b64decode(cover_base64)
        # For the sequential (non-LlamaGen) generation paths above, nothing has been
        # committed yet -- these calls do the full compositing/commit work now. For the
        # LlamaGen parallel path, each page was already committed as soon as it was
        # ready (see the helpers defined above); these are then no-ops, guarded by
        # _committed_pages, so both code paths converge on the same final state.
        _commit_cover(raw_cover_bytes)
        _commit_page1()
        _commit_page2()

        panel_dialogue = {
            str(n): normalize_dialogue(item.get("dialogue"))
            for n, (_, item) in panels.items()
        }
        panel_narration = {
            str(n): item.get("narration", "")
            for n, (_, item) in panels.items()
        }
        upload_bytes(
            key=f"panels/{proj.id}/dialogue.json",
            data=json.dumps(panel_dialogue).encode(),
            content_type="application/json",
        )
        upload_bytes(
            key=f"panels/{proj.id}/narration.json",
            data=json.dumps(panel_narration).encode(),
            content_type="application/json",
        )

        proj.status = models.ProjectStatus.preview_ready
        db.commit()


@router.get("/character-ready")
def character_ready(project_id: str, db: Session = Depends(get_db)):
    """
    Frontend polls this while showing the 'preparing your character' screen.
    PuLID path: ready immediately once photos are uploaded.
    LoRA path: waits for training to complete.
    """
    proj = _get_project_or_404(db, project_id)
    main = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == proj.id,
            models.Character.role == "main",
        )
        .first()
    )
    if not main:
        raise HTTPException(400, "No character found for this project")

    # No-training paths (GPT-Image-1, PuLID, InstantID) — ready as soon as photos exist
    if not settings.use_lora:
        has_photos = bool(main.photo_keys)
        return {
            "ready": has_photos,
            "lora_status": "ready" if has_photos else "pending",
            "message": "Your character is ready!" if has_photos else "Uploading your photos...",
        }

    # LoRA path — wait for training
    lora_status = main.fal_lora_status or "pending"
    return {
        "ready": lora_status == "ready",
        "lora_status": lora_status,
        "message": {
            "pending": "Getting your character ready...",
            "training": "Training your character's unique style (this takes about 3–5 minutes)...",
            "ready": "Your character is ready!",
            "failed": "Something went wrong. Please try uploading your photos again.",
        }.get(lora_status, "Preparing..."),
    }


@router.post("/reset-status")
def reset_preview_status(project_id: str, db: Session = Depends(get_db)):
    """Unstick a project frozen in 'generating' state so it can be regenerated."""
    proj = _get_project_or_404(db, project_id)
    proj.status = models.ProjectStatus.draft
    db.commit()
    return {"status": "reset", "project_id": project_id}


@router.post("/generate")
def generate_preview(project_id: str, force: bool = False, db: Session = Depends(get_db)):
    import datetime
    proj = _get_project_or_404(db, project_id)

    if not proj.selected_story_option_id:
        raise HTTPException(400, "No story has been selected for this project")

    if proj.status == models.ProjectStatus.generating:
        # Auto-unstick: if stuck in generating for >10 minutes, allow restart
        stuck = True
        if proj.updated_at:
            age = datetime.datetime.now(datetime.timezone.utc) - proj.updated_at.replace(tzinfo=datetime.timezone.utc) if proj.updated_at.tzinfo is None else datetime.datetime.now(datetime.timezone.utc) - proj.updated_at
            stuck = age > datetime.timedelta(minutes=10)
        if not stuck and not force:
            return {"status": "generating"}
        print(f"[preview] Project {proj.id} unstuck from generating state (force={force})")

    if proj.status == models.ProjectStatus.preview_ready and not force:
        pages = (
            db.query(models.PreviewPage)
            .filter(models.PreviewPage.project_id == proj.id)
            .order_by(models.PreviewPage.page_number.asc())
            .all()
        )
        return [schemas.PreviewPageOut(
            id=p.id, project_id=p.project_id, story_option_id=p.story_option_id,
            page_number=p.page_number, narration=p.narration,
            dialogue=normalize_dialogue(p.dialogue), image_prompt=p.image_prompt,
            image_url=generate_presigned_get_url(p.image_url) if p.image_url else None,
            created_at=p.created_at,
        ) for p in pages]

    if force:
        db.query(models.PreviewPage).filter(
            models.PreviewPage.project_id == proj.id
        ).delete()
        db.commit()

    main = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == proj.id,
            models.Character.role == "main",
        )
        .first()
    )
    if not main or not main.photo_keys:
        raise HTTPException(400, "Character has no uploaded photos. Please upload at least one photo first.")

    # LoRA path only: block generation until training is complete
    if settings.use_lora and main.fal_lora_status != "ready":
        import datetime
        lora_status = main.fal_lora_status or "pending"
        training_age = (
            datetime.datetime.now(datetime.timezone.utc) - main.updated_at
            if main.updated_at else datetime.timedelta(0)
        )
        timed_out = training_age > datetime.timedelta(minutes=10)
        if not timed_out:
            return {"status": "awaiting_lora", "lora_status": lora_status}
        print(f"⚠️  LoRA training timed out for character {main.id} — falling back to GPT-Image-1")

    proj.status = models.ProjectStatus.generating
    db.commit()

    threading.Thread(target=_generate_preview_background, args=(str(proj.id),), daemon=True).start()

    return {"status": "generating"}


@router.post("/recomposite", response_model=List[schemas.PreviewPageOut])
def recomposite_preview(project_id: str, db: Session = Depends(get_db)):
    """Re-run Pass 2 (Pillow compositing + speech bubbles) without touching AI generation."""
    proj = _get_project_or_404(db, project_id)

    genre_style = GENRE_STYLES.get(proj.genre or "adventure", GENRE_STYLES["adventure"])

    recomp_main = (
        db.query(models.Character)
        .filter(models.Character.project_id == proj.id, models.Character.role == "main")
        .first()
    )
    recomp_hero_name = recomp_main.name if recomp_main else ""

    try:
        dialogue_bytes = download_bytes(f"panels/{proj.id}/dialogue.json")
    except Exception:
        raise HTTPException(400, "No raw panel data found — run /generate first.")

    panel_dialogue = json.loads(dialogue_bytes)

    try:
        narration_bytes = download_bytes(f"panels/{proj.id}/narration.json")
        panel_narration = json.loads(narration_bytes)
    except Exception:
        panel_narration = {}

    pages = (
        db.query(models.PreviewPage)
        .filter(models.PreviewPage.project_id == proj.id)
        .order_by(models.PreviewPage.page_number.asc())
        .all()
    )
    pages_by_num = {p.page_number: p for p in pages}
    results = []

    # Recomposite cover (silently skipped if raw image not present from older generation)
    recomp_support = (
        db.query(models.Character)
        .filter(models.Character.project_id == proj.id, models.Character.role == "support")
        .limit(1)
        .first()
    )
    recomp_supporting_name = recomp_support.name if recomp_support else None
    cover_page = pages_by_num.get(0)
    if cover_page:
        try:
            raw_cover_bytes = download_bytes(f"panels/{proj.id}/cover-raw.png")
            cover_line = cover_page.dialogue[0]["line"] if cover_page.dialogue else None
            cover_hero = cover_page.dialogue[0]["speaker"] if cover_page.dialogue else ""
            cover_bytes = add_cover_text(
                image_bytes=raw_cover_bytes,
                title=cover_page.narration,
                hero_name=cover_hero,
                cover_line=cover_line,
                title_color=genre_style["title_color"],
                supporting_name=recomp_supporting_name,
                genre=proj.genre or "adventure",
            )
            upload_bytes(key=cover_page.image_url, data=cover_bytes, content_type="image/png")
            results.append(cover_page)
        except Exception as e:
            print(f"⚠️  [preview] cover recomposite failed for project {proj.id}: {e}")

    # Recomposite page 1: 2-panel spread (panels 1+2)
    try:
        panel_a_bytes = download_bytes(f"panels/{proj.id}/panel-1.png")
        panel_b_bytes = download_bytes(f"panels/{proj.id}/panel-2.png")
    except Exception:
        raise HTTPException(400, "Raw panels 1 or 2 missing — run /generate first.")

    def _recomp_narration(stored: str, dialogue_list: list, fallback: str = "") -> str:
        n = (stored or "").strip()
        if n:
            return n
        if dialogue_list:
            first_line = (dialogue_list[0].get("line") or "").strip()
            if first_line:
                return f'"{first_line}"'
        return fallback

    dialogue_a = panel_dialogue.get("1", [])
    dialogue_b = panel_dialogue.get("2", [])
    narration_a = panel_narration.get("1", "")
    narration_b = panel_narration.get("2", "")
    if not narration_a or not narration_b:
        db_p1 = pages_by_num.get(1)
        parts = (db_p1.narration or "").split("\n\n", 1) if db_p1 else []
        narration_a = narration_a or (parts[0] if parts else "")
        narration_b = narration_b or (parts[1] if len(parts) > 1 else "")
    narration_a = _recomp_narration(narration_a, dialogue_a, f"The adventure begins for {recomp_hero_name}.")
    narration_b = _recomp_narration(narration_b, dialogue_b)
    spread_bytes = compose_comic_page(
        panel_a_bytes, panel_b_bytes,
        dialogue_a=dialogue_a, dialogue_b=dialogue_b,
        narration_a=narration_a, narration_b=narration_b,
        hero_name=recomp_hero_name,
    )
    page = pages_by_num.get(1)
    if page:
        upload_bytes(key=page.image_url, data=spread_bytes, content_type="image/png")
        results.append(page)

    # Recomposite page 2: single panel (panel 3)
    try:
        panel_bytes = download_bytes(f"panels/{proj.id}/panel-3.png")
    except Exception:
        raise HTTPException(400, "Raw panel 3 missing — run /generate first.")

    dialogue = panel_dialogue.get("3", [])
    narration = panel_narration.get("3", "")
    if not narration:
        db_p2 = pages_by_num.get(2)
        narration = (db_p2.narration or "") if db_p2 else ""
    narration = _recomp_narration(narration, dialogue, "What happens next will change everything...")
    single_bytes = compose_single_panel_page(panel_bytes, dialogue=dialogue, narration=narration, hero_name=recomp_hero_name)
    page = pages_by_num.get(2)
    if page:
        upload_bytes(key=page.image_url, data=single_bytes, content_type="image/png")
        results.append(page)

    db.commit()

    response_pages = []
    for page in sorted(results, key=lambda p: p.page_number):
        image_url = generate_presigned_get_url(page.image_url) if page.image_url else None
        response_pages.append(
            schemas.PreviewPageOut(
                id=page.id,
                project_id=page.project_id,
                story_option_id=page.story_option_id,
                page_number=page.page_number,
                narration=page.narration,
                dialogue=normalize_dialogue(page.dialogue),
                image_prompt=page.image_prompt,
                image_url=image_url,
                created_at=page.created_at,
            )
        )

    return response_pages

