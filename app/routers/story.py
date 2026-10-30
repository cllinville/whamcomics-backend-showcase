import threading
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app import models
from app.services.ai import (
    generate_story_options_ai,
    generate_comedy_blueprint_ai,
    validate_comedy_blueprint_ai,
    generate_adventure_blueprint_ai,
    validate_adventure_blueprint_ai,
    generate_mystery_blueprint_ai,
    validate_mystery_blueprint_ai,
    generate_horror_blueprint_ai,
    validate_horror_blueprint_ai,
    generate_horror_kids_blueprint_ai,
    validate_horror_kids_blueprint_ai,
    generate_sci_fi_blueprint_ai,
    validate_sci_fi_blueprint_ai,
    generate_fantasy_blueprint_ai,
    validate_fantasy_blueprint_ai,
    generate_prose_story_ai,
    expand_prose_if_short_ai,
    check_story_pacing_ai,
    check_story_continuity_ai,
    check_story_specificity_ai,
    check_story_arc_shape_ai,
    check_adventure_fidelity_ai,
    check_mystery_fidelity_ai,
    check_comedy_fidelity_ai,
    check_horror_fidelity_ai,
    check_horror_kids_fidelity_ai,
    check_sci_fi_fidelity_ai,
    check_fantasy_fidelity_ai,
    check_panel_fidelity_ai,
    check_panel_pacing_ai,
    punch_up_prose_ai,
    generate_story_character_bible_ai,
    generate_preview_pages_ai,
)
from ..db import get_db, SessionLocal
from .. import models, schemas

router = APIRouter(prefix="/projects/{project_id}/story", tags=["story"])

def _get_project_or_404(db: Session, project_id: str) -> models.Project:
    from uuid import UUID
    try:
        pid = UUID(project_id)
    except Exception:
        raise HTTPException(422, "project_id must be a UUID")
    proj = db.get(models.Project, pid)
    if not proj:
        raise HTTPException(404, "Project not found")
    return proj


# ── Background job support for the long-running text-only test endpoints ──────
# Railway's gateway times out well before a 30-72 panel pipeline (prose ->
# expansion -> pacing -> continuity -> specificity -> npc_bible -> panel
# script, up to 7 sequential OpenAI calls) finishes, even though the work
# itself completes fine server-side. These endpoints now return a job_id
# immediately and the caller polls /jobs/{job_id} for the result.
_jobs: dict[str, dict] = {}


def _run_job(job_id: str, fn, *args, **kwargs) -> None:
    try:
        result = fn(*args, **kwargs)
        _jobs[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as e:
        _jobs[job_id] = {"status": "failed", "result": None, "error": str(e)}


def _start_job(fn, *args, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "result": None, "error": None}
    thread = threading.Thread(target=_run_job, args=(job_id, fn) + args, kwargs=kwargs, daemon=True)
    thread.start()
    return job_id


def _build_support_char_info(db: Session, project_id) -> list:
    all_support_chars = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == project_id,
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
        sc_pers = sc.personalization or {}
        if sc_pers.get("nickname_for_hero"):
            sc_info["nickname_for_hero"] = sc_pers["nickname_for_hero"].strip()
        if sc_pers.get("hero_nickname_for_them"):
            sc_info["hero_nickname_for_them"] = sc_pers["hero_nickname_for_them"].strip()
        support_char_info.append(sc_info)
    return support_char_info


@router.post("/options", response_model=List[schemas.StoryOptionOut])
def generate_story_options(
    project_id: str,
    count: int = Query(3, ge=1, le=5, description="Number of options to return (1–5)"),
    db: Session = Depends(get_db),
):
    proj = _get_project_or_404(db, project_id)

    # pull a name and traits for AI story generation
    main = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == proj.id,
            models.Character.role == "main",
        )
        .first()
    )
    hero_name = main.name if main else "Our hero"

    traits = []
    if main and main.traits and "personality" in main.traits:
        traits = main.traits["personality"]

    support_chars = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == proj.id,
            models.Character.role == "support",
        )
        .all()
    )
    support_names = []
    for sc in support_chars:
        rel = (sc.personalization or {}).get("relationship_to_main", "")
        label = f"{sc.name} ({hero_name}'s {rel})" if rel else sc.name
        support_names.append(label)

    # Clear any existing unselected options so a refresh gives a clean set
    db.query(models.StoryOption).filter(
        models.StoryOption.project_id == proj.id
    ).delete()
    proj.selected_story_option_id = None
    db.commit()

    try:
        selected = generate_story_options_ai(
            hero_name=hero_name,
            traits=traits,
            theme=proj.theme,
            count=count,
            personalization=main.personalization if main else None,
            genre=proj.genre,
            support_characters=support_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI story generation failed: {str(e)}")

    results = []

    for item in selected:
        option = models.StoryOption(
            project_id=proj.id,
            logline=item["logline"],
            outline=item["outline"],
            estimated_pages=item.get("estimated_pages", 24),
        )
        db.add(option)
        results.append(option)

    db.commit()

    for option in results:
        db.refresh(option)

    return results

@router.get("/options", response_model=List[schemas.StoryOptionOut])
def list_story_options(
    project_id: str,
    db: Session = Depends(get_db),
):
    proj = _get_project_or_404(db, project_id)

    options = (
        db.query(models.StoryOption)
        .filter(models.StoryOption.project_id == proj.id)
        .order_by(models.StoryOption.id.asc())
        .all()
    )

    return options

@router.post("/choose")
def choose_story_option(
    project_id: str,
    payload: schemas.StoryChooseIn,
    db: Session = Depends(get_db),
):
    proj = _get_project_or_404(db, project_id)

    option = db.get(models.StoryOption, payload.story_option_id)
    if not option or option.project_id != proj.id:
        raise HTTPException(404, "Story option not found for this project")

    proj.selected_story_option_id = option.id
    # Reset preview so the new story gets a fresh generation
    proj.status = models.ProjectStatus.draft
    db.query(models.PreviewPage).filter(
        models.PreviewPage.project_id == proj.id
    ).delete()

    db.commit()
    db.refresh(proj)

    return {
        "message": "Story selected successfully",
        "project_id": str(proj.id),
        "selected_story_option_id": str(option.id),
    }


def _run_prose_pipeline(project_id: str, page_count_override: int | None) -> dict:
    """Runs in a background thread — opens its own DB session."""
    db = SessionLocal()
    try:
        proj = db.get(models.Project, uuid.UUID(project_id))
        option = db.get(models.StoryOption, proj.selected_story_option_id)
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == proj.id, models.Character.role == "main")
            .first()
        )

        hero_name = main.name
        genre = proj.genre or "adventure"
        page_count = page_count_override or proj.panel_count or 24
        support_char_info = _build_support_char_info(db, proj.id)

        passes = []

        comedy_blueprint: dict = {}
        if genre == "comedy":
            try:
                comedy_blueprint = generate_comedy_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("comedy_blueprint")
            except Exception as e:
                passes.append(f"comedy_blueprint_failed: {e}")

            if comedy_blueprint:
                try:
                    comedy_blueprint = validate_comedy_blueprint_ai(
                        blueprint=comedy_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("comedy_blueprint_validated")
                except Exception as e:
                    passes.append(f"comedy_blueprint_validation_failed: {e}")

        adventure_blueprint: dict = {}
        if genre == "adventure":
            try:
                adventure_blueprint = generate_adventure_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("adventure_blueprint")
            except Exception as e:
                passes.append(f"adventure_blueprint_failed: {e}")

            if adventure_blueprint:
                try:
                    adventure_blueprint = validate_adventure_blueprint_ai(
                        blueprint=adventure_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("adventure_blueprint_validated")
                except Exception as e:
                    passes.append(f"adventure_blueprint_validation_failed: {e}")

        mystery_blueprint: dict = {}
        if genre == "mystery":
            try:
                mystery_blueprint = generate_mystery_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("mystery_blueprint")
            except Exception as e:
                passes.append(f"mystery_blueprint_failed: {e}")

            if mystery_blueprint:
                try:
                    mystery_blueprint = validate_mystery_blueprint_ai(
                        blueprint=mystery_blueprint,
                        hero_name=hero_name,
                        page_count=page_count,
                    )
                    passes.append("mystery_blueprint_validated")
                except Exception as e:
                    passes.append(f"mystery_blueprint_validation_failed: {e}")

        horror_blueprint: dict = {}
        if genre == "horror_adult":
            try:
                horror_blueprint = generate_horror_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("horror_blueprint")
            except Exception as e:
                passes.append(f"horror_blueprint_failed: {e}")

            if horror_blueprint:
                try:
                    horror_blueprint = validate_horror_blueprint_ai(
                        blueprint=horror_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("horror_blueprint_validated")
                except Exception as e:
                    passes.append(f"horror_blueprint_validation_failed: {e}")

        sci_fi_blueprint: dict = {}
        if genre == "sci_fi":
            try:
                sci_fi_blueprint = generate_sci_fi_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("sci_fi_blueprint")
            except Exception as e:
                passes.append(f"sci_fi_blueprint_failed: {e}")

            if sci_fi_blueprint:
                try:
                    sci_fi_blueprint = validate_sci_fi_blueprint_ai(
                        blueprint=sci_fi_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("sci_fi_blueprint_validated")
                except Exception as e:
                    passes.append(f"sci_fi_blueprint_validation_failed: {e}")

        horror_kids_blueprint: dict = {}
        if genre == "horror":
            try:
                horror_kids_blueprint = generate_horror_kids_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("horror_kids_blueprint")
            except Exception as e:
                passes.append(f"horror_kids_blueprint_failed: {e}")

            if horror_kids_blueprint:
                try:
                    horror_kids_blueprint = validate_horror_kids_blueprint_ai(
                        blueprint=horror_kids_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("horror_kids_blueprint_validated")
                except Exception as e:
                    passes.append(f"horror_kids_blueprint_validation_failed: {e}")

        fantasy_blueprint: dict = {}
        if genre == "fantasy":
            try:
                fantasy_blueprint = generate_fantasy_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("fantasy_blueprint")
            except Exception as e:
                passes.append(f"fantasy_blueprint_failed: {e}")

            if fantasy_blueprint:
                try:
                    fantasy_blueprint = validate_fantasy_blueprint_ai(
                        blueprint=fantasy_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("fantasy_blueprint_validated")
                except Exception as e:
                    passes.append(f"fantasy_blueprint_validation_failed: {e}")

        prose_story = generate_prose_story_ai(
            hero_name=hero_name,
            theme=proj.theme,
            logline=option.logline,
            outline=option.outline,
            personalization=main.personalization,
            genre=genre,
            additional_characters=support_char_info or None,
            page_count=page_count,
            comedy_blueprint=comedy_blueprint or None,
            adventure_blueprint=adventure_blueprint or None,
            mystery_blueprint=mystery_blueprint or None,
            horror_blueprint=horror_blueprint or None,
            sci_fi_blueprint=sci_fi_blueprint or None,
            horror_kids_blueprint=horror_kids_blueprint or None,
            fantasy_blueprint=fantasy_blueprint or None,
        )
        passes.append("prose")

        target_wc = 2000 if page_count <= 24 else 3000
        try:
            prose_story = expand_prose_if_short_ai(
                prose_story=prose_story,
                target_word_count=target_wc,
                hero_name=hero_name,
                max_passes=3 if page_count > 24 else 2,
            )
            passes.append("prose_expansion")
        except Exception as e:
            passes.append(f"prose_expansion_failed: {e}")

        try:
            prose_story = check_story_pacing_ai(
                prose_story=prose_story, hero_name=hero_name, genre=genre, page_count=page_count,
            )
            passes.append("pacing_check")
        except Exception as e:
            passes.append(f"pacing_check_failed: {e}")

        try:
            prose_story = check_story_continuity_ai(
                prose_story=prose_story, hero_name=hero_name, page_count=page_count,
            )
            passes.append("continuity_check")
        except Exception as e:
            passes.append(f"continuity_check_failed: {e}")

        try:
            prose_story = check_story_specificity_ai(
                prose_story=prose_story, hero_name=hero_name, page_count=page_count,
            )
            passes.append("specificity_check")
        except Exception as e:
            passes.append(f"specificity_check_failed: {e}")

        try:
            prose_story = check_story_arc_shape_ai(
                prose_story=prose_story, hero_name=hero_name, genre=genre, page_count=page_count,
            )
            passes.append("arc_shape_check")
        except Exception as e:
            passes.append(f"arc_shape_check_failed: {e}")

        if genre == "adventure" and adventure_blueprint:
            try:
                prose_story = check_adventure_fidelity_ai(
                    prose_story=prose_story, blueprint=adventure_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("adventure_fidelity_check")
            except Exception as e:
                passes.append(f"adventure_fidelity_check_failed: {e}")

        if genre == "mystery" and mystery_blueprint:
            try:
                prose_story = check_mystery_fidelity_ai(
                    prose_story=prose_story, blueprint=mystery_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("mystery_fidelity_check")
            except Exception as e:
                passes.append(f"mystery_fidelity_check_failed: {e}")

        if genre == "comedy" and comedy_blueprint:
            try:
                prose_story = check_comedy_fidelity_ai(
                    prose_story=prose_story, blueprint=comedy_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("comedy_fidelity_check")
            except Exception as e:
                passes.append(f"comedy_fidelity_check_failed: {e}")

        if genre == "horror_adult" and horror_blueprint:
            try:
                prose_story = check_horror_fidelity_ai(
                    prose_story=prose_story, blueprint=horror_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("horror_fidelity_check")
            except Exception as e:
                passes.append(f"horror_fidelity_check_failed: {e}")

        if genre == "sci_fi" and sci_fi_blueprint:
            try:
                prose_story = check_sci_fi_fidelity_ai(
                    prose_story=prose_story, blueprint=sci_fi_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("sci_fi_fidelity_check")
            except Exception as e:
                passes.append(f"sci_fi_fidelity_check_failed: {e}")

        if genre == "horror" and horror_kids_blueprint:
            try:
                prose_story = check_horror_kids_fidelity_ai(
                    prose_story=prose_story, blueprint=horror_kids_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("horror_kids_fidelity_check")
            except Exception as e:
                passes.append(f"horror_kids_fidelity_check_failed: {e}")

        if genre == "fantasy" and fantasy_blueprint:
            try:
                prose_story = check_fantasy_fidelity_ai(
                    prose_story=prose_story, blueprint=fantasy_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("fantasy_fidelity_check")
            except Exception as e:
                passes.append(f"fantasy_fidelity_check_failed: {e}")

        if genre == "comedy":
            try:
                prose_story = punch_up_prose_ai(
                    prose_story=prose_story,
                    hero_name=hero_name,
                    page_count=page_count,
                    additional_characters=support_char_info or None,
                )
                passes.append("punch_up")
            except Exception as e:
                passes.append(f"punch_up_failed: {e}")

        return {
            "project_id": str(proj.id),
            "hero_name": hero_name,
            "genre": genre,
            "page_count": page_count,
            "passes_completed": passes,
            "comedy_blueprint": comedy_blueprint or None,
            "adventure_blueprint": adventure_blueprint or None,
            "mystery_blueprint": mystery_blueprint or None,
            "horror_blueprint": horror_blueprint or None,
            "sci_fi_blueprint": sci_fi_blueprint or None,
            "horror_kids_blueprint": horror_kids_blueprint or None,
            "fantasy_blueprint": fantasy_blueprint or None,
            "prose": prose_story,
        }
    finally:
        db.close()


@router.post("/prose")
def generate_prose_preview(
    project_id: str,
    page_count_override: int = Query(None, alias="page_count", ge=12, le=72, description="Override panel count (default: project setting)"),
    db: Session = Depends(get_db),
):
    """
    Starts the full writing pipeline (blueprint -> prose -> pacing check ->
    continuity check -> specificity check -> punch-up) as a background job and
    returns immediately with a job_id — poll GET .../story/jobs/{job_id} for
    the result. Does NOT generate any images — zero LlamaGen credits.
    """
    proj = _get_project_or_404(db, project_id)

    if not proj.selected_story_option_id:
        raise HTTPException(400, "No story option selected — call /story/choose first")

    option = db.get(models.StoryOption, proj.selected_story_option_id)
    if not option:
        raise HTTPException(404, "Selected story option not found")

    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == proj.id, models.Character.role == "main")
        .first()
    )
    if not main:
        raise HTTPException(400, "No main character found for this project")

    job_id = _start_job(_run_prose_pipeline, project_id, page_count_override)
    return {
        "job_id": job_id,
        "status": "started",
        "poll_url": f"/projects/{project_id}/story/jobs/{job_id}",
    }


def _run_panels_pipeline(project_id: str, page_count_override: int | None, use_claude: bool = False) -> dict:
    """Runs in a background thread — opens its own DB session."""
    from app.services.book_generation import _build_character_lock

    db = SessionLocal()
    try:
        proj = db.get(models.Project, uuid.UUID(project_id))
        option = db.get(models.StoryOption, proj.selected_story_option_id)
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == proj.id, models.Character.role == "main")
            .first()
        )

        hero_name = main.name
        genre = proj.genre or "adventure"
        page_count = page_count_override or proj.panel_count or 24
        character_lock = _build_character_lock(main)
        support_char_info = _build_support_char_info(db, proj.id)

        passes = []

        comedy_blueprint: dict = {}
        if genre == "comedy":
            try:
                comedy_blueprint = generate_comedy_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("comedy_blueprint")
            except Exception as e:
                passes.append(f"comedy_blueprint_failed: {e}")

            if comedy_blueprint:
                try:
                    comedy_blueprint = validate_comedy_blueprint_ai(
                        blueprint=comedy_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("comedy_blueprint_validated")
                except Exception as e:
                    passes.append(f"comedy_blueprint_validation_failed: {e}")

        adventure_blueprint: dict = {}
        if genre == "adventure":
            try:
                adventure_blueprint = generate_adventure_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("adventure_blueprint")
            except Exception as e:
                passes.append(f"adventure_blueprint_failed: {e}")

            if adventure_blueprint:
                try:
                    adventure_blueprint = validate_adventure_blueprint_ai(
                        blueprint=adventure_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("adventure_blueprint_validated")
                except Exception as e:
                    passes.append(f"adventure_blueprint_validation_failed: {e}")

        mystery_blueprint: dict = {}
        if genre == "mystery":
            try:
                mystery_blueprint = generate_mystery_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("mystery_blueprint")
            except Exception as e:
                passes.append(f"mystery_blueprint_failed: {e}")

            if mystery_blueprint:
                try:
                    mystery_blueprint = validate_mystery_blueprint_ai(
                        blueprint=mystery_blueprint,
                        hero_name=hero_name,
                        page_count=page_count,
                    )
                    passes.append("mystery_blueprint_validated")
                except Exception as e:
                    passes.append(f"mystery_blueprint_validation_failed: {e}")

        horror_blueprint: dict = {}
        if genre == "horror_adult":
            try:
                horror_blueprint = generate_horror_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("horror_blueprint")
            except Exception as e:
                passes.append(f"horror_blueprint_failed: {e}")

            if horror_blueprint:
                try:
                    horror_blueprint = validate_horror_blueprint_ai(
                        blueprint=horror_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("horror_blueprint_validated")
                except Exception as e:
                    passes.append(f"horror_blueprint_validation_failed: {e}")

        sci_fi_blueprint: dict = {}
        if genre == "sci_fi":
            try:
                sci_fi_blueprint = generate_sci_fi_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("sci_fi_blueprint")
            except Exception as e:
                passes.append(f"sci_fi_blueprint_failed: {e}")

            if sci_fi_blueprint:
                try:
                    sci_fi_blueprint = validate_sci_fi_blueprint_ai(
                        blueprint=sci_fi_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("sci_fi_blueprint_validated")
                except Exception as e:
                    passes.append(f"sci_fi_blueprint_validation_failed: {e}")

        horror_kids_blueprint: dict = {}
        if genre == "horror":
            try:
                horror_kids_blueprint = generate_horror_kids_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("horror_kids_blueprint")
            except Exception as e:
                passes.append(f"horror_kids_blueprint_failed: {e}")

            if horror_kids_blueprint:
                try:
                    horror_kids_blueprint = validate_horror_kids_blueprint_ai(
                        blueprint=horror_kids_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("horror_kids_blueprint_validated")
                except Exception as e:
                    passes.append(f"horror_kids_blueprint_validation_failed: {e}")

        fantasy_blueprint: dict = {}
        if genre == "fantasy":
            try:
                fantasy_blueprint = generate_fantasy_blueprint_ai(
                    hero_name=hero_name,
                    logline=option.logline,
                    outline=option.outline,
                    personalization=main.personalization,
                    additional_characters=support_char_info or None,
                    page_count=page_count,
                )
                passes.append("fantasy_blueprint")
            except Exception as e:
                passes.append(f"fantasy_blueprint_failed: {e}")

            if fantasy_blueprint:
                try:
                    fantasy_blueprint = validate_fantasy_blueprint_ai(
                        blueprint=fantasy_blueprint,
                        hero_name=hero_name,
                    )
                    passes.append("fantasy_blueprint_validated")
                except Exception as e:
                    passes.append(f"fantasy_blueprint_validation_failed: {e}")

        prose_story = generate_prose_story_ai(
            hero_name=hero_name,
            theme=proj.theme,
            logline=option.logline,
            outline=option.outline,
            personalization=main.personalization,
            genre=genre,
            additional_characters=support_char_info or None,
            page_count=page_count,
            comedy_blueprint=comedy_blueprint or None,
            adventure_blueprint=adventure_blueprint or None,
            mystery_blueprint=mystery_blueprint or None,
            horror_blueprint=horror_blueprint or None,
            sci_fi_blueprint=sci_fi_blueprint or None,
            horror_kids_blueprint=horror_kids_blueprint or None,
            fantasy_blueprint=fantasy_blueprint or None,
        )
        passes.append("prose")

        target_wc = 2000 if page_count <= 24 else 3000
        try:
            prose_story = expand_prose_if_short_ai(
                prose_story=prose_story,
                target_word_count=target_wc,
                hero_name=hero_name,
                max_passes=3 if page_count > 24 else 2,
            )
            passes.append("prose_expansion")
        except Exception as e:
            passes.append(f"prose_expansion_failed: {e}")

        try:
            prose_story = check_story_pacing_ai(
                prose_story=prose_story, hero_name=hero_name, genre=genre, page_count=page_count,
            )
            passes.append("pacing_check")
        except Exception as e:
            passes.append(f"pacing_check_failed: {e}")

        try:
            prose_story = check_story_continuity_ai(
                prose_story=prose_story, hero_name=hero_name, page_count=page_count,
            )
            passes.append("continuity_check")
        except Exception as e:
            passes.append(f"continuity_check_failed: {e}")

        try:
            prose_story = check_story_specificity_ai(
                prose_story=prose_story, hero_name=hero_name, page_count=page_count,
            )
            passes.append("specificity_check")
        except Exception as e:
            passes.append(f"specificity_check_failed: {e}")

        try:
            prose_story = check_story_arc_shape_ai(
                prose_story=prose_story, hero_name=hero_name, genre=genre, page_count=page_count,
            )
            passes.append("arc_shape_check")
        except Exception as e:
            passes.append(f"arc_shape_check_failed: {e}")

        if genre == "adventure" and adventure_blueprint:
            try:
                prose_story = check_adventure_fidelity_ai(
                    prose_story=prose_story, blueprint=adventure_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("adventure_fidelity_check")
            except Exception as e:
                passes.append(f"adventure_fidelity_check_failed: {e}")

        if genre == "mystery" and mystery_blueprint:
            try:
                prose_story = check_mystery_fidelity_ai(
                    prose_story=prose_story, blueprint=mystery_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("mystery_fidelity_check")
            except Exception as e:
                passes.append(f"mystery_fidelity_check_failed: {e}")

        if genre == "comedy" and comedy_blueprint:
            try:
                prose_story = check_comedy_fidelity_ai(
                    prose_story=prose_story, blueprint=comedy_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("comedy_fidelity_check")
            except Exception as e:
                passes.append(f"comedy_fidelity_check_failed: {e}")

        if genre == "horror_adult" and horror_blueprint:
            try:
                prose_story = check_horror_fidelity_ai(
                    prose_story=prose_story, blueprint=horror_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("horror_fidelity_check")
            except Exception as e:
                passes.append(f"horror_fidelity_check_failed: {e}")

        if genre == "sci_fi" and sci_fi_blueprint:
            try:
                prose_story = check_sci_fi_fidelity_ai(
                    prose_story=prose_story, blueprint=sci_fi_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("sci_fi_fidelity_check")
            except Exception as e:
                passes.append(f"sci_fi_fidelity_check_failed: {e}")

        if genre == "horror" and horror_kids_blueprint:
            try:
                prose_story = check_horror_kids_fidelity_ai(
                    prose_story=prose_story, blueprint=horror_kids_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("horror_kids_fidelity_check")
            except Exception as e:
                passes.append(f"horror_kids_fidelity_check_failed: {e}")

        if genre == "fantasy" and fantasy_blueprint:
            try:
                prose_story = check_fantasy_fidelity_ai(
                    prose_story=prose_story, blueprint=fantasy_blueprint, hero_name=hero_name, page_count=page_count,
                )
                passes.append("fantasy_fidelity_check")
            except Exception as e:
                passes.append(f"fantasy_fidelity_check_failed: {e}")

        if genre == "comedy":
            try:
                prose_story = punch_up_prose_ai(
                    prose_story=prose_story,
                    hero_name=hero_name,
                    page_count=page_count,
                    additional_characters=support_char_info or None,
                )
                passes.append("punch_up")
            except Exception as e:
                passes.append(f"punch_up_failed: {e}")

        npc_bible: list = []
        try:
            npc_bible = generate_story_character_bible_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                support_char_names=[sc["name"] for sc in support_char_info],
            )
            passes.append("npc_bible")
        except Exception as e:
            passes.append(f"npc_bible_failed: {e}")

        generated_pages = generate_preview_pages_ai(
            hero_name=hero_name,
            theme=proj.theme,
            logline=option.logline,
            outline=option.outline,
            character_brief=character_lock,
            page_count=page_count,
            personalization=main.personalization,
            genre=genre,
            is_preview=False,
            additional_characters=support_char_info or None,
            prose_story=prose_story,
            npc_bible=npc_bible,
            use_claude=use_claude,
        )
        passes.append("panel_script" + ("_claude" if use_claude else ""))

        panel_fidelity_debug: dict = {}
        if prose_story:
            try:
                generated_pages = check_panel_fidelity_ai(
                    pages=generated_pages, prose_story=prose_story, hero_name=hero_name, page_count=page_count,
                    debug_sink=panel_fidelity_debug, use_claude=use_claude,
                )
                passes.append("panel_fidelity_check")
            except Exception as e:
                passes.append(f"panel_fidelity_check_failed: {e}")

        try:
            generated_pages = check_panel_pacing_ai(
                pages=generated_pages, hero_name=hero_name, genre=genre, page_count=page_count, use_claude=use_claude,
            )
            passes.append("panel_pacing_check")
        except Exception as e:
            passes.append(f"panel_pacing_check_failed: {e}")

        return {
            "project_id": str(proj.id),
            "hero_name": hero_name,
            "genre": genre,
            "page_count": page_count,
            "passes_completed": passes,
            "comedy_blueprint": comedy_blueprint or None,
            "adventure_blueprint": adventure_blueprint or None,
            "mystery_blueprint": mystery_blueprint or None,
            "horror_blueprint": horror_blueprint or None,
            "sci_fi_blueprint": sci_fi_blueprint or None,
            "horror_kids_blueprint": horror_kids_blueprint or None,
            "fantasy_blueprint": fantasy_blueprint or None,
            "prose_story": prose_story,
            "npc_bible": [{"name": n["name"], "description": n["description"]} for n in npc_bible],
            "pages": sorted(generated_pages, key=lambda x: x["page_number"]),
            "panel_fidelity_debug": panel_fidelity_debug or None,
        }
    finally:
        db.close()


@router.post("/panels-preview")
def generate_panels_preview(
    project_id: str,
    page_count_override: int = Query(None, alias="page_count", ge=12, le=72, description="Override panel count (default: project setting)"),
    use_claude: bool = Query(False, description="Test path: use Claude instead of gpt-4o-mini for the panel-adaptation call"),
    db: Session = Depends(get_db),
):
    """
    Starts the full text pipeline (blueprint -> prose -> pacing check ->
    continuity check -> specificity check -> punch-up -> NPC bible -> panel
    script adaptation -> panel fidelity check -> panel pacing check) as a
    background job and returns immediately with a job_id — poll GET
    .../story/jobs/{job_id} for the result, which includes
    "characters_present" per panel. Does NOT generate any images or NPC
    reference portraits — zero LlamaGen credits.
    """
    proj = _get_project_or_404(db, project_id)

    if not proj.selected_story_option_id:
        raise HTTPException(400, "No story option selected — call /story/choose first")

    option = db.get(models.StoryOption, proj.selected_story_option_id)
    if not option:
        raise HTTPException(404, "Selected story option not found")

    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == proj.id, models.Character.role == "main")
        .first()
    )
    if not main:
        raise HTTPException(400, "No main character found for this project")

    job_id = _start_job(_run_panels_pipeline, project_id, page_count_override, use_claude)
    return {
        "job_id": job_id,
        "status": "started",
        "poll_url": f"/projects/{project_id}/story/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}")
def get_story_job(project_id: str, job_id: str):
    """Poll for the result of a /prose or /panels-preview background job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found — it may have expired or the server restarted")
    return job
