from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..db import get_db
from .. import models, schemas
from ..schemas import CharacterPhotoReplace
import uuid
from app.services.ai import generate_character_reference_ai, extract_appearance_from_photo
from app.services import fal_service
from app.s3 import download_bytes, upload_bytes, make_key

router = APIRouter(prefix="/projects/{project_id}/characters", tags=["characters"])

def _get_project_or_404(db: Session, project_id: str) -> models.Project:
    try:
        pid = uuid.UUID(project_id)
    except Exception:
        raise HTTPException(422, "project_id must be a UUID")
    proj = db.get(models.Project, pid)
    if not proj:
        raise HTTPException(404, "Project not found")
    return proj

@router.post("", response_model=schemas.CharacterOut)
def create_character(project_id: str, payload: schemas.CharacterCreate, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)
    try:
        # Main character: replace any existing main (keep support chars)
        # Support character: just add alongside the main character
        if payload.role == "main":
            db.query(models.Character).filter(
                models.Character.project_id == proj.id,
                models.Character.role == "main",
            ).delete()
            db.flush()

        char = models.Character(
            project_id=proj.id,
            name=payload.name,
            role=payload.role,
            age=payload.age,
            traits=payload.traits or {},
            photo_keys=payload.photo_keys or [],
        )
        db.add(char)
        db.commit()
        db.refresh(char)
        return char
    except Exception as e:
        db.rollback()
        raise HTTPException(400, f"Could not create character: {e}")

@router.get("", response_model=List[schemas.CharacterOut])
def list_characters(project_id: str, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)
    return db.query(models.Character).filter(models.Character.project_id == proj.id).all()

@router.post("/{character_id}/personalization", response_model=schemas.CharacterOut)
@router.patch("/{character_id}/personalization", response_model=schemas.CharacterOut)
def update_personalization(
    project_id: str,
    character_id: str,
    payload: schemas.CharacterPersonalizationIn,
    db: Session = Depends(get_db),
):
    proj = _get_project_or_404(db, project_id)
    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")
    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")

    char.personalization = payload.model_dump(exclude_none=True)
    db.add(char)
    db.commit()
    db.refresh(char)
    return char


@router.post("/{character_id}/photos", response_model=schemas.CharacterOut)
def attach_photos(project_id: str, character_id: str, body: schemas.CharacterPhotoAttach, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)
    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")
    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")
    current = char.photo_keys or []
    for k in body.add_keys:
        if k not in current:
            current.append(k)
    char.photo_keys = current
    db.add(char)
    db.commit()
    db.refresh(char)
    return char

@router.put("/{character_id}/photos", response_model=schemas.CharacterOut)
def replace_photos(project_id: str, character_id: str, body: CharacterPhotoReplace, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)
    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")
    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")
    char.photo_keys = body.photo_keys
    db.add(char)
    db.commit()
    db.refresh(char)
    return char


@router.post("/{character_id}/reference", response_model=schemas.CharacterOut)
def generate_character_reference(
    project_id: str,
    character_id: str,
    db: Session = Depends(get_db)
):
    proj = _get_project_or_404(db, project_id)

    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")

    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")

    # 🚨 Make sure we have a photo
    if not char.photo_keys or len(char.photo_keys) == 0:
        raise HTTPException(400, "Character has no uploaded photo")

    photo_key = char.photo_keys[0]

    try:
        photo_bytes = download_bytes(photo_key)
        reference_bytes = generate_character_reference_ai(photo_bytes)

        reference_key = make_key(
            f"projects/{project_id}/characters/{character_id}/reference",
            "reference.png"
        )
        upload_bytes(reference_key, reference_bytes, content_type="image/png")

        char.reference_image_key = reference_key
        db.add(char)
        db.commit()
        db.refresh(char)

        return char

    except Exception as e:
        raise HTTPException(500, f"Reference generation failed: {str(e)}")


@router.post("/{character_id}/train", response_model=schemas.CharacterOut)
def trigger_training(project_id: str, character_id: str, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)

    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")

    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")

    if not char.photo_keys:
        raise HTTPException(400, "Character has no uploaded photos")

    if char.fal_lora_status in ("pending", "training"):
        raise HTTPException(400, f"Training already in progress (status: {char.fal_lora_status})")

    try:
        photo_bytes_list = [download_bytes(key) for key in char.photo_keys]

        # Extract appearance from all uploaded photos and store in traits
        try:
            appearance = extract_appearance_from_photo(
                photo_bytes_list[0],
                extra_photos=photo_bytes_list[1:] if len(photo_bytes_list) > 1 else None,
            )
            traits = dict(char.traits or {})
            traits["appearance"] = appearance
            char.traits = traits
        except Exception:
            pass  # appearance extraction is best-effort, don't block training

        from app.config import settings
        webhook_url = f"{settings.base_url}/webhooks/fal-lora/{char.id}"
        zip_url = fal_service.upload_training_images(photo_bytes_list)
        request_id = fal_service.trigger_lora_training(zip_url, webhook_url=webhook_url)

        char.fal_training_request_id = request_id
        char.fal_lora_status = "pending"
        char.fal_lora_url = None
        db.add(char)
        db.commit()
        db.refresh(char)

        return char

    except Exception as e:
        raise HTTPException(500, f"Training trigger failed: {str(e)}")


@router.get("/{character_id}/training-status", response_model=schemas.CharacterOut)
def check_training_status(project_id: str, character_id: str, db: Session = Depends(get_db)):
    proj = _get_project_or_404(db, project_id)

    try:
        cid = uuid.UUID(character_id)
    except Exception:
        raise HTTPException(422, "character_id must be a UUID")

    char = db.get(models.Character, cid)
    if not char or char.project_id != proj.id:
        raise HTTPException(404, "Character not found for this project")

    if not char.fal_training_request_id:
        raise HTTPException(400, "No training job found for this character")

    try:
        result = fal_service.get_training_status(char.fal_training_request_id)

        char.fal_lora_status = result["status"]
        if result["lora_url"]:
            char.fal_lora_url = result["lora_url"]

        db.add(char)
        db.commit()
        db.refresh(char)

        return char

    except Exception as e:
        raise HTTPException(500, f"Status check failed: {str(e)}")