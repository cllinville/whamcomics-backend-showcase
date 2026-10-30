from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models, schemas


class EmailUpdate(BaseModel):
    email: str

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("", response_model=schemas.ProjectOut)
def create_project(payload: schemas.ProjectCreate, db: Session = Depends(get_db)):
    if payload.genre not in schemas.VALID_GENRES:
        raise HTTPException(400, f"Invalid genre. Valid options: {sorted(schemas.VALID_GENRES)}")
    project = models.Project(theme=payload.theme, genre=payload.genre)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project

@router.get("/{project_id}", response_model=schemas.ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@router.patch("/{project_id}/email")
def save_email(project_id: str, body: EmailUpdate, db: Session = Depends(get_db)):
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    project.customer_email = body.email
    db.commit()
    return {"ok": True}
