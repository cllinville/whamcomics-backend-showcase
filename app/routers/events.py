from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from uuid import UUID

from app.db import get_db
from app import models

router = APIRouter(tags=["events"])


class EventPayload(BaseModel):
    project_id: Optional[str] = None
    event: str
    meta: Optional[dict] = None


@router.post("/events")
def log_event(payload: EventPayload, db: Session = Depends(get_db)):
    project_uuid = None
    if payload.project_id:
        try:
            project_uuid = UUID(payload.project_id)
        except ValueError:
            pass

    ev = models.FunnelEvent(
        project_id=project_uuid,
        event=payload.event,
        meta=payload.meta,
    )
    db.add(ev)
    db.commit()
    return {"ok": True}
