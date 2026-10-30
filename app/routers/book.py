from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from app import models
from app.db import get_db
from app.s3 import generate_presigned_get_url

router = APIRouter(prefix="/projects/{project_id}/book", tags=["book"])


@router.get("")
def get_book_status(project_id: str, db: Session = Depends(get_db)):
    try:
        pid = UUID(project_id)
    except Exception:
        raise HTTPException(422, "project_id must be a UUID")

    project = db.get(models.Project, pid)
    if not project:
        raise HTTPException(404, "Project not found")

    if project.status not in (
        models.ProjectStatus.paid,
        models.ProjectStatus.generating_book,
        models.ProjectStatus.book_ready,
        models.ProjectStatus.book_failed,
        models.ProjectStatus.delivered,
    ):
        raise HTTPException(
            400,
            f"Project has not been paid yet (status: {project.status})"
        )

    response = {
        "project_id": str(project.id),
        "status": project.status,
        "digital_download_url": None,
        "print_download_url": None,
        "print_cover_download_url": None,
    }

    # Download links only appear once an admin has approved the book via
    # /admin/review/{project_id} (status flips book_ready -> delivered there).
    # Found a real gap: this is a public, unauthenticated endpoint that the
    # frontend polls throughout generation, and it was handing out live
    # presigned download links the instant book_ready was reached -- the
    # customer's own frontend would show a working download button immediately,
    # with zero human review, even though the delivery EMAIL was correctly
    # held back pending approval. Gating on book_ready only blocked the email,
    # not the much more direct path a customer already has open in their browser.
    if project.status == models.ProjectStatus.delivered:
        if project.book_pdf_key:
            response["digital_download_url"] = generate_presigned_get_url(
                project.book_pdf_key, expires_in=86400 * 7
            )
        if project.book_interior_pdf_key:
            response["print_download_url"] = generate_presigned_get_url(
                project.book_interior_pdf_key, expires_in=86400 * 7
            )
        if project.book_cover_pdf_key:
            response["print_cover_download_url"] = generate_presigned_get_url(
                project.book_cover_pdf_key, expires_in=86400 * 7
            )

    return response
