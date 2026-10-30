from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import stripe

from app import models, schemas
from app.config import settings
from app.db import get_db

router = APIRouter(prefix="/checkout", tags=["checkout"])

# ── Pricing ──────────────────────────────────────────────────────────────────
# Set LAUNCH_SALE = False when you're ready to move to regular prices.
LAUNCH_SALE = True

PRODUCTS = {
    "digital": {
        24: {
            "name": "WhamComics Digital PDF — 16 Pages",
            "description": "Your personalized 16-page comic book — instant download as a high-quality PDF.",
            "price_cents": 2000 if LAUNCH_SALE else 2500,   # launch $20 / regular $25
        },
        48: {
            "name": "WhamComics Digital PDF — 32 Pages",
            "description": "Your personalized 32-page comic book — instant download as a high-quality PDF.",
            "price_cents": 3000 if LAUNCH_SALE else 3500,   # launch $30 / regular $35
        },
    },
    "print": {
        24: {
            "name": "WhamComics Printed Comic — 16 Pages",
            "description": "Your personalized 16-page comic book printed on authentic comic paper and shipped to you.",
            "price_cents": 3500 if LAUNCH_SALE else 4000,   # launch $35 / regular $40
        },
        48: {
            "name": "WhamComics Printed Comic — 32 Pages",
            "description": "Your personalized 32-page comic book printed on authentic comic paper and shipped to you.",
            "price_cents": 5000 if LAUNCH_SALE else 5900,   # launch $50 / regular $59
        },
    },
}


@router.post("", response_model=dict)
def start_checkout(payload: schemas.CheckoutRequest, db: Session = Depends(get_db)):
    if not settings.stripe_secret:
        raise HTTPException(status_code=500, detail="Stripe is not configured")

    if payload.product_type not in schemas.VALID_PRODUCT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid product_type. Choose: {sorted(schemas.VALID_PRODUCT_TYPES)}"
        )

    if payload.panel_count not in schemas.VALID_PANEL_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid panel_count. Choose: {sorted(schemas.VALID_PANEL_COUNTS)}"
        )

    project = db.get(models.Project, payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in (
        models.ProjectStatus.preview_ready,
        models.ProjectStatus.awaiting_payment,
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Project must have a completed preview before checkout (status: {project.status})"
        )

    option = db.get(models.StoryOption, payload.story_option_id) if payload.story_option_id else None
    if not option or option.project_id != project.id:
        if project.selected_story_option_id:
            option = db.get(models.StoryOption, project.selected_story_option_id)
        if not option or option.project_id != project.id:
            raise HTTPException(status_code=404, detail="Story option not found for this project")

    main = (
        db.query(models.Character)
        .filter(models.Character.project_id == project.id)
        .order_by(models.Character.role.desc())
        .first()
    )
    hero_name = main.name if main else "Your hero"

    product = PRODUCTS[payload.product_type][payload.panel_count]
    stripe.api_key = settings.stripe_secret

    session_params = dict(
        mode="payment",
        success_url=settings.stripe_success_url,
        cancel_url=settings.stripe_cancel_url,
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": product["price_cents"],
                    "product_data": {
                        "name": f"{product['name']} — {hero_name}",
                        "description": product["description"],
                    },
                },
                "quantity": 1,
            }
        ],
        metadata={
            "project_id":      str(project.id),
            "story_option_id": str(option.id),
            "product_type":    payload.product_type,
            "panel_count":     str(payload.panel_count),
        },
    )

    if payload.product_type == "print":
        session_params["shipping_address_collection"] = {
            "allowed_countries": ["US", "CA", "GB", "AU", "NZ", "IE"]
        }
        # Lulu requires a phone number on the shipping address for every print job.
        session_params["phone_number_collection"] = {"enabled": True}

    session = stripe.checkout.Session.create(**session_params)

    project.status = models.ProjectStatus.awaiting_payment
    project.product_type = payload.product_type
    project.panel_count = payload.panel_count
    db.commit()

    return {
        "checkout_url": session.url,
        "session_id":   session.id,
        "product_type": payload.product_type,
        "panel_count":  payload.panel_count,
        "price_cents":  product["price_cents"],
    }
