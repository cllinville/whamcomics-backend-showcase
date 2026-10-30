import stripe
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Header, HTTPException
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.db import get_db
from app.services.book_generation import generate_full_book_task, _send_order_confirmation_email, _send_shipped_email
import app.services.lulu_service as lulu_service

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
):
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured")

    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=stripe_signature,
            secret=settings.stripe_webhook_secret,
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

    if event["type"] == "checkout.session.completed":
        _handle_checkout_completed(event["data"]["object"], db, background_tasks)

    return {"ok": True}


def _handle_checkout_completed(
    session: dict, db: Session, background_tasks: BackgroundTasks
) -> None:
    metadata = session.get("metadata", {})
    project_id_str = metadata.get("project_id")

    if not project_id_str:
        print("⚠️  checkout.session.completed missing project_id in metadata")
        return

    try:
        project_id = UUID(project_id_str)
    except ValueError:
        print(f"⚠️  Invalid project_id in Stripe metadata: {project_id_str}")
        return

    project = db.get(models.Project, project_id)
    if not project:
        print(f"⚠️  Project {project_id} not found after Stripe payment")
        return

    # Capture customer email from Stripe session
    customer_details = session.get("customer_details") or {}
    email = customer_details.get("email")
    if email:
        project.customer_email = email

    # Capture shipping address for print orders (check both field names Stripe uses)
    shipping = session.get("shipping_details") or session.get("shipping") or {}
    if shipping:
        addr = shipping.get("address", {})
        project.shipping_address = {
            "name": shipping.get("name"),
            "line1": addr.get("line1"),
            "line2": addr.get("line2"),
            "city": addr.get("city"),
            "state": addr.get("state"),
            "postal_code": addr.get("postal_code"),
            "country": addr.get("country"),
            "phone": customer_details.get("phone"),  # required by Lulu
            "email": email,                          # required by Lulu
        }

    # Set selected story option if not already stored
    story_option_id_str = metadata.get("story_option_id")
    if story_option_id_str and not project.selected_story_option_id:
        try:
            project.selected_story_option_id = UUID(story_option_id_str)
        except ValueError:
            pass

    # Store product type and panel count from Stripe metadata
    product_type = metadata.get("product_type")
    if product_type:
        project.product_type = product_type

    panel_count_str = metadata.get("panel_count")
    if panel_count_str:
        try:
            project.panel_count = int(panel_count_str)
        except ValueError:
            pass

    # Guard against duplicate webhook fires — only queue if not already paid/generating/done
    if project.status in (models.ProjectStatus.paid, models.ProjectStatus.generating_book, models.ProjectStatus.book_ready):
        print(f"⚠️  Project {project_id} already in status '{project.status}' — skipping duplicate webhook")
        return

    project.status = models.ProjectStatus.paid
    db.commit()

    # Send immediate payment confirmation email
    if email:
        main_char = (
            db.query(models.Character)
            .filter(models.Character.project_id == project.id, models.Character.role == "main")
            .first()
        )
        hero_name = main_char.name if main_char else "your hero"
        order_ref = str(project.id).replace("-", "")[-8:].upper()
        _send_order_confirmation_email(
            customer_email=email,
            hero_name=hero_name,
            product_type=product_type or project.product_type or "digital",
            panel_count=project.panel_count or 24,
            amount_cents=session.get("amount_total") or 0,
            order_ref=order_ref,
        )

    print(f"✅ Project {project_id} marked as paid — queuing book generation")
    background_tasks.add_task(generate_full_book_task, str(project_id))


@router.post("/lulu")
async def lulu_webhook(request: Request, db: Session = Depends(get_db)):
    raw = await request.body()
    signature = request.headers.get("Lulu-HMAC-SHA256")
    if not lulu_service.verify_webhook_signature(raw, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.get("topic") != "PRINT_JOB_STATUS_CHANGED":
        return {"ok": True}

    job = payload.get("data") or {}
    external_id = job.get("external_id")
    lulu_id = job.get("id")

    project = None
    if external_id:
        try:
            project = db.get(models.Project, UUID(str(external_id)))
        except ValueError:
            project = None
    if not project and lulu_id:
        project = (
            db.query(models.Project)
            .filter(models.Project.lulu_print_job_id == str(lulu_id))
            .first()
        )

    if not project:
        print(f"⚠️  Lulu webhook: no matching project for external_id={external_id} lulu_id={lulu_id}")
        return {"ok": True}

    status_name = (job.get("status") or {}).get("name")
    project.lulu_print_job_status = status_name
    if lulu_id and not project.lulu_print_job_id:
        project.lulu_print_job_id = str(lulu_id)
    db.commit()

    if status_name == "SHIPPED" and project.customer_email:
        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == project.id, models.Character.role == "main")
            .first()
        )
        hero_name = main.name if main else "your hero"

        tracking_urls: list[str] = []
        for line_item in job.get("line_items", []):
            urls = ((line_item.get("status") or {}).get("messages") or {}).get("tracking_urls")
            if isinstance(urls, str):
                tracking_urls.append(urls)
            elif isinstance(urls, list):
                tracking_urls.extend(urls)

        _send_shipped_email(project.customer_email, hero_name, tracking_urls)

    return {"ok": True}


@router.post("/fal-lora/{character_id}")
async def fal_lora_webhook(character_id: UUID, request: Request, db: Session = Depends(get_db)):
    """Called by fal.ai when LoRA training completes."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    char = db.get(models.Character, character_id)
    if not char:
        return {"ok": True}  # silently ignore unknown characters

    status = payload.get("status")
    if status == "OK":
        lora_url = (
            payload.get("payload", {})
            .get("diffusers_lora_file", {})
            .get("url")
        )
        if lora_url:
            char.fal_lora_url = lora_url
            char.fal_lora_status = "ready"
            db.add(char)
            db.commit()
            print(f"✅ LoRA training complete for character {character_id}")
        else:
            print(f"⚠️  LoRA webhook OK but no lora_url for character {character_id}")
    else:
        char.fal_lora_status = "failed"
        db.add(char)
        db.commit()
        print(f"❌ LoRA training failed for character {character_id}: {payload.get('error')}")

    return {"ok": True}
