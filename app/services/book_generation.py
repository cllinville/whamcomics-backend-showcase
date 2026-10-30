import base64
import concurrent.futures
import json
import time
from io import BytesIO
from typing import Optional
from uuid import UUID

from PIL import Image
from sqlalchemy.orm import Session

from app import models
from app.db import SessionLocal
from app.s3 import upload_bytes, download_bytes, make_key, generate_presigned_get_url
from app.services.ai import (
    GENRE_STYLES,
    generate_comedy_blueprint_ai,
    validate_comedy_blueprint_ai,
    generate_adventure_blueprint_ai,
    validate_adventure_blueprint_ai,
    generate_mystery_blueprint_ai,
    validate_mystery_blueprint_ai,
    generate_sci_fi_blueprint_ai,
    validate_sci_fi_blueprint_ai,
    generate_horror_blueprint_ai,
    validate_horror_blueprint_ai,
    generate_horror_kids_blueprint_ai,
    validate_horror_kids_blueprint_ai,
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
    check_sci_fi_fidelity_ai,
    check_horror_fidelity_ai,
    check_horror_kids_fidelity_ai,
    check_fantasy_fidelity_ai,
    extract_mystery_suspect_names_ai,
    verify_and_fix_mystery_suspect_closure_ai,
    check_comedy_fidelity_ai,
    check_panel_fidelity_ai,
    check_panel_pacing_ai,
    check_nickname_usage_ai,
    punch_up_prose_ai,
    generate_story_character_bible_ai,
    generate_preview_pages_ai,
    generate_preview_pages_chunked_ai,
    generate_character_reference_ai,
    generate_image_with_reference,
    build_cartoon_page_prompt,
    build_back_cover_prompt,
    generate_back_cover_copy_ai,
    generate_book_title_ai,
    generate_cover_line_ai,
    verify_panel_image_matches_narration_ai,
)
from app.services.comic_layout import compose_comic_page, compose_single_panel_page
from app.services.speech_bubbles import add_cover_text, add_back_cover_text
from app.config import settings
import app.services.fal_service as fal_service
import app.services.llamagen_service as llamagen_service


def _normalize_dialogue(dialogue) -> list:
    if not dialogue:
        return []
    if isinstance(dialogue, list):
        out = []
        for item in dialogue:
            if isinstance(item, dict):
                out.append({
                    "speaker": item.get("speaker", ""),
                    "line": item.get("line") or item.get("text", ""),
                    "side": item.get("side", ""),
                })
        # Sort left speakers first so bubble positions match image layout
        if any(d["side"] for d in out):
            out.sort(key=lambda d: 0 if d.get("side", "").lower() == "left" else 1)
        return out
    return []


# Print interior trim conforms to the Lulu pod_package_id in settings.lulu_pod_package_id
# (default 6.625" x 10.25" comic trim) — verify against Lulu's sandbox before a real order ships.
PRINT_DPI = 150
PRINT_TRIM_W_IN = 6.625
PRINT_TRIM_H_IN = 10.25
PRINT_W = int(PRINT_TRIM_W_IN * PRINT_DPI)  # 993 px
PRINT_H = int(PRINT_TRIM_H_IN * PRINT_DPI)  # 1537 px


def _assemble_digital_pdf(image_bytes_list: list[bytes]) -> bytes:
    """Digital PDF — cover + interior pages (spreads + splashes) + back cover."""
    images = [Image.open(BytesIO(b)).convert("RGB") for b in image_bytes_list]
    if not images:
        raise RuntimeError("No pages to assemble into PDF")
    out = BytesIO()
    images[0].save(
        out,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=PRINT_DPI,
    )
    return out.getvalue()


def _fit_to_print_page(image_bytes: bytes, blank: bool = False) -> Image.Image:
    """Scale spread to fill print page width; canvas aspect ratio ensures near-zero vertical bars."""
    page = Image.new("RGB", (PRINT_W, PRINT_H), (255, 255, 255))
    if blank:
        return page
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    scale = PRINT_W / img.width
    new_h = int(img.height * scale)
    img = img.resize((PRINT_W, new_h), Image.LANCZOS)
    top = max(0, (PRINT_H - new_h) // 2)
    page.paste(img, (0, top))
    return page


def _assemble_interior_pdf(spread_bytes_list: list[bytes]) -> bytes:
    """Interior-only PDF for Lulu (no covers) — one print page per spread/splash."""
    pages = [_fit_to_print_page(b) for b in spread_bytes_list]
    if not pages:
        raise RuntimeError("No interior pages to assemble into PDF")
    out = BytesIO()
    pages[0].save(
        out,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=PRINT_DPI,
    )
    return out.getvalue()


def _assemble_cover_pdf(front_cover_bytes: bytes, back_cover_bytes: bytes, interior_page_count: int) -> bytes:
    """
    Wrap cover PDF for Lulu: back cover | spine | front cover, sized via Lulu's
    /cover-dimensions/ endpoint for the configured pod_package_id + interior page count.
    Falls back to a spineless (saddle-stitch) side-by-side spread if that call fails.
    """
    import app.services.lulu_service as lulu_service

    try:
        dims = lulu_service.get_cover_dimensions(interior_page_count, unit="pt")
        cover_w_px = int(float(dims["width"]) / 72 * PRINT_DPI)
        cover_h_px = int(float(dims["height"]) / 72 * PRINT_DPI)
    except Exception as e:
        print(f"  [lulu] cover-dimensions lookup failed, using spineless fallback: {e}")
        cover_w_px = PRINT_W * 2
        cover_h_px = PRINT_H

    canvas = Image.new("RGB", (cover_w_px, cover_h_px), (255, 255, 255))

    def _place(img_bytes: bytes, target_w: int, x: int) -> None:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        scale = target_w / img.width
        img = img.resize((target_w, int(img.height * scale)), Image.LANCZOS)
        canvas.paste(img, (x, max(0, (cover_h_px - img.height) // 2)))

    _place(back_cover_bytes, PRINT_W, 0)
    _place(front_cover_bytes, PRINT_W, cover_w_px - PRINT_W)

    out = BytesIO()
    canvas.save(out, format="PDF", resolution=PRINT_DPI)
    return out.getvalue()


def rebuild_pdf_from_panels(project_id: str) -> dict:
    """
    Recomposite all spreads from stored raw panel PNGs and rebuild the PDF.
    Used by the admin review flow after regenerating individual panels.
    Returns {"digital_key": ..., "print_key": ...} with the updated S3 keys.
    """
    from app.services.comic_layout import compose_comic_page, compose_single_panel_page
    from app.services.ai import GENRE_STYLES

    db = SessionLocal()
    try:
        project = db.get(models.Project, UUID(project_id))
        if not project:
            raise ValueError(f"Project {project_id} not found")

        main = (
            db.query(models.Character)
            .filter(models.Character.project_id == project.id, models.Character.role == "main")
            .first()
        )
        hero_name = main.name if main else ""
        panel_count = project.panel_count or 48
        genre_style = GENRE_STYLES.get(project.genre or "adventure", GENRE_STYLES["adventure"])

        # Load metadata
        dialogue_map = json.loads(download_bytes(f"panels/{project.id}/dialogue.json"))
        narration_map = json.loads(download_bytes(f"panels/{project.id}/narration.json"))

        # Composite pages: [A+B → spread] [C → full-page splash] repeating
        interior_pages = []
        num_groups = panel_count // 3
        for group_num in range(num_groups):
            a_num = group_num * 3 + 1
            b_num = group_num * 3 + 2
            c_num = group_num * 3 + 3
            panel_a_bytes = download_bytes(f"panels/{project.id}/panel-{a_num}.png")
            panel_b_bytes = download_bytes(f"panels/{project.id}/panel-{b_num}.png")
            panel_c_bytes = download_bytes(f"panels/{project.id}/panel-{c_num}.png")
            da = _normalize_dialogue(dialogue_map.get(str(a_num), []))
            db_ = _normalize_dialogue(dialogue_map.get(str(b_num), []))
            dc = _normalize_dialogue(dialogue_map.get(str(c_num), []))
            na = narration_map.get(str(a_num), "")
            nb = narration_map.get(str(b_num), "")
            nc = narration_map.get(str(c_num), "")
            spread_bytes = compose_comic_page(
                panel_a_bytes, panel_b_bytes,
                dialogue_a=da, dialogue_b=db_,
                narration_a=na, narration_b=nb,
                hero_name=hero_name,
            )
            interior_pages.append(spread_bytes)
            splash_bytes = compose_single_panel_page(
                panel_c_bytes,
                dialogue=dc,
                narration=nc,
                hero_name=hero_name,
            )
            interior_pages.append(splash_bytes)

        front_cover_bytes = download_bytes(f"panels/{project.id}/cover.png")
        back_cover_bytes = download_bytes(f"panels/{project.id}/back-cover.png")

        result = {}

        # Digital PDF
        digital_pages = [front_cover_bytes] + interior_pages + [back_cover_bytes]
        digital_pdf_bytes = _assemble_digital_pdf(digital_pages)
        digital_key = f"books/{project.id}/comic-digital.pdf"
        upload_bytes(key=digital_key, data=digital_pdf_bytes, content_type="application/pdf")
        project.book_pdf_key = digital_key
        result["digital_key"] = digital_key

        # Print PDFs (if applicable) — separate interior + wrap cover for Lulu
        if project.product_type == "print":
            interior_pdf_bytes = _assemble_interior_pdf(interior_pages)
            interior_key = f"books/{project.id}/comic-interior.pdf"
            upload_bytes(key=interior_key, data=interior_pdf_bytes, content_type="application/pdf")
            project.book_interior_pdf_key = interior_key
            result["interior_key"] = interior_key

            cover_pdf_bytes = _assemble_cover_pdf(front_cover_bytes, back_cover_bytes, len(interior_pages))
            cover_key = f"books/{project.id}/comic-cover.pdf"
            upload_bytes(key=cover_key, data=cover_pdf_bytes, content_type="application/pdf")
            project.book_cover_pdf_key = cover_key
            result["cover_key"] = cover_key

        db.commit()
        return result
    finally:
        db.close()


def _brevo_send(to_email: str, subject: str, html: str) -> None:
    from app.config import settings
    import httpx

    if not settings.brevo_api_key:
        print(f"[email] BREVO_API_KEY not set — skipping email to {to_email}")
        return

    try:
        response = httpx.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": settings.brevo_api_key, "Content-Type": "application/json"},
            json={
                "sender": {"name": "WhamComics", "email": "noreply@whamcomics.com"},
                "to": [{"email": to_email}],
                "subject": subject,
                "htmlContent": html,
            },
            timeout=15,
        )
        if response.status_code in (200, 201):
            print(f"[email] Sent to {to_email}")
        else:
            print(f"[email] Failed: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"[email] Error sending to {to_email}: {e}")


def _send_book_ready_email(customer_email: str, hero_name: str, download_url: str) -> None:
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h1 style="color:#6B21A8;">Your comic book is ready!</h1>
      <p>Hi there! Your personalized WhamComics comic book starring <strong>{hero_name}</strong> is ready to download.</p>
      <p style="margin:32px 0;">
        <a href="{download_url}"
           style="background:#6B21A8;color:white;padding:14px 28px;text-decoration:none;border-radius:8px;font-size:16px;font-weight:bold;">
          Download Your Comic Book
        </a>
      </p>
      <p style="color:#666;font-size:14px;">This download link expires in 7 days.</p>
      <p style="color:#666;font-size:14px;">Thank you for using WhamComics!</p>
    </div>
    """
    _brevo_send(customer_email, f"Your WhamComics comic starring {hero_name} is ready!", html)


def _send_print_order_email(customer_email: str, hero_name: str) -> None:
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h1 style="color:#6B21A8;">Your order is confirmed!</h1>
      <p>Hi there! We've received your order for a printed WhamComics comic book starring <strong>{hero_name}</strong>. Keep this email as your receipt.</p>
      <p style="margin-top:12px;">Your personalized comic is being reviewed and prepared for printing. We'll follow up with shipping details once your book is on its way.</p>
      <p style="color:#666;font-size:14px;margin-top:20px;">Thank you for using WhamComics!</p>
    </div>
    """
    _brevo_send(customer_email, f"Your WhamComics print order for {hero_name} is confirmed!", html)


def _send_admin_notification_email(project_id: str, hero_name: str, customer_email: str, product_type: str, quality_flags: list = None) -> None:
    from app.config import settings
    review_url = f"{settings.base_url}/admin/review/{project_id}?secret={settings.secret_key}"
    product_label = "print" if product_type == "print" else "digital"
    # Surfaces anything the generation pipeline itself wasn't fully confident
    # about (a panel-1 image that needed regenerating, a dedup pass that still
    # left repeated lines after its retries) right in the review-needed email,
    # instead of only ever reaching a Railway log line nobody's watching in
    # real time -- the point is to know about a likely quality issue before
    # the customer does, not just have it recorded somewhere.
    flags_html = ""
    if quality_flags:
        items = "".join(f"<li>{f}</li>" for f in quality_flags)
        flags_html = f"""
      <div style="background:#fff3cd;border:1px solid #ffecb5;border-radius:8px;padding:12px 16px;margin:20px 0;">
        <strong style="color:#856404;">⚠️ Worth a closer look:</strong>
        <ul style="color:#856404;margin:8px 0 0;padding-left:20px;">{items}</ul>
      </div>
        """
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h2 style="color:#ffd700;background:#111;padding:16px;border-radius:8px;">
        📚 New WhamComics book ready for review
      </h2>
      <p><strong>Hero:</strong> {hero_name}</p>
      <p><strong>Customer:</strong> {customer_email}</p>
      <p><strong>Product:</strong> {product_label}</p>
      <p><strong>Project ID:</strong> {project_id}</p>
      {flags_html}
      <p style="margin:28px 0;">
        <a href="{review_url}"
           style="background:#ffd700;color:#111;padding:14px 28px;text-decoration:none;border-radius:8px;font-size:16px;font-weight:bold;">
          Review &amp; Approve Comic
        </a>
      </p>
      <p style="color:#666;font-size:13px;">
        Once you approve, the delivery email will be sent to the customer automatically.
      </p>
    </div>
    """
    subject = f"[WhamComics] Review needed — {hero_name}'s comic is ready"
    if quality_flags:
        subject = f"[WhamComics] ⚠️ Review needed (flagged) — {hero_name}'s comic is ready"
    _brevo_send(settings.owner_email, subject, html)


def _send_book_failed_alert_email(project_id: str, hero_name: str, customer_email: str, product_type: str, reason: str) -> None:
    from app.config import settings
    review_url = f"{settings.base_url}/admin/review/{project_id}?secret={settings.secret_key}"
    product_label = "print" if product_type == "print" else "digital"
    reason_escaped = (reason or "Unknown error")[:1500]
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h2 style="color:#fff;background:#c0392b;padding:16px;border-radius:8px;">
        ⚠️ WhamComics book generation FAILED
      </h2>
      <p><strong>Hero:</strong> {hero_name}</p>
      <p><strong>Customer:</strong> {customer_email}</p>
      <p><strong>Product:</strong> {product_label}</p>
      <p><strong>Project ID:</strong> {project_id}</p>
      <p><strong>Reason:</strong></p>
      <pre style="background:#111;color:#f88;padding:12px;border-radius:8px;white-space:pre-wrap;font-size:13px;">{reason_escaped}</pre>
      <p style="margin:28px 0;">
        <a href="{review_url}"
           style="background:#c0392b;color:#fff;padding:14px 28px;text-decoration:none;border-radius:8px;font-size:16px;font-weight:bold;">
          Review &amp; Retry
        </a>
      </p>
      <p style="color:#666;font-size:13px;">
        This customer has already paid. Retrying does not charge them again.
      </p>
    </div>
    """
    _brevo_send(settings.owner_email, f"[WhamComics] ⚠️ Book generation failed — {hero_name}", html)


def _send_print_fulfillment_notification(
    hero_name: str, customer_email: str, shipping: dict, lulu_job_id: str, lulu_status: str
) -> None:
    from app.config import settings

    addr_parts = [
        shipping.get("name", ""),
        shipping.get("line1", ""),
        shipping.get("line2") or "",
        f"{shipping.get('city', '')}, {shipping.get('state', '')} {shipping.get('postal_code', '')}",
        shipping.get("country", ""),
    ]
    addr_html = "<br>".join(p for p in addr_parts if p.strip())

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h2>Print order submitted to Lulu — {hero_name}</h2>
      <p><strong>Customer email:</strong> {customer_email}</p>
      <p><strong>Lulu Print-Job ID:</strong> {lulu_job_id}</p>
      <p><strong>Status:</strong> {lulu_status}</p>
      <h3>Ship to:</h3>
      <p>{addr_html}</p>
      <p style="color:#666;font-size:14px;">Track progress and payment in your Lulu dashboard.</p>
    </div>
    """
    owner_email = settings.owner_email
    _brevo_send(owner_email, f"[WhamComics] Print order submitted to Lulu — {hero_name}", html)


def _send_shipped_email(customer_email: str, hero_name: str, tracking_urls: list[str]) -> None:
    links_html = "".join(
        f'<p><a href="{u}">{u}</a></p>' for u in tracking_urls
    ) or "<p>Tracking details will follow from the carrier shortly.</p>"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h1 style="color:#6B21A8;">Your comic is on its way!</h1>
      <p>Hi there! Your printed WhamComics comic book starring <strong>{hero_name}</strong> has shipped.</p>
      {links_html}
      <p style="color:#666;font-size:14px;">Thank you for using WhamComics!</p>
    </div>
    """
    _brevo_send(customer_email, f"Your WhamComics comic starring {hero_name} has shipped!", html)


def _send_order_confirmation_email(
    customer_email: str,
    hero_name: str,
    product_type: str,
    panel_count: int,
    amount_cents: int,
    order_ref: str,
) -> None:
    amount_str = f"${amount_cents / 100:.2f}"
    _panel_to_pages = {24: 16, 48: 32}
    page_count = _panel_to_pages.get(panel_count, panel_count)
    panel_label = f"{page_count}-page" if panel_count else ""
    product_label = "Digital Comic Book" if product_type == "digital" else "Printed Comic Book"

    if product_type == "print":
        next_steps = (
            "We're creating your personalized comic book now. Once your order ships, "
            "we'll send you tracking information by email."
        )
    else:
        next_steps = (
            "We're creating your personalized comic book now — please allow up to 4 hours for delivery. "
            "You'll receive another email with your download link as soon as it's ready."
        )

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;color:#1a1a1a;">
      <div style="background:#6B21A8;padding:24px;border-radius:8px 8px 0 0;text-align:center;">
        <h1 style="color:white;margin:0;font-size:24px;">Order Confirmed!</h1>
      </div>
      <div style="background:#f9f9f9;padding:24px;border-radius:0 0 8px 8px;border:1px solid #e5e5e5;border-top:none;">
        <p style="margin-top:0;">Thank you for your WhamComics order! Here's your receipt:</p>
        <table style="width:100%;border-collapse:collapse;margin:16px 0;">
          <tr style="border-bottom:1px solid #e5e5e5;">
            <td style="padding:10px 0;color:#666;">Order reference</td>
            <td style="padding:10px 0;text-align:right;font-weight:bold;font-family:monospace;">#{order_ref}</td>
          </tr>
          <tr style="border-bottom:1px solid #e5e5e5;">
            <td style="padding:10px 0;color:#666;">Product</td>
            <td style="padding:10px 0;text-align:right;">{panel_label} {product_label}</td>
          </tr>
          <tr style="border-bottom:1px solid #e5e5e5;">
            <td style="padding:10px 0;color:#666;">Starring</td>
            <td style="padding:10px 0;text-align:right;">{hero_name}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;color:#666;">Total paid</td>
            <td style="padding:10px 0;text-align:right;font-weight:bold;font-size:18px;color:#6B21A8;">{amount_str}</td>
          </tr>
        </table>
        <div style="background:#EDE9FE;border-radius:8px;padding:16px;margin-top:16px;">
          <p style="margin:0;color:#4C1D95;font-size:14px;">
            <strong>What happens next?</strong><br><br>
            {next_steps}
          </p>
        </div>
        <p style="color:#999;font-size:12px;margin-top:24px;margin-bottom:0;">
          Questions? Contact us at support@whamcomics.com
        </p>
      </div>
    </div>
    """
    _brevo_send(customer_email, f"Your WhamComics order is confirmed — ref #{order_ref}", html)


def generate_full_book_task(project_id: str) -> None:
    db = SessionLocal()
    try:
        _generate_full_book(project_id, db)
    except Exception as e:
        print(f"❌ Full book generation failed for project {project_id}: {e}")
        _mark_failed(project_id, reason=str(e))
    finally:
        db.close()


def _mark_failed(project_id: str, reason: str = "") -> None:
    db = SessionLocal()
    try:
        project = db.get(models.Project, UUID(project_id))
        if project:
            project.status = models.ProjectStatus.book_failed
            db.commit()
            try:
                main = (
                    db.query(models.Character)
                    .filter(models.Character.project_id == project.id, models.Character.role == "main")
                    .first()
                )
                _send_book_failed_alert_email(
                    project_id=project_id,
                    hero_name=main.name if main else "Unknown",
                    customer_email=project.customer_email or "",
                    product_type=project.product_type or "digital",
                    reason=reason,
                )
            except Exception as _ee:
                print(f"⚠️  book_failed alert email failed to send (non-fatal): {_ee}")
    except Exception:
        pass
    finally:
        db.close()


def _build_character_lock(character: models.Character) -> str:
    if not character:
        return "A kid superhero with a friendly comic-book look."
    traits = character.traits or {}
    appearance = traits.get("appearance", {})
    name = character.name or "the main character"
    skin = appearance.get("skin_tone", "")
    hair = appearance.get("hair", "")
    clothing = appearance.get("clothing", "")
    glasses = appearance.get("glasses", "")
    parts = [name]
    if skin:
        parts.append(f"with {skin} skin")
    if hair:
        parts.append(f"and {hair}")
    if glasses:
        parts.append(f"wearing {glasses}")
    if clothing:
        parts.append(f"(reference everyday clothes: {clothing} — use ONLY the outfit described in the scene prompt, do NOT layer reference clothes over the story outfit)")
    return " ".join(parts)


def _generate_full_book(project_id: str, db: Session) -> None:
    project = db.get(models.Project, UUID(project_id))
    if not project:
        raise RuntimeError(f"Project {project_id} not found")

    project.status = models.ProjectStatus.generating_book
    db.commit()

    # Collects human-readable notes about things the pipeline wasn't fully
    # confident about (a panel-1 image that needed regenerating, a dedup pass
    # that still left repeated lines after its retries) -- surfaced in the
    # admin review-needed email instead of only ever reaching a Railway log
    # line, so a review isn't a blind read-through of 48 panels but is
    # pointed at exactly what the system itself flagged.
    quality_flags: list = []

    option = db.get(models.StoryOption, project.selected_story_option_id)
    if not option:
        raise RuntimeError("No selected story option found")

    main = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == project.id,
            models.Character.role == "main",
        )
        .first()
    )
    if not main or not main.photo_keys:
        raise RuntimeError("No character with photos found")

    hero_name = main.name
    character_lock = _build_character_lock(main)
    genre = project.genre or "adventure"
    genre_style = GENRE_STYLES.get(genre, GENRE_STYLES["adventure"])
    use_lora = bool(settings.use_lora and main.fal_lora_url and main.fal_lora_status == "ready")
    use_instantid = bool(not use_lora and settings.use_instantid)
    use_pulid = bool(not use_lora and not use_instantid and settings.use_pulid)
    use_llamagen = bool(not use_lora and not use_instantid and not use_pulid and settings.use_llamagen and main.photo_keys)

    # Build appearance anchor string for LoRA path
    _app = (main.traits or {}).get("appearance", {})
    _gender = (main.personalization or {}).get("character_gender") or _app.get("gender")
    _anchor_parts = []
    if _app.get("age_range") and _gender:
        _anchor_parts.append(f"{_app['age_range']} {_gender}")
    if _app.get("skin_tone"):
        _anchor_parts.append(f"{_app['skin_tone']} skin")
    if _app.get("hair"):
        _anchor_parts.append(_app["hair"])
    if _app.get("glasses"):
        _anchor_parts.append(_app["glasses"])
    appearance_anchor = ", ".join(_anchor_parts)

    # expires_in is generous (24h, not the previous 2h) because this single URL is
    # computed once here and reused for every panel call across the WHOLE book --
    # LlamaGen's own team reported 403s fetching reference images, and real book
    # generations today ran past 2 hours under LlamaGen-side slowness, meaning this
    # URL could go stale mid-book and start silently breaking every panel after that
    # point (worse the longer generation drags on, since more of it falls after expiry).
    _LLAMAGEN_PHOTO_URL_TTL = 86400
    llamagen_photo_url = generate_presigned_get_url(main.photo_keys[0], expires_in=_LLAMAGEN_PHOTO_URL_TTL) if use_llamagen else ""
    llamagen_age = (main.personalization or {}).get("age") or main.age
    llamagen_gender = (main.personalization or {}).get("character_gender") or _app.get("gender", "")
    _lg_notes_parts = []
    if llamagen_age:
        _lg_notes_parts.append(f"{llamagen_age}-year-old")
    if llamagen_gender:
        _lg_notes_parts.append(llamagen_gender)
    if _app.get("glasses"):
        _lg_notes_parts.append(f"wears {_app['glasses']} — always include glasses")
    if _app.get("hair"):
        _lg_notes_parts.append(_app["hair"])
    llamagen_appearance_notes = ", ".join(_lg_notes_parts)
    # Query support chars early — needed for story generation and LlamaGen roles
    all_support_chars = (
        db.query(models.Character)
        .filter(
            models.Character.project_id == project.id,
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

    llamagen_extra_roles: list = []
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
    print(f"📚 Starting full book generation for project {project_id} — {hero_name}, genre={genre}, path={path}")

    # ── Raw photos + cartoon reference ───────────────────────────────────────
    raw_photo_bytes_list = [download_bytes(k) for k in main.photo_keys]

    if main.reference_image_key:
        cartoon_reference_bytes = download_bytes(main.reference_image_key)
    else:
        cartoon_reference_bytes = generate_character_reference_ai(
            raw_photo_bytes_list[0],
            extra_photos=raw_photo_bytes_list[1:] if len(raw_photo_bytes_list) > 1 else [],
        )
        ref_key = make_key(
            prefix=f"projects/{project.id}/characters/{main.id}/reference",
            filename="reference.png",
        )
        upload_bytes(ref_key, cartoon_reference_bytes, content_type="image/png")
        main.reference_image_key = ref_key
        db.add(main)
        db.commit()

    # ── Front cover (reuse from preview if it exists) ─────────────────────────
    cover_preview = (
        db.query(models.PreviewPage)
        .filter(
            models.PreviewPage.project_id == project.id,
            models.PreviewPage.page_number == 0,
        )
        .first()
    )

    pulid_reference = raw_photo_bytes_list[0] if raw_photo_bytes_list else None

    if cover_preview and cover_preview.image_url:
        print("Reusing existing front cover from preview...")
        front_cover_bytes = download_bytes(cover_preview.image_url)
        book_title = cover_preview.narration
    else:
        print("Generating front cover...")
        _audience_age = (main.personalization or {}).get("age") or 10
        book_title = generate_book_title_ai(hero_name, project.theme, option.logline, audience_age=_audience_age)
        cover_line = generate_cover_line_ai(hero_name, option.logline, audience_age=_audience_age)
        if use_lora:
            raw_cover_bytes_gen = fal_service.generate_cover_with_lora(
                hero_name=hero_name, theme=project.theme, logline=option.logline,
                lora_url=main.fal_lora_url, character_appearance=appearance_anchor,
            )
            raw_cover_b64 = base64.b64encode(raw_cover_bytes_gen).decode()
        elif use_instantid:
            raw_cover_bytes_gen = fal_service.generate_cover_with_instantcharacter(
                hero_name=hero_name, theme=project.theme, logline=option.logline,
                reference_image_bytes=pulid_reference, character_appearance=appearance_anchor,
            )
            raw_cover_b64 = base64.b64encode(raw_cover_bytes_gen).decode()
        elif use_pulid:
            raw_cover_bytes_gen = fal_service.generate_cover_with_pulid(
                hero_name=hero_name, theme=project.theme, logline=option.logline,
                reference_image_bytes=pulid_reference, character_appearance=appearance_anchor,
            )
            raw_cover_b64 = base64.b64encode(raw_cover_bytes_gen).decode()
        elif use_llamagen:
            cover_support_parts = []
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
            raw_cover_bytes_gen = llamagen_service.generate_cover_llamagen(
                hero_name=hero_name,
                theme=project.theme,
                logline=option.logline,
                photo_url=llamagen_photo_url,
                character_age=llamagen_age,
                character_gender=llamagen_gender,
                appearance_notes=appearance_anchor or llamagen_appearance_notes,
                extra_roles=llamagen_extra_roles,
                supporting_characters=", ".join(cover_support_parts),
                genre=genre,
            )
            raw_cover_b64 = base64.b64encode(raw_cover_bytes_gen).decode()
        else:
            from app.services.ai import build_cover_page_prompt
            cover_prompt = build_cover_page_prompt(
                hero_name=hero_name,
                theme=project.theme,
                logline=option.logline,
                character_brief=character_lock,
                genre_style=genre_style,
            )
            raw_cover_b64 = generate_image_with_reference(
                prompt=cover_prompt,
                reference_image_bytes=cartoon_reference_bytes,
                photo_bytes_list=raw_photo_bytes_list,
            )
        raw_cover_bytes = base64.b64decode(raw_cover_b64)
        front_cover_bytes = add_cover_text(
            image_bytes=raw_cover_bytes,
            title=book_title,
            hero_name=hero_name,
            cover_line=cover_line,
            title_color=genre_style["title_color"],
            genre=genre,
        )

    try:
        upload_bytes(key=f"panels/{project.id}/cover.png", data=front_cover_bytes, content_type="image/png")
    except Exception as _ce:
        print(f"  [admin] cover upload failed (non-fatal): {_ce}")

    is_print = project.product_type == "print"

    # ── Prose story (written first, then adapted into panels) ─────────────────
    page_count = project.panel_count or 48

    # Step 1a: Comedy blueprint (comedy only) — plan specific funny moments before writing prose
    comedy_blueprint: dict = {}
    if genre == "comedy":
        print("Generating comedy blueprint...")
        try:
            comedy_blueprint = generate_comedy_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Comedy blueprint failed — proceeding without: {e}")

        if comedy_blueprint:
            print("Validating comedy blueprint...")
            try:
                comedy_blueprint = validate_comedy_blueprint_ai(
                    blueprint=comedy_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Comedy blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.5: Adventure blueprint (adventure only) — lock the mission, obstacles,
    # and climax mechanism before writing prose, same rationale as comedy above.
    adventure_blueprint: dict = {}
    if genre == "adventure":
        print("Generating adventure blueprint...")
        try:
            adventure_blueprint = generate_adventure_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Adventure blueprint failed — proceeding without: {e}")

        if adventure_blueprint:
            print("Validating adventure blueprint...")
            try:
                adventure_blueprint = validate_adventure_blueprint_ai(
                    blueprint=adventure_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Adventure blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.6: Mystery blueprint (mystery only) — lock the culprit, motive, clues,
    # red herring, and climax mechanism before writing prose, same rationale as
    # the adventure blueprint above.
    mystery_blueprint: dict = {}
    if genre == "mystery":
        print("Generating mystery blueprint...")
        try:
            mystery_blueprint = generate_mystery_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Mystery blueprint failed — proceeding without: {e}")

        if mystery_blueprint:
            print("Validating mystery blueprint...")
            try:
                mystery_blueprint = validate_mystery_blueprint_ai(
                    blueprint=mystery_blueprint,
                    hero_name=hero_name,
                    page_count=page_count,
                )
            except Exception as e:
                print(f"⚠️  Mystery blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.7: Sci-fi blueprint (sci_fi only) — lock the technology's true
    # nature, the institution's real motive, and the third-act reversal
    # before writing prose, same rationale as the mystery blueprint above.
    # Without this, sci_fi silently fell into generate_prose_story_ai's
    # generic 8-beat fallback scene list, which is comedy-shaped ("something
    # goes hilariously wrong" ... ending on a "punchline") -- structurally
    # wrong for sci-fi regardless of the genre_guidance text, and it never
    # forces the story to state concrete stakes or resolve the central
    # premise (see generate_sci_fi_blueprint_ai's docstring for the exact
    # rationale this was built to fix -- it just was never wired in here).
    sci_fi_blueprint: dict = {}
    if genre == "sci_fi":
        print("Generating sci-fi blueprint...")
        try:
            sci_fi_blueprint = generate_sci_fi_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Sci-fi blueprint failed — proceeding without: {e}")

        if sci_fi_blueprint:
            print("Validating sci-fi blueprint...")
            try:
                sci_fi_blueprint = validate_sci_fi_blueprint_ai(
                    blueprint=sci_fi_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Sci-fi blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.8: Horror blueprint (horror_adult only) — lock the entity's true
    # backstory and the mechanism that resolves things before writing prose,
    # same rationale as the sci-fi blueprint above (see
    # generate_horror_blueprint_ai's docstring — same "generic comedy-shaped
    # fallback" gap this was built to fix but was never wired in here).
    horror_blueprint: dict = {}
    if genre == "horror_adult":
        print("Generating horror blueprint...")
        try:
            horror_blueprint = generate_horror_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Horror blueprint failed — proceeding without: {e}")

        if horror_blueprint:
            print("Validating horror blueprint...")
            try:
                horror_blueprint = validate_horror_blueprint_ai(
                    blueprint=horror_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Horror blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.9: Kid-friendly horror blueprint ("horror" genre, distinct from
    # horror_adult) — same rationale, mirrors generate_horror_blueprint_ai's
    # two-step architecture for the never-truly-frightening tone.
    horror_kids_blueprint: dict = {}
    if genre == "horror":
        print("Generating kid-friendly horror blueprint...")
        try:
            horror_kids_blueprint = generate_horror_kids_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Kid-friendly horror blueprint failed — proceeding without: {e}")

        if horror_kids_blueprint:
            print("Validating kid-friendly horror blueprint...")
            try:
                horror_kids_blueprint = validate_horror_kids_blueprint_ai(
                    blueprint=horror_kids_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Kid-friendly horror blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1a.10: Fantasy blueprint (fantasy only) — lock the magic system's
    # true nature and the third-act reversal before writing prose, same
    # rationale as the sci-fi blueprint above.
    fantasy_blueprint: dict = {}
    if genre == "fantasy":
        print("Generating fantasy blueprint...")
        try:
            fantasy_blueprint = generate_fantasy_blueprint_ai(
                hero_name=hero_name,
                logline=option.logline,
                outline=option.outline,
                personalization=main.personalization,
                additional_characters=support_char_info or None,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Fantasy blueprint failed — proceeding without: {e}")

        if fantasy_blueprint:
            print("Validating fantasy blueprint...")
            try:
                fantasy_blueprint = validate_fantasy_blueprint_ai(
                    blueprint=fantasy_blueprint,
                    hero_name=hero_name,
                )
            except Exception as e:
                print(f"⚠️  Fantasy blueprint validation failed — using unvalidated blueprint: {e}")

    # Step 1b: Prose story
    print(f"Generating prose story for {page_count}-panel comic...")
    try:
        prose_story = generate_prose_story_ai(
            hero_name=hero_name,
            theme=project.theme,
            logline=option.logline,
            outline=option.outline,
            personalization=main.personalization,
            genre=genre,
            additional_characters=support_char_info or None,
            page_count=page_count,
            comedy_blueprint=comedy_blueprint or None,
            adventure_blueprint=adventure_blueprint or None,
            mystery_blueprint=mystery_blueprint or None,
            sci_fi_blueprint=sci_fi_blueprint or None,
            horror_blueprint=horror_blueprint or None,
            horror_kids_blueprint=horror_kids_blueprint or None,
            fantasy_blueprint=fantasy_blueprint or None,
        )
    except Exception as e:
        print(f"⚠️  Prose story generation failed — falling back to outline-only: {e}")
        prose_story = None

    # Step 1b.5: Expand prose if too short
    if prose_story:
        target_wc = 2000 if page_count <= 24 else 3000
        try:
            prose_story = expand_prose_if_short_ai(
                prose_story=prose_story,
                target_word_count=target_wc,
                hero_name=hero_name,
                max_passes=3 if page_count > 24 else 2,
            )
        except Exception as e:
            print(f"⚠️  Prose expansion failed — using original: {e}")

    # Step 1c: Pacing check
    if prose_story:
        print("Checking story pacing...")
        try:
            prose_story = check_story_pacing_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                genre=genre,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Pacing check failed — using original prose: {e}")

    # Step 1c.5: Continuity check — dedicated single-purpose pass, separate from
    # pacing, so dropped characters/settings/setups aren't competing for attention
    # with timing rules in the same prompt.
    if prose_story:
        print("Checking story continuity...")
        try:
            prose_story = check_story_continuity_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Continuity check failed — using original prose: {e}")

    # Step 1c.6: Specificity/craft check — dedicated single-purpose pass for
    # earned solutions, concrete climax beats, judge neutrality, real friction,
    # and non-generic dialogue.
    if prose_story:
        print("Checking story specificity...")
        try:
            prose_story = check_story_specificity_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Specificity check failed — using original prose: {e}")

    # Step 1c.7: Arc-shape check — dedicated single-purpose pass for the
    # two-dip structure (a second, bigger setback before the climax) and,
    # for goal-driven genres, whether the objective was established concretely.
    if prose_story:
        print("Checking story arc shape...")
        try:
            prose_story = check_story_arc_shape_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                genre=genre,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Arc shape check failed — using original prose: {e}")

    # Step 1c.8: Adventure fidelity check (adventure only, when a blueprint exists) —
    # catches prose that drifted from its own planned mission/obstacles ("solution
    # hopping"), which the generic structural checks above can't detect since they
    # don't know what the intended mission was.
    if prose_story and genre == "adventure" and adventure_blueprint:
        print("Checking adventure blueprint fidelity...")
        try:
            prose_story = check_adventure_fidelity_ai(
                prose_story=prose_story,
                blueprint=adventure_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Adventure fidelity check failed — using original prose: {e}")

    # Step 1c.9: Mystery fidelity check (mystery only, when a blueprint exists) —
    # catches prose that drifted from its own planned culprit/motive/clues, the
    # mystery-genre version of "solution hopping," which the generic structural
    # checks above can't detect since they don't know the intended solution.
    if prose_story and genre == "mystery" and mystery_blueprint:
        print("Checking mystery blueprint fidelity...")
        try:
            prose_story = check_mystery_fidelity_ai(
                prose_story=prose_story,
                blueprint=mystery_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Mystery fidelity check failed — using original prose: {e}")

        # Ground-truth backstop, independent of the check above's own
        # self-report -- a real production book got "approved" prose where
        # two named suspects, built up with real evidence, never appeared
        # again anywhere in the story. Reads the actual prose text for each
        # suspect name rather than trusting the model's claim.
        try:
            suspect_names = extract_mystery_suspect_names_ai(mystery_blueprint, hero_name)
            closure_debug: dict = {}
            closure_result = verify_and_fix_mystery_suspect_closure_ai(
                prose_story=prose_story,
                suspect_names=suspect_names,
                hero_name=hero_name,
                page_count=page_count,
                debug_sink=closure_debug,
            )
            prose_story = closure_result["story"]
            if closure_result["unresolved_suspects"]:
                raise RuntimeError(
                    f"Mystery suspect-closure ground-truth check failed even after a corrective retry — "
                    f"{closure_result['unresolved_suspects']} never appear near the story's ending. "
                    f"Refusing to deliver a book with an abandoned suspect. Needs manual review."
                )
        except RuntimeError:
            raise
        except Exception as e:
            print(f"⚠️  Suspect-closure ground-truth check failed to run — proceeding without it: {e}")

    # Step 1c.9.5: Sci-fi fidelity check (sci_fi only, when a blueprint exists) —
    # catches prose that drifted from its own planned technology/institution
    # premise, the sci-fi version of "solution hopping" -- the technology's
    # internal logic drifting, the hidden truth being dumped all at once, the
    # third-act reversal flattening back into a straight investigate-and-
    # confirm line, or the climax resolving via a speech instead of a
    # concrete technical action.
    if prose_story and genre == "sci_fi" and sci_fi_blueprint:
        print("Checking sci-fi blueprint fidelity...")
        try:
            prose_story = check_sci_fi_fidelity_ai(
                prose_story=prose_story,
                blueprint=sci_fi_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Sci-fi fidelity check failed — using original prose: {e}")

    # Step 1c.9.6: Horror fidelity check (horror_adult only, when a blueprint
    # exists) — catches prose that drifted from its own planned entity
    # backstory/reversal, the horror version of "solution hopping."
    if prose_story and genre == "horror_adult" and horror_blueprint:
        print("Checking horror blueprint fidelity...")
        try:
            prose_story = check_horror_fidelity_ai(
                prose_story=prose_story,
                blueprint=horror_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Horror fidelity check failed — using original prose: {e}")

    # Step 1c.9.7: Kid-friendly horror fidelity check ("horror" genre only,
    # when a blueprint exists) — same rationale, reworked for the
    # never-truly-frightening tone.
    if prose_story and genre == "horror" and horror_kids_blueprint:
        print("Checking kid-friendly horror blueprint fidelity...")
        try:
            prose_story = check_horror_kids_fidelity_ai(
                prose_story=prose_story,
                blueprint=horror_kids_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Kid-friendly horror fidelity check failed — using original prose: {e}")

    # Step 1c.9.8: Fantasy fidelity check (fantasy only, when a blueprint
    # exists) — catches prose that drifted from its own planned magic-system
    # premise, the fantasy version of "solution hopping."
    if prose_story and genre == "fantasy" and fantasy_blueprint:
        print("Checking fantasy blueprint fidelity...")
        try:
            prose_story = check_fantasy_fidelity_ai(
                prose_story=prose_story,
                blueprint=fantasy_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Fantasy fidelity check failed — using original prose: {e}")

    # Step 1c.10: Comedy fidelity check (comedy only, when a blueprint exists) —
    # catches prose that drifted from its own planned funny_moments/npc_bits/
    # callback, the comedy-genre version of "solution hopping," which the
    # generic structural checks above can't detect since they don't know the
    # intended beats.
    if prose_story and genre == "comedy" and comedy_blueprint:
        print("Checking comedy blueprint fidelity...")
        try:
            prose_story = check_comedy_fidelity_ai(
                prose_story=prose_story,
                blueprint=comedy_blueprint,
                hero_name=hero_name,
                page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Comedy fidelity check failed — using original prose: {e}")

    # Step 1d: Comedy punch-up pass (comedy only) — amplify funniest scenes, cut dead panels
    if prose_story and genre == "comedy":
        print("Running comedy punch-up pass...")
        try:
            prose_story = punch_up_prose_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                page_count=page_count,
                additional_characters=support_char_info or None,
            )
        except Exception as e:
            print(f"⚠️  Comedy punch-up failed — using pre-punch-up prose: {e}")

    # ── NPC character bible (locked visual descriptions for all named NPCs) ────
    npc_bible: list = []
    if prose_story:
        print("Generating NPC character bible...")
        try:
            npc_bible = generate_story_character_bible_ai(
                prose_story=prose_story,
                hero_name=hero_name,
                support_char_names=[sc["name"] for sc in support_char_info],
            )
        except Exception as e:
            print(f"⚠️  NPC bible generation failed: {e}")

    # Save the finished prose for admin review -- non-fatal if S3 upload fails.
    # Without this, diagnosing a bad panel adaptation after the fact is
    # impossible: only the panel-level dialogue/narration/prompts were ever
    # persisted, not the prose they were adapted from, so there was no way to
    # tell whether a broken panel script was already broken in the prose or
    # invented fresh during adaptation.
    if prose_story:
        try:
            upload_bytes(
                key=f"panels/{project.id}/prose.txt",
                data=prose_story.encode("utf-8"),
                content_type="text/plain",
            )
        except Exception as _pe:
            print(f"  [admin] prose upload failed (non-fatal): {_pe}")

    # ── Panel script ─────────────────────────────────────────────────────────
    # check_panel_fidelity_ai's corrective retries genuinely help but aren't
    # guaranteed -- testing found a real case where they exhausted without
    # ever getting the climax to read as a resolved scene, and the pipeline
    # shipped that broken script straight into image generation and delivery
    # anyway. A fresh panel-adaptation attempt is a materially different roll
    # (same prose, new generation), not a rephrase of the same failed one, so
    # it's worth one full retry before giving up. If that retry ALSO fails,
    # stop here rather than deliver an unresolved story -- raising surfaces
    # up to generate_full_book_task, which marks the project book_failed for
    # manual review instead of silently completing the order.
    _UNRESOLVED_FIDELITY_OUTCOMES = {
        "sequential_retries_incomplete_used_best",
        "ground_truth_verification_failed_used_best",
        # The check's own first pass errored out (e.g. truncated at max_tokens)
        # before it could evaluate anything -- a real production book shipped
        # with an abandoned suspect and a suspect who never appeared at all
        # because this outcome wasn't treated as unresolved: the pipeline read
        # "the check never ran" as "nothing to fix" and shipped the completely
        # unverified first-draft panel script. No information is not the same
        # as a pass -- treat it exactly as seriously as a confirmed failure.
        "first_pass_failed",
    }

    def _adapt_and_check_panels() -> tuple[list, Optional[dict]]:
        pages = generate_preview_pages_chunked_ai(
            hero_name=hero_name,
            theme=project.theme,
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
            warnings_out=quality_flags,
        )
        if not prose_story:
            return pages, None
        fidelity_debug: dict = {}
        try:
            print("Checking panel fidelity (panels vs. prose)...")
            pages = check_panel_fidelity_ai(
                pages=pages, prose_story=prose_story, hero_name=hero_name, page_count=page_count,
                debug_sink=fidelity_debug,
            )
        except Exception as e:
            print(f"⚠️  Panel fidelity check failed — using original pages: {e}")
            fidelity_debug = {"outcome": "first_pass_failed"}
        return pages, fidelity_debug

    print(f"Adapting {'prose' if prose_story else 'outline'} into {page_count}-panel script...")
    generated_pages, fidelity_debug = _adapt_and_check_panels()

    if fidelity_debug is not None and fidelity_debug.get("outcome") in _UNRESOLVED_FIDELITY_OUTCOMES:
        print(f"⚠️  Panel fidelity check did not resolve even after corrective retries ({fidelity_debug.get('outcome')}) — regenerating the panel script from scratch instead of shipping an unresolved story")
        generated_pages, fidelity_debug = _adapt_and_check_panels()
        if fidelity_debug is not None and fidelity_debug.get("outcome") in _UNRESOLVED_FIDELITY_OUTCOMES:
            raise RuntimeError(
                f"Panel fidelity check failed on both the original and a from-scratch retry "
                f"({fidelity_debug.get('outcome')}) — refusing to deliver a book with an "
                f"unresolved climax or a dropped character. Needs manual review."
            )

    if prose_story:
        try:
            print("Checking panel pacing (post-climax panel budget)...")
            generated_pages = check_panel_pacing_ai(
                pages=generated_pages, hero_name=hero_name, genre=genre, page_count=page_count,
            )
        except Exception as e:
            print(f"⚠️  Panel pacing check failed — using original pages: {e}")

    generated_pages = check_nickname_usage_ai(
        pages=generated_pages, hero_name=hero_name,
        additional_characters=support_char_info or None, page_count=page_count,
    )

    sorted_pages = sorted(generated_pages, key=lambda x: x["page_number"])

    # Drop any NPC who didn't survive into the final panel script -- panel_script
    # and panel_pacing_check can both compress/cut scenes after the bible was
    # built from prose, and generating a reference portrait for a character who
    # ends up in zero panels wastes the LlamaGen credit. Checking this now,
    # after the panel script is final, instead of generating portraits for the
    # whole bible up front, is what actually guarantees no waste.
    if npc_bible:
        present_names = {
            str(n).strip().lower()
            for item in sorted_pages
            for n in (item.get("characters_present") or [])
        }
        kept_npcs = [npc for npc in npc_bible if npc["name"].strip().lower() in present_names]
        dropped_names = [npc["name"] for npc in npc_bible if npc["name"].strip().lower() not in present_names]
        if dropped_names:
            print(f"⚠️  Dropping NPC(s) absent from every panel — skipping reference portrait: {dropped_names}")
        npc_bible = kept_npcs

    # Give each remaining NPC a real reference photo (once per story, not per panel)
    # so LlamaGen anchors their likeness the same way it does for hero/support
    # characters instead of falling back to reusing an existing face when only a
    # text description is given.
    if npc_bible and use_llamagen:
        for npc in npc_bible:
            try:
                npc_age = int(npc["age"]) if npc.get("age") else None
            except (TypeError, ValueError):
                npc_age = None
            npc_gender = str(npc.get("gender") or "")
            try:
                print(f"  Generating reference portrait for NPC: {npc['name']}...")
                npc_portrait_bytes = llamagen_service.generate_npc_reference_llamagen(
                    npc_name=npc["name"],
                    description=npc["description"],
                    age=npc_age,
                    gender=npc_gender,
                )
                npc_slug = "".join(c if c.isalnum() else "-" for c in npc["name"].lower()).strip("-")
                npc_key = f"panels/{project.id}/npc-{npc_slug}.png"
                upload_bytes(key=npc_key, data=npc_portrait_bytes, content_type="image/png")
                npc["photo_url"] = generate_presigned_get_url(npc_key, expires_in=_LLAMAGEN_PHOTO_URL_TTL)
                npc["age"] = npc_age
                npc["gender"] = npc_gender
            except Exception as e:
                print(f"⚠️  NPC reference portrait failed for {npc.get('name')}, falling back to text-only: {e}")

    # ── Panel images ──────────────────────────────────────────────────────────
    panels: dict[int, tuple[bytes, dict]] = {}
    _panel_dialogue_map: dict = {}
    _panel_narration_map: dict = {}
    _panel_prompts_map: dict = {}
    # Maps a panel's "setting" label to the FULL image_prompt of the first panel
    # that used it. LlamaGen has no background/scene-reference API (only
    # comicRoles for characters), and the panel script's own "repeat the location
    # word-for-word" instruction is prompt-only -- unreliable the same way every
    # other "ask nicely" instruction has been this session. This makes it a code
    # guarantee instead: every later panel at the same location gets the FIRST
    # panel's own description injected verbatim, not a fresh LLM-remembered copy.
    setting_reference_prompts: dict[str, str] = {}
    # Maps (setting label, sorted characters present) to the FIRST panel's
    # "blocking" string for that exact group at that location -- e.g. who's
    # driving/front seat/back seat. Found via real admin review: the same car
    # scene had a passenger drawn in the driver's seat in one panel and the
    # back seat in another, purely because each panel re-invented blocking
    # independently. Keyed on the character set too, not just setting, so a
    # character joining/leaving correctly starts a new locked arrangement
    # instead of forcing a stale one that no longer has the right people in it.
    setting_blocking_prompts: dict[tuple, str] = {}
    # LlamaGen panels are deferred instead of generated inline -- see the
    # batching pass after this loop. Each entry holds everything needed to
    # either batch-generate or single-generate this panel once grouping is
    # decided; panel_bytes_gen stays None for these so the shared
    # upload/metadata tail below is skipped until the batched result exists.
    llamagen_deferred: list[dict] = []
    for item in sorted_pages:
        panel_num = item["page_number"]
        print(f"  Panel {panel_num}/{page_count}...")
        prompt = build_cartoon_page_prompt(
            image_prompt=item["image_prompt"],
            character_brief=character_lock,
            genre_style=genre_style,
        )
        if use_lora:
            panel_bytes_gen = fal_service.generate_page_with_lora(
                scene_prompt=item["image_prompt"],
                lora_url=main.fal_lora_url,
                character_appearance=appearance_anchor,
            )
        elif use_instantid:
            panel_bytes_gen = fal_service.generate_page_with_instantcharacter(
                scene_prompt=item["image_prompt"],
                reference_image_bytes=pulid_reference,
                character_appearance=appearance_anchor,
            )
        elif use_pulid:
            panel_bytes_gen = fal_service.generate_page_with_pulid(
                scene_prompt=item["image_prompt"],
                reference_image_bytes=pulid_reference,
                character_appearance=appearance_anchor,
            )
        elif use_llamagen:
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

            # Which named characters the story adapter says are actually drawn in this
            # panel. Authoritative when present — a character narratively absent (left,
            # missing, contacted remotely) must not be visually injected even though
            # they're a known support character. Falls back to "always include" only
            # if the story adapter didn't provide this field for this panel.
            _present_raw = item.get("characters_present")
            present_lower = (
                {str(n).strip().lower() for n in _present_raw if n} if _present_raw else None
            )
            panel_extra_roles = [
                r for r in llamagen_extra_roles
                if present_lower is None or r["name"].lower() in present_lower
            ]

            # Inject [Character: Name] tags for support chars with a photo who are
            # actually present in this panel. Do not gate on name-in-text — the AI
            # may refer to them by role label instead of name.
            for sc_info in support_char_info:
                if not any(r["name"] == sc_info["name"] for r in panel_extra_roles):
                    continue
                sc_name = sc_info["name"]
                sc_is_adult = (sc_info.get("age") is None) or (int(sc_info["age"]) >= 18)
                sc_tag_parts = []
                if sc_info.get("age"):
                    sc_tag_parts.append(f"{sc_info['age']}-year-old")
                if sc_info.get("gender"):
                    sc_tag_parts.append(sc_info["gender"])
                sc_tag_parts.append("adult" if sc_is_adult else "child")
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
            # Sort by side field so left speaker is always first (renders as left bubble)
            if any(d.get("side") for d in panel_dialogue):
                panel_dialogue = sorted(
                    panel_dialogue,
                    key=lambda d: 0 if d.get("side", "").lower() == "left" else 1,
                )
            panel_text = (item.get("narration", "") + " " + item.get("image_prompt", "")).lower()
            # Narration only — used for possession/warp cues, which must NOT scan
            # image_prompt (that text is generated under our own atmosphere rules,
            # so it self-reinforces spooky-sounding words unrelated to actual events).
            narration_text = (item.get("narration", "") or "").lower()

            # Inject NPC descriptions for any NPC who speaks OR is mentioned by name
            # in the narration/image_prompt. This keeps visual appearance locked across
            # all panels, not just the ones where the NPC has dialogue.
            panel_npc_roles: list = []
            if npc_bible:
                npc_lookup = {npc["name"].lower(): npc for npc in npc_bible}
                known_lower = {hero_name.lower()} | {sc["name"].lower() for sc in support_char_info}
                seen_injected = set()
                # Check speakers in dialogue
                for d in panel_dialogue:
                    spk = (d.get("speaker") or "").strip()
                    if spk and spk.lower() not in known_lower and spk.lower() in npc_lookup:
                        seen_injected.add(spk.lower())
                # Check the CONTENT of every dialogue line too, not just who's speaking —
                # e.g. Ariyah saying "I won't let you win, Lila!" means Lila is being
                # addressed/present even though she isn't the speaker of that line.
                dialogue_text = " ".join(
                    str(d.get("text") or d.get("line") or "") for d in panel_dialogue
                ).lower()
                # Also check narration box, image_prompt, and dialogue content for NPC name mentions
                for npc_name_lower in npc_lookup:
                    if npc_name_lower not in known_lower and (
                        npc_name_lower in panel_text or npc_name_lower in dialogue_text
                    ):
                        seen_injected.add(npc_name_lower)
                # Respect explicit presence data when available — an NPC only mentioned
                # in passing (not physically drawn this panel) shouldn't be injected.
                # EXCEPT when the NPC's name appears in the panel's own image_prompt --
                # that's the literal instruction driving what gets drawn, a stronger
                # presence signal than the separate characters_present list, which the
                # story adapter doesn't always keep in sync. Found via a real production
                # book where an NPC (Marcus) rendered as a different person -- a
                # different race -- in panels where characters_present had silently
                # dropped his name even though image_prompt still described him doing
                # something in the scene, so his locked appearance/photo never got
                # injected and the model invented his face from scratch each time.
                image_prompt_text = (item.get("image_prompt", "") or "").lower()
                if present_lower is not None:
                    seen_injected = {
                        n for n in seen_injected
                        if n in present_lower or n in image_prompt_text
                    }
                # Inject locked appearance block for every NPC found in this panel
                for npc_name_lower in seen_injected:
                    npc = npc_lookup[npc_name_lower]
                    npc_desc = npc["description"]
                    injection = (
                        f"MANDATORY — {npc['name'].upper()} MUST LOOK EXACTLY LIKE THIS: "
                        f"{npc_desc} "
                        f"This appearance is LOCKED and cannot vary between panels."
                    )
                    lg_scene = injection + "\n\n" + lg_scene
                    # Give the NPC a real photo reference (if one was generated) so the
                    # model anchors their face the same way it does hero/support —
                    # instead of defaulting to an existing character's likeness.
                    if npc.get("photo_url"):
                        npc_role: dict = {"name": npc["name"], "image": npc["photo_url"]}
                        if npc.get("age"):
                            npc_role["age"] = npc["age"]
                        if npc.get("gender"):
                            npc_role["gender"] = npc["gender"]
                        panel_npc_roles.append(npc_role)

            # Possession/haunting cue: when the narration signals a spirit manifesting
            # THROUGH a known character rather than as its own figure, push explicit
            # unsettling visual details onto that character for this panel only —
            # otherwise the model just renders a normal expression, not a haunting.
            _possession_keywords = (
                "spirit emerged", "spirit took", "spirit manifest", "spirit speaks through",
                "possess", "twisting", "twisted into", "glazed over", "otherworldly",
                "entity emerged", "spectral", "ghostly grip", "chilling presence", "sinister smile",
                "eyes turned", "eyes went", "eyes rolled back", "vacant stare", "hollow eyes",
                "wasn't herself", "wasn't himself", "not herself", "not himself", "not quite human",
                "consumed by", "corrupted", "under its control", "took control", "channeling",
                "spoke through", "voice wasn't", "wrong voice", "unnatural grin", "unnatural smile",
                "malevolent", "inhuman",
            )
            if any(kw in narration_text for kw in _possession_keywords):
                _present_names = [str(n) for n in (_present_raw or [])]
                _candidate_names = _present_names + [hero_name] + [sc["name"] for sc in support_char_info]
                _possessed_name = next(
                    (n for n in _candidate_names if n and n.lower() in narration_text),
                    None,
                )
                if _possessed_name:
                    lg_scene = (
                        f"MANDATORY VISUAL — {_possessed_name.upper()} IS BEING SUPERNATURALLY POSSESSED IN THIS PANEL: "
                        f"{_possessed_name}'s eyes must glow an unnatural white or sickly color with no visible pupils, "
                        f"their skin has an unnatural pale gray-green undertone, and their expression is a subtly wrong, "
                        f"too-wide or crooked smile that does not match a normal human expression — clearly unsettling "
                        f"and inhuman, not a normal happy or neutral face. This is a horror possession moment.\n\n"
                    ) + lg_scene

            # Environmental-manifestation cue: narration that describes the SCENE itself
            # warping/transforming rather than a specific character's face — a different
            # visual instruction than possession since there may be no named target.
            _environment_warp_keywords = (
                "nightmarish vision", "room transformed", "world twisted", "reality bent",
                "reality warped", "reality twisted", "walls seemed to", "shadows writhed",
                "environment distorted", "space twisted", "everything warped",
                "nightmarish", "hellish vision", "world shifted", "room shifted", "place transformed",
            )
            if any(kw in narration_text for kw in _environment_warp_keywords):
                lg_scene = (
                    "MANDATORY VISUAL — THE ENVIRONMENT ITSELF IS SUPERNATURALLY WARPED IN THIS PANEL: "
                    "walls, furniture, and surfaces are subtly wrong — unnatural angles, colors bleeding or "
                    "shifting to sickly greens and reds, shadows that fall in the wrong direction or seem to "
                    "move on their own, light sources flickering or pulsing unnaturally. The setting should "
                    "look like it's actively distorting around the characters, not a normal calm room. "
                    "This is a horror set-warp moment.\n\n"
                ) + lg_scene

            if len(panel_dialogue) >= 2:
                s0 = (panel_dialogue[0].get("speaker") or "").strip()
                s1 = (panel_dialogue[1].get("speaker") or "").strip()
                if s0 and s1:
                    lg_scene += f"\n\nPANEL LAYOUT: {s0} must be positioned on the LEFT side of the panel. {s1} must be positioned on the RIGHT side of the panel."

            llamagen_deferred.append({
                "panel_num": panel_num,
                "item": item,
                "lg_scene": lg_scene,
                "roles": panel_extra_roles + panel_npc_roles,
                # Empty setting means the script didn't label this panel's
                # location -- never batch those, since there's no reliable
                # signal they're actually the same scene.
                "setting": panel_setting,
                "group": panel_group,
            })
            panel_bytes_gen = None
        else:
            image_b64 = generate_image_with_reference(
                prompt=prompt,
                reference_image_bytes=cartoon_reference_bytes,
                photo_bytes_list=raw_photo_bytes_list,
            )
            panel_bytes_gen = base64.b64decode(image_b64)
        if panel_bytes_gen is not None:
            panels[panel_num] = (panel_bytes_gen, item)

            # Save raw panel for admin review — non-fatal if S3 upload fails
            try:
                upload_bytes(
                    key=f"panels/{project.id}/panel-{panel_num}.png",
                    data=panel_bytes_gen,
                    content_type="image/png",
                )
            except Exception as _ue:
                print(f"  [admin] panel {panel_num} upload failed (non-fatal): {_ue}")
            _panel_dialogue_map[str(panel_num)] = _normalize_dialogue(item.get("dialogue"))
            _panel_narration_map[str(panel_num)] = item.get("narration", "")
            _panel_prompts_map[str(panel_num)] = item.get("image_prompt", "")

    # ── Deferred LlamaGen generation, batched by scene ──────────────────────────
    # Panels sharing the same (setting, characters-present) are generated
    # together in one fixPanelNum>1 request instead of one call per panel --
    # admin review found independent single-panel calls had zero shared
    # context (the driver's seat swapped mid-scene, the scene randomly
    # relocated from inside a car to standing in the road), while a batched
    # test of the same scenario held the driver's seat and background
    # consistent across all panels in the batch. Capped at 3, not the
    # originally-tried 6 -- 3 is the only size actually proven to complete
    # reliably; real 6-panel batches in production timed out after their full
    # scaled poll window (720s) and fell back to one-at-a-time generation
    # anyway, just after a long dead wait first.
    _MAX_LLAMAGEN_BATCH = 3

    def _generate_panel_with_retry(entry: dict) -> bytes:
        """
        A single content-filter rejection or transient timeout on ONE panel
        used to take down the entire book with no retry at all -- found via a
        real failure where LlamaGen's content moderation (they run requests
        through Azure OpenAI's filter) never resolved within our poll window,
        surfacing to us as a plain timeout with no way to tell it apart from
        an ordinary slow generation. Retries once with the identical prompt
        (covers a transient timeout), then once more with an explicit
        safety-reframing prepended to the SAME scene content (covers an
        actual content-policy rejection) before giving up and letting the
        exception propagate to the existing book_failed safety net.
        """
        last_exc = None
        for attempt in range(3):
            scene_prompt = entry["lg_scene"]
            if attempt == 2:
                scene_prompt = (
                    "Render this scene safely and non-graphically -- convey tension, urgency, and physical "
                    "action through body language, framing, and motion cues rather than depicting contact, "
                    "impact, or harm directly. Avoid anything that could be read as violent or graphic.\n\n"
                ) + scene_prompt
            try:
                return llamagen_service.generate_panel_llamagen(
                    scene_prompt=scene_prompt,
                    photo_url=llamagen_photo_url,
                    character_name=hero_name,
                    character_age=llamagen_age,
                    character_gender=llamagen_gender,
                    appearance_notes=appearance_anchor or llamagen_appearance_notes,
                    extra_roles=entry["roles"],
                )
            except Exception as e:
                last_exc = e
                print(f"⚠️  LlamaGen panel {entry['panel_num']} generation failed on attempt {attempt + 1}/3 ({e})")
        raise last_exc

    # Circuit breaker: real production runs have seen LlamaGen's batch
    # endpoint fail 100% of the time for an entire book -- every batch
    # attempt burning the full ~360s timeout before falling back anyway,
    # repeated for every new scene group, turning a slow day into a
    # multi-hour generation. Once batching has failed this many times in a
    # row, stop attempting it for the rest of THIS book and go straight to
    # (parallel) single-panel generation -- batching's consistency benefit
    # is moot if it never actually succeeds, and this stops paying its
    # timeout tax over and over on a run where it clearly isn't working.
    _BATCH_CIRCUIT_BREAKER = 2
    _consecutive_batch_failures = 0

    def _generate_fallback_panels_parallel(batch: list) -> list:
        """Generates every panel in a failed batch independently, in
        parallel -- mirrors preview.py's parallel panel generation. The
        panels no longer share a single LlamaGen call once a batch fails, so
        there's no reason to generate them one at a time and pay the sum of
        their individual durations instead of just the slowest one.

        A panel that still fails after _generate_panel_with_retry's own 3
        attempts is logged and OMITTED from the returned list rather than
        raised -- letting one bad panel's exception surface here used to
        abort every other panel still in flight in the same batch (and, by
        propagating further, the rest of the book), throwing away completed
        prose/script/portraits and any panels that were about to succeed.
        The caller collects the omitted entries and gives them one final
        consolidated retry after every other panel in the book is done."""
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_entry = {executor.submit(_generate_panel_with_retry, entry): entry for entry in batch}
            for future in concurrent.futures.as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results[entry["panel_num"]] = (entry, future.result())
                except Exception as e:
                    print(f"⚠️  Panel {entry['panel_num']} failed after all retries ({e}) — deferring to a final retry pass")
        return [results[entry["panel_num"]] for entry in batch if entry["panel_num"] in results]

    def _finalize_panel(entry: dict, img_bytes: bytes) -> None:
        panel_num = entry["panel_num"]
        item = entry["item"]
        # Vision ground-truth check, scoped to panel 1 only (the highest-
        # visibility panel, and the one with a documented real incident: an
        # otherwise fully-verified book showed two people unpacking moving
        # boxes in panel 1 while its own narration described the hero
        # lounging on a chaise reading a novel -- an image-fidelity miss by
        # the model, not a pairing bug; see verify_panel_image_matches_
        # narration_ai's docstring). One bounded regeneration attempt if it
        # fails -- not a retry loop, to cap the added cost/latency.
        if panel_num == 1:
            try:
                if not verify_panel_image_matches_narration_ai(
                    img_bytes, item.get("narration", ""), item.get("image_prompt", "")
                ):
                    print("⚠️  Panel 1 image didn't match its own narration — regenerating once")
                    quality_flags.append("Panel 1's image didn't match its own narration on the first attempt and was regenerated — worth double-checking it actually landed right.")
                    img_bytes = _generate_panel_with_retry(entry)
            except Exception as e:
                print(f"⚠️  Panel 1 image verification/regeneration failed (non-fatal, keeping original): {e}")
        panels[panel_num] = (img_bytes, item)
        try:
            upload_bytes(
                key=f"panels/{project.id}/panel-{panel_num}.png",
                data=img_bytes,
                content_type="image/png",
            )
        except Exception as _ue:
            print(f"  [admin] panel {panel_num} upload failed (non-fatal): {_ue}")
        _panel_dialogue_map[str(panel_num)] = _normalize_dialogue(item.get("dialogue"))
        _panel_narration_map[str(panel_num)] = item.get("narration", "")
        _panel_prompts_map[str(panel_num)] = item.get("image_prompt", "")

    failed_entries: list = []

    i = 0
    while i < len(llamagen_deferred):
        batch = [llamagen_deferred[i]]
        j = i + 1
        while (
            j < len(llamagen_deferred)
            and len(batch) < _MAX_LLAMAGEN_BATCH
            and batch[0]["setting"]
            and llamagen_deferred[j]["setting"] == batch[0]["setting"]
            and llamagen_deferred[j]["group"] == batch[0]["group"]
        ):
            batch.append(llamagen_deferred[j])
            j += 1
        i = j

        if len(batch) == 1:
            entry = batch[0]
            try:
                batch_results = [(entry, _generate_panel_with_retry(entry))]
            except Exception as e:
                print(f"⚠️  Panel {entry['panel_num']} failed after all retries ({e}) — deferring to a final retry pass")
                failed_entries.append(entry)
                batch_results = []
        elif _consecutive_batch_failures >= _BATCH_CIRCUIT_BREAKER:
            # Batching has already failed enough times this book that it's
            # not worth paying its ~360s timeout again -- go straight to
            # parallel single-panel generation for this group.
            batch_results = _generate_fallback_panels_parallel(batch)
            succeeded = {e["panel_num"] for e, _ in batch_results}
            failed_entries.extend(e for e in batch if e["panel_num"] not in succeeded)
        else:
            # Union of every panel's roles in the batch, deduped by name --
            # two panels can share the same characters-present group but
            # mention different names in their own dialogue/narration, so
            # any single panel's role list alone could miss a photo
            # reference another panel in the batch actually needs.
            combined_roles = []
            seen_role_names = set()
            for e in batch:
                for r in e["roles"]:
                    rn = r.get("name", "")
                    if rn and rn not in seen_role_names:
                        seen_role_names.add(rn)
                        combined_roles.append(r)
            try:
                images = llamagen_service.generate_panel_batch_llamagen(
                    scene_prompts=[e["lg_scene"] for e in batch],
                    photo_url=llamagen_photo_url,
                    character_name=hero_name,
                    character_age=llamagen_age,
                    character_gender=llamagen_gender,
                    appearance_notes=appearance_anchor or llamagen_appearance_notes,
                    extra_roles=combined_roles,
                )
                batch_results = list(zip(batch, images))
                _consecutive_batch_failures = 0
            except Exception as e:
                _consecutive_batch_failures += 1
                tripped = _consecutive_batch_failures >= _BATCH_CIRCUIT_BREAKER
                print(
                    f"⚠️  LlamaGen batch of {len(batch)} panels failed ({e}) — "
                    f"falling back to parallel single-panel generation for this scene"
                    + (" (batching looks unreliable right now — skipping it for the rest of this book)" if tripped else "")
                )
                batch_results = _generate_fallback_panels_parallel(batch)
                succeeded = {e["panel_num"] for e, _ in batch_results}
                failed_entries.extend(e for e in batch if e["panel_num"] not in succeeded)

        for entry, img_bytes in batch_results:
            _finalize_panel(entry, img_bytes)

    if failed_entries:
        # A cluster of panel failures this late is almost always a transient
        # LlamaGen-side outage (observed in production as a run of straight
        # 500s), not a per-panel content issue -- each entry already used up
        # _generate_panel_with_retry's 3 attempts once. Rather than failing
        # the whole book (discarding the prose, full panel script, NPC
        # portraits, and every other panel already generated above), wait
        # for the outage to plausibly clear and give the survivors one more
        # consolidated shot before giving up on them.
        panel_nums = [e["panel_num"] for e in failed_entries]
        print(f"⚠️  {len(failed_entries)} panel(s) still failing after retries ({panel_nums}) — waiting 30s for a final attempt")
        time.sleep(30)
        retry_results = _generate_fallback_panels_parallel(failed_entries)
        for entry, img_bytes in retry_results:
            _finalize_panel(entry, img_bytes)
        succeeded = {e["panel_num"] for e, _ in retry_results}
        still_failed = [e["panel_num"] for e in failed_entries if e["panel_num"] not in succeeded]
        if still_failed:
            raise RuntimeError(f"LlamaGen panel generation failed after all retries for panel(s): {still_failed}")

    # Save metadata for admin review — non-fatal
    try:
        upload_bytes(f"panels/{project.id}/dialogue.json", json.dumps(_panel_dialogue_map).encode(), "application/json")
        upload_bytes(f"panels/{project.id}/narration.json", json.dumps(_panel_narration_map).encode(), "application/json")
        upload_bytes(f"panels/{project.id}/prompts.json", json.dumps(_panel_prompts_map).encode(), "application/json")
    except Exception as _me:
        print(f"  [admin] metadata upload failed (non-fatal): {_me}")

    # ── Back cover ─────────────────────────────────────────────────────────────
    print("Generating back cover...")
    back_copy = generate_back_cover_copy_ai(hero_name, option.logline, option.outline)
    back_cover_prompt = build_back_cover_prompt(
        hero_name=hero_name,
        logline=option.logline,
        character_brief=character_lock,
        genre_style=genre_style,
    )
    if use_lora:
        back_cover_gen = fal_service.generate_page_with_lora(
            scene_prompt=back_cover_prompt,
            lora_url=main.fal_lora_url,
            character_appearance=appearance_anchor,
        )
        back_b64 = base64.b64encode(back_cover_gen).decode()
    elif use_llamagen:
        back_cover_gen = llamagen_service.generate_panel_llamagen(
            scene_prompt=back_cover_prompt,
            photo_url=llamagen_photo_url,
            character_name=hero_name,
            character_age=llamagen_age,
            character_gender=llamagen_gender,
            appearance_notes=llamagen_appearance_notes,
            extra_roles=llamagen_extra_roles,
        )
        back_b64 = base64.b64encode(back_cover_gen).decode()
    else:
        back_b64 = generate_image_with_reference(
            prompt=back_cover_prompt,
            reference_image_bytes=cartoon_reference_bytes,
            photo_bytes_list=raw_photo_bytes_list,
        )
    back_cover_bytes = add_back_cover_text(
        image_bytes=base64.b64decode(back_b64),
        synopsis=back_copy.get("synopsis", ""),
        teaser=back_copy.get("teaser"),
        title_color=genre_style["title_color"],
    )
    try:
        upload_bytes(key=f"panels/{project.id}/back-cover.png", data=back_cover_bytes, content_type="image/png")
    except Exception as _bce:
        print(f"  [admin] back-cover upload failed (non-fatal): {_bce}")

    # ── Composite pages: [A+B → spread] [C → full-page splash] repeating ────
    print("Compositing pages...")
    interior_pages: list[bytes] = []
    num_groups = page_count // 3
    for group_num in range(num_groups):
        a_num = group_num * 3 + 1
        b_num = group_num * 3 + 2
        c_num = group_num * 3 + 3
        panel_a_bytes, item_a = panels[a_num]
        panel_b_bytes, item_b = panels[b_num]
        panel_c_bytes, item_c = panels[c_num]
        spread_bytes = compose_comic_page(
            panel_a_bytes, panel_b_bytes,
            dialogue_a=_normalize_dialogue(item_a.get("dialogue")),
            dialogue_b=_normalize_dialogue(item_b.get("dialogue")),
            narration_a=item_a.get("narration", ""),
            narration_b=item_b.get("narration", ""),
            hero_name=hero_name,
            use_vision=False,
        )
        interior_pages.append(spread_bytes)
        splash_bytes = compose_single_panel_page(
            panel_c_bytes,
            dialogue=_normalize_dialogue(item_c.get("dialogue")),
            narration=item_c.get("narration", ""),
            hero_name=hero_name,
            use_vision=False,
        )
        interior_pages.append(splash_bytes)

    all_pages = [front_cover_bytes] + interior_pages + [back_cover_bytes]
    total_pages = len(all_pages)

    digital_key = None
    cover_key = None
    interior_key = None

    if is_print:
        print(f"Assembling print interior + cover PDFs ({len(interior_pages)}pp interior)...")
        interior_pdf_bytes = _assemble_interior_pdf(interior_pages)
        interior_key = f"books/{project.id}/comic-interior.pdf"
        upload_bytes(key=interior_key, data=interior_pdf_bytes, content_type="application/pdf")

        cover_pdf_bytes = _assemble_cover_pdf(front_cover_bytes, back_cover_bytes, len(interior_pages))
        cover_key = f"books/{project.id}/comic-cover.pdf"
        upload_bytes(key=cover_key, data=cover_pdf_bytes, content_type="application/pdf")
    else:
        print(f"Assembling digital PDF ({total_pages}pp)...")
        digital_pdf_bytes = _assemble_digital_pdf(all_pages)
        digital_key = f"books/{project.id}/comic-digital.pdf"
        upload_bytes(key=digital_key, data=digital_pdf_bytes, content_type="application/pdf")

    # ── Finalise ──────────────────────────────────────────────────────────────
    project.book_pdf_key = digital_key
    project.book_cover_pdf_key = cover_key
    project.book_interior_pdf_key = interior_key
    project.status = models.ProjectStatus.book_ready
    db.commit()

    print(f"Full book ready for project {project_id} — {'print' if is_print else 'digital'}: {total_pages}pp ({page_count} panels)")
    print(f"[delivery] Holding delivery email for project {project_id} — awaiting admin approval at /admin/review/{project_id}")
    if is_print and project.customer_email:
        _send_print_order_email(project.customer_email, hero_name)
        print(f"[delivery] Print order receipt sent to {project.customer_email}")
    _send_admin_notification_email(project_id, hero_name, project.customer_email or "", project.product_type or "digital", quality_flags=quality_flags)
