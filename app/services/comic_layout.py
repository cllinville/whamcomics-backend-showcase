from PIL import Image, ImageDraw
from io import BytesIO
from app.services.speech_bubbles import draw_bubbles_on_panel
from app.services.ai import detect_speaker_positions

CANVAS_W = 1024
CANVAS_H = 1567  # Matches US Standard print page ratio (6.69" x 10.24") exactly at this width
BORDER = 8
GUTTER = 8

PANEL_W = CANVAS_W - 2 * BORDER          # 988
PANEL_H = (CANVAS_H - 2 * BORDER - GUTTER) // 2  # 758


def _autocrop_letterbox(img: Image.Image) -> Image.Image:
    """Strip near-white letterbox bars LlamaGen sometimes bakes into panel edges."""
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < 235 else 0)
    bbox = mask.getbbox()
    if bbox:
        return img.crop(bbox)  # Crop all 4 sides
    return img


def _fit_panel(img: Image.Image) -> Image.Image:
    """Scale-to-cover: fill panel slot completely, center-crop any overflow."""
    img = _autocrop_letterbox(img)
    scale = max(PANEL_W / img.width, PANEL_H / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - PANEL_W) // 2
    top = (new_h - PANEL_H) // 2
    return img.crop((left, top, left + PANEL_W, top + PANEL_H))


def _get_speaker_names(dialogue: list) -> list:
    return list({(d.get("speaker") or "").strip() for d in (dialogue or []) if d.get("speaker")})


def compose_comic_page(
    panel_a_bytes: bytes,
    panel_b_bytes: bytes,
    dialogue_a: list = None,
    dialogue_b: list = None,
    narration_a: str = None,
    narration_b: str = None,
    hero_name: str = None,
    use_vision: bool = True,
) -> bytes:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    panel_a = _fit_panel(Image.open(BytesIO(panel_a_bytes)).convert("RGB"))
    panel_b = _fit_panel(Image.open(BytesIO(panel_b_bytes)).convert("RGB"))

    y_a = BORDER
    y_b = BORDER + PANEL_H + GUTTER

    canvas.paste(panel_a, (BORDER, y_a))
    canvas.paste(panel_b, (BORDER, y_b))

    # Panel borders
    draw.rectangle([BORDER - 1, y_a - 1, BORDER + PANEL_W + 1, y_a + PANEL_H + 1],
                   outline=(20, 20, 20), width=3)
    draw.rectangle([BORDER - 1, y_b - 1, BORDER + PANEL_W + 1, y_b + PANEL_H + 1],
                   outline=(20, 20, 20), width=3)

    # Outer page border (double rule — classic comic style)
    draw.rectangle([3, 3, CANVAS_W - 3, CANVAS_H - 3], outline=(20, 20, 20), width=5)
    draw.rectangle([9, 9, CANVAS_W - 9, CANVAS_H - 9], outline=(20, 20, 20), width=2)

    # Detect character positions via vision for accurate bubble placement
    positions_a: dict = {}
    positions_b: dict = {}
    if use_vision:
        try:
            speakers_a = _get_speaker_names(dialogue_a)
            if speakers_a:
                positions_a = detect_speaker_positions(panel_a_bytes, speakers_a)
                print(f"  [vision] panel A positions: {positions_a}")
            speakers_b = _get_speaker_names(dialogue_b)
            if speakers_b:
                positions_b = detect_speaker_positions(panel_b_bytes, speakers_b)
                print(f"  [vision] panel B positions: {positions_b}")
        except Exception as e:
            print(f"  [vision] position detection failed, using defaults: {e}")

    # Narration boxes and speech bubbles composited on RGBA overlay
    if dialogue_a or dialogue_b or narration_a or narration_b:
        overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        draw_bubbles_on_panel(overlay_draw, dialogue_a or [], y_start=y_a, canvas_width=CANVAS_W, panel_height=PANEL_H, narration=narration_a, hero_name=hero_name, character_positions=positions_a)
        draw_bubbles_on_panel(overlay_draw, dialogue_b or [], y_start=y_b, canvas_width=CANVAS_W, panel_height=PANEL_H, narration=narration_b, hero_name=hero_name, character_positions=positions_b)

        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def compose_single_panel_page(
    panel_bytes: bytes,
    dialogue: list = None,
    narration: str = None,
    hero_name: str = None,
    use_vision: bool = True,
) -> bytes:
    """One panel per page — used for the digital PDF (1 panel per page)."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(20, 20, 20))

    img = Image.open(BytesIO(panel_bytes)).convert("RGB")
    # Scale to cover (fill) the canvas, then center-crop — eliminates blank borders
    scale = max(CANVAS_W / img.width, CANVAS_H / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    x_off = (new_w - CANVAS_W) // 2
    y_off = (new_h - CANVAS_H) // 2
    img = img.crop((x_off, y_off, x_off + CANVAS_W, y_off + CANVAS_H))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([BORDER - 1, -1, CANVAS_W - BORDER + 1, CANVAS_H + 1],
                   outline=(20, 20, 20), width=3)
    draw.rectangle([3, 3, CANVAS_W - 3, CANVAS_H - 3], outline=(20, 20, 20), width=5)
    draw.rectangle([9, 9, CANVAS_W - 9, CANVAS_H - 9], outline=(20, 20, 20), width=2)

    # Detect character positions via vision for accurate bubble placement
    positions: dict = {}
    if use_vision:
        try:
            speakers = _get_speaker_names(dialogue)
            if speakers:
                positions = detect_speaker_positions(panel_bytes, speakers)
                print(f"  [vision] single panel positions: {positions}")
        except Exception as e:
            print(f"  [vision] position detection failed, using defaults: {e}")

    if dialogue or narration:
        overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw_bubbles_on_panel(overlay_draw, dialogue or [], y_start=0, canvas_width=CANVAS_W, panel_height=CANVAS_H, narration=narration, hero_name=hero_name, character_positions=positions)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()
