from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import os
import glob

BORDER_WIDTH = 3
BUBBLE_PADDING = 14
LINE_SPACING = 6
TAIL_HEIGHT = 44
TAIL_WIDTH = 20

_font_cache: dict[int, ImageFont.ImageFont] = {}

# Characters the AI writer commonly uses that render as a blank tofu-box glyph
# in production -- found via a real printed page showing an empty rectangle in
# place of an em dash. Likely means the bundled DejaVuSans.ttf isn't present on
# that particular Railway build and PIL silently fell back to its built-in
# bitmap font (ASCII-only, no glyph for em/en dashes, curly quotes, or
# ellipsis). Rather than depend on font availability, strip these to safe
# ASCII equivalents before drawing -- a comic speech bubble doesn't need
# typographic punctuation, and this is correct even if the real font is found.
_UNSAFE_CHAR_MAP = {
    "—": "-",    # em dash —
    "–": "-",    # en dash –
    "‘": "'",    # left single quote '
    "’": "'",    # right single quote '
    "“": '"',    # left double quote "
    "”": '"',    # right double quote "
    "…": "...",  # ellipsis …
    " ": " ",    # non-breaking space
}


def _sanitize_text(text: str) -> str:
    if not text:
        return text
    for bad, good in _UNSAFE_CHAR_MAP.items():
        text = text.replace(bad, good)
    return text


_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLED_FONT = os.path.join(_APP_DIR, "fonts", "DejaVuSans.ttf")


def _find_font_path() -> str | None:
    candidates = [
        _BUNDLED_FONT,  # copied from Nix store during Railway build
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    # Broad search — covers Nix store (hashed paths), apt layouts, etc.
    for pattern in [
        "/usr/share/fonts/**/DejaVuSans.ttf",
        "/usr/share/fonts/**/LiberationSans-Regular.ttf",
        "/nix/store/**/DejaVuSans.ttf",
        "/usr/local/share/fonts/**/*.ttf",
    ]:
        found = glob.glob(pattern, recursive=True)
        if found:
            candidates.extend(found)

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    path = _find_font_path()
    if path:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except Exception:
            pass
    print(f"[font] WARNING: no TTF font found, falling back to bitmap default")
    return ImageFont.load_default()


def _text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_bubble(
    draw: ImageDraw.ImageDraw,
    text: str,
    speaker: str,
    font: ImageFont.ImageFont,
    speaker_font: ImageFont.ImageFont,
    anchor_x: int,
    y_top: int,
    img_width: int,
    max_chars: int,
    tail_up: bool = False,
    tail_tip_x: int = None,
    tail_tip_y: int = None,
) -> None:
    text = _sanitize_text(text)
    lines = textwrap.wrap(text, width=max_chars) or [text]

    line_sizes = [_text_size(font, line) for line in lines]
    text_w = max(w for w, _ in line_sizes)
    text_h = sum(h for _, h in line_sizes) + LINE_SPACING * (len(lines) - 1)

    speaker_h = 0
    show_speaker = False

    bubble_w = text_w + BUBBLE_PADDING * 2
    bubble_h = text_h + speaker_h + BUBBLE_PADDING * 2

    # Clamp so bubble stays within image bounds
    x1 = max(12, min(img_width - bubble_w - 12, anchor_x - bubble_w // 2))
    x2 = x1 + bubble_w
    y1 = y_top
    y2 = y1 + bubble_h

    tail_cx = (x1 + x2) // 2
    # tip_x: where the tail actually points — toward the speaker's head when provided
    tip_x = tail_tip_x if tail_tip_x is not None else tail_cx
    # Clamp tip inside image so the tail line never leaves the canvas
    tip_x = max(4, min(img_width - 4, tip_x))
    if tail_up:
        tip_y = tail_tip_y if tail_tip_y is not None else y1 - TAIL_HEIGHT
        actual_len = max(TAIL_HEIGHT, y1 - tip_y)
        # Scale width proportionally so long tails stay triangular, not needle-like
        dynamic_w = max(TAIL_WIDTH, actual_len // 4)
        tail_left = (tail_cx - dynamic_w // 2, y1)
        tail_right = (tail_cx + dynamic_w // 2, y1)
        tail_tip = (tip_x, tip_y)
    else:
        tip_y = tail_tip_y if tail_tip_y is not None else y2 + TAIL_HEIGHT
        actual_len = max(TAIL_HEIGHT, tip_y - y2)
        dynamic_w = max(TAIL_WIDTH, actual_len // 4)
        tail_left = (tail_cx - dynamic_w // 2, y2)
        tail_right = (tail_cx + dynamic_w // 2, y2)
        tail_tip = (tip_x, tip_y)

    # Draw fills first so white is solid before adding outlines
    draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill="white")
    draw.polygon([tail_left, tail_right, tail_tip], fill="white")

    # Draw outlines on top of fills
    draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill=None, outline="black", width=BORDER_WIDTH)
    draw.line([tail_left, tail_tip], fill="black", width=BORDER_WIDTH)
    draw.line([tail_right, tail_tip], fill="black", width=BORDER_WIDTH)

    # Text
    cur_y = y1 + BUBBLE_PADDING
    if show_speaker:
        draw.text((x1 + BUBBLE_PADDING, cur_y), speaker, fill="#555555", font=speaker_font)
        cur_y += speaker_h

    for line, (_, lh) in zip(lines, line_sizes):
        draw.text((x1 + BUBBLE_PADDING, cur_y), line, fill="black", font=font)
        cur_y += lh + LINE_SPACING


def _draw_barcode(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    """Draw a decorative (non-scannable) barcode pattern."""
    bar_pattern = [3,1,2,1,3,2,1,2,1,3,1,2,2,1,3,1,2,1,2,3,1,1,2,3,1,2,1,2,3,1]
    total_units = sum(bar_pattern)
    unit_w = w / total_units
    cx = x
    for i, units in enumerate(bar_pattern):
        bar_w = max(1, int(units * unit_w))
        if i % 2 == 0:
            draw.rectangle([cx, y, cx + bar_w - 1, y + h], fill=(0, 0, 0, 230))
        cx += bar_w
    # White background behind barcode
    draw.rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline=(255, 255, 255, 200), width=2)


def _load_logo(max_size: int = 130) -> Image.Image | None:
    logo_path = os.path.join(_APP_DIR, "static", "whamcomics_logo.png")
    if not os.path.exists(logo_path):
        return None
    try:
        logo = Image.open(logo_path).convert("RGBA")
        # Remove white background
        data = logo.getdata()
        new_data = [
            (r, g, b, 0) if (r > 230 and g > 230 and b > 230) else (r, g, b, a)
            for r, g, b, a in data
        ]
        logo.putdata(new_data)
        # Crop transparent padding so placement coords match visible content
        bbox = logo.getbbox()
        if bbox:
            logo = logo.crop(bbox)
        logo.thumbnail((max_size, max_size), Image.LANCZOS)
        return logo
    except Exception as e:
        print(f"[cover] Could not load logo: {e}")
        return None


def add_cover_text(image_bytes: bytes, title: str, hero_name: str, cover_line: str = None, title_color: tuple = (255, 215, 0), supporting_name: str = None, genre: str = "") -> bytes:
    ai_img = Image.open(BytesIO(image_bytes)).convert("RGBA")

    COVER_W = 1024
    COVER_H = 1567      # Matches US Standard print page ratio (6.69" x 10.24")
    TITLE_H = 320       # vertical space reserved for title text (gradient zone)
    BOTTOM_H = 80       # barcode strip

    # Scale AI art to fill the ENTIRE canvas — scale-to-cover without square-cropping first,
    # so portrait source images (768x1024) show nearly the full illustration
    ai_w, ai_h = ai_img.size
    scale = max(COVER_W / ai_w, COVER_H / ai_h)
    new_w = int(ai_w * scale)
    new_h = int(ai_h * scale)
    ai_scaled = ai_img.resize((new_w, new_h), Image.LANCZOS)
    cx = (new_w - COVER_W) // 2
    cy = (new_h - COVER_H) // 2
    ai_art = ai_scaled.crop((cx, cy, cx + COVER_W, cy + COVER_H))

    canvas = Image.new("RGBA", (COVER_W, COVER_H), (0, 0, 0, 255))
    canvas.paste(ai_art, (0, 0))

    overlay = Image.new("RGBA", (COVER_W, COVER_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w = COVER_W
    img_h = COVER_H

    # Dark gradient at top — solid near-black for first 80px, then fades to transparent.
    # Solid band ensures white title text is always legible over any illustration color.
    # Horror covers are already dark/oppressive by design, so this needs much less
    # forced darkening than a bright genre's cover — otherwise it reads as one big
    # black block instead of a subtle fade over the art.
    is_horror_adult = genre == "horror_adult"
    GRADIENT_H = (TITLE_H - 100) if is_horror_adult else (TITLE_H + 100)
    SOLID_BAND = 10 if is_horror_adult else 80
    PEAK_ALPHA = 95 if is_horror_adult else 245
    for y in range(GRADIENT_H):
        if y < SOLID_BAND:
            alpha = PEAK_ALPHA
        else:
            fade_y = y - SOLID_BAND
            fade_h = GRADIENT_H - SOLID_BAND
            alpha = int(PEAK_ALPHA * (1.0 - fade_y / fade_h) ** 1.4)
        draw.line([(0, y), (img_w, y)], fill=(0, 0, 0, alpha))

    # Logo at 200px max — keeps badge visible but leaves ample room for the title text
    logo = _load_logo(max_size=200)

    title_font = _load_font(64)
    subtitle_font = _load_font(34)

    # Reserve space on the right for the logo so title never overlaps it.
    # lx = COVER_W - logo.width - 20, so reserved = logo.width + 20 + 15 gap.
    logo_reserved = (logo.width + 35) if logo else 0
    text_area_w = COVER_W - logo_reserved
    wrap_chars = 18 if logo else 22

    title_lines = textwrap.wrap(title, width=wrap_chars) or [title]
    line_sizes = [_text_size(title_font, line) for line in title_lines]

    total_h = sum(lh + 8 for _, lh in line_sizes)
    block_y = 50

    def draw_outlined_text(x, y, text, font, fill, stroke_width=5):
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if abs(dx) >= stroke_width - 1 or abs(dy) >= stroke_width - 1:
                    draw.text((x + dx, y + dy), text, fill=(0, 0, 0, 255), font=font)
        draw.text((x, y), text, fill=fill, font=font)

    cur_y = block_y
    for line, (lw, lh) in zip(title_lines, line_sizes):
        lx = (text_area_w - lw) // 2
        draw_outlined_text(lx, cur_y, line, title_font, fill=(255, 255, 255, 255), stroke_width=5)
        cur_y += lh + 8

    # ── Bottom section ─────────────────────────────────────────────────────────
    # Tagline: gold text in a dark band just above the barcode strip
    TAGLINE_H = 90
    tagline_top = img_h - BOTTOM_H - TAGLINE_H

    # Dark gradient rising from the bottom — transparent at top, near-opaque at barcode strip
    tagline_peak_alpha = 120 if is_horror_adult else 210
    for y in range(TAGLINE_H):
        a = int(tagline_peak_alpha * (y / TAGLINE_H) ** 1.1)
        draw.line([(0, tagline_top + y), (img_w, tagline_top + y)], fill=(0, 0, 0, a))

    tagline = cover_line if cover_line else f"Starring {hero_name}"
    tagline_lines = textwrap.wrap(tagline, width=36) or [tagline]
    tagline_sizes = [_text_size(subtitle_font, line) for line in tagline_lines]
    tagline_total_h = sum(lh + 4 for _, lh in tagline_sizes)
    tag_y = tagline_top + (TAGLINE_H - tagline_total_h) // 2
    for line, (lw, lh) in zip(tagline_lines, tagline_sizes):
        lx = (img_w - lw) // 2
        draw_outlined_text(lx, tag_y, line, subtitle_font, fill=(255, 215, 0, 255), stroke_width=3)
        tag_y += lh + 4

    # Barcode strip at the very bottom — just the barcode, no vol/price text
    bottom_y = img_h - BOTTOM_H
    draw.rectangle([0, bottom_y, img_w, img_h], fill=(0, 0, 0, 230))

    bc_w, bc_h = 220, 58
    bc_x = 20   # bottom-left
    bc_y = bottom_y + (BOTTOM_H - bc_h) // 2
    draw.rectangle([bc_x - 6, bc_y - 4, bc_x + bc_w + 6, bc_y + bc_h + 4], fill=(255, 255, 255, 255))
    _draw_barcode(draw, bc_x, bc_y, bc_w, bc_h)

    # WhamComics logo stamp — top-right corner, aligned with title block
    if logo:
        lx = COVER_W - logo.width - 20
        ly = 50
        result_pre = Image.alpha_composite(canvas, overlay)
        result_pre.paste(logo, (lx, ly), logo)
        overlay = Image.new("RGBA", (COVER_W, COVER_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        canvas = result_pre

    # Comic border
    border = 10
    draw.rectangle([border // 2, border // 2, img_w - border // 2, img_h - border // 2],
                   outline=(0, 0, 0, 255), width=border)

    result = Image.alpha_composite(canvas, overlay).convert("RGB")
    out = BytesIO()
    result.save(out, format="PNG")
    return out.getvalue()


def add_back_cover_text(
    image_bytes: bytes,
    synopsis: str,
    teaser: str = None,
    title_color: tuple = (255, 215, 0),
) -> bytes:
    # Standardize to the same canvas dimensions as the front cover so every
    # page in the PDF is the same size (required for print-on-demand).
    BACK_W = 1024
    BACK_H = 1567       # Matches US Standard print page ratio (6.69" x 10.24")
    img_raw = Image.open(BytesIO(image_bytes)).convert("RGBA")
    raw_w, raw_h = img_raw.size
    # Center-crop to square, scale-to-cover full canvas (no black at bottom)
    sq = min(raw_w, raw_h)
    sq_img = img_raw.crop(((raw_w - sq) // 2, (raw_h - sq) // 2,
                            (raw_w + sq) // 2, (raw_h + sq) // 2))
    scale = max(BACK_W / sq, BACK_H / sq)
    scaled_sz = int(sq * scale)
    sq_scaled = sq_img.resize((scaled_sz, scaled_sz), Image.LANCZOS)
    cx = (scaled_sz - BACK_W) // 2
    sq_fitted = sq_scaled.crop((cx, 0, cx + BACK_W, BACK_H))
    canvas = Image.new("RGBA", (BACK_W, BACK_H), (0, 0, 0, 255))
    canvas.paste(sq_fitted, (0, 0))
    img = canvas

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = BACK_W, BACK_H

    synopsis_font = _load_font(24)
    teaser_font = _load_font(20)
    vol_font = _load_font(20)

    # Dark text band — bottom 42% of image (more room so text never overflows)
    band_y = int(img_h * 0.58)
    draw.rectangle([0, band_y, img_w, img_h], fill=(0, 0, 0, 200))

    # Synopsis — white text, wrapped
    synopsis_lines = textwrap.wrap(synopsis, width=44) or [synopsis]
    cur_y = band_y + 20
    for line in synopsis_lines:
        lw, lh = _text_size(synopsis_font, line)
        draw.text(((img_w - lw) // 2, cur_y), line, fill=(255, 255, 255, 255), font=synopsis_font)
        cur_y += lh + 6

    # Teaser — genre color
    if teaser:
        cur_y += 10
        teaser_lines = textwrap.wrap(f'Next issue: "{teaser}"', width=48) or [teaser]
        for line in teaser_lines:
            lw, lh = _text_size(teaser_font, line)
            draw.text(((img_w - lw) // 2, cur_y), line, fill=(*title_color[:3], 255), font=teaser_font)
            cur_y += lh + 4

    # Bottom strip — barcode + vol info
    bottom_strip_h = 50
    bottom_y = img_h - bottom_strip_h
    draw.rectangle([0, bottom_y, img_w, img_h], fill=(0, 0, 0, 220))

    vol_text = "Vol. 1  •  Issue #1  •  $19.99"
    vol_w, vol_h = _text_size(vol_font, vol_text)
    draw.text((18, bottom_y + (bottom_strip_h - vol_h) // 2), vol_text, fill=(255, 215, 0, 255), font=vol_font)

    bc_w, bc_h = 100, 36
    bc_x = img_w - bc_w - 16
    bc_y = bottom_y + (bottom_strip_h - bc_h) // 2
    draw.rectangle([bc_x - 3, bc_y - 2, bc_x + bc_w + 3, bc_y + bc_h + 2], fill=(255, 255, 255, 240))
    _draw_barcode(draw, bc_x, bc_y, bc_w, bc_h)

    # Border
    border = 10
    draw.rectangle(
        [border // 2, border // 2, img_w - border // 2, img_h - border // 2],
        outline=(0, 0, 0, 255),
        width=border,
    )

    result = Image.alpha_composite(img, overlay).convert("RGB")

    # WhamComics logo — top-right corner (same as front cover)
    logo = _load_logo(max_size=200)
    if logo:
        lx = img_w - logo.width + 10
        ly = -15
        result.paste(logo, (lx, ly), logo)

    out = BytesIO()
    result.save(out, format="PNG")
    return out.getvalue()


def draw_narration_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    y_start: int,
    canvas_width: int,
) -> None:
    if not text or not text.strip():
        return

    text = _sanitize_text(text)
    font = _load_font(24)
    lines = textwrap.wrap(text.strip(), width=48)[:4]
    if not lines:
        return

    padding_x, padding_y = 12, 7
    line_sizes = [_text_size(font, line) for line in lines]
    text_w = max(w for w, _ in line_sizes)
    text_h = sum(h for _, h in line_sizes) + LINE_SPACING * (len(lines) - 1)

    box_w = min(canvas_width - 36, text_w + padding_x * 2)
    box_h = text_h + padding_y * 2

    x1, y1 = 18, y_start + 8
    x2, y2 = x1 + box_w, y1 + box_h

    draw.rectangle([x1, y1, x2, y2], fill=(255, 252, 200, 175))
    draw.rectangle([x1, y1, x2, y2], outline=(20, 20, 20), width=2)

    cur_y = y1 + padding_y
    for line, (_, lh) in zip(lines, line_sizes):
        draw.text((x1 + padding_x, cur_y), line, fill=(20, 20, 20), font=font)
        cur_y += lh + LINE_SPACING


def draw_bubbles_on_panel(
    draw: ImageDraw.ImageDraw,
    dialogue: list,
    y_start: int,
    canvas_width: int,
    panel_height: int = None,
    narration: str = None,
    hero_name: str = None,
    character_positions: dict = None,
) -> None:
    """Draw narration caption and speech bubbles on a panel region."""
    if narration:
        draw_narration_box(draw, narration, y_start, canvas_width)

    lines = [d for d in dialogue if d.get("line")][:2]
    if not lines:
        return

    # Smaller than before (was 34) and more chars per line (was 18/13) -- a
    # smaller font needs fewer wrapped lines for the same dialogue, which
    # shrinks bubble height the most (bubbles grow downward from the 65%-down
    # anchor and were tall enough on longer lines to cover a character's body).
    font = _load_font(26)
    max_chars = 24 if len(lines) == 1 else 18

    positions = character_positions or {}
    hero = (hero_name or "").strip().lower()

    x_positions = []
    for i, line_data in enumerate(lines):
        speaker = (line_data.get("speaker") or "").strip().lower()
        if speaker and speaker in positions:
            # Vision-detected position — accurate horizontal placement
            x_positions.append(int(positions[speaker] * canvas_width))
        elif len(lines) == 1:
            # Single speaker fallback: hero left, other right, unknown center
            if hero and speaker == hero:
                x_positions.append(canvas_width // 4)
            elif hero and speaker and speaker != hero:
                x_positions.append(3 * canvas_width // 4)
            else:
                x_positions.append(canvas_width // 2)
        else:
            # Two-speaker fallback: first left, second right
            x_positions.append(canvas_width // 4 if i == 0 else 3 * canvas_width // 4)

    # Place bubbles at ~65% down the panel — leaves a 35% buffer above the bottom border
    if panel_height:
        y_top = y_start + int(panel_height * 0.65)
    else:
        y_top = y_start + 12

    for line_data, anchor_x in zip(lines, x_positions):
        _draw_bubble(
            draw=draw,
            text=line_data.get("line", ""),
            speaker="",
            font=font,
            speaker_font=font,
            anchor_x=anchor_x,
            y_top=y_top,
            img_width=canvas_width,
            max_chars=max_chars,
            tail_up=True,
        )


def add_speech_bubbles(image_bytes: bytes, dialogue: list) -> bytes:
    if not dialogue:
        return image_bytes

    lines = [d for d in dialogue if d.get("line")][:2]
    if not lines:
        return image_bytes

    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    img_w, img_h = img.size

    # Fewer chars per line when two bubbles share the width
    max_chars = 22 if len(lines) == 1 else 16

    font = _load_font(30)

    if len(lines) == 1:
        x_positions = [img_w // 2]
        tail_targets = [img_w // 2]
    else:
        x_positions = [img_w // 4, 3 * img_w // 4]
        # Bubbles at top, tails point DOWN toward characters below
        tail_targets = [int(img_w * 0.20), int(img_w * 0.80)]

    for line_data, anchor_x, tip_x in zip(lines, x_positions, tail_targets):
        _draw_bubble(
            draw=draw,
            text=line_data.get("line", ""),
            speaker="",
            font=font,
            speaker_font=font,
            anchor_x=anchor_x,
            y_top=20,
            img_width=img_w,
            max_chars=max_chars,
            tail_tip_x=tip_x,
        )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out = BytesIO()
    result.save(out, format="PNG")
    return out.getvalue()
