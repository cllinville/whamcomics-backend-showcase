import time
from typing import Optional

import httpx

from app.config import settings

LLAMAGEN_BASE = "https://api.llamagen.ai"
_POLL_INTERVAL = 5
# History: raised from 300s to 480s on 2026-10-10 after a panel that had
# genuinely completed got killed by our timeout and discarded. Revisited on
# 2026-10-16 after LlamaGen deployed a server-side fix: 3 of 4 generations in
# a test batch completed normally in 2-3 min, but one job hung and ate the
# full 480s before timing out -- its retry then succeeded in ~3 min, same as
# the healthy ones. That pattern (healthy jobs finish in 2-3 min; a stuck job
# doesn't recover just by waiting longer) means a long timeout no longer buys
# correctness, it just adds dead time before we retry. Lowered to 180s --
# comfortably above every normal completion we've observed, short enough to
# roughly halve the worst-case wait on a stuck job.
_POLL_TIMEOUT = 180  # 3 minutes

# The delivered page canvas (comic_layout.py) is 1024x1567 -- a tall ~0.65:1
# ratio matching the physical print page. The old landscape default
# (1024x768, ratio 1.33:1) required center-cropping roughly HALF the image's
# width to fit that shape; a real generated 1024x768 panel needed ~530px cut
# from each side. What matters for the crop math is the RATIO, not the
# absolute pixel count -- "1024x1536" (ratio 0.667:1) fixed the cropping but
# doubled the pixel count LlamaGen has to render per panel (~1.6MP vs the old
# ~0.8MP), and batch reliability visibly degraded once it shipped: 6-panel
# batches that previously completed started timing out at the full scaled
# poll window. "768x1152" holds the same 0.667:1 ratio at roughly the old
# pixel count, to fix the crop without also roughly doubling render load.
PAGE_MATCHED_SIZE = "768x1152"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.llamagen_api_key}",
        "Content-Type": "application/json",
    }


def _sanitize_prompt(text: str) -> str:
    """Strip stray ASCII control bytes (keeping \\t\\n\\r) before this prompt
    leaves our backend. Found in production: an upstream GPT rewrite step
    occasionally emits a literal control character (e.g. 0x14) in place of an
    em dash, with no validation catching it. The resulting prompt still looked
    like normal text when printed/logged, but every panel built from it
    rendered a correct background with ZERO characters -- consistent with
    LlamaGen's own prompt parser treating a raw control byte as a terminator
    and silently dropping everything after it, which is exactly where
    character descriptions and reference-photo instructions live in these
    prompts. This is the last line of defense regardless of which caller or
    upstream rewrite introduces the bad byte."""
    if not text:
        return text
    return "".join(c for c in text if ord(c) >= 32 or c in "\t\n\r")


def _submit(payload: dict) -> str:
    if "prompt" in payload:
        payload = {**payload, "prompt": _sanitize_prompt(payload["prompt"])}
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{LLAMAGEN_BASE}/v1/comics/generations",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        return resp.json()["id"]


def _poll_until_done(generation_id: str) -> bytes:
    deadline = time.time() + _POLL_TIMEOUT
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            resp = client.get(
                f"{LLAMAGEN_BASE}/v1/comics/generations/{generation_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "PROCESSED":
                # Response nests assetUrl at comics[].panels[].assetUrl
                asset_url = (data.get("data") or {}).get("assetUrl") or data.get("assetUrl")
                if not asset_url:
                    for comic in (data.get("comics") or []):
                        for panel in (comic.get("panels") or []):
                            asset_url = panel.get("assetUrl")
                            if asset_url:
                                break
                        if asset_url:
                            break
                if not asset_url:
                    raise RuntimeError(f"LlamaGen {generation_id}: PROCESSED but no assetUrl — {data}")
                img_resp = client.get(asset_url, timeout=60)
                img_resp.raise_for_status()
                return img_resp.content
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"LlamaGen {generation_id} {status}: {data.get('error')}")
            print(f"  [LlamaGen] {generation_id} status={status}, waiting...")
    raise TimeoutError(f"LlamaGen generation {generation_id} timed out after {_POLL_TIMEOUT}s")


def download_asset(asset_url: str) -> bytes:
    """Download an assetUrl from LlamaGen's CDN. s.llamagen.ai URLs are
    signed/presigned -- sending our API bearer token on top breaks their own
    signature validation (400), so fetch anonymously, matching what
    _poll_until_done already does for the single-panel path."""
    with httpx.Client(timeout=60) as client:
        resp = client.get(asset_url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.content


def debug_poll_raw(generation_id: str) -> dict:
    """
    Same polling loop as _poll_until_done, but returns the full raw completed
    JSON instead of downloading and returning a single image's bytes. Built to
    inspect what a fixPanelNum > 1 request actually returns -- how many panel
    entries come back, whether each has its own assetUrl or they're baked into
    one combined page image -- before committing to how batched generation
    should be integrated. _poll_until_done only ever grabs the FIRST assetUrl
    it finds and discards the rest, which was never a problem while fixPanelNum
    was hardcoded to 1, but would silently drop every other panel in a batch.
    """
    deadline = time.time() + _POLL_TIMEOUT
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            resp = client.get(
                f"{LLAMAGEN_BASE}/v1/comics/generations/{generation_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "PROCESSED":
                return data
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"LlamaGen {generation_id} {status}: {data.get('error')}")
            print(f"  [LlamaGen debug] {generation_id} status={status}, waiting...")
    raise TimeoutError(f"LlamaGen generation {generation_id} timed out after {_POLL_TIMEOUT}s")


def _build_role(name: str, photo_url: str, age: Optional[int], gender: str) -> dict:
    role: dict = {"name": name, "image": photo_url}
    if age:
        role["age"] = age
    if gender:
        role["gender"] = gender
    return role


def _build_shared_prompt_blocks(
    character_name: str,
    character_age: Optional[int],
    character_gender: str,
    appearance_notes: str,
    extra_roles: list,
) -> tuple[str, str, str]:
    """
    Factored out of generate_panel_llamagen so generate_panel_batch_llamagen
    can build these once for a whole batch instead of once per panel -- none
    of this varies panel to panel for a fixed character/roles set. Returns
    (before_block, char_ref, after_block); the caller sandwiches the
    panel-specific scene text between before_block and after_block, with
    char_ref prefixed to it the same way the original single-panel prompt did.
    """
    age_desc = f"{character_age}-year-old " if character_age else ""
    gender_desc = f"{character_gender} " if character_gender else ""
    # Default to adult when age is unknown — WhamComics customers are adults
    is_adult = (character_age is None) or (character_age >= 18)
    age_group = "adult" if is_adult else "child"
    age_enforcement = (
        f"{character_name} must always appear as an adult, never as a child or teenager."
        if is_adult else
        f"{character_name} must always appear as a child, never as an adult or teenager."
    )
    notes = f", {appearance_notes}" if appearance_notes else ""
    char_ref = f"[Character: {character_name}] {age_desc}{gender_desc}{age_group}{notes}."

    # Build a mandatory appearance lock block for the hero — same pattern as NPC injection
    if appearance_notes:
        appearance_lock = (
            f"MANDATORY — {character_name.upper()} MUST LOOK EXACTLY LIKE THIS IN EVERY SINGLE PANEL: "
            f"{appearance_notes}. "
            f"Match the reference photo precisely. "
            f"These features are LOCKED and cannot change between panels under any circumstances."
        )
    else:
        appearance_lock = (
            f"MANDATORY — {character_name.upper()} must match their reference photo exactly in every panel. "
            f"Do not change their face, hair, skin tone, or any physical features."
        )

    # Extract skin tone from appearance_notes for explicit negative prompt
    skin_note = ""
    if appearance_notes:
        for part in appearance_notes.split(","):
            if "skin" in part.lower():
                skin_note = f" Never change {character_name}'s skin tone — must always be {part.strip()}."
                break
    # Build consistency enforcement for every named character with a reference photo
    extra_consistency = ""
    for role in (extra_roles or []):
        role_name = role.get("name", "")
        if role_name:
            extra_consistency += (
                f" Do not change [Character: {role_name}]'s hairstyle, hair color, skin tone, age, or facial features between panels."
                f" [Character: {role_name}] must look identical to their reference photo in every panel."
            )
    negative = (
        f"Do not change [Character: {character_name}]'s hairstyle, hair color, skin tone, age, or glasses between panels.{skin_note}"
        f"{extra_consistency} "
        f"Do not add grey or white hair to any character unless their reference photo shows grey hair. "
        f"{age_enforcement} "
        f"Every supporting character in this scene is a FULLY GROWN ADULT — do not render any of them as a child, "
        f"teenager, or young person. No character should have a child's face, child's body proportions, or child's height. "
        f"Every character must have a sharp, crisp, fully detailed face — no blurry faces, no out-of-focus faces, no smudged or softened faces, no blank faces, no faceless figures, no obscured or missing facial features. Face must be in sharp focus with clearly defined eyes, nose, and mouth. "
        f"Never show the same character appearing twice in the same image — each named person must appear exactly once. "
        f"Every character must look VISUALLY DISTINCT from every other character — no two characters should share the same hairstyle, outfit color, or jewelry. "
        f"No speech bubbles, no dialogue boxes, no thought clouds, no caption boxes, no text overlays of any kind."
    )
    safe_zone = (
        "ORIENTATION: All characters are standing upright on the floor with normal gravity — "
        "feet at the bottom, heads at the top. Never show characters upside-down, sideways, floating, or falling. "
        "The floor is at the bottom of the image, the ceiling or sky is at the top. "
        "COMPOSITION: Keep all characters and important visual elements strictly within "
        "the center 65% of the frame — the leftmost 17% and rightmost 17% of the image "
        "must be clear of characters, faces, and text. "
        "FULL BODY: Every character must be shown from head to toe — no character cropped below the waist. "
        "FACE QUALITY: Every character's face must be sharp, crisp, and in clear focus — detailed eyes, nose, mouth. Never blurry, never softened, never smudged. "
        "FACE VISIBILITY IN DARK SCENES: Even in dark or atmospheric scenes, every character's face must be clearly illuminated and visible — use a practical light source (flashlight, candle, lamp, moonlight, doorway glow) to light the face. Never render a face as a dark silhouette or black void. "
        "GAZE: Characters must look at each other or at something in the scene — NEVER stare directly into the camera or look straight at the viewer. "
        "NO speech bubbles, NO dialogue boxes, NO thought clouds, NO caption boxes anywhere in the image — "
        "the image must be pure illustration with no embedded text."
    )
    style_block = (
        "STYLE: comic book illustration — bold ink linework, cel-shaded coloring, "
        "vivid saturated flat colors with hard shadow edges, detailed illustrated background. "
        "NOT a photograph. NOT photorealistic. NOT live-action. "
        "This image must look like a hand-drawn comic panel, never like a real photo."
    )
    no_bubbles_header = (
        "⚠️ ABSOLUTE RULE #1 — ZERO TOLERANCE: NO speech bubbles. NO dialogue boxes. NO thought clouds. "
        "NO caption boxes. NO text of ANY kind anywhere in this image — not a single letter, word, or symbol. "
        "This image is a PURE ILLUSTRATION. Any embedded text or speech bubble makes it unusable. "
        "This rule overrides every other instruction.\n\n"
    )
    all_names = [character_name] + [r.get("name", "") for r in (extra_roles or []) if r.get("name")]
    presence_rule = (
        f"⚠️ ABSOLUTE RULE #0 — EVERY NAMED CHARACTER MUST BE PHYSICALLY PRESENT AND VISIBLE: "
        f"This scene MUST show {', '.join(all_names)} as actual human figures in the image — "
        f"not implied, not off-screen, not represented only by an object or empty space. "
        f"Do not simplify the composition by dropping a character for framing or clutter reasons. "
        f"If a character is named in the scene description, they are IN the image, full body, "
        f"clearly visible.\n\n"
    ) if len(all_names) > 1 else ""
    before_block = f"{no_bubbles_header}{presence_rule}{appearance_lock}"
    after_block = f"{safe_zone}\n\n{style_block}\n\n{negative}"
    return before_block, char_ref, after_block


def generate_panel_llamagen(
    scene_prompt: str,
    photo_url: str,
    character_name: str,
    character_age: Optional[int] = None,
    character_gender: str = "",
    appearance_notes: str = "",
    extra_roles: list = None,
) -> bytes:
    before_block, char_ref, after_block = _build_shared_prompt_blocks(
        character_name, character_age, character_gender, appearance_notes, extra_roles,
    )
    prompt = f"{before_block}\n\n{char_ref} {scene_prompt}\n\n{after_block}"
    roles = [_build_role(character_name, photo_url, character_age, character_gender)]
    roles.extend(extra_roles or [])
    size = PAGE_MATCHED_SIZE
    payload = {
        "prompt": prompt,
        "fixPanelNum": 1,
        "size": size,
        "preset": settings.llamagen_preset,
        "comicRoles": roles,
    }
    generation_id = _submit(payload)
    print(f"  [LlamaGen] panel submitted: {generation_id}")
    return _poll_until_done(generation_id)


def generate_panel_batch_llamagen(
    scene_prompts: list[str],
    photo_url: str,
    character_name: str,
    character_age: Optional[int] = None,
    character_gender: str = "",
    appearance_notes: str = "",
    extra_roles: list = None,
) -> list[bytes]:
    """
    Generates multiple panels of the SAME scene (same setting, same characters
    present) in a single LlamaGen request via fixPanelNum, instead of one
    independent call per panel. Testing (admin review of project e8c92edc)
    showed independent single-panel calls have no shared context at all --
    the driver's seat swapped between two different characters mid-scene, and
    the scene randomly relocated from inside a car to standing in the road.
    A batched fixPanelNum=3 test of the same scenario kept the driver in the
    driver's seat and the background consistent across all 3 panels. Returns
    images in the same order as scene_prompts. Capped by LlamaGen at 20
    panels per request.
    """
    n = len(scene_prompts)
    before_block, char_ref, after_block = _build_shared_prompt_blocks(
        character_name, character_age, character_gender, appearance_notes, extra_roles,
    )
    panels_text = "\n".join(f"Panel {i + 1}: {sp}" for i, sp in enumerate(scene_prompts))
    prompt = (
        f"{before_block}\n\n{char_ref}\n\n"
        f"This is a {n}-panel sequence, all part of ONE continuous scene. Each panel below shows a different "
        f"specific moment, but the background, environment, and every character's position (who is where) MUST "
        f"stay identical across all {n} panels unless a panel's own description explicitly says someone moved.\n\n"
        f"{panels_text}\n\n{after_block}"
    )
    roles = [_build_role(character_name, photo_url, character_age, character_gender)]
    roles.extend(extra_roles or [])
    size = PAGE_MATCHED_SIZE
    payload = {
        "prompt": prompt,
        "fixPanelNum": n,
        "size": size,
        "preset": settings.llamagen_preset,
        "comicRoles": roles,
    }
    generation_id = _submit(payload)
    print(f"  [LlamaGen] batch of {n} panels submitted: {generation_id}")
    return _poll_batch_until_done(generation_id, n)


def _poll_batch_until_done(generation_id: str, expected_count: int) -> list[bytes]:
    """Like _poll_until_done, but for a fixPanelNum > 1 request -- downloads
    EVERY panel's assetUrl in panel order, instead of grabbing just the first
    one the way _poll_until_done does (fine for fixPanelNum=1, would silently
    drop every other panel in a real batch).

    Timeout scales with expected_count: a real 6-panel batch timed out at the
    flat single-panel _POLL_TIMEOUT (300s), which is worse than just slow --
    our side gives up and pays again via the one-at-a-time fallback while
    LlamaGen's job likely keeps running (and billing) server-side regardless.
    """
    batch_timeout = max(_POLL_TIMEOUT, 120 * expected_count)
    deadline = time.time() + batch_timeout
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            resp = client.get(
                f"{LLAMAGEN_BASE}/v1/comics/generations/{generation_id}",
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "PROCESSED":
                panel_entries = []
                for comic in (data.get("comics") or []):
                    panel_entries.extend(comic.get("panels") or [])
                panel_entries.sort(key=lambda p: p.get("panel", 0))
                if len(panel_entries) != expected_count:
                    raise RuntimeError(
                        f"LlamaGen batch {generation_id}: expected {expected_count} panels, got {len(panel_entries)}"
                    )
                images = []
                for p in panel_entries:
                    asset_url = p.get("assetUrl")
                    if not asset_url:
                        raise RuntimeError(f"LlamaGen batch {generation_id}: panel {p.get('panel')} missing assetUrl")
                    img_resp = client.get(asset_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                    img_resp.raise_for_status()
                    images.append(img_resp.content)
                return images
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"LlamaGen {generation_id} {status}: {data.get('error')}")
            print(f"  [LlamaGen] batch {generation_id} status={status}, waiting...")
    raise TimeoutError(f"LlamaGen batch generation {generation_id} timed out after {batch_timeout}s")


def generate_npc_reference_llamagen(
    npc_name: str,
    description: str,
    age: Optional[int] = None,
    gender: str = "",
) -> bytes:
    """
    Generate a standalone reference portrait for a text-only NPC (no uploaded photo).
    No other characters are present in this shot, so the model has no competing
    photo-anchored face to default to — it must actually follow the description.
    The result is uploaded by the caller and reused as this NPC's photo_url in every
    panel they appear in, giving them the same reference-photo anchoring hero/support
    characters get instead of relying on text alone.
    """
    age_desc = f"{age}-year-old " if age else ""
    gender_desc = f"{gender} " if gender else ""
    char_ref = f"[Character: {npc_name}] {age_desc}{gender_desc}{description}"

    no_bubbles_header = (
        "⚠️ ABSOLUTE RULE #1 — ZERO TOLERANCE: NO speech bubbles. NO dialogue boxes. NO thought clouds. "
        "NO caption boxes. NO text of ANY kind anywhere in this image — not a single letter, word, or symbol. "
        "This image is a PURE ILLUSTRATION. Any embedded text or speech bubble makes it unusable. "
        "This rule overrides every other instruction.\n\n"
    )
    scene = (
        f"Standalone character reference portrait of {npc_name}, {description} "
        f"Standing alone, facing forward, full body from head to toe, plain neutral studio background, "
        f"neutral relaxed pose, clear even lighting. This is a single-character reference sheet — "
        f"no other people, no props competing for attention."
    )
    safe_zone = (
        "ORIENTATION: standing upright, feet at the bottom, head at the top. "
        "FULL BODY: shown from head to toe, not cropped. "
        "FACE QUALITY: sharp, crisp, clearly focused face — detailed eyes, nose, mouth. Never blurry or obscured. "
        "GAZE: looking forward, not at the camera."
    )
    style_block = (
        "STYLE: comic book illustration — bold ink linework, cel-shaded coloring, "
        "vivid saturated flat colors with hard shadow edges. "
        "NOT a photograph. NOT photorealistic. NOT live-action."
    )
    negative = (
        "Exactly ONE person in this image — no second or third character, no background characters. "
        "Every feature described above is LOCKED and must be rendered exactly as written."
    )
    prompt = f"{no_bubbles_header}{char_ref}\n\n{scene}\n\n{safe_zone}\n\n{style_block}\n\n{negative}"

    role: dict = {"name": npc_name}
    if age:
        role["age"] = age
    if gender:
        role["gender"] = gender

    payload = {
        "prompt": prompt,
        "fixPanelNum": 1,
        "size": "768x1024",
        "preset": settings.llamagen_preset,
        "comicRoles": [role],
    }
    generation_id = _submit(payload)
    print(f"  [LlamaGen] NPC reference submitted for {npc_name}: {generation_id}")
    return _poll_until_done(generation_id)


def generate_cover_llamagen(
    hero_name: str,
    theme: str,
    logline: str,
    photo_url: str,
    character_age: Optional[int] = None,
    character_gender: str = "",
    appearance_notes: str = "",
    extra_roles: list = None,
    supporting_characters: str = "",
    genre: str = "",
    retry_note: str = "",
) -> bytes:
    age_desc = f"{character_age}-year-old " if character_age else ""
    gender_desc = f"{character_gender} " if character_gender else ""
    # Default to adult when age is unknown — WhamComics customers are adults
    is_hero_adult = (character_age is None) or (character_age >= 18)
    age_group = "mature adult" if is_hero_adult else "child"
    notes_suffix = f", {appearance_notes}" if appearance_notes else ""
    char_ref = f"[Character: {hero_name}] {age_desc}{gender_desc}{age_group}{notes_suffix}."
    notes = f" {appearance_notes}." if appearance_notes else ""
    support_bg = (
        f"\n\nSUPPORTING CHARACTER ON COVER: {supporting_characters}. "
        f"This character appears BEHIND and to the side of {hero_name}, visible from head to toe, positioned well within "
        f"the frame — NOT in a corner, NOT touching the edge of the image, NOT partially cut off by the frame border. "
        f"{hero_name} is the clear HERO and largest figure — the supporting character appears roughly half {hero_name}'s height due to distance. "
        f"The supporting character must be clearly visible and recognizable, fully inside the frame with margin on every side — "
        f"NOT a tiny silhouette, NOT cropped, NOT hidden."
        if supporting_characters else ""
    )
    adult_note = (
        "ALL CHARACTERS IN THIS IMAGE ARE FULLY GROWN MATURE ADULTS (30s age range). "
        "No teenagers, no children — every person shown must look like a grown adult.\n\n"
        if is_hero_adult else ""
    )
    # Genre-specific mood guidance so the cover reflects the story's emotional tone
    genre_mood = {
        "comedy": f"The scene should feel energetic and chaotic — {hero_name}'s expression shows exasperation, disbelief, or being caught mid-disaster. NOT serious or stoic.",
        "sci_fi": f"The scene should feel tense and atmospheric — {hero_name}'s expression shows determination, urgency, or unease. The environment reflects the sci-fi world.",
        "horror_adult": f"The scene should feel ominous and unsettling — {hero_name}'s expression shows fear, dread, or grim resolve. Dark dramatic lighting.",
        "mystery": f"The scene should feel sharp and investigative — {hero_name}'s expression shows focus, suspicion, or a dawning realization.",
        "romance": f"The scene should feel warm and emotionally charged — {hero_name}'s expression shows longing, joy, or vulnerability.",
        "adventure": f"The scene should feel exciting and kinetic — {hero_name} is in the middle of action, expression showing intensity or thrill.",
    }.get(genre or "", f"The scene should reflect the emotional tone of the story. {hero_name}'s expression and body language must fit the moment.")

    no_bubbles_header = (
        "⚠️ ABSOLUTE RULE #1 — ZERO TOLERANCE: NO speech bubbles. NO dialogue boxes. NO thought clouds. "
        "NO caption boxes. NO text of ANY kind anywhere in this image — not a single letter, word, or symbol. "
        "This image is a PURE ILLUSTRATION. Any embedded text or speech bubble makes it unusable. "
        "This rule overrides every other instruction.\n\n"
    )
    presence_rule = (
        f"⚠️ ABSOLUTE RULE #0 — {hero_name.upper()} MUST BE PHYSICALLY PRESENT AND CLEARLY VISIBLE: "
        f"This is a cover for {hero_name}'s story — she MUST appear as an actual human figure in this "
        f"image, full body, in the lower portion of the frame. Do not render a pure environment/establishing "
        f"shot with no character in it, however dramatic the setting looks without her.\n\n"
    )
    retry_block = (
        f"⚠️ YOUR PREVIOUS ATTEMPT AT THIS COVER FAILED — {hero_name} did not appear as a visible human figure, "
        f"the image was environment/background only. This time she MUST be physically present and clearly "
        f"visible, full body, in the lower portion of the frame.\n\n"
        if retry_note else ""
    )
    no_bubbles_header = f"{no_bubbles_header}{retry_block}{presence_rule}"
    if genre == "horror_adult":
        # The rest of a horror cover is already dark/moody by design (see genre_mood
        # above), so a full 25% flat "empty dark" band has nothing to contrast against
        # and the whole cover reads as one solid black mass. Reserve less space and
        # keep it textured/moody rather than a plain void, so it looks intentional.
        top_band_rule = (
            f"⚠️ ABSOLUTE RULE #2 — TOP 15% RESERVED FOR TITLE, KEEP IT MOODY NOT EMPTY: The top 15% "
            f"of this image must stay relatively uncluttered for a title overlay, but it should still show "
            f"dim, moody environmental detail — a night sky with stars or moon, distant rooftops, hanging "
            f"signage, fog, tree branches — NOT a flat, textureless, pure-black void. "
            f"NO characters, NO faces in this top band. "
            f"{hero_name}'s head and body must appear in the LOWER 85% of the image only. "
            f"This space is reserved for the comic title — it will look like an empty mistake if left "
            f"completely blank, so keep faint atmospheric detail visible even though it's dark.\n\n"
        )
    else:
        top_band_rule = (
            f"⚠️ ABSOLUTE RULE #2 — TOP 25% MUST BE EMPTY DARK SKY/BACKGROUND: The top quarter of this image "
            f"(top 25% of height) must be a plain dark background — sky, night sky, dark wall, or dark solid color. "
            f"NO characters, NO faces, NO action in this top band. "
            f"{hero_name}'s head and body must appear in the LOWER 75% of the image only. "
            f"This space is reserved for the comic title — it will be ruined if a character or action is there.\n\n"
        )
    composition_rule = (
        f"⚠️ ABSOLUTE RULE #3 — KEEP EVERY CHARACTER WELL INSIDE THE FRAME: All characters must be positioned "
        f"within the center of the frame with clear margin on every side — never in a corner, never touching or "
        f"crossing the left/right/bottom edge of the image, never partially cut off by the frame border. "
        f"Every character shown must be fully visible head to toe with breathing room around them.\n\n"
    )
    prompt = (
        f"{no_bubbles_header}"
        f"{top_band_rule}"
        f"{composition_rule}"
        f"{char_ref}\n\n"
        f"{adult_note}"
        f"Comic book front cover. "
        f"SCENE: Show a specific dramatic moment from this story — {logline}. "
        f"{hero_name} is the main character and the LARGEST figure in the image, visible from head to toe in the LOWER 75% of the frame. "
        f"Her body language, expression, and surroundings must reflect what is actually happening in this story. "
        f"{genre_mood} "
        f"The setting behind her should reflect the world of the story: {theme}. "
        f"Vibrant colors, cinematic composition. "
        f"{hero_name}'s face is sharp, crisp, and in clear focus — detailed eyes, nose, and mouth visible. Not blurry, not softened, not smudged.{notes}"
        f"{support_bg}"
        f"\n\nEvery character must have a sharp, crisp, fully detailed face — no blurry faces, no out-of-focus faces, no smudged or softened faces, no blank faces, no faceless figures. Face must be in sharp focus with clearly defined eyes, nose, and mouth. "
        f"Do not add grey or white hair to any character — maintain exact hair color from their reference photo."
    )
    roles = [_build_role(hero_name, photo_url, character_age, character_gender)]
    roles.extend(extra_roles or [])
    payload = {
        "prompt": prompt,
        "fixPanelNum": 1,
        "size": PAGE_MATCHED_SIZE,
        "preset": settings.llamagen_preset,
        "comicRoles": roles,
    }
    generation_id = _submit(payload)
    print(f"  [LlamaGen] cover submitted: {generation_id}")
    return _poll_until_done(generation_id)
