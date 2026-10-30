import os
import zipfile
import requests
from io import BytesIO

import fal_client

from app.config import settings

FAL_TRAINING_MODEL = "fal-ai/flux-lora-portrait-trainer"
FAL_INFERENCE_MODEL = "fal-ai/flux-lora"
TRIGGER_WORD = "OHWX"


def _setup():
    os.environ["FAL_KEY"] = settings.fal_api_key


def upload_training_images(photo_bytes_list: list[bytes]) -> str:
    _setup()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, photo_bytes in enumerate(photo_bytes_list):
            zf.writestr(f"image_{i:03d}.jpg", photo_bytes)
    buf.seek(0)
    return fal_client.upload(buf.read(), content_type="application/zip")


def trigger_lora_training(zip_url: str, webhook_url: str = None) -> str:
    _setup()
    arguments = {
        "images_data_url": zip_url,
        "trigger_word": TRIGGER_WORD,
        "steps": 1000,
        "is_input_format_already_prepared": False,
        "create_masks": True,
    }
    handler = fal_client.submit(
        FAL_TRAINING_MODEL,
        arguments=arguments,
        webhook_url=webhook_url,
    )
    return handler.request_id


def get_training_status(request_id: str) -> dict:
    _setup()
    status = fal_client.status(FAL_TRAINING_MODEL, request_id, with_logs=False)
    status_type = type(status).__name__

    if status_type == "Completed":
        result = fal_client.result(FAL_TRAINING_MODEL, request_id)
        lora_url = result["diffusers_lora_file"]["url"]
        return {"status": "ready", "lora_url": lora_url}
    elif status_type == "InProgress":
        return {"status": "training", "lora_url": None}
    elif status_type == "Queued":
        return {"status": "pending", "lora_url": None}
    else:
        return {"status": "failed", "lora_url": None}


def generate_page_with_pulid(scene_prompt: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    hair_lock = ""
    if character_appearance:
        for part in character_appearance.split(","):
            part_stripped = part.strip()
            if any(w in part_stripped.lower() for w in ["loc", "dread", "braid", "hair", "curl", "afro"]):
                hair_lock = part_stripped + ", same hair length and style, "
                break

    style_trigger = f"{settings.style_lora_trigger} " if settings.style_lora_trigger else ""

    full_prompt = (
        f"{style_trigger}child{appearance_anchor} "
        f"{scene_prompt}, "
        f"{hair_lock}"
        # Framing
        "WIDE SHOT — all characters visible fully from head to toe, "
        "character head occupies no more than 15% of total image height, "
        "environment clearly visible around all characters, "
        "no close-ups, no face portraits, "
        # Style
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "rich detailed background, cinematic comic book composition, "
        # Supporting characters
        "ALL other characters look completely different from the main character: "
        "NO dreadlocks NO locs NO white glasses on any secondary character, "
        "ADULT characters are taller with adult face proportions — not child-sized, "
        + ""
    )

    loras = []
    if settings.style_lora_url:
        loras.append({"path": settings.style_lora_url, "scale": 0.65})

    arguments = {
        "prompt": full_prompt,
        "reference_image_url": reference_url,
        "id_weight": 0.85,
        "guidance_scale": 7.5,
        "num_inference_steps": 35,
        "image_size": "square_hd",
        "negative_prompt": (
            "close-up, face close-up, head filling frame, portrait, "
            "anime, manga, chibi, oversized eyes, huge eyes, anime eyes, "
            "bobblehead, big head, oversized head, "
            "photorealistic, photograph, photo, hyperrealistic, 3D render, "
            "looking at viewer, direct eye contact, "
            "two characters with dreadlocks, two characters with locs, "
            "two characters with white glasses, "
            "background characters with dreadlocks, background characters with locs, "
            "background characters with white glasses, "
            "adult drawn as child, coach with child face, all characters same height, "
            "extra hands, extra fingers, extra limbs, bad anatomy, "
            "ugly, blurry, low quality, deformed, watermark"
        ),
    }
    if loras:
        arguments["loras"] = loras

    result = fal_client.run("fal-ai/flux-pulid", arguments=arguments)

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_page_with_instantid(scene_prompt: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    hair_lock = ""
    if character_appearance:
        for part in character_appearance.split(","):
            part_stripped = part.strip()
            if any(w in part_stripped.lower() for w in ["loc", "dread", "braid", "hair", "curl", "afro"]):
                hair_lock = part_stripped + ", same hair length and style, "
                break

    full_prompt = (
        f"child{appearance_anchor} "
        f"{scene_prompt}, "
        f"{hair_lock}"
        "WIDE SHOT — all characters visible fully from head to toe, "
        "character head occupies no more than 15% of total image height, "
        "environment clearly visible around all characters, "
        "no close-ups, no face portraits, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "rich detailed background, cinematic comic book composition, "
        "ALL other characters look completely different from the main character: "
        "NO dreadlocks NO locs NO white glasses on any secondary character, "
        "ADULT characters are taller with adult face proportions — not child-sized, "
        + ""
    )

    result = fal_client.run(
        "fal-ai/flux-dev-with-instantid",
        arguments={
            "prompt": full_prompt,
            "image_url": reference_url,
            "controlnet_conditioning_scale": 0.8,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "num_images": 1,
            "image_size": "square_hd",
            "negative_prompt": (
                "close-up, face close-up, head filling frame, portrait, "
                "anime, manga, chibi, oversized eyes, huge eyes, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, "
                "looking at viewer, direct eye contact, "
                "two characters with dreadlocks, two characters with locs, "
                "two characters with white glasses, "
                "background characters with dreadlocks, background characters with locs, "
                "background characters with white glasses, "
                "adult drawn as child, all characters same height, "
                "extra hands, extra fingers, bad anatomy, "
                "ugly, blurry, low quality, deformed, watermark"
            ),
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_cover_with_instantid(hero_name: str, theme: str, logline: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    full_prompt = (
        f"child{appearance_anchor} "
        f"comic book cover illustration — {hero_name}: {logline}, "
        "heroic ground-level pose, hands on hips or arms at sides, confident smile, "
        "character occupies the bottom 55% of the image — top 40% is background only, "
        "no other characters on the cover, single character only, "
        "wide establishing shot, character fully visible from head to toe, "
        "character head takes up no more than 20% of image height, "
        "dramatic bright background matching the story theme, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "premium children's comic book cover composition, "
        "NO text NO words NO letters NO speech bubbles anywhere in the image"
    )

    result = fal_client.run(
        "fal-ai/flux-dev-with-instantid",
        arguments={
            "prompt": full_prompt,
            "image_url": reference_url,
            "controlnet_conditioning_scale": 0.8,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "num_images": 1,
            "image_size": "square_hd",
            "negative_prompt": (
                "close-up, face filling frame, portrait, head and shoulders only, "
                "anime, manga, chibi, oversized eyes, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, "
                "looking at viewer, direct eye contact, "
                "text, words, letters, watermark, speech bubble, "
                "multiple characters, two characters, background people"
            ),
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_page_with_instantcharacter(scene_prompt: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    hair_lock = ""
    if character_appearance:
        for part in character_appearance.split(","):
            if any(w in part.lower() for w in ["loc", "dread", "braid", "hair", "curl", "afro"]):
                hair_lock = part.strip() + ", same hair length and style, "
                break

    ca_lower = character_appearance.lower()
    if any(w in ca_lower for w in ["boy", " male"]):
        gender_word = "boy"
        gender_negative = "girl, female, feminine, feminine face, girly features, "
    elif any(w in ca_lower for w in ["girl", "female"]):
        gender_word = "girl"
        gender_negative = "boy, male, masculine, masculine face, "
    else:
        gender_word = "child"
        gender_negative = ""

    full_prompt = (
        f"{gender_word}{appearance_anchor} "
        f"{scene_prompt}, "
        f"{hair_lock}"
        "EXTREME WIDE SHOT — every character's full body visible from head to toe including feet on the ground, "
        "characters are small figures in a large open environment, "
        "character heads are tiny — no more than 8% of total image height, "
        "vast background fills at least 60% of the frame, "
        "no close-ups, no face portraits, no head-filling shots, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "rich detailed background, cinematic comic book panel composition, "
        "ALL supporting characters are school-age children 8-12 years old — "
        "EXACTLY the same height as the main character, equal height, side by side at equal height, "
        "NOT shorter, NOT smaller, NOT babies NOT toddlers, "
        "EVERY supporting character looks completely different from the main character — "
        "different hair, different face, different clothing: "
        "NO dreadlocks NO locs NO white glasses on any supporting character — "
        "give them short hair, braids, or afro — never locs never white glasses, "
        "RIVAL characters: short buzzcut, cornrows, or short curly — never dreadlocks, "
        + ""
    )

    result = fal_client.run(
        "fal-ai/instant-character",
        arguments={
            "prompt": full_prompt,
            "image_url": reference_url,
            "scale": 0.5,
            "guidance_scale": 5.5,
            "num_inference_steps": 30,
            "num_images": 1,
            "image_size": "landscape_4_3",
            "negative_prompt": (
                f"{gender_negative}"
                "close-up, face close-up, head filling frame, portrait, face portrait, "
                "medium close-up, tight shot, head and shoulders only, bust shot, "
                "baby, toddler, infant, tiny child, miniature person, short person, "
                "characters different heights, one character shorter, size mismatch, "
                "anime, manga, chibi, oversized eyes, huge eyes, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, "
                "looking at viewer, direct eye contact, "
                "two characters with same hairstyle, two characters with dreadlocks, "
                "two characters with locs, two characters with white glasses, "
                "background characters with dreadlocks, background characters with locs, "
                "background characters with white glasses, "
                "duplicate character, clones, identical twins, "
                "extra hands, extra fingers, extra limbs, bad anatomy, "
                "ugly, blurry, low quality, deformed, watermark"
            ),
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_cover_with_instantcharacter(hero_name: str, theme: str, logline: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    ca_lower = character_appearance.lower()
    if any(w in ca_lower for w in ["boy", " male"]):
        gender_word = "boy"
        gender_negative = "girl, female, feminine, feminine face, girly features, "
    elif any(w in ca_lower for w in ["girl", "female"]):
        gender_word = "girl"
        gender_negative = "boy, male, masculine, masculine face, "
    else:
        gender_word = "child"
        gender_negative = ""

    full_prompt = (
        f"{gender_word}{appearance_anchor} "
        f"comic book cover illustration — {hero_name}: {logline}, "
        "heroic ground-level pose, hands on hips or arms at sides, confident smile, "
        "character occupies the bottom 55% of the image — top 40% is background only, "
        "no other characters on the cover, single character only, "
        "wide establishing shot, character fully visible from head to toe, "
        "character head takes up no more than 20% of image height, "
        "dramatic bright background matching the story theme, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "premium children's comic book cover composition, "
        "NO text NO words NO letters NO speech bubbles anywhere in the image"
    )

    result = fal_client.run(
        "fal-ai/instant-character",
        arguments={
            "prompt": full_prompt,
            "image_url": reference_url,
            "scale": 0.75,
            "guidance_scale": 4.5,
            "num_inference_steps": 28,
            "num_images": 1,
            "image_size": "square_hd",
            "negative_prompt": (
                f"{gender_negative}"
                "close-up, face filling frame, portrait, head and shoulders only, "
                "anime, manga, chibi, oversized eyes, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, "
                "looking at viewer, direct eye contact, "
                "text, words, letters, watermark, speech bubble, "
                "multiple characters, two characters, background people"
            ),
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_cover_with_pulid(hero_name: str, theme: str, logline: str, reference_image_bytes: bytes, character_appearance: str = "") -> bytes:
    _setup()

    reference_url = fal_client.upload(reference_image_bytes, content_type="image/jpeg")

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    style_trigger = f"{settings.style_lora_trigger} " if settings.style_lora_trigger else ""

    full_prompt = (
        f"{style_trigger}child{appearance_anchor} "
        f"comic book cover illustration — {hero_name}: {logline}, "
        "heroic ground-level pose, hands on hips or arms at sides, confident smile, "
        "character occupies the bottom 55% of the image — top 40% is background only, "
        "no other characters on the cover, single character only, "
        "wide establishing shot, character fully visible from head to toe, "
        "character head takes up no more than 20% of image height, "
        "dramatic bright background matching the story theme, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "premium children's comic book cover composition, "
        "NO text NO words NO letters NO speech bubbles anywhere in the image"
    )

    loras = []
    if settings.style_lora_url:
        loras.append({"path": settings.style_lora_url, "scale": 0.65})

    arguments = {
        "prompt": full_prompt,
        "reference_image_url": reference_url,
        "id_weight": 0.85,
        "guidance_scale": 7.5,
        "num_inference_steps": 35,
        "image_size": "square_hd",
        "negative_prompt": (
            "close-up, face filling frame, portrait, head and shoulders only, "
            "anime, manga, chibi, oversized eyes, "
            "photorealistic, photograph, photo, hyperrealistic, 3D render, "
            "looking at viewer, direct eye contact, "
            "text, words, letters, watermark, speech bubble, "
            "multiple characters, two characters, background people"
        ),
    }
    if loras:
        arguments["loras"] = loras

    result = fal_client.run("fal-ai/flux-pulid", arguments=arguments)

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_page_with_lora(scene_prompt: str, lora_url: str, character_appearance: str = "") -> bytes:
    _setup()

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    hair_lock = ""
    if character_appearance:
        for part in character_appearance.split(","):
            if any(w in part.lower() for w in ["loc", "dread", "braid", "hair", "curl", "afro"]):
                hair_lock = part.strip() + ", same hair length and style, "
                break

    style_trigger = f"{settings.style_lora_trigger} " if settings.style_lora_trigger else ""

    full_prompt = (
        f"{style_trigger}{TRIGGER_WORD} is the main character{appearance_anchor} "
        f"{scene_prompt}, "
        f"{hair_lock}"
        # Framing rules
        "WIDE SHOT — main character and any other characters visible fully from head to toe, "
        "character head occupies no more than 15% of total image height, "
        "environment and background clearly visible around all characters, "
        "no close-ups, no face portraits, camera at least 15 feet away, "
        # Action — characters must be doing something, not posing
        "all characters fully engaged in the scene action, "
        "characters looking at each other or at objects in the scene — NOT at the camera, "
        "dynamic action pose, characters interacting with their environment, "
        # Style
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "rich detailed background, cinematic comic book panel composition, "
        # Supporting character differentiation — CRITICAL
        "EVERY non-main character must look exactly as described in the scene — "
        "follow their described hair color, style, clothing, and age precisely, "
        "ALL child supporting characters: NO dreadlocks NO locs NO white glasses — "
        "give them short hair, braids, or afro — never locs never white glasses, "
        "ADULT characters (coaches, teachers, parents, neighbors, rivals who are adults): "
        "must have adult face proportions, adult body height, adult bone structure — "
        "NOT child-sized, NOT child-faced — clearly taller and older than the child characters, "
        "RIVAL characters: must have completely different hair from the main character — "
        "short buzzcut, cornrows, or short curly — never dreadlocks or locs, "
        + ""
    )

    result = fal_client.run(
        FAL_INFERENCE_MODEL,
        arguments={
            "prompt": full_prompt,
            "negative_prompt": (
                "looking at viewer, looking at camera, staring at viewer, direct eye contact, "
                "posing for photo, posing for camera, facing camera, front-facing portrait, "
                "close-up, face close-up, head filling frame, portrait, face portrait, "
                "medium close-up, tight shot, head and shoulders only, "
                "anime, manga, chibi, oversized eyes, huge eyes, anime eyes, "
                "bobblehead, big head, oversized head, disproportionate head, "
                "infant proportions, toddler proportions, "
                "extra hands, extra fingers, extra limbs, bad anatomy, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, CGI, "
                "two characters with same hairstyle, two characters with dreadlocks, "
                "two characters with locs, two characters with white glasses, "
                "background characters with dreadlocks, background characters with locs, "
                "background characters with white glasses, "
                "duplicate character, clones, identical twins, "
                "adult drawn as child, coach with child face, teacher with child proportions, "
                "parent with child body, all characters same height, "
                "ugly, blurry, low quality, deformed, watermark, signature"
            ),
            "loras": [
                {"path": lora_url, "scale": 0.75},
                *([{"path": settings.style_lora_url, "scale": 0.7}] if settings.style_lora_url else []),
            ],
            "image_size": "square_hd",
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "num_images": 1,
            "output_format": "png",
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content


def generate_cover_with_lora(hero_name: str, theme: str, logline: str, lora_url: str, character_appearance: str = "") -> bytes:
    """Generates a cover page using the character LoRA with a FLUX-appropriate prompt."""
    _setup()

    appearance_anchor = (
        f" {character_appearance}," if character_appearance else ","
    )

    style_trigger = f"{settings.style_lora_trigger} " if settings.style_lora_trigger else ""

    full_prompt = (
        f"{style_trigger}{TRIGGER_WORD} is the main character{appearance_anchor} "
        f"comic book cover illustration for a children's story: {hero_name} — {logline}, "
        "heroic ground-level pose, hands on hips or arms at sides, confident smile, "
        "character occupies the bottom 55% of the image — top 40% is background only, "
        "no other characters on the cover, "
        "wide establishing shot, character fully visible from head to toe, "
        "character head takes up no more than 20% of image height, "
        "dramatic bright background matching the story theme, "
        "comic book illustration style, detailed ink linework, "
        "cel-shaded coloring with vivid saturated colors, "
        "premium children's comic book cover composition, "
        "NO text, NO words, NO letters, NO speech bubbles anywhere in the image"
    )

    result = fal_client.run(
        FAL_INFERENCE_MODEL,
        arguments={
            "prompt": full_prompt,
            "negative_prompt": (
                "close-up, face filling frame, portrait, head and shoulders only, "
                "anime, manga, chibi, oversized eyes, "
                "photorealistic, photograph, photo, hyperrealistic, 3D render, "
                "looking at viewer, direct eye contact, "
                "text, words, letters, watermark, signature, speech bubble"
            ),
            "loras": [
                {"path": lora_url, "scale": 0.75},
                *([{"path": settings.style_lora_url, "scale": 0.7}] if settings.style_lora_url else []),
            ],
            "image_size": "square_hd",
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "num_images": 1,
            "output_format": "png",
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return response.content
