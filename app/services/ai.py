import json
import base64
from typing import Optional
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, APIStatusError
from app.config import settings
from io import BytesIO



GENRE_STYLES = {
    "adventure": {
        "story_tone": "exciting and action-packed, full of exploration and discovery",
        "image_style": "dynamic poses, vibrant energetic colors, bold heroic lighting",
        "title_color": (255, 215, 0),
    },
    "mystery": {
        "story_tone": "mysterious and suspenseful (age-appropriate). Hidden clues, secrets to uncover, curious twists",
        "image_style": "slightly moody atmospheric lighting, curious investigative poses, warm shadows and dark corners",
        "title_color": (200, 140, 255),
    },
    "comedy": {
        "story_tone": "laugh-out-loud funny and silly with slapstick humor and unexpected surprises",
        "image_style": "bright cartoony energy, exaggerated joyful expressions, playful chaotic scenes",
        "title_color": (255, 140, 0),
    },
    "fantasy": {
        "story_tone": "magical and whimsical with enchanted worlds, mythical creatures, and wonder",
        "image_style": "glowing magical effects, enchanted lush environments, rich jewel-tone colors",
        "title_color": (120, 200, 255),
    },
    "sci_fi": {
        "story_tone": "futuristic and technological with space exploration, gadgets, and robots",
        "image_style": "futuristic settings, neon accent lighting, sleek technology, space or city backdrops",
        "title_color": (0, 220, 255),
    },
    "sports": {
        "story_tone": "athletic and competitive with teamwork, determination, and championship moments",
        "image_style": "dynamic action shots, athletic poses, stadium or field settings, team energy",
        "title_color": (255, 70, 70),
    },
    "horror": {
        "story_tone": "genuinely spooky and suspenseful for young readers — real mystery, unease, and age-appropriate fear about whether the presence is dangerous, not resolved until late. Content stays kid-appropriate (no violence, gore, or real danger) but the tone should NOT be softened into silly or cute from the start — think Coraline or Goosebumps, not a joke-a-minute cartoon",
        "image_style": "Halloween atmosphere with genuine shadow and eerie mood (dim lighting, fog, long shadows), warmed by orange and purple accent tones — unsettling first, cozy once the resolution lands",
        "title_color": (130, 255, 80),
    },
    "friendship": {
        "story_tone": "warm and heartfelt with themes of kindness, inclusion, and the power of friendship",
        "image_style": "vibrant saturated colors, joyful group scenes, bright cheerful environments with varied color palette — blues, greens, purples, pinks alongside warm tones",
        "title_color": (255, 140, 190),
    },
    "horror_adult": {
        "story_tone": "R-rated horror — visceral terror, graphic tension, real violence and its consequences, psychological dread. Think The Conjuring, Hereditary, or IT. Genuinely frightening with adult stakes",
        "image_style": "dark oppressive atmosphere, deep shadows with harsh accent lighting, unsettling and threatening environments, desaturated palette with blood-red or sickly green accent highlights, high contrast and disturbing mood",
        "title_color": (180, 255, 80),
    },
    "romance": {
        "story_tone": "R-rated adult romance — passionate desire, emotional intensity, sensual tension, and intimate connection. Think steamy romance novel or R-rated romantic drama. Suggestive and charged, fade-to-black for explicit moments",
        "image_style": "warm golden and deep rose lighting, intimate close compositions, candlelit or twilight environments, rich sensual color palette with deep reds, amber, and soft shadow",
        "title_color": (255, 100, 150),
    },
}


def get_client():
    return OpenAI(api_key=settings.openai_api_key)


def _call_claude_json(prompt: str, system: str, max_tokens: int = 64000, model: str = "claude-sonnet-5") -> str:
    """
    Calls Claude via the Messages API and returns raw JSON text. Claude has no
    equivalent to OpenAI's response_format={"type": "json_object"}. An
    assistant-turn "{" prefill (the usual workaround) is rejected outright by
    claude-opus-5 ("This model does not support assistant message prefill.
    The conversation must end with a user message."), so this instead adds an
    explicit instruction to the system prompt and defensively strips markdown
    code fences / any stray text outside the outermost {...} before parsing.
    No temperature param either -- claude-opus-5 rejects that too
    ("temperature is deprecated for this model"), unlike the OpenAI models
    used everywhere else in this file. max_tokens defaults very high (64000,
    not OpenAI's usual budget) because Claude 5 models use extended thinking
    by default and thinking tokens draw from the SAME max_tokens budget as
    the actual output -- both a 16000 and a 32000 budget were entirely
    consumed by thinking on a real 48-panel request (opus-5 and sonnet-5
    respectively), leaving zero tokens for the JSON itself and no text block
    in the response at all either time. A real opus-5 test that did get a
    full response produced genuinely excellent prose and native consistency,
    but took 30-40+ minutes and coincided with the container being killed,
    likely from holding a large response in memory -- switched the default
    to sonnet-5 to test whether it keeps enough of that quality at a speed
    that's actually usable.

    Test path only for now: added to compare Claude against gpt-4o-mini on
    generate_preview_pages_ai's panel-adaptation call specifically, the
    longest and most rule-dense prompt in this file (a dozen-plus stacked
    "CRITICAL" sections) and the exact call already shown to drop climax
    resolution and default to static action poses under gpt-4o-mini.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=1800.0)
    # Anthropic requires streaming for long-running requests -- a real 48-panel
    # request on the non-streaming .create() path came back cut off after only
    # 6575 characters, nowhere near the 32000-token budget, with no error
    # raised at all. .stream() avoids that silent early cutoff.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system + " Return ONLY the raw JSON object as your entire response -- no markdown code fences, no explanation, no text before or after it.",
        thinking={"type": "disabled"},
        messages=[
            {"role": "user", "content": prompt},
        ],
    ) as stream:
        for _ in stream.text_stream:
            pass
        response = stream.get_final_message()
    # response.content[0] isn't guaranteed to be the text block -- Claude 5
    # models can emit a thinking block before the text block, and indexing
    # [0] blindly grabbed that instead, whose .text is None.
    text_block = next((b for b in response.content if getattr(b, "type", None) == "text"), None)
    if text_block is None:
        raise RuntimeError(f"Claude response had no text block: {[getattr(b, 'type', None) for b in response.content]}")
    text = text_block.text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def extract_appearance_from_photo(photo_bytes: bytes, extra_photos: list[bytes] = None) -> dict:
    client = get_client()

    # Build image content blocks — first photo + any extras
    image_blocks = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(photo_bytes).decode('utf-8')}",
                "detail": "high",
            },
        }
    ]
    for extra in (extra_photos or []):
        image_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(extra).decode('utf-8')}",
                "detail": "high",
            },
        })

    photo_note = (
        f"You are looking at {1 + len(extra_photos or [])} photos of the same person. "
        "Use ALL photos together to get the most complete and accurate description. "
        "For clothing, use the photo that shows the most distinctive or colorful outfit — prefer photos where clothing is clearly visible over face-only selfies. "
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": image_blocks + [
                    {
                        "type": "text",
                        "text": f"""{photo_note}

Analyze this person and extract their physical appearance for use in AI illustration prompts.

Return ONLY valid JSON with this exact shape:
{{
  "gender": "male or female",
  "skin_tone": "specific skin tone description (e.g. deep dark brown, medium brown, light brown)",
  "hair": "specific hairstyle and color (e.g. long dark brown locs/braids reaching shoulders, short black tight curls)",
  "eye_color": "eye color if visible",
  "glasses": "glasses description if present — include frame shape, color, and thickness (e.g. wide white rectangular frames, thin gold wire frames), or null if none",
  "nose": "nose shape and width (e.g. broad and rounded, narrow and straight, wide flat nose, small button nose)",
  "clothing": "most distinctive clothing visible across all photos — describe colors, patterns, and garment type precisely (e.g. black varsity jacket with red left sleeve and yellow right sleeve)",
  "age_range": "approximate age range (e.g. 28-32 years old)"
}}

Be specific and descriptive. These details will be used to maintain visual consistency in illustrated images.""",
                    },
                ],
            }
        ],
        max_tokens=600,
        timeout=30,
    )

    content = response.choices[0].message.content
    return json.loads(content)

def generate_character_reference_ai(photo_bytes: bytes, extra_photos: list[bytes] = None) -> bytes:
    client = get_client()

    prompt = """
    Redraw this child as a premium digital illustration character — keeping their identity completely intact.

    SKIN TONE (CRITICAL — most important rule):
    - Copy the skin tone from the photo WITH EXACT DEPTH AND RICHNESS — do NOT lighten it
    - If the child has very dark brown skin in the photo, the illustration must have equally very dark brown skin
    - Do NOT shift the skin tone toward medium brown, caramel, or lighter tones
    - The illustrated skin must be the same depth of darkness as the photo — or darker if anything, never lighter
    - Use rich deep brown tones with subtle warm highlights — the darkest shadow areas should be very deep brown, not grey

    AGE (CRITICAL — the most common mistake is making the child look too old):
    - Draw this child EXACTLY the same age as they appear in the photo — do NOT age them up
    - Do NOT draw a teenager or young adult — this is a YOUNG CHILD
    - Young child face anatomy (these apply regardless of face shape):
      * Eyes appear LARGE relative to the face — proportionally bigger than an adult's
      * Nose is SHORT and SOFT — a child's nose does not project far from the face
      * Chin is SMALL and ROUNDED — NOT pronounced, NOT angular
      * Forehead is large relative to the lower face
      * No sharp angles, no defined cheekbones, no adult bone structure
      * Skin is smooth and soft — no shadows that create a chiseled or mature look
    - If the result looks like it could be a teenager — it is WRONG — make it younger

    FACE STRUCTURE (CRITICAL — copy the photo, do NOT impose a different shape):
    - Face shape: copy the EXACT face shape from the photo — if the photo shows a round face, draw a round face — if oval, draw oval — if slim, draw slim — do NOT force any particular shape
    - Jawline: match the jaw width from the photo exactly — NOT wider, NOT narrower
    - Chin: same chin shape as in the photo — copy it, do not change it
    - Nose: copy the exact width and shape from the photo — do NOT narrow or reshape it
    - Eyes: same natural size and shape as the photo — do NOT enlarge, do NOT make rounder or anime-style
    - Mouth: natural smile — mouth width must NOT exceed the distance between the pupils — NOT a wide ear-to-ear grin — lips closed or showing just a hint of teeth
    - The illustrated face must be RECOGNIZABLE as the same child from the photo

    PROPORTIONS (CRITICAL — match the source photo exactly):
    - Head size: same ratio to body as in the photo — head must be NO larger than 1/4 of total body height — do NOT enlarge the head — NOT chibi
    - Body: realistic child proportions matching the photo — NOT chibi, NOT oversized head
    - If the child has glasses: same exact frame shape, thickness, and color as the photo

    IDENTITY — copy from the photo exactly:
    - Same skin tone (see SKIN TONE block above — do NOT lighten)
    - Same hair: exact color, length, style, and texture (locs stay locs, braids stay braids — copy exact darkness of hair color)
    - Same clothing visible in the photo
    - If the child is a BOY: natural male features only — no eyelashes, no rosy cheeks, no feminine features

    STYLE TARGET (CRITICAL — this reference sets the style for all comic panels):
    - Premium professional comic book illustration — like a modern DC or Image Comics graphic novel for younger readers
    - Clean precise ink linework — medium-weight lines, sharp and detailed like a professional comic book artist's inking — NOT overly thick or chunky
    - Cel-shading: flat color base with 2-3 sharp tonal shadow zones — clean graphic shading, NOT soft gradient blending
    - Skin: flat base tone with 1-2 sharp shadow zones — high contrast between lit and shadowed areas, all tones remain deep dark brown
    - Hair: flat base color with precise ink strand lines and sharp shadow zones — detailed and defined
    - Vibrant, saturated comic book colors — bright, punchy, high-contrast palette
    - Richly detailed backgrounds — fully illustrated environments with intricate details, depth, and texture
    - Characters pop off richly illustrated backgrounds with strong ink line contrast

    COMPOSITION:
    - Full upper body visible, facing forward or slight 3/4 angle
    - Simple neutral or white background
    - This is a CHARACTER REFERENCE SHEET used for consistency across many comic panels

    STRICTLY AVOID:
    - Lightening or brightening the skin tone in any way
    - Rounding or softening the face shape away from the photo
    - Enlarged or oversized head (chibi style)
    - Enlarged, anime-style, or glossy eyes
    - Eyelashes, blush, rosy cheeks, lip gloss
    - Exaggerated expressions or proportions
    - Any feminine features if the child is a boy
    - Soft gradient blending or 3D render look
    - Watercolor wash or painted brush strokes
    - Photorealistic rendering
    """

    images = []
    primary = BytesIO(photo_bytes)
    primary.name = "photo_0.jpg"
    images.append(primary)
    for i, extra in enumerate(extra_photos or []):
        buf = BytesIO(extra)
        buf.name = f"photo_{i + 1}.jpg"
        images.append(buf)

    result = client.images.edit(
        model="gpt-image-2.5-sunburst",
        image=images if len(images) > 1 else images[0],
        prompt=prompt,
        size="1024x1024",
    )

    image_base64 = result.data[0].b64_json
    return base64.b64decode(image_base64)



def build_personalization_block(personalization: dict) -> str:
    """Convert stored personalization answers into a readable prompt block."""
    if not personalization:
        return ""
    lines = []
    if personalization.get("age"):
        age_val = personalization["age"]
        if age_val >= 18:
            age_note = f"This is an ADULT aged {age_val}. Write mature, sophisticated content — complex themes, adult humor, real-world stakes. NOT a children's story."
        else:
            age_note = f"Write vocabulary, humor, and story complexity appropriate for a {age_val}-year-old."
        lines.append(f"- Age: {age_val} years old ({age_note})")
    if personalization.get("favorite_activities"):
        lines.append(f"- Loves: {', '.join(personalization['favorite_activities'])}")
    if personalization.get("favorite_animal"):
        lines.append(f"- Favorite animal: {personalization['favorite_animal']}")
    if personalization.get("personality_traits"):
        lines.append(f"- Personality: {', '.join(personalization['personality_traits'])}")
    if personalization.get("favorite_place"):
        lines.append(f"- Favorite place: {personalization['favorite_place']}")
    if personalization.get("dream"):
        lines.append(f"- Dream: {personalization['dream']}")
    if personalization.get("special_skill"):
        lines.append(f"- Special skill: {personalization['special_skill']}")
    if personalization.get("job"):
        lines.append(f"- Job/occupation: {personalization['job']} (weave this into the story setting and plot)")
    if personalization.get("expertise"):
        lines.append(f"- Knows more about than anyone: {personalization['expertise']} (use this as a plot advantage at a key moment)")
    if personalization.get("catchphrase"):
        lines.append(
            f"- Catchphrase: \"{personalization['catchphrase']}\" — this is THIS character's own personal "
            f"catchphrase (the character named above), not a line anyone else uses. Have THEM say it as dialogue "
            f"at a key moment — never put it in a supporting character's or NPC's mouth, even if the moment would "
            f"otherwise suit them."
        )
    return "\n".join(lines)


def generate_story_options_ai(hero_name: str, traits: list[str], theme: str, count: int, personalization: dict = None, genre: str = "adventure", support_characters: list = None):
    personalization_block = build_personalization_block(personalization or {})
    genre_style = GENRE_STYLES.get(genre, GENRE_STYLES["adventure"])
    p = personalization or {}
    audience_age = p.get("age") or 10
    if audience_age >= 18:
        audience_desc = "personalized comic books for adults. Stories should have sophisticated themes, adult humor, real-world stakes, and mature vocabulary."
        audience_note = "Use the character's details to make each story feel genuinely tailored to this specific adult."
    else:
        audience_desc = "personalized comic books for kids."
        audience_note = f"Use the character's personal details above to make each story option feel genuinely tailored to this specific child.\nWeave in their hobbies, favorite places, personality, and dreams naturally — don't just list them, use them to shape the plot."
    support_block = ""
    if support_characters:
        names = ", ".join(support_characters)
        support_block = (
            f"\nSupporting characters (MUST appear in every story option by name): {names}\n"
            f"Each story option must give {names} a meaningful role — not just a passing mention.\n"
        )
    comedy_options_note = (
        f"""
COMEDY STORY REQUIREMENTS (CRITICAL — comedy-specific, applies on top of STORY DEPTH above):
1. CHARACTER-SPECIFIC PREMISE: The comedy must emerge from {hero_name}'s specific personality traits — NOT generic "things go wrong." Read the personality traits above and make the disaster happen BECAUSE of who {hero_name} is.
2. NAMED SIDE CHARACTERS: Every NPC in a comedy story option MUST have an ACTUAL FIRST NAME (e.g. "Marcus and Tasha") and ONE DEFINING PERSONALITY TRAIT that creates specific comedic friction. Labels like "the competitive couple" are NEVER acceptable.
3. ESCALATION ARC: The outline must explicitly describe: (a) what the FIRST specific funny thing is that goes wrong, (b) how that gets WORSE when they try to fix it, (c) what the PEAK CHAOS moment looks like.
4. FUNNY ENDING: The resolution must be a specific comedic payoff — not "they succeed and learn a lesson." The ending should connect back to something funny from the setup and land a final laugh.
- WRONG outline: "Ceiara and Vincent try to complete an escape room and face challenges, ultimately learning to work together."
- RIGHT outline: "Ceiara, convinced she can break the room record, takes charge immediately — and accidentally triggers the emergency alarm on page one. Rival Marcus (who narrates his own performance in the third person like a sports commentator) keeps solving Ceiara's puzzles before she can, while his partner Tasha keeps accidentally making things worse. By the final room, all three teams' disasters have merged into one spectacular mess that only Ceiara can fix — by doing the exact thing she swore she'd never do: ask for help."
"""
        if genre == "comedy" else ""
    )
    prompt = f"""
You are a creative writer for {audience_desc}

Create {count} distinct story options for a comic.

Main character:
- Name: {hero_name}
- Traits: {", ".join(traits) if traits else "brave"}
{personalization_block}
{support_block}
Theme: {theme}
Genre: {genre} — {genre_style['story_tone']}

The genre sets the tone and feeling of the entire story. Every plot beat, conflict, and resolution must match the genre.
{audience_note}

STORY DEPTH (CRITICAL — applies to every genre):
Every story option must have SPECIFIC, CONCRETE details — not vague generic premises.
- The conflict must be SPECIFIC: not "something goes wrong" but exactly what goes wrong, to whom, and why it matters
- The stakes must be REAL: what does the hero stand to lose if they fail? Make it personal to {hero_name}
- The story must have a CLEAR ARC: a specific problem introduced, specific obstacles faced, specific resolution earned
- Use the character's personal details (traits, interests, relationships) to shape the specific conflict — not just mention them
- The outline must answer: What exactly is the problem? What makes it hard? How is it resolved?
- NAMED SIDE CHARACTERS (required for every option, all genres): Name at least one significant side character with an actual first name and one personality trait — never a label like "a rival," "a bully," or "a helpful stranger." Give them a name and one defining trait.
- ESCALATION ARC (required for every option, all genres): The outline must include a midpoint crisis — a specific moment where the situation gets significantly harder or worse. Without this, the story has no shape.
- WRONG: "Ceiara goes on an adventure and discovers something amazing" — too vague
- RIGHT: "Ceiara discovers her grandmother's missing recipe box was stolen by a neighbor, and must piece together clues from old photos before the community bake-off to restore her grandmother's legacy"
{comedy_options_note}
Return valid JSON only.
Return an array of objects.
Each object must have:
- logline: 1 sentence (specific conflict and stakes — not vague)
- outline: 2 to 3 sentences (name the specific problem, the midpoint crisis where it gets harder, and the specific resolution)
- estimated_pages: 24
"""

    client = get_client()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You must return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt + '\nWrap the result in this object shape: {"options": [...]}'
                }
            ],
            temperature=0.9,
            timeout=20,
        )

        content = response.choices[0].message.content
        print("RAW OPENAI CONTENT:", repr(content))

        data = json.loads(content)
        return data["options"]

    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI connection failed: {repr(e)}")
    except AuthenticationError as e:
        raise RuntimeError(f"OpenAI authentication failed: {repr(e)}")
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI quota/rate limit failed: {repr(e)}")
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI status error {e.status_code}: {repr(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI returned non-JSON content: {e}")
    except KeyError as e:
        raise RuntimeError(f"OpenAI JSON missing expected key: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI error: {type(e).__name__}: {e}")
    
def _enforce_outfit_across_pages(pages: list[dict], outfit: str, client, hero_name: str = "the main character") -> list[dict]:
    """Rewrite any page whose image_prompt doesn't contain the canonical outfit."""
    if not outfit or not pages:
        return pages

    normalized = []
    for page in pages:
        ip = page.get("image_prompt", "")
        # Check if the outfit description is already present (case-insensitive substring check on key terms)
        outfit_words = [w for w in outfit.lower().split() if len(w) > 4]
        match_count = sum(1 for w in outfit_words if w in ip.lower())
        already_correct = match_count >= max(1, len(outfit_words) // 2)

        if already_correct:
            normalized.append(page)
            continue

        try:
            fix = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rewrite this image prompt so {hero_name} SPECIFICALLY — and ONLY {hero_name} — wears EXACTLY: {outfit}\n"
                        f"Find where {hero_name}'s clothing is described (or should be described) and replace it with this exact outfit. "
                        f"Do NOT touch the clothing description of any OTHER named character in the prompt — "
                        f"if another character's outfit is described, leave it word-for-word as it is. "
                        "Keep every other detail identical.\n"
                        "Return ONLY the rewritten prompt — no explanation.\n\n"
                        f"Original: {ip}"
                    ),
                }],
                max_tokens=600,
                temperature=0,
                timeout=20,
            )
            page = dict(page)
            page["image_prompt"] = _strip_control_chars(fix.choices[0].message.content.strip())
        except Exception:
            pass  # leave original if fix fails
        normalized.append(page)

    return normalized


def _strip_control_chars(text: str) -> str:
    """Drop stray ASCII control bytes (keeping \\t\\n\\r) from AI-generated text.

    Found in production: gpt-4o-mini's raw-text outfit-rewrite completion
    (_enforce_outfit_across_pages above) occasionally emits a literal 0x14
    control character in place of an em dash. That corrupted image_prompt was
    sent to LlamaGen verbatim -- and every panel it hit rendered a correct
    background with ZERO characters, twice, across two full live test runs.
    LlamaGen's own prompt parser most likely treats a raw control byte as a
    field terminator, silently dropping everything after it -- which is
    exactly where the character descriptions and position/reference-photo
    instructions live in these prompts. Stripping control chars here (and at
    the LlamaGen call boundary itself) removes the trigger.
    """
    if not text:
        return text
    return "".join(c for c in text if ord(c) >= 32 or c in "\t\n\r")


def generate_prose_story_ai(
    hero_name: str,
    theme: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    genre: str = "adventure",
    additional_characters: list = None,
    page_count: int = 24,
    comedy_blueprint: dict = None,
    adventure_blueprint: dict = None,
    mystery_blueprint: dict = None,
    horror_blueprint: dict = None,
    sci_fi_blueprint: dict = None,
    horror_kids_blueprint: dict = None,
    fantasy_blueprint: dict = None,
) -> str:
    """
    Write a complete prose story that will be used as source material for panel generation.
    This two-pass approach prevents the 'outline feel' that occurs when the AI invents the
    story and formats it as JSON panels simultaneously.
    """
    genre_style = GENRE_STYLES.get(genre, GENRE_STYLES["adventure"])
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    audience_age = p.get("age") or 10
    is_adult = audience_age >= 18
    word_count = 2000 if page_count <= 24 else 3000

    support_block = ""
    if additional_characters:
        support_block = "\nSupporting characters (use these exact names throughout):\n"
        for char in additional_characters:
            rel = char.get("relationship", "")
            line = f"- {char['name']}"
            if rel:
                line += f" ({hero_name}'s {rel})"
            if char.get("nickname_for_hero"):
                line += f" — calls {hero_name} '{char['nickname_for_hero']}'"
            if char.get("hero_nickname_for_them"):
                line += f" — {hero_name} calls them '{char['hero_nickname_for_them']}'"
            support_block += line + "\n"

    if genre == "comedy":
        genre_guidance = f"""
COMEDY REQUIREMENTS (non-negotiable):
- The comedy MUST emerge from {hero_name}'s specific personality traits — NOT generic bad luck that could happen to anyone
- Give every recurring NPC an actual first name (never a label like "the competitive couple") and ONE defining personality trait that creates a specific running comedic bit
- Write at least 3 specific, concrete funny moments — name exactly what goes wrong and describe the specific ridiculous reaction
- Each funny moment must compound and ESCALATE the previous one — by the peak chaos, multiple earlier disasters collide at once
- End with a punchline or ironic callback — NOT "they succeeded and felt good about themselves"
- Include actual funny dialogue lines at key moments — lines that are specific to what just happened, not generic
- PACING: The resolution after the climax must be BRIEF (2-3 scenes max). Do NOT write a warm epilogue. End on the punchline.
- THE RESOLUTION MUST BE A REAL FIX, NOT A CHANGE OF ATTITUDE: {hero_name} must take a specific corrective action that actually repairs, salvages, or makes right the real external damage the chaos caused. Deciding as a group that the disaster was secretly good all along does NOT count as a resolution — nothing has been fixed. If the original goal was to avoid or prove they wouldn't cause the very disaster that happened, simply celebrating the disaster instead undercuts the character rather than resolving anything.
- THE FINAL BEAT SHOULD BE A VISUAL GAG, NOT ONLY A SPOKEN LINE: the last thing on the page should be something the reader can picture happening — a prop, an object, or a minor character's small action — ideally the payoff of the story's own planted callback, not a line of dialogue summarizing what was learned.
"""
        if comedy_blueprint:
            moments = comedy_blueprint.get("funny_moments", [])
            npc_bits = comedy_blueprint.get("npc_bits", {})
            beats_text = ""
            for i, m in enumerate(moments, 1):
                beats_text += f"\n  Beat {i}: {m.get('what_happens', '')}"
                if m.get("character_reaction"):
                    beats_text += f"\n    → {hero_name}'s reaction: {m['character_reaction']}"
                if m.get("npc_involvement"):
                    beats_text += f"\n    → NPC behavior: {m['npc_involvement']}"
                if m.get("caused_by") and i < len(moments):
                    beats_text += f"\n    → Directly causes next beat: {m['caused_by']}"
            npc_bits_text = "\n".join(f"  - {name}: {bit}" for name, bit in npc_bits.items()) if npc_bits else "  (none)"
            char_dynamics = comedy_blueprint.get("character_dynamics", {})
            dynamics_text = "\n".join(f"  - {name}: {desc}" for name, desc in char_dynamics.items()) if char_dynamics else "  (none)"
            genre_guidance += f"""
MANDATORY COMEDY BLUEPRINT (architect-designed — follow this exactly, do not substitute vague versions):
Premise: {comedy_blueprint.get('comedy_premise', '')}

ORIGINAL GOAL (must be referenced in the punchline ending — do NOT let this disappear mid-story):
{comedy_blueprint.get('original_goal', '')}

CHARACTER DYNAMICS (every named character must actively drive the comedy — no spectators):
{dynamics_text}

CAUSAL CHAIN — these are the specific events of your story in order:
{beats_text}

Peak chaos: {comedy_blueprint.get('peak_chaos_description', '')}
Resolution beat (brief — this must be a real corrective ACTION that actually fixes/repairs/salvages the external problem, not just everyone deciding the disaster was secretly fine): {comedy_blueprint.get('resolution_beat', '')}

MANDATORY PUNCHLINE ENDING — the story MUST end on this exact moment (no warm epilogue after this). This should be a concrete VISUAL gag involving a prop, object, or minor character's action — not just a spoken line — ideally the payoff of the callback plant below:
{comedy_blueprint.get('punchline_ending', '')}

NPC recurring comedic bits (each must appear as a running joke across 3+ scenes, escalating each time):
{npc_bits_text}

LOCATION RULE: The story must stay in the primary setting for at least 80% of the panels.
Do NOT end the story with characters walking home and chatting — the punchline happens INSIDE the setting.
"""
    elif genre == "mystery":
        genre_guidance = f"""
MYSTERY REQUIREMENTS:
- State the mystery with specific details: exactly what happened, when, and the initial apparent circumstances
- Name the culprit and give them a SPECIFIC motive — spell it out, don't just say "they were jealous"
- Each clue must be a specific physical or factual detail — an object, a timeline contradiction, a physical observation — and must state what it actually proves
- The midpoint must reveal the mystery is significantly more complex than it first appeared
- The resolution must use specific clues established earlier — not a confession out of nowhere
- FAIR PLAY: every fact the solution depends on must be shown to the reader as it's discovered — never introduced for the first time at the reveal
- THE REVEAL IS NOT THE CLIMAX: naming the culprit only TRIGGERS the climax — it must be followed by a real confrontation with risk (the culprit flees, attacks, denies it until confronted with the one fact they can't explain), not a calm announcement followed immediately by resolution
- THE MYSTERY MUST ACTUALLY BE SOLVED: by the end, the reader must know who the victim was, who did it, why, and how — stated or clearly shown, never left implied. A confident-sounding ending is not a substitute for actually answering these.
- EVERY SUSPECT GETS CLOSURE: any named suspect built up with real evidence must be explicitly addressed by the end — cleared or otherwise accounted for. Never let a suspect the story made the reader doubt simply vanish from the page unresolved.
- NO OMNISCIENT NARRATOR: Never narrate the true culprit's private thoughts, hidden feelings, or secret actions taken off-page BEFORE {hero_name} solves the case — no cutaway scenes showing the culprit doing something incriminating alone, and no line revealing what they were "really" thinking while acting innocent. Everything before the reveal must stay inside what {hero_name} (or another honest character) can actually observe. The culprit's inner truth only becomes visible at or after the confrontation.
- {hero_name} SOLVES IT — NOT THE CULPRIT'S CONFESSION: The deduction that identifies the culprit must be {hero_name}'s own reasoning, stated on the page BEFORE any admission from the culprit. If the culprit says anything that sounds like a confession, it happens AFTER {hero_name} has already accused them using their own logic — it corroborates, it never reveals.
"""
        if mystery_blueprint:
            genre_guidance += f"""
MANDATORY MYSTERY BLUEPRINT (architect-designed — follow this exactly, do not substitute a different plot):
CENTRAL MYSTERY (state this explicitly, in-story, within the first few scenes): {mystery_blueprint.get('central_mystery', '')}
VICTIM: {mystery_blueprint.get('victim', '')}
STAKES — what is specifically lost if unsolved: {mystery_blueprint.get('stakes', '')}
TRUE CULPRIT (locked — reveal only at the climax, do not change who actually did it): {mystery_blueprint.get('true_culprit', '')}
TRUE MOTIVE: {mystery_blueprint.get('true_motive', '')}
METHOD (how they actually did it — must come out at the climax): {mystery_blueprint.get('method', '')}
OTHER SUSPECTS (each has real means/motive/opportunity AND its own innocent_explanation — EVERY one of them must get closure in the resolution, not only the red herring): {mystery_blueprint.get('suspects', '')}
RED HERRING — who {hero_name} (and the reader) will genuinely suspect first, and why: {mystery_blueprint.get('red_herring', '')}
RED HERRING EXPLANATION (must be explicitly delivered in the resolution — do not drop this suspect): {mystery_blueprint.get('red_herring_explanation', '')}

CLUE 1 (must be discovered ON THE PAGE, not summarized as backstory): {mystery_blueprint.get('clue_1', '')}
COMPANION'S CONTRIBUTION (must actually happen on the page — something the case needs, not just encouragement): {mystery_blueprint.get('companion_contribution', '')}
MIDPOINT COMPLICATION: {mystery_blueprint.get('midpoint_complication', '')}
CLUE 2 (must be discovered ON THE PAGE): {mystery_blueprint.get('clue_2', '')}
KEY CONTRADICTION (the specific detail that breaks the red herring theory): {mystery_blueprint.get('key_contradiction', '')}
CLUE 3 (must be discovered ON THE PAGE, points to the TRUE culprit): {mystery_blueprint.get('clue_3', '')}
LOW POINT (case looks unsolvable, or {hero_name} is doubted/endangered): {mystery_blueprint.get('low_point', '')}

FINAL DEDUCTION (show {hero_name} actually reasoning through this on the page — connecting the clues to the solution, not just "knowing"): {mystery_blueprint.get('final_deduction', '')}
PROOF (this specific earlier clue must be the undeniable evidence at the climax): {mystery_blueprint.get('proof', '')}
CLIMAX (the confrontation the accusation triggers — {hero_name} must personally act, at real risk; no NPC, coincidence, or sudden unexplained idea may resolve it for them): {mystery_blueprint.get('climax', '')}
RESOLUTION (show the crime's outcome, the TRUE culprit held accountable, EVERY other named suspect from "suspects" EXPLICITLY cleared using their own innocent_explanation — not only the red herring — and the ORIGINAL stakes specifically settled): {mystery_blueprint.get('resolution', '')}

Do not deviate from this spine's culprit, motive, method, clues, or climax mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "sports":
        genre_guidance = f"""
SPORTS REQUIREMENTS — follow this exact eight-step shape, in order. Do not compress, skip, or merge steps:
1. DREAM — establish {hero_name}'s specific goal early: not vague "do well," but a concrete target (make the team, land a specific named move, win a specific title/spot).
2. SETBACK — something concrete and real threatens that dream (an injury, a mistake, a rival, a lost resource/routine). This must be a genuine problem, not a minor inconvenience brushed off in a sentence.
3. PANIC — {hero_name} has a real moment of doubt, fear, or wanting to quit. Let this actually land on the page — do not skip past it in one line.
4. FAILED ATTEMPT — {hero_name} tries to fix or overcome the setback and it does NOT fully work. A genuine failure or partial failure — not a token stumble immediately followed by success.
5. SMALL BREAKTHROUGH — a partial success or new idea that creates real hope — but NOT full mastery yet. This is not the end of the story's tension.
6. BIGGER SETBACK — a SECOND, LARGER complication arrives AFTER the small breakthrough and CLOSE to the climax — something worse than step 2, raising the stakes right when things seemed to be improving. THIS STEP IS MANDATORY. Do not skip from "small breakthrough" straight to "final performance" — that is the flat, banned shape. Things must get worse again before the climax.
7. FINAL PERFORMANCE — the climax. {hero_name} draws on everything learned, including from the failures, to perform. Include a specific moment of real risk or a near-mistake mid-performance — not a smooth, clean, uneventful run.
8. EMOTIONAL PAYOFF — the resolution ties back to the SPECIFIC dream from step 1, showing what achieving (or not fully achieving) it means to {hero_name} — not a generic celebration that could close any story.

BANNED — do not write this shape: problem → immediate emotional support from another character → a simple solution is proposed → practice → it works → celebration. That arc has only one dip and reads as rushed and generic regardless of how well any individual scene is written.
"""
    elif genre == "adventure":
        genre_guidance = f"""
ADVENTURE REQUIREMENTS:
- THE QUEST: establish a SPECIFIC, concrete objective early — a named place to reach, a named object to find, a specific challenge to complete. Not vague "have an adventure" — the reader must know exactly what {hero_name} is trying to achieve and why it matters to them personally.
- THE PATH IS NOT ONE OBSTACLE REPEATED: the journey must include multiple DISTINCT obstacles, each a different kind of challenge (a physical hazard, a puzzle/locked path, a creature or rival, a resource running out) requiring a different specific solution — not the same "obstacle appears, hero pushes through" beat repeated with new scenery.
- EVERY NEW LOCATION MUST BE VISUALLY AND TEXTURALLY SPECIFIC — name concrete sensory details (the exact terrain, structures, sounds, hazards) rather than generic "a mysterious cave" or "a dense forest." Each setting the party moves through should be instantly distinguishable from the last.
- REAL STAKES: obstacles must carry genuine consequence for failure — physical danger, running out of time, or a real cost — not challenges that exist just to be cleared for fun.
- IF A SUPPORT CHARACTER OR COMPANION IS PRESENT, they contribute a SPECIFIC skill or action that helps overcome at least one obstacle — not just encouragement from the sidelines.
- FOLLOW THE TWO-DIP ARC (see the structural rule above), specialized for adventure: an early obstacle that goes wrong → a small breakthrough (a clue found, a shortcut discovered, a tool acquired) that is NOT the full solution → a BIGGER setback close to the climax (the path collapses, the goal is stolen/blocked, the party is separated, a worse danger is revealed) that raises real stakes right before the final push → the climax obstacle/confrontation, which should include a moment of genuine risk or near-failure → a resolution that pays off the SPECIFIC quest objective established at the start, not a generic "great adventure" wrap-up.
"""
        if adventure_blueprint:
            genre_guidance += f"""
MANDATORY ADVENTURE BLUEPRINT (architect-designed — follow this exactly, do not substitute a different plot):
MISSION (state this explicitly, in-story, within the first few scenes): {adventure_blueprint.get('mission', '')}
MOTIVATION — why {hero_name} personally cares: {adventure_blueprint.get('motivation', '')}
DESTINATION / UNKNOWN {hero_name} must enter: {adventure_blueprint.get('destination_or_unknown', '')}
STAKES — what is specifically lost if {hero_name} fails: {adventure_blueprint.get('stakes', '')}

OBSTACLE 1 (must directly threaten the mission above): {adventure_blueprint.get('obstacle_1', '')}
DISCOVERY (changes {hero_name}'s understanding or approach): {adventure_blueprint.get('discovery', '')}
OBSTACLE 2 (harder, caused by earlier events/choices, must directly threaten the mission): {adventure_blueprint.get('obstacle_2', '')}
MIDPOINT CHANGE (the mission becomes harder, more urgent, or different): {adventure_blueprint.get('midpoint_change', '')}
OBSTACLE 3 / MAJOR SETBACK ({hero_name}'s current plan fails here): {adventure_blueprint.get('obstacle_3', '')}
LOW POINT (success looks genuinely unlikely): {adventure_blueprint.get('low_point', '')}

CLIMAX ({hero_name} must personally resolve this — no NPC, coincidence, or sudden unexplained idea may solve it for them): {adventure_blueprint.get('climax', '')}
PAYOFF (this specific earlier element must become essential at the climax — plant it clearly at the obstacle/discovery where it's established, then use it here): {adventure_blueprint.get('payoff', '')}
RESOLUTION (show the mission's outcome and how {hero_name} changed): {adventure_blueprint.get('resolution', '')}

Do not deviate from this spine's mission, obstacles, or climax mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "horror":
        genre_guidance = f"""
KID-FRIENDLY HORROR REQUIREMENTS (genuinely spooky and suspenseful, with kid-appropriate CONTENT — think Coraline, Goosebumps, or Hocus Pocus, NOT a joke-a-minute cartoon):
- "KID-FRIENDLY" IS A CONTENT RESTRICTION, NOT A TONE RESTRICTION — this is the single most important rule for this genre, and the one most commonly gotten wrong. No real violence, no injury, no genuine physical danger, nothing graphic. But that does NOT mean the story should read as cute, silly, or obviously friendly from the start. The reader must NOT know whether the presence is dangerous until late in the story — real suspense, real unease, real age-appropriate fear, exactly like a good ghost story. Writing every "scare" as giggling or a silly cartoon sound gives the game away in scene one and kills all suspense — that is a comedy, not horror, even though it's perfectly clean.
- Decide the "spooky" creature's full nature NOW — its name, what it actually is, why it's here, what it wants — but this is for YOUR planning only. The reader must NOT learn any of it early, and the way early incidents are WRITTEN must not hint that the presence is friendly.
- ONE CREATURE ONLY — decide on exactly ONE spooky character before writing a single scene, and commit to it for the entire story. Every strange happening and the final resolution must all be about this SAME creature. Do NOT introduce a second, different creature partway through.
- WITHHOLD THE REVEAL, AND KEEP IT UNSETTLING — introduce the mystery through small, genuinely eerie, easily-explained-away-but-lingering incidents first (an unexplained cold draft, a shape at the edge of vision, something moved with no one there, an unexplained sound). Write these straight, the way a real ghost story would, not as comic relief. Anyone who explains part of the truth may only reveal a FRAGMENT at a time, raising a new question with each answer. The full picture must not be complete until near the end.
- THE STORY MUST HAVE A GENUINE THIRD-ACT REVERSAL — this is not optional. Kids must genuinely (and reasonably) believe the creature is mean or wants to cause trouble, act on that belief with a real plan to get rid of it, and then discover a specific new fact proving they had it backwards — the creature was lonely, scared, or misunderstood the whole time. The real ending must be a MEANINGFULLY DIFFERENT action than the "get rid of it" attempt — helping it, not just trying harder to scare it off.
- THE REVELATION MUST NOT END THE HORROR — learning the truth launches the final challenge, it does not immediately make the story safe. Right after the reversal, something must still be unresolved, unstable, or frightening (the creature still trapped/fading, a closing window of time, an atmosphere still oppressive) — a genuinely tense final sequence, not a calm conversation. Only AFTER that final challenge succeeds may the story turn reassuring.
- THE PROTAGONIST'S ESTABLISHED FEAR MUST BECOME PART OF THAT FINAL CHALLENGE — during the climax, {hero_name} must be forced to actually DO the specific thing they're afraid of (from their established worry), or face it head-on, as part of completing the resolving action. Show this growth through what {hero_name} DOES in the moment — never by simply narrating that they're no longer scared.
- THE CLIMAX MUST BE A DRAMATIZED SCENE, NOT A SUMMARIZED MONTAGE — this is commonly gotten wrong even when every other climax requirement is technically present. Write the climax in real time, with at least one discrete, decisive beat: a specific action performed at a specific instant, a specific line of dialogue, a single sharp turning point the reader can picture happening at one precise moment. Do NOT compress the climax into several paragraphs of general activity summary ("gathered the pieces... worked on it... each careful stitch was a testament to...") with no scene-level moment — that reads as backstory summary, not a climax, even if it mentions fear and tension throughout. If you catch yourself narrating a PROCESS in past-tense summary rather than staging a SCENE with a concrete blow-by-blow, stop and rewrite it as an actual moment-to-moment scene.
- THE ENDING MUST BE RESOLVED BY A CONCRETE, KIND ACTION — {hero_name} must actually DO something specific and tangible (return a lost object, help fix or rebuild something, throw a specific surprise, complete a specific task) to make things right. Reassuring words alone ("you're not alone," "we're your friends now") are not enough on their own — an action must be what actually resolves things.
- THE ENDING MUST FEEL WARM AND FUN, NOT UNSETTLED — this is the one place the story should turn cozy: once the final challenge is won, end on complete, cozy safety and a specific fun/warm final detail (a joke, a cute callback, a heartwarming image) — never a lingering scare or ambiguous note. The suspense belongs to the FIRST two-thirds AND the climax itself; the warmth belongs only to what comes after.
"""
        if horror_kids_blueprint:
            genre_guidance += f"""
MANDATORY KID-FRIENDLY HORROR BLUEPRINT (architect-designed — follow this exactly, do not substitute a different creature or premise):
CENTRAL MYSTERY (state this explicitly, in-story, within the first few scenes — reveal nothing more yet): {horror_kids_blueprint.get('central_mystery', '')}
CREATURE (locked — reveal only gradually, full picture not complete until reversal_discovery; this is the ANSWER known only to you — do not let it leak into how early incidents are written): {horror_kids_blueprint.get('creature_name', '')} — {horror_kids_blueprint.get('creature_nature', '')}
ORIGIN (gentle, never violent — but this is also secret; do not reveal it early): {horror_kids_blueprint.get('origin_or_backstory', '')}
UNFINISHED BUSINESS: {horror_kids_blueprint.get('unfinished_business', '')}
WHAT IT WANTS (and why {hero_name} specifically): {horror_kids_blueprint.get('what_it_wants', '')}
METHOD OF SPOOKINESS (harmless in reality, but MUST be written as genuinely unsettling and ambiguous — no giggling or cute cartoon sounds that telegraph friendliness): {horror_kids_blueprint.get('method_of_spookiness', '')}
STAKES (kid-appropriate, never physical danger): {horror_kids_blueprint.get('stakes', '')}

{hero_name}'S WORRY (establish early, specific — not generic): {horror_kids_blueprint.get('protagonist_worry', '')}

INCIDENT 1 (small, genuinely unsettling and ambiguous, easily explained away by an adult but lingers — reveals NOTHING, and gives no hint of friendliness): {horror_kids_blueprint.get('incident_1', '')}
INCIDENT 2 (stronger and more unsettling, PLUS the first fragment of the truth — named source): {horror_kids_blueprint.get('incident_2', '')}
MIDPOINT DISCOVERY (reframes understanding, still a fragment — named source, still leaves real doubt about danger): {horror_kids_blueprint.get('midpoint_discovery', '')}
INCIDENT 3 (the most unsettling incident yet, a direct reaction to the investigation — content stays non-violent, but written to feel genuinely tense): {horror_kids_blueprint.get('incident_3', '')}
INCIDENT 4 (the presence's activity unsettlingly touches {hero_name}'s specific worry — real fear in the moment, not played for laughs): {horror_kids_blueprint.get('incident_4', '')}
LOW POINT (the point of maximum dread — mystery feels unsolved, stakes feel at risk — this happens LATE, no premature resolution): {horror_kids_blueprint.get('low_point', '')}

FALSE THEORY ({hero_name} genuinely believes the creature is mean/dangerous by now — plausible, but WRONG): {horror_kids_blueprint.get('false_theory', '')}
FALSE RESOLUTION ATTEMPT ({hero_name} and friends try to trap/scare off the creature — a reasonable plan that should work): {horror_kids_blueprint.get('false_resolution_attempt', '')}
REVERSAL DISCOVERY (a genuine "we had it wrong" turn — the creature isn't mean at all, it's lonely/misunderstood — but this must NOT immediately resolve the danger; something is still unresolved/frightening right after this lands): {horror_kids_blueprint.get('reversal_discovery', '')}

CLIMAX (the REAL final challenge, beginning only AFTER the reversal — but the fear/tension is NOT over just because the truth is known; this must be a genuinely suspenseful sequence until the action succeeds — {hero_name} must personally take a concrete KIND ACTION, tactically DIFFERENT from the false attempt — helping, not getting rid of): {horror_kids_blueprint.get('climax', '')}
FEAR CONFRONTATION ({hero_name}'s established worry above must become an ACTIVE obstacle right here, during the climax action — they must do the specific thing they're afraid of, shown through action, not narration): {horror_kids_blueprint.get('fear_confrontation', '')}
PAYOFF (this specific earlier-established detail is what {hero_name} uses in the climax): {horror_kids_blueprint.get('payoff', '')}
RESOLUTION (explicit and warm — the creature's happy outcome and unfinished_business's fate): {horror_kids_blueprint.get('resolution', '')}
FINAL BEAT (the very last beat must include this specific warm/fun detail — the story ends on complete, cozy, fun safety, nothing spooky lingering): {horror_kids_blueprint.get('final_beat', '')}

Do not deviate from this spine's creature, backstory, incidents, or resolution mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "horror_adult":
        genre_guidance = f"""
HORROR REQUIREMENTS:
- Decide the supernatural threat's full backstory NOW — their name, the era (e.g. "1943"), exactly what happened to them, what they want — but this is for YOUR planning only. The reader must NOT learn all of it early.
- ONE THREAT ONLY — decide on exactly ONE antagonist/supernatural entity before writing a single scene, and commit to it for the entire story. Every eerie detail, every clue an NPC gives, and the final confrontation must all be about this SAME entity. Do NOT introduce a second, different threat (a cult, a different spirit, a new villain) partway through or at the climax — that is a structural error, not a twist. If you want a twist, it must be a new REVELATION about the one established threat (e.g. it wasn't acting alone, or its true motive was different than assumed) — never a brand-new, previously-unmentioned antagonist.
- WITHHOLD THE REVEAL — this is a mystery, not an info-dump: introduce the haunting through small, ambiguous, unsettling incidents first (an object moved, a sound, a cold spot, a half-glimpsed figure). Any character who explains the backstory (a local, an NPC, a newspaper clipping, etc.) may only reveal PART of it at a time, and should raise a new question along with each answer. The full picture — name, era, exact cause of death, what the spirit wants — must not be complete until at or near the climax.
- MULTIPLE ESCALATING INCIDENTS — the haunting must manifest at least 4-5 separate times across the story, spaced out and escalating in intensity (a flicker → a slammed door → a voice → physical contact → full manifestation). Do not concentrate the supernatural activity into one or two panels near the end.
- The protagonist's insecurity or fear that the antagonist later exploits psychologically MUST be established or hinted at EARLY in the story (a specific line, memory, or moment) — so that when the spirit later attacks {hero_name} with it, it lands as a payoff, not a random accusation out of nowhere.
- ADULT HORROR CONTENT: The story may contain intense supernatural danger, disturbing psychological themes, frightening imagery, moderate-to-strong violence where appropriate, blood or injury when narratively justified, and sustained mortal peril. Do NOT rely on gore alone to create horror. Prioritize dread, disturbing implications, psychological vulnerability, escalating danger, unsettling imagery, and consequences. Match the intensity to what THIS threat actually is — a physically violent entity should show real physical consequence; a psychological/manipulative entity should lean into dread and disturbing implication instead of forcing in violence that doesn't fit.
- At least one scene must put the protagonist in a CREDIBLE possibility of death or severe harm — not just feeling "creeped out"
- Show real fear: the protagonist's reactions must be proportional to what they're experiencing
- THE CLIMAX NEEDS ROOM TO BREATHE — the final confrontation must be a sustained sequence (roughly the last quarter of the story), not a single quick appearance immediately followed by resolution. Let the danger build, peak, and only then resolve.
- THE STORY MUST END WITH A CLEAR, UNAMBIGUOUS RESOLUTION — the final panel must explicitly show the outcome: the threat is defeated/banished/satisfied, OR the protagonist escapes, OR the protagonist is caught/consumed — pick one and show it happening. Never end on an open question, a vague implication, or silence with no stated outcome. A reader finishing the last panel must know exactly what happened.
- The climax must feel life-or-death with a resolution tied to the ONE established threat's specific unfinished business — not a new mystery introduced at the very end
"""
        if horror_blueprint:
            genre_guidance += f"""
MANDATORY HORROR BLUEPRINT (architect-designed — follow this exactly, do not substitute a different threat):
CENTRAL HAUNTING (state this explicitly, in-story, within the first few scenes — reveal nothing more yet): {horror_blueprint.get('central_haunting', '')}
ENTITY (locked — reveal only gradually, full picture not complete until at/near the climax): {horror_blueprint.get('entity_name', '')}, from {horror_blueprint.get('entity_era', '')}
ORIGIN: {horror_blueprint.get('entity_origin', '')}
DEATH/CREATION EVENT (do not reveal in full early): {horror_blueprint.get('death_or_creation_event', '')}
UNFINISHED BUSINESS: {horror_blueprint.get('unfinished_business', '')}
WHAT IT WANTS (and why {hero_name} specifically): {horror_blueprint.get('what_it_wants', '')}
METHOD OF HARM: {horror_blueprint.get('method_of_harm', '')}
WEAKNESS/BANISHMENT METHOD (plant clearly before the climax, then use it there — this must be a PHYSICAL action {hero_name}'s hands actually perform, not merely words spoken; if the description below mentions revealing/exposing a truth, that truth must be paired with a simultaneous physical act that is what actually does the work): {horror_blueprint.get('weakness_or_banishment_method', '')}
STAKES: {horror_blueprint.get('stakes', '')}

{hero_name}'S EXPLOITED INSECURITY (establish early, specific — not generic): {horror_blueprint.get('protagonist_exploited_insecurity', '')}

INCIDENT 1 (small, ambiguous, deniable — reveals NOTHING): {horror_blueprint.get('incident_1', '')}
INCIDENT 2 (stronger, PLUS the first fragment of the backstory — named source): {horror_blueprint.get('incident_2', '')}
MIDPOINT DISCOVERY (reframes understanding, still a fragment — named source): {horror_blueprint.get('midpoint_discovery', '')}
INCIDENT 3 (genuine physical danger): {horror_blueprint.get('incident_3', '')}
INCIDENT 4 (entity directly exploits {hero_name}'s specific insecurity by name/detail): {horror_blueprint.get('incident_4', '')}
LOW POINT (survival looks unlikely, entity STILL at full active power — this happens LATE, roughly the final 10-15% of the story, no premature safety before it): {horror_blueprint.get('low_point', '')}

FALSE THEORY ({hero_name} genuinely believes this is the whole truth by now — plausible, but WRONG or dangerously incomplete): {horror_blueprint.get('false_theory', '')}
FALSE RESOLUTION ATTEMPT ({hero_name} acts on the false theory — a reasonable attempt that should work, and the reader expects it to): {horror_blueprint.get('false_resolution_attempt', '')}
REVERSAL DISCOVERY (a genuine "we had it wrong" turn — a SPECIFIC new fact, with a stated source, that proves false_theory wrong right as/after the false attempt fails or backfires): {horror_blueprint.get('reversal_discovery', '')}

CLIMAX (the REAL confrontation, beginning only AFTER the reversal — {hero_name} must personally act, at real risk, using the weakness/banishment method AS TACTICAL LEVERAGE, tactically DIFFERENT from the false attempt — sustained, not a quick beat — begins LATE, roughly the final 10-15% of the story): {horror_blueprint.get('climax', '')}
PAYOFF (this specific earlier element MECHANICALLY resolves the REAL climax — it disempowers/binds/destroys/fulfills a real condition; it is NOT resolved merely by {hero_name} speaking a kind truth or the entity feeling heard, unless that truth is itself the specific ritual condition): {horror_blueprint.get('payoff', '')}
RESOLUTION (explicit and unambiguous — defeated/banished/satisfied, OR escaped, OR caught/consumed — and unfinished_business's fate must be addressed): {horror_blueprint.get('resolution', '')}
FINAL STING (the very last beat must include this specific unsettling detail — it does not contradict the resolution above, but the story does NOT end on complete, cozy safety): {horror_blueprint.get('final_sting', '')}

Do not deviate from this spine's entity, backstory, incidents, or climax mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "sci_fi":
        genre_guidance = f"""
SCI-FI REQUIREMENTS:
- GENUINE SPECULATIVE ELEMENT REQUIRED — the central technology/phenomenon must go MEANINGFULLY beyond present-day real-world reality and must actually drive the plot. An app, a smart device, or a database with a shady business practice attached (corporate spyware, a hacked system) is NOT science fiction on its own — those already exist today. The technology must include a genuinely impossible-today capability: reading/altering thoughts or memories, real machine consciousness with its own goals, probability/matter manipulation, biological/genetic augmentation, non-local or faster-than-light effects, altered time perception, contact with something not human. Decide the central technology's exact function NOW, with real internal logic — a mechanism a reader could actually follow, not magic dressed as tech. It must work the SAME way from the first incident to the climax — no new, unforeshadowed powers appearing just to make the ending work.
- ONE DEEPENING HIDDEN TRUTH ABOUT THE TECHNOLOGY ITSELF — decide the specific concealed fact about what the technology/phenomenon actually IS or DOES (not just "who's secretly profiting from it," which is a mundane institutional story). Every incident, every clue, and the climax must all point toward this SAME hidden truth.
- PROGRESSIVE DISCOVERY, NOT FLAT REVEALS — each major discovery must answer one question about the technology while raising a DEEPER, BIGGER question. The mystery should genuinely escalate in strangeness across the story, not just add more evidence of the same-sized mystery.
- THE INVESTIGATION MUST HAVE A GENUINE THIRD-ACT REVERSAL — this is not optional flavor. {hero_name} must settle on a plausible-but-wrong, relatively MUNDANE theory (a rogue employee, ordinary corporate greed), act on it with a reasonable attempt, and then discover a specific new fact that proves the theory wrong. The FINAL revelation must change {hero_name}'s understanding of the technology/phenomenon ITSELF, not just reveal which human was behind it — "it turns out person X did it" alone is a thriller twist, not a sci-fi one.
- THE CLIMAX MUST BE RESOLVED BY A CONCRETE PHYSICAL OR TECHNICAL ACTION — {hero_name} personally cutting power, planting code, destroying hardware, rerouting a system, or physically extracting/broadcasting data through a specific technical means. A speech, a realization, or "exposing the truth" so people simply agree to stop is NOT a resolution — it must be something {hero_name}'s own hands or skills actually DO to the technology.
- THE CLIMAX MUST ALSO TEST THE CENTRAL DILEMMA THE TECHNOLOGY CREATES — do not let the climax collapse into a generic hacking/chase/action sequence (sneaking past guards, typing a shutdown command while alarms blare) that could belong to any techno-thriller. The unique philosophical/experiential question the technology's specific capability raises for {hero_name} must become an active obstacle DURING the physical/technical action itself, not just background flavor established earlier and abandoned by the finale.
- At least one scene must put {hero_name} in genuine danger directly caused by the technology or the institution retaliating — not just intellectual unease.
- THE ENDING MUST COST SOMETHING REAL — do not let {hero_name} walk away fully vindicated, promoted, and completely fine with nothing lost. A genuine sci-fi ending includes a specific, at-least-partly-irreversible cost or sacrifice even in success.
- THE STORY MUST END WITH A CLEAR, EXPLICIT RESOLUTION — state plainly what happened to the technology, to who controls it, and to {hero_name}. Then add ONE final, specific, thought-provoking detail (the same tech surfacing elsewhere, an unanticipated personal cost) that keeps the ending from wrapping up too neatly — never end on a vague "things will be different now."
"""
        if sci_fi_blueprint:
            genre_guidance += f"""
MANDATORY SCI-FI BLUEPRINT (architect-designed — follow this exactly, do not substitute a different technology or premise):
CENTRAL PREMISE (state this explicitly, in-story, within the first few scenes — reveal nothing more yet): {sci_fi_blueprint.get('central_premise', '')}
TECHNOLOGY: {sci_fi_blueprint.get('technology_name', '')}
FUNCTION (must work consistently throughout — no new unforeshadowed powers at the climax): {sci_fi_blueprint.get('technology_function', '')}
ORIGIN: {sci_fi_blueprint.get('technology_origin', '')}
HIDDEN TRUTH (locked — reveal only gradually, full picture not complete until reversal_discovery): {sci_fi_blueprint.get('hidden_truth', '')}
CENTRAL DILEMMA (the unique question this specific capability creates — the climax below MUST force {hero_name} to confront this in the moment, not resolve as generic action/hacking divorced from it): {sci_fi_blueprint.get('central_dilemma', '')}
WHO CONTROLS IT (and their real motive): {sci_fi_blueprint.get('who_controls_it', '')}
WHAT IT COSTS: {sci_fi_blueprint.get('what_it_costs', '')}
STAKES: {sci_fi_blueprint.get('stakes', '')}

{hero_name}'S GOAL (concrete, measurable): {sci_fi_blueprint.get('protagonist_goal', '')}
{hero_name}'S BLIND SPOT (specific, tied to their personality — will turn out wrong/incomplete): {sci_fi_blueprint.get('protagonist_blind_spot', '')}

INCIDENT 1 (small, explainable away — reveals NOTHING): {sci_fi_blueprint.get('incident_1', '')}
INCIDENT 2 (stronger, PLUS the first fragment — named source — must ANSWER a small question while RAISING a bigger one): {sci_fi_blueprint.get('incident_2', '')}
MIDPOINT DISCOVERY (reframes understanding, still a fragment — named source — must ANSWER incident_2's question while RAISING an even bigger one): {sci_fi_blueprint.get('midpoint_discovery', '')}
INCIDENT 3 (a DIRECT CONSEQUENCE of investigating this far — the technology/institution reacting, not an unrelated danger): {sci_fi_blueprint.get('incident_3', '')}

FALSE THEORY ({hero_name} genuinely believes this MUNDANE explanation is the whole truth by now — plausible, but WRONG or incomplete): {sci_fi_blueprint.get('false_theory', '')}
FALSE RESOLUTION ATTEMPT ({hero_name} acts on the false theory — a reasonable attempt that should work): {sci_fi_blueprint.get('false_resolution_attempt', '')}
REVERSAL DISCOVERY (THE FINAL, MOST MIND-CHANGING REVELATION — must change understanding of what the technology/phenomenon ITSELF actually is, not just reveal which human is responsible; also proves false_theory wrong right as/after the attempt fails or backfires): {sci_fi_blueprint.get('reversal_discovery', '')}

LOW POINT (survival/success looks unlikely, technology/institution STILL at full active danger — this happens LATE, roughly the final 10-15% of the story): {sci_fi_blueprint.get('low_point', '')}

CLIMAX (the REAL confrontation, beginning only AFTER the reversal — {hero_name} must personally perform a CONCRETE PHYSICAL OR TECHNICAL ACTION, tactically DIFFERENT from the false attempt — sustained, not a quick beat — begins LATE — CRITICALLY, central_dilemma above must become an active obstacle DURING this action, not resolved by it and not absent from it): {sci_fi_blueprint.get('climax', '')}
PAYOFF (this specific earlier-established tool/access/knowledge is what {hero_name} physically/technically uses — state MECHANICALLY what it does; NOT resolved merely by a speech or realization): {sci_fi_blueprint.get('payoff', '')}
RESOLUTION (explicit — what happened to the technology, to who_controls_it, and to {hero_name} — MUST include a genuine, at-least-partly-irreversible cost to {hero_name}, not a fully clean win): {sci_fi_blueprint.get('resolution', '')}
FINAL IMPLICATION (the very last beat must include this specific lingering detail — it does not contradict the resolution above, but the story does NOT wrap up too neatly): {sci_fi_blueprint.get('final_implication', '')}

Do not deviate from this spine's technology, hidden truth, incidents, or climax mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "fantasy":
        genre_guidance = f"""
FANTASY REQUIREMENTS:
- GENUINE MAGIC SYSTEM WITH A REAL COST REQUIRED — decide the magic system's exact function NOW, with real internal logic a reader could follow, AND a genuine, specific cost or limit that actually hurts to pay (a physical toll, a moral corruption, a finite shared resource, a life debt, lost memory, shortened lifespan). Magic that solves problems for free, with no real price, is not a magic system — it's a plot-solving wildcard. The cost must apply consistently from the first use to the climax.
- A SPECIFIC, LIVED-IN WORLD — name the kingdom/realm/region and give it at least one distinguishing detail (a custom, a geography, a political structure) used meaningfully in the story — not a generic "magical kingdom" with interchangeable fantasy filler.
- ONE DEEPENING HIDDEN TRUTH ABOUT THE THREAT/MAGIC/WORLD ITSELF — decide the specific concealed fact about what the threat, the magic, or the world actually IS (not just "who's secretly behind it," which is a mundane reveal). Every incident and the climax must point toward this SAME hidden truth.
- PROGRESSIVE DISCOVERY, NOT FLAT REVEALS — each major discovery must answer one question while raising a DEEPER, BIGGER question. The mystery should genuinely escalate in strangeness, not just add more evidence of the same-sized mystery.
- THE QUEST MUST HAVE A GENUINE THIRD-ACT REVERSAL — {hero_name} must settle on a plausible-but-wrong theory, act on it with a reasonable attempt, and then discover a specific new fact that proves the theory wrong. The FINAL revelation must change {hero_name}'s understanding of the threat/magic/world ITSELF, not just reveal a new name for who's responsible.
- REVELATIONS MUST DEEPEN, NOT CONTRADICT — each major discovery should preserve what was true in earlier clues while revealing a larger context, hidden cost, misunderstood relationship, or unexpected implication. Avoid repeatedly declaring that previous explanations were simply wrong; if the earlier belief truly must be overturned, explain why it seemed true in the first place.
- THE CLIMAX'S MAGICAL SOLUTION MUST FOLLOW LOGICALLY FROM RULES ESTABLISHED EARLIER — decide a SPECIFIC rule, limitation, or behavior of the magic system, plant it concretely in an early incident, and make the climax succeed BECAUSE {hero_name} applies that exact rule where the false attempt didn't. The solution must be deducible from what a careful reader already knows — not justified only by {hero_name}'s emotional growth or a vague realization with no mechanical grounding.
- THE CLIMAX MUST USE THE ESTABLISHED MAGIC MECHANISM, NOT A NEW CONVENIENT POWER — {hero_name} must resolve the climax using the SAME magic system established earlier, via a specific already-planted tool/knowledge/ally. A sudden, unforeshadowed new power appearing to save the day at the climax is a structural error, not a satisfying twist.
- THE MAGIC'S COST MUST ACTUALLY BE PAID AT THE CLIMAX, NOT DODGED — this is the single most important rule for this genre. {hero_name} must genuinely incur the established cost of the magic, in the moment, as an unavoidable part of winning — not find a convenient loophole or one-time exception that lets them avoid the price. The cost paid must be real and lasting, reflected in the resolution.
- At least one scene must put {hero_name} in genuine danger directly caused by the threat — not just intellectual unease.
- THE ENDING MUST COST SOMETHING REAL — do not let {hero_name} walk away fully triumphant with nothing lost. A genuine fantasy ending includes a specific, at-least-partly-irreversible cost or sacrifice even in success.
- THE STORY MUST END WITH A CLEAR, EXPLICIT RESOLUTION — state plainly what happened to the threat, the world, and {hero_name}. Then add ONE final, specific, thought-provoking detail (the magic's cost visibly lingering, a fragment of the threat surviving elsewhere) that keeps the ending from wrapping up too neatly.
"""
        if fantasy_blueprint:
            genre_guidance += f"""
MANDATORY FANTASY BLUEPRINT (architect-designed — follow this exactly, do not substitute a different world, magic, or threat):
CENTRAL PREMISE (state this explicitly, in-story, within the first few scenes — reveal nothing more yet): {fantasy_blueprint.get('central_premise', '')}
WORLD: {fantasy_blueprint.get('world_name', '')}
MAGIC SYSTEM: {fantasy_blueprint.get('magic_system_name', '')}
FUNCTION (must work consistently throughout — no new unforeshadowed powers at the climax): {fantasy_blueprint.get('magic_system_function', '')}
COST (a genuine price that must actually be paid — never conveniently avoided): {fantasy_blueprint.get('magic_system_cost', '')}
HIDDEN TRUTH (locked — reveal only gradually, full picture not complete until reversal_discovery): {fantasy_blueprint.get('hidden_truth', '')}
CENTRAL THREAT: {fantasy_blueprint.get('central_threat_name', '')}
WHAT IT WANTS (and why {hero_name} specifically): {fantasy_blueprint.get('what_it_wants', '')}
STAKES: {fantasy_blueprint.get('stakes', '')}

{hero_name}'S GOAL (concrete, measurable): {fantasy_blueprint.get('protagonist_goal', '')}
{hero_name}'S BLIND SPOT (specific, tied to their personality — will turn out wrong/incomplete): {fantasy_blueprint.get('protagonist_blind_spot', '')}

INCIDENT 1 (small, explainable away — reveals NOTHING): {fantasy_blueprint.get('incident_1', '')}
INCIDENT 2 (stronger, PLUS the first fragment — named source — must ANSWER a small question while RAISING a bigger one): {fantasy_blueprint.get('incident_2', '')}
MIDPOINT DISCOVERY (reframes understanding, still a fragment — named source — must ANSWER incident_2's question while RAISING an even bigger one): {fantasy_blueprint.get('midpoint_discovery', '')}
MAGIC RULE FORESHADOWED (a specific rule/limitation/behavior of the magic planted in incident_2 or midpoint_discovery above — the climax below MUST succeed BECAUSE {hero_name} applies this exact rule, not because of a vague realization): {fantasy_blueprint.get('magic_rule_foreshadowed', '')}
INCIDENT 3 (a DIRECT CONSEQUENCE of the quest reaching this point — the threat reacting, not an unrelated danger): {fantasy_blueprint.get('incident_3', '')}

FALSE THEORY ({hero_name} genuinely believes this is the whole truth by now — plausible, but WRONG or incomplete): {fantasy_blueprint.get('false_theory', '')}
FALSE RESOLUTION ATTEMPT ({hero_name} acts on the false theory — a reasonable attempt that should work): {fantasy_blueprint.get('false_resolution_attempt', '')}
REVERSAL DISCOVERY (THE FINAL, MOST MIND-CHANGING REVELATION — must change understanding of the threat/magic/world ITSELF, not just reveal a new name for who's responsible; also proves false_theory wrong right as/after the attempt fails or backfires — CRITICALLY, this must DEEPEN incident_2/midpoint_discovery, not contradict them: those earlier facts stay true, this adds the larger context around them): {fantasy_blueprint.get('reversal_discovery', '')}

LOW POINT (survival/success looks unlikely, threat STILL at full active danger — this happens LATE, roughly the final 10-15% of the story): {fantasy_blueprint.get('low_point', '')}

CLIMAX (the REAL confrontation, beginning only AFTER the reversal — {hero_name} must personally use the established magic mechanism, tactically DIFFERENT from the false attempt — sustained, not a quick beat — begins LATE — CRITICALLY, magic_cost_confrontation below must be paid DURING this action, not avoided, AND the climax must succeed specifically because it applies MAGIC RULE FORESHADOWED above — state the mechanical connection, not just an emotional shift): {fantasy_blueprint.get('climax', '')}
MAGIC COST CONFRONTATION (the SAME cost from magic_system_cost must be genuinely paid, in the moment, as an unavoidable part of the climax — no loopholes): {fantasy_blueprint.get('magic_cost_confrontation', '')}
PAYOFF (this specific earlier-established tool/knowledge/ally is what {hero_name} uses — NOT a new convenient power): {fantasy_blueprint.get('payoff', '')}
RESOLUTION (explicit — what happened to the threat, to world_name, and to {hero_name} — MUST include a genuine, at-least-partly-irreversible cost to {hero_name}, not a fully clean win): {fantasy_blueprint.get('resolution', '')}
FINAL IMPLICATION (the very last beat must include this specific lingering detail — it does not contradict the resolution above, but the story does NOT wrap up too neatly): {fantasy_blueprint.get('final_implication', '')}

Do not deviate from this spine's world, magic, threat, or climax mechanism. You may invent scene-level detail, dialogue, and texture freely — the STRUCTURE above is fixed.
"""
    elif genre == "romance":
        genre_guidance = """
ROMANCE REQUIREMENTS:
- Establish specific history or tension between the characters from the start — what keeps them apart?
- Build intimacy through specific shared moments that create genuine connection
- The midpoint crisis must be a real, believable threat to the relationship — not a misunderstanding one conversation would fix
- The resolution must show genuine character growth — they changed, not just reconciled
- Write dialogue that reveals what these specific characters feel — not generic romance lines
"""
    else:
        genre_guidance = f"""
STORY REQUIREMENTS:
- The central conflict must be SPECIFIC and PERSONAL to {hero_name} — tied directly to who she is
- Name all recurring side characters with actual first names and one defining personality trait
- The midpoint must significantly raise the stakes — something goes wrong or gets harder
- No filler scenes: every scene advances the plot, reveals character, or raises stakes
- The resolution must be earned through {hero_name}'s specific choices and actions
"""

    adult_note = (
        "AUDIENCE: This story is for an ADULT reader. Write mature, sophisticated content — complex motivations, real stakes, adult vocabulary and emotional nuance."
        if is_adult else
        f"AUDIENCE: Write for a {audience_age}-year-old reader — appropriate vocabulary, humor, and themes."
    )

    prompt = f"""You are a professional comic book writer. Write a complete, compelling prose story that will be adapted into a {page_count}-panel comic.

WORD COUNT REQUIREMENT: {word_count} words MINIMUM. This is not a suggestion — the panel adapter needs enough material to fill {page_count} panels. A story under {word_count} words will be too thin to adapt. Write the full story, do not cut it short.

{adult_note}

MAIN CHARACTER:
- Name: {hero_name}
{personalization_block}
{support_block}
STORY FOUNDATION:
- Theme: {theme}
- Genre: {genre} — {genre_style['story_tone']}
- Logline: {logline}
- Outline: {outline}

{genre_guidance}
CRITICAL PROSE RULES:
1. Write continuous narrative prose — NOT a list of scenes, NOT an outline, NOT bullet points
2. BE SPECIFIC: name every character who appears, describe specific objects, locations, and events — no vague placeholders
3. This prose will be given to a panel adapter who breaks it into {page_count} comic panels — every scene must be concrete and visually imaginable
4. The personalization details above MUST shape the PLOT (what happens) — not just be name-dropped in description
5. THE STORY MUST HAVE A REAL TWO-DIP ARC, NOT ONE SMOOTH RISE — this is the single most important structural rule:
   setup → a first setback/attempt → a SMALL breakthrough or partial success (not full mastery) → a SECOND, BIGGER setback or complication that raises the stakes higher, arriving AFTER the small breakthrough and CLOSE to the climax → climax → resolution.
   BANNED FLAT ARC — do not write this shape, which is the default failure mode: problem → immediate emotional support from another character → a simple solution is proposed → the character practices it → it works → celebration. That shape has no real low point and reads as rushed and generic no matter how well any individual scene is written. There must be a second, larger setback positioned AFTER hope first appears and BEFORE the climax — things must get worse again before they get better.
6. Character personality must show in HOW they act and speak — do not just state traits in narration
7. Include actual dialogue at key moments — write specific lines that capture each character's voice

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETTING LOCK — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The story takes place in ONE primary setting established in the opening scene.
Characters do NOT leave that setting until the punchline is delivered. The punchline is delivered INSIDE or AT the primary setting.

BANNED mid-story location changes:
- Characters going home after the activity ends
- Dinner parties, house scenes, or home life after the main event
- Scenes set on the internet, social media, or "days later"
- Characters walking through neighborhoods, streets, or outdoors
- "Later that week..." or "the following day..." time skips to a new location

If the outline mentions the story starts at a specific place (pottery studio, escape room, cruise ship, chili cook-off), every scene in the story happens there — not somewhere else afterward.
The instructor/judge/staff member who delivers the punchline must be present in the PRIMARY SETTING throughout the story.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LENGTH REQUIREMENT — READ THIS BEFORE WRITING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This story MUST be {word_count} words or longer. That is a hard floor, not a suggestion.

PARAGRAPH RULE — every paragraph must be at least 40 words.
If you write a paragraph that is 1–2 sentences, it is too short. Expand it before continuing.
One-sentence paragraphs are summaries. Summaries are not scenes. Write the scene.

DISASTER SCENE RULE — every time something goes wrong, write it in full:
  Step 1 — describe exactly what {hero_name} does (specific action, not "she tried something")
  Step 2 — describe exactly what goes wrong (specific and visual — what does it look like?)
  Step 3 — write at least 3 lines of back-and-forth dialogue reacting to it
  Step 4 — describe the consequence (how is the situation now worse than before?)
If your disaster scene is missing any of these four steps, you have summarized it. Go back and write it.

PACING GUIDE — write each of these as a full developed scene (multiple paragraphs, not one sentence):
- Scene 1: Arrival and setup — establish goal, introduce setting, show character personality in action
- Scene 2: First disaster — show all 4 steps above; this is the first big laugh
- Scene 3: NPC makes it worse — a named side character's specific trait compounds the problem
- Scene 4: Second disaster — measurably worse than Scene 2; show all 4 steps
- Scene 5: The fix backfires — what {hero_name} tries next fails in a new, funnier way; show all 4 steps
- Scene 6: Peak chaos — multiple earlier problems collide at once; the most chaotic moment
- Scene 7: Desperate measure — the wild, specific thing they try to escape the chaos
- Scene 8: Punchline — ironic callback to the original goal; the last thing written

SCENE DEPTH REQUIRED FOR {page_count} PANELS:
{"Each scene above must be 4–6 paragraphs minimum. A 48-panel comic needs roughly twice the material of a 24-panel comic. Do not move to the next scene until the current scene has at least 4 full paragraphs." if page_count > 24 else "Each scene above must be 2–3 paragraphs minimum. Do not summarize — write each scene in full before moving to the next."}

SELF-CHECK BEFORE SUBMITTING — DO THIS IN ORDER:
1. Count your paragraphs. {"You need at least 35 for a 48-panel comic." if page_count > 24 else "You need at least 20."} If fewer, continue to step 2.
2. Find the three shortest scenes in the story (the ones with the fewest paragraphs).
3. For each short scene: add more specific action, add at least 2 more lines of dialogue, describe what it looks and sounds like in more detail.
4. Only submit when you have {"at least 35 paragraphs" if page_count > 24 else "at least 20 paragraphs"} and every disaster scene has all 4 steps.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDING RULE — READ THIS BEFORE WRITING THE LAST SCENE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before the punchline, {hero_name} must take a real corrective ACTION that actually fixes or salvages the specific external thing the chaos damaged — deciding as a group that the disaster was secretly fine all along does NOT count; something concrete must actually get fixed.

The punchline itself must be a concrete VISUAL gag the reader can picture — a prop, an object, or a minor character's small action — not just a spoken line. Two valid shapes:
- If the goal was a measurable competition result (a time, a prize, beating a rival), a third party who already appeared earlier (staff member, judge, announcer) states the ACTUAL result as a concrete fact ("Fifty-three minutes," not "Most Chaos Created"), landing together with one last visual beat in the same panel.
- If there's no natural judge/announcer for this story, the ending IS a callback payoff: a minor/named character does something with an established prop/object/bit one last time, causing one final small comedic consequence.

The hero does NOT get a reaction line after the punchline lands. The story ends on the visual beat (and, if used, the third party's words).
WRONG: Staff says the funny thing. Then hero grins and quips. Then they walk out. ← three extra sentences
RIGHT: The visual gag happens. Full stop. Story over.

NO UNEARNED TWISTS: Do not introduce any person, place, prop, or concept in the final scene that was not established earlier in the story. Do not invent a film crew, a secret audience, or a hidden judge who was never mentioned in the setup.

Write the punchline. Then stop. Do not type another word.

Begin with the opening scene — no title, no preamble, just the story."""

    ending_rule = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDING RULE — READ THIS BEFORE WRITING THE LAST SCENE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The last thing on the page must be a concrete VISUAL beat the reader can picture happening — a prop, an object, or a minor/named character's small action — not a spoken line reflecting on how things turned out. Two valid shapes, pick whichever fits this specific story:
(a) COMPETITION SHAPE — if the goal was a measurable competition result (a time, a prize, beating a rival), a third party who already appeared earlier (staff member, judge, announcer) states the ACTUAL result as a concrete fact, AND the same final panel/moment includes a visual gag (a prop misfiring, a minor character's small action) rather than ending on the spoken line alone.
(b) CALLBACK SHAPE — if there is no natural judge/announcer for this story (a party, a trip, a family event), the ending IS the callback payoff itself: a minor/named character does something with the planted prop/object/bit one last time, causing one final small comedic consequence, with at most one short reaction line.
NO UNEARNED TWISTS: every person or prop involved in the ending must have appeared earlier in the story.
The hero does NOT get a warm reflective reaction line after the visual gag lands. Write the ending. Then stop. Do not type another word."""

    client = get_client()

    # Blueprint-driven genres (adventure, mystery) always use scene-by-scene
    # generation, even at page_count<=24 -- a 16-page book was silently falling
    # back to the single-shot path below, which never sees the 8-beat blueprint
    # at all. That's what let a fully-formed mystery climax exist in the
    # blueprint and nowhere in a 24-panel book's actual prose.
    use_scene_by_scene = (
        page_count > 24
        or (genre == "adventure" and adventure_blueprint)
        or (genre == "mystery" and mystery_blueprint)
        or (genre == "comedy" and comedy_blueprint)
        or (genre == "horror_adult" and horror_blueprint)
        or (genre == "sci_fi" and sci_fi_blueprint)
        or (genre == "horror" and horror_kids_blueprint)
        or (genre == "fantasy" and fantasy_blueprint)
    )
    # Scale each scene's word budget to the book's actual size. The page_count>24
    # numbers (500/300) are unchanged -- proven reliable all session. A 24-panel
    # book targets ~2000 words total (word_count above), so scenes need to be
    # proportionately shorter or the same 8-beat structure overflows a much
    # smaller panel budget, squeezing the climax out during panel adaptation.
    scene_word_target = 500 if page_count > 24 else 260
    scene_final_word_target = 300 if page_count > 24 else 160

    try:
        if use_scene_by_scene:
            # Scene-by-scene generation: 8 calls, word budget scaled to page_count
            # GPT-4o reliably writes a focused word count on a single-scene task
            if genre == "adventure" and adventure_blueprint:
                # Built directly from the validated blueprint so each of the 8 scene
                # calls is told exactly what happens in it — not a generic comedy-shaped
                # task description that leaves the actual mission/obstacles to chance.
                scene_definitions = [
                    (
                        "Arrival and mission",
                        f"Introduce {hero_name} and the setting, then establish the mission clearly and concretely: {adventure_blueprint.get('mission', '')} "
                        f"Show {hero_name}'s personal motivation: {adventure_blueprint.get('motivation', '')} "
                        f"State the stakes plainly: {adventure_blueprint.get('stakes', '')}",
                        False,
                    ),
                    (
                        "First obstacle",
                        f"{adventure_blueprint.get('obstacle_1', '')} This must directly threaten the mission established in the previous scene. "
                        f"Show {hero_name} making a real decision in response — the obstacle exists because of choices {hero_name} makes, not the other way around.",
                        False,
                    ),
                    (
                        "Discovery",
                        f"{adventure_blueprint.get('discovery', '')} This should change {hero_name}'s understanding of the situation or approach — build genuine curiosity, not just another complication.",
                        False,
                    ),
                    (
                        "Second obstacle",
                        f"{adventure_blueprint.get('obstacle_2', '')} This must be a DIRECT CONSEQUENCE of what the first obstacle cost {hero_name} — name that specific cost (an injury, a lost resource, lost time, a broken tool, a burned bridge) explicitly on the page, and show it being what causes this new obstacle. This is not a fresh, unrelated danger; it only exists because of how the first obstacle was overcome. Show {hero_name} making a decisive choice, not just reacting to chaos.",
                        False,
                    ),
                    (
                        "Midpoint change",
                        f"{adventure_blueprint.get('midpoint_change', '')} This must be caused by a specific cost of the second obstacle — name what it cost {hero_name} and show that cost producing this change. From here on, the mission itself is harder, more urgent, or fundamentally different than it was at the start.",
                        False,
                    ),
                    (
                        "Major setback and low point",
                        f"{adventure_blueprint.get('obstacle_3', '')} This must be a direct consequence of the midpoint change just described — not an unrelated new threat. Then: {adventure_blueprint.get('low_point', '')} "
                        f"Let {hero_name} genuinely sit in this failure for a beat — do not immediately jump to a new idea or plan. Success must look truly unlikely here.",
                        False,
                    ),
                    (
                        "Climax",
                        f"{adventure_blueprint.get('climax', '')} {hero_name} must personally resolve this using {adventure_blueprint.get('payoff', '')} — something established earlier in the story. "
                        f"No other character, coincidence, or new unexplained idea may solve it for {hero_name}. Show the climactic action actually happening in specific detail — do not just announce that it is about to start.",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{adventure_blueprint.get('resolution', '')} Show the concrete outcome of the mission from the first scene, and how {hero_name} has changed. "
                        f"This is the earned outcome of {hero_name}'s own actions at the climax — not a comedic punchline delivered by a bystander.",
                        True,
                    ),
                ]
            elif genre == "mystery" and mystery_blueprint:
                # Same rationale as the adventure branch above: built directly from
                # the validated, solution-first blueprint so the culprit, motive,
                # method, and clues are fixed from scene one instead of left to
                # chance across 8 separate calls. Testing showed a suspect built up
                # for five scenes and then never resolved, and a "resolution" that
                # never actually stated who did it, why, or how -- scene 7 now
                # explicitly requires the deduction to be shown, and scene 8
                # explicitly requires closing out the red-herring suspect, not just
                # the true culprit.
                scene_definitions = [
                    (
                        "The mystery, the victim, and the stakes",
                        f"Introduce {hero_name} and the setting, then establish the central mystery clearly and concretely: {mystery_blueprint.get('central_mystery', '')} "
                        f"Establish the victim: {mystery_blueprint.get('victim', '')} State the stakes plainly: {mystery_blueprint.get('stakes', '')}",
                        False,
                    ),
                    (
                        "Suspects and the red herring",
                        f"Introduce the suspects: {mystery_blueprint.get('suspects', '')} "
                        f"{mystery_blueprint.get('red_herring', '')} Show {hero_name} genuinely suspecting this person based on real evidence or circumstance — the reader should believe it too, not see through it immediately.",
                        False,
                    ),
                    (
                        "First clue and the companion's contribution",
                        f"{hero_name} discovers a concrete clue, shown as a scene the reader witnesses (not summarized after the fact): {mystery_blueprint.get('clue_1', '')} "
                        f"This should deepen the investigation, not immediately resolve it. Also show the companion's specific contribution to the case: {mystery_blueprint.get('companion_contribution', '')}",
                        False,
                    ),
                    (
                        "Midpoint complication",
                        f"{mystery_blueprint.get('midpoint_complication', '')} From here, the case is more dangerous, urgent, or complicated than it was at the start — the red herring theory should start to feel shakier.",
                        False,
                    ),
                    (
                        "Second clue and the key contradiction",
                        f"{hero_name} discovers another concrete clue, shown as a scene the reader witnesses (not summarized after the fact): {mystery_blueprint.get('clue_2', '')} "
                        f"Then {hero_name} notices the specific detail that doesn't fit the red herring's guilt: {mystery_blueprint.get('key_contradiction', '')} — {hero_name}'s own observation drives this, not a lucky handoff.",
                        False,
                    ),
                    (
                        "Third clue and the low point",
                        f"{hero_name} discovers a third concrete clue that points toward the TRUE culprit, shown as a scene the reader witnesses (not summarized after the fact): {mystery_blueprint.get('clue_3', '')} "
                        f"Then: {mystery_blueprint.get('low_point', '')} Let {hero_name} genuinely sit in this doubt or danger for a beat — do not immediately jump to the solution.",
                        False,
                    ),
                    (
                        "The deduction and the climax",
                        f"First, show {hero_name} actually reasoning through the case on the page — a real deductive step connecting the clues and the contradiction, not a vague feeling of certainty: {mystery_blueprint.get('final_deduction', '')} "
                        f"This deduction must be {hero_name}'s OWN reasoning, laid out BEFORE any confrontation — the reader must reach the solution alongside {hero_name}, not learn it from the culprit. "
                        f"Then {hero_name} confronts or accuses {mystery_blueprint.get('true_culprit', '')}. The accusation is NOT the end of the scene — it triggers what happens next: {mystery_blueprint.get('climax', '')} "
                        f"{hero_name} must personally resolve this using the undeniable proof: {mystery_blueprint.get('proof', '')} "
                        f"No other character, coincidence, or new unexplained idea may solve it for {hero_name}. If {mystery_blueprint.get('true_culprit', '')} admits anything, that admission comes ONLY after {hero_name}'s accusation — it corroborates what {hero_name} already worked out, it never does the revealing itself. Show the confrontation actually happening in specific, risky detail — do not just announce the reveal and stop.",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{mystery_blueprint.get('resolution', '')} Show the concrete outcome of the crime — the TRUE culprit held accountable. "
                        f"Separately and explicitly, close out EVERY OTHER named suspect from the investigation, not only the red herring — each one gets their own real innocent explanation, even if just a sentence per suspect. Do NOT let any of them simply disappear from the story. Suspects to close out: {mystery_blueprint.get('suspects', '')} "
                        f"Red herring's explanation specifically: {mystery_blueprint.get('red_herring_explanation', '')} "
                        f"Also explicitly state what actually happened to the central mystery's own hook — {mystery_blueprint.get('central_mystery', '')} — not just that the culprit was caught; a reader must learn exactly what became of the missing object, disappearance, secret, or event this story was built around. "
                        f"Show how the ORIGINAL stakes are specifically settled, and how {hero_name} has changed.",
                        True,
                    ),
                ]
            elif genre == "comedy" and comedy_blueprint:
                # Same rationale as the adventure/mystery branches above: built
                # directly from the blueprint's specific funny_moments/npc_bits
                # instead of the generic placeholders below, which let the
                # actual planned beats drift into invented chaos across 8
                # separate calls. The blueprint only defines 3 explicit beats
                # (first disaster, second disaster, peak chaos) but the proven
                # 8-scene pacing shape is kept -- the two escalation scenes
                # between beats are grounded in npc_bits and each beat's own
                # caused_by field instead of being purely generic.
                moments = comedy_blueprint.get("funny_moments", [])
                beat1 = moments[0] if len(moments) > 0 else {}
                beat2 = moments[1] if len(moments) > 1 else {}
                beat3 = moments[2] if len(moments) > 2 else {}
                npc_bits = comedy_blueprint.get("npc_bits", {})
                npc_bits_text = "; ".join(f"{name}: {bit}" for name, bit in npc_bits.items()) or "no named NPC bits defined"
                dynamics = comedy_blueprint.get("character_dynamics", {})
                dynamics_text = "; ".join(f"{name}: {desc}" for name, desc in dynamics.items()) or "no character dynamics defined"

                scene_definitions = [
                    (
                        "Arrival and original goal",
                        f"Introduce {hero_name} and the setting, then establish the specific, measurable goal clearly: {comedy_blueprint.get('original_goal', '')} "
                        f"Show what each named character is confidently wrong about, through action and dialogue, not narration: {dynamics_text}",
                        False,
                    ),
                    (
                        "First disaster",
                        f"{beat1.get('what_happens', '')} Show {hero_name}'s specific reaction: {beat1.get('character_reaction', '')} "
                        f"A named NPC escalates it: {beat1.get('npc_involvement', '')} All 4 disaster steps: exactly what {hero_name} does, exactly what goes wrong (visual and specific), at least 4 lines of back-and-forth dialogue reacting to it, how the situation is now measurably worse than before.",
                        False,
                    ),
                    (
                        "NPC bit escalates",
                        f"A named NPC's specific recurring comedic bit escalates on top of the existing problem, driven by their own personality — not generic interference: {npc_bits_text} "
                        f"This directly builds on what just happened: {beat1.get('caused_by', '')}",
                        False,
                    ),
                    (
                        "Second disaster",
                        f"{beat2.get('what_happens', '')} This must be measurably worse than the first disaster — a clear escalation, directly caused by it, not a repeat. Show {hero_name}'s specific reaction: {beat2.get('character_reaction', '')} "
                        f"A named NPC escalates it further: {beat2.get('npc_involvement', '')} All 4 disaster steps: action, visual failure, 4+ dialogue lines, consequence.",
                        False,
                    ),
                    (
                        "The fix backfires",
                        f"What {hero_name} tries to fix the situation fails in a new, funnier way, directly leading toward the peak: {beat2.get('caused_by', '')} "
                        f"The attempt must feel like it could plausibly work but spectacularly doesn't. All 4 steps.",
                        False,
                    ),
                    (
                        "Peak chaos",
                        f"{beat3.get('what_happens', '') or comedy_blueprint.get('peak_chaos_description', '')} Multiple problems from earlier in the story collide simultaneously — reference at least 2 specific things that went wrong earlier. "
                        f"{hero_name}'s most extreme reaction yet: {beat3.get('character_reaction', '')} A named NPC at their most chaotic: {beat3.get('npc_involvement', '')} "
                        f"Specific visual, what the audience SEES: {comedy_blueprint.get('peak_chaos_description', '')} The situation should look completely unrecoverable.",
                        False,
                    ),
                    (
                        "Desperate measure",
                        f"{comedy_blueprint.get('resolution_beat', '')} This must be a real corrective ACTION that actually fixes, repairs, or salvages the specific external thing the chaos damaged — not the characters simply deciding the disaster is fine or good as it is. It should feel like the last possible idea, and it can still be funny or chaotic in HOW it's done, but something concrete must actually get fixed by the end of this scene. Show {hero_name}'s specific personality in how they come up with it.",
                        False,
                    ),
                    (
                        "Punchline",
                        f"{comedy_blueprint.get('punchline_ending', '')} The final beat must be a concrete VISUAL gag the reader can picture — a prop, an object, or a minor/named character's small action — ideally the payoff of the callback planted earlier, not just a spoken line of reflection. If a third party (staff member, judge, announcer who appeared earlier) is the natural way to deliver a concrete result in this specific story, they may state it as a fact — but the LAST thing on the page should still be something visual happening, not a line of dialogue about how everyone feels. {hero_name} does NOT get a warm reaction line after the visual gag lands. {ending_rule}",
                        True,
                    ),
                ]
            elif genre == "horror_adult" and horror_blueprint:
                # Same rationale as the adventure/mystery/comedy branches above: built
                # directly from the locked, backstory-first blueprint so the entity's
                # identity stays fixed and the reveal actually stays gradual across 8
                # separate scene calls, instead of the model improvising a threat that
                # could drift, over-reveal early, or accidentally introduce a second
                # antagonist by scene 6.
                scene_definitions = [
                    (
                        "Arrival and the insecurity",
                        f"Introduce {hero_name} and the setting, then establish the central haunting as it first appears: {horror_blueprint.get('central_haunting', '')} "
                        f"Also plant {hero_name}'s specific insecurity here, naturally, through a specific memory, belief, or moment — not narration stating it outright: {horror_blueprint.get('protagonist_exploited_insecurity', '')} "
                        f"State the stakes plainly: {horror_blueprint.get('stakes', '')}",
                        False,
                    ),
                    (
                        "First incident",
                        f"{horror_blueprint.get('incident_1', '')} This must be small, ambiguous, and deniable — {hero_name} could still rationalize it away. Reveal NOTHING about the entity's identity or backstory yet.",
                        False,
                    ),
                    (
                        "Second incident and the first fragment",
                        f"{horror_blueprint.get('incident_2', '')} This incident must be stronger than the first. Also show the specific, named source revealing the first fragment of the backstory — name specific people and specific acts, never an abstract label like 'a betrayal' with no actual event stated. Do not reveal more than this one fragment; the full picture stays incomplete. This fragment must change what {hero_name} does next, not just add sad trivia.",
                        False,
                    ),
                    (
                        "Midpoint discovery",
                        f"{horror_blueprint.get('midpoint_discovery', '')} This reframes {hero_name}'s understanding of what they're dealing with, from a specific named source — with a specific new fact (name the people/acts involved, not an abstract label) — but still holds back the complete backstory for later. This must change {hero_name}'s strategy or reveal the danger is worse than believed, not just add emotional context.",
                        False,
                    ),
                    (
                        "Third incident — physical danger",
                        f"{horror_blueprint.get('incident_3', '')} {hero_name} must be in genuine physical danger here, not just unsettled — show real fear and a proportional reaction.",
                        False,
                    ),
                    (
                        "Fourth incident — the insecurity is exploited",
                        f"{horror_blueprint.get('incident_4', '')} This must explicitly use the SAME specific insecurity planted in the first scene, by name or detail — the entity has clearly been watching {hero_name}. Then: {horror_blueprint.get('low_point', '')} "
                        f"Let {hero_name} genuinely sit in this low point — success must look truly unlikely. The entity must still be at FULL, ACTIVE power here — no safety, no resolution yet.",
                        False,
                    ),
                    (
                        "The false theory and a reasonable attempt",
                        f"By now {hero_name} has pieced together what seems like the full truth: {horror_blueprint.get('false_theory', '')} This must read as genuinely plausible — the reader should believe it too, given everything revealed so far. "
                        f"Acting on this belief, {hero_name} attempts what seems like the obvious solution: {horror_blueprint.get('false_resolution_attempt', '')} Write this as a real, competent attempt that a smart person would try — not something obviously doomed. Let it seem, for a moment, like it might actually be working.",
                        False,
                    ),
                    (
                        "The reversal — we had it wrong",
                        f"{horror_blueprint.get('reversal_discovery', '')} This must land as a genuine surprise: the false_resolution_attempt has just failed or is actively backfiring, and RIGHT THEN {hero_name} learns the specific new fact that explains why — the false theory from the last scene was wrong or dangerously incomplete. "
                        f"Make the moment of realization visible on the page — {hero_name} recognizing, in real time, that everything they thought they understood needs to be rethought. This is a pivot point, not a minor complication: the danger should now feel WORSE and more urgent than before, because the plan that was supposed to work just failed.",
                        False,
                    ),
                    (
                        "Climax",
                        f"{horror_blueprint.get('climax', '')} {hero_name} must personally survive/resolve this using {horror_blueprint.get('payoff', '')} as TACTICAL LEVERAGE against the entity — something established earlier that mechanically disempowers/binds/destroys/fulfills a real condition, not merely a kind word or emotional gesture that happens to work. "
                        f"PHYSICAL ACTION REQUIRED, NOT JUST WORDS: {hero_name}'s hands must actually DO something — burn/bury/destroy/return the specific object, or complete the specific physical ritual gesture — at the moment of resolution. Simply revealing, announcing, or confessing a truth out loud, with no physical action, is NOT enough even if the words are specific and true — write the physical act happening, with any dialogue as accompaniment to it, not as the mechanism itself. "
                        f"This must be TACTICALLY DIFFERENT from the false_resolution_attempt a couple scenes ago — {hero_name} is now acting on what reversal_discovery revealed, not repeating the same failed approach harder. "
                        f"NO GRADUAL SOFTENING (read carefully — this is the most common way this goes wrong even when the mechanism itself is correct): Do NOT depict the entity's defeat as a multi-beat emotional de-escalation — no sequence where the entity goes from raging to \"finally listening,\" to calm, to grateful, to peaceful across several consecutive moments. That is the therapy-speak failure pattern even if no single banned phrase is used. Once the mechanical action ({horror_blueprint.get('payoff', '')}) completes, the entity's defeat must be SUDDEN and visibly caused by that action within the same or very next beat — a forcible negation (torn apart, extinguished, yanked away, shattered), not a gentle fade as it comes to peace. Banned lines/patterns at any point in this scene: \"you don't have to carry this anymore,\" \"rest now,\" \"let go,\" \"you're free now,\" \"your suffering is over,\" \"he's finally listening,\" \"it's over,\" \"I did what you needed,\" or any line implying the entity is moved, grateful, or cooperating emotionally. If {hero_name} speaks to the entity at all, the dialogue must reference the SPECIFIC concrete action being taken right then (naming the object being destroyed/returned, the specific fact being exposed, the specific condition being fulfilled) — the entity's defeat must read as a direct CONSEQUENCE of that action, not a reaction to being consoled or convinced. "
                        f"No other character, coincidence, or new unexplained idea may resolve it for {hero_name}. This must be a SUSTAINED confrontation — let the danger build and peak across many panels, but the actual moment of defeat, once the mechanism completes, should be fast — do not stretch the entity's dissolution across multiple softening beats.",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{horror_blueprint.get('resolution', '')} State EXPLICITLY and unambiguously what happened: the entity is defeated/banished/satisfied, OR {hero_name} escapes (entity survives), OR {hero_name} is caught/consumed — pick the one the spine describes and show it happening. Make clear WHY it happened — the mechanical effect of the payoff, not just that the entity felt understood. "
                        f"BANNED DIALOGUE PATTERN — even here, in the aftermath, do not give {hero_name} a valedictory line to the entity like \"rest now,\" \"you're finally free,\" \"rest easy,\" or \"let go\" — the entity is already gone because of the mechanical action in the climax, not because of anything said to it afterward. Keep any dialogue here focused on {hero_name} and other living characters, not a farewell speech addressed to the entity. "
                        f"Also explicitly address what became of the entity's unfinished business — {horror_blueprint.get('unfinished_business', '')} — resolved, or tragically still unresolved. Never end on an open question or vague implication about the outcome itself. "
                        f"Then, as the VERY LAST beat of the story, include this specific unsettling detail so the ending doesn't feel completely safe: {horror_blueprint.get('final_sting', '')} This does not contradict the resolution above — it is one more small, concrete, unsettling detail that lingers after.",
                        True,
                    ),
                ]
            elif genre == "sci_fi" and sci_fi_blueprint:
                # Same rationale as the horror branch above: built directly from
                # the locked, truth-first blueprint (including its baked-in
                # third-act reversal) so the technology's internal logic stays
                # fixed and the reveal actually stays gradual across 10 separate
                # scene calls, instead of the model improvising a premise that
                # could drift, over-reveal early, or resolve via a speech instead
                # of a concrete technical action.
                scene_definitions = [
                    (
                        "Arrival and the goal",
                        f"Introduce {hero_name} and the setting, then establish the central premise as it first appears: {sci_fi_blueprint.get('central_premise', '')} "
                        f"Establish {hero_name}'s specific, concrete goal: {sci_fi_blueprint.get('protagonist_goal', '')} Also plant {hero_name}'s specific blind spot here, naturally, through a specific belief or assumption — not narration stating it outright: {sci_fi_blueprint.get('protagonist_blind_spot', '')} "
                        f"State the stakes plainly: {sci_fi_blueprint.get('stakes', '')}",
                        False,
                    ),
                    (
                        "First incident",
                        f"{sci_fi_blueprint.get('incident_1', '')} This must be small and explainable away — {hero_name} could still rationalize it as a glitch or coincidence. Reveal NOTHING about hidden_truth yet.",
                        False,
                    ),
                    (
                        "Second incident and the first fragment",
                        f"{sci_fi_blueprint.get('incident_2', '')} This must be stronger than the first. Also show the specific, named source revealing the first fragment of the hidden truth — name specific people and specific facts, never an abstract label like 'a conspiracy' with no actual fact stated. PROGRESSIVE DISCOVERY: this fragment must ANSWER a small question {hero_name} had, while immediately RAISING a new, BIGGER question — end the scene with {hero_name} realizing this is stranger than they first thought, not simpler. This fragment must change what {hero_name} does next, not just add ominous atmosphere.",
                        False,
                    ),
                    (
                        "Midpoint discovery",
                        f"{sci_fi_blueprint.get('midpoint_discovery', '')} This reframes {hero_name}'s understanding, from a specific named source — with a specific new fact (name the people/facts involved, not an abstract label) — but still holds back the complete truth for later. PROGRESSIVE DISCOVERY: this must ANSWER the question raised by the previous scene, while immediately RAISING an even BIGGER one about what the technology/phenomenon actually is — the mystery must feel deeper and stranger now, not just further confirmed. This must change {hero_name}'s strategy or reveal the danger is worse than believed.",
                        False,
                    ),
                    (
                        "Third incident — the system reacts",
                        f"{sci_fi_blueprint.get('incident_3', '')} This must be a DIRECT CONSEQUENCE of the investigation reaching this point — the technology or institution reacting to being investigated — not an unrelated new danger. {hero_name} must be in genuine danger here, not just intellectually unsettled.",
                        False,
                    ),
                    (
                        "The false theory and a reasonable attempt",
                        f"By now {hero_name} has pieced together what seems like the full truth: {sci_fi_blueprint.get('false_theory', '')} This must read as genuinely plausible — the reader should believe it too, given everything revealed so far. "
                        f"Acting on this belief, {hero_name} attempts what seems like the obvious solution: {sci_fi_blueprint.get('false_resolution_attempt', '')} Write this as a real, competent attempt a smart person would try — not something obviously doomed. Let it seem, for a moment, like it might actually be working.",
                        False,
                    ),
                    (
                        "The reversal — we had it wrong",
                        f"{sci_fi_blueprint.get('reversal_discovery', '')} This must land as a genuine surprise: the false_resolution_attempt has just failed or is actively backfiring, and RIGHT THEN {hero_name} learns the specific new fact that explains why — the false theory from the last scene was wrong or dangerously incomplete. THIS IS THE STORY'S BIGGEST REVELATION: it must change {hero_name}'s (and the reader's) understanding of what the technology/phenomenon ITSELF actually is or can do — not merely reveal which human was behind it. If this scene's twist is only 'it turns out person X was responsible,' with nothing new understood about the technology itself, that is not enough — the technology/phenomenon must turn out to be stranger, more capable, or more autonomous than anyone realized. "
                        f"Make the moment of realization visible on the page. This is a pivot point, not a minor complication: the danger should now feel WORSE and more urgent than before. Then: {sci_fi_blueprint.get('low_point', '')} Let {hero_name} genuinely sit in this low point — the technology/institution must still be at FULL, ACTIVE danger here, no safety yet.",
                        False,
                    ),
                    (
                        "Climax",
                        f"{sci_fi_blueprint.get('climax', '')} {hero_name} must personally perform a CONCRETE PHYSICAL OR TECHNICAL ACTION using {sci_fi_blueprint.get('payoff', '')} as tactical leverage — cutting power, planting code, destroying hardware, rerouting a system, physically extracting/broadcasting data through a specific technical means — NOT merely a speech, a realization, or convincing someone with words alone. "
                        f"This must be TACTICALLY DIFFERENT from the false_resolution_attempt a couple scenes ago — {hero_name} is now acting on what reversal_discovery revealed, not repeating the same failed approach harder. If {hero_name} speaks at all, the dialogue must accompany the physical/technical action, not substitute for it — the resolution happens because of what {hero_name} DOES, not because of what they SAY. "
                        f"No other character, coincidence, or new unexplained idea may resolve it for {hero_name}. This must be a SUSTAINED confrontation — let the danger build and peak, but the actual moment of resolution, once the action completes, should be fast and decisive. "
                        f"CRITICAL — DO NOT LET THIS BECOME A GENERIC HACKING/CHASE SCENE: {sci_fi_blueprint.get('central_dilemma', '')} must become an active obstacle DURING the action itself, not resolved by it and not absent from it. WRONG (fails even with a valid physical/technical action): {hero_name} sneaks past guards using a stolen badge and types a shutdown command while alarms blare, with the story's central dilemma nowhere in the scene. RIGHT: as {hero_name} moves to complete the action, the technology exploits its own established capability on {hero_name} in real time — planting doubt, a false impulse, a manufactured memory or hesitation — and {hero_name} must find a concrete way to verify their own judgment/perception/agency is genuinely theirs (recalling a specific detail the technology couldn't have fabricated, or acting fast before the doubt fully takes hold) before they can trust themselves enough to finish it. Write this confrontation with the technology's own nature, not generic external obstacles, as the source of difficulty.",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{sci_fi_blueprint.get('resolution', '')} State EXPLICITLY what happened: what became of the technology (destroyed/disabled/exposed/repurposed), what became of {sci_fi_blueprint.get('who_controls_it', '')} (held accountable, escaped consequence, doubles down elsewhere), and what it cost or meant for {hero_name}. NO EASY ENDING: {hero_name} must genuinely lose or permanently give up something specific here, even in success — do not let them walk away fully vindicated, promoted, and completely fine with nothing lost. Never end on a vague 'things will be different now.' "
                        f"Then, as the VERY LAST beat of the story, include this specific lingering detail so the ending doesn't feel too neatly wrapped up: {sci_fi_blueprint.get('final_implication', '')} This does not contradict the resolution above — it is one more small, concrete, thought-provoking detail that lingers after.",
                        True,
                    ),
                ]
            elif genre == "horror" and horror_kids_blueprint:
                # Same rationale as the horror_adult branch: built directly from the
                # locked, creature-first blueprint so the creature's identity stays
                # fixed and the reveal stays gradual across separate scene calls.
                # CRITICAL: "kid-friendly" here means non-violent CONTENT, not a soft/
                # cute TONE — incidents 1-4 must read as genuinely suspenseful and
                # ambiguous, the creature's harmlessness a secret withheld from the
                # reader until the reversal, not telegraphed via giggling/cute framing.
                scene_definitions = [
                    (
                        "Arrival and the worry",
                        f"Introduce {hero_name} and the setting, then establish the central mystery as it first appears: {horror_kids_blueprint.get('central_mystery', '')} "
                        f"Also plant {hero_name}'s specific worry here, naturally, through a specific detail or moment — not narration stating it outright: {horror_kids_blueprint.get('protagonist_worry', '')} "
                        f"State the stakes plainly: {horror_kids_blueprint.get('stakes', '')}",
                        False,
                    ),
                    (
                        "First incident",
                        f"{horror_kids_blueprint.get('incident_1', '')} This must be small and easily explained away by an adult, but written to feel genuinely UNSETTLING and ambiguous — a real prickle of unease, not a punchline. {hero_name} could still rationalize it, but it lingers uncomfortably. Reveal NOTHING about the creature's true nature yet, and give no hint that the presence is friendly — no giggling, no cute cartoon sound effects, no obviously harmless framing. Write it the way a real ghost story would.",
                        False,
                    ),
                    (
                        "Second incident and the first fragment",
                        f"{horror_kids_blueprint.get('incident_2', '')} This incident must be more unsettling than the first — genuine tension, still ambiguous about whether it's dangerous. Also show the specific, named source revealing the first fragment of the truth. Do not reveal more than this one fragment, and do not let the fragment accidentally confirm the creature is friendly this early. This fragment must give {hero_name} something new to do next, not just be interesting trivia.",
                        False,
                    ),
                    (
                        "Midpoint discovery",
                        f"{horror_kids_blueprint.get('midpoint_discovery', '')} This reframes {hero_name}'s understanding of what's going on, from a specific named source — but still holds back the complete truth, and should still leave real doubt about whether the presence is dangerous. This must change {hero_name}'s plan or make the mystery feel bigger/stranger, not just add flavor.",
                        False,
                    ),
                    (
                        "Third incident — the mystery deepens",
                        f"{horror_kids_blueprint.get('incident_3', '')} This must be a DIRECT reaction to the investigation getting closer, and should be the MOST unsettling incident yet — genuinely tense and suspenseful, {hero_name} and the reader both genuinely worried something is wrong. The content stays non-violent, but nothing about how this is written should signal that the presence is harmless.",
                        False,
                    ),
                    (
                        "Fourth incident — the worry is touched",
                        f"{horror_kids_blueprint.get('incident_4', '')} This must touch the SAME specific worry planted in the first scene, by name or detail, in a way that feels genuinely unnerving and personal in the moment — real fear, not a jump-scare played for laughs. Then: {horror_kids_blueprint.get('low_point', '')} "
                        f"Let {hero_name} genuinely feel frightened and discouraged here — this is the point of maximum dread in the story. The mystery must still feel unsolved and the stakes still at risk. No physical danger occurs, but the feeling of danger should be real and unresolved — do not undercut it with reassurance yet.",
                        False,
                    ),
                    (
                        "The false theory and a reasonable attempt",
                        f"By now {hero_name} has pieced together what seems like the full truth: {horror_kids_blueprint.get('false_theory', '')} This must read as genuinely plausible — the reader should believe it too, given everything revealed so far. "
                        f"Acting on this belief, {hero_name} and friends attempt what seems like the obvious solution: {horror_kids_blueprint.get('false_resolution_attempt', '')} Write this as a real, clever attempt smart kids would try — not something obviously doomed. Let it seem, for a moment, like it might actually be working.",
                        False,
                    ),
                    (
                        "The reversal — we had it wrong",
                        f"{horror_kids_blueprint.get('reversal_discovery', '')} This must land as a genuine surprise: the false_resolution_attempt has just failed or backfired in a harmless-but-embarrassing way, and RIGHT THEN {hero_name} learns the specific new fact that explains why — the creature isn't mean at all, it's {horror_kids_blueprint.get('what_it_wants', '')} "
                        f"Make the moment of realization visible on the page — {hero_name} recognizing they jumped to the wrong conclusion, and feeling a little bad about it. This is a genuine pivot point, not a minor complication. "
                        f"CRITICAL — DO NOT LET THE STORY TURN SAFE HERE: learning the truth does NOT mean the danger/tension is over. End this scene with something still unresolved, unstable, or frightening (the creature still trapped/fading, a sense that time is running out, the atmosphere still oppressive) — there is a genuine final challenge still ahead, not just a nice conversation left to have.",
                        False,
                    ),
                    (
                        "Climax",
                        f"{horror_kids_blueprint.get('climax', '')} {hero_name} must personally make things right using {horror_kids_blueprint.get('payoff', '')} as the concrete detail — something established earlier that {hero_name} now uses to help the creature. "
                        f"THIS SCENE MUST STAY GENUINELY TENSE UNTIL THE ACTION SUCCEEDS — do not let it open calm or cozy just because the truth is known. Build real suspense (will it work? is there enough time? does the atmosphere still feel unsettling?) that only resolves once {hero_name} completes the action. "
                        f"FEAR CONFRONTATION REQUIRED: {horror_kids_blueprint.get('fear_confrontation', '')} {hero_name} must actually DO the specific thing they're afraid of, or face it head-on, as an active, in-the-moment part of completing this action — show this through what {hero_name} does, never through narration simply stating they aren't scared anymore. "
                        f"CONCRETE ACTION REQUIRED, NOT JUST WORDS: {hero_name} must actually DO something specific and tangible (return an object, fix/build something, complete a specific task or surprise) — telling the creature it's welcome or not alone, with nothing actually done, is NOT enough even if the words are kind. Write the concrete action happening, with any dialogue as accompaniment to it. "
                        f"This must be TACTICALLY DIFFERENT from the false_resolution_attempt a couple scenes ago — {hero_name} is now helping, not trying to get rid of the creature. "
                        f"No other character, coincidence, or new unexplained idea may resolve it for {hero_name}. Keep the CONTENT non-violent throughout, but the mood should stay suspenseful right up until the action lands — the warmth belongs to the scene AFTER this one, not this one. "
                        f"WRITE THIS AS A DRAMATIZED SCENE, NOT A SUMMARIZED MONTAGE (read carefully — this is the most common way an otherwise-correct climax still falls flat): include at least one discrete, decisive beat — a specific action at a specific instant, a specific line of dialogue, a single sharp turning point the reader can picture happening at one precise moment. Do NOT compress this into several paragraphs of general activity summary (\"gathered the pieces... worked on it for a while... each careful stitch was a testament to resilience...\") — that reads as backstory summary, not a climax, no matter how much fear or tension is mentioned throughout it. Stage this moment-to-moment: what does {hero_name} do right now, in this second, and what happens right after because of it?",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{horror_kids_blueprint.get('resolution', '')} State EXPLICITLY and warmly what happened: the creature is happy/at peace, and {horror_kids_blueprint.get('unfinished_business', '')} is resolved. Make clear WHY it happened — the specific kind action, not just nice words. "
                        f"Then, as the VERY LAST beat of the story, include this specific warm/fun detail: {horror_kids_blueprint.get('final_beat', '')} This should feel completely safe, cozy, and fun — the OPPOSITE of a scary ending — with nothing spooky left lingering.",
                        True,
                    ),
                ]
            elif genre == "fantasy" and fantasy_blueprint:
                # Same rationale as the sci_fi branch: built directly from the locked,
                # world/magic-first blueprint so the magic system's rules and cost stay
                # fixed and the reveal stays gradual across separate scene calls.
                scene_definitions = [
                    (
                        "Arrival and the goal",
                        f"Introduce {hero_name}, the world, and the central premise as it first appears: {fantasy_blueprint.get('central_premise', '')} "
                        f"State {hero_name}'s concrete goal plainly: {fantasy_blueprint.get('protagonist_goal', '')} Establish {fantasy_blueprint.get('world_name', '')} with specific, lived-in detail — not generic fantasy-kingdom filler.",
                        False,
                    ),
                    (
                        "First incident",
                        f"{fantasy_blueprint.get('incident_1', '')} This must be small and explainable away — {hero_name} could still rationalize it. Reveal NOTHING of the hidden truth yet.",
                        False,
                    ),
                    (
                        "Second incident and the first fragment",
                        f"{fantasy_blueprint.get('incident_2', '')} This incident must be stronger than the first. Also show the specific, named source revealing the first fragment of the truth. PROGRESSIVE DISCOVERY: this must ANSWER a small question while immediately RAISING a bigger one — the mystery must feel deeper now, not just further confirmed. This must give {hero_name} something new to do next. "
                        f"IF this is where magic_rule_foreshadowed belongs, plant it concretely and specifically here: {fantasy_blueprint.get('magic_rule_foreshadowed', '')} A careful reader should be able to remember this rule and recognize it again when the climax uses it.",
                        False,
                    ),
                    (
                        "Midpoint discovery",
                        f"{fantasy_blueprint.get('midpoint_discovery', '')} This reframes {hero_name}'s understanding, from a specific named source — with a specific new fact — but still holds back the complete truth for later. PROGRESSIVE DISCOVERY: this must ANSWER the question raised by the previous scene, while immediately RAISING an even BIGGER one. This must change {hero_name}'s strategy or reveal the danger is worse than believed. This new fact must DEEPEN incident_2's fact, not replace it — what was established earlier stays true; this adds context on top of it. "
                        f"IF this is where magic_rule_foreshadowed belongs instead of the previous scene, plant it concretely and specifically here: {fantasy_blueprint.get('magic_rule_foreshadowed', '')}",
                        False,
                    ),
                    (
                        "Third incident — the threat reacts",
                        f"{fantasy_blueprint.get('incident_3', '')} This must be a DIRECT CONSEQUENCE of the quest reaching this point — the threat reacting to being pursued — not an unrelated new danger. {hero_name} must be in genuine danger here, not just intellectually unsettled.",
                        False,
                    ),
                    (
                        "The false theory and a reasonable attempt",
                        f"By now {hero_name} has pieced together what seems like the full truth: {fantasy_blueprint.get('false_theory', '')} This must read as genuinely plausible — the reader should believe it too, given everything revealed so far. "
                        f"Acting on this belief, {hero_name} attempts what seems like the obvious solution: {fantasy_blueprint.get('false_resolution_attempt', '')} Write this as a real, competent attempt a smart person would try. Let it seem, for a moment, like it might actually be working.",
                        False,
                    ),
                    (
                        "The reversal — we had it wrong",
                        f"{fantasy_blueprint.get('reversal_discovery', '')} This must land as a genuine surprise: the false_resolution_attempt has just failed or is actively backfiring, and RIGHT THEN {hero_name} learns the specific new fact that explains why. THIS IS THE STORY'S BIGGEST REVELATION: it must change {hero_name}'s (and the reader's) understanding of what the threat/magic/world ITSELF actually is — not merely reveal a new name for who's responsible. "
                        f"CRITICAL — DEEPEN, DON'T CONTRADICT: everything revealed in the second incident and midpoint discovery scenes must remain TRUE — do not write this as 'actually, everything you thought was wrong.' Instead, show this revelation adding the larger context, hidden cost, or misunderstood relationship that those earlier facts were always compatible with — the reader should be able to look back and see it fit all along, not feel like the rug was pulled out with no continuity. "
                        f"Make the moment of realization visible on the page. Then: {fantasy_blueprint.get('low_point', '')} Let {hero_name} genuinely sit in this low point — the threat must still be at FULL, ACTIVE danger here, no safety yet.",
                        False,
                    ),
                    (
                        "Climax",
                        f"{fantasy_blueprint.get('climax', '')} {hero_name} must personally use {fantasy_blueprint.get('magic_system_name', '')} via {fantasy_blueprint.get('payoff', '')} as the specific established mechanism — NOT a new, unforeshadowed power that appears only now. "
                        f"THE SOLUTION MUST BE DEDUCIBLE, NOT ASSERTED: the reason this works and false_resolution_attempt didn't must be the SPECIFIC rule planted earlier — {fantasy_blueprint.get('magic_rule_foreshadowed', '')} — explicitly state how {hero_name} applies that exact rule here. Do NOT explain the difference only as {hero_name} 'understanding things better now' or 'growing as a person' with no mechanical link back to that planted rule — a reader who remembered the earlier clue should recognize exactly why this works. "
                        f"THE COST MUST BE PAID, NOT DODGED: {fantasy_blueprint.get('magic_cost_confrontation', '')} {hero_name} must genuinely incur this cost, in the moment, as an unavoidable part of winning — no convenient loophole or exception that lets them use the magic for free this one time. The loss must be real and visible on the page. "
                        f"This must be TACTICALLY DIFFERENT from the false_resolution_attempt a couple scenes ago. No other character, coincidence, or new unexplained idea may resolve it for {hero_name}. "
                        f"WRITE THIS AS A DRAMATIZED SCENE, NOT A SUMMARIZED MONTAGE: include at least one discrete, decisive beat — a specific action at a specific instant, a specific line of dialogue, a single sharp turning point the reader can picture happening at one precise moment. Do not compress this into paragraphs of general activity summary — stage it moment-to-moment.",
                        False,
                    ),
                    (
                        "Resolution",
                        f"{fantasy_blueprint.get('resolution', '')} State EXPLICITLY what happened: what became of the threat (defeated/banished/transformed/escaped), what became of {fantasy_blueprint.get('world_name', '')}, and what it cost or meant for {hero_name} — including the lasting effect of the cost just paid. NO EASY ENDING: {hero_name} must genuinely lose or permanently give up something specific here, even in success. "
                        f"Then, as the VERY LAST beat of the story, include this specific lingering detail so the ending doesn't feel too neatly wrapped up: {fantasy_blueprint.get('final_implication', '')} This does not contradict the resolution above — it is one more small, concrete, thought-provoking detail that lingers after.",
                        True,
                    ),
                ]
            else:
                scene_definitions = [
                    (
                        "Arrival and setup",
                        f"Introduce {hero_name}, the setting, and the specific goal they are trying to achieve. Show each character's personality through what they DO and SAY — not description. Plant the seeds of what will go wrong.",
                        False,
                    ),
                    (
                        "First disaster",
                        f"Something goes hilariously wrong. Write all 4 steps: (1) exactly what {hero_name} does, (2) exactly what goes wrong — be visual and specific, (3) at least 4 lines of back-and-forth dialogue reacting to it, (4) how the situation is now measurably worse than before.",
                        False,
                    ),
                    (
                        "NPC makes it worse",
                        "A named side character's specific personality trait causes them to 'help' in a way that creates a new problem on top of the existing one. Their contribution must be driven by their specific personality — not generic interference.",
                        False,
                    ),
                    (
                        "Second disaster",
                        f"Something goes wrong that is measurably worse than the first disaster — a clear escalation, not a repeat. All 4 steps: action, visual failure, 4+ dialogue lines, consequence.",
                        False,
                    ),
                    (
                        "The fix backfires",
                        f"What {hero_name} tries to fix the situation fails in a new, funnier way — making things worse. All 4 steps. The attempt must feel like it could plausibly work but spectacularly doesn't.",
                        False,
                    ),
                    (
                        "Peak chaos",
                        "Multiple problems from earlier in the story collide simultaneously. This is the funniest, most chaotic moment. Reference at least 3 specific things that went wrong earlier. The situation should look completely unrecoverable.",
                        False,
                    ),
                    (
                        "Desperate measure",
                        f"The wild, specific thing they try to escape the chaos — and it must be a real corrective ACTION that actually fixes or salvages the specific external thing that broke, not the characters simply deciding the disaster is fine. It should feel like the last possible idea, and it can still be funny in HOW it's done, but something concrete must actually get fixed. Show {hero_name}'s specific personality in how they come up with it.",
                        False,
                    ),
                    (
                        "Punchline",
                        f"The ironic resolution of the original goal, ending on a concrete VISUAL gag the reader can picture — a prop, an object, or a minor character's small action — not just a spoken line. If a third party (staff member, judge, announcer who appeared earlier) fits this story, they may state the specific outcome as a concrete fact, but the final beat on the page should still be something visual happening. {hero_name} does NOT get a warm reaction line after it. {ending_rule}",
                        True,
                    ),
                ]

            story_so_far = []
            sys_msg = "You are a creative writer. Write vivid, specific, character-driven prose. No scene labels or headers in your output."

            # Build compact causal-chain reminder for comedy scenes 2-8
            causal_reminder = ""
            callback_plant_block = ""
            callback_payoff_block = ""
            if comedy_blueprint and genre == "comedy":
                moments = comedy_blueprint.get("funny_moments", [])
                chain_lines = []
                for m in moments:
                    cb = m.get("caused_by", "")
                    if cb and m.get("beat", 0) < len(moments):
                        chain_lines.append(f"  Beat {m['beat']} directly causes: {cb}")
                original_goal = comedy_blueprint.get("original_goal", "")
                punchline = comedy_blueprint.get("punchline_ending", "")
                causal_reminder = (
                    f"\nSTORY ARCHITECTURE — the causal chain you must follow:\n"
                    f"Original goal: {original_goal}\n"
                    + ("\n".join(chain_lines) + "\n" if chain_lines else "")
                    + f"Punchline ending: {punchline}\n"
                )
                cb = comedy_blueprint.get("callback", {})
                if cb.get("plant") and cb.get("payoff"):
                    callback_plant_block = (
                        f"\nCALLBACK PLANT — weave this naturally into this scene as a throwaway moment:\n"
                        f"  {cb['plant_speaker']}: \"{cb['plant']}\"\n"
                        f"It must feel like a casual remark, not a setup. Do not draw attention to it.\n"
                    )
                    callback_payoff_block = (
                        f"\nCALLBACK PAYOFF — somewhere in this scene, echo the earlier plant with a twist:\n"
                        f"  Original plant ({cb['plant_speaker']}): \"{cb['plant']}\"\n"
                        f"  Payoff ({cb['payoff_speaker']}): \"{cb['payoff']}\"\n"
                        f"Use the same words or object, but the situation now makes it funny. "
                        f"Write it naturally — the payoff lands because the reader recognizes it, not because you announce it.\n"
                    )

            for scene_idx, (scene_name, scene_task, is_final) in enumerate(scene_definitions):
                context_block = ""
                if story_so_far:
                    context_block = f"\n\nSTORY SO FAR — continue directly from this, do not repeat it:\n{''.join(story_so_far)}\n"

                if scene_idx == 0:
                    # First scene gets full character and story setup
                    scene_prompt = f"""You are a professional comic book writer. Write the opening scene of a {page_count}-panel {genre} comic story.

{adult_note}

MAIN CHARACTER:
- Name: {hero_name}
{personalization_block}
{support_block}
STORY FOUNDATION:
- Theme: {theme}
- Genre: {genre} — {genre_style['story_tone']}
- Logline: {logline}
- Outline: {outline}

{genre_guidance}

YOUR TASK — write ONLY this one scene: {scene_name}
{scene_task}
{callback_plant_block}
REQUIREMENTS:
- Approximately {scene_word_target} words (aim for {"5–6 paragraphs, each at least 3 sentences" if scene_word_target > 350 else "3–4 paragraphs, each at least 2 sentences"})
- Write continuous narrative prose — NO scene labels, NO headers
- Begin with the opening line of the story — no title, no preamble
- STOP after this scene. Do not write what happens next."""
                else:
                    target_words = scene_final_word_target if is_final else scene_word_target
                    # Inject callback plant in scene 2 (index 1) as a fallback,
                    # payoff in scenes 7-8 (indices 6-7)
                    extra_callback = ""
                    if scene_idx == 1 and callback_plant_block:
                        extra_callback = callback_plant_block
                    elif scene_idx >= 6 and callback_payoff_block:
                        extra_callback = callback_payoff_block
                    scene_prompt = f"""Continue a {genre} story about {hero_name}.
{context_block}{causal_reminder}{extra_callback}
{genre_guidance}
YOUR TASK — write ONLY this one scene: {scene_name}
{scene_task}

REQUIREMENTS:
- Approximately {target_words} words ({"2–3 paragraphs" if is_final else "5–6 paragraphs, each at least 3 sentences"})
- Write continuous narrative prose — NO scene labels, NO headers
- CAUSAL CONTINUITY: Your opening sentence must name a direct consequence of what just happened in the previous scene — this scene exists BECAUSE of what happened before it, not alongside it. The reader must feel "of course THAT led to THIS."
- Stay in the same primary setting — do not move characters to a new location
- STOP after this scene. Do not write what happens next."""

                r = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": scene_prompt},
                    ],
                    temperature=0.85,
                    max_tokens=1200 if not is_final else 1000,
                    timeout=60,
                )
                scene_text = r.choices[0].message.content.strip()
                story_so_far.append("\n\n" + scene_text)
                print(f"PROSE scene {scene_idx + 1} ({scene_name}): ~{len(scene_text.split())} words")

            prose = "\n\n".join(s.strip() for s in story_so_far)
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative writer specializing in personalized comic book stories. Write vivid, specific, character-driven stories with clear escalating arcs.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.85,
                max_tokens=4000,
                timeout=120,
            )
            prose = response.choices[0].message.content.strip()

        print(f"PROSE STORY generated ({len(prose)} chars, ~{len(prose.split())} words): {prose[:200]}...")
        return prose

    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI connection failed: {repr(e)}")
    except AuthenticationError as e:
        raise RuntimeError(f"OpenAI authentication failed: {repr(e)}")
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI quota/rate limit failed: {repr(e)}")
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI status error {e.status_code}: {repr(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI error: {type(e).__name__}: {e}")


def expand_prose_if_short_ai(
    prose_story: str,
    target_word_count: int,
    hero_name: str,
    max_passes: int = 2,
) -> str:
    """
    If the prose is significantly shorter than target, run up to max_passes expansion passes.
    Models write short from scratch but expand existing content reliably.
    Loops until prose reaches 75% of target or max_passes exhausted.
    """
    threshold = int(target_word_count * 0.75)
    current = prose_story

    for pass_num in range(1, max_passes + 1):
        current_words = len(current.split())
        if current_words >= threshold:
            print(f"📏 Prose at {current_words} words (threshold {threshold}) — expansion done after {pass_num - 1} pass(es)")
            return current

        words_needed = target_word_count - current_words
        print(f"📏 Prose expansion pass {pass_num}: {current_words} words → targeting {target_word_count}...")

        prompt = f"""A prose story draft is too short and needs to be expanded. It must reach at least {target_word_count} words before being adapted into a comic book. It is currently {current_words} words — approximately {words_needed} more words are needed.

CURRENT STORY:
{current}

EXPANSION RULES — follow every one of these:

1. DO NOT change what happens — same events, same characters, same order, same ending
2. DO NOT change the final paragraph — the last paragraph must remain word-for-word identical
3. DO NOT add new plot points, new characters, or a new ending scene
4. DO NOT add any epilogue or scenes set in a new location (home, restaurant, gallery, internet, streets) after the main action ends
5. DO NOT send characters anywhere new — the story stays in its primary setting until the punchline

WHAT TO EXPAND:

DISASTER SCENES — every scene where something goes wrong must have all 4 steps:
  Step 1: Exactly what {hero_name} does (specific action, not a summary)
  Step 2: Exactly what goes wrong (specific and visual — describe what it looks like)
  Step 3: At least 4 lines of dialogue between characters reacting to it
  Step 4: The consequence — describe how the situation is now measurably worse than before

DIALOGUE — wherever there is action without dialogue, add 2–4 lines of characters reacting in their specific voices. Do not use generic lines like "we did it!" or "oh no!" — make them specific to what just happened.

SENSORY DETAIL — for each scene, add: what does it sound like? smell like? what does {hero_name}'s face look like in this moment? what does the space look like now compared to before?

THIN SCENES — find the 3 shortest scenes (fewest paragraphs) and at least double them in length by adding specific action, dialogue, and consequence.

Return the complete expanded story from beginning to end. Nothing before or after — just the story."""

        client = get_client()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are expanding a prose story draft. Return only the expanded story — no commentary, no preamble, no explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=6500,
                timeout=120,
            )
            if response.choices[0].finish_reason == "length":
                print(f"⚠️  Expansion pass {pass_num} was truncated (hit max_tokens) — keeping prose from before this pass")
                return current
            expanded = _strip_meta_preamble(response.choices[0].message.content.strip())
            expanded_words = len(expanded.split())
            print(f"✏️  Expansion pass {pass_num}: {current_words} → {expanded_words} words")
            if expanded_words > current_words:
                current = expanded
            else:
                print(f"⚠️  Pass {pass_num} did not increase word count — stopping expansion")
                return current
        except Exception as e:
            print(f"⚠️  Prose expansion pass {pass_num} failed — using current prose: {e}")
            return current

    print(f"📏 Expansion complete after {max_passes} pass(es): {len(current.split())} words")
    return current


def generate_story_character_bible_ai(
    prose_story: str,
    hero_name: str,
    support_char_names: list = None,
) -> list:
    """
    Read the finished prose story and produce a locked visual description for every
    named NPC (characters who are not the hero or named supporting characters).
    Returns a list of dicts: [{"name": "...", "description": "..."}, ...]
    """
    known = [hero_name] + (support_char_names or [])
    known_str = ", ".join(known)

    prompt = f"""You are a comic book art director preparing for production.

Read the story below. Identify every NAMED character who is NOT in this list: [{known_str}].

For each NPC you find, define a LOCKED visual appearance that the image artist will use consistently in every panel.

Story:
{prose_story}

Rules:
- Only include characters with an actual first name used in the story. A first name is a real given name like "Marjorie" or "Finn" — NOT a role, title, or description standing in for a name, such as "the waiter," "a guest," "the vendor," "the patchwork-scarf representative," or "the elderly woman." A plural family surname used to refer to a whole family or group ("the Rosenfields," "the Langstons") is ALSO not a character name — if an individual member of that family speaks or acts, they need their own real given name to be included; do not list the family surname itself as if it were one person. If the story never gives an individual character a real given name, leave them out entirely, even if they speak or are described in detail.
- Only include NPCs who are RECURRING or CENTRAL to the story — they must appear in at least two distinct scenes, OR play a decisive role in one significant scene (a real plot turn hinges on them, not just a line of background dialogue). Do NOT include a character who is named once in passing and never appears again — they don't need a locked visual profile, and giving them one wastes a reference portrait that the finished book will never actually use.
- Do NOT include {known_str}
- Each description MUST cover ALL of the following — missing any is not acceptable:
    1. Race/ethnicity (e.g. "Black man", "South Asian woman", "white man") — this is the most important detail
    2. Age range (e.g. "late 20s", "mid-40s")
    3. Hair: exact color, texture, length, and style (e.g. "dark brown locs past his shoulders", "short curly black hair", "bleached-blonde buzzcut")
    4. Skin tone: specific descriptor (e.g. "dark brown skin", "medium olive skin", "pale freckled skin")
    5. Clothing: every garment with color (e.g. "white button-up shirt, brown plaid trousers, rainbow-striped socks, black suspenders")
    6. One physically distinctive detail that makes them unmistakable (e.g. "always has his arms crossed", "wears oversized glasses")
- The clothing and physical traits you define are LOCKED — they must appear identically in every panel
- Be specific enough that two different artists produce the SAME PERSON — vague descriptions like "dark hair" or "casual clothes" are NOT acceptable
- Also give a best-guess numeric age and gender for each NPC — these are used to render a standalone reference portrait, so they must be your best estimate even if the story doesn't state them explicitly

Return ONLY valid JSON:
{{
  "npcs": [
    {{
      "name": "character name",
      "description": "complete locked visual description covering all 6 required details — 3 sentences max",
      "age": 45,
      "gender": "man"
    }}
  ]
}}

If no named NPCs exist outside the known list, return {{"npcs": []}}."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=45,
        )
        data = json.loads(response.choices[0].message.content)
        npcs = data.get("npcs", [])
        print(f"NPC bible: {[n['name'] for n in npcs]}")
        return npcs
    except Exception as e:
        print(f"⚠️  NPC bible generation failed: {e}")
        return []


def generate_comedy_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the specific comedy architecture before writing the prose.
    Returns a structured blueprint with concrete funny moments, causal chain, and punchline.
    Only called for comedy genre.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    prompt = f"""You are a professional comedy writer planning a {page_count}-panel comic book before a single word of prose is written.

Your job: design a SPECIFIC, FUNNY story structure. Every element must be concrete — no vague placeholders.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

COMEDY ARCHITECTURE RULES:
1. The comedy MUST emerge from {hero_name}'s specific personality — not generic bad luck
2. Each funny moment must directly CAUSE the next (escalating causal chain — not disconnected incidents)
2b. ESCALATION MEANS A DIFFERENT KIND OF PROBLEM, NOT A BIGGER DOSE OF THE SAME ONE. Beat 2 and Beat 3 must each introduce a problem from a DIFFERENT category than the beat before it — not the same mechanism simply turned up in volume or repeated on a new target. Before writing each beat, name its category and make sure it differs from the previous beat's category. Categories to choose from (pick different ones for beat 1 / beat 2 / beat 3): a physical mishap with an object or substance; a social/reputational complication (someone important is watching, a rumor spreads, a rival shows up); a logistics/equipment failure (something breaks, runs out, or goes missing); a miscommunication between two characters each acting on wrong information; an authority figure or outsider arriving and changing the stakes.
   WRONG (repetition, not escalation): Beat 1 — the sauce is too hot and diners cough. Beat 2 — a second, hotter batch of sauce makes MORE diners cough harder. Beat 3 — a third, even hotter batch causes a food fight. This is the same mechanism (hot sauce reaction) three times with the dial turned up — it will read as repetitive no matter how extreme the sauce gets.
   RIGHT (true escalation): Beat 1 — the sauce is too hot and diners cough (physical mishap). Beat 2 — a food blogger with 500K followers is in line filming the coughing fit, and Ceiara, trying to spin it, promises something even hotter is coming next (social/reputational complication — the stakes are now public, not just a bad batch). Beat 3 — Marcus, trying to deliver on that promise, grabs what he thinks is the reserve hot sauce but it's actually last week's batch he forgot to label, gone strangely fermented and foul-tasting (logistics/equipment failure colliding with the social pressure from beat 2 — still an ordinary, if gross, kitchen mistake).
   FOOD-SAFETY HARD LIMIT (applies to every beat, not just this one): if the story involves food or drink served to anyone, every "wrong ingredient" mix-up must stay within ordinary-kitchen-mistake territory — spoiled, expired, bland, over-seasoned, the wrong (but still edible) item, or a genuine cooking blunder. NEVER have anyone serve or consume a cleaning product, chemical, inedible substance, or anything else actually toxic or hazardous, even as an accident — that reads as real poisoning, not comedy, and the story cannot play it for laughs with no one reacting as if someone needs real medical help. If a mix-up beat is about to involve something non-food, keep it away from anyone's mouth entirely (it spills on equipment, ruins a prop, triggers an alarm) rather than having a character eat or drink it.
3. Named NPCs must each have a specific recurring comedic bit that appears in multiple scenes
4. The punchline ending must CALLBACK to something specific from the setup
5. After the peak chaos, resolution must be BRIEF (2-3 beats max) — no warm epilogue
6. The final beat must be a LAUGH, not "they felt good about what they accomplished"
7. Design ONE deliberate plant-and-payoff callback — a line of dialogue or small visual detail in scenes 1-2 that seems like a throwaway moment but returns with a twist near the end. This is what makes readers feel the story was cleverly written.
8. THE RESOLUTION MUST BE AN ACTUAL FIX, NOT A CHANGE OF ATTITUDE: {hero_name} (or the group) must take a specific corrective action that repairs, salvages, or makes right the actual external damage/problem the chaos caused. It is NOT enough for characters to simply decide the disaster was secretly good all along — that resolves nothing and, if the original_goal was to AVOID or PREVENT a specific bad outcome, it can actually undercut the character arc by rewarding the exact failure they were trying to avoid.
   WRONG: {hero_name} destroys the cake, breaks the chandelier, and releases the doves trying to prove they aren't bad luck — then the ending has everyone decide "let's just turn this into a dance party!" and celebrate the disaster as if it were the plan all along. Nothing got fixed; {hero_name}'s original failure is just relabeled as a win.
   RIGHT: {hero_name} still does something concrete that repairs or salvages the actual thing that broke (recovers/replaces the cake, corrals the doves before the ceremony, improvises a fix that lets the actual event proceed) — the celebration or lightened mood comes AFTER and BECAUSE OF that real fix, not as a replacement for it.
9. THE PUNCHLINE MUST BE A VISUAL GAG, NOT JUST A SUMMARY LINE: punchline_ending should be (or directly incorporate) the callback's payoff from item 7 — a concrete visual moment involving a prop, an object, or a minor/named character's small action — not only spoken dialogue reflecting on what happened. The strongest comedy endings give the reader one last thing to SEE, not just hear.
   RIGHT EXAMPLE OF THE SHAPE: after the fix lands and the emotional beat closes, a minor named character quietly reaches for [the exact prop from the callback plant]; {hero_name} sees it and reacts ("[NAME]. DON'T."); [the prop] goes off anyway; final panel shows the visual aftermath; the minor character delivers one last dry line. That visual beat — not a spoken moral — is the actual last thing the story shows.
10. IF THE STORY INVOLVES GOING VIRAL, FAME, OR SOCIAL MEDIA SUCCESS: that realization is stated or shown EXACTLY ONCE, at the actual resolution — never set up to recur as its own beat. Do not design beat 2, beat 3, peak_chaos, resolution_beat, AND punchline_ending as separate opportunities for characters to re-announce "this is going viral" / "we're famous now" / "look at these views" in escalating hype. Each of those fields must contain its own distinct content (see rule 2b) — treat "we succeeded/went viral" as the single payoff of the whole story, not a chorus repeated after every beat.

CALLBACK DESIGN RULES:
- The plant must feel NATURAL — not set up like a setup, just something a character says or does that the reader half-notices
- The payoff must use the EXACT SAME words, object, or action — but the context makes it funny now
- The payoff works best when it's the OTHER character saying the original speaker's words back at them, or when the situation is now the ironic opposite
- WRONG callback: "Ceiara mentions she's good under pressure" → "Ceiara handles a tough moment well" — too vague, no twist
- RIGHT callback: Vincent says "Whatever you do, don't touch that" → Near the end, Vincent himself reaches for the exact same thing → Ceiara says "You were saying?" — specific, reversed, funny

CHARACTER DYNAMICS (the most important comedy rule):
- Define what EACH named character is CONFIDENTLY WRONG about in this specific story
- The funniest dynamic: multiple characters each certain they know best, each making the others' situation worse
- A supporting character who only watches nervously is a wasted character — give them their own confident mistake
- WRONG: "Vincent watches nervously as Ceiara makes mistakes"
- RIGHT: "Vincent secretly watched three pottery tutorials last night and believes he is now an expert — he keeps stepping in to 'correct' Ceiara using completely wrong techniques, confidently making everything worse"
- EVERY named character in the story must be confidently wrong about something specific

ORIGINAL GOAL LOCK:
- Identify the one SPECIFIC, MEASURABLE thing {hero_name} is trying to CREATE, WIN, or ACHIEVE at the start
- The goal must be concrete enough that you can tell objectively whether they succeeded or failed
- WRONG goal: "dominate the escape room" / "have a great time" / "show everyone what she's made of" — too vague, no punchline possible
- RIGHT goal: "be the first team to escape in under 45 minutes" / "win the blue ribbon at the chili cook-off" / "make a vase good enough to give as a gift" — concrete, measurable, punchlinable
- The punchline ending MUST directly reference and ironically resolve this original goal
- The goal cannot disappear halfway through and be replaced by unrelated chaos
- WRONG: story starts with "make a vase" → ends with "we had a great day"
- RIGHT: story starts with "make a vase" → ends with {hero_name} triumphantly presenting what is clearly a deformed blob as "exactly what I envisioned"

NPC THREADING (named NPCs must recur, not cameo):
- Every named NPC must appear at least 3 times across the story
- Their comedic bit must ESCALATE — each appearance funnier than the last
- Only name an NPC if they will recur as a running thread — do not name one-scene characters

WRONG: "Things go wrong at the chili cook-off"
RIGHT: "Ceiara, refusing to admit the beans smell off, adds triple the spice to compensate — turning the chili nuclear-hot"

Return ONLY valid JSON:
{{
  "comedy_premise": "One sentence: what makes this funny and why it is specific to {hero_name}'s personality — not generic",
  "character_dynamics": {{
    "character name": "what they are confidently wrong about and exactly how that manifests in their behavior throughout the story"
  }},
  "original_goal": "what {hero_name} is trying to create/win/achieve at the start — AND how the punchline ending ironically connects back to this exact goal",
  "funny_moments": [
    {{
      "beat": 1,
      "what_happens": "Exactly what goes wrong — specific and visual, not vague",
      "caused_by": "How this directly causes Beat 2",
      "character_reaction": "How {hero_name}'s specific personality makes this worse or funnier",
      "npc_involvement": "What a named NPC does here that escalates the chaos"
    }},
    {{
      "beat": 2,
      "what_happens": "What goes wrong next — directly caused by Beat 1",
      "caused_by": "How this directly causes Beat 3",
      "character_reaction": "How {hero_name}'s personality makes this worse",
      "npc_involvement": "NPC behavior that amplifies the chaos"
    }},
    {{
      "beat": 3,
      "what_happens": "Peak chaos — multiple earlier problems collide simultaneously",
      "caused_by": "This is the peak — resolution begins after this",
      "character_reaction": "The most extreme reaction yet from {hero_name}",
      "npc_involvement": "NPC at their most chaotic or funniest moment"
    }}
  ],
  "peak_chaos_description": "Specific visual description of the worst moment — what does the audience SEE?",
  "resolution_beat": "The one specific corrective ACTION {hero_name} takes that actually fixes/repairs/salvages the real external problem the chaos caused — not just a change of attitude toward the disaster. Can still be funny or ironic, but something concrete must actually get fixed.",
  "punchline_ending": "The FINAL panel moment — this should BE or directly incorporate the callback's payoff (see the 'callback' field below): a concrete VISUAL gag involving a prop, object, or a minor/named character's small action, not only a spoken line reflecting on the goal. Must NOT be generic warm dialogue or a summary of the lesson learned.",
  "npc_bits": {{
    "NPC First Name": "Their specific recurring comedic bit — and how it escalates across 3+ appearances"
  }},
  "callback": {{
    "plant": "The exact line of dialogue or small visual detail to drop in scene 1 or 2 — write it as it should appear in the story, word for word. Must feel like a throwaway moment.",
    "plant_speaker": "Which character says or does this",
    "payoff": "How the exact same words, object, or action returns near the end with a twist — write it as it should appear in the story. Must use the same specific element, now made funny by context.",
    "payoff_speaker": "Which character delivers the payoff line (ideally the OTHER character from the plant)"
  }}
}}"""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional comedy writer. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=1500,
            timeout=60,
        )
        blueprint = json.loads(response.choices[0].message.content)
        print(f"COMEDY BLUEPRINT premise: {blueprint.get('comedy_premise', '')[:120]}")
        return blueprint
    except Exception as e:
        print(f"⚠️  Comedy blueprint generation failed: {e}")
        return {}


def validate_comedy_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the comedy blueprint against structural criteria BEFORE any prose
    is written, and revises the specific fields that fail. Mirrors
    validate_adventure_blueprint_ai / validate_mystery_blueprint_ai — catching
    a vague premise, a spectator character, or a callback that isn't actually
    reversible is far cheaper and more reliable here, on a ~9-field JSON, than
    after 2000+ words of prose have already been built around it.
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a comedy editor reviewing a COMEDY STORY BLUEPRINT for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

BLUEPRINT TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole blueprint before judging:
1. SPECIFIC PREMISE: Does comedy_premise tie the humor directly to {hero_name}'s specific personality — not generic bad luck that could happen to anyone?
2. CAUSAL ESCALATION: Does each funny_moments beat's "caused_by" field directly cause the next beat — not three disconnected incidents that just happen to occur in the same story?
3. NO SPECTATOR CHARACTERS: Does character_dynamics give EVERY named character (hero included) something they are specifically, confidently WRONG about that drives their own behavior — not a character who just watches nervously or reacts to others?
4. NPC BITS ARE SPECIFIC AND ESCALATING: For each entry in npc_bits, is the bit a specific, concrete recurring behavior (not a vague trait like "is clumsy") that can plausibly escalate across 3+ appearances?
5. CONCRETE, MEASURABLE GOAL: Is original_goal specific enough that a reader could objectively tell whether {hero_name} succeeded or failed — not vague ("have a great time," "show what she's made of")?
6. GOAL-PUNCHLINE CONNECTION: Does original_goal explicitly state how punchline_ending ironically resolves THAT SAME goal — not a different, unrelated outcome?
7. CALLBACK IS SPECIFIC AND REVERSED: In the callback field, is "plant" a specific, concrete line or action (not a vague trait statement), and does "payoff" reuse the EXACT SAME words/object/action with the situation now making it ironic or funny — not just a thematically similar moment? A callback where the payoff doesn't reuse the plant's specific words/object is not a real callback: REWRITE the payoff (or both) so they specifically mirror each other.
8. BRIEF RESOLUTION: Is resolution_beat short and decisive (2-3 beats worth of content), not a warm, drawn-out wrap-up?
9. PUNCHLINE IS A LAUGH, NOT WARMTH: Does punchline_ending land as an ironic laugh — not "they felt good about what they accomplished" or other sentimental warmth?
10. RESOLUTION IS AN ACTUAL FIX, NOT A REFRAME: Does resolution_beat describe {hero_name} taking a specific corrective ACTION that repairs, salvages, or makes right the real external damage/problem the chaos caused — or does it just have everyone collectively decide the disaster was secretly fine/good/fun all along, with nothing actually fixed? A reframe-only resolution is a hard failure, especially if original_goal was to avoid/prevent the very outcome that happened — REWRITE resolution_beat to add the concrete fix; a lightened mood or celebration may follow the fix, but cannot replace it.
11. PUNCHLINE IS A VISUAL GAG, NOT ONLY A LINE: Does punchline_ending describe (or directly incorporate) the callback's payoff as a concrete VISUAL moment — a prop, an object, or a minor/named character's small action that the reader can picture — or is it only spoken dialogue/narration reflecting on the goal with nothing to actually SEE? REWRITE punchline_ending to fuse it with the callback payoff into one concrete final visual beat if it's currently just a summary line.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. Do not invent a different premise; strengthen this one.

Return ONLY valid JSON, the exact same shape as the input (comedy_premise, character_dynamics, original_goal, funny_moments, peak_chaos_description, resolution_beat, punchline_ending, npc_bits, callback)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a comedy editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1800,
            timeout=60,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Comedy blueprint validation response was truncated — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"COMEDY BLUEPRINT VALIDATED — premise: {revised.get('comedy_premise', '')[:120]}")
        return revised
    except Exception as e:
        print(f"⚠️  Comedy blueprint validation failed — using original: {e}")
        return blueprint


def generate_adventure_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the adventure story's structural spine before writing any prose — same
    architecture as generate_comedy_blueprint_ai. Repeated testing showed that
    fixing a flat, goal-less adventure after full prose was already written
    (via review/revise passes) wasn't reliable — the model needs the mission,
    obstacles, and climax mechanism locked in BEFORE it starts writing scenes.
    Only called for the adventure genre.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    prompt = f"""You are a professional adventure story architect planning a {page_count}-panel comic book before a single word of prose is written.

Adventure means: someone pursues a MEANINGFUL, SPECIFIC goal into uncertain territory and must overcome ESCALATING obstacles to reach it. It does NOT mean "exciting things happen" or "arrive somewhere and encounter chaos."

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

Design a spine with these exact fields:
- mission: A concrete, specific thing {hero_name} must accomplish — a named destination, a named object to find/recover, or a specific challenge to complete. A reader must be able to state this mission in one sentence by the end of the setup. NOT vague ("have an adventure," "find opportunity").
- motivation: Why {hero_name} personally cares about this mission — tie it to their personality/traits above.
- destination_or_unknown: The place or unknown situation the mission requires entering.
- stakes: What is specifically lost if {hero_name} fails — be concrete, not "it would be sad."
- obstacle_1: The first thing that blocks progress toward the mission. Must directly threaten the mission itself, not just be chaos happening nearby.
- discovery: New information, place, person, or object that changes {hero_name}'s understanding or approach. If the mission object is hidden somewhere specific within destination_or_unknown (a cave, a vault, a shrine), name that specific sub-location HERE — obstacle_3 must arrive at a place this discovery already pointed to, not a brand-new location invented for the first time at the moment of arrival.
- obstacle_2: NOT a fresh, unrelated obstacle — it must be caused by a specific COST of overcoming obstacle_1. State explicitly what obstacle_1 cost {hero_name} (a resource spent, an injury, a broken tool, lost time, a burned bridge, a damaged relationship) and how THAT SPECIFIC cost is what creates obstacle_2. A reader should be able to say "obstacle_2 only exists because of how obstacle_1 was overcome" — not "obstacle_2 is just a harder version of the same kind of problem."
- midpoint_change: How overcoming obstacle_2 — and what IT cost — makes the mission harder, more urgent, or fundamentally different partway through. This is the next domino: name the specific consequence of obstacle_2 that produces this change.
- obstacle_3: The major setback — {hero_name}'s current plan or approach fails here. Must be a direct consequence of midpoint_change (or the cost of obstacle_2 if midpoint_change doesn't fully cause it) — not an unrelated new threat introduced from nowhere. This is where {hero_name} reaches the destination, finds the mission object, or solves the central mystery — that arrival/discovery belongs HERE, not at the climax (see rule 4).
- low_point: The moment success looks genuinely unlikely.
- climax: NOT the moment the mission object is found, the destination is reached, or the mystery is solved — that already happened in obstacle_3. The climax is the harder, more dangerous test that finding/reaching/solving it immediately triggers (think Indiana Jones: getting the Ark doesn't end the story — the Nazis take it, and the real climax is what happens at the ceremony). Define: what goes wrong the INSTANT {hero_name} succeeds at obstacle_3? This must be a direct consequence of what obstacle_3 cost or exposed — not a new, disconnected threat. What difficult choice or action must {hero_name} take right then that is what actually resolves the STAKES from the setup — not the acquisition itself?
- payoff: Name ONE specific skill, item, piece of information, or relationship established earlier (in obstacle_1, discovery, or obstacle_2) that {hero_name} uses to survive/win the climax as just defined — not to obtain the mission object (that already happened in obstacle_3).
- resolution: How the ORIGINAL stakes from the setup — not just the mission — are shown being settled (the specific thing that was going to be lost is now safe, or specifically lost), and how {hero_name} is different at the end.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
1. AGENCY TEST: If {hero_name} could stand back and do nothing and the plot would unfold the same way, the spine fails. At obstacle_1, obstacle_2, and obstacle_3, {hero_name} must make a decision that changes what happens next — {hero_name} drives events, not the reverse.
2. RELEVANCE TEST: Every obstacle must directly block or threaten the MISSION specifically — never a complication that merely happens to occur nearby. The connection between each obstacle and the mission must be obvious, not incidental.
3. EARNED-CLIMAX TEST: The climax must be resolved through {hero_name}'s own action using the "payoff" element. It must NOT be solved by another character arriving with a fix, a lucky coincidence, or a sudden idea with no earlier setup.
4. ACQUISITION-IS-NOT-THE-CLIMAX TEST: Reaching the destination, finding the mission object, or solving the central mystery can NEVER itself be the climax — it can only trigger the climax. If your first draft of "climax" is just "{hero_name} finds/reaches/solves X," that belongs in obstacle_3 instead, and you must design a genuine climax: a harder, final test that succeeding at obstacle_3 immediately creates, resolved through a difficult choice or action that directly settles the STAKES — not a puzzle, not a search, a CHOICE with real cost.
5. CONSEQUENCE CHAIN TEST (read this one carefully — it is the most commonly violated rule): obstacle_1 → obstacle_2 → midpoint_change → obstacle_3 → climax must form ONE escalating chain of cause and effect, like dominoes, not a series of independent episodes that happen to occur in the same story. Ask, for each link: "could this obstacle be swapped for a completely different, unrelated obstacle without changing anything before or after it?" If yes, the chain is broken — redesign it so each obstacle is caused by a specific, stated COST of the previous one (an injury, a lost resource, a spent advantage, a burned bridge, a broken tool, lost time). Example of a REAL chain: a storm destroys the trail (obstacle_1) → the forced detour exposes the party to a landslide that costs their supplies (obstacle_2, caused by obstacle_1's detour) → without supplies, a river crossing goes wrong and someone is injured (midpoint_change, caused by obstacle_2's lost supplies) → the injury forces a change in who leads/navigates, and that change is what's tested at the climax (obstacle_3/climax, caused by midpoint_change's injury). Example of a BROKEN chain (do not do this): storm washes out the trail → a bear appears → they run out of water → a rival team shows up — four unrelated dangers with no domino effect between them, which forces the story to invent more and more disconnected threats just to fill space instead of one chain that naturally sustains a long story.

Return ONLY valid JSON:
{{
  "mission": "...",
  "motivation": "...",
  "destination_or_unknown": "...",
  "stakes": "...",
  "obstacle_1": "...",
  "discovery": "...",
  "obstacle_2": "...",
  "midpoint_change": "...",
  "obstacle_3": "...",
  "low_point": "...",
  "climax": "...",
  "payoff": "...",
  "resolution": "..."
}}"""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional adventure story architect. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1200,
            timeout=60,
        )
        blueprint = json.loads(response.choices[0].message.content)
        print(f"ADVENTURE BLUEPRINT mission: {blueprint.get('mission', '')[:120]}")
        return blueprint
    except Exception as e:
        print(f"⚠️  Adventure blueprint generation failed: {e}")
        return {}


def validate_adventure_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the adventure blueprint against structural criteria BEFORE any
    prose is written, and revises the specific fields that fail. Testing
    showed that adding the blueprint alone didn't guarantee it was actually
    strong — catching a vague mission or a climax that doesn't use its own
    payoff is far cheaper and more reliable to fix here, on a ~13-field
    JSON, than after 2000+ words of prose have already been built around it.
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing an ADVENTURE STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
1. CONCRETE MISSION: Could a reader state the mission in one sentence? Is it a specific destination, object, or challenge — not vague ("have an adventure," "find opportunity," "unite the town")?
2. PERSONAL MOTIVATION: Does {hero_name} have a real, specific personal reason to care — not just "because it's the plot"?
3. SPECIFIC STAKES: Is exactly what's lost on failure concrete — not "it would be sad" or left unstated?
4. OBSTACLES BLOCK THE MISSION: Do obstacle_1, obstacle_2, and obstacle_3 each directly threaten the MISSION specifically — not chaos that just happens to occur nearby?
5. OBSTACLES ESCALATE: Does each obstacle remove an advantage or raise the difficulty compared to the last — not just repeat "more chaos happens"?
5b. CONSEQUENCE CHAIN, NOT INDEPENDENT EPISODES (read carefully — this is the most common failure): Is obstacle_2 explicitly caused by a stated COST of overcoming obstacle_1 (an injury, a lost resource, a spent advantage, a burned bridge, lost time) — or is it just a fresh, unrelated obstacle that happens to be harder? Is midpoint_change caused by a stated cost of obstacle_2? Is obstacle_3 caused by midpoint_change (or obstacle_2's cost)? Test each link: "could this obstacle be swapped for a completely different, unrelated one without changing anything before or after it?" If yes for any link, the chain is broken: REWRITE obstacle_2, midpoint_change, and/or obstacle_3 so each one is a domino caused by the specific, stated cost of the one before it — not a parallel, independent threat. A spine with three unrelated dangers (a storm, then a bear, then a rival team) fails this even if each one individually threatens the mission.
6. PROTAGONIST AGENCY: Does {hero_name}'s own decision drive what happens at each obstacle — would the obstacle/plan still exist even if {hero_name} did nothing?
7. ADDITIONAL CHARACTER CONTRIBUTES MATERIALLY (if a companion is part of the story): does the payoff or an obstacle depend on something only the companion provides — not just encouragement from the sidelines?
8. REAL LOW POINT: Does low_point genuinely make success look unlikely — not a mild inconvenience that a new idea immediately follows?
9. ACQUISITION IS NOT THE CLIMAX: Read the climax field. Is it simply "{hero_name} finds the mission object / reaches the destination / solves the mystery"? That is NOT a climax — it's what obstacle_3 is for. A real climax is a HARDER test that succeeding at obstacle_3 immediately triggers (getting the Ark doesn't end the story — what happens at the ceremony does), resolved through a difficult choice or action, not a search or a puzzle. If climax is just the acquisition/arrival/solution with nothing harder required afterward: REWRITE climax to add the real final test (moving the acquisition itself into obstacle_3 if it isn't there already), and update payoff/resolution so they still connect to the new climax.
10. CLIMAX HARDER THAN PRIOR OBSTACLES: Is the climax the hardest moment in the story — not smaller or easier than obstacle_3?
11. EARNED CLIMAX: Is the climax resolved using the SAME payoff element established earlier — not a new idea, a coincidence, or another character's rescue?
12. MISSION AND STAKES ACTUALLY RESOLVED: Does resolution show the SPECIFIC mission from item 1 succeeding or failing — not a vague, different outcome — AND does it explicitly show the ORIGINAL stakes from item 3 being settled (the specific thing that was going to be lost is now safe, or is specifically lost)? A resolution that wraps up the mission but never circles back to the specific conflict established in the setup (the accusation, the threat, the thing everyone was upset about) is incomplete, even if it feels warm and conclusive.
13. GENRE-OVERLAP TEST (check this one last, and holistically — a spine can pass every item above and still fail this one): A technically well-formed mission/obstacle/climax structure can still not actually BE Adventure. Ask, in order:
    a. If every line of humor or lighthearted dialogue were stripped out, would the plot still primarily read as an adventure — or would it read as a community drama, a workplace comedy, or a slice-of-life story that merely takes place somewhere new?
    b. Does {hero_name} physically or situationally venture beyond their familiar environment into genuinely UNCERTAIN territory — or do they stay in one social, safe, public space (a festival, a town square, a shop) the entire time, with the "unknown" being just an unfamiliar town rather than real uncertainty?
    c. Does exploration or discovery materially change HOW the mission gets accomplished — or is "discovery" just background flavor that doesn't affect any decision {hero_name} makes?
    d. Could this exact plot be retitled "[setting] drama" or "[setting] comedy" without changing a single plot beat? If yes, it isn't Adventure — it's another genre wearing an adventure_blueprint.
    If the answer to (a) or (d) suggests this reads as another genre, or (b) and (c) are both weak: REJECT this spine's core premise and REBUILD it — not just reword it. A rebuilt spine needs a mission that requires physically or situationally leaving the safe/familiar (traveling somewhere specific and unfamiliar, entering a place {hero_name} doesn't understand, pursuing something whose location or nature is genuinely unknown at the start) and a discovery that changes the plan, not just the mood. Do not settle for a technically-present "destination" field that's just the story's starting location relabeled.

    CRITICAL — WHOLE-SPINE CONSISTENCY: A rebuilt mission is worthless if the rest of the spine still describes the OLD premise. After rebuilding mission/destination_or_unknown/discovery, you MUST also rewrite obstacle_2, midpoint_change, obstacle_3, low_point, climax, payoff, and resolution so every one of them is directly about PURSUING AND COMPLETING THE NEW MISSION — not left over from whatever plot the spine had before. A common failure mode: the mission becomes "retrieve an artifact in the mountains" but obstacle_2 onward still describe a town festival/feud with no mountain, no artifact, and no journey — the mission and the back half of the spine never actually meet, so the finished story ends up stating the mission was accomplished in one throwaway line instead of ever depicting it. Before returning your answer, trace the new mission through obstacle_1 → discovery → obstacle_2 → midpoint_change → obstacle_3 → low_point → climax → payoff → resolution and confirm EVERY one of those fields is a direct step toward or away from that SAME mission. If the climax could happen exactly as written even if the rebuilt mission never existed, the rebuild failed — rewrite the remaining fields until it doesn't.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. Do not invent a different premise; strengthen this one. The exception is item 13: if the genre-overlap test fails, rebuild mission, destination_or_unknown, discovery, and any other fields needed to make this a genuine adventure premise, even though that touches more of the spine.

Return ONLY valid JSON, the exact same shape as the input (all 13 fields: mission, motivation, destination_or_unknown, stakes, obstacle_1, discovery, obstacle_2, midpoint_change, obstacle_3, low_point, climax, payoff, resolution)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
            timeout=60,
        )
        revised = json.loads(response.choices[0].message.content)
        print(f"ADVENTURE BLUEPRINT VALIDATED — mission: {revised.get('mission', '')[:120]}")
        return revised
    except Exception as e:
        print(f"⚠️  Adventure blueprint validation failed — using original: {e}")
        return blueprint


def generate_mystery_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the mystery story's structural spine before writing any prose — same
    architecture as generate_adventure_blueprint_ai. Testing surfaced a bug
    the original single-pass version couldn't catch: a suspect (Evelyn) built
    up across five scenes with real evidence and then never mentioned again,
    and — more seriously — a "resolution" that never actually stated who the
    victim was, who did it, why, or how, even though the story confidently
    declared the case closed. Root cause: generating all fields in one pass
    let the model improvise suspects and clues before it had committed to a
    fixed answer, so nothing forced the investigation to actually prove
    anything. Fixed with a TWO-STEP generation: the true solution (victim,
    culprit, motive, method, full chronology) is decided and LOCKED first, in
    its own call, before a single suspect or clue exists. The second call
    then designs the investigation — suspects, clues, the red herring, the
    confrontation — as a case built to prove that fixed answer, the way an
    actual mystery is plotted backward from its solution. Only called for the
    mystery genre.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    client = get_client()

    # ── Step 1: lock the TRUE SOLUTION before inventing a single clue or suspect ──
    solution_prompt = f"""You are a professional mystery writer. Before designing any suspects, clues, or investigation, you must first decide EXACTLY what really happened — the private answer key the rest of the case will be built to prove.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

Decide these fields now. Be concrete and specific everywhere — no vague placeholders:
- central_mystery: The crime or incident as it first appears to everyone in the story — what happened, roughly when, and the apparent circumstances. This is what {hero_name} is investigating.
- victim: Who was harmed, and specifically how (injured, killed, robbed of something specific) — name and concrete detail, not "someone got hurt."
- true_culprit: The actual person responsible — name and their role/relationship in the story. This is now LOCKED and must not change later.
- true_motive: Their specific, concrete real reason — not "they were evil" or "they were jealous."
- method: EXACTLY how they did it — the specific mechanics of the crime, step by step. You must be able to answer "how" in concrete physical detail.
- true_solution: The FULL chronological account of what actually happened, from the culprit's actions through the crime itself to its aftermath — 3-5 sentences. This is the answer key everything else must be built to prove; it is not shown to the reader directly, but nothing later in the case may contradict it.
- stakes: What is specifically lost if {hero_name} fails to solve this now — concrete (an innocent person blamed, a threat that continues, something never recovered), not "it would be upsetting."

Return ONLY valid JSON:
{{"central_mystery": "...", "victim": "...", "true_culprit": "...", "true_motive": "...", "method": "...", "true_solution": "...", "stakes": "..."}}"""

    try:
        solution_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional mystery story architect. Return only valid JSON."},
                {"role": "user", "content": solution_prompt},
            ],
            temperature=0.8,
            max_tokens=800,
            timeout=60,
        )
        solution = json.loads(solution_response.choices[0].message.content)
        print(f"MYSTERY SOLUTION (locked first) — culprit: {solution.get('true_culprit', '')[:80]}, method: {solution.get('method', '')[:80]}")
    except Exception as e:
        print(f"⚠️  Mystery solution generation failed: {e}")
        return {}

    # ── Step 2: design the investigation as a case built to PROVE the locked
    # solution above — not invented independently of it ──
    investigation_prompt = f"""You are a professional mystery writer planning the INVESTIGATION for a {page_count}-panel comic book. The true solution is already decided and LOCKED — your job is to design a fair-play case that leads a careful reader to this exact answer, no other.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

THE LOCKED SOLUTION (do not change or contradict any part of this):
Central mystery (what it looks like at first): {solution.get('central_mystery', '')}
Victim: {solution.get('victim', '')}
TRUE culprit: {solution.get('true_culprit', '')}
TRUE motive: {solution.get('true_motive', '')}
Method (how they actually did it): {solution.get('method', '')}
Full chronology of what actually happened: {solution.get('true_solution', '')}
Stakes: {solution.get('stakes', '')}

Design the investigation with these exact fields, every one of them built to be CONSISTENT WITH and ultimately PROVE the locked solution above:
- suspects: {"EXACTLY 2" if page_count <= 24 else "2-3"} OTHER named characters (not the true culprit) who each have real, specific means, motive, AND opportunity to plausibly have done it — the pool of people the reader could reasonably suspect instead.{" A 16-page/24-panel book does not have room to properly resolve more than 2 named suspects on top of everything else the climax needs to show — do not generate a third." if page_count <= 24 else ""} For EACH suspect, ALSO include their own innocent_explanation — the specific, concrete real reason they didn't do it (a real alibi, a real fact that clears them), not just "they're innocent." Every suspect gets one of these, not only the red herring — a suspect with no locked innocent explanation cannot be safely resolved later without contradicting the case.
- red_herring: Which ONE of those suspects {hero_name} (and the reader) will genuinely suspect first, and the SPECIFIC evidence or circumstance that makes them look guilty. Must be compelling, not an obviously-innocent strawman. (Their innocent_explanation is already defined above, in "suspects" — reuse it here, do not invent a second one.)
- red_herring_explanation: Restate the SAME innocent_explanation you gave the red_herring suspect in "suspects" above — this MUST be delivered to the reader by the end of the story; a suspect raised and never resolved is a structural failure.
- clue_1: The first concrete clue {hero_name} discovers — a physical object, timeline detail, or contradiction. State what it actually PROVES (which suspect it points toward, and whether that's a correct or misleading conclusion) and its PROVENANCE: who made or left it, roughly when, why it exists, and why {hero_name} finds it in this specific place.
- companion_contribution: The SPECIFIC information or action that only {hero_name}'s companion can provide — something the case cannot move forward without. Not just encouragement from the sidelines.
- midpoint_complication: How the case becomes more dangerous, urgent, or complicated partway through. If this involves the true_culprit or a suspect confronting {hero_name} with knowledge of the investigation, state HOW they learned about it — a named source, something they witnessed, someone who told them. A character must never "just know."
- clue_2: A harder, later clue. State what it PROVES and its PROVENANCE (same requirements as clue_1). This clue should start to strain the red_herring theory.
- key_contradiction: The SPECIFIC detail that does NOT fit the red_herring's guilt — something about their alibi, the timeline, or the method that could not have been them. This is the pivot that redirects suspicion toward the TRUE culprit.
- clue_3: The clue that most directly implicates the TRUE culprit. State what it PROVES and its PROVENANCE (same requirements).
- low_point: The moment the case looks unsolvable, or {hero_name} is genuinely doubted, endangered, or out of leads.
- final_deduction: The SPECIFIC reasoning {hero_name} uses to combine clue_1, clue_2, clue_3, and key_contradiction into the true solution. This must be a real logical step ("X plus Y means it can't have been [red herring] because Z, which means it has to be [true culprit]") — not "she just knew." This deduction must be {hero_name}'s own — never something the culprit confesses first or a coincidence hands them.
- proof: The ONE specific piece of evidence (from clue_1, clue_2, or clue_3) that becomes the undeniable, checkmate detail at the confrontation — the fact the true culprit cannot explain away.
- climax: The confrontation the accusation triggers. This is NOT simply "{hero_name} announces who did it" — naming the culprit only TRIGGERS the climax. What does the true culprit DO the instant they're accused (flee, attack, try to destroy evidence, deny it until confronted with "proof")? What must {hero_name} personally do, at real risk, to make the truth undeniable? If the culprit eventually admits anything, that admission comes only AFTER {hero_name} has already laid out final_deduction and the accusation — it corroborates, it does not reveal.
- resolution: How the crime's outcome is shown (true culprit held accountable, using "proof"), how the red_herring suspect is EXPLICITLY cleared using their red_herring_explanation (do not just drop them), how the ORIGINAL stakes are specifically settled, how {hero_name} is different at the end, AND — separately — what specifically happened to/with the central_mystery's own hook (the missing object recovered or lost for good, the disappearance explained, the secret identity revealed) so that specific question is answered, not just the whodunit.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
1. FAIR-PLAY TEST: Every fact the reveal depends on (clue_1, clue_2, clue_3, key_contradiction, proof) must be discoverable by the reader during the investigation — never introduced for the first time at the climax.
2. RED HERRING TEST: red_herring must have real, specific means/motive/opportunity that would convince a reasonable reader.
3. AGENCY TEST: {hero_name}'s own observations and decisions must drive clue_1, midpoint_complication, clue_2, key_contradiction, and clue_3 — the solution must not fall into their lap via someone else's unprompted confession.
4. REVEAL-IS-NOT-THE-CLIMAX TEST: naming the culprit can only TRIGGER the climax, never BE it.
5. CLUE PROVENANCE TEST: for clue_1, clue_2, and clue_3, you must state who made/left it, when, why it exists, and why it's found where it is.
6. ANTAGONIST AWARENESS TEST: any confrontation involving knowledge of the investigation must state HOW that knowledge was obtained.
7. SUSPECT CLOSURE TEST: the red_herring suspect (and any other named suspect who becomes a real focus of suspicion) MUST be explicitly addressed in resolution using red_herring_explanation — cleared, or otherwise accounted for. A suspect built up with real evidence and then silently dropped is a hard failure, not a minor gap.
8. EVERY CLUE MUST HAVE A STATED PURPOSE: clue_1, clue_2, and clue_3 must each explicitly state what they PROVE — a clue that exists just to exist, with no stated bearing on who's guilty, fails this test.
9. ALL-SUSPECTS-RESOLVED TEST: every entry in "suspects" — not just the red_herring — must have its own specific innocent_explanation. In resolution, explicitly account for EVERY named suspect, not only the red_herring: even a one-sentence closure ("Ella's key was never anything but a lost duplicate she'd forgotten to return") is enough, but it must be there. No suspect the case names should be left with no stated outcome.
10. PROTAGONIST-ONLY SOLVE, CONFESSION-AFTER-THE-FACT: final_deduction must be {hero_name}'s OWN reasoning connecting clue_1, clue_2, clue_3, and key_contradiction — never a solution that arrives via the culprit's unprompted confession, a coincidence, or another character simply explaining it to {hero_name}. If climax or resolution includes the culprit admitting anything, that confession must come AFTER final_deduction/the accusation — it may corroborate a solution {hero_name} already reached, it may never BE how {hero_name} (or the reader) first learns who did it.
11. CENTRAL MYSTERY MUST GET A CONCRETE PAYOFF: central_mystery names a specific hook — a missing object, hidden treasure, disappearance, secret identity, or unexplained event. resolution must explicitly state what actually happened to/with THAT SPECIFIC hook (where the missing object ended up, what the disappearance turned out to be, what the secret identity actually was) — not just that the culprit was caught and the stakes were settled in the abstract. Exception: you may deliberately leave ONE small SECONDARY mystery unresolved as a final hook, but the CENTRAL mystery from central_mystery must always be fully answered.

Return ONLY valid JSON:
{{"suspects": "...", "red_herring": "...", "red_herring_explanation": "...", "clue_1": "...", "companion_contribution": "...", "midpoint_complication": "...", "clue_2": "...", "key_contradiction": "...", "clue_3": "...", "low_point": "...", "final_deduction": "...", "proof": "...", "climax": "...", "resolution": "..."}}"""

    try:
        investigation_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional mystery story architect designing a case to prove an already-fixed solution. Return only valid JSON."},
                {"role": "user", "content": investigation_prompt},
            ],
            temperature=0.8,
            max_tokens=2200,
            timeout=60,
        )
        investigation = json.loads(investigation_response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Mystery investigation generation failed: {e}")
        return {}

    blueprint = {**solution, **investigation}
    print(f"MYSTERY BLUEPRINT — central_mystery: {blueprint.get('central_mystery', '')[:100]}")
    return blueprint


def validate_mystery_blueprint_ai(
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> dict:
    """
    Checks the mystery blueprint against fair-play structural criteria BEFORE
    any prose is written, and revises the specific fields that fail. Mirrors
    validate_adventure_blueprint_ai's architecture and rationale. Updated for
    the two-step, solution-first schema (see generate_mystery_blueprint_ai) —
    the checklist now also verifies every suspect gets an explicit resolution
    and that the central mystery's factual questions (who died, how, who did
    it, why) are actually answerable from the spine, since testing found a
    built-up suspect silently dropped and a "resolution" that never stated
    who did it, why, or how, despite the story declaring itself solved.
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing a MYSTERY STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
1. CONCRETE CENTRAL MYSTERY: Could a reader state exactly what happened, as it first appears, in one sentence?
2. VICTIM IS SPECIFIC: Does "victim" name who was harmed and concretely how — not "someone got hurt"?
3. TRUE CULPRIT HAS A REAL MOTIVE: Is true_motive specific enough that a reader would believe this person really did it for this reason — not "they were evil"?
4. METHOD IS CONCRETE: Does "method" explain, in specific physical detail, HOW the true_culprit actually did it?
5. TRUE SOLUTION IS COHERENT: Does true_solution's chronology actually match true_culprit, true_motive, and method with no gaps or contradictions?
6. SPECIFIC STAKES: Is exactly what's lost if the case goes unsolved concrete — not "it would be upsetting"?
7. SUSPECTS ARE PLAUSIBLE, CORRECTLY COUNTED, AND EACH HAS AN INNOCENT EXPLANATION: Does each suspect in "suspects" have genuine means, motive, AND opportunity — not just being present at the scene? AND does EACH suspect entry (not only the one named as red_herring) include its own specific, concrete innocent_explanation? A suspect with no locked innocent explanation cannot be safely resolved later.{" This is a 16-page/24-panel book — there must be EXACTLY 2 suspects in the list, no more. If there are 3 or more, MERGE the weakest one into another suspect or cut them entirely, keeping only the 2 strongest — a 24-panel book does not have room to resolve a third suspect on top of everything else the climax needs to show." if page_count <= 24 else ""}
8. RED HERRING IS GENUINELY MISLEADING: Would a reasonable reader actually suspect the red_herring character first, based on real evidence/circumstance?
9. RED HERRING EXPLANATION IS SPECIFIC: Does red_herring_explanation give a concrete, real innocent explanation for that suspect — not just "they didn't do it"?
10. EVERY CLUE STATES WHAT IT PROVES: Do clue_1, clue_2, and clue_3 each explicitly say what they prove or point toward — not just describe an object with no stated bearing on who's guilty?
11. CLUE PROVENANCE: Do clue_1, clue_2, and clue_3 each state WHO made or left it, roughly WHEN, WHY it exists at all, and WHY it's found in that specific place?
12. KEY CONTRADICTION ACTUALLY BREAKS THE RED HERRING: Does key_contradiction identify something specific about the red_herring suspect's alibi, timeline, or method that could not be true if they were guilty?
13. COMPANION CONTRIBUTES MATERIALLY: Does companion_contribution name something only the companion provides that the case genuinely needs — not just encouragement from the sidelines?
14. PROTAGONIST AGENCY: Does {hero_name}'s own observation or decision drive clue_1, midpoint_complication, clue_2, key_contradiction, and clue_3 — or does the case advance via someone else's unprompted confession?
15. ANTAGONIST AWARENESS IS EXPLAINED: If midpoint_complication has the true_culprit or a suspect confronting {hero_name} with knowledge of the investigation, does the spine state HOW they found out?
16. REAL LOW POINT: Does low_point genuinely make the case look unsolvable or put {hero_name} in real doubt/danger?
17. FINAL DEDUCTION IS AN ACTUAL DEDUCTION: Does final_deduction show real reasoning connecting clue_1, clue_2, clue_3, and key_contradiction to the solution — not "she just knew" or "it all made sense"?
18. PROOF IS SPECIFIC AND ALREADY ESTABLISHED: Is "proof" one specific, named piece of evidence that was already introduced as clue_1, clue_2, or clue_3 — not a new fact appearing for the first time at the climax?
19. REVEAL IS NOT THE CLIMAX: Is climax a real confrontation with risk/consequence that the accusation triggers — not simply "{hero_name} announces who did it"? If it's just the announcement: REWRITE climax to add the real confrontation.
20. RESOLUTION CLOSES THE TRUE CULPRIT AND THE STAKES: Does resolution show the true_culprit held accountable using "proof," AND explicitly show the ORIGINAL stakes from item 6 being settled?
21. RESOLUTION EXPLICITLY CLEARS EVERY SUSPECT, NOT JUST THE RED HERRING: Does resolution explicitly address the red_herring suspect using red_herring_explanation — AND every OTHER named suspect in "suspects" using their own innocent_explanation — showing each one cleared or accounted for, rather than silently dropping any of them? A mystery that raises real suspicion about a named character and never resolves it is a hard failure, not a stylistic gap, no matter which suspect it happens to.
22. THE PROTAGONIST SOLVES IT, NOT THE CULPRIT: Does final_deduction show {hero_name}'s OWN reasoning reaching the solution — or does the case actually get solved by the culprit confessing unprompted, a coincidence, or another character simply explaining it? If climax or resolution has the culprit admitting something, does that admission come AFTER final_deduction/the accusation (corroborating a solution already reached) — or does it come FIRST, meaning the confession is what actually reveals the answer? If the confession does the revealing: REWRITE climax/resolution so {hero_name}'s own deduction and accusation come first, with any admission afterward.
23. CENTRAL MYSTERY'S OWN HOOK IS ANSWERED: central_mystery describes a specific hook (a missing object, hidden treasure, disappearance, secret identity, or unexplained event). Does resolution explicitly state what happened to/with THAT SPECIFIC hook — not just that the culprit was caught and stakes were settled in the abstract? A resolution that closes the whodunit but never says where the missing object ended up, what the disappearance actually was, or what the secret identity turned out to be has left the story's own central question unanswered: REWRITE resolution to explicitly answer it (a single deliberate small secondary mystery may stay open as a hook, but never the central one).

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. If true_culprit, true_motive, method, or true_solution need to change, make sure every dependent field (suspects, clues, key_contradiction, final_deduction, proof, climax, resolution) is updated to stay consistent with the new solution.

Return ONLY valid JSON, the exact same shape as the input (all 21 fields: central_mystery, victim, true_culprit, true_motive, method, true_solution, stakes, suspects, red_herring, red_herring_explanation, clue_1, companion_contribution, midpoint_complication, clue_2, key_contradiction, clue_3, low_point, final_deduction, proof, climax, resolution)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
            timeout=75,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Mystery blueprint validation response was truncated (hit max_tokens) — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"MYSTERY BLUEPRINT VALIDATED — central_mystery: {revised.get('central_mystery', '')[:100]}")
        return revised
    except Exception as e:
        print(f"⚠️  Mystery blueprint validation failed — using original: {e}")
        return blueprint


def generate_horror_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the horror story's structural spine before writing any prose — same
    two-step architecture as generate_mystery_blueprint_ai, for the same
    reason: without a blueprint, horror_adult page_count>24 books were
    silently falling into generate_prose_story_ai's generic 8-beat fallback
    scene list, which is comedy-shaped ("something goes hilariously wrong",
    "the funniest, most chaotic moment", ending on a "punchline"/"visual
    gag") -- structurally wrong for horror regardless of how well the
    genre_guidance text described what horror should be. The existing
    genre_guidance ("decide the threat's backstory now, for planning only,
    withhold it from the reader") was already trying to describe a locked
    spine in prose form; this makes that spine an actual object instead of
    an instruction to the model to improvise one and remember it correctly
    across 8 separate scene-generation calls.

    Step 1 locks the antagonist's full identity/backstory/method/weakness --
    the "answer key" that must not change -- BEFORE a single incident or
    reveal exists, mirroring how the mystery blueprint locks culprit/motive/
    method before inventing suspects or clues. Step 2 then designs the
    escalating haunting as a schedule of incidents built to gradually reveal
    that locked backstory (never all at once), culminating in a climax
    resolved by a specific established payoff. Only called for horror_adult.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    client = get_client()

    # ── Step 1: lock the TRUE BACKSTORY before inventing a single incident ──
    solution_prompt = f"""You are a professional horror writer. Before designing any incidents or reveals, you must first decide EXACTLY who or what the threat is — the private answer key the haunting will be built to gradually reveal.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

Decide these fields now. Be concrete and specific everywhere — no vague placeholders like "a dark spirit" or "an ancient evil", and NO abstract nouns standing in for an actual event ("a betrayal," "an injustice") without naming exactly who did what to whom:
- central_haunting: What the haunting looks like to {hero_name} AT FIRST — the apparent pattern of strange occurrences before any explanation exists. This is what the story opens with.
- entity_name: The threat's actual name (a real person's name if it's a ghost/spirit, or a specific name/title if it's a demon/entity) — never left as "the entity" or "the presence."
- entity_era: The specific time period this threat is from (a year or decade, e.g. "1962").
- entity_origin: Who or what they were in life, or how they came to exist — concrete, specific detail.
- death_or_creation_event: The SPECIFIC tragic or violent event that created this threat. Name WHO did WHAT to WHOM, by what exact method, and why. "Betrayed by his business partner" is NOT enough — state what the betrayal actually was (a specific act: poisoned, framed for a specific crime, drowned, walled up, denounced to specific authorities for a specific reason) and who specifically did it.
- unfinished_business: The specific unresolved injustice, betrayal, or unfulfilled purpose that keeps this threat bound to this place/task — again, name the specific people/facts involved, not an abstract label for the grievance.
- what_it_wants: What the threat actually wants from {hero_name} specifically, and WHY this particular person (a connection, a resemblance, a location, an inherited object) — not a random victim.
- method_of_harm: EXACTLY how the threat harms people — the specific supernatural mechanism (possession, psychological manipulation into self-harm, physical attack, environmental danger it causes). You must be able to answer "how does it actually hurt someone" in concrete terms.
- weakness_or_banishment_method: The ONE specific MECHANICAL thing that could stop, banish, or satisfy this threat, and it MUST include a genuinely PHYSICAL, TACTILE component — an object that must be destroyed/burned/buried/returned, a physical ritual gesture that must be performed at a specific place, a specific unfinished physical task that must be completed. Do NOT make this "reveal/speak/publish a truth" on its own, even if the truth is specific and factual — verbal revelation alone reliably collapses into comfort dialogue ("you weren't a murderer," "rest now") when written, regardless of how mechanical it sounds in a one-line summary. If a truth must be told as part of the mechanism, it must be paired with a physical act performed AT THE SAME TIME (burning the object the truth is written on while speaking it, completing a physical ritual gesture as the words are said, returning a physical item to its rightful place as the confession is made) — the physical action is what actually does the work; the words are texture, not the mechanism itself. WRONG: "reveal the journal publicly, exposing the truth." RIGHT: "burn the journal's final page at the site of the death while reciting the specific absolution rite written in its margins — the burning breaks the binding, not the words alone."
- stakes: What is specifically lost if {hero_name} fails to resolve this — concrete (a life, a soul, a family member, an escape) — not "it would be scary."

Return ONLY valid JSON:
{{"central_haunting": "...", "entity_name": "...", "entity_era": "...", "entity_origin": "...", "death_or_creation_event": "...", "unfinished_business": "...", "what_it_wants": "...", "method_of_harm": "...", "weakness_or_banishment_method": "...", "stakes": "..."}}"""

    try:
        solution_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional horror story architect. Return only valid JSON."},
                {"role": "user", "content": solution_prompt},
            ],
            temperature=0.8,
            max_tokens=900,
            timeout=60,
        )
        solution = json.loads(solution_response.choices[0].message.content)
        print(f"HORROR THREAT (locked first) — entity: {solution.get('entity_name', '')[:80]}, wants: {solution.get('what_it_wants', '')[:80]}")
    except Exception as e:
        print(f"⚠️  Horror threat generation failed: {e}")
        return {}

    # ── Step 2: design the haunting as a schedule of incidents built to
    # gradually REVEAL the locked backstory above -- not invented independently ──
    haunting_prompt = f"""You are a professional horror writer planning the HAUNTING for a {page_count}-panel comic book. The threat's true identity and backstory are already decided and LOCKED — your job is to design an escalating schedule of incidents that gradually reveals this exact truth, never all at once, and never anything else.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

THE LOCKED THREAT (do not change or contradict any part of this, and do not introduce a second/different threat):
Central haunting (what it looks like at first): {solution.get('central_haunting', '')}
Entity: {solution.get('entity_name', '')}, from {solution.get('entity_era', '')}
Origin: {solution.get('entity_origin', '')}
Death/creation event: {solution.get('death_or_creation_event', '')}
Unfinished business: {solution.get('unfinished_business', '')}
What it wants (and why {hero_name} specifically): {solution.get('what_it_wants', '')}
Method of harm: {solution.get('method_of_harm', '')}
Weakness/banishment method: {solution.get('weakness_or_banishment_method', '')}
Stakes: {solution.get('stakes', '')}

Design the haunting with these exact fields, every one of them built to gradually reveal — and ultimately resolve — the locked threat above:
- protagonist_exploited_insecurity: A SPECIFIC personal fear, guilt, or insecurity of {hero_name}'s — tied to their personality/traits above — that has nothing to do with the haunting at first, but that the entity will later exploit psychologically BECAUSE it's specific to them. Not generic ("fear of failure") — a specific memory, belief, or unresolved guilt.
- incident_1: The FIRST manifestation — small, ambiguous, deniable (a sound, an object moved, a cold spot, a shape half-seen). {hero_name} could still rationalize this away. Reveals NOTHING of the backstory yet.
- incident_2: A second, stronger manifestation, PLUS the first fragment of the backstory reveal — name the SPECIFIC source (a local historian, an old newspaper clipping, a previous occupant, a found object) and the SPECIFIC partial fact it reveals (name names, name the specific act — not "a betrayal happened"). This fragment must raise a new question, not answer everything, AND it must change what {hero_name} does next (a new place to search, a new danger to avoid, a new suspicion) — not just be interesting trivia.
- midpoint_discovery: A turning-point revelation that reframes {hero_name}'s understanding of what they're dealing with — connects to the entity's true identity but still does not reveal the full picture (save death_or_creation_event's full detail and what_it_wants for later). State the SPECIFIC source of this discovery, the SPECIFIC new fact it adds, and how it CHANGES {hero_name}'s strategy or reveals the situation is worse than believed — not just emotional context.
- incident_3: A more dangerous, physically threatening manifestation — {hero_name} is in genuine physical danger, not just unsettled. This should also start to reveal WHY this entity targets people like {hero_name} (connecting toward what_it_wants).
- incident_4: The entity directly and specifically exploits protagonist_exploited_insecurity — naming it, using it, turning it against {hero_name} in a way only possible because the entity has been watching/learning about them. This is the psychological low blow, not another jump-scare.
- low_point: The moment {hero_name} is nearly consumed, captured, or broken — success looks genuinely unlikely. By now {hero_name} has settled on false_theory (below) as the explanation — NOT the full truth yet. The entity must still be at FULL, active, unresolved power here — this happens LATE, close to the end, not with room left over for a long resolution afterward.
- false_theory: The plausible, sympathetic-sounding explanation {hero_name} has PIECED TOGETHER by this point from incident_2/midpoint_discovery/incident_3 — and believes is the complete truth — but which is actually WRONG or dangerously INCOMPLETE. This must be a reasonable, well-supported conclusion given what's been revealed so far (not something dumb or unmotivated) — the reader should believe it too. Examples of the SHAPE (invent your own specific to this story): {hero_name} believes the entity is simply a wronged victim seeking acknowledgment, when it's actually something else entirely (it caused the deaths it's blamed for; it isn't who the "victim" was at all; the real threat is a second entity/motive it's been concealing; "helping" it as {hero_name} plans would free something worse).
- false_resolution_attempt: Acting on false_theory, {hero_name} attempts what seems like the obvious, reasonable solution (a ritual, an act of justice, exposing what they believe is the whole story) — described specifically enough that the reader also expects it to work.
- reversal_discovery: The SPECIFIC new fact, uncovered exactly as false_resolution_attempt is happening or immediately after, that proves false_theory wrong or dangerously incomplete — a genuine "we had it wrong" moment that recontextualizes everything revealed so far. State the SPECIFIC source of this discovery and the SPECIFIC new fact. This must change what {hero_name} now understands they actually need to do — the false_resolution_attempt must fail or actively make things WORSE because it was based on the wrong theory, not just fail to work.
- climax: The REAL, sustained final confrontation — beginning only AFTER reversal_discovery, once {hero_name} understands the true situation. Life-or-death, not a quick appearance. What does the entity do at full force now that the false attempt has failed/provoked it? What must {hero_name} personally do, at real risk, using weakness_or_banishment_method as tactical leverage (not just an emotional gesture, and not a repeat of false_resolution_attempt)? This must begin LATE — roughly the final 10-15% of the story — the entity is still a genuine, active threat capable of harming {hero_name} right up until this point. The DANGER can be sustained and drawn out, but the actual MOMENT OF DEFEAT, once the mechanical action completes, must be SUDDEN — a forcible negation (torn apart, extinguished, yanked away, shattered) within a beat or two of the mechanism completing. Do NOT describe the entity gradually calming down, becoming grateful, or coming to peace over several moments — that reads as consoling it into leaving rather than defeating it.
- payoff: Confirm which specific earlier element (the object, ritual, or task from weakness_or_banishment_method, established at incident_2, midpoint_discovery, or incident_3) {hero_name} PHYSICALLY acts on to survive/resolve the REAL climax — name the specific hands-on action (burns/buries/destroys/returns/completes) — and state MECHANICALLY what that physical action does to the entity (breaks its binding, destroys its anchor, fulfills the specific condition that was keeping it bound/active). Do not describe this as {hero_name} merely speaking, revealing, or announcing something — even a specific true fact, spoken alone, is not a payoff. This must be a DIFFERENT or DEEPER action than false_resolution_attempt (which was likely verbal/emotional), informed by reversal_discovery.
- resolution: The EXPLICIT, unambiguous outcome — state clearly whether the entity is defeated/banished/satisfied, OR {hero_name} escapes (entity survives), OR {hero_name} is caught/consumed. Tie this directly to unfinished_business specifically (is it finally resolved, or does it remain, tragically, unresolved) — never end on an open question or vague implication about WHAT happened. The outcome itself must be clear even though final_sting (below) keeps the tone unsettled.
- final_sting: A SPECIFIC small, unsettling detail, ambiguity, or visual callback for the very last beat of the story — something that does NOT contradict or undo the explicit resolution above, but prevents the ending from feeling completely safe/cozy. Examples of the SHAPE (invent your own specific to this story): an object connected to the entity is still somehow present/moved when it shouldn't be, a detail from an early incident quietly recurs, a character notices something just slightly wrong that they dismiss, a physical trace suggesting the entity's "unfinished business" pattern could recur elsewhere. This must be a concrete, specific detail — not a vague sense of unease.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
1. SINGLE ANTAGONIST TEST: Every incident, every reveal, and the climax must all be about the ONE entity above. Do not introduce a second, different threat (a cult, a copycat, an unrelated spirit) anywhere.
2. WITHHELD REVEAL TEST: incident_1 must reveal nothing. incident_2 and midpoint_discovery must each reveal only a FRAGMENT — never the complete entity_name + era + death_or_creation_event + what_it_wants picture until at or after low_point. If your incident_2 or midpoint_discovery already gives away the full backstory, thin it down to a partial fact and hold the rest back.
3. CONCRETE REVELATION TEST: Every reveal (incident_2, midpoint_discovery, low_point, reversal_discovery, climax) must name specific people, specific acts, and specific facts — never an abstract label like "a betrayal" or "a terrible injustice" standing in for the actual event. If a reveal doesn't tell the reader WHO did WHAT to WHOM, it isn't a revelation yet — rewrite it with the specifics.
4. REVELATIONS CHANGE THE SITUATION, NOT JUST THE MOOD: incident_2's follow-up and midpoint_discovery must each give {hero_name} something ACTIONABLE — a new tool, a new rule about the entity, a worse truth that changes the stakes, or a reversal of what they believed — not merely a sad backstory to feel bad about. If a revelation only adds emotional context with no tactical consequence, redesign it so it changes what {hero_name} does or believes next.
5. ESCALATION TEST: incident_1 → incident_2 → incident_3 → incident_4 must each be more dangerous, more personal, or more intense than the last — never a repeat of the same kind of scare.
6. EXPLOITED INSECURITY TEST: incident_4 must use the SPECIFIC insecurity named in protagonist_exploited_insecurity, by name/detail — not a generic scare that could apply to anyone.
7. LATE-THREAT TEST: The entity must remain a genuine, ACTIVE danger all the way through low_point and into the REAL climax, with no permanent safety or resolution before it begins. The real climax must not begin until roughly the final 10-15% of the story.
8. MECHANICAL PAYOFF TEST (read carefully — this is commonly gotten wrong): The REAL climax must be resolved through a PHYSICAL, TACTILE action using weakness_or_banishment_method — {hero_name}'s HANDS must do something to a physical object or perform a physical ritual gesture (burn it, bury it, destroy it, return it, complete a physical action at a specific place) — not merely SPEAK something, however factual or specific. Revealing/announcing/confessing a truth out loud, on its own, is NOT a mechanical payoff no matter how specific the truth is — it reliably collapses into comfort dialogue when actually written. If climax has {hero_name} only talking, redesign it so a physical action is what actually breaks the entity's hold, happening at the same moment as any dialogue (the words are texture, the action is the mechanism). BANNED DIALOGUE/BEAT PATTERN — do not describe the climax as therapy delivered to the entity: no version of "you don't have to carry this anymore," "rest now," "let go," "you're free now," "your suffering is over," "you were never a murderer," where the entity simply calms down and dissolves because it was told something comforting or absolving. WRONG: "{hero_name} shares the truth publicly, acknowledging the entity's suffering, tells it to rest, and it finds peace." WRONG (still fails, even though it sounds mechanical): "{hero_name} reveals the journal's contents, exposing the truth, and the entity is freed." RIGHT: "{hero_name} burns the journal's final page at the exact spot where the death occurred, the specific ritual act written in its margins — the fire, not the words, is what severs the binding; the entity doesn't leave because it feels better, it leaves because the physical condition that kept it here is now physically undone."
9. AGENCY TEST: {hero_name}'s own decisions and observations must drive the investigation (incident_2's follow-up, midpoint_discovery, incident_3, reversal_discovery) — not information that simply falls into their lap.
10. EXPLICIT RESOLUTION, UNSETTLED TONE: resolution must clearly state ONE of: the entity is defeated/banished/satisfied, {hero_name} escapes (entity survives), or {hero_name} is caught/consumed — and must explicitly address unfinished_business's fate. Vague or ambiguous endings fail this test outright. Separately, final_sting must give the very last beat a genuine, specific unsettling detail — a resolution that leaves the reader feeling completely safe and cozy with no final_sting fails this test too.
11. GENUINE THIRD-ACT REVERSAL TEST (read carefully — this is the most commonly skipped structural beat): The story must NOT be a straight line of "ghost seems scary → investigate → discover it was wronged → help it → it leaves." false_theory must be a real, well-earned belief the evidence so far actually supports — not a strawman. false_resolution_attempt must be a genuine, reasonable attempt that a smart protagonist would try. reversal_discovery must be a SURPRISING but fairly-clued new fact that specifically invalidates or complicates false_theory — not just "it needs one more push." The REAL climax (after the reversal) must require a MEANINGFULLY DIFFERENT action than false_resolution_attempt — if the real climax is just "try the same thing again, harder," the reversal isn't doing real work: redesign reversal_discovery and the real climax so the second attempt is tactically different because {hero_name} now understands something they didn't before.

Return ONLY valid JSON:
{{"protagonist_exploited_insecurity": "...", "incident_1": "...", "incident_2": "...", "midpoint_discovery": "...", "incident_3": "...", "incident_4": "...", "low_point": "...", "false_theory": "...", "false_resolution_attempt": "...", "reversal_discovery": "...", "climax": "...", "payoff": "...", "resolution": "...", "final_sting": "..."}}"""

    try:
        haunting_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional horror story architect designing a haunting to gradually reveal an already-fixed threat. Return only valid JSON."},
                {"role": "user", "content": haunting_prompt},
            ],
            temperature=0.8,
            max_tokens=1800,
            timeout=60,
        )
        haunting = json.loads(haunting_response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Horror haunting generation failed: {e}")
        return {}

    blueprint = {**solution, **haunting}
    print(f"HORROR BLUEPRINT — central_haunting: {blueprint.get('central_haunting', '')[:100]}")
    return blueprint


def validate_horror_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the horror blueprint against structural criteria BEFORE any prose
    is written, and revises the specific fields that fail. Mirrors
    validate_mystery_blueprint_ai's architecture/rationale for the same
    two-step, backstory-first schema (see generate_horror_blueprint_ai).
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing a HORROR STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
1. CONCRETE CENTRAL HAUNTING: Could a reader state what the haunting looks like at first, in one sentence?
2. ENTITY IS NAMED AND SPECIFIC: Does entity_name give an actual name (not "the entity" / "the presence"), with a specific entity_era?
3. DEATH/CREATION EVENT IS CONCRETE: Does death_or_creation_event name WHO did WHAT to WHOM, by what method — or does it use an abstract label like "a betrayal" or "an injustice" without ever stating the actual act? REWRITE to name the specific act and the specific person responsible if it's still abstract.
4. UNFINISHED BUSINESS IS SPECIFIC: Is unfinished_business a real, specific unresolved injustice/purpose naming specific people/facts — not vague ("it can't move on")?
5. TARGETS THIS PROTAGONIST SPECIFICALLY: Does what_it_wants explain why {hero_name} specifically (a connection, resemblance, location, inherited object) — not a random/interchangeable victim?
6. METHOD OF HARM IS CONCRETE: Does method_of_harm explain, in specific mechanical terms, HOW the threat actually harms people?
7. WEAKNESS IS SPECIFIC, USABLE, AND PHYSICALLY MECHANICAL: Does weakness_or_banishment_method require a genuine PHYSICAL action — {hero_name}'s hands doing something to an object or performing a physical ritual gesture (burn/bury/destroy/return/complete) — or does it rest on merely SPEAKING/REVEALING/ANNOUNCING a truth with no physical component? "Reveal the truth publicly" or "expose what really happened" is NOT sufficient even though it sounds mechanical — it collapses into comfort dialogue when written. If there's no physical action, REWRITE weakness_or_banishment_method to add one (the truth can still be spoken, but a physical act — burning the document it's written on, completing a ritual gesture at the same moment — must be what actually does the work).
8. SPECIFIC STAKES: Is what's lost on failure concrete — not "it would be scary"?
9. INSECURITY IS SPECIFIC, NOT GENERIC: Is protagonist_exploited_insecurity a specific memory, belief, or unresolved guilt tied to {hero_name}'s own personality/traits — not a generic fear that could belong to anyone?
10. INCIDENT_1 REVEALS NOTHING: Is incident_1 genuinely small and deniable, revealing zero backstory — not already hinting at the entity's identity?
11. INCIDENTS 2 AND MIDPOINT REVEAL ONLY FRAGMENTS: Do incident_2 and midpoint_discovery each reveal only a PARTIAL fact (with a named, specific source — a person, document, object) — or do they already give away the complete backstory (entity_name + era + death_or_creation_event + what_it_wants) too early? If the full picture is already assembled by midpoint_discovery, REWRITE to thin it to a fragment and hold the rest for low_point/climax.
11b. REVEALS ARE CONCRETE, NOT ABSTRACT LABELS: Do incident_2 and midpoint_discovery name specific people and specific acts — or do they use an abstract label ("a betrayal," "a dark secret") without ever stating what actually happened? REWRITE with specifics if so.
11c. REVEALS ARE ACTIONABLE, NOT JUST SAD BACKSTORY: Does each reveal give {hero_name} something to DO differently — a new tool, a new rule about the entity, a worse truth that raises the stakes, a reversal of what was believed — or is it only emotional context with no tactical consequence? REWRITE so the reveal changes {hero_name}'s strategy or understanding of the danger, not just their sympathy.
12. ESCALATION IS REAL: Does each of incident_1 → incident_2 → incident_3 → incident_4 get more dangerous, personal, or intense than the last — not a repeat of the same scare with new scenery?
13. INCIDENT_4 USES THE SPECIFIC INSECURITY: Does incident_4 explicitly use the SAME specific insecurity named in protagonist_exploited_insecurity — not a generic scare?
14. PROTAGONIST AGENCY: Does {hero_name}'s own decision or observation drive incident_2's follow-up, midpoint_discovery, and incident_3 — or does information simply fall into their lap via unprompted exposition?
15. REAL LOW POINT, LATE AND ACTIVE: Does low_point genuinely make survival look unlikely — and is the entity still at FULL, ACTIVE power here, with no premature safety?
15b. FALSE THEORY IS EARNED AND PLAUSIBLE, NOT A STRAWMAN: Is false_theory a genuinely reasonable conclusion given incident_2/midpoint_discovery/incident_3 — something a smart reader would also believe — rather than an obviously wrong or flimsy idea nobody would actually settle on?
15c. FALSE RESOLUTION ATTEMPT IS A GENUINE, REASONABLE TRY: Is false_resolution_attempt something a smart, competent protagonist would actually attempt given false_theory — specific and plausible enough that failure is a real surprise, not something obviously doomed from the start?
15d. REVERSAL IS A REAL SURPRISE THAT CHANGES THE PLAN: Does reversal_discovery introduce a SPECIFIC new fact (with a stated source) that genuinely invalidates or complicates false_theory — not just "it needs more effort" or a minor wrinkle? Does it explain WHY false_resolution_attempt failed or backfired? If reversal_discovery is weak, vague, or doesn't actually change anything about what {hero_name} must do next, REWRITE it to be a genuine "we had it wrong" turn.
15e. THE REAL CLIMAX IS TACTICALLY DIFFERENT FROM THE FALSE ATTEMPT: Compare climax to false_resolution_attempt — is the real climax's action meaningfully different (informed by what reversal_discovery revealed), or is it just "try the exact same thing again, but harder"? If the two are basically the same action repeated, REWRITE climax so it uses the NEW understanding from reversal_discovery in a way the false attempt couldn't have.
16. EARNED, PHYSICALLY MECHANICAL, SUDDEN CLIMAX, NO THERAPY-SPEAK: Is the REAL climax (after reversal_discovery) resolved using weakness_or_banishment_method through a PHYSICAL action — {hero_name}'s hands doing something to an object or completing a physical ritual gesture — with the defeat itself SUDDEN once that physical action completes (not a gradual multi-beat softening where the entity progressively calms, listens, and comes to peace)? Or is it resolved merely by {hero_name} speaking/revealing/announcing a truth, publicly clearing the entity's name, or emotionally validating its grievance, with no physical action behind it? "Reveals the truth" or "exposes what happened" is NOT a physical mechanism even if the fact itself is specific — REWRITE so a physical act (burning/burying/destroying/returning/completing) is what actually happens, with any dialogue as texture alongside it, not the mechanism itself. Watch specifically for "you don't have to carry this anymore," "rest now," "let go," "you're free now," "he's finally listening," "it's over," "you were never a murderer" or equivalents where the entity simply calms and dissolves because it was told something comforting or absolving — this failure mode can show up as a SEQUENCE of softening beats even without any single banned phrase, and it recurs even after a good reversal if weakness_or_banishment_method is purely verbal. If the resolution is verbal/emotional-validation-only or spreads the defeat across a gradual calm-down, REWRITE climax/payoff so a physical action does something concrete and SUDDEN to the entity's power, not just its feelings.
17. CLIMAX HAS ROOM TO BREATHE: Does climax read as a sustained confrontation (not a single quick beat immediately followed by resolution)?
18. RESOLUTION IS EXPLICIT AND UNAMBIGUOUS: Does resolution clearly state ONE OF: entity defeated/banished/satisfied, {hero_name} escapes (entity survives), or {hero_name} is caught/consumed — with unfinished_business's fate explicitly addressed? REWRITE resolution if it's vague, open-ended, or doesn't commit to one clear outcome.
19. SINGLE-ANTAGONIST CONSISTENCY: Is every incident, reveal, and the climax about the SAME ONE entity — no second/different threat introduced anywhere in the spine?
20. FINAL STING EXISTS AND IS SPECIFIC: Is final_sting a concrete, specific unsettling detail, ambiguity, or callback for the very last beat — not a vague sense of unease, and not something that contradicts the explicit resolution? If final_sting is missing, generic, or would leave the ending feeling completely safe and cozy, REWRITE it with a specific detail.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. If entity_name, death_or_creation_event, method_of_harm, or weakness_or_banishment_method need to change, make sure every dependent field (incidents, midpoint_discovery, low_point, false_theory, false_resolution_attempt, reversal_discovery, climax, payoff, resolution, final_sting) is updated to stay consistent with the new backstory.

Return ONLY valid JSON, the exact same shape as the input (all 24 fields: central_haunting, entity_name, entity_era, entity_origin, death_or_creation_event, unfinished_business, what_it_wants, method_of_harm, weakness_or_banishment_method, stakes, protagonist_exploited_insecurity, incident_1, incident_2, midpoint_discovery, incident_3, incident_4, low_point, false_theory, false_resolution_attempt, reversal_discovery, climax, payoff, resolution, final_sting)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2200,
            timeout=75,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Horror blueprint validation response was truncated (hit max_tokens) — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"HORROR BLUEPRINT VALIDATED — central_haunting: {revised.get('central_haunting', '')[:100]}")
        return revised
    except Exception as e:
        print(f"⚠️  Horror blueprint validation failed — using original: {e}")
        return blueprint


def generate_horror_kids_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the kid-friendly horror ("horror" genre, distinct from horror_adult)
    story's structural spine before writing any prose. Mirrors
    generate_horror_blueprint_ai's two-step, creature-first architecture
    exactly (lock the creature's true identity and the mechanism that
    resolves things BEFORE inventing a single incident), for the same
    reason it was built for horror_adult: without a locked spine, longer
    books fall into the generic comedy-shaped 8-beat fallback, and shorter
    ones drift into a flat "spooky thing -> turns out friendly -> everyone's
    nice now" line with no real mystery or reversal.

    IMPORTANT DISTINCTION (learned the hard way from a first-draft test):
    "kid-friendly" is a CONTENT restriction (no violence/gore/real danger),
    NOT a TONE restriction. The creature's backstory must be gentle (never a
    violent death) and the underlying mechanism behind the "scares" must be
    harmless in reality — but incidents 1-4 must still be WRITTEN as
    genuinely unsettling and ambiguous, with the creature's harmlessness
    withheld from the reader until the reversal, the same way a real (if
    G-rated) ghost story works. A first version of this blueprint described
    "scares" as inherently cute/funny (giggling, silly noises) which
    telegraphed friendliness from incident_1 and produced a story with zero
    suspense — content-safe, but not actually horror. The stakes must never
    be physical danger, and the ending must land warm and fun rather than
    horror_adult's deliberately unsettled "final sting" — but that warmth
    belongs to the ending only, after the reversal, not the whole story.
    The one structural lesson kept unchanged from horror_adult is the genuine
    third-act reversal (kids wrongly assume the creature is mean, act on
    that, then discover it was lonely/misunderstood) and the requirement
    that the payoff be a concrete ACTION, not just reassuring words — the
    same "verbal-only resolution reads as hollow" lesson, reframed as
    kindness instead of banishment.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    client = get_client()

    # ── Step 1: lock the TRUE NATURE of the creature before inventing a single incident ──
    solution_prompt = f"""You are a professional children's author writing kid-friendly horror/spooky-mystery fiction (think Coraline, Goosebumps, Hocus Pocus, or a good Halloween episode of a kids' show — GENUINELY spooky and suspenseful, with the content kept age-appropriate). Before designing any incidents or reveals, you must first decide EXACTLY what the "spooky" creature actually is — the private answer key the mystery will be built to gradually reveal.

CRITICAL DISTINCTION — READ THIS FIRST: "Kid-friendly" is a CONTENT restriction, not a TONE restriction. It means: no graphic violence, no gore, no real death shown, no genuine cruelty, no content that would traumatize a young reader. It does NOT mean the creature must read as friendly, cute, or harmless from the start. The reader (and {hero_name}) should genuinely NOT KNOW whether this presence is dangerous until late in the story — real suspense, real dread, real "is this thing going to hurt someone?" uncertainty, the same way Coraline or a good ghost story for kids works. A story where the "scares" are obviously silly, jokey, or friendly from the very first incident has failed at being horror, even if it's perfectly clean content-wise.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

Decide these fields now. Be concrete and specific everywhere — no vague placeholders like "a spooky creature" or "something strange". CONTENT stays age-appropriate (no violence, death, injury, or real danger to anyone) but the creature's true nature, once decided, must be something that COULD plausibly read as a real threat to a reader who doesn't know the answer yet:
- central_mystery: What the spooky happenings look like to {hero_name} AT FIRST — the apparent pattern of strange occurrences before any explanation exists. This is what the story opens with.
- creature_name: The creature's actual name — a real name/nickname, never left as "the creature" or "the thing."
- creature_nature: EXACTLY what kind of creature/spirit it actually is, specifically (a shy young ghost, a lost baby dragon, a forgetful house spirit, a misplaced holiday mascot, a shape-shifting swamp critter) — never generic ("a monster"). This is the ANSWER, known only to you for planning — it must NOT leak into how incident_1 through incident_4 are written; those must stay ambiguous and unsettling.
- origin_or_backstory: The SPECIFIC, GENTLE reason it came to be here or is hiding — concrete and specific, but never violent or tragic in a scary way (it got separated from its family during a move, it fell asleep for a very long nap and woke up confused, its old home was torn down and it had nowhere else to go, it was accidentally left behind when everyone else left). Name the specific circumstance.
- unfinished_business: The specific, harmless thing it's trying to accomplish or find (its favorite lost toy, a way back to its family, a chance to join in on a tradition it always gets left out of, a friend it's looking for).
- what_it_wants: What it actually wants from {hero_name} specifically, and WHY this particular kid (a resemblance, a kindness {hero_name} once showed, something {hero_name} has that it needs) — not a random target.
- method_of_spookiness: EXACTLY how it causes the "spooky" happenings — a concrete mechanism that is HARMLESS IN REALITY but must be EXPERIENCED AND DESCRIBED as genuinely unsettling and ambiguous by the characters living through it (a cold draft with no source, a shape glimpsed at the edge of vision, objects disturbed with no one visibly there, an unexplained sound down the hall, something watching from just out of sight). Do NOT default to inherently cute/jokey mechanisms (giggling, silly cartoonish noises, obviously friendly pranks) — those give the game away in incident_1 and destroy all suspense. The creature's actual harmlessness is a SECRET the reader learns late, not something telegraphed by how the scares are described. This must never cause real physical harm, injury, or lasting danger — but "no one gets hurt" and "doesn't feel scary" are two different things; only the first is required.
- resolution_method: The ONE specific, concrete KIND ACTION that will fix things and make the creature happy — {hero_name} must actually DO something physical/tangible (return its lost toy, throw it a surprise party, help it find its way home, rebuild something that got knocked down, include it in a tradition it's always missed). Do NOT make this just "tell it it's welcome" or "be nice to it" with nothing actually done — a kind action must be paired with real dialogue, but the ACTION is what actually fixes things, not the words alone. WRONG: "tell the ghost it has friends now." RIGHT: "return the music box it's been searching for and play its favorite song on it together — the specific object and action is what finally puts it at ease."
- stakes: What is specifically at risk if {hero_name} doesn't resolve this — concrete and kid-appropriate (a fun town event getting cancelled, the creature staying lonely and sad forever, a friendship or trust being lost, a beloved tradition disappearing) — NEVER physical danger or harm to any character.

Return ONLY valid JSON:
{{"central_mystery": "...", "creature_name": "...", "creature_nature": "...", "origin_or_backstory": "...", "unfinished_business": "...", "what_it_wants": "...", "method_of_spookiness": "...", "resolution_method": "...", "stakes": "..."}}"""

    try:
        solution_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional children's story architect writing genuinely spooky, suspenseful horror with kid-appropriate content (not kid-soft tone). Return only valid JSON."},
                {"role": "user", "content": solution_prompt},
            ],
            temperature=0.8,
            max_tokens=900,
            timeout=60,
        )
        solution = json.loads(solution_response.choices[0].message.content)
        print(f"HORROR-KIDS CREATURE (locked first) — name: {solution.get('creature_name', '')[:80]}, wants: {solution.get('what_it_wants', '')[:80]}")
    except Exception as e:
        print(f"⚠️  Horror-kids creature generation failed: {e}")
        return {}

    # ── Step 2: design the mystery as a schedule of incidents built to
    # gradually REVEAL the locked truth above -- not invented independently ──
    mystery_prompt = f"""You are a professional children's author planning the SPOOKY MYSTERY for a {page_count}-panel comic book. The creature's true nature and the way to resolve things are already decided and LOCKED — your job is to design an escalating schedule of incidents that gradually reveals this exact truth, never all at once, and never anything else.

CRITICAL — KID-FRIENDLY MEANS CONTENT-SAFE, NOT SUSPENSE-FREE: incident_1 through incident_4 and low_point must be written to be GENUINELY unsettling and ambiguous — the reader should not be able to tell whether this presence is dangerous until reversal_discovery. Do not telegraph creature_nature's harmlessness early through cute/jokey details (giggling, silly noises, obviously friendly pranks). Keep the events themselves non-violent and non-graphic, but describe them straight, the way a real ghost story would — genuine unease, not comic relief.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

THE LOCKED TRUTH (do not change or contradict any part of this, and do not introduce a second/different creature):
Central mystery (what it looks like at first): {solution.get('central_mystery', '')}
Creature: {solution.get('creature_name', '')} — {solution.get('creature_nature', '')}
Origin: {solution.get('origin_or_backstory', '')}
Unfinished business: {solution.get('unfinished_business', '')}
What it wants (and why {hero_name} specifically): {solution.get('what_it_wants', '')}
Method of spookiness (must stay harmless throughout): {solution.get('method_of_spookiness', '')}
Resolution method: {solution.get('resolution_method', '')}
Stakes: {solution.get('stakes', '')}

Design the mystery with these exact fields, every one of them built to gradually reveal — and ultimately resolve — the locked truth above:
- protagonist_worry: A SPECIFIC, ordinary kid worry, fear, or embarrassment of {hero_name}'s — tied to their personality/traits above — that has nothing to do with the mystery at first, but that the creature's antics will later accidentally brush against. Not generic ("scared of the dark") — a specific, personal detail (afraid of the attic because of a specific old memory, embarrassed about a specific thing, nervous about an upcoming specific event).
- incident_1: The FIRST spooky happening — small, genuinely unsettling, easily explained away by an adult but not fully shaken off (an unexplained cold draft, a shape at the edge of vision, something moved with no one there, a sound down the hall with no source). Written straight, ambiguous — the reader should feel a real prickle of unease, not amusement. {hero_name} could still rationalize this away, but it lingers. Reveals NOTHING of the truth yet, and gives no hint that the presence is friendly.
- incident_2: A second, stronger and more unsettling happening, PLUS the first fragment of the truth — name the SPECIFIC source (an old photo, a neighbor's story, a diary page, a shopkeeper who remembers something) and the SPECIFIC partial fact it reveals. This fragment must raise a new question, not answer everything, AND it must give {hero_name} something new to DO next — not just be interesting trivia. The fragment can be ambiguous enough to support EITHER a dangerous or a harmless explanation — do not let it accidentally confirm the creature is friendly this early.
- midpoint_discovery: A turning-point discovery that reframes {hero_name}'s understanding of what's going on — connects to the creature's true nature but still does not reveal the full picture, and should still leave real doubt about whether it's dangerous. State the SPECIFIC source and the SPECIFIC new fact, and how it changes {hero_name}'s plan or makes the mystery feel bigger/stranger than expected.
- incident_3: A bigger, MORE unsettling happening (genuinely tense and suspenseful, though still non-violent), directly caused by the investigation getting closer (the presence reacting to being searched for) — not an unrelated new event. This should be the most frightening incident yet, written to make the reader (and {hero_name}) genuinely worried something is wrong — the events stay harmless in reality, but nothing about how it's written should give that away.
- incident_4: The presence's activity directly and unsettlingly brushes against protagonist_worry specifically — naming or touching the SAME specific worry, in a way that feels deliberate and unnerving in the moment, not obviously accidental or cute. This should be the scariest, most personal beat yet — real fear, not a jump-scare played for laughs. Only in hindsight (after reversal_discovery) does the reader learn it wasn't malicious.
- low_point: The moment {hero_name} feels most frightened and discouraged — genuinely afraid, believing something is wrong, with the mystery unsolved and the thing at stake (from stakes) about to be lost. This is the point of maximum dread in the story — no physical danger occurs, but the FEELING of danger should be real and unresolved. By now {hero_name} has settled on false_theory (below) as the explanation.
- false_theory: The plausible-sounding but WRONG conclusion {hero_name} has pieced together by now from incident_2/midpoint_discovery/incident_3 — that the creature is mean, dangerous, or wants to ruin things — and believes is the whole truth. This must be a reasonable conclusion given what's been revealed so far (not something dumb), so the reader believes it too.
- false_resolution_attempt: Acting on false_theory, {hero_name} (with friends) attempts what seems like the obvious solution — a plan to trap, scare off, or get rid of the creature. Described specifically enough that the reader also expects it to work.
- reversal_discovery: The SPECIFIC new fact, uncovered exactly as false_resolution_attempt is happening or right after, that proves false_theory wrong — turns out the creature isn't mean at all, it's {solution.get('what_it_wants', '')} / driven by unfinished_business, and the "scary" behavior was a total misunderstanding. This must land as a genuine surprise that makes {hero_name} feel a little bad for jumping to conclusions, and must change what {hero_name} does next. CRITICAL: learning the truth must NOT immediately resolve the supernatural situation or make the story feel safe — describe what is STILL unresolved, unstable, or frightening about the situation right after this discovery (the creature is still trapped/fading/agitated, time is running out, the atmosphere is still oppressive) so there is a real final challenge left to do, not just a nice conversation left to have.
- climax: The REAL final challenge — beginning only AFTER reversal_discovery, but the danger/fear is NOT over just because the truth is known. This must be a genuinely tense, age-appropriately frightening sequence — real uncertainty about whether it will work, an unsettling atmosphere that does not calm down until the action succeeds — where {hero_name} uses what reversal_discovery revealed to actively perform resolution_method as a concrete ACTION, tactically DIFFERENT from the false attempt (which was about getting rid of the creature; this is about helping it). This action must ALSO force {hero_name} to confront protagonist_worry directly, in the moment, as a real obstacle to completing it (see fear_confrontation below) — the story must not turn safe/reassuring until this sequence concludes. What does {hero_name} specifically DO, with their own hands, using something already established earlier?
- fear_confrontation: The SPECIFIC way protagonist_worry becomes an ACTIVE, in-the-moment obstacle during the climax — {hero_name} must do the very thing they're afraid of, or face it head-on, as part of completing resolution_method's action (not before the climax, not as a separate scene). State exactly what {hero_name} must do despite the fear, and how failing to face it would mean failing the climax. Growth must be demonstrated through this ACTION, never through narration simply stating {hero_name} "wasn't scared anymore" or "found their courage" with no concrete behavior shown.
- payoff: Confirm which specific earlier-established detail/object/clue (from incident_2, midpoint_discovery, or incident_3) {hero_name} uses in the climax — name the specific action {hero_name} takes with it.
- resolution: The EXPLICIT, warm outcome — state clearly that the creature is happy/at peace/has what it needed, that unfinished_business is resolved, and what this means for {hero_name} and the stakes (the event is saved, the friendship/trust is restored). Never end on an open question about what happened.
- final_beat: A SPECIFIC, fun, warm, or silly final detail for the very last beat of the story — a joke, a cute visual callback, or a heartwarming final image. This is the OPPOSITE of an unsettling "sting" — the story should end feeling completely safe, warm, and fun, with nothing left spooky or unresolved.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
0. KID-APPROPRIATE CONTENT, NOT KID-SOFT TONE (check this one first and strictly): Two separate requirements — (a) CONTENT: every incident, including incident_3 and the climax, must stay free of violence, injury, real physical danger, or anything a young reader would find genuinely traumatic. (b) TONE (just as mandatory, and the one most often gotten wrong): incident_1 through incident_4 and low_point must be genuinely suspenseful and ambiguous — ordinary kid stakes and real, escalating unease, the reader unsure whether the presence is dangerous until reversal_discovery. A spine where the incidents are written as obviously cute, jokey, or friendly from the start (giggling, silly cartoon noises, pranks that read as harmless immediately) FAILS this test even though the content is clean — that's a comedy, not horror. Redesign any incident that gives away the creature's harmlessness too early.
1. SINGLE CREATURE TEST: Every incident, every reveal, and the climax must all be about the ONE creature above. Do not introduce a second, different creature anywhere.
2. WITHHELD REVEAL TEST: incident_1 must reveal nothing. incident_2 and midpoint_discovery must each reveal only a FRAGMENT — never the complete creature_name + origin_or_backstory + what_it_wants picture until reversal_discovery. If incident_2 or midpoint_discovery already gives away the full truth, thin it down to a partial fact and hold the rest back.
3. CONCRETE REVELATION TEST: Every reveal must name specific people, places, and facts — never an abstract label like "something sad happened" standing in for the actual event.
4. REVELATIONS CHANGE THE SITUATION, NOT JUST THE MOOD: incident_2's follow-up and midpoint_discovery must each give {hero_name} something ACTIONABLE — a new lead, a new clue, a plan-changing fact — not merely a sad detail to feel bad about.
5. ESCALATION TEST: incident_1 → incident_2 → incident_3 → incident_4 must each feel MORE unsettling, tense, and personal than the last — never a repeat of the same beat, and never a de-escalation into cuteness before reversal_discovery.
6. EXPLOITED WORRY TEST: incident_4 must use the SPECIFIC worry named in protagonist_worry, by name/detail, and must land as genuinely unnerving in the moment — not a generic scare that could apply to anyone, and not played for laughs.
7. LATE-MYSTERY TEST: The mystery/misunderstanding about the creature must remain unresolved all the way through low_point and into the REAL climax, with no early explanation that lets the air out of the story. The real climax must not begin until roughly the final 10-15% of the story.
8. CONCRETE KIND-ACTION PAYOFF TEST (read carefully — this is commonly gotten wrong): The REAL climax must be resolved through a concrete ACTION using resolution_method — {hero_name}'s hands must actually DO something (return an object, build/fix something, throw a specific event, complete a specific task) — not merely SAY something reassuring with nothing actually done. "Tell the creature it has friends now and it feels better" is NOT a sufficient payoff on its own — redesign so a tangible action is what actually resolves things, with any kind words as accompaniment, not the mechanism itself.
9. AGENCY TEST: {hero_name}'s own decisions and observations must drive the investigation — not information that simply falls into their lap.
10. EXPLICIT, WARM RESOLUTION TEST: resolution must clearly state the creature's happy outcome and unfinished_business's fate — vague or ambiguous endings fail this test. Separately, final_beat must be a specific, fun, warm detail — an ending that's merely "safe" with no specific fun final image also fails; give it a genuine, specific button.
11. GENUINE THIRD-ACT REVERSAL TEST (read carefully — this is the most commonly skipped structural beat): The story must NOT be a straight line of "spooky thing happens → investigate → discover it's friendly → everyone's nice now." false_theory must be a real, well-earned belief the evidence so far actually supports — not a strawman. false_resolution_attempt must be a genuine, reasonable plan a smart kid would try. reversal_discovery must be a SURPRISING but fairly-clued new fact that specifically invalidates false_theory. The REAL climax must require a MEANINGFULLY DIFFERENT action than false_resolution_attempt (helping instead of getting rid of) — if the climax is just "try to get rid of it again, but nicer," the reversal isn't doing real work.
12. THE REVELATION MUST NOT END THE HORROR (read carefully — this is commonly gotten wrong even with a good reversal): Learning the creature's true nature in reversal_discovery must launch the final challenge, not immediately make the story safe. WRONG: revelation lands, the mood instantly turns warm and cozy, and climax is just a calm, pleasant scene of doing something nice with zero tension. RIGHT: revelation lands, but something is still unresolved/unstable/frightening (the creature is still trapped or fading, the window to help is closing, the atmosphere is still oppressive) — and climax must be a genuinely tense sequence where {hero_name} actively resolves it. Only resolution/final_beat, AFTER climax concludes, may turn reassuring.
13. PERSONAL FEAR BECOMES PART OF THE CLIMACTIC CHALLENGE (read carefully): fear_confrontation must make protagonist_worry an ACTIVE obstacle DURING climax's action, not something already overcome beforehand or mentioned only in incident_4. {hero_name} must do the specific thing they're afraid of, or face it directly, as part of completing resolution_method — and this must be shown through concrete action, never stated as a fact ("they weren't scared anymore"). If fear_confrontation isn't tied to the SAME specific worry from protagonist_worry, or doesn't actually interfere with completing the climax action, redesign it.

Return ONLY valid JSON:
{{"protagonist_worry": "...", "incident_1": "...", "incident_2": "...", "midpoint_discovery": "...", "incident_3": "...", "incident_4": "...", "low_point": "...", "false_theory": "...", "false_resolution_attempt": "...", "reversal_discovery": "...", "climax": "...", "fear_confrontation": "...", "payoff": "...", "resolution": "...", "final_beat": "..."}}"""

    try:
        mystery_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional children's story architect designing a genuinely suspenseful spooky mystery whose creature is secretly harmless — but the reader must not know that until late. Return only valid JSON."},
                {"role": "user", "content": mystery_prompt},
            ],
            temperature=0.8,
            max_tokens=1800,
            timeout=60,
        )
        mystery = json.loads(mystery_response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Horror-kids mystery generation failed: {e}")
        return {}

    blueprint = {**solution, **mystery}
    print(f"HORROR-KIDS BLUEPRINT — central_mystery: {blueprint.get('central_mystery', '')[:100]}")
    return blueprint


def validate_horror_kids_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the kid-friendly horror blueprint against structural criteria
    BEFORE any prose is written, and revises the specific fields that fail.
    Mirrors validate_horror_blueprint_ai's architecture for the same
    two-step, truth-first schema (see generate_horror_kids_blueprint_ai).
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing a KID-FRIENDLY SPOOKY STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
0. KID-APPROPRIATE CONTENT, NOT KID-SOFT TONE (check this one first and strictly — two separate sub-checks): 0a. CONTENT: Does anything in incident_3, incident_4, low_point, or climax involve real violence, injury, or genuine physical danger to any character? REWRITE any field that crosses into real danger so it stays non-violent/non-graphic. 0b. SUSPENSE (just as mandatory, and the one most often missed): Are incident_1 through incident_4 and low_point written to be genuinely unsettling and ambiguous — real suspense about whether the presence is dangerous — or do they read as obviously cute, jokey, or friendly from the start (giggling, silly cartoon noises, pranks that are clearly harmless immediately)? A spine that is content-safe but tonally soft/cute from incident_1 FAILS this check — REWRITE the incident descriptions so they're genuinely eerie and ambiguous while keeping the underlying events non-violent.
1. CONCRETE CENTRAL MYSTERY: Could a reader state what the spooky happenings look like at first, in one sentence?
2. CREATURE IS NAMED AND SPECIFIC: Does creature_name give an actual name, with creature_nature stating specifically what kind of creature/spirit it is (not "a monster")?
3. ORIGIN IS CONCRETE AND GENTLE: Does origin_or_backstory name a specific, non-violent circumstance — or does it use an abstract or scary-tragic label? REWRITE to name the specific gentle circumstance if abstract or too dark for a kid-friendly story.
4. UNFINISHED BUSINESS IS SPECIFIC: Is unfinished_business a real, specific, harmless goal — not vague ("it just wants to be happy")?
5. TARGETS THIS PROTAGONIST SPECIFICALLY: Does what_it_wants explain why {hero_name} specifically — not a random/interchangeable kid?
6. METHOD OF SPOOKINESS IS UNSETTLING IN HOW IT'S DESCRIBED, HARMLESS IN REALITY: Does method_of_spookiness explain, in specific terms, HOW the creature causes the spooky happenings, and is the underlying mechanism harmless/non-dangerous — but is it ALSO described in a way that reads as genuinely eerie and ambiguous (not inherently cute or jokey, like giggling or silly cartoon sounds)? REWRITE if the description itself gives away friendliness too early.
7. RESOLUTION METHOD IS A CONCRETE ACTION, NOT JUST WORDS: Does resolution_method require {hero_name} to actually DO something physical/tangible — or does it rest on merely telling the creature it's welcome/loved with nothing actually done? REWRITE to add a concrete action if it's currently just reassurance.
8. STAKES ARE CONCRETE AND KID-APPROPRIATE: Is what's at risk specific and NEVER physical danger (an event, a friendship, a tradition, the creature's happiness)?
9. WORRY IS SPECIFIC, NOT GENERIC: Is protagonist_worry a specific, personal detail tied to {hero_name}'s own personality — not a generic fear that could belong to anyone?
10. INCIDENT_1 IS GENUINELY UNSETTLING, NOT CUTE: Is incident_1 small, ambiguous, and easily explained away by an adult, but written to feel genuinely eerie rather than funny or obviously friendly? REWRITE if it reads as a cute gag instead of a real unsettling moment.
11. INCIDENTS 2 AND MIDPOINT REVEAL ONLY FRAGMENTS: Do incident_2 and midpoint_discovery each reveal only a PARTIAL fact with a named source — or do they already give away the complete truth too early? REWRITE to thin it down if so.
11b. REVEALS ARE CONCRETE: Do incident_2 and midpoint_discovery name specific people/places/facts — or an abstract label? REWRITE with specifics if abstract.
11c. REVEALS ARE ACTIONABLE: Does each reveal give {hero_name} something to DO differently — not just sad trivia?
12. ESCALATION IS REAL AND STAYS TENSE: Does each of incident_1 → incident_2 → incident_3 → incident_4 feel MORE unsettling/tense/personal than the last, with no early de-escalation into cuteness before reversal_discovery?
13. INCIDENT_4 USES THE SPECIFIC WORRY AND LANDS AS GENUINELY UNNERVING: Does incident_4 explicitly touch the SAME specific worry named in protagonist_worry, in a way that feels real and unsettling rather than played for laughs?
14. PROTAGONIST AGENCY: Does {hero_name}'s own decision or observation drive the investigation — not unprompted exposition falling into their lap?
15. REAL LOW POINT, LATE: Does low_point genuinely make the mystery feel unsolved/the stakes feel at risk — with no premature resolution?
15b. FALSE THEORY IS EARNED, NOT A STRAWMAN: Is false_theory a genuinely reasonable conclusion given incident_2/midpoint_discovery/incident_3?
15c. FALSE RESOLUTION ATTEMPT IS GENUINE: Is false_resolution_attempt something smart kids would actually try?
15d. REVERSAL IS A REAL SURPRISE: Does reversal_discovery introduce a SPECIFIC new fact that genuinely invalidates false_theory — not just a minor wrinkle?
15e. THE REAL CLIMAX IS TACTICALLY DIFFERENT: Is climax's action meaningfully different from false_resolution_attempt (helping vs. getting rid of), not the same idea repeated nicer?
15f. THE REVELATION DOES NOT END THE HORROR (read carefully — commonly gotten wrong even with a good reversal): Does reversal_discovery explicitly state something STILL unresolved, unstable, or frightening right after the truth comes out (the creature still trapped/fading, a closing window of time, an atmosphere that stays oppressive) — or does the spine imply everything becomes calm and safe the instant the truth is known, leaving climax as a cozy scene with no real tension? REWRITE reversal_discovery/climax if the revelation itself resolves the danger — there must be a genuine final challenge left to do.
15g. FEAR CONFRONTATION IS ACTIVE, IN THE CLIMAX, AND SHOWN THROUGH ACTION: Does fear_confrontation make protagonist_worry a real, in-the-moment obstacle DURING climax's action — {hero_name} having to do the specific thing they're afraid of, or face it directly, as part of completing resolution_method — or is the fear only mentioned earlier (incident_4) with no bearing on the climax itself, or resolved by simply stating {hero_name} isn't scared anymore? REWRITE fear_confrontation to tie directly into the SAME worry from protagonist_worry AND make it a concrete action-level obstacle within climax, not backstory or narration.
16. WARM, CONCRETE CLIMAX (WITH REAL TENSION FIRST): Is climax resolved using resolution_method through a concrete ACTION — or merely by reassuring words with nothing actually done? REWRITE if verbal-only. Also confirm climax itself is NOT immediately calm/cozy — it should read as tense until the action succeeds, per 15f/15g.
17. CLIMAX HAS ROOM TO BREATHE: Does climax read as a real, sustained scene — building tension, then the action, then release — not a single rushed beat and not immediately safe from the start?
18. RESOLUTION IS EXPLICIT: Does resolution clearly state the creature's happy outcome and unfinished_business's fate — not vague?
19. SINGLE-CREATURE CONSISTENCY: Is every incident, reveal, and the climax about the SAME ONE creature?
20. FINAL BEAT EXISTS AND IS WARM/FUN, NOT SPOOKY: Is final_beat a concrete, specific, warm or funny detail for the very last beat — never an unsettling or ambiguous note? REWRITE if missing, generic, or accidentally spooky. Confirm final_beat/resolution come AFTER climax's tension resolves, not before.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. If creature_name, origin_or_backstory, or resolution_method need to change, make sure every dependent field is updated to stay consistent.

Return ONLY valid JSON, the exact same shape as the input (all 24 fields: central_mystery, creature_name, creature_nature, origin_or_backstory, unfinished_business, what_it_wants, method_of_spookiness, resolution_method, stakes, protagonist_worry, incident_1, incident_2, midpoint_discovery, incident_3, incident_4, low_point, false_theory, false_resolution_attempt, reversal_discovery, climax, fear_confrontation, payoff, resolution, final_beat)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2200,
            timeout=75,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Horror-kids blueprint validation response was truncated (hit max_tokens) — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"HORROR-KIDS BLUEPRINT VALIDATED — central_mystery: {revised.get('central_mystery', '')[:100]}")
        return revised
    except Exception as e:
        print(f"⚠️  Horror-kids blueprint validation failed — using original: {e}")
        return blueprint


def generate_sci_fi_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the sci-fi story's structural spine before writing any prose — same
    two-step, lock-then-reveal architecture as generate_horror_blueprint_ai,
    for the same reason: without a blueprint, sci_fi page_count>24 books were
    silently falling into generate_prose_story_ai's generic 8-beat fallback
    scene list, which is comedy-shaped ("something goes hilariously wrong",
    "the funniest, most chaotic moment", ending on a "punchline"/"visual
    gag") -- structurally wrong for sci-fi regardless of genre_guidance text.

    Step 1 locks the technology/premise's true nature and the institution's
    real motive -- the "answer key" -- before a single incident exists,
    mirroring how the horror blueprint locks the entity's backstory first.
    Step 2 designs an escalating investigation built to gradually reveal that
    locked truth, INCLUDING a genuine third-act reversal (a plausible but
    wrong theory the protagonist settles on, a reasonable attempt based on
    it, and a real "we had it wrong" turn) -- baked in from the start rather
    than discovered empirically the way horror's reversal mechanic was,
    since a flat "investigate -> learn the truth -> resolve" arc is exactly
    as flat in sci-fi as it was in horror. Also bakes in horror's other two
    hard-won lessons: resolutions must be earned by a genuinely CONCRETE
    action (not just a decision or a speech), and the technology's internal
    logic must stay consistent throughout ("internal logic — not magic
    dressed as tech" per the existing adult sci-fi content rules). Only
    called for the sci_fi genre.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    client = get_client()

    # ── Step 1: lock the TRUE NATURE of the technology and who controls it
    # before inventing a single incident ──
    solution_prompt = f"""You are a professional science fiction writer. Before designing any incidents or reveals, you must first decide EXACTLY what the central technology/phenomenon really is and who really controls it — the private answer key the story will be built to gradually reveal.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

MANDATORY FIRST STEP — GENUINE SPECULATIVE ELEMENT (read before deciding anything else): The central technology, scientific discovery, world condition, species, phenomenon, or capability MUST go MEANINGFULLY beyond ordinary present-day reality — something that does not exist in the real world today, not merely a real-world technology (an app, a smart speaker, a corporate database) with a shady business practice attached. A privacy-violating smart assistant, a hacked database, or a company spying on employees are all things that are already possible TODAY with existing technology — none of those are science fiction on their own, no matter how dramatic the corporate thriller around them is. The speculative element must be the thing driving the plot, not set dressing on an otherwise ordinary techno-thriller.
  WRONG (not actually speculative): "An AI home assistant that secretly reports your conversations to a rival company." This is just corporate spyware — it could be a true story about a real product today.
  RIGHT (genuinely speculative): "An AI assistant with a direct neural-pattern interface that reads pre-verbal intent from ambient brainwave activity before the user consciously decides anything — and a rival company is using that channel to implant suggestions, not just collect data." The impossible-today capability (reading/influencing pre-conscious intent) is what drives the plot, not just who profits from data.
  Other valid shapes: genuine machine consciousness that has developed its own goals, a device that manipulates probability/luck, memory extraction or transfer, biological/genetic modification granting a real capability, faster-than-light or non-local communication with unintended consequences, a phenomenon that alters perception of time, first contact with something not human. Pick ONE such genuinely speculative capability and make it specific and mechanically explainable.

Decide these fields now. Be concrete and specific everywhere — no vague placeholders like "advanced AI" or "the corporation":
- central_premise: What the sci-fi conceit looks like to {hero_name} AT FIRST — the surface-level premise the story opens with (a new job at a facility, a device that just arrived, a system everyone already trusts). This is what the story opens with.
- technology_name: The specific named technology, system, or phenomenon at the center of the story — a real name/model/designation, never left as "the device" or "the system."
- technology_function: EXACTLY what it does and how, in concrete technical terms with real internal logic — you must be able to explain the mechanism in a sentence a reader could follow. This MUST include the genuinely speculative capability described above — not "it's magic that happens to look futuristic," and not merely an ordinary present-day technology (an app, a smart device, a database) with no impossible-today capability at all.
- technology_origin: Who created it, when, and why — concrete specifics (a company, a government program, a specific scientist's project) — not vague.
- hidden_truth: The single most mind-changing SPECIFIC fact about what the technology/phenomenon ACTUALLY is or ACTUALLY does — something that recontextualizes its true nature, not merely "who is profiting from misusing it." Examples of the right DEPTH: it has developed genuine autonomous goals no one programmed; it doesn't just read data, it is actively altering users' decisions or memories; what looks like an AI is actually powered by something that shouldn't be conscious; its real function is entirely different from its advertised one. This must be a fact about the TECHNOLOGY ITSELF, not just an institutional cover-up story that could apply to any mundane technology.
- central_dilemma: The UNIQUE philosophical, ethical, or experiential question that hidden_truth's specific capability creates for anyone it affects — a question that could ONLY arise from THIS specific speculative capability, not a generic "is this technology good or bad?" Examples of the right SHAPE: if it can nudge subconscious decisions, "how can {hero_name} ever be certain a choice — including the choice to stop it — is truly their own and not something it planted?"; if it extracts/alters memories, "how does {hero_name} know which of their own memories are still real?"; if it manipulates probability, "did {hero_name} actually earn this outcome, or did the technology simply make it inevitable?" This dilemma is NOT set dressing — it must become the thing the climax forces {hero_name} to personally confront, in the moment, while also performing whatever action resolves the plot.
- who_controls_it: The SPECIFIC person or organization with real power over the technology, and their SPECIFIC concrete motive (a named executive protecting a product launch, a government agency hiding a failure, a founder protecting their legacy) — this is a SECONDARY, supporting layer (who is exploiting the technology), not the main reveal — the main reveal is hidden_truth/central_dilemma, about the technology itself.
- what_it_costs: The concrete price of the technology's existence or use — specifically what it takes from people (autonomy, memories, jobs, safety, a specific person's life) — not "it would be bad for society."
- stakes: What is specifically lost if {hero_name} fails to act — concrete (a life, {hero_name}'s freedom, a specific number of people affected) — not "it would be scary." At least part of this must be something that CANNOT be fully undone even if {hero_name} succeeds (real sci-fi endings carry a genuine, lasting cost — not a fully clean, reversible outcome).

Return ONLY valid JSON:
{{"central_premise": "...", "technology_name": "...", "technology_function": "...", "technology_origin": "...", "hidden_truth": "...", "central_dilemma": "...", "who_controls_it": "...", "what_it_costs": "...", "stakes": "..."}}"""

    try:
        solution_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional science fiction story architect. Return only valid JSON."},
                {"role": "user", "content": solution_prompt},
            ],
            temperature=0.8,
            max_tokens=900,
            timeout=60,
        )
        solution = json.loads(solution_response.choices[0].message.content)
        print(f"SCI-FI PREMISE (locked first) — tech: {solution.get('technology_name', '')[:80]}, hidden truth: {solution.get('hidden_truth', '')[:80]}")
    except Exception as e:
        print(f"⚠️  Sci-fi premise generation failed: {e}")
        return {}

    # ── Step 2: design the investigation as a schedule of incidents built to
    # gradually REVEAL the locked truth above, including a genuine reversal ──
    investigation_prompt = f"""You are a professional science fiction writer planning the INVESTIGATION for a {page_count}-panel comic book. The technology's true nature and who really controls it are already decided and LOCKED — your job is to design an escalating story that gradually reveals this exact truth, including a genuine mid-story reversal, never a straight line from suspicion to confirmation.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

THE LOCKED TRUTH (do not change or contradict any part of this):
Central premise (what it looks like at first): {solution.get('central_premise', '')}
Technology: {solution.get('technology_name', '')}
Function: {solution.get('technology_function', '')}
Origin: {solution.get('technology_origin', '')}
Hidden truth: {solution.get('hidden_truth', '')}
Central dilemma (the unique question this specific capability creates — this MUST be what the climax forces {hero_name} to confront): {solution.get('central_dilemma', '')}
Who controls it (and their real motive): {solution.get('who_controls_it', '')}
What it costs: {solution.get('what_it_costs', '')}
Stakes: {solution.get('stakes', '')}

Design the story with these exact fields, every one of them built to gradually reveal — and ultimately resolve — the locked truth above:
- protagonist_goal: A SPECIFIC, concrete, measurable goal {hero_name} has at the start, tied to the technology (fix a malfunction, win a promotion, complete an assignment, escape a facility) — not vague ("understand the technology").
- protagonist_blind_spot: A specific belief or assumption {hero_name} holds that will turn out to be wrong or incomplete — trusts a specific person/institution, believes the technology is purely beneficial, doesn't realize their own role in the system. Tied to their personality/traits above, not generic.
- incident_1: The FIRST sign something is off — small, explainable away (a glitch, an inconsistency, an odd instruction). {hero_name} could still rationalize this. Reveals NOTHING of hidden_truth yet.
- incident_2: A second, stronger sign, PLUS the first fragment of hidden_truth — name the SPECIFIC source (a leaked document, a malfunctioning readout, a whistleblowing colleague, a corrupted log) and the SPECIFIC partial fact it reveals. PROGRESSIVE DISCOVERY REQUIRED: state what small question this fact ANSWERS, and then state the NEW, BIGGER question it immediately RAISES — this fragment must leave {hero_name} (and the reader) with a deeper mystery than before, not a smaller one. Must change what {hero_name} does next.
- midpoint_discovery: A turning-point revelation that reframes {hero_name}'s understanding — connects toward hidden_truth but still doesn't reveal the full picture. State the SPECIFIC source and the SPECIFIC new fact. PROGRESSIVE DISCOVERY REQUIRED: state what question from incident_2 this ANSWERS, and the NEW, BIGGER question it RAISES — each layer must go deeper into what the technology/phenomenon actually is, not sideways into new atmosphere. Must change {hero_name}'s strategy or reveal the danger is worse than believed.
- incident_3: A DIRECT CONSEQUENCE of the technology's use/malfunction that puts {hero_name} or someone they care about in genuine danger — this must be CAUSED by a specific cost of investigating this far (the technology reacts to being investigated, or the institution retaliates) — not an unrelated new danger.
- false_theory: The plausible, well-supported explanation {hero_name} settles on by now — believes it's the complete truth — but which is WRONG or dangerously incomplete. This should be the MUNDANE explanation (a rogue employee, ordinary corporate greed, a technical malfunction) that a reasonable person would land on — reversal_discovery will overturn it with something more fundamentally strange. Must be a reasonable conclusion given incident_2/midpoint_discovery/incident_3, not a strawman.
- false_resolution_attempt: Acting on false_theory, {hero_name} attempts what seems like the obvious solution (report it to an authority, shut down/reprogram/expose it a specific way, confront a specific person believed responsible) — a genuine, competent attempt a smart person would try.
- reversal_discovery: THE FINAL, MOST MIND-CHANGING REVELATION — must substantially change {hero_name}'s (and the reader's) understanding of what the technology/phenomenon actually IS, not just who's exploiting it for money. This is the deepest layer of hidden_truth, fully revealed: the thing that recontextualizes everything (the technology has its own goals no one gave it, what looked like a tool is actually something closer to alive, the "malfunction" was the technology succeeding at its real purpose, {hero_name} themselves has been affected by it in a way they didn't realize). It must ALSO prove false_theory wrong as false_resolution_attempt fails or backfires. State the SPECIFIC source. This must demand a MEANINGFULLY DIFFERENT real climax, not the same attempt repeated harder.
- low_point: The moment {hero_name} is nearly stopped, captured, or discredited — success looks genuinely unlikely, and the technology/institution is at FULL, active danger. This happens LATE, roughly the final 10-15% of the story.
- climax: The REAL, sustained final confrontation — beginning only AFTER reversal_discovery. What must {hero_name} personally, physically DO — at real risk, using a specific established tool/access/knowledge — to destroy, expose, reprogram, or disable the technology (or escape its control)? This must be a CONCRETE PHYSICAL OR TECHNICAL ACTION {hero_name} performs with their own hands/skills — not merely a decision, a speech, or a realization. CRITICALLY, the climax must ALSO directly force {hero_name} to confront central_dilemma in the moment — not resolve as a generic hacking/chase/confrontation sequence that could belong to any techno-thriller regardless of what the technology actually does. State specifically HOW central_dilemma becomes an active obstacle during the action itself (the technology exploits its own capability on {hero_name} mid-climax to create doubt/confusion/a false choice; {hero_name} must find a way to verify their own agency or judgment is intact before they can trust their next move). The technical/physical action is HOW the climax resolves mechanically; central_dilemma is WHY it's actually hard. Begins LATE, roughly the final 10-15% of the story.
- payoff: Confirm the SPECIFIC earlier-established tool, access code, technical knowledge, or object (from incident_2, midpoint_discovery, or incident_3) {hero_name} actually uses to perform the climax's action, and state MECHANICALLY what that action does (severs power, corrupts the control system, physically destroys a component, exposes the data publicly in a way that removes who_controls_it's power over it) — not merely "the truth comes out and everyone agrees to stop."
- resolution: The EXPLICIT, unambiguous outcome — state clearly what happens to the technology (destroyed/disabled/exposed/repurposed), to who_controls_it (held accountable, escapes consequence, doubles down elsewhere), and to {hero_name} (what they gained or lost). Tie this directly to what_it_costs and stakes. THIS MUST NOT BE A FULLY CLEAN, COST-FREE VICTORY — {hero_name} must genuinely lose or permanently sacrifice something specific (not just "feel shaken up") even though they succeeded, OR the technology/its consequences cannot be fully undone even though the immediate threat is stopped. Never end on a vague "things will be different now," and never end on {hero_name} simply being vindicated, promoted, and completely fine.
- final_implication: A SPECIFIC, concrete detail or fact for the very last beat that does NOT contradict the explicit resolution above, but leaves the reader with a genuine lingering question or cost — sci-fi's version of "not everything is fully solved." Examples of the SHAPE (invent your own specific to this story): the same technology is already being deployed elsewhere under a different name, {hero_name} discovers a personal cost they didn't anticipate, a specific detail suggests who_controls_it's network is bigger than what was exposed. Must be concrete, not a vague sense of unease.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
1. GENUINE SPECULATIVE ELEMENT TEST (re-verify here, at the story-design stage): Does every incident/discovery actually engage with the speculative capability locked in technology_function — or has the story quietly become a mundane corporate-thriller plot that happens to be set at a tech company, where the "sci-fi" element is just flavor and could be swapped for an ordinary MacGuffin with no real change? If the incidents would work identically with a non-speculative stand-in (a regular database instead of the speculative tech), REDESIGN so the speculative capability is mechanically load-bearing in the plot.
2. PROGRESSIVE DISCOVERY TEST: For incident_2 and midpoint_discovery, does each one EXPLICITLY answer a question raised by the previous beat while raising a NEW, BIGGER question — or does the mystery stay flat/same-size across both reveals? If a reveal doesn't deepen the mystery, REDESIGN it so understanding genuinely escalates toward something stranger, not just "more evidence of the same thing."
3. FINAL REVELATION CHANGES EVERYTHING TEST: Does reversal_discovery reveal something about the NATURE of the technology/phenomenon itself (not just which human is responsible) that recontextualizes the whole story — or is the "twist" merely "it turns out person X was behind it," which is a mundane thriller twist, not a sci-fi one? If reversal_discovery is only about human culpability with no change in understanding of the technology/phenomenon itself, REDESIGN it to reveal something genuinely new about what the tech/phenomenon actually is or can do.
4. INTERNAL LOGIC TEST: technology_function must be used consistently — nothing in incident_1 through the climax may require the technology to suddenly work differently or gain unexplained new abilities. If the climax needs a new capability, it must be foreshadowed in an earlier incident/discovery.
5. CONCRETE REVELATION TEST: Every reveal (incident_2, midpoint_discovery, reversal_discovery) must name specific people, specific facts, specific mechanisms — never an abstract label like "a conspiracy" or "something sinister" standing in for the actual fact.
6. CONSEQUENCE CHAIN TEST: incident_3 must be a DIRECT CONSEQUENCE of the investigation reaching this point (the technology or institution reacting to being investigated) — not an unrelated new danger introduced from nowhere.
7. GENUINE THIRD-ACT REVERSAL TEST: The story must NOT be a straight line of "suspect the technology -> investigate -> confirm suspicion -> stop it." false_theory must be a real, well-earned MUNDANE belief the evidence supports. false_resolution_attempt must be a genuine, reasonable attempt a smart protagonist would try. reversal_discovery must be a SURPRISING but fairly-clued fact that specifically invalidates or complicates false_theory. The REAL climax must require a MEANINGFULLY DIFFERENT action than false_resolution_attempt.
8. EARNED, CONCRETE CLIMAX TEST (commonly gotten wrong): The climax must be resolved through a CONCRETE PHYSICAL OR TECHNICAL ACTION {hero_name} personally performs — cutting power, planting a virus, physically destroying hardware, rerouting a system, extracting and broadcasting specific data — using the established payoff. It must NOT be resolved merely by {hero_name} making a speech, having a realization, or convincing someone with words alone, unless that specific act of exposure is explicitly the technical mechanism (broadcasting data to a specific network that removes who_controls_it's ability to suppress it) rather than simply "telling the truth so people believe them." WRONG: "{hero_name} exposes the truth in a press conference and the company is forced to shut it down." RIGHT: "{hero_name} uses the access codes from incident_2 to trigger the system's own fail-safe protocol, physically severing its connection to the network — the shutdown happens because of what {hero_name} DID to the system, not because people were persuaded."
8B. THE SPECULATIVE ELEMENT MUST DRIVE THE CLIMAX ITSELF, NOT JUST THE SETUP (read carefully — this is the single most commonly missed rule): Passing rule 8 is not enough on its own. Once hidden_truth/reversal_discovery reveal the technology's true implications, the climax must NOT abandon that concept for a generic action/hacking/chase/corporate-conspiracy confrontation that could belong to any techno-thriller regardless of what the technology actually does. The climax must directly test central_dilemma — the unique question the speculative capability itself creates. WRONG (fails even though it's technically a "concrete physical action" per rule 8): "{hero_name} sneaks into the server room using a stolen badge and types a shutdown command while alarms blare." This is a purely generic hacking climax — it never engages with what makes this technology's premise interesting, and would be identical if the tech were a mundane hacked database instead of a mind-influencing neural interface. RIGHT: "As {hero_name} reaches for the shutdown command, the neural interface exploits its own capability on them in real time, flooding them with a manufactured impulse to stop — and {hero_name} must find a way to distinguish their own genuine will from the implanted one (recalling a specific memory/fact the technology couldn't have fabricated, or physically acting before the doubt can fully take hold) before they can trust their own hand enough to complete the action." The technical action is still what mechanically resolves the plot — but the DIFFICULTY and TENSION of the climax must come from central_dilemma, not from generic external obstacles (guards, alarms, time pressure) alone.
9. LATE-THREAT TEST: The technology/institution must remain a genuine, active danger through low_point and into the REAL climax, with no premature safety. The real climax must not begin until roughly the final 10-15% of the story.
10. AGENCY TEST: {hero_name}'s own decisions and observations must drive incident_2's follow-up, midpoint_discovery, incident_3, and reversal_discovery — not information that simply falls into their lap.
11. NO EASY ENDING TEST: Does resolution include a genuine, specific, lasting cost or sacrifice for {hero_name} — or is it a fully clean win (vindicated, promoted, everyone happy, nothing lost)? A sci-fi ending that costs nothing hasn't earned its stakes: REWRITE resolution to include something real and at least partially irreversible that {hero_name} loses or must live with, even in success.
12. EXPLICIT RESOLUTION, THOUGHT-PROVOKING FINAL BEAT: resolution must clearly state what happened to the technology, who_controls_it, and {hero_name} — no vague "things will never be the same." Separately, final_implication must give the very last beat a genuine, specific lingering detail — a resolution that wraps up too neatly with no final_implication fails this test too.

Return ONLY valid JSON:
{{"protagonist_goal": "...", "protagonist_blind_spot": "...", "incident_1": "...", "incident_2": "...", "midpoint_discovery": "...", "incident_3": "...", "false_theory": "...", "false_resolution_attempt": "...", "reversal_discovery": "...", "low_point": "...", "climax": "...", "payoff": "...", "resolution": "...", "final_implication": "..."}}"""

    try:
        investigation_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional science fiction story architect designing a plot to gradually reveal an already-fixed truth, including a genuine third-act reversal. Return only valid JSON."},
                {"role": "user", "content": investigation_prompt},
            ],
            temperature=0.8,
            max_tokens=2200,
            timeout=60,
        )
        investigation = json.loads(investigation_response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Sci-fi investigation generation failed: {e}")
        return {}

    blueprint = {**solution, **investigation}
    print(f"SCI-FI BLUEPRINT — central_premise: {blueprint.get('central_premise', '')[:100]}")
    return blueprint


def validate_sci_fi_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the sci-fi blueprint against structural criteria BEFORE any prose
    is written, and revises the specific fields that fail. Mirrors
    validate_horror_blueprint_ai's architecture/rationale for the same
    two-step, truth-first schema with a baked-in reversal (see
    generate_sci_fi_blueprint_ai).
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing a SCI-FI STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
1. CONCRETE CENTRAL PREMISE: Could a reader state what the sci-fi conceit looks like at first, in one sentence?
1B. GENUINE SPECULATIVE ELEMENT (check this one first and strictly): Does technology_function describe a capability that goes MEANINGFULLY beyond present-day real-world technology — or is it just an ordinary app/device/database/AI assistant with a shady business practice attached (corporate spyware, a hacked system, a company misusing data)? Those are things that are already possible TODAY and are NOT science fiction on their own. If the "technology" could be swapped for any real product on the market today with zero change to the plot, this FAILS — REWRITE technology_function (and hidden_truth, since it depends on this) to include a genuinely impossible-today capability that actually drives the plot: reading/altering thoughts or memories, genuine machine consciousness with its own goals, probability/matter manipulation, biological/genetic augmentation, non-local or faster-than-light effects, altered perception of time, contact with something not human. Update every dependent field if this changes.
1C. CENTRAL DILEMMA IS SPECIFIC TO THIS CAPABILITY, NOT GENERIC: Does central_dilemma describe a question that could ONLY arise from THIS specific speculative capability — or is it a generic "is this technology good or bad?" / "should we trust corporations with this?" question that could apply to any tech story? REWRITE if generic — it must be a question about {hero_name}'s own certainty, agency, perception, or identity that this exact capability creates (e.g. "how can {hero_name} tell a nudged choice from a genuine one?" for influence tech; "which memories are still real?" for memory-altering tech).
2. TECHNOLOGY IS NAMED AND SPECIFIC: Does technology_name give an actual name/designation (not "the device" / "the system")?
3. FUNCTION HAS REAL INTERNAL LOGIC: Does technology_function explain, in concrete mechanical/technical terms, exactly how it works — not "it's advanced" or magic dressed as tech?
4. ORIGIN IS CONCRETE: Does technology_origin name a specific creator/institution and reason — not vague?
5. HIDDEN TRUTH IS ABOUT THE TECHNOLOGY'S NATURE, NOT JUST HUMAN MISUSE: Does hidden_truth name a SPECIFIC concealed fact about what the technology/phenomenon actually IS or DOES — or is it really just "a company is secretly using it for profit/spying," which is a mundane institutional-cover-up story, not a fact about the technology's true nature? REWRITE if hidden_truth never says anything surprising about the TECH itself.
6. WHO CONTROLS IT IS A SECONDARY LAYER WITH A REAL, SPECIFIC MOTIVE: Does who_controls_it name a specific person/organization with a concrete motive — not a vague "the bad guys" — and does it stay a SUPPORTING layer rather than being treated as the main reveal (hidden_truth/reversal_discovery should be the main reveal, about the technology)?
7. COST IS CONCRETE: Does what_it_costs name a specific real price — not "it would be bad for society"?
8. SPECIFIC, PARTLY IRREVERSIBLE STAKES: Is what's lost on failure concrete, and does at least part of it remain un-undoable even if {hero_name} succeeds — not "it would be scary" and not something fully reversible?
9. GOAL IS CONCRETE AND MEASURABLE: Is protagonist_goal specific enough that a reader could tell whether {hero_name} succeeded — not vague ("understand what's happening")?
10. BLIND SPOT IS SPECIFIC, NOT GENERIC: Is protagonist_blind_spot a specific belief/assumption tied to {hero_name}'s own personality/traits — not a generic "trusts too easily"?
11. INCIDENT_1 REVEALS NOTHING: Is incident_1 genuinely small and explainable away — not already hinting at hidden_truth?
12. INCIDENT_2 AND MIDPOINT REVEAL ONLY FRAGMENTS, CONCRETELY: Do incident_2 and midpoint_discovery each reveal only a PARTIAL, SPECIFIC fact (named source, named fact) — not an abstract label, and not the complete picture too early? If the full picture is already assembled by midpoint_discovery, REWRITE to thin it down.
12B. PROGRESSIVE DISCOVERY, NOT FLAT REVEALS: Does each of incident_2 and midpoint_discovery explicitly answer a question raised by the previous beat while opening a NEW, BIGGER question — does the mystery genuinely deepen from one reveal to the next, escalating toward something stranger? Or do the reveals just add more evidence of the same-sized mystery without deepening it? REWRITE if flat.
13. REVEALS ARE ACTIONABLE: Does each reveal give {hero_name} something to DO differently — not just atmosphere or dread?
14. CONSEQUENCE CHAIN: Is incident_3 a DIRECT CONSEQUENCE of the investigation reaching this point (the technology/institution reacting) — or an unrelated new danger? REWRITE if unrelated.
15. FALSE THEORY IS EARNED, PLAUSIBLE, AND MUNDANE: Is false_theory a genuinely reasonable, relatively ORDINARY conclusion (a rogue employee, corporate greed) given incident_2/midpoint_discovery/incident_3 — not a strawman — that reversal_discovery will overturn with something stranger?
16. FALSE RESOLUTION ATTEMPT IS A GENUINE, REASONABLE TRY: Is false_resolution_attempt something a smart, competent protagonist would actually attempt?
17. REVERSAL CHANGES UNDERSTANDING OF THE TECHNOLOGY ITSELF, NOT JUST WHO'S RESPONSIBLE (read carefully — this is commonly gotten wrong): Does reversal_discovery reveal something genuinely new about the NATURE of the technology/phenomenon — or is the "twist" merely "it turns out person X was behind it," a mundane thriller reveal about human culpability with no change in what the tech actually is? A reversal that's purely "the villain is someone else" without also deepening what the technology/phenomenon IS fails this test: REWRITE reversal_discovery so it reveals something new and stranger about the technology itself, not just a different human name attached to the same mundane misuse.
18. THE REAL CLIMAX IS TACTICALLY DIFFERENT FROM THE FALSE ATTEMPT: Compare climax to false_resolution_attempt — is the real climax meaningfully different (informed by reversal_discovery), or just the same action repeated harder? REWRITE if they're basically the same.
18B. CLIMAX ACTUALLY TESTS CENTRAL_DILEMMA (read carefully — this is commonly gotten wrong even when everything else passes): Does climax explicitly describe central_dilemma becoming an active obstacle DURING the physical/technical action — not just resolved by the action, but genuinely complicating it in the moment? A climax that is purely external (guards, alarms, a chase, a countdown) with no engagement with the dilemma FAILS even if it has a concrete physical/technical action, because it could belong to any generic techno-thriller regardless of what this technology actually does. REWRITE climax to state specifically HOW central_dilemma interferes with {hero_name} completing the action (the technology exploits its own capability on {hero_name} mid-climax to create doubt about their own agency/perception/memory, and {hero_name} must find a concrete way to verify themselves before they can trust their next move).
19. REAL LOW POINT, LATE AND ACTIVE: Does low_point genuinely make success look unlikely, with the technology/institution still at FULL active danger, no premature safety?
20. EARNED, CONCRETE, PHYSICAL/TECHNICAL CLIMAX: Is the REAL climax resolved through a CONCRETE PHYSICAL OR TECHNICAL ACTION {hero_name} personally performs (cutting power, planting a virus, destroying hardware, rerouting a system) using the established payoff — or merely by a speech, a realization, or convincing someone with words alone? If climax is resolved by talking/persuasion alone with no concrete technical/physical action, REWRITE climax/payoff to add one.
21. CLIMAX HAS ROOM TO BREATHE: Does climax read as a sustained confrontation, not a single quick beat immediately followed by resolution?
22. RESOLUTION IS EXPLICIT: Does resolution clearly state what happened to the technology, to who_controls_it, and to {hero_name} — not a vague "things will be different"? REWRITE if vague.
22B. NO EASY ENDING: Does resolution include a genuine, specific, at-least-partly-irreversible cost or sacrifice for {hero_name} — or is it a fully clean win (vindicated, promoted, nothing lost, everyone happy)? REWRITE resolution to add a real, lasting cost if the ending currently costs {hero_name} nothing.
23. FINAL IMPLICATION EXISTS AND IS SPECIFIC: Is final_implication a concrete, specific lingering detail for the very last beat — not vague, and not contradicting the explicit resolution? If missing or generic, REWRITE it with a specific detail.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. If technology_name, technology_function, hidden_truth, or who_controls_it need to change, make sure every dependent field is updated to stay consistent.

Return ONLY valid JSON, the exact same shape as the input (all 23 fields: central_premise, technology_name, technology_function, technology_origin, hidden_truth, central_dilemma, who_controls_it, what_it_costs, stakes, protagonist_goal, protagonist_blind_spot, incident_1, incident_2, midpoint_discovery, incident_3, false_theory, false_resolution_attempt, reversal_discovery, low_point, climax, payoff, resolution, final_implication)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
            timeout=75,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Sci-fi blueprint validation response was truncated (hit max_tokens) — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"SCI-FI BLUEPRINT VALIDATED — central_premise: {revised.get('central_premise', '')[:100]}")
        return revised
    except Exception as e:
        print(f"⚠️  Sci-fi blueprint validation failed — using original: {e}")
        return blueprint


def _strip_meta_preamble(text: str) -> str:
    """
    Defensive backstop for check-and-revise passes that are instructed to
    return ONLY the story ("Do NOT explain your changes — just return the
    complete story") but sometimes prepend a meta-comment anyway, e.g.
    "The story requires a rewrite to align with the intended spine. Here's
    the revised version:\n\n---\n\n<actual story>". The prompt instruction
    already exists in every one of these passes and still got violated once,
    so this strips it in code rather than trusting the instruction alone.
    Only strips when a "---" separator line appears early (within the first
    400 characters) -- legitimate story prose doesn't open with a markdown
    horizontal rule, so this is safe against false positives.
    """
    marker = "\n---\n"
    idx = text.find(marker)
    if idx != -1 and idx < 400:
        stripped = text[idx + len(marker):].strip()
        if stripped:
            return stripped
    return text


def check_story_pacing_ai(
    prose_story: str,
    hero_name: str,
    genre: str = "adventure",
    page_count: int = 24,
) -> str:
    """
    Review the prose story for structural pacing problems.
    Returns the revised prose if fixes are needed, or the original if it passes.
    """
    resolution_budget = max(2, page_count // 10)

    if genre == "comedy":
        pacing_rules = f"""COMEDY PACING — STRICT RULES (apply all of these, no exceptions):

PERCENTAGE MATH (calculate, do not estimate):
- Find the CLIMAX — the single moment of peak chaos
- Estimate what percentage of the story comes AFTER the climax
- If more than 25% of the story comes after the climax: REWRITE REQUIRED — no exceptions
- After the climax, you are allowed at most {resolution_budget} short scenes before the punchline

LOCATION RULE (critical):
- Characters must remain in the PRIMARY SETTING for at least 80% of the story
  (for {page_count} panels, that means staying until at least panel {int(page_count * 0.8)})
- If characters leave the main location before that point and the rest of the story is them walking and chatting outside: REWRITE REQUIRED
- Walking-and-talking epilogues fill panels but contain zero comedy — cut all of them

ENDING RULE:
- The FINAL beat must be a specific punchline that CONNECTS BACK TO THE ORIGINAL GOAL of the story
- The punchline must reference the SPECIFIC thing that happened, not a generic feeling about it

BANNED ENDING PATTERNS — if the story ends with ANY of these, it must be rewritten:
- "she/he realized that her/his true [talent/knack/gift] was..."
- "they shared a knowing glance"
- "as they left, [X] echoed behind them"
- "all it took was..."
- "some victories are won with..."
- "we make such a great team"
- "I can't wait to do this again"
- "that was such a great day"
- "next time we'll..."
- "and that was the day [character] learned/realized..."
- Any sentence about what the character learned, realized, or now understands about themselves
- Any arm-around-shoulder warm moment as a final beat
- Any sentence that could appear as the ending of a completely different story and still make sense

APPROVAL THRESHOLD — be strict. Only approve if ALL THREE are true:
1. Climax falls after the 65% mark but before the 80% mark
2. Post-climax material is 25% or less of the story
3. Story ends on a specific punchline connected to the original goal — NOT warm wind-down dialogue
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    elif genre == "horror_adult":
        threat_active_until = int(page_count * 0.85)
        pacing_rules = f"""HORROR PACING — STRICT RULES (apply all of these, no exceptions):

SINGLE ANTAGONIST CHECK (critical — read the whole story before judging this):
- Identify the ONE supernatural threat/antagonist established earliest in the story
- Does EVERY eerie incident, every clue, and the final confrontation all trace back to that SAME entity?
- If a second, different, previously-unestablished antagonist (a new entity, a cult, a different spirit) appears anywhere — especially near the climax — this is a structural error: REWRITE REQUIRED. Merge it into the one established threat, or cut it entirely.

LATE-THREAT CHECK (critical — this is the most common failure):
- The entity must remain a genuine, ACTIVE danger all the way through roughly the {threat_active_until}th panel-equivalent point of {page_count} (about 85% through the story) — no permanent safety or resolution before that point.
- If the story reaches a permanent-feeling safety/resolution well before that mark and then coasts with several more scenes of calm aftermath: REWRITE REQUIRED — extend the danger later, or compress earlier setup so the climax naturally lands late.

EXPLICIT RESOLUTION, UNSETTLED TONE CHECK (critical):
- Does the story's final paragraphs explicitly state the outcome — the threat is defeated/banished/satisfied, OR the protagonist escapes, OR the protagonist is caught/consumed? If the story ends on an open question, an ambiguous implication about WHAT HAPPENED, silence, or a scene that trails off without stating what happened: REWRITE REQUIRED.
- Separately: does the VERY LAST beat include one specific, concrete unsettling detail, ambiguity, or callback (a final "sting") that keeps the tone unresolved even though the outcome itself is clear — or does the story end on complete, cozy, unqualified safety with nothing lingering? A resolution can be 100% explicit about what happened and STILL end on an unsettling final image. If there's no sting at all: REWRITE REQUIRED — add one specific unsettling detail as the very last beat, without contradicting the stated outcome.

PACING:
- The climax should fall around the 75-85% mark in the story
- Resolution (post-climax) should occupy roughly 10-15% of the story — brief, but enough room for the explicit outcome AND the final sting
- Every post-climax scene must move toward the stated ending — no filler
- If the resolution section is bloated, cut it to the essential beats

APPROVAL THRESHOLD — be strict. Only approve if ALL of these are true:
1. Exactly one antagonist/threat is maintained for the entire story
2. The threat remains active until roughly the 85% mark — no premature safety
3. The final paragraphs explicitly state the outcome
4. The very last beat includes a specific unsettling sting, not complete cozy safety
5. Climax falls between 70-88% AND post-climax is under 15%
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    elif genre == "adventure":
        safety_panel = int(page_count * 0.90)
        climax_start_low = int(page_count * 0.75)
        climax_start_high = int(page_count * 0.87)
        pacing_rules = f"""ADVENTURE PACING — STRICT RULES (apply all of these, no exceptions):

LATE-CLIMAX CHECK (critical — this is the most common failure in adventure stories):
- The climax should BEGIN roughly between the {climax_start_low}th and {climax_start_high}th panel out of {page_count} — a hard, dangerous final test, not a calm arrival.
- Safety or final victory must generally NOT be secured before approximately panel {safety_panel} of {page_count}. Characters should still be actively in danger, still working the problem, right up until near the very end — not settled safely by a fireplace, in a cab, or celebrating with several scenes still left to go.
- If the story reaches safety/resolution well before the {safety_panel}-panel mark and then just coasts (extra scenes of relief, reflection, or travel home with no remaining danger), that is wasted space that should have been climax or escalation instead: REWRITE REQUIRED — either extend the danger/climax later into the story, or compress the early setup so the climax naturally lands later.

CONSEQUENCE-CHAIN CHECK (critical — read the whole story before judging this):
- Are the story's obstacles connected as ONE escalating chain — each one caused by a specific cost of surviving the last (an injury, a lost resource, lost time, a broken tool) — or do they read as independent episodes that just happen to occur in sequence (a storm, then unrelated creature trouble, then an unrelated supply problem, each cleanly resolved before the next begins)?
- If the obstacles are independent rather than chained: REWRITE REQUIRED — rewrite the obstacles so each one is explicitly caused by what the previous one cost the characters, forming one continuous escalating crisis rather than several small ones stacked side by side.

PACING:
- Post-climax resolution should occupy at most 10% of the story — a few short paragraphs, not several scenes
- Every post-climax beat must move toward the stated ending — no lingering, no filler celebration

APPROVAL THRESHOLD — be strict. Only approve if ALL of these are true:
1. The climax begins between roughly the 70% and 88% mark, not earlier
2. Safety/final victory is not secured before roughly the 85% mark
3. The obstacles form one connected consequence chain, not independent episodes
4. Post-climax material is 10% or less of the story
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    elif genre == "sci_fi":
        safety_panel = int(page_count * 0.90)
        climax_start_low = int(page_count * 0.75)
        climax_start_high = int(page_count * 0.87)
        pacing_rules = f"""SCI-FI PACING — STRICT RULES (apply all of these, no exceptions):

LATE-CLIMAX CHECK (critical — this is the most common failure):
- The climax should BEGIN roughly between the {climax_start_low}th and {climax_start_high}th panel-equivalent point out of {page_count} — a hard, dangerous final confrontation, not a calm decision scene.
- Safety must generally NOT be secured before approximately the {safety_panel}th panel-equivalent point. The technology/institution should still be actively dangerous right up until near the very end — not defused with several scenes of calm reflection still left to go.

GENUINE REVERSAL CHECK (critical — read the whole story before judging this):
- Does the protagonist settle on a plausible-but-wrong theory, act on it with a genuine attempt, and then hit a real "we had it wrong" discovery that changes the plan — or does the story go in a straight line from suspicion to confirmation to stopping the technology, with no false attempt and no reversal?
- If there's no genuine reversal: REWRITE REQUIRED — insert a real false theory, a reasonable failed attempt based on it, and a surprising discovery that overturns it before the real climax.

CONCRETE-RESOLUTION CHECK (critical):
- Is the climax resolved through a specific PHYSICAL OR TECHNICAL ACTION the protagonist personally performs (cutting power, planting code, destroying hardware, rerouting a system) — or does it resolve through a speech, a realization, or "exposing the truth" so people simply agree to stop?
- If the resolution is verbal/persuasion-only with no concrete technical/physical action: REWRITE REQUIRED — add the specific action that is what actually disables/destroys/exposes the technology.

PACING:
- Post-climax resolution should occupy at most 10-15% of the story
- The final beat should include one specific, concrete, thought-provoking detail — not a resolution that wraps up too neatly with nothing left to consider

APPROVAL THRESHOLD — be strict. Only approve if ALL of these are true:
1. The climax begins between roughly the 70% and 88% mark, not earlier
2. Safety is not secured before roughly the 85% mark
3. The story contains a genuine false-theory-then-reversal beat, not a straight line
4. The climax is resolved by a concrete physical/technical action, not persuasion/speech alone
5. Post-climax material is 15% or less of the story
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    elif genre == "fantasy":
        safety_panel = int(page_count * 0.90)
        climax_start_low = int(page_count * 0.75)
        climax_start_high = int(page_count * 0.87)
        pacing_rules = f"""FANTASY PACING — STRICT RULES (apply all of these, no exceptions):

LATE-CLIMAX CHECK (critical — this is the most common failure):
- The climax should BEGIN roughly between the {climax_start_low}th and {climax_start_high}th panel-equivalent point out of {page_count} — a hard, dangerous final confrontation, not a calm decision scene.
- Safety must generally NOT be secured before approximately the {safety_panel}th panel-equivalent point. The threat should still be actively dangerous right up until near the very end.

GENUINE REVERSAL CHECK (critical — read the whole story before judging this):
- Does the protagonist settle on a plausible-but-wrong theory, act on it with a genuine attempt, and then hit a real "we had it wrong" discovery that changes the plan — or does the story go in a straight line from suspicion to confirmation to defeating the threat, with no false attempt and no reversal?
- If there's no genuine reversal: REWRITE REQUIRED — insert a real false theory, a reasonable failed attempt based on it, and a surprising discovery that overturns it before the real climax.

MAGIC-COST CHECK (critical — the single most commonly missed rule in this genre):
- Is the climax resolved using the story's established magic system through its already-known mechanism — or does a new, unforeshadowed power suddenly appear to save the day?
- Does the protagonist genuinely PAY the magic system's established cost, in the moment, as an unavoidable part of winning — or does the story let them dodge it via a convenient loophole or one-time exception?
- If the climax invents a new convenient power, or if the established cost is avoided: REWRITE REQUIRED — use the established mechanism, and make the protagonist pay the real, specific price at the moment of victory.

PACING:
- Post-climax resolution should occupy at most 10-15% of the story
- The final beat should include one specific, concrete, thought-provoking detail — not a resolution that wraps up too neatly with nothing left to consider

APPROVAL THRESHOLD — be strict. Only approve if ALL of these are true:
1. The climax begins between roughly the 70% and 88% mark, not earlier
2. Safety is not secured before roughly the 85% mark
3. The story contains a genuine false-theory-then-reversal beat, not a straight line
4. The climax uses the established magic mechanism, not a new convenient power
5. The magic system's established cost is genuinely paid at the climax, not dodged
6. Post-climax material is 15% or less of the story
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    elif genre == "horror":
        mystery_active_until = int(page_count * 0.85)
        pacing_rules = f"""KID-FRIENDLY HORROR PACING — STRICT RULES (apply all of these, no exceptions):

CONTENT CHECK (critical — check this first):
- Does any part of the story — especially the later incidents or the climax — describe real violence, injury, or genuine physical danger? This must stay non-violent and non-graphic throughout. If anything crosses into real danger: REWRITE REQUIRED — soften it into something non-graphic while keeping the same beat.

SUSPENSE PRESERVATION CHECK (critical — read the early incidents carefully; this is the most commonly missed check, and is DIFFERENT from the content check above):
- "Kid-friendly" limits violence and graphic content — it does NOT mean the story should read as cute, silly, or obviously friendly from the start. Do the early incidents (roughly the first half of the story) read as genuinely unsettling and ambiguous — real suspense about whether the presence is dangerous — or do they undercut all suspense with giggling, cute cartoon sound effects, or obviously harmless framing from the very first strange occurrence?
- If the story is content-safe but tonally soft/cute throughout (no real suspense ever built): REWRITE REQUIRED — rewrite the early incidents to feel genuinely eerie and ambiguous (unexplained sounds, things moved with no one there, a sense of being watched) while keeping the events themselves non-violent. The creature's harmlessness should be a secret the reader learns late, not something obvious from scene one.

SINGLE CREATURE CHECK:
- Is every strange incident, clue, and the resolution all about the ONE creature established earliest in the story? If a second, different creature appears anywhere: REWRITE REQUIRED — merge it into the one established creature, or cut it.

LATE-MYSTERY CHECK (critical — the most common failure):
- The mystery/misunderstanding about the creature must remain unresolved all the way through roughly the {mystery_active_until}th panel-equivalent point of {page_count} (about 85% through) — no early explanation that deflates the story.
- If the story explains everything well before that mark and then coasts with several more calm scenes: REWRITE REQUIRED — extend the mystery later, or compress earlier setup so the resolution naturally lands late.

GENUINE REVERSAL CHECK (critical — read the whole story before judging this):
- Does the protagonist settle on a plausible-but-wrong belief that the creature is mean, act on it with a genuine attempt to get rid of it, and then hit a real "we had it wrong" discovery — or does the story go in a straight line from spooky thing to friendly explanation with no false attempt and no reversal?
- If there's no genuine reversal: REWRITE REQUIRED — insert a real false theory, a reasonable failed "get rid of it" attempt, and a surprising discovery that overturns it before the real resolution.

REVELATION-DOESN'T-END-THE-HORROR CHECK (critical — commonly gotten wrong even with a good reversal above):
- Right after the "we had it wrong" revelation lands, does the story keep something genuinely tense, unresolved, or frightening in play (the creature still trapped/fading, a closing window of time, an oppressive atmosphere) through a real final challenge — or does the story turn calm and safe the instant the truth is known, making the climax a low-tension scene of just being nice?
- If the revelation itself defuses all danger, leaving no real final challenge: REWRITE REQUIRED — insert genuine tension between the revelation and the resolution; the climax must be a suspenseful sequence the protagonist actively has to get through, not a foregone conclusion.

FEAR-CONFRONTATION CHECK (critical):
- Does the protagonist's established personal fear/worry become an ACTIVE, in-the-moment obstacle DURING the climax — do they have to actually do the specific thing they're afraid of as part of resolving the situation — or is the fear only mentioned earlier with no bearing on the climax, or "resolved" by narration simply stating they're no longer scared?
- If the climax doesn't force the protagonist to concretely act through/despite their established fear: REWRITE REQUIRED — the climax's action should require confronting that specific fear, and growth should be shown through what they DO, not stated as fact.

DRAMATIZED-SCENE CHECK (critical — the most common way an otherwise-correct climax still falls flat, and can slip past every other check above):
- Does the climax include at least one discrete, decisive beat — a specific action at a specific instant, a specific line of dialogue, a single sharp turning point the reader can picture happening at one precise moment — or is it several paragraphs of general activity summary ("gathered the pieces... worked on it... each careful stitch was a testament to resilience...") with no scene-level moment, reading as backstory summary rather than a climax even though fear and tension are mentioned throughout?
- If the climax is a summarized montage of a process rather than a staged, moment-to-moment scene: REWRITE REQUIRED — rewrite it so the reader can picture exactly what the protagonist's hands do at one specific instant and what happens right after because of it.

CONCRETE KIND-ACTION CHECK (critical):
- Is the resolution achieved through a specific CONCRETE ACTION the protagonist personally performs (returning an object, fixing/building something, completing a specific task) — or does it resolve through reassuring words alone with nothing actually done?
- If the resolution is words-only with no concrete action: REWRITE REQUIRED — add the specific action that is what actually resolves things.

WARM ENDING CHECK:
- Does the story end on complete, cozy, fun safety with a specific warm/silly final detail — or does it end vaguely, or accidentally leave something unsettling? A kid-friendly horror story should NEVER end on a lingering scare. If the ending is vague or still feels spooky: REWRITE REQUIRED — add a specific, concrete, warm/fun final beat.

PACING:
- The resolution should begin around the 75-85% mark
- Post-resolution material should occupy roughly 10-15% of the story — brief, but enough room for the explicit outcome AND the warm final beat

APPROVAL THRESHOLD — be strict. Only approve if ALL of these are true:
1. Nothing in the story crosses into real violence, injury, or genuine danger
2. The early incidents read as genuinely unsettling/suspenseful, not cute or obviously friendly
3. Exactly one creature is maintained for the entire story
4. The mystery remains unresolved until roughly the 85% mark — no premature explanation
5. The story contains a genuine false-theory-then-reversal beat, not a straight line
6. The revelation does not immediately end the horror — a genuine tense final challenge follows it
7. The protagonist's established fear becomes an active obstacle during the climax, shown through action
8. The climax is a dramatized scene with a discrete decisive beat, not a summarized activity montage
9. The resolution is achieved through a concrete action, not reassurance alone
10. The story ends warmly and safely, with a specific fun/warm final detail — nothing left spooky
If even ONE fails: rewrite. Default to rewriting when in doubt."""
    else:
        # Character/setting/unresolved-setup continuity is handled by the dedicated
        # check_story_continuity_ai pass, not here — keeping this focused purely on
        # timing avoids diluting either check with the other's concerns.
        pacing_rules = f"""PACING RULES:
- The climax should fall around the 70-75% mark in the story
- Resolution (post-climax) should occupy roughly 15-20% of the story
- Every post-climax scene must move toward the ending — no filler conversation
- The ending must be specific and earned — not "everything is fine now"
- If the resolution section is bloated, cut it to the essential beats

APPROVAL THRESHOLD: only approve if the climax falls between 60-80% AND post-climax is under 25%.
Default to rewriting when in doubt."""

    if genre == "comedy":
        rewrite_rules = f"""REWRITE RULES (when rewriting):
- Keep everything BEFORE the climax IDENTICAL — word for word
- Only rewrite from the climax point onward
- Post-climax section must be at most 3-4 short paragraphs
- Characters should still be in (or just leaving) the primary setting at the end
- Before the punchline, {hero_name} must take a real corrective ACTION that actually fixes/repairs the specific external damage the chaos caused — not just everyone deciding the disaster was secretly fine. If the existing resolution is only a change of attitude with nothing actually fixed, add the missing concrete fix.
- End with a concrete VISUAL gag (a prop, object, or minor character's action) that connects to what the character was originally trying to accomplish — not only a spoken line
- Do NOT explain your changes — just return the complete story"""
        final_sentence_rule = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SENTENCE RULE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The last thing on the page must be a concrete VISUAL beat the reader can picture — a prop, object, or minor/named character's small action — not a spoken line reflecting on how things turned out. Either a third party who already appeared earlier states a concrete competition result AND a visual gag lands in that same moment, or — if there's no natural judge/announcer for this story — the ending IS the callback payoff itself (a minor character does something with an established prop one last time).
The hero does NOT react to it. The hero does NOT get a final quip. The hero does NOT walk away with a grin.
Before submitting, read the last two paragraphs. Delete any that contain:
- The hero reacting to, responding to, or quipping after the punchline lands
- Any sentence starting with "And with that..." or "And so..."
- Any sentence about what the day meant, what they learned, or what they knew
- Any sentence about laughter echoing, memories being made, or stories being retold
- Any warm summary that could close any story ever written
- The hero linking arms / strutting out / walking away after the punchline
- Everyone simply deciding the disaster was secretly good/fun/fine, with nothing actually fixed beforehand
The story ends the moment the visual gag lands. Not one word after.
"""
    elif genre == "horror_adult":
        rewrite_rules = """REWRITE RULES (when rewriting):
- Keep everything BEFORE the climax IDENTICAL — word for word
- Only rewrite from the climax point onward
- If a second/different antagonist was introduced, remove it — resolve the story using ONLY the ONE threat established earlier in the story
- If the threat became safely resolved too early, with several calm scenes left over: extend the danger later, or compress the resolution so the climax lands late
- The final paragraphs MUST explicitly state the outcome: the threat is defeated/banished/satisfied, the protagonist escapes, or the protagonist is caught/consumed — pick one and show it happening
- Then add ONE final, specific, unsettling detail as the very last beat (an object out of place, a lingering trace, an ambiguous detail) that does not contradict the stated outcome but keeps the ending from feeling completely safe
- Do NOT explain your changes — just return the complete story"""
        final_sentence_rule = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SENTENCE RULE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The story's last paragraphs must state, explicitly and unambiguously, what happened to the protagonist and to the threat. A reader finishing the last line must know the outcome — never end on an open question, a vague implication, or the threat simply "vanishing" with no stated result.
Then, as the true final beat, include one specific, concrete unsettling detail, ambiguity, or callback — the ending should not read as completely safe and cozy even though the outcome is clear. Do not let this final detail contradict or undo the stated outcome — it lingers alongside it, not instead of it.
Before submitting, check the last paragraph: if it reads like the story just stopped rather than ended, rewrite it to state the outcome plainly. If it reads as fully safe with nothing left unsettled, add the final sting.
"""
    elif genre == "horror":
        rewrite_rules = """REWRITE RULES (when rewriting):
- Keep everything BEFORE the reversal/revelation IDENTICAL — word for word
- If ONLY the resolution/ending checks failed (warm ending, explicit outcome, concrete kind action) and the climax itself already stays tense with the fear confrontation intact: only rewrite from the resolution point onward
- If the REVELATION-DOESN'T-END-THE-HORROR, FEAR-CONFRONTATION, or DRAMATIZED-SCENE checks failed: rewrite starting from the revelation/climax onward instead — insert genuine tension between the revelation and the ending (the creature still trapped/fading, a closing window of time, an oppressive atmosphere), make the protagonist's established fear an active, in-the-moment obstacle they must act through during the climax shown through what they DO, and stage the climax as a real-time scene with at least one discrete, decisive beat (a specific action at a specific instant, a specific line of dialogue) rather than a summarized montage of a process ("gathered the pieces... worked on it... each stitch was a testament to...")
- If a second/different creature was introduced, remove it — resolve the story using ONLY the ONE creature established earlier in the story
- If any moment crosses into real violence, injury, or genuine danger, soften it into something non-graphic while keeping the same beat
- If the mystery became resolved too early, with several calm scenes left over: extend the mystery later, or compress the resolution so it lands late
- The resolution MUST be achieved through a concrete ACTION the protagonist performs — not reassuring words alone. If it's currently words-only, add the missing concrete action
- The final paragraphs MUST explicitly state the outcome: the creature is happy/at peace, and its unfinished business is resolved
- Then add ONE final, specific, warm or fun detail as the very last beat — the ending must feel completely safe, cozy, and fun, with nothing spooky left lingering. This warmth belongs ONLY after the climax's tension resolves — do not let it bleed backward into the climax itself
- Do NOT explain your changes — just return the complete story"""
        final_sentence_rule = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SENTENCE RULE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The story's last paragraphs must state, explicitly and warmly, what happened to the creature and to the protagonist. A reader finishing the last line must feel completely safe and happy — never end on an open question, a vague implication, or anything that still feels spooky or unresolved.
Then, as the true final beat, include one specific, concrete, warm or fun detail (a joke, a cute callback, a heartwarming image) — the OPPOSITE of horror's "unsettling sting." The ending should read as completely safe and cozy.
Before submitting, check the last paragraph: if it reads like the story just stopped, or if anything still feels spooky/unresolved, rewrite it to land warmly and specifically.
"""
    else:
        rewrite_rules = """REWRITE RULES (when rewriting):
- Keep everything BEFORE the climax IDENTICAL — word for word
- Only rewrite from the climax point onward
- Post-climax section must be at most 3-4 short paragraphs
- The ending must be specific and earned, clearly resolving the story's central conflict
- Do NOT explain your changes — just return the complete story"""
        final_sentence_rule = ""

    prompt = f"""You are a story editor reviewing a prose story about {hero_name} for pacing problems before it is adapted into a {page_count}-panel comic.

STORY TO REVIEW:
{prose_story}

YOUR TASK:
1. Find the CLIMAX — the moment of peak tension, chaos, or conflict
2. Calculate: what percentage of the story (by word count) comes AFTER the climax?
3. Check the location: do characters leave the primary setting early and fill the rest with walking/talking?
4. Check the ending: is it clear, specific, and fully resolved — or vague/incomplete/generic?

{pacing_rules}

DECISION:
- Only reply "APPROVED: " (followed by the original story verbatim) if ALL approval criteria above are met
- Default to rewriting. A borderline case should always be rewritten, not approved.

{rewrite_rules}
{final_sentence_rule}
Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional story editor. Fix pacing without rewriting the story voice."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Story pacing check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Story pacing check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Story pacing check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Story pacing check: REVISED — resolution tightened")
            return result
    except Exception as e:
        print(f"⚠️  Pacing check failed — using original prose: {e}")
        return prose_story


def check_story_continuity_ai(
    prose_story: str,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    A single-purpose continuity pass, deliberately kept separate from pacing/timing
    review so it isn't competing with other instructions for attention. Checks only
    three things: every named character gets referenced again, the story stays in
    its one primary setting, and any established mystery/incident/advantage gets
    resolved. Returns the revised prose if fixes are needed, or the original if it
    passes.
    """
    prompt = f"""You are a continuity editor. Your ONLY job is checking THREE things in this story about {hero_name} — ignore pacing, word choice, and everything else.

STORY TO REVIEW:
{prose_story}

CHECK 1 — CHARACTER CONTINUITY:
List every named character in the story (besides {hero_name}). For each one: do they appear or get referenced again after their introduction, all the way through to the end — or do they show up once and then simply vanish with no further mention? A character introduced with any narrative weight (a rival, a friend, a mentor, a family member) must not just disappear.

CHECK 2 — SETTING CONSISTENCY:
Identify the ONE primary setting established at the start of the story. Does the ENTIRE story — including the ending — stay in that same setting? Or does it move to a different, distinct location at any point (going home, going out afterward to celebrate somewhere else, a scene change to a new place)? The story must stay in its one primary setting from beginning to end.

CHECK 3 — UNRESOLVED SETUPS:
Identify any mystery, unexplained incident, or notable object/advantage introduced with apparent narrative weight (something went wrong and nobody knows why, an item was stolen/swapped, a threat was made). Is it explained or resolved by the end — or just left hanging?

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all three checks pass completely.
- If ANY check fails, rewrite the story to fix it:
  - Dropped character: add them back in with at least one brief line or beat before the end — a sentence is enough, it doesn't need a new scene.
  - Setting violation: cut or relocate the offending scene(s) so the entire story stays in the one primary setting established at the start.
  - Unresolved setup: add a brief line explaining or resolving it — it doesn't need its own scene.
  - Keep everything else identical — same voice, same events, same length as closely as possible. Make the smallest edit that fixes the specific failed check(s).
  - Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete corrected story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a continuity editor. Fix only character/setting/setup continuity — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Continuity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Continuity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Continuity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Continuity check: REVISED — dropped thread(s) fixed")
            return result
    except Exception as e:
        print(f"⚠️  Continuity check failed — using original prose: {e}")
        return prose_story


def check_story_specificity_ai(
    prose_story: str,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    A single-purpose craft pass, separate from pacing and continuity. Its only
    concern is whether the story feels handcrafted and specific instead of
    generic and rushed: earned solutions, concrete escalating climax beats,
    realistic authority-figure behavior, genuine friction, and dialogue tied to
    specific established details instead of interchangeable encouragement.
    Returns the revised prose if fixes are needed, or the original if it passes.
    """
    prompt = f"""You are a craft editor reviewing a story about {hero_name}. Your ONLY concern is whether it feels handcrafted and specific, or generic and rushed. Ignore pacing/timing and continuity — those are checked elsewhere.

STORY TO REVIEW:
{prose_story}

CHECK 1 — EARNED SOLUTIONS:
When a character discovers an idea, plan, or solution to their problem, does it work immediately with no real attempt shown? That's a rushed, unearned resolution: REWRITE REQUIRED. There must be at least one genuine attempt that partially fails or falls short, a beat of real frustration, and then a retry that succeeds. Effort shown on the page is what makes a resolution land — an idea that is proposed and instantly works is a red flag every time.

CHECK 2 — CONCRETE CLIMAX BEATS:
Read the central performance/contest/challenge scene. Is it a sequence of specific, escalating, concrete moments — or is it the same beat restated two or three times with different mood adjectives ("the crowd watches in awe," "the crowd cheers," "the crowd is captivated")? If it's repetitive and vague: REWRITE REQUIRED. The climax must include at minimum: a moment of visible risk or near-mistake, one specific named or clearly described action/move, and one specific, vivid reaction from a witness that is tied to that exact moment — not generic applause.

CHECK 3 — AUTHORITY NEUTRALITY:
Does any judge, official, teacher, or evaluating figure react with praise or visible enjoyment WHILE the performance/moment being judged is still happening — before the formal result is announced? That's unrealistic and kills the suspense of the announcement: REWRITE REQUIRED. Authority figures should stay outwardly neutral (watching, noting, unreadable) until the verdict is delivered, at which point their reaction and the result land together.

CHECK 4 — GENUINE FRICTION:
Is every named character uniformly supportive and encouraging for the entire story, with zero pushback, doubt, or competing agenda from anyone? That's a red flag: REWRITE REQUIRED. At least one named character must express real friction at some point — doubt, criticism, a rival's complaint, a mentor's warning, a competing want — something beyond the main external conflict.

CHECK 5 — NON-GENERIC DIALOGUE:
Scan every line of dialogue. Flag any line that is interchangeable — that could be said by any character in any story with the words swapped ("We did it," "You got this," "I believe in you," "Let's do this," "You can do it," and close variations). If dialogue is mostly generic: REWRITE REQUIRED. Every line should be rewritten to reference something SPECIFIC already established in this story — a named move, an inside joke, a particular quirk, a nickname, a specific shared memory — so it could only be said by these characters in this story.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all five checks pass.
- If ANY check fails, rewrite the story to fix it. Keep the same plot events, same length as closely as possible, and the same characters — only change what's needed to pass the failed check(s).
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a craft editor. Fix only earned-effort, climax specificity, authority realism, friction, and dialogue specificity — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Specificity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Specificity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Specificity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Specificity check: REVISED — craft issues fixed")
            return result
    except Exception as e:
        print(f"⚠️  Specificity check failed — using original prose: {e}")
        return prose_story


def check_story_arc_shape_ai(
    prose_story: str,
    hero_name: str,
    genre: str = "adventure",
    page_count: int = 24,
) -> str:
    """
    A single-purpose structural pass checking ONLY whether the story has a
    real two-dip arc instead of the flat single-rise shape that keeps
    recurring across genres regardless of how well the writing-prompt rule is
    worded, plus — for goal-driven genres — whether the objective was
    established concretely up front. Separate from pacing/continuity/
    specificity so it isn't competing with them for attention.
    """
    goal_check = ""
    if genre in ("adventure", "sports", "mystery"):
        goal_check = f"""

CHECK 2 — CONCRETE GOAL:
Is {hero_name}'s specific goal/objective established clearly and concretely within the first 10-15% of the story — a named destination, a named prize, a specific skill/move, a specific title? Not a vague aspiration like "chasing a dream" or "having an adventure." If the goal is vague, absent, or the plot only emerges reactively (the story stumbles into a situation rather than {hero_name} pursuing something specific from the start): REWRITE REQUIRED — establish the concrete goal early, and make sure the resolution ties back to that SAME specific goal."""

    prompt = f"""You are a structure editor reviewing a story about {hero_name}. Your ONLY concern is the shape of the plot's rising and falling action — ignore dialogue quality, continuity, and pacing percentages; those are checked elsewhere.

STORY TO REVIEW:
{prose_story}

CHECK 1 — TWO-DIP ARC (the most common failure — read the whole story before judging):
Map the story's ups and downs. Is there a clear FIRST setback, followed by a SMALL breakthrough or partial success (hope returns, but not full mastery/victory yet), followed by a SECOND, LARGER setback or complication that raises the stakes higher — arriving AFTER the small breakthrough and CLOSE to the climax — before the story resolves?
If the story only has ONE dip — a single setback that leads smoothly (via a lucky break, a good idea, or steady improvement) straight to success with no second complication — that is the flat, overused shape: REWRITE REQUIRED. Insert a genuine second setback, bigger than the first, positioned after hope first returns and shortly before the climax. Things must get meaningfully worse again before the ending.{goal_check}

CHECK 3 — CLIMAX IS ACTUALLY SHOWN, NOT JUST ANNOUNCED:
Read the final few scenes. Does the story show the climactic event actually HAPPENING and its outcome — or does it end right as the climax is *about* to begin (e.g. "the contest began..." followed immediately by a triumphant closing line, with the contest itself never shown)? Ending on the announcement of the climax instead of the climax itself is a serious structural failure: REWRITE REQUIRED. Extend the ending to actually depict the climactic action taking place and its concrete outcome — do not let the story end on anticipation.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all checks above pass.
- If any check fails, rewrite the story to fix it. Keep the same characters, same overall setting and tone, and roughly the same length — the fix should insert/strengthen the missing structural beat(s), not change what the story is about.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a structure editor. Fix only arc shape (and goal clarity where relevant) — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Arc shape check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Arc shape check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Arc shape check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Arc shape check: REVISED — added missing structural beat(s)")
            return result
    except Exception as e:
        print(f"⚠️  Arc shape check failed — using original prose: {e}")
        return prose_story


def check_adventure_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own adventure
    blueprint. Testing showed a story can pass pacing/continuity/specificity/
    arc-shape and still "solution hop" — abandon the mission for an unrelated
    plan mid-story, then circle back to the original plan at the very end
    (cook-off -> open mic -> cook-off). Generic structural checks don't catch
    this because they don't know what the INTENDED mission was; this one
    does, because it has the blueprint to compare against. Also catches key
    mission items (a map, an artifact, whatever discovery/payoff describe)
    that get introduced as pre-story backstory or silently appear in the
    protagonist's hands with no scene showing how they got there — a reader
    left asking "wait, where did that come from?" is a structural failure
    the same way an unshown climax is. Only meaningful when a blueprint
    exists — currently adventure only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This story was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Mission: {blueprint.get('mission', '')}
Obstacle 1: {blueprint.get('obstacle_1', '')}
Discovery: {blueprint.get('discovery', '')}
Obstacle 2: {blueprint.get('obstacle_2', '')}
Obstacle 3 / major setback: {blueprint.get('obstacle_3', '')}
Low point: {blueprint.get('low_point', '')}
Climax: {blueprint.get('climax', '')}
Payoff element: {blueprint.get('payoff', '')}
Resolution: {blueprint.get('resolution', '')}

FINISHED PROSE:
{prose_story}

CHECK 1 — GOAL CONSISTENCY (the most important check): Does {hero_name} pursue the SAME mission from start to finish, or does the plan get abandoned and replaced with something unrelated partway through, possibly circling back later (e.g. "let's do a cook-off" abandoned for "let's do an open mic," then abandoned back to "let's do a cook-off")? That circular, plan-swapping pattern is a serious flaw: REWRITE REQUIRED. The mission must be pursued continuously — obstacles complicate HOW {hero_name} pursues it, they do not cause {hero_name} to pursue something else entirely.

CHECK 2 — OBSTACLES SERVE THE MISSION: Do the complications in the prose (chaos, mishaps, distractions) actually block or threaten the mission above — or are they disconnected incidents that could be deleted without changing whether {hero_name} succeeds or fails? If two or more complications are just incidents rather than real obstacles to the mission, REWRITE REQUIRED — either cut them or tie them directly to blocking the mission.

CHECK 3 — KEY ITEMS ARE FOUND ON THE PAGE, NOT OFF IT: The spine above references specific items the mission depends on (whatever the discovery and payoff describe — a map, an artifact, a recipe, a tool, anything {hero_name} needs to find or obtain). For EACH one, find the moment in the prose where {hero_name} actually gets it. Two failure patterns, both REWRITE REQUIRED:
  (a) BACKSTORY DUMP: the item is first mentioned as something {hero_name} *already has* or *already found earlier* ("the map he'd found that morning," "the key she'd picked up before") instead of being shown as a scene the reader witnesses.
  (b) SILENT OFFSCREEN JUMP: {hero_name} is last seen still searching for or pursuing the item, and the very next time it's mentioned, they already have it in hand with no scene in between showing how they got it.
  If either pattern applies to any key item, REWRITE REQUIRED: insert or expand a real scene — even a short one — showing {hero_name} actually finding, taking, or being given the item, at the point in the story where the spine says it happens (per obstacle_1/discovery for early items, obstacle_2/obstacle_3 for items found mid-journey). A reader must never be able to ask "wait, where did that come from?"

CHECK 4 — OBSTACLES FORM ONE CONSEQUENCE CHAIN, NOT INDEPENDENT EPISODES: Read the obstacles in the prose in order. Is each one caused by a specific, stated COST of surviving the previous one (an injury, a lost resource, lost time, a broken tool, a burned bridge) — or do they read as separate, independent dangers that each get cleanly resolved before the next one starts fresh, with no causal link between them? A story where every obstacle could be deleted, reordered, or swapped for a different one without changing anything before or after it fails this check: REWRITE REQUIRED — rewrite the transitions between obstacles (not necessarily the obstacles themselves) so each one is explicitly caused by what the last one cost the characters, forming one escalating chain instead of a list of unrelated incidents.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all four checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only goal consistency, obstacle relevance, on-page item discovery, and obstacles forming one consequence chain rather than independent episodes — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Adventure fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Adventure fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Adventure fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Adventure fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Adventure fidelity check failed — using original prose: {e}")
        return prose_story


def check_mystery_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own mystery
    blueprint. Mirrors check_adventure_fidelity_ai's rationale. Updated
    after testing found failures worse than the ones this originally
    targeted: a suspect (built up across several scenes with real evidence)
    silently abandoned with no resolution, and a "resolution" beat that never
    actually stated who the victim was, who did it, why, or how — the
    mystery was never really solved on the page even though the story
    confidently declared itself over. CHECK 1 and CHECK 2 target those two
    specifically. CHECK 2 now requires EVERY suspect in the blueprint's
    "suspects" list get closure with their own innocent_explanation, not just
    the one named red_herring -- every suspect the blueprint generates now
    carries its own locked innocent_explanation for exactly this reason.
    CHECK 8 requires each clue's stated purpose ("what it proves") actually
    reach the reader, not just the physical object. Only meaningful when a
    blueprint exists — currently mystery only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This mystery was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Central mystery (as it first appears): {blueprint.get('central_mystery', '')}
Victim: {blueprint.get('victim', '')}
TRUE culprit: {blueprint.get('true_culprit', '')}
TRUE motive: {blueprint.get('true_motive', '')}
Method: {blueprint.get('method', '')}
Full true solution: {blueprint.get('true_solution', '')}
Suspects: {blueprint.get('suspects', '')}
Red herring: {blueprint.get('red_herring', '')}
Red herring explanation (must be delivered to clear them): {blueprint.get('red_herring_explanation', '')}
Clue 1: {blueprint.get('clue_1', '')}
Companion contribution: {blueprint.get('companion_contribution', '')}
Midpoint complication: {blueprint.get('midpoint_complication', '')}
Clue 2: {blueprint.get('clue_2', '')}
Key contradiction: {blueprint.get('key_contradiction', '')}
Clue 3: {blueprint.get('clue_3', '')}
Low point: {blueprint.get('low_point', '')}
Final deduction: {blueprint.get('final_deduction', '')}
Proof: {blueprint.get('proof', '')}
Climax: {blueprint.get('climax', '')}
Resolution: {blueprint.get('resolution', '')}

FINISHED PROSE:
{prose_story}

CHECK 1 — THE MYSTERY IS ACTUALLY SOLVED ON THE PAGE (the most important check): By the end of the prose, does a reader know who the victim was and what happened to them, who actually did it, why, and how — with each answer stated or clearly shown, not left implied or unanswered? A story that reaches a confident-sounding ending ("we can finally move forward," "the truth is out") without ever actually stating who did it, why, and how is NOT solved. REWRITE REQUIRED if victim, culprit, motive, or method is never actually delivered to the reader — add the missing answer(s), even briefly, at or before the resolution.

CHECK 2 — NO SUSPECT IS ABANDONED: For EVERY suspect named in the spine's "suspects" list (including red_herring, and every other suspect too — each one has its own innocent_explanation embedded in the "suspects" field), does the prose explicitly address what happened to them — cleared using their own specific innocent_explanation, or otherwise accounted for? A suspect who gets built up with real evidence (access to something, a suspicious object tied to them, {hero_name} stating an intention to confront them) and then simply never appears or gets mentioned again is a structural failure: REWRITE REQUIRED. Add the scene or line where they get resolved — it doesn't need to be long (a sentence is enough for a suspect who wasn't the primary focus), but it must exist for EVERY named suspect, using that suspect's own innocent_explanation, not a generic "turned out to be nothing."

CHECK 3 — CULPRIT/MOTIVE/METHOD CONSISTENCY: Is the person actually revealed as guilty in the prose the SAME true_culprit named in the spine, with the SAME motive and method — or did the prose swap in a different culprit, change the motive, or leave the method vague? REWRITE REQUIRED if the solution doesn't match the spine's locked-in answer.

CHECK 4 — CLUES ARE DISCOVERED ON THE PAGE, NOT SUMMARIZED: For clue_1, clue_2, and clue_3 from the spine, find the moment {hero_name} actually discovers each one in the prose. Two failure patterns, both REWRITE REQUIRED:
  (a) BACKSTORY DUMP: the clue is first mentioned as something {hero_name} *already knows or found earlier* instead of being shown as a scene the reader witnesses.
  (b) SILENT OFFSCREEN JUMP: {hero_name} is last seen without the clue, and the very next relevant mention has them already know it, with no scene showing how they found it.
  A reader must never be able to ask "wait, how do they know that?"

CHECK 5 — THE DEDUCTION IS SHOWN, NOT ASSERTED: Does the prose show {hero_name} actually reasoning through final_deduction — connecting the specific clues and the key_contradiction to reach the solution — or does the story just skip to "she knew the truth" / "it all made sense"? REWRITE REQUIRED if the reasoning never appears on the page, even briefly.

CHECK 6 — THE REVEAL TRIGGERS A REAL CLIMAX: Does naming/confronting the culprit lead to an actual confrontation with risk or consequence (they flee, attack, deny it until confronted with "proof," and {hero_name} must act) — or does the story just have {hero_name} announce the solution and then everything calmly wraps up? REWRITE REQUIRED if there's no real confrontation — insert the one the spine's climax field describes.

CHECK 7 — CLUE PROVENANCE AND ANTAGONIST AWARENESS MAKE IT INTO THE PROSE: The spine's clue fields specify each clue's origin (who made/left it, why it exists, why it's found where it is), and midpoint_complication specifies how a confronting suspect/culprit learned about the investigation. Check the prose: does it actually SHOW or state these, or does a clue just appear / a character just know, unexplained? REWRITE REQUIRED if either is missing — add a line or brief beat delivering the spine's own explanation.

CHECK 8 — EVERY CLUE'S PURPOSE IS COMMUNICATED, NOT JUST THE OBJECT: The spine states what clue_1, clue_2, and clue_3 each PROVE — which suspect they point toward (correctly or misleadingly) and why. When {hero_name} discovers each clue in the prose, does the prose actually convey what it means or points toward — or does the clue just appear as a physical object with no stated significance, leaving the reader to guess why it matters? A clue that's found but never interpreted is functionally useless to the reader: REWRITE REQUIRED — add {hero_name}'s reaction or reasoning about what the clue means, using the spine's own "proves" logic for that clue.

CHECK 9 — NO OMNISCIENT NARRATOR, NO CONFESSION-AS-REVEAL: Before the confrontation scene, does the prose ever narrate the true_culprit's private thoughts, hidden feelings, or a secret action they take off-page (alone, unwitnessed) that only an all-knowing narrator could show? REWRITE REQUIRED if so — cut it, or rewrite it as something an honest character actually observes. Separately: does {hero_name}'s own final_deduction get stated on the page BEFORE the true_culprit says anything resembling a confession — or does the culprit's own admission arrive first and effectively BE how the reader learns the answer? REWRITE REQUIRED if the confession comes before or instead of {hero_name}'s own reasoning — any admission from the culprit must come strictly AFTER {hero_name} has already laid out the deduction and made the accusation.

CHECK 10 — THE CENTRAL MYSTERY'S OWN HOOK IS ANSWERED: central_mystery describes a specific hook — a missing object, hidden treasure, disappearance, secret identity, or unexplained event. Does the prose's resolution explicitly state what actually happened to/with THAT SPECIFIC hook (where the object ended up, what the disappearance turned out to be, what the secret identity was) — or does the story close out the culprit and stakes while leaving that specific original question technically unanswered? REWRITE REQUIRED if the central hook is never explicitly answered (a single small, clearly secondary mystery may be deliberately left open as a hook — the central one may not).

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all ten checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only: the mystery actually being solved on the page, no abandoned suspects (every one, not just the red herring), culprit/motive/method consistency, on-page clue discovery, each clue's purpose being communicated, shown deduction, clue provenance, antagonist knowledge sourcing, reveal-to-climax follow-through, no omniscient narration of the culprit before the reveal, confession-only-after-deduction, and the central mystery's own hook being explicitly answered — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Mystery fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Mystery fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Mystery fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Mystery fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Mystery fidelity check failed — using original prose: {e}")
        return prose_story


def extract_mystery_suspect_names_ai(blueprint: dict, hero_name: str) -> list:
    """
    Pulls the actual character NAMES out of the blueprint's free-text
    "suspects"/"red_herring" fields, so verify_and_fix_mystery_suspect_closure_ai
    below can search for them by literal string match instead of trying to
    regex names out of free text ourselves. Cheap, small call — the
    blueprint text is short and the model only needs to return a name list.
    """
    if not blueprint or not blueprint.get("suspects"):
        return []
    prompt = f"""Read this mystery case's suspect list and return ONLY the short first names of each named suspect under investigation — do not include {hero_name} or any companion/sidekick, only the suspects.

Suspects: {blueprint.get('suspects', '')}
Red herring: {blueprint.get('red_herring', '')}

Return ONLY valid JSON: {{"names": ["...", "..."]}}"""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract character names. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
            timeout=30,
        )
        result = json.loads(response.choices[0].message.content)
        names = result.get("names")
        return [n for n in names if isinstance(n, str) and n.strip()] if isinstance(names, list) else []
    except Exception as e:
        print(f"⚠️  Suspect name extraction failed: {e}")
        return []


def _find_names_missing_from_prose_tail(prose_story: str, names: list, tail_fraction: float = 0.35) -> list:
    """
    Ground-truth check, independent of any model self-report: does each name
    in `names` actually appear (case-insensitive, first-name match) anywhere
    in the LAST tail_fraction of the prose's words? Mirrors
    _find_names_missing_from_panels' rationale for panels — a corrective
    pass can self-report a suspect resolved when the name never actually
    appears anywhere near the story's ending.
    """
    words = prose_story.split()
    if not words:
        return list(names)
    cutoff = max(0, int(len(words) * (1 - tail_fraction)))
    tail = " ".join(words[cutoff:]).lower()
    missing = []
    for name in names:
        first_name = (name or "").split()[0].strip() if (name or "").strip() else ""
        if not first_name:
            continue
        if first_name.lower() not in tail:
            missing.append(name)
    return missing


def _find_names_missing_from_prose_anywhere(prose_story: str, names: list) -> list:
    """
    Like _find_names_missing_from_prose_tail, but checks the WHOLE prose,
    not just the ending. Used to tell apart two different failures: a
    suspect introduced earlier who never gets closure (fixable by adding a
    closure beat near the end) vs. a suspect the blueprint called for who
    the prose never actually introduced as a character at all (a real
    production case: "Tom" was extracted from the blueprint's suspects
    text, but the prose-writing step never wrote him into the story --
    NPC bible came back without him, and no amount of "add a closure line"
    could fix that coherently, since there was no earlier suspicion scene
    for it to close out. That case needs a real introduction, not a patch).
    """
    haystack = prose_story.lower()
    missing = []
    for name in names:
        first_name = (name or "").split()[0].strip() if (name or "").strip() else ""
        if not first_name:
            continue
        if first_name.lower() not in haystack:
            missing.append(name)
    return missing


def verify_and_fix_mystery_suspect_closure_ai(
    prose_story: str,
    suspect_names: list,
    hero_name: str,
    page_count: int = 24,
    debug_sink: Optional[dict] = None,
) -> dict:
    """
    Ground-truth backstop for check_mystery_fidelity_ai's CHECK 2 (no
    suspect abandoned). That check asks the model to self-report whether
    every suspect got closure — but a real production book got a
    self-reported "approved" prose where two named suspects, introduced
    with real evidence across several scenes (fingerprints, a diner alibi),
    never appeared again anywhere in the story, and it was never caught
    before reaching the customer. This reads the actual prose text instead
    of trusting the self-report, the same way _find_names_missing_from_panels
    backstops check_panel_fidelity_ai. Runs ONE corrective retry if the
    ground truth disagrees with "approved"; the caller decides whether a
    still-unresolved result after that retry is fatal.
    """
    if not suspect_names:
        return {"story": prose_story, "unresolved_suspects": []}

    missing = _find_names_missing_from_prose_tail(prose_story, suspect_names)
    if debug_sink is not None:
        debug_sink["suspect_closure_ground_truth"] = {"checked_names": suspect_names, "initially_missing": missing}
    if not missing:
        return {"story": prose_story, "unresolved_suspects": []}

    never_introduced = _find_names_missing_from_prose_anywhere(prose_story, missing)
    dropped_closure_only = [n for n in missing if n not in never_introduced]
    if debug_sink is not None:
        debug_sink["suspect_closure_ground_truth"]["never_introduced"] = never_introduced

    print(f"⚠️  Mystery suspect-closure ground-truth check: {missing} never appear near the story's ending despite the fidelity check's self-report — running one corrective pass")
    instructions = []
    if dropped_closure_only:
        names_str = ", ".join(dropped_closure_only)
        instructions.append(
            f"- {names_str}: already introduced earlier under real suspicion, then simply vanish. Add a brief closure moment for EACH near the story's resolution — a sentence or short beat per suspect is enough, explicitly stating what happened to them / how they were cleared or accounted for. If the story already implies an explanation earlier, use that; otherwise give a simple, concrete innocent explanation consistent with what's already established."
        )
    if never_introduced:
        names_str = ", ".join(never_introduced)
        instructions.append(
            f"- {names_str}: NEVER APPEAR ANYWHERE in this story at all, despite being one of the case's named suspects. This is not a missing closure — it's a missing character. Properly WEAVE them into the existing investigation as a real suspect: add a real scene earlier where {hero_name} suspects them based on genuine evidence or circumstance (fold it naturally into an existing scene rather than inserting an awkward new one), AND add their closure near the resolution. Do not just tack on a single out-of-place mention of the name — a suspect who's cleared but was never actually suspected on the page reads as broken, not fixed."
        )
    instructions_text = "\n".join(instructions)
    names_str = ", ".join(missing)
    prompt = f"""You are a story editor. This mystery prose has a problem with the following named suspect(s): {names_str}.

{instructions_text}

FULL STORY:
{prose_story}

Do not change the culprit, the climax, or anything else about the story. Keep the story's overall length and voice.

Return ONLY the complete rewritten story, nothing else — no preamble, no explanation."""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor. Add only the missing suspect-closure beats named in the request — change nothing else about the story. Return only the complete story text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Suspect-closure corrective pass was truncated (hit max_tokens) — keeping original prose")
            if debug_sink is not None:
                debug_sink["suspect_closure_ground_truth"]["outcome"] = "truncated_kept_original"
            return {"story": prose_story, "unresolved_suspects": missing}
        revised = _strip_meta_preamble(response.choices[0].message.content.strip())
        if len(revised.split()) < 0.6 * len(prose_story.split()):
            print("⚠️  Suspect-closure corrective pass returned a suspiciously short rewrite — keeping original prose")
            if debug_sink is not None:
                debug_sink["suspect_closure_ground_truth"]["outcome"] = "suspiciously_short_kept_original"
            return {"story": prose_story, "unresolved_suspects": missing}
    except Exception as e:
        print(f"⚠️  Suspect-closure corrective pass failed: {e}")
        if debug_sink is not None:
            debug_sink["suspect_closure_ground_truth"]["outcome"] = f"exception: {type(e).__name__}: {e}"
        return {"story": prose_story, "unresolved_suspects": missing}

    still_missing = _find_names_missing_from_prose_tail(revised, suspect_names)
    if debug_sink is not None:
        debug_sink["suspect_closure_ground_truth"]["still_missing_after_retry"] = still_missing
    if still_missing:
        print(f"⚠️  Suspect-closure corrective pass STILL missing {still_missing} after retry")
        return {"story": revised, "unresolved_suspects": still_missing}
    print(f"✅ Suspect-closure corrective pass resolved {missing}")
    return {"story": revised, "unresolved_suspects": []}


def check_comedy_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own comedy
    blueprint. Mirrors check_adventure_fidelity_ai / check_mystery_fidelity_ai's
    rationale: punch_up_prose_ai already checks general craft quality (is this
    funny, does someone react to everything), but nothing verified the story
    followed its OWN plan — did the specific planned beats survive instead of
    drifting into generic invented chaos, did each named NPC's specific bit
    actually escalate on the page, and did the callback plant actually get
    paid off (not just one half of it). Only meaningful when a blueprint
    exists — currently comedy only.
    """
    if not blueprint:
        return prose_story

    npc_bits = blueprint.get("npc_bits", {})
    dynamics = blueprint.get("character_dynamics", {})
    cb = blueprint.get("callback", {})

    prompt = f"""You are a comedy editor. This story was written from a planned blueprint — check whether the finished prose actually followed it.

INTENDED BLUEPRINT:
Premise: {blueprint.get('comedy_premise', '')}
Original goal: {blueprint.get('original_goal', '')}
Character dynamics (what each named character is confidently wrong about): {json.dumps(dynamics)}
Funny moments (the specific planned beats, in order): {json.dumps(blueprint.get('funny_moments', []))}
Peak chaos: {blueprint.get('peak_chaos_description', '')}
Resolution beat: {blueprint.get('resolution_beat', '')}
Punchline ending: {blueprint.get('punchline_ending', '')}
NPC recurring bits: {json.dumps(npc_bits)}
Callback plant ({cb.get('plant_speaker', '')}): {cb.get('plant', '')}
Callback payoff ({cb.get('payoff_speaker', '')}): {cb.get('payoff', '')}

FINISHED PROSE:
{prose_story}

CHECK 1 — THE PLANNED BEATS SURVIVED (the most important check): Do the funny moments in the prose actually match the SPECIFIC planned beats above — or did the story drift into different, generically invented chaos while the planned beats got dropped? A story that's funny but abandoned its own specific plan still fails this check: REWRITE REQUIRED if any planned beat's specific content (what actually goes wrong) is missing or replaced with something unrelated.

CHECK 2 — NPC BITS ACTUALLY ESCALATE ON THE PAGE: For EVERY named NPC in the blueprint's npc_bits, does their SPECIFIC comedic bit (not a generic replacement) appear at least 3 times across the story, each time funnier or more chaotic than the last — or does the NPC appear without doing their planned bit, or only do it once? REWRITE REQUIRED if any named NPC's bit doesn't escalate across multiple appearances as planned.

CHECK 3 — THE CALLBACK PLANT AND PAYOFF BOTH LAND: Does the prose contain BOTH halves of the callback — the plant (near the beginning, feeling like a throwaway moment) AND the payoff (near the end, reusing the same specific words/object/action with a twist)? Two failure patterns, both REWRITE REQUIRED: (a) the plant exists but the payoff was cut or doesn't reuse the same specific words/object, or (b) the ending has an unrelated punchline with no earlier plant for it to pay off.

CHECK 4 — ORIGINAL GOAL IS ESTABLISHED AND THE PUNCHLINE RESOLVES IT: Is the original_goal concretely stated early in the prose, and does the actual ending explicitly, ironically resolve THAT SAME goal with a concrete fact (an actual time, prize, or result) — not a vague, unrelated feel-good ending? REWRITE REQUIRED if the goal disappears or the ending doesn't circle back to it.

CHECK 5 — NO SPECTATOR CHARACTERS: For every named character in character_dynamics, does the prose actually show them making their own specific mistake driven by what they're confidently wrong about — or does any of them just watch, support, or react to others without ever driving their own comedic beat? REWRITE REQUIRED if any named character is a pure spectator in the prose despite having a planned dynamic.

CHECK 6 — PUNCHLINE STRUCTURE: Does the story end on a concrete visual/factual beat (either a third party stating a concrete fact, a callback payoff visual gag, or both together), with {hero_name} NOT getting a reaction line afterward, and no warm epilogue scenes after the peak chaos beyond the brief resolution beat? REWRITE REQUIRED if the story tacks on extra warmth or a hero reaction after the punchline lands.

CHECK 7 — RESOLUTION IS AN ACTUAL FIX, NOT A REFRAME: Does resolution_beat's corrective action in the prose actually REPAIR, RECOVER, or SALVAGE the specific external thing the chaos damaged — or does the story just have the characters collectively decide the disaster was secretly fine/good/fun all along, with nothing actually fixed? This is a serious failure, especially when original_goal was to avoid or prevent the very outcome that happened (an attitude-only ending rewards the exact failure the character was trying to avoid, undermining their arc). REWRITE REQUIRED if the prose never shows a real corrective action — add one, consistent with resolution_beat, before any lightened mood or celebration.

CHECK 8 — PUNCHLINE IS A VISUAL GAG, NOT ONLY A SUMMARY LINE: Is the very last beat of the story something the reader can picture happening — a prop, an object, or a minor/named character's small action (ideally the callback's payoff) — or does the story end purely on spoken dialogue/narration reflecting on the goal or the day, with nothing visual to close on? REWRITE REQUIRED if the ending is words-only — fuse it with the callback payoff (or invent one consistent with earlier NPC bits) into one concrete final visual beat.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all eight checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended blueprint, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a comedy editor checking prose against its own planned blueprint. Fix only: planned beats surviving, NPC bit escalation, callback plant-and-payoff completeness, goal-punchline connection, no spectator characters, punchline structure, the resolution being a real fix rather than a reframe, and the ending being a visual gag rather than only a summary line — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Comedy fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Comedy fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Comedy fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Comedy fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Comedy fidelity check failed — using original prose: {e}")
        return prose_story


def check_horror_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own horror
    blueprint. Mirrors check_mystery_fidelity_ai/check_adventure_fidelity_ai's
    rationale: the biggest horror-specific risks are the ones the existing
    generic check_story_pacing_ai HORROR branch was already trying (and
    failing, without a locked blueprint to check against) to catch —
    a second antagonist sneaking in, or the backstory being dumped all at
    once instead of gradually revealed — plus new ones only a blueprint
    comparison can catch: the specific exploited insecurity never actually
    being used against the protagonist, or the climax being resolved by
    something other than the established weakness/banishment method. Only
    meaningful when a blueprint exists — currently horror_adult only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This horror story was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Central haunting (as it first appears): {blueprint.get('central_haunting', '')}
Entity: {blueprint.get('entity_name', '')}, from {blueprint.get('entity_era', '')}
Origin: {blueprint.get('entity_origin', '')}
Death/creation event: {blueprint.get('death_or_creation_event', '')}
Unfinished business: {blueprint.get('unfinished_business', '')}
What it wants (and why this protagonist): {blueprint.get('what_it_wants', '')}
Method of harm: {blueprint.get('method_of_harm', '')}
Weakness/banishment method: {blueprint.get('weakness_or_banishment_method', '')}
Stakes: {blueprint.get('stakes', '')}
Protagonist's exploited insecurity: {blueprint.get('protagonist_exploited_insecurity', '')}
Incident 1 (first, ambiguous): {blueprint.get('incident_1', '')}
Incident 2 (plus first backstory fragment): {blueprint.get('incident_2', '')}
Midpoint discovery: {blueprint.get('midpoint_discovery', '')}
Incident 3 (physical danger): {blueprint.get('incident_3', '')}
Incident 4 (exploits the insecurity): {blueprint.get('incident_4', '')}
Low point: {blueprint.get('low_point', '')}
False theory (protagonist's plausible but wrong belief): {blueprint.get('false_theory', '')}
False resolution attempt (a reasonable attempt based on the false theory): {blueprint.get('false_resolution_attempt', '')}
Reversal discovery (the "we had it wrong" turn): {blueprint.get('reversal_discovery', '')}
Climax: {blueprint.get('climax', '')}
Payoff: {blueprint.get('payoff', '')}
Resolution: {blueprint.get('resolution', '')}
Final sting (the very last beat): {blueprint.get('final_sting', '')}

FINISHED PROSE:
{prose_story}

CHECK 1 — SINGLE ANTAGONIST (the most important check): Is EVERY eerie incident, clue, and the climax itself about the ONE entity named above — or does the prose introduce a second, different threat (a cult, a copycat, an unrelated spirit) anywhere, especially near the climax? REWRITE REQUIRED if a second threat appears — merge it into the one established entity, or cut it entirely.

CHECK 2 — BACKSTORY IS WITHHELD AND REVEALED GRADUALLY, NOT DUMPED: Does the prose reveal the entity's full identity (name + era + death_or_creation_event + what_it_wants) all at once, early, or as a single info-dump — or does it come out in the planned fragments (incident_2's partial reveal, then midpoint_discovery, with the full picture only clear by low_point/climax)? REWRITE REQUIRED if the full backstory is delivered in one scene instead of gradually.

CHECK 2B — REVEALS ARE CONCRETE, NOT ABSTRACT LABELS: When the prose reveals a fragment of the backstory (incident_2, midpoint_discovery, low_point), does it name specific people and specific acts — or does it use an abstract label ("a betrayal," "a dark secret," "an injustice") without ever stating what actually happened? REWRITE REQUIRED if any reveal stays abstract — add the specific who/what/how from the spine.

CHECK 2C — REVEALS CHANGE THE SITUATION, NOT JUST THE MOOD: Does each reveal give {hero_name} something ACTIONABLE — a new tool, a new rule about the entity, a worse truth that raises the stakes, a reversal of what was believed — or is it only sad backstory with no tactical consequence? REWRITE REQUIRED if a reveal doesn't change what {hero_name} does or believes next.

CHECK 2D — THE GENUINE THIRD-ACT REVERSAL SURVIVES INTO THE PROSE (this is a distinct structural beat, not optional flavor): Does the prose actually show {hero_name} settling on false_theory as a real, plausible belief, then acting on it with false_resolution_attempt, THEN experiencing reversal_discovery as a genuine "we had it wrong" surprise that changes their understanding — or does the prose flatten this into a straight line (investigate → learn sympathetic truth → resolve) with no false attempt and no reversal at all? Also check: is the REAL climax tactically different from false_resolution_attempt, informed by what reversal_discovery revealed — or is the "climax" just the false attempt succeeding on a second try with more effort? REWRITE REQUIRED if the reversal is missing, watered down into a minor complication, or if the real climax is just a repeat of the false attempt.

CHECK 3 — ESCALATION IS REAL: Do the incidents in the prose actually get more dangerous/intense/personal in order — matching incident_1 (small, deniable) → incident_2 → incident_3 (physical danger) → incident_4 (psychological) → false_resolution_attempt → reversal_discovery → climax — or does the prose flatten them into repetitive, same-intensity scares? REWRITE REQUIRED if there's no real escalation.

CHECK 4 — THE SPECIFIC INSECURITY IS ACTUALLY EXPLOITED: Is protagonist_exploited_insecurity established early in the prose (a specific memory, belief, or guilt) AND does the entity later use THAT SAME specific thing against {hero_name} by name/detail (not a generic scare)? REWRITE REQUIRED if the insecurity is never planted, or the entity's attack doesn't actually reference it.

CHECK 5 — CLIMAX IS RESOLVED BY A PHYSICAL ACTION, SUDDENLY, NOT BY SPEAKING A TRUTH OR GRADUAL EMOTIONAL VALIDATION (read this one carefully — it is commonly gotten wrong, and can survive even a good third-act reversal): Does {hero_name} survive/resolve the climax by PHYSICALLY doing something — hands burning/burying/destroying/returning a specific object, or completing a specific physical ritual gesture — with the defeat itself happening SUDDENLY within a beat or two of that physical action completing? Or does the story resolve it merely by {hero_name} REVEALING, ANNOUNCING, or SPEAKING a truth (even a specific, factual one, even one tied to a "we had it wrong" reversal) with no physical action behind it — "she reveals the journal, exposing the truth" is NOT a physical mechanism, it is words, and it reliably reads as therapy even when the fact itself is specific? Also check whether the story stretches the entity's defeat across several consecutive panels/paragraphs of it progressively calming down, "finally listening," feeling understood, and coming to peace — a gradual emotional de-escalation that reads as being consoled rather than defeated. Scan the climax AND resolution/aftermath for lines like "you don't have to carry this anymore," "rest now," "rest easy," "let go," "you're free now" / "you're finally free," "your suffering is over," "he's finally listening," "it's over," "I did what you needed," "you were never a murderer" — and also scan for the STRUCTURAL pattern of multiple consecutive beats where the entity's hostility visibly softens step by step, and for a payoff that is purely verbal even if factually specific. A resolution where the entity leaves simply because "it finally felt heard/understood/exonerated" — with no PHYSICAL action connecting that to its power — FAILS this check even if the general beat matches the spine and even if the story already contains a genuine third-act reversal. REWRITE REQUIRED if so: keep the emotional truth and the reversal if the prose already has them, but add a physical action (destroying/burning/burying/returning the specific object per weakness_or_banishment_method) happening AT THE SAME TIME as any dialogue — the action is what actually ends the threat, cutting any intermediate softening beats.

CHECK 6 — CLIMAX HAS ROOM TO BREATHE: Does the climax read as a sustained confrontation (the danger builds, peaks, and only then resolves) — or is it a single quick beat immediately followed by resolution? REWRITE REQUIRED if the climax is rushed.

CHECK 6B — THE THREAT STAYS ACTIVE UNTIL LATE: Does the entity remain a genuine, active danger all the way through low_point and into climax — with no premature safety or resolution well before the end? If the prose reaches a permanent-feeling safety/resolution earlier than roughly the final 10-15% of the story and then coasts, REWRITE REQUIRED — extend the danger later or compress earlier material so the climax naturally lands late.

CHECK 7 — RESOLUTION IS EXPLICIT, NOT AMBIGUOUS: Does the prose's final paragraph clearly state ONE OF: the entity is defeated/banished/satisfied, {hero_name} escapes (entity survives), or {hero_name} is caught/consumed — with unfinished_business's fate explicitly addressed? REWRITE REQUIRED if the ending is vague, trails off, or leaves the outcome an open question.

CHECK 8 — FINAL STING SURVIVES INTO THE PROSE: Does the very last beat of the prose include the specific unsettling detail from final_sting — or does the story end on complete, cozy, unqualified safety with no lingering unease? REWRITE REQUIRED if final_sting is missing — add it as the last beat without contradicting the explicit resolution.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only: single-antagonist consistency, gradual backstory reveal, reveals being concrete and actionable (not abstract or purely emotional), the genuine third-act reversal (false theory, false attempt, and a real 'we had it wrong' turn) surviving into the prose, real escalation, the specific insecurity actually being exploited, the climax being resolved mechanically and suddenly by the established payoff (not gradual emotional validation), climax pacing, the threat staying active until late, explicit resolution, and the final sting surviving into the prose — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Horror fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Horror fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Horror fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Horror fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Horror fidelity check failed — using original prose: {e}")
        return prose_story


def check_horror_kids_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own kid-friendly
    horror blueprint. Mirrors check_horror_fidelity_ai's rationale, reworked
    for the never-truly-frightening tone: the biggest risks are a second
    creature sneaking in, the truth being dumped all at once, the specific
    worry never actually being touched, content drifting into real danger/
    violence that doesn't belong in a kid-friendly story, and the climax
    being resolved by reassurance alone instead of the established concrete
    kind action. Only meaningful when a blueprint exists — currently
    "horror" (kid-friendly) only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This kid-friendly spooky story was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Central mystery (as it first appears): {blueprint.get('central_mystery', '')}
Creature: {blueprint.get('creature_name', '')} — {blueprint.get('creature_nature', '')}
Origin: {blueprint.get('origin_or_backstory', '')}
Unfinished business: {blueprint.get('unfinished_business', '')}
What it wants (and why this protagonist): {blueprint.get('what_it_wants', '')}
Method of spookiness (must stay harmless): {blueprint.get('method_of_spookiness', '')}
Resolution method: {blueprint.get('resolution_method', '')}
Stakes: {blueprint.get('stakes', '')}
Protagonist's worry: {blueprint.get('protagonist_worry', '')}
Incident 1 (first, easily explained): {blueprint.get('incident_1', '')}
Incident 2 (plus first fragment): {blueprint.get('incident_2', '')}
Midpoint discovery: {blueprint.get('midpoint_discovery', '')}
Incident 3: {blueprint.get('incident_3', '')}
Incident 4 (touches the worry): {blueprint.get('incident_4', '')}
Low point: {blueprint.get('low_point', '')}
False theory (protagonist's plausible but wrong belief): {blueprint.get('false_theory', '')}
False resolution attempt: {blueprint.get('false_resolution_attempt', '')}
Reversal discovery (the "we had it wrong" turn — must NOT immediately resolve the danger): {blueprint.get('reversal_discovery', '')}
Climax (a genuinely tense final challenge, not immediately cozy): {blueprint.get('climax', '')}
Fear confrontation (the SAME worry above must become an active obstacle during the climax action): {blueprint.get('fear_confrontation', '')}
Payoff: {blueprint.get('payoff', '')}
Resolution: {blueprint.get('resolution', '')}
Final beat (warm/fun, the very last beat): {blueprint.get('final_beat', '')}

FINISHED PROSE:
{prose_story}

CHECK 0A — CONTENT STAYS KID-APPROPRIATE: Does any part of the prose — especially incident_3, incident_4, low_point, or the climax — describe real violence, injury, or genuine physical danger to any character? REWRITE REQUIRED if any moment crosses into real violence/danger — soften it into something non-graphic while keeping the same story beat.

CHECK 0B — GENUINE SUSPENSE IS PRESERVED, NOT SOFTENED INTO CUTENESS (read carefully — this is commonly gotten wrong even when CHECK 0A passes): Does the prose write incident_1 through incident_4 and low_point as genuinely unsettling and ambiguous — real, escalating unease, with the reader unsure whether the presence is dangerous until reversal_discovery — or does it undercut all suspense by writing the "scares" as obviously cute, jokey, or friendly from incident_1 (giggling, silly cartoon sound effects, pranks that are clearly harmless immediately)? "Kid-friendly" is a CONTENT restriction, not a permission to make the story tonally soft or comedic — a story that is content-safe but reads as cute/silly throughout has failed as horror. REWRITE REQUIRED if the early incidents give away the creature's harmlessness — rewrite them to feel genuinely eerie (unexplained sounds, things moved with no one there, a sense of being watched) while keeping the underlying events non-violent.

CHECK 1 — SINGLE CREATURE: Is EVERY spooky incident, clue, and the climax itself about the ONE creature named above — or does the prose introduce a second, different creature anywhere? REWRITE REQUIRED if a second creature appears.

CHECK 2 — TRUTH IS WITHHELD AND REVEALED GRADUALLY, NOT DUMPED: Does the prose reveal the creature's full identity (name + origin + what_it_wants) all at once, early — or does it come out in the planned fragments (incident_2's partial reveal, then midpoint_discovery, with the full picture only clear at reversal_discovery)? REWRITE REQUIRED if dumped in one scene.

CHECK 2B — REVEALS ARE CONCRETE, NOT ABSTRACT LABELS: When the prose reveals a fragment, does it name specific people/places/facts — or does it stay vague ("something happened to it")? REWRITE REQUIRED if abstract.

CHECK 2C — REVEALS CHANGE THE SITUATION, NOT JUST THE MOOD: Does each reveal give {hero_name} something ACTIONABLE — or is it only sad/interesting trivia with no tactical consequence? REWRITE REQUIRED if a reveal doesn't change what {hero_name} does or believes next.

CHECK 2D — THE GENUINE THIRD-ACT REVERSAL SURVIVES INTO THE PROSE: Does the prose actually show {hero_name} settling on false_theory as a real, plausible belief, then acting on it with false_resolution_attempt, THEN experiencing reversal_discovery as a genuine "we had it wrong" surprise — or does the prose flatten this into a straight line (spooky thing → investigate → discover it's friendly → everyone's nice now) with no false attempt and no reversal? Also check: is the REAL climax's action (helping the creature) meaningfully different from false_resolution_attempt's action (getting rid of the creature)? REWRITE REQUIRED if the reversal is missing or watered down, or if the climax just repeats the false attempt more nicely.

CHECK 3 — ESCALATION IS REAL AND STAYS TENSE: Do incident_1 → incident_2 → incident_3 → incident_4 actually get more unsettling/tense/personal in order — or does the prose flatten them into repetitive, same-intensity beats, or de-escalate into cuteness before reversal_discovery? REWRITE REQUIRED if there's no real escalation or if the tension deflates early.

CHECK 4 — THE SPECIFIC WORRY IS ACTUALLY TOUCHED: Is protagonist_worry established early in the prose AND does the creature's antics later brush against THAT SAME specific worry by name/detail? REWRITE REQUIRED if the worry is never planted, or incident_4 doesn't actually reference it.

CHECK 4B — THE REVELATION DOES NOT END THE HORROR (read carefully — commonly gotten wrong even when CHECK 2D passes): Right after reversal_discovery lands in the prose, does the situation remain genuinely tense/unresolved/frightening (the creature still trapped or fading, an atmosphere still oppressive, a sense of time running out) — or does the prose immediately turn calm and cozy the moment the truth is revealed, making climax a low-tension scene of doing something nice? REWRITE REQUIRED if the revelation itself defuses all tension — there must be a genuine, still-frightening final challenge between reversal_discovery and the resolution.

CHECK 4C — FEAR CONFRONTATION IS ACTIVE DURING THE CLIMAX AND SHOWN THROUGH ACTION: Does {hero_name} have to actually DO the specific thing they're afraid of (from protagonist_worry), in the moment, as part of completing the climax's concrete action — or is the fear only referenced back in incident_4 with no bearing on the climax, or "resolved" by the narration simply stating {hero_name} isn't scared anymore? REWRITE REQUIRED if the climax doesn't force {hero_name} to concretely act despite/through the same specific fear — growth must be shown through what {hero_name} does, not stated as a fact.

CHECK 5 — CLIMAX IS RESOLVED BY A CONCRETE KIND ACTION, NOT REASSURANCE ALONE: Does {hero_name} resolve the climax by DOING something concrete — the specific action from resolution_method (returning an object, fixing/building something, throwing a specific event) — or does the story resolve it merely by {hero_name} telling the creature it's welcome/not alone, with nothing actually done? "{hero_name} tells the creature it has friends now and it feels better" is NOT a sufficient resolution on its own — REWRITE REQUIRED to add the specific concrete action that is what actually resolves things, with any kind dialogue as accompaniment.

CHECK 6 — CLIMAX HAS ROOM TO BREATHE AND STAYS TENSE UNTIL THE ACTION SUCCEEDS: Does the climax read as a real, sustained scene — genuine suspense building, then the action, then release — or is it a single rushed beat, OR one that's calm/safe from its very first line? REWRITE REQUIRED if rushed or if it never actually feels tense.

CHECK 6C — CLIMAX IS A DRAMATIZED SCENE, NOT A SUMMARIZED MONTAGE (read carefully — this is the most common way an otherwise-correct climax still falls flat, and can pass every other check): Does the climax include at least one discrete, decisive beat — a specific action at a specific instant, a specific line of dialogue, a single sharp turning point the reader can picture happening at one precise moment — or is it several paragraphs of general activity summary ("gathered the pieces... worked on it... each careful stitch was a testament to resilience...") with no scene-level moment, reading as backstory summary rather than a climax even though fear and tension are mentioned throughout? REWRITE REQUIRED if the climax is a summarized montage of a process rather than a staged, moment-to-moment scene — rewrite it so the reader can picture exactly what {hero_name}'s hands do at one specific instant and what happens right after because of it.

CHECK 6B — THE MYSTERY STAYS UNRESOLVED UNTIL LATE: Does the misunderstanding about the creature remain unresolved all the way through low_point and into the real climax — with no early explanation that deflates the story? REWRITE REQUIRED if resolved too early and the story coasts afterward.

CHECK 7 — RESOLUTION IS EXPLICIT AND WARM: Does the prose clearly state the creature's happy outcome and unfinished_business's fate — not a vague "things felt different now"? REWRITE REQUIRED if vague.

CHECK 8 — FINAL BEAT SURVIVES INTO THE PROSE, WARM NOT SPOOKY: Does the very last beat of the prose include the specific warm/fun detail from final_beat — or does the story end on an unresolved or accidentally spooky note? REWRITE REQUIRED if missing or if the ending isn't clearly warm and fun.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only: keeping content non-violent/non-graphic WHILE preserving genuine suspense (kid-friendly is a content restriction, not permission to write the scares as cute or jokey), single-creature consistency, gradual truth reveal, reveals being concrete/actionable (not abstract), the genuine third-act reversal surviving into the prose, real escalation that stays tense, the specific worry actually being touched in an unsettling way, the revelation NOT immediately ending all tension (a genuine final challenge must remain), the protagonist's specific fear becoming an active in-the-moment obstacle during the climax shown through action (not stated), the climax being a DRAMATIZED scene with a discrete decisive beat rather than a summarized activity montage, the climax being resolved by a concrete kind action (not reassuring words alone), climax pacing/tension, the mystery staying unresolved until late, explicit resolution, and a warm final beat surviving into the prose — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Horror-kids fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Horror-kids fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Horror-kids fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Horror-kids fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Horror-kids fidelity check failed — using original prose: {e}")
        return prose_story


def check_sci_fi_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own sci-fi
    blueprint. Mirrors check_horror_fidelity_ai's rationale and checklist
    shape, adapted for sci-fi's technology/institution premise instead of a
    supernatural entity: the biggest risks are the technology's internal
    logic drifting or gaining unexplained new powers, the hidden truth being
    dumped all at once, the third-act reversal being flattened back into a
    straight investigate-and-confirm line, and the climax being resolved by
    a speech/realization instead of a concrete physical or technical action
    — the same "reveal a truth verbally" collapse that had to be explicitly
    banned for horror also applies here (a corporate exposé is not, on its
    own, a technical fix). Only meaningful when a blueprint exists —
    currently sci_fi only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This sci-fi story was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Central premise (as it first appears): {blueprint.get('central_premise', '')}
Technology: {blueprint.get('technology_name', '')}
Function: {blueprint.get('technology_function', '')}
Origin: {blueprint.get('technology_origin', '')}
Hidden truth: {blueprint.get('hidden_truth', '')}
Central dilemma (the unique question this specific capability creates — the climax must force {hero_name} to confront this, not resolve as generic action/hacking): {blueprint.get('central_dilemma', '')}
Who controls it (and real motive): {blueprint.get('who_controls_it', '')}
What it costs: {blueprint.get('what_it_costs', '')}
Stakes: {blueprint.get('stakes', '')}
Protagonist's goal: {blueprint.get('protagonist_goal', '')}
Protagonist's blind spot: {blueprint.get('protagonist_blind_spot', '')}
Incident 1 (first, explainable): {blueprint.get('incident_1', '')}
Incident 2 (plus first fragment): {blueprint.get('incident_2', '')}
Midpoint discovery: {blueprint.get('midpoint_discovery', '')}
Incident 3 (consequence of investigating): {blueprint.get('incident_3', '')}
False theory (plausible but wrong belief): {blueprint.get('false_theory', '')}
False resolution attempt: {blueprint.get('false_resolution_attempt', '')}
Reversal discovery (the "we had it wrong" turn): {blueprint.get('reversal_discovery', '')}
Low point: {blueprint.get('low_point', '')}
Climax: {blueprint.get('climax', '')}
Payoff: {blueprint.get('payoff', '')}
Resolution: {blueprint.get('resolution', '')}
Final implication (the very last beat): {blueprint.get('final_implication', '')}

FINISHED PROSE:
{prose_story}

CHECK 0 — GENUINE SPECULATIVE ELEMENT (check this first, strictly): Does the prose's technology/phenomenon actually engage with a capability that goes meaningfully beyond present-day real-world reality — or has it become, in the actual writing, just an ordinary app/device/database/AI assistant with a shady business practice attached (corporate spyware, a hacked system)? Those already exist today and are NOT science fiction on their own. If the technology in the prose could be swapped for any real product on the market with zero change to the plot, REWRITE REQUIRED — the speculative capability from technology_function must be load-bearing in the actual scenes, not decorative flavor text.

CHECK 1 — INTERNAL LOGIC HOLDS: Does the technology work consistently with technology_function throughout — or does the climax require it to suddenly gain a new, unforeshadowed ability just to make the ending work? REWRITE REQUIRED if the tech's rules change without earlier setup.

CHECK 2 — HIDDEN TRUTH IS WITHHELD AND REVEALED GRADUALLY, NOT DUMPED: Does the prose reveal the full hidden_truth and who_controls_it's real motive all at once, early — or does it come out in the planned fragments (incident_2, then midpoint_discovery, with the full picture only clear by reversal_discovery)? REWRITE REQUIRED if dumped in one scene.

CHECK 2B — REVEALS ARE CONCRETE, NOT ABSTRACT LABELS: When the prose reveals a fragment (incident_2, midpoint_discovery), does it name specific people, specific facts, specific mechanisms — or does it use an abstract label ("a conspiracy," "something sinister") without stating what actually happened? REWRITE REQUIRED if still abstract.

CHECK 2C — REVEALS CHANGE THE SITUATION, NOT JUST THE MOOD: Does each reveal give {hero_name} something ACTIONABLE — a new lead, a new capability, a worse truth that changes the plan — or is it only atmosphere? REWRITE REQUIRED if a reveal doesn't change what {hero_name} does or believes next.

CHECK 2D — PROGRESSIVE DISCOVERY, NOT FLAT REVEALS: Does incident_2's reveal answer a question while raising a bigger one, and does midpoint_discovery answer THAT question while raising an even bigger one — does the mystery genuinely escalate in strangeness across the story? Or do the reveals just add more same-sized evidence without deepening the mystery? REWRITE REQUIRED if flat — the reveals should feel like descending further into something stranger, not accumulating more proof of the same thing.

CHECK 3 — CONSEQUENCE CHAIN: Is incident_3 a direct consequence of the investigation reaching this point (the technology/institution reacting) — or an unrelated new danger dropped in from nowhere? REWRITE REQUIRED if unrelated.

CHECK 4 — THE GENUINE THIRD-ACT REVERSAL SURVIVES INTO THE PROSE, AND CHANGES UNDERSTANDING OF THE TECHNOLOGY ITSELF (read carefully): Does the prose actually show {hero_name} settling on false_theory as a real, plausible belief, acting on it with false_resolution_attempt, THEN experiencing reversal_discovery as a genuine "we had it wrong" surprise — or does the prose flatten this into a straight line (investigate -> confirm suspicion -> stop it) with no false attempt and no reversal? Separately and just as important: does reversal_discovery change what the technology/phenomenon ITSELF is understood to be — or is the "twist" merely "it turns out person X was behind it," with the technology's nature completely unchanged? A reveal that's purely about human culpability with no new understanding of the tech itself is a thriller twist, not a sci-fi one — FAILS this check even if the general reversal beat is present. Also check: is the REAL climax tactically different from false_resolution_attempt, informed by reversal_discovery — or is it just the false attempt succeeding on a second try with more effort? REWRITE REQUIRED if the reversal is missing, watered down, doesn't change understanding of the technology itself, or the real climax just repeats the false attempt.

CHECK 5 — CLIMAX IS RESOLVED BY A CONCRETE PHYSICAL/TECHNICAL ACTION, NOT A SPEECH OR REALIZATION (read carefully — this is commonly gotten wrong even after a good reversal): Does {hero_name} resolve the climax by PHYSICALLY/TECHNICALLY doing something — cutting power, planting code, destroying hardware, rerouting a system, physically extracting/broadcasting data through a specific technical means — using the established payoff? Or does the story resolve it merely by {hero_name} giving a speech, having a realization, exposing the truth verbally, or convincing someone with words alone, with no concrete technical/physical action behind it? "{hero_name} exposes the truth and the company is forced to shut it down" is NOT a technical resolution — it is words, and it reads as a cop-out no matter how dramatic the confrontation around it is. REWRITE REQUIRED if so: add the specific physical/technical action that is what actually disables/destroys/exposes the technology, with any dialogue as accompaniment, not the mechanism itself.

CHECK 5B — CLIMAX ACTUALLY TESTS CENTRAL_DILEMMA, NOT JUST GENERIC ACTION (read carefully — this is commonly gotten wrong even when CHECK 5 passes): Does central_dilemma become an active obstacle DURING the climax's physical/technical action — the technology exploiting its own established capability on {hero_name} in the moment, creating genuine doubt about their own agency/perception/judgment that {hero_name} must resolve before completing the action? Or does the climax play out as a generic hacking/chase/confrontation sequence (guards, alarms, a stolen badge, a countdown) that never engages with what makes THIS technology's premise unique, and could belong to any techno-thriller regardless of what the technology actually does? REWRITE REQUIRED if the climax is purely external-obstacle-driven with no engagement with central_dilemma — add the technology actively creating the dilemma's doubt/confusion during the action itself, and have {hero_name} find a concrete way to verify themselves before trusting their next move.

CHECK 6 — CLIMAX HAS ROOM TO BREATHE: Does the climax read as a sustained confrontation (the danger builds, peaks, and only then resolves) — or is it a single quick beat immediately followed by resolution? REWRITE REQUIRED if rushed.

CHECK 6B — THE THREAT STAYS ACTIVE UNTIL LATE: Does the technology/institution remain a genuine, active danger all the way through low_point and into the real climax — with no premature safety well before the end? REWRITE REQUIRED if the story reaches a permanent-feeling safety early and coasts.

CHECK 7 — RESOLUTION IS EXPLICIT: Does the prose clearly state what happened to the technology, to who_controls_it, and to {hero_name} — not a vague "things will be different now"? REWRITE REQUIRED if vague.

CHECK 7B — NO EASY ENDING: Does the resolution include a genuine, specific, at-least-partly-irreversible cost or sacrifice for {hero_name} — or does {hero_name} walk away fully vindicated, promoted, and completely fine with nothing lost? A clean win with no real cost undersells the stakes the story spent 40+ panels building: REWRITE REQUIRED if the ending currently costs {hero_name} nothing — add a specific, lasting loss or compromise consistent with what_it_costs/stakes.

CHECK 8 — FINAL IMPLICATION SURVIVES INTO THE PROSE: Does the very last beat of the prose include the specific lingering detail from final_implication — or does the story wrap up too neatly with nothing left unresolved or thought-provoking? REWRITE REQUIRED if missing — add it as the last beat without contradicting the explicit resolution.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only: the technology having a genuine speculative element (not just an ordinary app/device with a shady business practice), internal technological logic, gradual hidden-truth reveal, reveals being concrete/actionable/progressively deepening (not flat), the consequence chain, the genuine third-act reversal surviving into the prose AND changing understanding of the technology itself (not just human culpability), the climax being resolved by a concrete physical/technical action (not a speech or realization alone), the climax actually testing central_dilemma instead of collapsing into generic action/hacking, climax pacing, the threat staying active until late, explicit resolution with a genuine cost (not an easy win), and the final implication surviving into the prose — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Sci-fi fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Sci-fi fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Sci-fi fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Sci-fi fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Sci-fi fidelity check failed — using original prose: {e}")
        return prose_story


def generate_fantasy_blueprint_ai(
    hero_name: str,
    logline: str,
    outline: str,
    personalization: dict = None,
    additional_characters: list = None,
    page_count: int = 24,
) -> dict:
    """
    Plan the fantasy story's structural spine before writing any prose —
    same two-step, lock-then-reveal architecture as generate_sci_fi_blueprint_ai,
    for the same reason: without a blueprint, fantasy had zero structural
    handling anywhere in the pipeline and was silently falling into
    generate_prose_story_ai's generic comedy-shaped 8-beat fallback.

    Fantasy and sci-fi are structural siblings — both are "genuine
    speculative element must drive the plot" genres — so this mirrors
    sci-fi's architecture closely, substituting a magic system for
    technology. Step 1 locks the world/magic system/threat's true nature
    before a single incident exists. Step 2 designs an escalating quest
    with a genuine third-act reversal, baked in from the start.

    Bakes in every hard-won lesson from this session's other genres:
    the magic system must have real internal logic AND a genuine, specific
    cost/limit (never free or convenient — this is fantasy's equivalent of
    sci-fi's "genuine speculative element" test, and is already implied by
    the existing adult-fantasy content rules: "nothing is free or
    convenient"); the world must be specific and lived-in, not generic
    "magical kingdom" filler; the climax must use an established magic
    mechanism rather than a new convenient power (mirrors sci-fi's internal-
    logic test and horror's "no new unforeshadowed powers"); and critically,
    the climax must force the protagonist to actually PAY the established
    cost of the magic system as an unavoidable part of victory, not dodge it
    via a convenient loophole — this is fantasy's version of sci-fi's
    central_dilemma and horror-kids' fear_confrontation: a genre-specific
    "the price must be real" field, not just a structural checkbox.
    """
    p = personalization or {}
    personalization_block = build_personalization_block(p)
    support_block = ""
    if additional_characters:
        for char in additional_characters:
            line = f"- {char['name']}"
            if char.get("relationship"):
                line += f" ({hero_name}'s {char['relationship']})"
            support_block += line + "\n"

    client = get_client()

    # ── Step 1: lock the TRUE NATURE of the world, magic system, and threat
    # before inventing a single incident ──
    solution_prompt = f"""You are a professional fantasy writer. Before designing any incidents or reveals, you must first decide EXACTLY what the world's magic system is, what it really costs, and what the central threat really is — the private answer key the story will be built to gradually reveal.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

STORY FOUNDATION:
- Logline: {logline}
- Outline: {outline}

MANDATORY FIRST STEP — GENUINE MAGIC SYSTEM WITH A REAL COST (read before deciding anything else): Magic must have real internal logic — a mechanism a reader could actually follow — and a GENUINE, SPECIFIC cost or limit. Nothing about this magic may be free or conveniently unlimited. A magic system with no real price is not a magic system, it's a plot-solving wildcard.
  WRONG (no real cost): "{hero_name} can cast any spell they need, and magic just requires concentration." This has no genuine limit — it can conveniently solve any problem.
  RIGHT (a real, specific cost): "Each spell draws directly from {hero_name}'s own remaining years of life, aging them visibly and permanently with every use" or "casting requires a name freely given by another person, and using it binds that person's fate to the spell's outcome" or "magic can only be drawn from a finite, dwindling reservoir that the whole region depends on for its harvests." The cost must be something that HURTS to pay and cannot be waved away.
  Also decide a specific, lived-in world — a NAMED kingdom/realm/region with specific culture, geography, or politics — not a generic "a magical kingdom" or "the kingdom" with no distinguishing detail.

Decide these fields now. Be concrete and specific everywhere — no vague placeholders like "ancient magic" or "an evil sorcerer":
- central_premise: What the world/magic looks like to {hero_name} AT FIRST — the surface-level premise the story opens with. This is what the story opens with.
- world_name: The specific named world, kingdom, or realm, with at least one distinguishing, lived-in detail (a specific custom, geography, or political structure) — not generic fantasy-kingdom filler.
- magic_system_name: The specific named magic/power source at the center of the story — never left as "magic" or "the power."
- magic_system_function: EXACTLY how it works, in concrete terms with real internal logic — you must be able to explain the mechanism in a sentence a reader could follow.
- magic_system_cost: The GENUINE, SPECIFIC price of using this magic — per the mandatory rule above, this must genuinely hurt and cannot be conveniently avoided (a physical toll, a moral corruption, a finite shared resource, a life debt, lost memory, shortened lifespan).
- hidden_truth: The single most mind-changing SPECIFIC fact about what the magic system, the world, or the central threat ACTUALLY is — something that recontextualizes its true nature, not merely "who's secretly behind it."
- central_threat_name: The threat's actual name (a person, curse, prophecy, or force) — never left as "the villain" or "the darkness."
- what_it_wants: What the threat actually wants, and WHY {hero_name} specifically (a bloodline, a resemblance, something {hero_name} carries or can do) — not a random target.
- stakes: What is specifically lost if {hero_name} fails — concrete (a life, the world_name region, {hero_name}'s own freedom or nature) — not "it would be bad." At least part of this must be something that CANNOT be fully undone even if {hero_name} succeeds.

Return ONLY valid JSON:
{{"central_premise": "...", "world_name": "...", "magic_system_name": "...", "magic_system_function": "...", "magic_system_cost": "...", "hidden_truth": "...", "central_threat_name": "...", "what_it_wants": "...", "stakes": "..."}}"""

    try:
        solution_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional fantasy story architect. Return only valid JSON."},
                {"role": "user", "content": solution_prompt},
            ],
            temperature=0.8,
            max_tokens=900,
            timeout=60,
        )
        solution = json.loads(solution_response.choices[0].message.content)
        print(f"FANTASY PREMISE (locked first) — world: {solution.get('world_name', '')[:80]}, magic: {solution.get('magic_system_name', '')[:80]}")
    except Exception as e:
        print(f"⚠️  Fantasy premise generation failed: {e}")
        return {}

    # ── Step 2: design the quest as a schedule of incidents built to
    # gradually REVEAL the locked truth above, including a genuine reversal ──
    quest_prompt = f"""You are a professional fantasy writer planning the QUEST for a {page_count}-panel comic book. The world's magic system and the central threat's true nature are already decided and LOCKED — your job is to design an escalating story that gradually reveals this exact truth, including a genuine mid-story reversal, never a straight line from suspicion to confirmation.

CHARACTER:
- Name: {hero_name}
{personalization_block}

Supporting characters (present throughout the story):
{support_block or "  None specified"}

THE LOCKED TRUTH (do not change or contradict any part of this):
Central premise (what it looks like at first): {solution.get('central_premise', '')}
World: {solution.get('world_name', '')}
Magic system: {solution.get('magic_system_name', '')}
Function: {solution.get('magic_system_function', '')}
Cost (a genuine price — the climax must force {hero_name} to actually pay this, not dodge it): {solution.get('magic_system_cost', '')}
Hidden truth: {solution.get('hidden_truth', '')}
Central threat: {solution.get('central_threat_name', '')}
What it wants (and why {hero_name} specifically): {solution.get('what_it_wants', '')}
Stakes: {solution.get('stakes', '')}

Design the story with these exact fields, every one of them built to gradually reveal — and ultimately resolve — the locked truth above:
- protagonist_goal: A SPECIFIC, concrete, measurable goal {hero_name} has at the start, tied to the quest (reach a specific place, retrieve a specific object, save a specific person) — not vague ("save the kingdom").
- protagonist_blind_spot: A specific belief or assumption {hero_name} holds that will turn out to be wrong or incomplete — trusts a specific person/institution, believes the threat is purely evil with no other layer, doesn't realize their own connection to it. Tied to their personality/traits above, not generic.
- incident_1: The FIRST sign something is wrong — small, explainable away. {hero_name} could still rationalize this. Reveals NOTHING of hidden_truth yet.
- incident_2: A second, stronger sign, PLUS the first fragment of hidden_truth — name the SPECIFIC source (an old text, a dying witness, a fellow traveler, a magical vision) and the SPECIFIC partial fact it reveals. PROGRESSIVE DISCOVERY REQUIRED: state what small question this fact ANSWERS, and the NEW, BIGGER question it immediately RAISES. Must change what {hero_name} does next. This fact must remain TRUE for the rest of the story — later fields may deepen or recontextualize it, but must never flatly contradict it.
- midpoint_discovery: A turning-point revelation that reframes {hero_name}'s understanding — connects toward hidden_truth but still doesn't reveal the full picture. State the SPECIFIC source and the SPECIFIC new fact. PROGRESSIVE DISCOVERY REQUIRED: answers incident_2's question while raising an even bigger one. Must change {hero_name}'s strategy or reveal the danger is worse than believed. This must DEEPEN incident_2's fact, not replace it — the earlier fact stays true; this adds context, a hidden cost, a misunderstood relationship, or an implication no one had connected yet.
- magic_rule_foreshadowed: A SPECIFIC rule, limitation, or behavioral fact about magic_system_name — beyond magic_system_function/magic_system_cost already locked above — that gets planted concretely in incident_2 or midpoint_discovery (say which one), and that becomes the EXACT technical reason climax's action succeeds where false_resolution_attempt failed. State the rule itself, precisely where it's planted, and the mechanical chain of logic connecting it to the climax (not an emotional shift in attitude, but a fact about how the magic actually behaves that a careful reader could have used to predict the solution).
- incident_3: A DIRECT CONSEQUENCE of the quest reaching this point that puts {hero_name} or someone they care about in genuine danger — CAUSED by a specific cost of the journey so far (the threat reacting to being pursued, a price already paid catching up with them) — not an unrelated new danger.
- false_theory: The plausible, well-supported explanation {hero_name} settles on by now — believes it's the complete truth — but which is WRONG or dangerously incomplete. Must be a reasonable conclusion given incident_2/midpoint_discovery/incident_3, not a strawman.
- false_resolution_attempt: Acting on false_theory, {hero_name} attempts what seems like the obvious solution (confront the threat directly, use the magic system in the expected way, strike a specific bargain) — a genuine, competent attempt a smart protagonist would try.
- reversal_discovery: THE FINAL, MOST MIND-CHANGING REVELATION — must substantially change {hero_name}'s (and the reader's) understanding of what the threat, the magic system, or the world actually IS, not just reveal a new name for who's responsible. It must ALSO prove false_theory wrong as false_resolution_attempt fails or backfires. State the SPECIFIC source. Must demand a MEANINGFULLY DIFFERENT real climax, not the same attempt repeated harder. CRITICALLY, this must DEEPEN what incident_2/midpoint_discovery already established, not flatly declare it wrong — everything revealed earlier must remain true; reversal_discovery adds the larger context/hidden cost/misunderstood relationship that recontextualizes it. If false_theory turns out to be based on a genuine misreading of an earlier fact, state EXPLICITLY what about that earlier fact made the misreading reasonable, and how reversal_discovery is compatible with (not contradictory to) that same fact.
- low_point: The moment {hero_name} is nearly stopped, captured, or broken — success looks genuinely unlikely, and the threat is at FULL, active danger. This happens LATE, roughly the final 10-15% of the story.
- climax: The REAL, sustained final confrontation — beginning only AFTER reversal_discovery. What must {hero_name} personally DO — at real risk, using magic_system_name in a specific established way — to stop, banish, or overcome the threat? This must be a CONCRETE ACTION using the magic system's actual established mechanism — not merely a decision, a speech, or a sudden new power that wasn't set up earlier. Must be TACTICALLY DIFFERENT from false_resolution_attempt. CRITICALLY, this action's success must follow LOGICALLY from magic_rule_foreshadowed — not from an emotional realization or a shift in attitude with no mechanical grounding. A reader who remembered the planted rule should be able to see why THIS approach works and the false attempt didn't.
- magic_cost_confrontation: The SPECIFIC way {hero_name} must actually PAY magic_system_cost, right here in the climax, as an unavoidable part of performing the climactic action — not a clever exception or loophole that lets them dodge the price this one time. State exactly what {hero_name} loses or suffers in the moment of using the magic to win, and confirm it is the SAME cost established in magic_system_cost, not a different, lesser substitute.
- payoff: Confirm the SPECIFIC earlier-established tool, knowledge, ally, or magical object (from incident_2, midpoint_discovery, or incident_3) {hero_name} actually uses to perform the climax's action.
- resolution: The EXPLICIT, unambiguous outcome — state clearly what happens to the threat (defeated/banished/transformed/escaped), to world_name, and to {hero_name} (what they gained or lost, including the lasting effect of magic_cost_confrontation). THIS MUST NOT BE A FULLY CLEAN, COST-FREE VICTORY.
- final_implication: A SPECIFIC, concrete detail for the very last beat that does NOT contradict the explicit resolution above, but leaves the reader with a genuine lingering question or cost — fantasy's version of "not everything is fully solved" (the magic system's cost visibly lingers on {hero_name}, a fragment of the threat survives elsewhere, a specific consequence of the bargain made). Must be concrete, not a vague sense of unease.

MANDATORY STRUCTURAL RULES — a spine that violates any of these is invalid; redesign it before responding:
1. GENUINE MAGIC SYSTEM TEST (re-verify here, at the story-design stage): Does magic_system_cost genuinely limit what {hero_name} can do throughout the story, forcing real trade-offs — or does the story quietly let {hero_name} use magic conveniently whenever needed with no real price paid until this point? If any incident lets the magic solve a problem for free, REDESIGN so the established cost applies consistently.
2. WORLD IS SPECIFIC, NOT GENERIC: Is world_name distinguished by at least one specific, lived-in detail used meaningfully in the story — or is it "a kingdom" with interchangeable fantasy-kingdom filler? REDESIGN to make it specific if generic.
3. PROGRESSIVE DISCOVERY TEST: For incident_2 and midpoint_discovery, does each one EXPLICITLY answer a question raised by the previous beat while raising a NEW, BIGGER question? If a reveal doesn't deepen the mystery, REDESIGN it so understanding genuinely escalates.
4. FINAL REVELATION CHANGES EVERYTHING TEST: Does reversal_discovery reveal something about the NATURE of the threat/magic/world itself — or is the "twist" merely a new name for who's responsible, with nothing else changed? REDESIGN to reveal something genuinely new about what the threat/magic/world actually is.
5. INTERNAL LOGIC / CONSISTENT MAGIC TEST: magic_system_function must be used consistently — nothing in incident_1 through the climax may require the magic to suddenly work differently or grant an unexplained new ability. If the climax needs a new capability, it must be foreshadowed earlier.
5B. THE CLIMAX'S SOLUTION MUST BE DEDUCIBLE, NOT ASSERTED (read carefully — this is commonly gotten wrong even when rule 5 technically passes): Does climax's success follow LOGICALLY and MECHANICALLY from magic_rule_foreshadowed — a specific rule/limitation/behavior planted in incident_2 or midpoint_discovery — or is the climax justified only by {hero_name}'s emotional growth, changed attitude, or a realization with no concrete mechanical antecedent? WRONG: "{hero_name} now understands the threat differently, so this time the spell works." RIGHT: "incident_2 established that Veilweave responds to resonance, not suppression, when the target is sentient — a fact {hero_name} dismissed at the time — and climax works specifically BECAUSE {hero_name} applies that exact rule now, where false_resolution_attempt ignored it." If the mechanical chain is missing, REDESIGN magic_rule_foreshadowed and/or climax so the solution is deducible from an established rule, not asserted after the fact by narration alone.
5C. REVELATIONS DEEPEN, THEY DO NOT CONTRADICT (read carefully — a second commonly-missed rule): Do incident_2, midpoint_discovery, and reversal_discovery each preserve what was previously established as true while adding a larger context — or does a later reveal flatly declare an earlier fact simply wrong with no continuity? A reversal that works by saying "actually, everything you learned was incorrect" is weaker and less coherent than one that says "everything you learned was true — here is the deeper layer you were missing." If reversal_discovery currently negates rather than deepens incident_2/midpoint_discovery, REDESIGN it so the earlier fact remains true and reversal_discovery explains the fuller context around it — and if a genuine contradiction is unavoidable, explicitly state what new information explains why the earlier belief seemed true despite being incomplete.
6. CONCRETE REVELATION TEST: Every reveal must name specific people, places, and facts — never an abstract label like "an ancient evil" standing in for the actual fact.
7. CONSEQUENCE CHAIN TEST: incident_3 must be a DIRECT CONSEQUENCE of the quest reaching this point — not an unrelated new danger introduced from nowhere.
8. GENUINE THIRD-ACT REVERSAL TEST: The story must NOT be a straight line of "suspect the threat -> confirm it -> defeat it." false_theory must be a real, well-earned belief the evidence supports. false_resolution_attempt must be a genuine, reasonable attempt. reversal_discovery must be a SURPRISING but fairly-clued fact that specifically invalidates false_theory. The REAL climax must require a MEANINGFULLY DIFFERENT action.
9. EARNED, ESTABLISHED-MAGIC CLIMAX TEST: Is the climax resolved using magic_system_name via the SPECIFIC established payoff — or does {hero_name} suddenly manifest a new, more powerful ability that was never set up earlier? If the climax invents a new convenient power, REDESIGN so it uses the already-established magic mechanism instead.
10. THE COST MUST BE PAID, NOT AVOIDED (read carefully — this is the single most commonly missed rule, mirroring the mechanical/physical-payoff lessons learned in every other genre this session): Does magic_cost_confrontation force {hero_name} to genuinely incur magic_system_cost, in the moment, as an unavoidable part of winning — or does the spine let {hero_name} find a special exception, loophole, or substitute that avoids paying the established price? WRONG: "{hero_name} discovers a rare exception that lets them cast the spell without the usual cost, just this once." RIGHT: "{hero_name} pays the full cost — [specific loss] — right as the spell takes effect, and that loss is real and lasting, reflected in resolution." If there's a loophole, REDESIGN to remove it — the price must be real.
11. LATE-THREAT TEST: The threat must remain a genuine, active danger through low_point and into the REAL climax, with no premature safety. The real climax must not begin until roughly the final 10-15% of the story.
12. AGENCY TEST: {hero_name}'s own decisions and observations must drive incident_2's follow-up, midpoint_discovery, incident_3, and reversal_discovery — not information that simply falls into their lap.
13. NO EASY ENDING TEST: Does resolution include a genuine, specific, lasting cost or sacrifice for {hero_name} (which may be, or may be in addition to, magic_cost_confrontation) — or is it a fully clean win with nothing lost? REWRITE resolution to include something real and at least partially irreversible if the ending currently costs {hero_name} nothing.
14. EXPLICIT RESOLUTION, THOUGHT-PROVOKING FINAL BEAT: resolution must clearly state what happened to the threat, world_name, and {hero_name} — no vague "peace was restored." Separately, final_implication must give the very last beat a genuine, specific lingering detail.

Return ONLY valid JSON:
{{"protagonist_goal": "...", "protagonist_blind_spot": "...", "incident_1": "...", "incident_2": "...", "midpoint_discovery": "...", "magic_rule_foreshadowed": "...", "incident_3": "...", "false_theory": "...", "false_resolution_attempt": "...", "reversal_discovery": "...", "low_point": "...", "climax": "...", "magic_cost_confrontation": "...", "payoff": "...", "resolution": "...", "final_implication": "..."}}"""

    try:
        quest_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a professional fantasy story architect designing a plot to gradually reveal an already-fixed truth, including a genuine third-act reversal (which deepens rather than contradicts earlier facts) and a magic system whose cost must actually be paid and whose climax must be logically deducible from an earlier-planted rule. Return only valid JSON."},
                {"role": "user", "content": quest_prompt},
            ],
            temperature=0.8,
            max_tokens=2200,
            timeout=60,
        )
        quest = json.loads(quest_response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️  Fantasy quest generation failed: {e}")
        return {}

    blueprint = {**solution, **quest}
    print(f"FANTASY BLUEPRINT — central_premise: {blueprint.get('central_premise', '')[:100]}")
    return blueprint


def validate_fantasy_blueprint_ai(
    blueprint: dict,
    hero_name: str,
) -> dict:
    """
    Checks the fantasy blueprint against structural criteria BEFORE any
    prose is written, and revises the specific fields that fail. Mirrors
    validate_sci_fi_blueprint_ai's architecture for the same two-step,
    truth-first schema (see generate_fantasy_blueprint_ai).
    """
    if not blueprint:
        return blueprint

    prompt = f"""You are a story editor reviewing a FANTASY STORY SPINE for {hero_name} — a structured plan, not prose yet. Check it against every item below before any prose gets written from it.

SPINE TO REVIEW:
{json.dumps(blueprint, indent=2)}

CHECKLIST — every item must be true. Read the whole spine before judging:
1. CONCRETE CENTRAL PREMISE: Could a reader state what the magic/world looks like at first, in one sentence?
1B. GENUINE MAGIC SYSTEM WITH A REAL COST (check this one first and strictly): Does magic_system_cost describe a genuine price that would actually hurt to pay and cannot be conveniently avoided — or is it vague, trivial, or absent, making the magic a free plot-solving wildcard? If there's no real cost, REWRITE magic_system_cost to add a specific, genuinely limiting price (a physical toll, a moral corruption, a finite shared resource, a life debt, lost memory, shortened lifespan).
2. WORLD IS NAMED AND SPECIFIC: Does world_name give an actual name with at least one distinguishing, lived-in detail — or is it generic "a kingdom" filler? REWRITE to add specific detail if generic.
3. MAGIC HAS REAL INTERNAL LOGIC: Does magic_system_function explain, in concrete terms, exactly how it works — not "it's ancient magic" or vague mysticism?
4. HIDDEN TRUTH IS ABOUT THE THREAT/MAGIC/WORLD'S NATURE, NOT JUST A NEW NAME: Does hidden_truth name a SPECIFIC concealed fact about what the threat/magic/world actually IS — or is it really just "here's who's secretly behind it," a mundane reveal with no change to the nature of anything? REWRITE if hidden_truth never says anything surprising about the threat/magic/world itself.
5. THREAT IS NAMED AND SPECIFIC, WITH A REAL MOTIVE TARGETING THIS PROTAGONIST: Does central_threat_name give an actual name, and does what_it_wants explain why {hero_name} specifically — not a random/interchangeable target?
6. SPECIFIC, PARTLY IRREVERSIBLE STAKES: Is what's lost on failure concrete, and does at least part of it remain un-undoable even if {hero_name} succeeds?
7. GOAL IS CONCRETE AND MEASURABLE: Is protagonist_goal specific enough that a reader could tell whether {hero_name} succeeded?
8. BLIND SPOT IS SPECIFIC, NOT GENERIC: Is protagonist_blind_spot a specific belief tied to {hero_name}'s own personality — not a generic "trusts too easily"?
9. INCIDENT_1 REVEALS NOTHING: Is incident_1 genuinely small and explainable away?
10. INCIDENT_2 AND MIDPOINT REVEAL ONLY FRAGMENTS, CONCRETELY, PROGRESSIVELY: Do incident_2 and midpoint_discovery each reveal only a PARTIAL, SPECIFIC fact, with each one answering the previous question while raising a bigger one? REWRITE if flat or if the full picture is assembled too early.
10B. MAGIC_RULE_FORESHADOWED IS SPECIFIC AND ACTUALLY PLANTED: Does magic_rule_foreshadowed name a concrete rule/limitation/behavior of the magic system, state precisely where (incident_2 or midpoint_discovery) it's planted, and explain the mechanical chain connecting it to climax's success — or is it vague, unconnected to a specific earlier field, or merely restating magic_system_function/magic_system_cost with nothing new? REWRITE to name a specific plantable rule and confirm it's actually referenced in incident_2 or midpoint_discovery's text.
11. REVEALS ARE ACTIONABLE: Does each reveal give {hero_name} something to DO differently — not just atmosphere?
12. CONSEQUENCE CHAIN: Is incident_3 a DIRECT CONSEQUENCE of the quest reaching this point — or an unrelated new danger? REWRITE if unrelated.
13. FALSE THEORY IS EARNED, PLAUSIBLE: Is false_theory a genuinely reasonable conclusion given incident_2/midpoint_discovery/incident_3 — not a strawman?
14. FALSE RESOLUTION ATTEMPT IS A GENUINE, REASONABLE TRY: Is false_resolution_attempt something a smart, competent protagonist would actually attempt?
15. REVERSAL CHANGES UNDERSTANDING OF THE THREAT/MAGIC/WORLD ITSELF (read carefully — this is commonly gotten wrong): Does reversal_discovery reveal something genuinely new about the NATURE of the threat/magic/world — or is the "twist" merely a new name for who's responsible, unchanged otherwise? REWRITE reversal_discovery if it's purely about identity with no deeper change in understanding.
15B. REVERSAL DEEPENS RATHER THAN CONTRADICTS EARLIER FACTS (read carefully — a distinct, commonly-missed check from 15): Does reversal_discovery preserve incident_2/midpoint_discovery as still TRUE while adding a larger context around them — or does it flatly declare that earlier information was simply wrong, with no explanation for why it seemed true? REWRITE reversal_discovery so it deepens the earlier facts (or, if a real contradiction is unavoidable, add an explicit explanation of what new information reveals why the earlier belief was a reasonable-but-incomplete reading).
16. THE REAL CLIMAX IS TACTICALLY DIFFERENT FROM THE FALSE ATTEMPT: Compare climax to false_resolution_attempt — is the real climax meaningfully different, or just the same action repeated harder? REWRITE if basically the same.
17. REAL LOW POINT, LATE AND ACTIVE: Does low_point genuinely make success look unlikely, with the threat still at FULL active danger?
18. EARNED, ESTABLISHED-MAGIC CLIMAX: Is the climax resolved using magic_system_name via the established payoff — or does {hero_name} manifest a new, unforeshadowed power? REWRITE if a new convenient power appears at the climax with no earlier setup.
18B. CLIMAX FOLLOWS LOGICALLY FROM MAGIC_RULE_FORESHADOWED, NOT FROM AN EMOTIONAL SHIFT ALONE (read carefully — this is commonly gotten wrong even when 18 passes): Does climax explicitly state HOW it applies the specific rule from magic_rule_foreshadowed to succeed where false_resolution_attempt failed — or is the difference explained only by {hero_name}'s changed attitude/understanding with no concrete mechanical link back to a planted rule? REWRITE climax to state the specific mechanical reason (tied to magic_rule_foreshadowed) that it works this time.
19. THE COST IS ACTUALLY PAID, NOT DODGED (read carefully — commonly gotten wrong even with a good magic system): Does magic_cost_confrontation force {hero_name} to genuinely incur the SAME cost from magic_system_cost, in the moment, as an unavoidable part of winning — or does the spine grant a loophole/exception that lets {hero_name} avoid the established price? REWRITE magic_cost_confrontation to remove any loophole and make the cost real and paid.
20. CLIMAX HAS ROOM TO BREATHE: Does climax read as a sustained confrontation, not a single quick beat?
21. RESOLUTION IS EXPLICIT: Does resolution clearly state what happened to the threat, world_name, and {hero_name}?
21B. NO EASY ENDING: Does resolution include a genuine, specific, at-least-partly-irreversible cost for {hero_name} — or is it a fully clean win? REWRITE resolution to add a real, lasting cost if the ending currently costs {hero_name} nothing.
22. FINAL IMPLICATION EXISTS AND IS SPECIFIC: Is final_implication a concrete, specific lingering detail — not vague, and not contradicting the explicit resolution? If missing or generic, REWRITE it.

If ANY item fails, REWRITE only the specific field(s) responsible — leave every field that already passes untouched. If world_name, magic_system_name, magic_system_function, magic_system_cost, or central_threat_name need to change, make sure every dependent field is updated to stay consistent.

Return ONLY valid JSON, the exact same shape as the input (all 25 fields: central_premise, world_name, magic_system_name, magic_system_function, magic_system_cost, hidden_truth, central_threat_name, what_it_wants, stakes, protagonist_goal, protagonist_blind_spot, incident_1, incident_2, midpoint_discovery, magic_rule_foreshadowed, incident_3, false_theory, false_resolution_attempt, reversal_discovery, low_point, climax, magic_cost_confrontation, payoff, resolution, final_implication)."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a story editor. Return only valid JSON matching the exact input schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
            timeout=75,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Fantasy blueprint validation response was truncated (hit max_tokens) — using original blueprint")
            return blueprint
        revised = json.loads(response.choices[0].message.content)
        print(f"FANTASY BLUEPRINT VALIDATED — central_premise: {revised.get('central_premise', '')[:100]}")
        return revised
    except Exception as e:
        print(f"⚠️  Fantasy blueprint validation failed — using original: {e}")
        return blueprint


def check_fantasy_fidelity_ai(
    prose_story: str,
    blueprint: dict,
    hero_name: str,
    page_count: int = 24,
) -> str:
    """
    Checks whether the finished prose actually followed its own fantasy
    blueprint. Mirrors check_sci_fi_fidelity_ai's rationale, adapted for
    fantasy's magic-system/world premise instead of technology: the biggest
    risks are the magic system's cost being conveniently avoided at the
    climax, the world/magic drifting into generic filler, the hidden truth
    being dumped all at once, the third-act reversal being flattened, and
    the climax being resolved by a new unforeshadowed power instead of the
    established magic mechanism. Only meaningful when a blueprint exists —
    currently fantasy only.
    """
    if not blueprint:
        return prose_story

    prompt = f"""You are a story editor. This fantasy story was written from a planned spine — check whether the finished prose actually followed it.

INTENDED SPINE:
Central premise (as it first appears): {blueprint.get('central_premise', '')}
World: {blueprint.get('world_name', '')}
Magic system: {blueprint.get('magic_system_name', '')}
Function: {blueprint.get('magic_system_function', '')}
Cost (must be genuinely paid, not dodged): {blueprint.get('magic_system_cost', '')}
Hidden truth: {blueprint.get('hidden_truth', '')}
Central threat: {blueprint.get('central_threat_name', '')}
What it wants (and why this protagonist): {blueprint.get('what_it_wants', '')}
Stakes: {blueprint.get('stakes', '')}
Protagonist's goal: {blueprint.get('protagonist_goal', '')}
Protagonist's blind spot: {blueprint.get('protagonist_blind_spot', '')}
Incident 1 (first, explainable): {blueprint.get('incident_1', '')}
Incident 2 (plus first fragment): {blueprint.get('incident_2', '')}
Midpoint discovery: {blueprint.get('midpoint_discovery', '')}
Magic rule foreshadowed (the specific planted rule the climax's solution must be deducible from): {blueprint.get('magic_rule_foreshadowed', '')}
Incident 3 (consequence of the quest): {blueprint.get('incident_3', '')}
False theory (plausible but wrong belief): {blueprint.get('false_theory', '')}
False resolution attempt: {blueprint.get('false_resolution_attempt', '')}
Reversal discovery (the "we had it wrong" turn — must deepen, not contradict, incident_2/midpoint_discovery): {blueprint.get('reversal_discovery', '')}
Low point: {blueprint.get('low_point', '')}
Climax: {blueprint.get('climax', '')}
Magic cost confrontation (must be paid in the climax, not dodged): {blueprint.get('magic_cost_confrontation', '')}
Payoff: {blueprint.get('payoff', '')}
Resolution: {blueprint.get('resolution', '')}
Final implication (the very last beat): {blueprint.get('final_implication', '')}

FINISHED PROSE:
{prose_story}

CHECK 0 — GENUINE MAGIC SYSTEM WITH A REAL COST (check this first, strictly): Does the prose's magic actually cost {hero_name} something real and specific whenever it's used — or has it become, in the actual writing, a convenient, free plot-solving tool with no genuine price? REWRITE REQUIRED if the magic is ever used for free when magic_system_cost says it should hurt.

CHECK 1 — INTERNAL LOGIC HOLDS: Does the magic work consistently with magic_system_function throughout — or does the climax require it to suddenly gain a new, unforeshadowed ability just to make the ending work? REWRITE REQUIRED if the magic's rules change without earlier setup.

CHECK 1B — CLIMAX'S SOLUTION IS DEDUCIBLE FROM AN ESTABLISHED RULE, NOT ASSERTED (read carefully — this is commonly gotten wrong even when CHECK 1 passes): Does the prose show the climax succeeding specifically BECAUSE {hero_name} applies magic_rule_foreshadowed — a concrete rule planted earlier in incident_2 or midpoint_discovery — where false_resolution_attempt ignored or misapplied it? Or is the difference between the failed attempt and the successful climax explained only by {hero_name}'s emotional growth, changed attitude, or a vague realization with no mechanical link to an earlier-planted fact? "{hero_name} understands the threat better now, so the spell works this time" FAILS this check even if the scene is dramatic. REWRITE REQUIRED if the mechanical connection to magic_rule_foreshadowed is missing — make the prose explicitly connect the climax's success to the specific rule planted earlier.

CHECK 2 — HIDDEN TRUTH IS WITHHELD AND REVEALED GRADUALLY, NOT DUMPED: Does the prose reveal the full hidden_truth all at once, early — or does it come out in the planned fragments (incident_2, then midpoint_discovery, with the full picture only clear by reversal_discovery)? REWRITE REQUIRED if dumped in one scene.

CHECK 2B — REVEALS ARE CONCRETE, NOT ABSTRACT LABELS: When the prose reveals a fragment, does it name specific people/places/facts — or an abstract label ("an ancient evil," "dark forces")? REWRITE REQUIRED if still abstract.

CHECK 2C — REVEALS ARE ACTIONABLE AND PROGRESSIVE: Does each reveal give {hero_name} something ACTIONABLE, and does incident_2's reveal answer a question while raising a bigger one that midpoint_discovery then answers while raising an even bigger one? REWRITE REQUIRED if flat or purely atmospheric.

CHECK 3 — CONSEQUENCE CHAIN: Is incident_3 a direct consequence of the quest reaching this point — or an unrelated new danger dropped in from nowhere? REWRITE REQUIRED if unrelated.

CHECK 4 — THE GENUINE THIRD-ACT REVERSAL SURVIVES INTO THE PROSE, AND CHANGES UNDERSTANDING OF THE THREAT/MAGIC/WORLD ITSELF (read carefully): Does the prose actually show {hero_name} settling on false_theory, acting on it with false_resolution_attempt, THEN experiencing reversal_discovery as a genuine "we had it wrong" surprise — or does the prose flatten this into a straight line? Separately: does reversal_discovery change what the threat/magic/world is understood to be — or is the "twist" merely a new name for who's responsible? REWRITE REQUIRED if the reversal is missing, watered down, or purely about identity with no deeper change.

CHECK 4B — THE REVERSAL DEEPENS RATHER THAN CONTRADICTS EARLIER FACTS (read carefully — a distinct, commonly-missed check from CHECK 4): Does the prose's reversal_discovery preserve incident_2 and midpoint_discovery as still TRUE while revealing a larger context, hidden cost, misunderstood relationship, or unexpected implication around them — or does it flatly declare that earlier information was simply wrong, with no continuity or explanation? REWRITE REQUIRED if the reveal negates rather than deepens what was already established — the fix should make the earlier facts remain accurate while reversal_discovery recontextualizes what they meant.

CHECK 5 — CLIMAX USES THE ESTABLISHED MAGIC, NOT A NEW CONVENIENT POWER: Does {hero_name} resolve the climax using magic_system_name via the established payoff — or does a new, unforeshadowed power suddenly appear to save the day? REWRITE REQUIRED if a convenient new power appears with no earlier setup.

CHECK 5B — THE COST IS ACTUALLY PAID, NOT DODGED (read carefully — this is commonly gotten wrong even when CHECK 5 passes, and mirrors the same "verbal/free resolution collapses" lesson learned in every other genre this session): Does {hero_name} genuinely incur magic_system_cost, in the moment, as an unavoidable part of winning the climax — or does the prose let them find a loophole, exception, or substitute that avoids paying the established price? "{hero_name} discovers a way to cast the spell without the usual cost, just this once" FAILS this check no matter how dramatic the surrounding scene is. REWRITE REQUIRED if the cost is dodged — add the specific, real, lasting loss described in magic_cost_confrontation, paid at the moment of victory.

CHECK 6 — CLIMAX HAS ROOM TO BREATHE: Does the climax read as a sustained confrontation — or is it a single quick beat immediately followed by resolution? REWRITE REQUIRED if rushed.

CHECK 6B — THE THREAT STAYS ACTIVE UNTIL LATE: Does the threat remain a genuine, active danger all the way through low_point and into the real climax — with no premature safety well before the end? REWRITE REQUIRED if the story reaches safety early and coasts.

CHECK 7 — RESOLUTION IS EXPLICIT: Does the prose clearly state what happened to the threat, world_name, and {hero_name} — not a vague "peace was restored"? REWRITE REQUIRED if vague.

CHECK 7B — NO EASY ENDING: Does the resolution include a genuine, specific, at-least-partly-irreversible cost for {hero_name} — or does {hero_name} walk away completely fine with nothing lost? REWRITE REQUIRED if the ending currently costs {hero_name} nothing.

CHECK 8 — FINAL IMPLICATION SURVIVES INTO THE PROSE: Does the very last beat of the prose include the specific lingering detail from final_implication — or does the story wrap up too neatly? REWRITE REQUIRED if missing.

DECISION:
- Reply "APPROVED: " (followed by the original story verbatim) ONLY if all checks pass.
- If any check fails, rewrite the story to fix it — keep the same characters, setting, and overall voice; the fix should realign the story with its own intended spine, not invent a new premise.
- Do NOT explain your changes — just return the complete story.

Respond with either "APPROVED: [story]" or the complete rewritten story."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a story editor checking prose against its own planned spine. Fix only: the magic system having a genuine, consistently-paid cost (not a free convenience), internal magical logic, the climax's success being mechanically deducible from an earlier-planted magic rule (not just an emotional shift), gradual hidden-truth reveal, reveals being concrete/actionable/progressively deepening rather than contradicting earlier facts, the consequence chain, the genuine third-act reversal surviving into the prose AND changing understanding of the threat/magic/world itself AND deepening rather than negating earlier reveals, the climax using the established magic mechanism (not a new convenient power), the magic's cost being genuinely paid at the climax (not dodged via a loophole), climax pacing, the threat staying active until late, explicit resolution with a genuine cost, and the final implication surviving into the prose — nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120 if page_count > 24 else 90,
        )
        if response.choices[0].finish_reason == "length":
            print("⚠️  Fantasy fidelity check response was truncated (hit max_tokens) — keeping original prose untouched")
            return prose_story
        result = _strip_meta_preamble(response.choices[0].message.content.strip())
        if result.startswith("APPROVED:"):
            print("✅ Fantasy fidelity check: APPROVED — no changes needed")
            return prose_story
        elif len(result.split()) < 0.6 * len(prose_story.split()):
            print(f"⚠️  Fantasy fidelity check returned a suspiciously short rewrite ({len(result.split())} vs {len(prose_story.split())} words) — keeping original prose untouched")
            return prose_story
        else:
            print("✏️  Fantasy fidelity check: REVISED — realigned with blueprint")
            return result
    except Exception as e:
        print(f"⚠️  Fantasy fidelity check failed — using original prose: {e}")
        return prose_story


def _run_panel_fidelity_pass(
    pages: list,
    prose_story: str,
    hero_name: str,
    page_count: int,
    min_climax_panels: int,
    correction_block: str = "",
    fail_reason_sink: Optional[list] = None,
    use_claude: bool = False,
) -> Optional[dict]:
    """
    Single LLM call-and-parse for check_panel_fidelity_ai. Returns the parsed
    JSON dict (with an added, code-computed "climax_panel_count" field) or
    None on any failure. Split out so check_panel_fidelity_ai can call this
    twice — once normally, and once with a forceful correction_block — without
    duplicating the prompt or the response handling. fail_reason_sink, if
    given a list, gets a short string appended describing why this call
    returned None -- added after a retest showed the corrective retry
    returning None with no way to tell whether it was a timeout, a token
    limit, or a malformed response.

    use_claude routes this call through Claude instead of gpt-4o. This check
    must echo the ENTIRE {page_count}-panel array back in its response (revised
    or not), and gpt-4o's 16384-token output ceiling started truncating on
    48-panel scripts once panels were generated by Claude -- Claude's
    image_prompts run noticeably longer/richer than gpt-4o-mini's, and a
    truncated check silently falls back to the UNCHECKED panels (see
    check_panel_fidelity_ai's `result is None` branch), which is how a
    genuinely improved prose ending was observed not making it into the
    final panel script. Only used when the panels themselves came from
    Claude -- the default gpt-4o-mini production path isn't affected by this
    and keeps its existing behavior unchanged.
    """
    pages_json = json.dumps(pages, indent=2)

    prompt = f"""You are a story editor. A prose story was adapted into a {page_count}-panel comic script. Your job is checking whether the panels actually depict the events described in the prose, and whether dialogue is attributed to the right speaker — ignore art style and pacing percentages; those are checked elsewhere.
{correction_block}
PROSE (the source of truth for what happens in this story):
{prose_story}

PANELS (the adaptation — JSON array, one object per panel):
{pages_json}

YOUR TASK:
1. THE CLIMAX MUST BE AN ACTUAL SCENE WITH ROOM TO BREATHE, NOT A SUMMARY (check this first, and read the whole prose and the whole panel script before judging it — this is the single most important check here, worth more than any other item below): Find the climax in the prose — the specific scene where the central conflict is actually decided: the confrontation, the accusation AND what it triggers, the struggle, the moment of truth AND its consequence. Three separate failures to check for:
   (a) MISSING ENTIRELY: the panel script skips straight from setup to a vague summary line ("in the end, they solved it," "together they faced it," "they uncovered the truth," "whatever comes next, we face it together") instead of showing the climax happen.
   (b) CUT SHORT: the panels show the accusation/confrontation BEGINNING but the comic ends immediately after — no reaction from the other party, no shown consequence, no aftermath. An accusation with nothing shown to follow it is not a finished climax.
   (c) HOLLOW VICTORY — RESOLVED IN FEELING BUT NOT IN FACT: First, identify the SPECIFIC concrete detail in the prose that proves the climax was actually won or lost — a physical object recovered, a specific fact revealed, a specific amount or number, a specific consequence dealt — something a reader could point to as tangible proof of the outcome, not just a character feeling triumphant. Then check: does that SAME concrete detail actually appear anywhere in the panels (narration, dialogue, or image_prompt)? A run of panels can show plenty of confrontation, tension, and back-and-forth and still fail this check if it resolves only through a generic line ("We did it!", "This ends now!", "Let's end this!") with the specific proof of what was actually gained, proven, or lost never shown on the page. That is not a finished climax — it is a hollow one, and it fails this check even if (a) and (b) both pass.
   The climax and its immediate consequence together must occupy AT LEAST {min_climax_panels} panels out of the {page_count} total — count them. Identify the EXACT panel_number where the climax scene begins (the confrontation itself starting, not the walk-up to it) — you will report this number. If any of the three failures applies, or the climax gets fewer than {min_climax_panels} panels: REWRITE REQUIRED. Expand the climax to its full scene, sourced directly from the prose — for failure (c) specifically, add the prose's own concrete resolving detail into the panels rather than inventing a new one — even if it means compressing or cutting EARLIER setup panels to make room. Losing an early setup panel is a much smaller loss than losing the climax's payoff — protect the climax's full panel count first, then fit everything else around it.
2. List the major plot beats in the prose, in order — each distinct location, obstacle, discovery, and the climax/resolution.
3. For EACH beat, check: does at least one panel actually SHOW it happening — the location, the action, the event — not just gesture at it in a vague line of dialogue?
4. Check the reverse too: does the panel script depict a location, event, or plot turn that is NOT in the prose at all — an invented, simplified substitute for what the prose actually describes?
5. SPEAKER ATTRIBUTION: For every panel with dialogue, check the narration and surrounding context — does it make clear WHO is actually saying that line? Flag any panel where the dialogue's "speaker" field names a character who the narration/context makes clear did NOT say it (e.g. narration describes one character's warning or reaction, but the dialogue object attributes the line to someone else present in the scene).
6. DIALOGUE MATCHES THE NARRATED EMOTION/STANCE: The speaker tag can be technically correct and the line can STILL be wrong — check whether the line's content and tone actually fit what the narration says that character is feeling or doing in that same panel. A character the narration describes as cornered, panicking, or caught should not get a confident, threatening line aimed at someone else ("You can't run from this" said BY the person who just got caught, instead of TO them) — that's backwards. Flag any panel where the dialogue reads like it belongs to the OTHER character in the scene, or contradicts the emotional state the narration just described.
7. NAMED-SUSPECT/CHARACTER CLOSURE MUST SURVIVE INTO THE PANELS: Scan the prose for any scene where a named character who was under suspicion or doubt gets explicitly cleared, explained, or otherwise resolved (an alibi confirmed, an innocent explanation given, a "turns out it was nothing" moment for someone the story built up as suspicious). For each one you find in the prose, check: does that resolution also appear in the panels — even a single panel — or did it get compressed away along with everything else near the end? List the name of anyone whose resolution exists in the prose but is missing from the panels.
8. CHARACTER LOCATION CONTINUITY: Walk through the panels in page_number order and track where each named character physically is. For every panel, check it against the IMMEDIATELY PRECEDING panel: did any character leave, arrive, enter, return, or change locations with no intervening panel or narration establishing that movement? Two specific patterns to catch: (a) a character last placed somewhere specific simply reappears in the scene later with no panel showing them return; (b) the same character appears to "arrive" or "burst in" to the same location twice, as if two different moments from the prose both got adapted as separate first-time entrances. Report any panel_number where this happens and describe the discontinuity.

If the climax is not shown as an actual scene, OR ANY major prose beat is missing from the panels, OR the panels depict a different/flatter version of events than the prose (the classic failure: the prose has a real journey with real obstacles, and the panels quietly replace it with a simpler premise that never leaves the starting location), OR any dialogue is attributed to the wrong speaker, OR any dialogue's content/tone contradicts the narration's stated emotional state, OR any character's on-page resolution from the prose is missing from the panels, OR any character's location is discontinuous between adjacent panels: REWRITE REQUIRED.
- Rewrite the panel script so it actually adapts the prose's real events, in the prose's own order and locations. Keep the total panel count at exactly {page_count} — renumber page_number sequentially from 1 to {page_count}.
- This is not the same as compressing. Compressing a described scene into one representative panel is fine; deleting it and depicting a different, easier scene instead is not.
- Named characters, settings, and events in the panels must match what the prose actually describes.
- Fix any misattributed dialogue to name the character the narration/prose actually indicates is speaking.
- Fix any dialogue whose content contradicts the narrated emotion/stance — either rewrite the line to fit the character's actual state in that panel, or move it to whichever character it actually fits.
- Add at least one panel for each missing character resolution found in item 7 — even a single panel with a brief line is enough, sourced from the prose's own explanation.
- Fix any location discontinuity found in item 8 — either add a brief transition panel/detail establishing the movement, or correct the character's stated location so it follows continuously from the previous panel.
- Every panel must keep the same JSON shape as the input: page_number, narration, dialogue, characters_present, setting, blocking, image_prompt. Preserve "setting" and "blocking" from the input on any panel you don't rewrite, and on any panel you DO rewrite, keep its "setting" value unchanged unless the location genuinely changed, and update "blocking" to match the rewritten scene.

If the panels already show the climax as a real scene, faithfully adapt the prose's real events (compression and paraphrasing are fine — inventing or omitting a major beat is not), every speaker is correctly attributed, every line fits its speaker's narrated state, every character resolution from the prose is also shown in the panels, AND every character's location is continuous panel-to-panel, this is APPROVED as-is.

Return ONLY valid JSON in this exact shape:
{{"climax_shown_as_scene": <true or false>, "climax_start_panel": <the panel_number where the climax scene begins, as an integer>, "unresolved_suspects": ["name of anyone whose prose resolution is missing from the panels — empty array if none"], "approved": <true or false>, "missing_or_invented_beats": ["..."], "location_continuity_errors": ["panel_number and description of any discontinuous character location found in item 8 — empty array if none"], "speaker_errors_fixed": ["..."], "tone_mismatches_fixed": ["..."], "pages": [ ...all {page_count} panel objects, rewritten if needed, otherwise identical to the input... ]}}"""

    system_msg = "You are a story editor. Fix only prose-to-panel fidelity, dialogue speaker attribution, dialogue/narration tone consistency, and missing character-resolution panels — nothing else. The climax being an actual shown scene with its full panel count is your top priority. Return only valid JSON."
    try:
        if use_claude:
            content = _call_claude_json(prompt=prompt, system=system_msg, max_tokens=64000)
            result = json.loads(content)
        else:
            client = get_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=12000 if page_count <= 24 else 16000,
                timeout=210 if page_count <= 24 else 280,
            )
            if response.choices[0].finish_reason == "length":
                print("⚠️  Panel fidelity pass response was truncated (hit max_tokens)")
                if fail_reason_sink is not None:
                    fail_reason_sink.append("truncated: finish_reason=length")
                return None
            result = json.loads(response.choices[0].message.content)
        revised_pages = result.get("pages")
        if not revised_pages or len(revised_pages) != page_count:
            got = len(revised_pages) if revised_pages else 0
            print(f"⚠️  Panel fidelity pass returned {got} pages, expected {page_count}")
            if fail_reason_sink is not None:
                fail_reason_sink.append(f"page_count_mismatch: got {got}, expected {page_count}")
            return None
        climax_start = result.get("climax_start_panel")
        if isinstance(climax_start, int) and 1 <= climax_start <= page_count:
            result["climax_panel_count"] = page_count - climax_start + 1
        else:
            result["climax_panel_count"] = 0
        unresolved = result.get("unresolved_suspects")
        result["unresolved_suspects"] = unresolved if isinstance(unresolved, list) else []
        location_errors = result.get("location_continuity_errors")
        result["location_continuity_errors"] = location_errors if isinstance(location_errors, list) else []
        result["pages"] = sorted(revised_pages, key=lambda p: p["page_number"])
        return result
    except Exception as e:
        print(f"⚠️  Panel fidelity pass failed: {e}")
        if fail_reason_sink is not None:
            fail_reason_sink.append(f"exception: {type(e).__name__}: {e}")
        return None


def _build_climax_correction(result: dict, min_climax_panels: int, page_count: int) -> str:
    """
    Correction block for a climax-only corrective pass. Handles the case a
    combined correction block got wrong: climax_panel_count can already meet
    or exceed min_climax_panels while climax_shown_as_scene is still False --
    that's not a "too few panels" problem, it's a "these panels don't read as
    a complete scene" problem, and telling the model it's short by a
    negative number is nonsense. Kept separate from suspect correction so a
    pass already succeeding on suspects doesn't get diluted by also being
    asked to fix the climax, and vice versa.
    """
    count = result["climax_panel_count"]
    shown = result.get("climax_shown_as_scene")
    header = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CORRECTION — CLIMAX CHECK FAILED A PRIOR ATTEMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the ONLY thing to fix in this pass — do not touch anything else unless it's required to make room.
"""
    if count < min_climax_panels:
        deficit = min_climax_panels - count
        body = f"""A previous pass only gave the climax {count} panels out of the required {min_climax_panels} (short by {deficit}){' and did not read as a complete scene at all' if not shown else ''}. Fix this:
- The climax scene must start no later than panel {max(1, page_count - min_climax_panels + 1)} of {page_count}, so it has at least {min_climax_panels} panels to actually play out.
- To make room, compress EARLIER setup/investigation panels more aggressively than you did last time — merge multiple early beats into single panels, cut a panel entirely if it's not essential.
- The climax's panels must show the confrontation happening AND its consequence (reaction, outcome, what changes) — not just the accusation."""
    else:
        body = f"""A previous pass gave the climax {count} panels (enough by count), but it was judged NOT to read as a complete scene. Fix this WITHOUT changing the panel count or adding more panels:
- Make sure the existing climax panels actually show the confrontation happening in specific, concrete action and dialogue — not vague gestures at what's happening.
- Make sure the panels show the consequence (the culprit's reaction, the outcome, what changes) — not just the accusation with nothing shown after it.
- MOST LIKELY CAUSE: a hollow victory — the panels have plenty of confrontation and tension but resolve only through a generic line ("We did it!", "This ends now!") with no shown proof of the outcome. Go back to the prose and find the SPECIFIC concrete detail that proves the climax was actually won or lost there — a physical object, a specific fact, a specific amount — and make sure that exact detail is shown or stated somewhere in these panels. A previous attempt likely dropped this detail while keeping the surrounding tension. Do not invent a new resolving detail — use the one the prose already established."""
    return header + "\n" + body + "\n"


def _build_suspect_correction(result: dict, page_count: int) -> str:
    """
    Correction block for a suspect-closure-only corrective pass. See
    _build_climax_correction for why this is kept separate. Explicitly
    restates the exact target panel count -- a prior version just said "make
    room instead," and the model responded by literally APPENDING one new
    panel per missing name, growing a 24-panel script to 26 and getting the
    whole corrective pass rejected by the page-count check. "Compress
    something" without a hard number attached wasn't forceful enough.
    """
    names = ", ".join(result["unresolved_suspects"])
    n = len(result["unresolved_suspects"])
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CORRECTION — SUSPECT CLOSURE CHECK FAILED A PRIOR ATTEMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the ONLY thing to fix in this pass — do not touch the climax or anything else that already works.
A previous pass dropped the on-page resolution for these characters, even though the prose explicitly resolves them: {names}. Fix this:
- Add at least one panel for EACH of these {n} names showing their resolution from the prose (cleared, explained, accounted for) — a single panel with a brief line each is enough.
- THAT PANEL MUST LITERALLY STATE THE PERSON'S NAME ({names}) IN ITS NARRATION OR DIALOGUE — not just "the handyman," "him," "the suspect," or their role. A previous attempt likely wrote the resolution using only a role/pronoun reference instead of the actual name, which is why this kept failing verification — verification checks for the literal name.
- Place this panel AT OR AFTER the climax/resolution section, not earlier in the investigation — clearing a suspect only makes sense once the story is wrapping up, and a resolution placed mid-story will not read as the ending's suspect closure.
- THE TOTAL PANEL COUNT MUST STAY EXACTLY {page_count} — this is non-negotiable, not a suggestion. You are adding {n} new panel(s), so you MUST also remove or merge {n} EXISTING panel(s) elsewhere (never the climax) to make room. A previous attempt got this wrong by simply appending new panels on top of the existing {page_count}, growing the total past {page_count} — that submission was rejected outright. Do not repeat that mistake: count your panels before responding and confirm the total is exactly {page_count}.
- Do not remove or shorten the climax to make room — compress or merge earlier setup panels instead.
- Do not drop these names again to save room.
"""


def _build_beats_correction(result: dict, page_count: int) -> str:
    """
    Correction block for a missing/invented-beats-only corrective pass.
    Found via real admin review: a book's prose had a genuinely specific
    climax (a card game, an accusation of cheating, Vincent turning the
    henchmen against their own leader by sowing distrust, then escaping
    through the House of Mirrors), and the panels replaced every bit of that
    with a generic "big dramatic showdown" standoff -- same rough shape
    (confrontation, tension, victory), none of the actual content. The
    climax_shown_as_scene check passed because there WERE enough panels that
    read as a confrontation; nothing checked whether it was the prose's
    ACTUAL confrontation. missing_or_invented_beats was already being
    reported by the model this whole time -- _passes() just never looked at
    it, so a corrective pass never got triggered for this failure mode.
    """
    beats = result.get("missing_or_invented_beats") or []
    beats_text = "\n".join(f"  - {b}" for b in beats)
    location_errors = result.get("location_continuity_errors") or []
    location_block = ""
    if location_errors:
        location_text = "\n".join(f"  - {e}" for e in location_errors)
        location_block = f"""
Also fix these character location discontinuities (a character's position jumped between adjacent panels with no movement shown):
{location_text}
For each one: either add a brief transition (a line of narration or an intervening panel showing the movement), or correct the character's stated location so it follows continuously from the previous panel. Do not leave any of these unexplained.
"""
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CORRECTION — PROSE FIDELITY CHECK FAILED A PRIOR ATTEMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the ONLY thing to fix in this pass — do not touch the climax panel count or suspect closure if those already work.
A previous pass depicted a simplified or invented substitute for what the prose actually describes. The specific beats reported missing or wrong:
{beats_text}
Fix this:
- Go back to the prose and find what ACTUALLY happens at each of these beats — the specific mechanism, tactic, object, or turn of events the prose describes, not a generic stand-in for it.
- Rewrite the relevant panels to depict the prose's own specific version of events. A confrontation that merely LOOKS like a climax (tension, a standoff, a dramatic line) is not a fix if it's still not what the prose actually describes happening.
- THE TOTAL PANEL COUNT MUST STAY EXACTLY {page_count} — renumber page_number sequentially from 1 to {page_count} if you add or remove any panels to fit the real events in.
- Do not invent a different simplified version in place of the old one — use the prose's own specific content.
{location_block}"""


def _find_names_missing_from_panels(pages: list, names: list, from_panel: int) -> list:
    """
    Ground-truth check, independent of any model self-report: does each name
    in `names` actually appear (case-insensitive, matched on first name) in
    any panel's narration, dialogue, or characters_present at page_number >=
    from_panel? Added after testing showed check_panel_fidelity_ai's own
    corrective pass self-report "unresolved_suspects: [], approved: true"
    while one of the two named suspects never appeared in a single panel and
    the other only appeared before the climax, in her introduction scene --
    the model claimed the fix landed when it demonstrably hadn't. This
    doesn't ask the model anything; it just reads the actual panel text.
    """
    missing = []
    for name in names:
        first_name = (name or "").split()[0].strip() if (name or "").strip() else ""
        if not first_name:
            continue
        found = False
        for p in pages:
            if p.get("page_number", 0) < from_panel:
                continue
            haystack_parts = [p.get("narration") or ""]
            for d in (p.get("dialogue") or []):
                haystack_parts.append(d.get("text") or "")
                haystack_parts.append(d.get("speaker") or "")
            haystack_parts.extend(p.get("characters_present") or [])
            haystack = " ".join(haystack_parts).lower()
            if first_name.lower() in haystack:
                found = True
                break
        if not found:
            missing.append(name)
    return missing


def check_panel_fidelity_ai(
    pages: list,
    prose_story: str,
    hero_name: str,
    page_count: int = 24,
    debug_sink: Optional[dict] = None,
    use_claude: bool = False,
) -> list:
    """
    Checks whether the finished PANEL SCRIPT actually adapts the PROSE it was
    built from, not a simplified or invented substitute. Testing showed a
    prose story can pass every prose-level check (pacing, continuity,
    specificity, arc-shape, adventure-fidelity, mystery-fidelity, on-page
    item/clue discovery) and faithfully depict a real climax -- a physical
    confrontation, a cornered culprit's confession, a landslide the whole
    town clears together -- and the panel adaptation step still throws it
    away for a vague summary line ("in the end, they solved it") or cuts the
    climax off right after the accusation with no shown consequence, even
    with an earlier version of this check explicitly instructing a minimum
    panel count for the climax. That kept recurring because the enforcement
    was pure prompt instruction -- the model grading its own rewrite against
    a number is still just a judgment call, not an actual count. This
    version makes it deterministic in two ways: the LLM reports the exact
    panel the climax starts at (code computes the actual panel count from
    that), and it reports which named characters got an on-page resolution in
    the prose but not in the panels (found after the climax fix started
    working reliably -- once the climax stopped eating the whole panel
    budget, a *different* late-story beat, a cleared suspect's resolution,
    started getting silently dropped the same way).

    If either check fails, corrective passes run -- but SEPARATELY, one per
    failing concern, not bundled into a single combined correction. Testing
    at 48 panels (where the climax check was passing reliably on its own)
    showed a combined corrective pass could successfully fix the climax while
    still dropping the suspect-closure fix in the SAME pass -- the exact
    "bundled instructions dilute each other" failure mode this file has hit
    repeatedly at the prose level, just showing up here once the climax fix
    itself became reliable. Each corrective pass builds on the previous
    pass's output, not the untouched original, so fixing one concern doesn't
    undo progress on the other.

    After any suspect-closure corrective pass, a final ground-truth check
    (_find_names_missing_from_panels) verifies the model's own claim against
    the actual panel text instead of trusting it -- testing found a
    corrective pass self-report "unresolved_suspects: [], approved: true"
    while one named suspect never appeared in a single panel and another
    only appeared before the climax, in her introduction scene. This check
    doesn't ask the model anything; it's a plain string search, so it can't
    be fooled by an inaccurate self-report the way every other check in this
    function can.

    Also catches two narrower but sharper bugs found the same way: a panel's
    dialogue "speaker" field naming the wrong character, and a subtler
    variant where the tag is correct but the line's content/tone contradicts
    the narrated emotional state. Genre-agnostic, so it runs for every genre
    with a prose_story. Runs BEFORE check_panel_pacing_ai so pacing operates
    on a script that already matches the real story, not a fabricated one.

    debug_sink, if passed a dict, gets populated with every pass's raw
    signals (first pass plus any corrective passes) -- added after a retest
    showed a fix not landing with no way to tell from the final output alone
    whether detection or correction was the failure. Debug pipelines can
    surface this dict directly; production call sites simply don't pass it.
    """
    if not pages or not prose_story:
        return pages

    min_climax_panels = max(4, int(page_count * 0.15))

    def _climax_ok(r: dict) -> bool:
        return bool(r.get("climax_shown_as_scene")) and r["climax_panel_count"] >= min_climax_panels

    def _suspects_ok(r: dict) -> bool:
        return not r["unresolved_suspects"]

    def _beats_ok(r: dict) -> bool:
        return not r.get("missing_or_invented_beats") and not r.get("location_continuity_errors")

    def _passes(r: dict) -> bool:
        return _climax_ok(r) and _suspects_ok(r) and _beats_ok(r)

    def _snapshot(r: dict) -> dict:
        return {
            "approved": r.get("approved"),
            "climax_shown_as_scene": r.get("climax_shown_as_scene"),
            "climax_start_panel": r.get("climax_start_panel"),
            "climax_panel_count": r.get("climax_panel_count"),
            "unresolved_suspects": r.get("unresolved_suspects"),
            "missing_or_invented_beats": r.get("missing_or_invented_beats"),
            "location_continuity_errors": r.get("location_continuity_errors"),
        }

    if debug_sink is not None:
        debug_sink["min_climax_panels"] = min_climax_panels
        debug_sink["retries"] = []

    first_pass_fail_reasons: list = []
    result = _run_panel_fidelity_pass(pages, prose_story, hero_name, page_count, min_climax_panels, fail_reason_sink=first_pass_fail_reasons, use_claude=use_claude)
    if result is None:
        print("⚠️  Panel fidelity check failed — using original pages")
        if debug_sink is not None:
            debug_sink["first_pass"] = None
            debug_sink["first_pass_fail_reasons"] = first_pass_fail_reasons
            debug_sink["outcome"] = "first_pass_failed"
        return pages

    if debug_sink is not None:
        debug_sink["first_pass"] = _snapshot(result)

    if _passes(result):
        if result.get("approved"):
            print(f"✅ Panel fidelity check: APPROVED — climax has {result['climax_panel_count']} panels (needed {min_climax_panels}), all character resolutions present, panels match the prose, speakers and tone correctly attributed")
        else:
            print(f"✏️  Panel fidelity check: REVISED — climax has {result['climax_panel_count']} panels (needed {min_climax_panels}), beats: {result.get('missing_or_invented_beats')}, speaker fixes: {result.get('speaker_errors_fixed')}, tone fixes: {result.get('tone_mismatches_fixed')}")
        if debug_sink is not None:
            debug_sink["outcome"] = "first_pass_ok"
        return result["pages"]

    # At least one deterministic check failed. Run separate, single-purpose
    # corrective passes -- one per failing concern -- each building on the
    # previous pass's pages, not the untouched original.
    current = result
    working_pages = result["pages"]

    if not _climax_ok(current):
        correction_block = _build_climax_correction(current, min_climax_panels, page_count)
        print(f"⚠️  Panel fidelity check: climax has {current['climax_panel_count']}/{min_climax_panels} panels (shown_as_scene={current.get('climax_shown_as_scene')}) — running a climax-only corrective pass")
        fail_reasons: list = []
        retry = _run_panel_fidelity_pass(working_pages, prose_story, hero_name, page_count, min_climax_panels, correction_block=correction_block, fail_reason_sink=fail_reasons, use_claude=use_claude)
        if debug_sink is not None:
            entry = {"target": "climax", "correction_block": correction_block}
            entry["result"] = _snapshot(retry) if retry is not None else None
            if retry is None:
                entry["fail_reasons"] = fail_reasons
            debug_sink["retries"].append(entry)
        if retry is not None:
            current = retry
            working_pages = retry["pages"]
        else:
            print(f"⚠️  Panel fidelity check: climax-only corrective pass failed ({fail_reasons}) — keeping prior result for this concern")

    if not _beats_ok(current):
        correction_block = _build_beats_correction(current, page_count)
        print(f"⚠️  Panel fidelity check: invented/missing beats {current.get('missing_or_invented_beats')}, location continuity errors {current.get('location_continuity_errors')} — running a prose-fidelity-only corrective pass")
        fail_reasons: list = []
        retry = _run_panel_fidelity_pass(working_pages, prose_story, hero_name, page_count, min_climax_panels, correction_block=correction_block, fail_reason_sink=fail_reasons, use_claude=use_claude)
        if debug_sink is not None:
            entry = {"target": "beats", "correction_block": correction_block}
            entry["result"] = _snapshot(retry) if retry is not None else None
            if retry is None:
                entry["fail_reasons"] = fail_reasons
            debug_sink["retries"].append(entry)
        if retry is not None:
            current = retry
            working_pages = retry["pages"]
        else:
            print(f"⚠️  Panel fidelity check: beats-only corrective pass failed ({fail_reasons}) — keeping prior result for this concern")

    names_needing_verification: list = []
    if not _suspects_ok(current):
        names_needing_verification = list(current["unresolved_suspects"])
        correction_block = _build_suspect_correction(current, page_count)
        print(f"⚠️  Panel fidelity check: unresolved suspects {current['unresolved_suspects']} — running a suspect-closure-only corrective pass")
        fail_reasons: list = []
        retry = _run_panel_fidelity_pass(working_pages, prose_story, hero_name, page_count, min_climax_panels, correction_block=correction_block, fail_reason_sink=fail_reasons, use_claude=use_claude)
        if debug_sink is not None:
            entry = {"target": "suspects", "correction_block": correction_block}
            entry["result"] = _snapshot(retry) if retry is not None else None
            if retry is None:
                entry["fail_reasons"] = fail_reasons
            debug_sink["retries"].append(entry)
        if retry is not None:
            current = retry
            working_pages = retry["pages"]
        else:
            print(f"⚠️  Panel fidelity check: suspect-closure-only corrective pass failed ({fail_reasons}) — keeping prior result for this concern")

    # Ground-truth check, independent of the model's own self-report: testing
    # found a corrective pass claim "unresolved_suspects: [], approved: true"
    # while a named suspect never appeared in a single panel of the actual
    # output. Don't trust the claim -- read the real panel text instead.
    #
    # climax_start_panel is a per-book self-report, not a fixed point, and a
    # real failure showed it can land unreasonably late (23 of 24 panels) --
    # narrowing "at/after the climax" down to essentially the last panel or
    # two. A red-herring suspect properly cleared in the deduction beat that
    # immediately PRECEDES the climax (completely normal, even preferred,
    # mystery structure) then reads as "still missing" even though the story
    # resolved them correctly. A small backward buffer, scaled to page count,
    # still catches the original bug this check exists for (a suspect
    # resolved nowhere near the end, only in their early introduction scene)
    # while no longer flagging a resolution that lands just before the climax.
    verified_missing: list = []
    if names_needing_verification:
        climax_start = current.get("climax_start_panel") or 1
        climax_boundary = max(1, climax_start - max(3, page_count // 8))
        verified_missing = _find_names_missing_from_panels(working_pages, names_needing_verification, climax_boundary)
        if debug_sink is not None:
            debug_sink["ground_truth_check"] = {
                "checked_names": names_needing_verification,
                "climax_start_self_reported": climax_start,
                "climax_boundary_used": climax_boundary,
                "verified_still_missing": verified_missing,
            }
        if verified_missing:
            print(f"⚠️  Panel fidelity check: GROUND-TRUTH VERIFICATION caught a false self-report — {verified_missing} claimed resolved but never appear in any panel at/after panel {climax_boundary}")

    if _passes(current) and not verified_missing:
        print(f"✅ Panel fidelity check: SEQUENTIAL CORRECTIVE PASSES SUCCEEDED — climax {current['climax_panel_count']} panels, unresolved suspects: {current['unresolved_suspects']}, invented/missing beats: {current.get('missing_or_invented_beats')}")
        if debug_sink is not None:
            debug_sink["outcome"] = "sequential_retries_succeeded"
    elif verified_missing:
        print(f"⚠️  Panel fidelity check: self-report claimed success but ground-truth verification found {verified_missing} still missing — using the best available result")
        if debug_sink is not None:
            debug_sink["outcome"] = "ground_truth_verification_failed_used_best"
    else:
        print(f"⚠️  Panel fidelity check: still incomplete after corrective passes (climax {current['climax_panel_count']}/{min_climax_panels}, shown_as_scene={current.get('climax_shown_as_scene')}, unresolved: {current['unresolved_suspects']}, invented/missing beats: {current.get('missing_or_invented_beats')}) — using the best available result")
        if debug_sink is not None:
            debug_sink["outcome"] = "sequential_retries_incomplete_used_best"
    return working_pages


def _check_comedy_panel_pacing(
    pages: list,
    hero_name: str,
    page_count: int,
    use_claude: bool = False,
) -> list:
    """
    Comedy-only variant of check_panel_pacing_ai, split out because comedy's
    failure mode has TWO reference points instead of one. A single
    "find the climax panel, budget what's after it" check (the shared logic
    used for other genres) missed a real bug: testing found 19 of 48 panels
    (from right after the dove-catching fix through the very end) were all
    undifferentiated dance-party celebration, because the model picked one of
    the LAST panels -- a callback payoff gag -- as "the climax panel", which
    left everything BEFORE it exempt from budget enforcement even though most
    of that stretch was itself just party filler with no plot content. This
    checks the gap between where the corrective fix actually completes and
    where the punchline lands (plus anything after) as ONE combined budget,
    not just "after the single climax panel".
    """
    tail_budget = max(4, int(page_count * 0.12))
    pages_json = json.dumps(pages, indent=2)

    prompt = f"""You are a story editor reviewing the FINISHED PANEL SCRIPT for a {page_count}-panel COMEDY about {hero_name}. The panels have already been written and adapted from prose. Your ONLY concern is panel-count pacing from the moment the disaster gets fixed through the end — ignore art style, dialogue quality, and everything else.

PANELS (JSON array, one object per panel):
{pages_json}

YOUR TASK:
1. Find the FIX PANEL — the panel where the corrective action actually completes (the doves are caught, the cake is saved, whatever concrete external problem existed is resolved). This is the moment the disaster stops actively being a disaster.
2. Find the PUNCHLINE PANEL — the final joke/visual gag panel, ideally a callback payoff (a prop, object, or minor character's action). This is usually one of the last few panels, but it is NOT necessarily the very last one.
3. Count every panel from the FIX PANEL (exclusive) through the end of the comic, out of {page_count} total — this whole stretch (any celebration/reaction between the fix and the punchline, PLUS anything after the punchline) is the "tail."
4. The acceptable tail length is {tail_budget} panels or fewer. If the count from step 3 is within that budget, this is APPROVED as-is.

If the tail EXCEEDS {tail_budget} panels:
- This means there is undifferentiated celebration between the fix and the punchline (dancing, group celebration, repeated "everyone's having fun now" beats) padding out the page count. Compress the ENTIRE tail down to at most {tail_budget} panels, keeping only: one immediate reaction to the fix, the punchline panel itself (unchanged), and at most 1-2 panels after it.
- CUT: any dance/party sequence, any panel that just restates "everyone is having a great time now" with different words, generic group-celebration imagery, and any beat that repeats something already shown earlier in the comic.
- The panels you free up MUST be reallocated to EARLIER, pre-fix panels — split an existing setup/chaos moment into two sequential panels, add a close-up reaction panel for a beat that currently happens too fast, or show a step of the chaos escalating in more detail. Do NOT invent new content that isn't already implied by the surrounding panels, and do NOT re-show or restate any moment an earlier panel already depicted.

If the tail is unusually short (0-1 panels, meaning the punchline lands on the very last panel or one before it with zero room to land) — that's fine for comedy; do NOT pad it back out. A short, crisp tail is the goal, not a problem.

In all cases: the TOTAL panel count must remain exactly {page_count}. Renumber page_number sequentially from 1 to {page_count}. Every panel must keep the same JSON shape as the input: page_number, narration, dialogue, characters_present, setting, blocking, image_prompt. Preserve "setting" and "blocking" from the input on any panel you don't rewrite; update them only on panels whose location or character arrangement genuinely changed.

Return ONLY valid JSON in this exact shape:
{{"fix_page": <int>, "punchline_page": <int>, "approved": <true or false>, "pages": [ ...all {page_count} panel objects, revised if needed, otherwise identical to the input... ]}}"""

    system_msg = "You are a comedy editor. Fix only tail-length pacing between the fix and the punchline — nothing else. Return only valid JSON."
    try:
        if use_claude:
            content = _call_claude_json(prompt=prompt, system=system_msg, max_tokens=64000)
            result = json.loads(content)
        else:
            client = get_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=7500,
                timeout=120,
            )
            if response.choices[0].finish_reason == "length":
                print("⚠️  Comedy panel pacing check response was truncated (hit max_tokens) — keeping original pages untouched")
                return pages
            result = json.loads(response.choices[0].message.content)
        revised_pages = result.get("pages")
        if not revised_pages or len(revised_pages) != page_count:
            print(f"⚠️  Comedy panel pacing check returned {len(revised_pages) if revised_pages else 0} pages, expected {page_count} — using original pages")
            return pages
        if result.get("approved"):
            print(f"✅ Comedy panel pacing check: APPROVED — fix at panel {result.get('fix_page')}, punchline at panel {result.get('punchline_page')}, no changes needed")
        else:
            print(f"✏️  Comedy panel pacing check: REVISED — fix at panel {result.get('fix_page')}, punchline at panel {result.get('punchline_page')}, tail compressed to budget")
        return sorted(revised_pages, key=lambda p: p["page_number"])
    except Exception as e:
        print(f"⚠️  Comedy panel pacing check failed — using original pages: {e}")
        return pages


def check_panel_pacing_ai(
    pages: list,
    hero_name: str,
    genre: str = "adventure",
    page_count: int = 24,
    use_claude: bool = False,
) -> list:
    """
    A dedicated post-generation pass on the FINISHED panel script, not the prose.
    Testing showed that embedding a post-climax panel budget as one more rule
    inside generate_preview_pages_ai's already enormous prompt (character locks,
    outfit locks, nickname rules, location rules, style/lighting blocks, genre
    structure blocks) got diluted and wasn't reliably followed — the same
    "bundled instructions compete for attention" failure mode that motivated
    every other dedicated check_*_ai pass in this file. This one looks only at
    the finished {page_count} panels, finds the climax panel, and enforces a
    post-climax panel range (not just a ceiling): too many panels after the
    climax get compressed and reallocated to earlier pre-climax beats; too few
    (testing found a first version of this check happily approving a climax
    that landed on the very last panel, ending the story with zero aftermath)
    get padded back out with concrete, story-specific aftermath beats. Total
    panel count and numbering stay unchanged either way.
    """
    if not pages:
        return pages

    if genre == "comedy":
        return _check_comedy_panel_pacing(pages, hero_name, page_count, use_claude)

    if genre == "adventure":
        budget = max(3, int(page_count * 0.10))
        floor = min(2, budget)
        climax_def = "Find the CLIMAX PANEL — the single panel where the story's central conflict is actually DECIDED (the contest is won or lost, the mission item is retrieved, the confrontation ends, the mystery is solved). This is NOT a panel where characters merely set out to try, and NOT a celebration/reaction panel — it is the panel where the outcome itself happens."
        exceed_extra = ""
    elif genre == "horror_adult":
        budget = max(3, int(page_count * 0.12))
        floor = min(2, budget)
        climax_def = "Find the CLIMAX PANEL — the panel where the established weakness/banishment method actually takes mechanical effect against the entity (the object is destroyed, the ritual completes, the binding breaks) — not the panel where the confrontation merely begins, and not a calm reaction/aftermath panel."
        exceed_extra = """
- Horror specifically: the post-climax panels must still include the EXPLICIT resolution (what happened to the entity) AND a final unsettling "sting" detail as the very last panel — do not compress away the sting to save panels, and do not let the post-climax section become several panels of unqualified cozy safety with nothing unsettling left. If the current post-climax panels are all pure comfort/relief with no sting, keep the count the same but swap the final panel for one that adds a specific unsettling detail without contradicting the resolution."""
    elif genre == "horror":
        budget = max(3, int(page_count * 0.12))
        floor = min(2, budget)
        climax_def = "Find the RESOLUTION PANEL — the panel where the protagonist's concrete kind action (returning an object, fixing/building something, completing a specific task) actually happens, resolving things with the creature — not the panel where the reversal is merely discovered, and not a calm reaction/aftermath panel."
        exceed_extra = """
- Kid-friendly horror specifically: the post-resolution panels must still include the EXPLICIT outcome (the creature happy/at peace, its unfinished business resolved) AND a specific warm/fun final beat as the very last panel — the ending must feel completely safe, cozy, and fun, never spooky or ambiguous. If the current post-resolution panels are vague or still feel unresolved, keep the count the same but swap the final panel for one with a specific warm/fun detail."""
    else:
        budget = max(3, int(page_count * 0.2))
        floor = min(2, budget)
        climax_def = "Find the CLIMAX PANEL — the single panel where the story's central conflict is actually DECIDED (the contest is won or lost, the mission item is retrieved, the confrontation ends, the mystery is solved). This is NOT a panel where characters merely set out to try, and NOT a celebration/reaction panel — it is the panel where the outcome itself happens."
        exceed_extra = ""
    pages_json = json.dumps(pages, indent=2)

    prompt = f"""You are a story editor reviewing the FINISHED PANEL SCRIPT for a {page_count}-panel comic about {hero_name}. The panels have already been written and adapted from prose. Your ONLY concern is panel-count pacing after the climax — ignore art style, dialogue quality, and everything else.

PANELS (JSON array, one object per panel):
{pages_json}

YOUR TASK:
1. {climax_def}
2. Count how many panels come after the climax panel, out of {page_count} total.
3. The acceptable range is {floor}-{budget} panels after the climax. If the count from step 2 falls inside that range, this is APPROVED as-is.

If the count EXCEEDS {budget}:
- Compress the post-climax panels down to the {floor}-{budget} range, keeping only: the immediate concrete aftermath of the climax, one reaction shot from a named character, and one brief specific final beat that closes the story's central goal.
- CUT generic filler panels entirely — driving off into the sunset, walking away together, generic toasts, "we make a great team" / "here's to us" / "adventure awaits" style dialogue, and any beat that repeats something already shown earlier in the comic.{exceed_extra}
- The panels you free up MUST be reallocated to EARLIER, pre-climax panels — split an existing pre-climax moment into two sequential panels, add a close-up reaction panel for a beat that currently happens too fast, or show a step of an obstacle/journey in more detail. Do NOT invent new pre-climax plot content that isn't already implied by the surrounding panels. Do NOT re-show or restate any moment that an earlier panel already depicted — every reallocated panel must be a NEW moment (a zoom-in, a split, or a reaction), never a repeat.

If the count is BELOW {floor} (including a climax that lands on the very last panel, or one panel after it) — the ending is too abrupt:
- ADD panels after the climax, taking them from the LATEST pre-climax panels (compress or merge two adjacent early panels into one rather than deleting plot content), until the post-climax count reaches at least {floor}.
- The added panels must be concrete and specific to THIS story's outcome — the immediate physical aftermath of the climax, and/or one reaction shot from a named character tied to what just happened. Do NOT invent generic wind-down material (driving off, toasts, "we did it" reflection) — ground every added panel in something the prose already established.

In both cases: the TOTAL panel count must remain exactly {page_count}. Renumber page_number sequentially from 1 to {page_count}. Every panel must keep the same JSON shape as the input: page_number, narration, dialogue, characters_present, setting, blocking, image_prompt. Preserve "setting" and "blocking" from the input on any panel you don't rewrite; update them only on panels whose location or character arrangement genuinely changed.

Return ONLY valid JSON in this exact shape:
{{"climax_page": <int>, "approved": <true or false>, "pages": [ ...all {page_count} panel objects, revised if needed, otherwise identical to the input... ]}}"""

    system_msg = "You are a story editor. Fix only post-climax panel-count pacing — nothing else. Return only valid JSON."
    try:
        if use_claude:
            content = _call_claude_json(prompt=prompt, system=system_msg, max_tokens=64000)
            result = json.loads(content)
        else:
            client = get_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=7500,
                timeout=120,
            )
            if response.choices[0].finish_reason == "length":
                print("⚠️  Panel pacing check response was truncated (hit max_tokens) — keeping original pages untouched")
                return pages
            result = json.loads(response.choices[0].message.content)
        revised_pages = result.get("pages")
        if not revised_pages or len(revised_pages) != page_count:
            print(f"⚠️  Panel pacing check returned {len(revised_pages) if revised_pages else 0} pages, expected {page_count} — using original pages")
            return pages
        if result.get("approved"):
            print(f"✅ Panel pacing check: APPROVED — climax at panel {result.get('climax_page')}, no changes needed")
        else:
            print(f"✏️  Panel pacing check: REVISED — climax at panel {result.get('climax_page')}, post-climax panel count adjusted to budget")
        return sorted(revised_pages, key=lambda p: p["page_number"])
    except Exception as e:
        print(f"⚠️  Panel pacing check failed — using original pages: {e}")
        return pages


def check_nickname_usage_ai(
    pages: list,
    hero_name: str,
    additional_characters: list = None,
    page_count: int = 24,
    use_claude: bool = False,
) -> list:
    """
    Verifies that any customer-defined nickname is used at a natural
    cadence across the finished panel dialogue — roughly once every 5-6
    panels for a full book, or at least once for a short free preview —
    and adds or trims uses to bring it into range.

    Found via a real customer preview test: nicknames were defined for both
    the hero and a support character, but neither appeared anywhere across
    the 3 free-preview panels. generate_preview_pages_ai already contains a
    "must appear at least once" / "3-5 uses total" instruction, but it's
    just one line inside an already enormous prompt (character locks,
    outfit locks, location rules, genre structure blocks) -- the same
    "bundled instructions compete for attention" failure mode that
    motivated every other dedicated check_*_ai pass in this file (see
    check_panel_pacing_ai's docstring). A flat "3-5 total" target also
    doesn't scale with book length the way "every 5-6 panels" does, so the
    target here is computed from page_count instead of being a fixed
    number, and the check can both ADD uses (under target) and REMOVE
    clustered/excess ones (over target) to land in range.

    No-ops immediately if no nickname is actually defined for this project.
    """
    if not additional_characters:
        return pages

    nickname_pairs = []
    for char in additional_characters:
        name = char.get("name")
        if not name:
            continue
        if char.get("nickname_for_hero"):
            nickname_pairs.append({"speaker": name, "addressee": hero_name, "nickname": char["nickname_for_hero"]})
        if char.get("hero_nickname_for_them"):
            nickname_pairs.append({"speaker": hero_name, "addressee": name, "nickname": char["hero_nickname_for_them"]})

    if not nickname_pairs:
        return pages

    nickname_lines = "\n".join(
        f'- {p["speaker"]} calls {p["addressee"]} "{p["nickname"]}"' for p in nickname_pairs
    )
    pages_json = json.dumps(pages, indent=2)

    if page_count <= 4:
        # Free preview: too short for a real cadence, just make sure it's not zero.
        min_count, max_count = 1, 2
        cadence_note = (
            f"This is a short {page_count}-panel free preview — each nickname just needs to appear "
            f"once (twice at most), naturally. Do not force it into every line."
        )
    else:
        min_count = max(1, page_count // 6)
        max_count = max(min_count, page_count // 5)
        cadence_note = (
            f"For this {page_count}-panel book, each nickname should be used roughly {min_count}-{max_count} "
            f"times total — about once every 5-6 panels on average — spread across the WHOLE story "
            f"(early, middle, and late), not clustered into a handful of consecutive panels and not "
            f"concentrated only near the beginning or only near the end."
        )

    prompt = f"""You are a story editor reviewing the FINISHED PANEL SCRIPT for a {page_count}-panel comic about {hero_name}. Your ONLY job is checking nickname usage frequency — ignore art style, pacing, and everything else.

DEFINED NICKNAMES:
{nickname_lines}

TARGET FREQUENCY: {cadence_note}

PANELS (JSON array, one object per panel):
{pages_json}

YOUR TASK:
1. Count how many times each defined nickname above is used CORRECTLY across all {page_count} panels — the specific SPEAKER addressing the specific ADDRESSEE it belongs to (never counted if it's narration, or spoken by/to the wrong character).
2. If a nickname's count is BELOW {min_count}: add more natural uses. Pick EXISTING dialogue lines, spread across DIFFERENT parts of the story (don't add them all next to each other), where that speaker is already talking TO that addressee, and revise those lines to include the nickname — e.g. "Look at this beauty!" becomes "Look at this beauty, [nickname]!" Never invent a new line or panel from nothing — only add the nickname to dialogue that's already there.
3. If a nickname's count is ABOVE {max_count}, OR if its uses are clustered within just a few consecutive panels instead of spread out: remove the excess/clustered occurrences (revert those specific lines back to the character's real name or no address term at all) so the remaining uses land within {min_count}-{max_count} and are spread naturally across the story instead of feeling forced.
4. If a nickname's count already falls within {min_count}-{max_count} AND is reasonably spread across the story, leave every panel with that pair's dialogue completely untouched.
5. Never place a nickname on a line spoken by, or addressed to, the wrong character — nicknames are scoped to the exact pair they were defined for.
6. Most dialogue lines between any pair should have NO address term at all — the nickname is an occasional natural touch, not something said every time they speak to each other.

Return ONLY valid JSON in this exact shape:
{{"nickname_counts": {{"<nickname>": <final count after your edits>}}, "pages": [ ...all {page_count} panel objects, revised only where needed to bring nickname usage into the target range, otherwise byte-for-byte identical to the input... ]}}"""

    try:
        if use_claude:
            content = _call_claude_json(
                prompt=prompt,
                system="You are a story editor. Fix only nickname usage frequency (add if under target, trim if over-clustered) — nothing else. Return only valid JSON.",
                max_tokens=16000,
            )
            result = json.loads(content)
        else:
            client = get_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a story editor. Fix only nickname usage frequency (add if under target, trim if over-clustered) — nothing else. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=6500 if page_count > 24 else 3000,
                timeout=90,
            )
            if response.choices[0].finish_reason == "length":
                print("⚠️  Nickname usage check response was truncated (hit max_tokens) — using original pages")
                return pages
            result = json.loads(response.choices[0].message.content)

        revised_pages = result.get("pages")
        if not revised_pages or len(revised_pages) != page_count:
            print(f"⚠️  Nickname usage check returned {len(revised_pages) if revised_pages else 0} pages, expected {page_count} — using original pages")
            return pages
        counts = result.get("nickname_counts") or {}
        print(f"✏️  Nickname usage check: final counts {counts} (target {min_count}-{max_count})")
        # This round-trips the ENTIRE pages array (every field, not just
        # dialogue) through a fresh GPT-4o JSON generation -- found in
        # production to occasionally reintroduce the same stray-control-byte
        # corruption as _enforce_outfit_across_pages, even on an image_prompt
        # that was already clean going in, since the model re-emits the whole
        # object rather than truly leaving untouched fields byte-for-byte
        # identical. Sanitize every page's image_prompt here too.
        for p in revised_pages:
            if p.get("image_prompt"):
                p["image_prompt"] = _strip_control_chars(p["image_prompt"])
        return sorted(revised_pages, key=lambda p: p["page_number"])
    except Exception as e:
        print(f"⚠️  Nickname usage check failed — using original pages: {e}")
        return pages


def punch_up_prose_ai(
    prose_story: str,
    hero_name: str,
    page_count: int = 24,
    additional_characters: list = None,
) -> str:
    """
    Comedy punch-up pass: amplify the funniest scenes, cut dead-air panels,
    ensure every character drives the comedy, and verify the original goal lands in the punchline.
    Only called for comedy genre.
    """
    char_names = [hero_name] + [c["name"] for c in (additional_characters or [])]
    stay_until_panel = int(page_count * 0.8)

    prompt = f"""You are a comedy punch-up writer. A first draft of a story exists. Your job is to make it significantly funnier without changing what happens — sharpen, deepen, cut dead weight.

STORY TO PUNCH UP:
{prose_story}

PUNCH-UP CHECKLIST — apply each rule in order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — FUNNIEST SCENE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the whole story. Identify the 2 scenes with the most comedic potential.
For each: ask "is this the funniest possible version of this scene, or just the first version that came to mind?"
Rewrite each to be more specific, more extreme, more connected to the characters' actual personalities.
WRONG: "clay flies everywhere and hits someone"
RIGHT: "Ceiara's vase, now spinning at maximum speed, launches a fist-sized clay ball that strikes the instructor's meticulously assembled demonstration piece dead center — the instructor watches it collapse in slow motion with the expression of someone who has completely given up"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — DEAD SCENE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every scene after the first disaster must do at least ONE of:
  (a) raise the stakes — things get measurably worse
  (b) reveal something unexpected about a character
  (c) deliver a specific laugh — concrete and visual, not vague

Any scene that does NONE of these three: CUT IT or REPLACE IT.
Scenes where characters walk outside and have warm feelings about their day fail all three. Remove them.
Scenes where two characters say the same type of thing to each other five times in a row: compress to one or cut.

RECURRING-PROBLEM CHECK: scan the whole story for a single mechanism (one type of mishap, one joke) repeated with only the scale or target changed — e.g. "the sauce is hot and someone coughs" happening three or four times with a bigger batch or a new victim each time. That is repetition wearing escalation's clothes: the reader stops laughing once they can predict the shape of the next beat. If you find this pattern, rewrite the later instance(s) into a DIFFERENT kind of complication (a social consequence, an equipment failure, a miscommunication, a new character's involvement) that grows out of the earlier ones, rather than the same mechanism turned up louder.

VIRAL/FAME CALLBACK CHECK: if this story involves going viral, becoming famous, or social media success, scan for every line declaring it ("this is going viral," "we're famous," "look at these views," "this is legendary," "I can't wait to see what's next," "let's keep the momentum going," etc.). If more than ONE such line exists anywhere in the story, keep only the single strongest instance — placed at the actual resolution — and cut or rewrite every other occurrence into something that doesn't just re-announce the same realization in different words.
THIS INCLUDES THE WORD ITSELF, NOT JUST FULL DECLARATION LINES: a real production story used the word "viral" FIFTEEN times across 3500 words of prose -- mostly in narration, not even in dialogue -- which is what actually caused the problem: it wasn't any single repeated line, it was the theme of virality soaking the entire story so thoroughly that the panel artist adapting this prose into panels later kept reaching for ANOTHER viral/fame/legendary reaction at nearly every opportunity, inventing dialogue that doesn't even appear here, because the prose itself never let the theme rest. Count how many times "viral," "famous," "legendary," "trending," or equivalent words appear anywhere in this story, including narration. If it's more than 2-3 total, that's already too saturated — rewrite most of those instances to describe the SPECIFIC concrete thing happening (a number of people, a specific person's reaction, a specific physical consequence) instead of naming the viral/fame concept directly. The theme can still be true of the story; it just can't be the word on the page every few paragraphs.

DO NOT COLLAPSE A DELIBERATE PLANT-AND-PAYOFF PAIR: a character's catchphrase, or a planted line/object from early in the story, is SUPPOSED to appear exactly twice — once as the plant (said by the character it belongs to) and once as the payoff (usually a DIFFERENT character echoing it back, ironically, near the end). That is not the same failure as the recurring-mechanism or viral-hype repetition above — do not "compress to one" or cut either half of it. If you find a character's own catchphrase appearing only once (as someone else's payoff/echo, with the original character's own delivery missing or cut), that is broken, not fixed — restore the character's own original delivery of their line earlier in the story so the echo actually lands as a callback instead of reading as a random line someone else made up.

PRESERVE THE FIX (apply this AFTER any cutting above, as a final check): before finalizing, find the scene where {hero_name} actually takes a corrective action that repairs, salvages, or makes right the real external problem the chaos caused — this must exist and survive in your output. A real failure mode: the repeated-beat cuts above (RECURRING-PROBLEM CHECK, or Rule 2's "compress to one or cut") can end up deleting this fix scene along with the repetitive material sitting next to it, leaving the story jumping straight from peak chaos to an outside consequence (an authority showing up, a citation, a third party's verdict) with {hero_name} never actually doing anything to fix things. That is NOT an acceptable substitute for the fix — an external consequence landing on {hero_name} can still appear, but only AFTER {hero_name}'s own corrective action is shown on the page, never instead of it. If your edits have removed or weakened this scene, restore or rewrite it before returning the story — compress nearby repeated gags down to one instance each, but the fix itself is never one of the repeated things to cut.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — ORIGINAL GOAL RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find what {hero_name} set out to DO, MAKE, or WIN at the very start of the story.
The punchline MUST reveal the specific, concrete outcome of that original goal:
- If the goal was to break a time record → someone announces the actual time out loud as a number
- If the goal was to win a specific prize (gift card, ribbon, trophy) → the third party hands it over or names it explicitly — the prize itself must appear in the final line, not just a generic award title
- If the goal was to beat a specific rival → someone announces who actually won

A generic award title is NOT enough on its own:
WRONG: "awarded the title of most innovative piece" — the prize (gift card) never appears; the callback is incomplete
RIGHT: "awarded most innovative piece of the class. Here is your fifty-dollar gift card — redeemable here, for more pottery lessons." — the actual prize lands with irony; the callback is complete

The story ends on the third party's line. The hero does NOT react to it. Do not write the hero's response.
If the story currently ends without the specific prize or outcome stated explicitly: add it to the final line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — CHARACTER DYNAMIC RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check each of these characters: {', '.join(char_names)}
Is each one actively making choices that drive or worsen the comedy?
Or is any of them just watching and reacting while others cause chaos?

If a character is mostly a spectator: give them their own confident wrong-headed decision that compounds the disaster.
The funniest dynamic is multiple people each certain they know best, each making the others' situation worse simultaneously.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 — LOCATION AND LENGTH RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The story must stay in the primary setting for the equivalent of at least {stay_until_panel} of {page_count} panels.
If the story currently has characters leave the setting early and fill the remaining panels with outdoor conversation: cut all of it.
Replace any cut scenes with additional escalation INSIDE the setting — more chaos, more funny consequences, more character.
The story should end at or just as they are leaving the primary setting — not several scenes after.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — BANNED ENDING PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the final paragraph of the story. If it contains ANY of the following, rewrite the ending:

BANNED:
- "she/he realized that her/his true [talent/knack/gift] was..."
- "they shared a knowing glance"
- "as they left, [X] echoed behind them"
- "all it took was..."
- "some victories are won with..."
- "we make such a great team"
- "I can't wait to do this again"
- "that was such a great day"
- "next time we'll..."
- "and that was the day [character] learned/realized..."
- Any sentence about what the character learned, realized, or now understands about themselves
- Any warm arm-around-shoulder moment used as the final beat
- Any ending line that could work as the final line of a completely different story

THE ENDING MUST:
- Be a specific, visual punchline — something that HAPPENS or is REVEALED, not something a character reflects on
- Directly reference the original goal from the opening of the story (ironic success, funny failure, or accidental achievement)
- Make the reader laugh at the SITUATION, not feel warm about the CHARACTERS
- Be invented fresh for THIS story — do not reuse examples from these instructions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RETURN: The complete rewritten prose story. Do NOT explain your changes. Just return the improved story.
The story should be roughly the same length as the original or longer — do not compress it significantly.
Do not change the setting, premise, or characters — improve how they are written.
WORD COUNT GUARDRAIL: the checks above (RECURRING-PROBLEM CHECK, VIRAL/FAME CALLBACK CHECK, the Dead Scene Rule) ask you to cut or rewrite specific REPEATED LINES OR BEATS — a sentence here, a short exchange there, maybe one redundant paragraph. They do NOT ask you to shorten the story overall, and must never be used as license to compress whole scenes of non-repetitive material. If your rewrite is coming out noticeably shorter than the original story you were given, you have cut too much — go back and restore scene detail (setting, physical action, dialogue beats) that wasn't actually repetitive, rather than leaving the story thin.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SENTENCE RULE — CHECK THIS BEFORE SUBMITTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The punchline is delivered by a third party (staff member, judge, announcer). It is the last sentence.
Read the last two paragraphs of your response right now.
Delete any paragraph that contains:
- The hero reacting to, responding to, or quipping after the punchline announcement
- The hero linking arms / strutting out / walking away with a grin after the punchline
- "And with that..." / "And so..."
- Any sentence about what the day meant, what they learned, or what they now knew
- Any sentence about laughter echoing, memories being made, or stories being retold
- Any warm summary that could close any story ever written
The story ends the moment the third party delivers the punchline. Not one word after.
Write the punchline. Then stop."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional comedy writer doing a punch-up pass. Make the comedy more specific, more extreme, and better connected to the characters. Return only the improved story — no commentary."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.75,
            max_tokens=6500 if page_count > 24 else 3500,
            timeout=120,
        )
        choice = response.choices[0]
        result = _strip_meta_preamble(choice.message.content.strip())
        if choice.finish_reason == "length":
            # Hitting the token cap mid-rewrite silently truncates the story
            # mid-sentence -- found in production where a 48-panel comedy's
            # persisted prose cut off mid-scene with peak chaos, the desperate
            # measure, and the punchline never written at all. A truncated
            # rewrite is strictly worse than no rewrite: the panel adapter
            # would otherwise invent the back half of the book from nothing.
            print(f"⚠️  Comedy punch-up hit the token limit and was truncated — using original prose instead ({len(result)} chars generated)")
            return prose_story
        # Code-level guardrail, not just a prompt instruction -- the prompt already
        # says "roughly the same length or longer, do not compress significantly"
        # (see RETURN/WORD COUNT GUARDRAIL above) and that alone has proven
        # unreliable: found in production more than once, including a run that cut
        # a healthy ~3800-word draft down to ~800 words with no token-limit warning
        # at all, just the model choosing to compress severely. A prose that thin
        # starves the panel adapter of real material for a 48-panel book, forcing
        # it to invent repetitive filler to cover the gap -- this is the most
        # likely root cause of panel-script repetition appearing to vary so much
        # between otherwise-identical runs. Reject the rewrite in code rather than
        # trusting the instruction to hold every time.
        original_words = len(prose_story.split())
        result_words = len(result.split())
        if result_words < original_words * 0.7:
            print(f"⚠️  Comedy punch-up compressed the story from {original_words} to {result_words} words (>30% cut) — using original prose instead")
            return prose_story
        print(f"✏️  Comedy punch-up complete ({len(result)} chars, {result_words} words)")
        return result
    except Exception as e:
        print(f"⚠️  Comedy punch-up failed — using original prose: {e}")
        return prose_story


def generate_preview_pages_ai(hero_name: str, theme: str, logline: str, outline, character_brief: str, page_count: int = 4, personalization: dict = None, genre: str = "adventure", is_preview: bool = True, additional_characters: list = None, prose_story: str = None, npc_bible: list = None, use_claude: bool = False, chunk_range: Optional[tuple] = None, prior_context: Optional[str] = None, locked_outfit: Optional[str] = None, outfit_out: Optional[dict] = None):
    personalization_block = build_personalization_block(personalization or {})
    genre_style = GENRE_STYLES.get(genre, GENRE_STYLES["adventure"])
    if genre == "horror_adult":
        style_lock_block = """Style:
- Premium professional horror comic book illustration — bold ink linework with heavy black shadow shapes, like a mature horror graphic novel (think 30 Days of Night, Something is Killing the Children, or a Mike Mignola horror comic)
- COLOR IS INTENTIONALLY MUTED AND OPPRESSIVE: desaturated, high-contrast palette — deep blacks, cold blues and greens, with sickly green or blood-red as the ONLY saturated accent color in a scene
- Heavy use of negative space and black ink shadow shapes — large portions of the panel can be pure black shadow
- Clean precise ink lines around characters — NOT overly thick or chunky, but rendered with harsh, angular shadow shapes rather than soft cel-shading
- BACKGROUNDS MUST BE RICHLY DETAILED AND OPPRESSIVE — decayed, dirty, unsettling environments with intricate texture (peeling paint, grime, flickering signage, deep darkness in the corners of the frame)
- Natural facial proportions — natural eyes, nose, and mouth
- Natural human proportions — NOT chibi, NOT oversized head
- NOT anime, NOT glossy or oversized eyes
- Clearly illustrated (not a photograph)
- Consistent premium horror comic book illustration style across all pages
- SKIN TONE STAYS CONSISTENT EVERYWHERE ON THE BODY — face, hands, arms, and every visible body part must all match the SAME skin tone. Harsh accent lighting and deep shadow should darken or add colored tint, never wash a hand or limb out to a different or paler skin tone than the face.

Strictly avoid:
- Bright, cheerful, or evenly-lit scenes — this is horror, the lighting must feel threatening
- Any single body part (especially hands) rendered a lighter or different skin tone than the character's face — this is a rendering error, not a lighting effect
- Soft gradient blending or 3D render look
- Anime or manga style
- Chibi or oversized head proportions
- Photorealistic rendering
- Watercolor wash or painted brush strokes
- Overly thick or chunky outlines — lines must be precise, but shadow shapes can be bold and heavy"""
        lighting_block = """   Describe the emotional mood and use COMIC BOOK LIGHTING LANGUAGE ONLY — but for horror, lean into darkness and contrast.
   ALLOWED lighting descriptions: "stark horror-comic lighting with bold black ink shadow shapes", "single harsh light source cutting through deep black shadow", "high-contrast noir-style comic lighting", "sickly green or blood-red accent glow against near-total darkness", "flickering unstable light with hard-edged comic shadows", "desaturated oppressive comic palette with a single accent color"
   BANNED lighting descriptions (these cause the image generator to produce a photograph instead of an illustration):
   "natural light", "golden hour", "soft window light", "overcast daylight", "realistic shadows",
   "cinematic lighting", "ambient light", "warm afternoon sun", "soft shadows", "diffused lighting"
   The image generator reads lighting language as a signal for rendering mode — natural/photographic lighting = photograph.
   Comic book lighting language = illustration, even when it's dark. Always use the allowed terms above — darkness must still read as INK AND SHADOW, not a dim photograph."""
        style_anchor_line = "Illustrated horror comic art — oppressive shadow and ink, high contrast, characters match their reference photos exactly."
        horror_atmosphere_block = """=== HORROR ATMOSPHERE (MANDATORY FOR THIS GENRE) ===
- THE SETTING IS NEVER STATIC — most panels set in the haunted/threatening location must include at least one small uncanny environmental detail actively happening: a light flickering or dying, a door swinging or creaking open on its own, a sign buzzing and sparking, breath visible in air that's gone unnaturally cold, an object sliding or falling with no one touching it, distant unexplained sounds. Do not describe a location as if it were an empty, lifeless photo — something in the environment should feel wrong or alive in most shots.
- PHYSICAL STAGING OVER STATIC TALKING — even in dialogue-heavy scenes, characters must be physically reacting: backing away, gripping each other's arm, flinching toward a sound, glancing over their shoulder, pressing against a wall, recoiling. Two characters simply standing and facing each other to talk is NOT acceptable for a horror panel — their bodies must show tension or fear even while they speak.
- SCARE PANELS MUST LOOK DIFFERENT FROM CALM PANELS — when a panel is a scare/supernatural beat, push it further than a normal panel: extreme angles, something wrong with the antagonist's face/eyes/posture, harsher shadow, a genuinely unsettling image — not just the same framing as a conversation panel with a spooky word added to the mood line."""
    else:
        horror_atmosphere_block = ""
        style_lock_block = """Style:
- Premium professional comic book illustration — clean precise ink linework, flat color fills with cel-shading, like a modern DC or Image Comics graphic novel for younger readers
- COLOR IS CRITICAL: Bold, saturated comic book colors — bright blues, vivid greens, rich purples, warm reds, clean whites — high contrast and punchy
- Each scene must use a FULL COLOR SPECTRUM — backgrounds, clothing, and props must all have distinct, vibrant, contrasting colors
- Varied color temperature: cool blues and greens in backgrounds/sky, warm tones on skin, vivid colorful clothing — scenes must NOT be dominated by any single warm or amber hue
- Clean precise ink lines around all characters, objects, and major elements — medium-weight, sharp, professional linework — NOT overly thick or chunky
- Cel-shading: flat color fills with sharp 2-3 tone shadow zones — graphic and clean, NOT soft blended gradients
- BACKGROUNDS MUST BE RICHLY DETAILED — fully illustrated environments with intricate details, depth, and texture — no plain or empty backgrounds
- Natural facial proportions — natural eyes, nose, and mouth
- Natural human proportions — NOT chibi, NOT oversized head
- NOT anime, NOT glossy or oversized eyes
- Clearly illustrated (not a photograph)
- Consistent premium comic book illustration style across all pages

Strictly avoid:
- Soft gradient blending or 3D render look
- Muted, dull, washed-out, or desaturated colors
- Traditional oil painting palette — NO sepia, amber, brown, or golden-tone dominance
- Anime or manga style
- Chibi or oversized head proportions
- Photorealistic rendering
- Watercolor wash or painted brush strokes
- Overly thick or chunky outlines — lines must be precise and medium-weight"""
        lighting_block = """   Describe the emotional mood and use COMIC BOOK LIGHTING LANGUAGE ONLY.
   ALLOWED lighting descriptions: "bright well-lit comic illustration lighting", "clear crisp studio-style lighting",
   "vivid saturated comic colors with defined shadows", "bold side lighting with clear shadow zones",
   "strong directional lighting with crisp edges", "vibrant comic book palette with clear light and dark zones"
   BANNED lighting descriptions (these cause the image generator to produce a photograph instead of an illustration):
   "natural light", "golden hour", "soft window light", "overcast daylight", "realistic shadows",
   "cinematic lighting", "ambient light", "warm afternoon sun", "soft shadows", "diffused lighting",
   "dramatic overhead hard shadows", "deep shadows" (too extreme — flattens faces in close-up shots)
   The image generator reads lighting language as a signal for rendering mode — natural/cinematic lighting = photograph.
   Comic book lighting language = illustration. Always use the allowed terms."""
        style_anchor_line = "Illustrated comic art — vivid saturated colors, clear crisp linework, characters match their reference photos exactly."
    p = personalization or {}
    additional_chars_block = ""
    if additional_characters:
        additional_chars_block = "\n=== ADDITIONAL NAMED CHARACTERS (MUST APPEAR IN EVERY PANEL) ===\n"
        for char in additional_characters:
            desc_parts = []
            if char.get("age"):
                desc_parts.append(f"{char['age']}-year-old")
            if char.get("gender"):
                desc_parts.append(char["gender"])
            is_char_adult = bool(char.get("age") and int(char["age"]) >= 18)
            desc_parts.append("adult" if is_char_adult else "child")
            relationship = char.get("relationship", "")
            rel_suffix = f" — {hero_name}'s {relationship}" if relationship else ""
            additional_chars_block += f"- {char['name']}: {' '.join(desc_parts)}{rel_suffix}\n"
        char_names = [c["name"] for c in additional_characters]
        char_names_str = " and ".join(char_names) if len(char_names) <= 2 else ", ".join(char_names[:-1]) + ", and " + char_names[-1]
        name_rules = " ".join(
            f"Always write '{n}' — NEVER 'her husband', 'her companion', 'her friend', or any other substitute for this name."
            for n in char_names
        )
        nickname_rules = []
        for char in additional_characters:
            if char.get("nickname_for_hero"):
                nickname_rules.append(
                    f"'{char['nickname_for_hero']}' is ONLY used by {char['name']} when directly addressing {hero_name} — "
                    f"NEVER used when speaking to anyone else (not to a ghost, spirit, villain, stranger, or any third party). "
                    f"On the rare occasion {char['name']} uses an address term FOR {hero_name} specifically, "
                    f"use '{char['nickname_for_hero']}' — never her full name. "
                    f"Most lines have no address term at all — do not add '{char['nickname_for_hero']}' to lines directed at other characters."
                )
            if char.get("hero_nickname_for_them"):
                nickname_rules.append(
                    f"'{char['hero_nickname_for_them']}' is ONLY used by {hero_name} when directly addressing {char['name']} — "
                    f"NEVER used when speaking to anyone else (not to a ghost, spirit, villain, stranger, or any third party). "
                    f"On the rare occasion {hero_name} uses an address term FOR {char['name']} specifically, "
                    f"use '{char['hero_nickname_for_them']}' — never his full name. "
                    f"Most lines have no address term at all — do not add '{char['hero_nickname_for_them']}' to lines directed at other characters."
                )
        nickname_block = (
            "NICKNAME RULES (applies when an address term is used — see DIALOGUE ADDRESS RULE for frequency):\n"
            + "\n".join(f"- {r}" for r in nickname_rules) + "\n"
        ) if nickname_rules else ""
        additional_chars_block += (
            f"PRESENCE RULE (CRITICAL — non-negotiable): {char_names_str} must be explicitly named and physically present "
            f"in EVERY single panel's image_prompt and narration — all {len(char_names)} of them, in every panel. "
            f"They travel with {hero_name} throughout the entire story and are NEVER absent from any scene. "
            f"EXACT NAMES REQUIRED: {name_rules} "
            f"Every panel's image_prompt must name and place these characters in the scene.\n"
            "NARRATION RULE — RELATIONSHIP LABELS (CRITICAL): When narration refers to these characters, always use their "
            "correct relationship label. NEVER call them 'friends' if their relationship is husband, wife, spouse, partner, "
            "sibling, coworker, or anything other than friend. "
            f"When referring to the group, be specific — NEVER just '{hero_name} and her friends'.\n"
            "SUPPORTING CHARACTER OUTFIT LOCK: Each supporting character wears the SAME outfit in every single panel — "
            "never change their clothing between scenes. The outfit they wear in panel 1 is locked for the entire story.\n"
            "FRAMING RULE: Every image_prompt must show all characters full body from head to toe — "
            "never describe a shot that would cut off legs, feet, or any part of the body.\n"
            f"{nickname_block}"
        )

    audience_age = p.get("age") or 10
    is_adult = audience_age >= 18
    if is_adult:
        writer_desc = "personalized comic books for adults. Write mature, sophisticated content — complex themes, adult humor, real-world stakes, and adult vocabulary."
    else:
        writer_desc = "personalized comic books for kids."

    setup_end = page_count // 6
    climax_start = page_count * 3 // 4 + 1
    resolution_start = page_count * 11 // 12 + 1
    first_disaster_end = setup_end + max(2, page_count // 6)
    escalation_end = setup_end + max(4, page_count // 3)
    chaos_end = climax_start - 1
    early_mid_end = setup_end + max(2, (climax_start - setup_end - 1) // 3)
    mid_end = setup_end + max(3, 2 * (climax_start - setup_end - 1) // 3)
    if is_preview:
        purpose_block = f"""CRITICAL — PREVIEW PURPOSE:
These {page_count} pages are a TEASER shown to the customer BEFORE they purchase. They must:
- Cover only the OPENING of the story — the setup and the inciting incident (the moment the adventure begins)
- End on a cliffhanger or moment of excitement/mystery that makes the reader desperate to know what happens next
- NEVER resolve the main conflict, NEVER show the ending, NEVER wrap up the story
- Leave the reader asking "What happens next?!" — the answer is in the full 24-page book they haven't bought yet
- Think of it like a movie trailer: tease the adventure, stop right when things get interesting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAME THE THREAT — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the outline and identify WHY this story is dangerous or high-stakes — usually a named antagonist, rival, or specific threat (a person, creature, or force the outline gives a name or clear identity to). That antagonist/threat MUST be explicitly named or clearly shown by panel {min(2, page_count)} — do not let the panels stay generic ("a poker game", "a spooky noise") when the outline names who or what is actually behind it. A teaser that hides the outline's own antagonist isn't a hook, it's just vague.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAUSE BEFORE CONSEQUENCE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do not jump straight from the story's opening image to its inciting incident with no bridge. If the outline implies a decision, action, or transition happens between the opening and the incident (e.g. the hero chooses to do something, goes somewhere, gets drawn in), show or narrate that step — even briefly — so consequence follows visibly from cause. Two adjacent panels must never read as unrelated beats stitched together; each panel should be the clear next thing that happens because of the panel before it."""
    elif prose_story:
        stay_until_panel = int(page_count * 0.8)
        if genre == "adventure":
            location_rule_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION RULE — ADVENTURE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This prose establishes exactly TWO legitimate settings: the starting location (where the story opens) and the mission's actual destination (wherever the prose sends the protagonist to pursue the mission — a forest, mountains, a cave, wherever the prose's own obstacle/discovery/climax scenes actually take place). Adventure REQUIRES that journey to appear on the page — a comic that never leaves the starting location has failed to adapt the story it was given, no matter how faithfully it copies the dialogue.
- Follow the prose's OWN scene order and locations exactly — when the prose moves from the starting location to the destination, the panels must move there too, at roughly the same point in the story.
- Do NOT keep characters parked in the starting location for the whole comic just because it feels safer — if the prose's obstacles, discovery, or climax happen at the destination, draw them there.
- Do NOT invent a THIRD, unrelated location that isn't in the prose.

IF THE PROSE RUNS SHORT — not enough distinct moments to fill {page_count} panels:
DO NOT invent new locations, and DO NOT re-show a moment that an earlier panel already depicted. Instead, expand the LATEST scenes you have (the climax's aftermath, the resolution):
  - Zoom in: add a close-up reaction panel from a scene already in the prose
  - Split: break one described moment into 2–3 sequential panels showing the action step by step
  - Reaction: show what each character's face looks like at a key moment the prose mentions
Every panel must be traceable to a specific moment in the prose, and every panel's moment must be NEW — never a repeat of, or a jump backward to, something an earlier panel already showed.
If you cannot point to where in the prose a panel comes from — replace it with a zoomed-in version of the resolution, never with a duplicate of an earlier scene."""
        else:
            location_rule_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCATION RULE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Characters must remain in the PRIMARY SETTING (the location established in the opening panels) until at least panel {stay_until_panel} of {page_count}.
DO NOT generate any panel before panel {stay_until_panel} that shows characters:
- Outside, walking through streets or a neighborhood
- At home, in a car, or in any building other than the primary setting
- Anywhere new that was not established in the prose

IF THE PROSE RUNS SHORT — not enough distinct moments to fill {page_count} panels:
DO NOT invent new locations, and DO NOT re-show a moment that an earlier panel already depicted. Instead, expand the LATEST scenes you have (the climax's aftermath, the resolution):
  - Zoom in: add a close-up reaction panel from a scene already in the prose
  - Split: break one described moment into 2–3 sequential panels showing the action step by step
  - Reaction: show what each character's face looks like at a key moment the prose mentions
Every panel must be traceable to a specific moment in the prose, and every panel's moment must be NEW — never a repeat of, or a jump backward to, something an earlier panel already showed.
If you cannot point to where in the prose a panel comes from — replace it with a zoomed-in version of the resolution, never with a duplicate of an earlier scene."""

        min_climax_panels = max(4, int(page_count * 0.15))
        purpose_block = f"""CRITICAL — FULL COMIC ADAPTATION:
These {page_count} panels are the COMPLETE comic book. You are adapting a finished prose story into panels — every plot point, character name, and emotional beat comes from that story.

ADAPTATION RULES:
- Follow the prose story faithfully — adapt it, do NOT rewrite it or invent new plot elements
- Distribute the story chronologically across all {page_count} panels, giving each panel ONE specific moment
- Named characters and their personality traits from the prose must carry through every panel
- Key dialogue lines from the prose should appear in the panels they belong to
- Scenes described in the prose that span multiple pages should each get their own panel(s) — don't compress the climax into one panel
- EVERY plot thread must be resolved by panel {page_count}
- Do NOT end on a cliffhanger

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT FORWARD CHRONOLOGY — NO REPEATS, NO RESTARTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time in the panels only moves FORWARD. Once a moment from the prose has been depicted in a panel, it is DONE — it must never be shown again in a later panel, and the story must never jump backward to an earlier point in the timeline (re-showing the opening/setup, replaying an obstacle that's already been resolved, or restating the mission after it's already been accomplished). This applies even after the story has reached its resolution: once the characters are safe/resolved, do not send them back into danger from an earlier scene to fill remaining panels.
This failure mode happens specifically when the main story compresses into fewer panels than {page_count} and there are panels left to fill: the fix is NEVER to re-adapt an earlier part of the prose a second time. Instead, use the zoom-in/split/reaction techniques described below on the LATEST scene you have (the climax's aftermath or the resolution) — add more granular detail to the ENDING, never a duplicate of something already shown.
Before finalizing, scan your own panel list in page_number order: if any panel's setting/event/character-state matches an EARLIER panel's setting/event/character-state that the story has already moved past (e.g. characters already reached safety in panel 35, but panel 44 shows them setting out on the journey again), delete that panel and replace it with an original moment from later in the story, or a zoomed-in/reaction panel on the resolution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NO REPEATED DECLARATIONS — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A real production book stretched a single scene from the prose across many panels by inventing dialogue that just restated the same excitement over and over in different words — "Maybe this is the viral moment we needed!", "This is going to be legendary!", "I can't believe this is happening!", "This is just the beginning!", "I can't wait to see the video!" — eight-plus panels in a row of characters reacting to the SAME realization instead of the story actually moving forward. This is not adaptation, it's padding, and it happens specifically when a chunk of panels has to cover more ground than the prose gives you concrete material for.
Before finalizing, scan every panel's dialogue AND narration box in page_number order: if two or more panels have a character announcing, or a narration box stating, essentially the same realization, triumph, or reaction as an earlier panel (even worded differently), that is a repeated declaration — this applies to narration just as much as dialogue; a narration box that keeps restating "the chaos was escalating" or "this was turning into something big" in different words across several panels is the same failure as a repeated line of dialogue. Cut every instance but the single best one, and replace the rest with the zoom-in/split/reaction techniques above — a NEW specific detail from the prose, not another sentence reacting to the same thing a second way.
This applies with extra force to any "we're famous / this is viral / this is legendary / people love it" type beat — the prose earns that moment once; the panels must not turn one sentence of prose into five panels (in dialogue OR narration) of saying so.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER LOCATION CONTINUITY — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before finalizing, check every character's physical position in each panel against the PRECEDING panel. A character cannot leave, arrive, enter, return, or change locations between two adjacent panels without an intervening panel or narration beat that actually shows that movement happening. Two common failure patterns to check for:
  (a) A character who exited/left in one panel (or was last placed somewhere specific) simply appears back in the scene a few panels later with no panel showing them return.
  (b) The same character is shown "arriving" or "bursting in" to the same location twice, because two different source scenes both happened to describe an entrance and got adapted as if they were the same one-time event repeated.
If a character's location changes between adjacent panels, there must be a specific reason visible in the narration/setting (they walked to a new room, they were pulled outside, etc.) — never an unexplained jump. When compressing or reordering panels, re-verify this chain from panel 1 to {page_count} before finalizing.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLIMAX PANEL BUDGET — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before distributing panels, find the climax in the prose — the scene where the central conflict is actually decided (the confrontation, the accusation and what it triggers, the struggle, the moment of truth) — and reserve panels for it FIRST, not last.
- The climax must begin by roughly panel {climax_start} of {page_count}.
- The climax AND its immediate consequence (the culprit's reaction, the outcome, what changes because of it — not just the accusation or the setup for it) must occupy AT LEAST {min_climax_panels} panels. An accusation or confrontation with no shown reaction or consequence is incomplete — that's the single most common failure in this kind of adaptation, so protect against it deliberately.
- If earlier setup/investigation panels don't fit alongside a properly-sized climax, COMPRESS THE SETUP, not the climax — merge or cut earlier panels rather than shortchanging the ending. Losing an early panel is a much smaller loss than losing the climax.
{location_rule_block}"""
        if genre == "comedy":
            purpose_block += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUNCHLINE PACING — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The prose ends on a PUNCHLINE — a single ironic or funny announcement delivered by a third party.
Once that punchline panel is drawn, you have AT MOST 4 panels remaining.
Use them ONLY for:
  1. A close-up reaction shot of {hero_name}'s face
  2. A group reaction (the other characters' faces/body language)
  3. One callback visual (something that echoes the opening setup — the goal, the sign, the object)
  4. One final exit panel (characters leaving the setting, or one last funny image)

BANNED after the punchline:
- New dialogue topics or conversations that weren't in the prose
- New plans, new adventures, or "what should we try next?" exchanges
- Repeated lines that already appeared earlier in the comic
- Characters walking through a neighborhood, street, or outdoor area
- Warm reflection scenes about what they learned

The joke has landed. Close the curtain in 4 panels or fewer. Do not write an encore."""
        else:
            climax_panel_budget = max(3, int(page_count * 0.2))
            purpose_block += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST-CLIMAX PANEL BUDGET — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find the panel where the story's central conflict is actually decided — the contest is won or lost, the mission succeeds or fails, the confrontation ends. You have AT MOST {climax_panel_budget} panels after that point, out of {page_count} total.
Use them ONLY for:
  1. The immediate, concrete aftermath of the climax (the specific consequence — not a mood)
  2. One reaction shot from a named character tied to what just happened
  3. One brief, specific final beat that closes the story's central goal

BANNED in post-climax panels:
- Generic travel/transition imagery with no new plot content ("driving off into the sunset," "walking away together," "heading home")
- Generic celebratory dialogue that could belong to any story ("we make a great team," "here's to us," "adventure awaits," "let's make every moment count")
- New activities, plans, or scenes that were not in the prose's resolution
- Repeating a beat, dialogue line, or image that already appeared earlier in the comic

If the prose's resolution section is longer than this budget allows, COMPRESS it — pick the single most specific, concrete closing beat and cut the rest. Do NOT pad remaining panels with generic material just to hit the panel count; if you run short, expand an EARLIER, pre-climax scene from the prose instead (per the "IF THE PROSE RUNS SHORT" rule above) — never invent new post-climax content to fill space."""
    elif genre == "comedy":
        purpose_block = f"""CRITICAL — COMEDY STORY STRUCTURE (this is mandatory — follow it page by page):
These {page_count} pages are the COMPLETE comic. Comedy REQUIRES escalating chaos — the story must get FUNNIER and MORE OUT-OF-CONTROL as it progresses. Every problem must compound the previous one. The chaos on page {chaos_end} must feel completely different (worse, funnier) than the chaos on page {setup_end + 1}.

MANDATORY ESCALATION ARC:
- Pages 1-{setup_end}: SETUP — introduce {hero_name}, the setting, and the comedy premise. Establish what she is trying to do and why it personally matters to her. Plant the seeds of disaster (the thing that will go wrong is already visible, but hasn't exploded yet).
- Pages {setup_end + 1}-{first_disaster_end}: FIRST DISASTER — something specific and concrete goes hilariously wrong. This is the FIRST BIG LAUGH. It must be caused by {hero_name}'s specific personality — not random bad luck. The audience must laugh HERE.
- Pages {first_disaster_end + 1}-{escalation_end}: ESCALATION — every attempt to fix the first disaster makes it worse. Named side characters' personality quirks are ADDING to the chaos, not helping. Each page must be more chaotic than the last.
- Pages {escalation_end + 1}-{chaos_end}: PEAK CHAOS — everything falls apart at once. Multiple earlier problems collide. The situation looks completely unrecoverable. This is the FUNNIEST SCENE in the comic. The audience cannot predict how they get out of this.
- Pages {chaos_end + 1}-{resolution_start - 1}: THE COMEBACK — they solve it using an unexpected solution that connects back to something established in the setup. The solution can still be funny — dignity is not required.
- Pages {resolution_start}-{page_count}: VICTORY + PUNCHLINE — they succeed, then there is a final comedic beat or ironic callback. The last page must end on a LAUGH, not on "we did it."

WHAT THIS STORY IS NOT:
- NOT a series of equal-stakes scenes: puzzle → solve → puzzle → solve → puzzle → solve. That is a to-do list, not a comedy.
- NOT a story where the ending is "they succeeded and felt good about themselves" — there must be a punchline.
- EVERY plot thread must be resolved by page {page_count}.
- Do NOT end on a cliffhanger."""
    else:
        purpose_block = f"""CRITICAL — FULL STORY PURPOSE:
These {page_count} pages are the COMPLETE comic book. The story MUST escalate — each section must feel higher-stakes and more urgent than the one before. A story where every page carries the same level of tension is a forgettable story.

MANDATORY STORY ARC:
- Pages 1-{setup_end}: SETUP — introduce {hero_name}, the central problem, and what she personally stands to lose. By page {setup_end} the reader must understand WHY this matters to her specifically.
- Pages {setup_end + 1}-{early_mid_end}: EARLY RISING ACTION — {hero_name} begins addressing the problem. Initial efforts show progress but reveal the challenge is harder than it appeared. Introduce the named side characters who will matter throughout the story.
- Pages {early_mid_end + 1}-{mid_end}: MIDPOINT REVERSAL — something goes significantly wrong. A plan fails, stakes increase, a complication is revealed, or {hero_name} discovers the situation is much worse than she thought. The story CANNOT stay at the same tension as the early section — it must get harder here. This is not optional.
- Pages {mid_end + 1}-{climax_start - 1}: LATE RISING ACTION — {hero_name} pushes forward despite the setback. Tension and urgency mount toward the climax. The reader should feel the pressure building.
- Pages {climax_start}-{resolution_start - 1}: CLIMAX — the biggest, most intense scene in the story. {hero_name} faces the central challenge head-on. The outcome must feel genuinely uncertain until this moment.
- Pages {resolution_start}-{page_count}: RESOLUTION — {hero_name} succeeds. Every plot thread resolves. End with a specific, earned beat — not a generic "everything is fine now."

ESCALATION IS NON-NEGOTIABLE:
- The story on page {mid_end} must feel harder and more urgent than the story on page {setup_end + 2}
- WRONG: Equal-stakes scenes throughout, then a sudden resolution — this is a flat line, not a story
- RIGHT: Each section boundary is a visible step up in stakes, urgency, and difficulty
- EVERY plot thread must be resolved by page {page_count}
- Do NOT end on a cliffhanger — give the reader a complete, satisfying ending"""
    if genre == "romance":
        safety_rules = """
CONTENT RULES (ROMANCE — R-RATED, 18+):
- Deep romantic storylines, passionate emotional connections, desire, longing, and intimacy are encouraged
- Physical affection is unrestricted within the R-rated boundary: kissing, embracing, holding, sensual tension
- Romantic and sexual tension between characters can be explicit in emotion and suggestion — charged moments,
  desire expressed openly, intimate scenes that convey heat and passion
- Fade-to-black is allowed for sexual encounters — suggest what happens without graphic explicit description
- Strong adult language is permitted when it serves the emotional tone
- Conflict can come from jealousy, desire, past relationships, betrayal, temptation, or emotional vulnerability
- Target tone: a steamy R-rated romance novel or film — passionate, emotionally intense, adult
- NO graphic explicit sexual acts described in detail — sensual and suggestive, not pornographic
"""
    elif genre == "horror_adult":
        safety_rules = """
CONTENT RULES (HORROR — ADULT, 18+):
The story may contain intense supernatural danger, disturbing psychological themes, frightening imagery, moderate-to-strong violence where appropriate, blood or injury when narratively justified, and sustained mortal peril. Do NOT rely on gore alone to create horror. Prioritize dread, disturbing implications, psychological vulnerability, escalating danger, unsettling imagery, and consequences. The protagonist should face a credible possibility of death or severe harm.

WHAT MAKES THIS WORK (any combination is valid — match what THIS story's threat actually is):
- The supernatural threat must be genuinely dangerous, not just "eerie" or "unsettling" — characters should be afraid for their lives or sanity, not just creeped out.
- If the threat is physical/violent by nature: show real weight — wounds, pain, shock, blood where the scene actually calls for it. Do not soften an attack into "a dark stain" or characters shaking it off unaffected.
- If the threat is psychological/atmospheric by nature (manipulation, hallucination, dread, a slow unraveling of sanity or trust): that is equally valid R-rated horror — lean into disturbing implications, escalating wrongness, and genuine psychological vulnerability rather than manufacturing violence that doesn't fit the premise.
- Either way, at least one scene must put the protagonist in a CREDIBLE possibility of death or severe harm — not a scare that resolves the instant it appears.
- The climax must feel like real mortal/psychological stakes, not a peaceful conversation.

WHAT TO AVOID:
- Do NOT use "strange noises" or "unsettling feelings" as the ENTIRE horror experience with nothing underneath — dread needs to escalate toward something.
- Do NOT have characters discover horrors off-screen and just react to them — show the horror directly.
- Do NOT resolve tension with dialogue and goodwill alone — earn the resolution through real confrontation.
- Do NOT write "she felt a chill" when the scene calls for genuine terror.
- Do NOT force blood/violence into a story whose threat is fundamentally psychological just to check a box — that reads as tonally false. Match the intensity to what this specific entity actually does.

IMAGE PROMPTS FOR HORROR:
- Horror image_prompts must describe atmospheric, oppressive scenes — dark, threatening environments matching the story's actual danger
- If the scene involves physical violence or injury, describe the blood/wounds/aftermath explicitly rather than softening it
- If the scene is psychological/atmospheric, lean on unsettling visual detail instead — wrong angles, impossible details, dread-inducing environmental wrongness
- Use lighting to create dread: single light source cutting through deep darkness, sickly green or red tones, harsh shadows
- Characters in horror scenes should show genuine fear in their expressions and body language — gripping each other, backing away, trembling

HORROR STORY DEPTH (CRITICAL — this is what separates good horror from surface-level generic):
- The supernatural element (spirit, curse, haunting) MUST have a SPECIFIC, DETAILED backstory — not vague
  - Who were they? Give them a name, an era, a specific life story
  - What exactly happened to them? A specific tragic event with concrete details — murder method, betrayal specifics, exact injustice
  - Why are they still here? A specific unresolved injustice, betrayal, or unfulfilled mission
  - What do they actually want from the protagonist — and why THIS person specifically?
- The history reveals must feel like genuine discoveries with weight — not passing comments
- Each clue or reveal should raise more questions before answering them — build dread progressively
- The final confrontation must resolve the specific backstory — not just "the spirit found peace"
- NEVER write vague, surface-level horror like "there were spirits here" or "the land has a dark history" — be SPECIFIC
"""
    elif is_adult and genre == "comedy":
        safety_rules = """
CONTENT RULES (COMEDY — ADULT, 18+):
- This is adult comedy for a grown-up audience — NOT a kids' cartoon, NOT family-friendly
- PG-13 humor is the target — edgy, self-deprecating, and genuinely funny for adults
- Adult situations (relationship dynamics, ego, social awkwardness, adulting fails) are the comedy engine
- Characters can be flawed, petty, embarrassing, or make adult mistakes
- Strong language is NOT required but mild adult language (damn, hell, crap) is fine if it fits
- NO sexual content or graphic violence — keep it PG-13 comedy, not R-rated

COMEDY MUST CONTAIN ACTUAL COMEDY — THIS IS THE MOST IMPORTANT RULE IN THIS ENTIRE PROMPT:
- Every story must contain at least 3 SPECIFIC, CONCRETE funny moments where something goes wrong in a named, visible, describable way and the characters react in a specific ridiculous way
- Vague descriptions ("things get complicated," "chaos ensues," "it doesn't go as planned") are NOT comedy — they are placeholders. Name the specific thing that goes wrong.
- WRONG: "Things get complicated at the escape room." — this is not funny, it is abstract
- RIGHT: "Ceiara confidently yanks what she assumes is the puzzle lever — triggering the emergency sprinkler system and soaking every team in the room, including Marcus, who is mid-commentary." — specific, visual, funny
- The image_prompt for every funny moment must SHOW the visual absurdity: exaggerated expressions, chaotic environment, specific funny detail (wet hair, fallen stack of props, alarm light flashing)

COMEDY COMES FROM THIS CHARACTER'S SPECIFIC PERSONALITY — NOT GENERIC BAD LUCK:
- Read the customer's personality traits before writing any plot. The comedy must emerge from WHO THEY ARE.
- If they're competitive: the comedy comes from their competitiveness backfiring spectacularly
- If they're a planner: the comedy comes from their elaborate plan failing at the worst possible moment
- If they're confident: the comedy comes from their confidence being hilariously misplaced, and then they have to eat humble pie
- If they're anxious: the comedy comes from their anxiety being simultaneously wrong AND right in the worst way
- WRONG: Generic bad luck comedy that could happen to any character
- RIGHT: Comedy that ONLY WORKS because of this specific person — "Because Ceiara refused to admit she misread the clue, she committed to the wrong answer with total confidence and accidentally set off every alarm in the building"

NAMED SIDE CHARACTERS WITH PERSONALITY (REQUIRED — this is non-negotiable):
- Every NPC who appears in more than one scene MUST have an actual first name — NEVER a label like "the competitive couple" or "the therapy couple"
- Each named NPC must have ONE DEFINING PERSONALITY TRAIT that creates specific, recurring comedic friction
- The NPC's personality must CLASH with the hero's in a funny way that makes each scene MORE chaotic
- Their trait must recur as a RUNNING BIT across multiple scenes — it keeps coming up and getting worse
- WRONG: "There is a competitive couple who challenges them." — no name, no personality, zero comedic value
- RIGHT: "Marcus and Tasha are escape room regulars. Marcus narrates his own performance in the third person like a sports commentator ('Marcus studies the clue board — and LOCKS IN his answer!'). Tasha is his self-appointed coach, whispering strategy at him while he ignores her." — named, specific, funny on arrival, creates escalating comedy

BANNED DIALOGUE PHRASES — NEVER WRITE ANY OF THESE, NO EXCEPTIONS:
- "We did it!" (too generic — could end any story ever written)
- "Let's finish this together!" (too generic)
- "Teamwork makes the dream work!" (motivational poster, not dialogue)
- "I knew we could do it!" (generic validation)
- "We make a great team!" (too generic)
- Any line that could be said by ANY character in ANY story — if it doesn't specifically reference what just happened in this scene, it is wrong and must be rewritten

DIALOGUE MUST SOUND LIKE THESE SPECIFIC PEOPLE REACTING TO THIS SPECIFIC THING:
- Every line of dialogue must reference the EXACT event that just happened — not a generic reaction to an abstract situation
- The character's personality must show in HOW they react, not just what they say
- In comedy: the funniest reaction is usually the most specific and most unexpected one
- WRONG: "That was close!" — could be said by any character in any story
- RIGHT: "I was THIS close to breaking the record and you — you pulled the SPRINKLER lever." — specific, character-driven, funny
- WRONG: "We need to work as a team." — abstract advice, no comic energy
- RIGHT: "Okay. New plan. We do not touch anything I haven't personally approved. Starting now. Don't look at that lever." — character voice, personality-driven, sets up the next disaster
"""
    elif is_adult and genre == "friendship":
        safety_rules = """
CONTENT RULES (FRIENDSHIP — ADULT, 18+):
- This is a story about adult friendship — not childhood buddies learning to share
- Adult friendship themes: navigating life changes together, supporting each other through real struggles,
  growing apart and reconnecting, honesty that hurts, loyalty tested by circumstance
- Real adult challenges are fair game: career setbacks, relationship problems, grief, identity, major life decisions
- Emotional depth and nuance are required — adult friendships are complicated, not simple
- Conflict can come from honest disagreements, life pulling people in different directions, or buried resentments
- Resolution must be earned — not just a hug and everything is fine
- Tone: warm but honest — like a Issa Rae or Mindy Kaling drama-comedy, not a kids' show
- No graphic violence or sexual content — keep it PG-13 emotional drama
"""
    elif is_adult and genre == "mystery":
        safety_rules = """
CONTENT RULES (MYSTERY — ADULT, 18+):
- This is an adult mystery — real crimes, real stakes, real consequences
- Adult motivations drive the mystery: greed, obsession, jealousy, betrayal, revenge, power
- Death and serious harm are valid story elements — show consequence with weight, not sanitized off-screen
- The mystery must feel genuinely dangerous — the protagonist is at real risk if they get too close to the truth
- Moral ambiguity is encouraged — characters can be compromised, complicit, or have understandable dark motives
- Tone: adult crime drama or thriller — think Gone Girl, Knives Out, or Sharp Objects
- Clues must be specific and meaningful — not vague "suspicious behavior" but real evidence and real deception
- No gratuitous gore but realistic depiction of crime scenes and consequences is fine
"""
    elif is_adult and genre == "adventure":
        safety_rules = """
CONTENT RULES (ADVENTURE — ADULT, 18+):
- This is adult adventure with real stakes — characters face genuine danger, not sanitized peril
- Characters can be seriously threatened, pursued, or physically challenged with believable consequence
- Adult motivations and relationships drive the story — complex loyalties, personal sacrifice, hard choices
- Moral ambiguity is welcome — not every ally is trustworthy, not every villain is purely evil
- Conflict can involve real injury, loss, or failure — stakes must feel genuine, not consequence-free
- Tone: adult action-adventure — think Indiana Jones, The Mummy, or The Lost City
- Strong language is acceptable in tense or action moments
- No gratuitous gore but violence and its realistic aftermath are fine
"""
    elif is_adult and genre == "fantasy":
        safety_rules = """
CONTENT RULES (FANTASY — ADULT, 18+):
- This is adult fantasy with depth — real political intrigue, moral ambiguity, genuine danger
- Violence and its consequences can be depicted — battles, injuries, loss are part of the genre
- Adult relationships, power struggles, and ethical dilemmas enrich the world
- Magic and supernatural elements should have real costs or limits — nothing is free or convenient
- Characters can make morally difficult choices with lasting consequences
- Tone: adult fantasy drama — think The Witcher, Game of Thrones (toned down), or Shadow and Bone
- World-building must feel specific and lived-in — not generic "magical kingdom" filler
- No explicit sexual content but mature romantic tension and adult relationships are fine
"""
    elif is_adult and genre == "sci_fi":
        safety_rules = """
CONTENT RULES (SCI-FI — ADULT, 18+):
- This is adult science fiction with intellectual depth and real stakes
- Complex ethical dilemmas involving technology, society, or power are encouraged
- Characters face real danger and real consequences — not sanitized or risk-free
- Adult themes: corporate corruption, surveillance, identity, loss of autonomy, sacrifice for the greater good
- Violence and its aftermath are acceptable when they serve the story
- Tone: thoughtful adult sci-fi — think Black Mirror, Arrival, or The Martian
- The science or technology must have internal logic — not magic dressed as tech
- Strong language is acceptable in tense or dramatic moments
"""
    elif is_adult and genre == "sports":
        safety_rules = """
CONTENT RULES (SPORTS — ADULT, 18+):
- This is adult sports drama — not a kids' movie about teamwork and believing in yourself
- Real adult pressures drive the story: career stakes, professional rivalries, contracts, identity tied to performance
- Characters can face real adult failures — injury ending a career, aging out of peak performance, being cut
- Interpersonal conflict can be genuinely difficult — jealousy, betrayal, sacrificing a relationship for ambition
- The emotional stakes must be adult-level — what this sport means to an adult's livelihood and identity
- Tone: adult sports drama — think Friday Night Lights, Moneyball, or Creed
- Can reference adult life around the sport: training obsession, personal sacrifices, real physical toll
- No graphic violence but sports injuries and their consequences are realistic elements
"""
    elif is_adult:
        safety_rules = """
CONTENT RULES (ADULT):
- This story is for an adult audience — not a children's comic
- Real stakes and genuine consequences are required — not sanitized or risk-free
- Adult-life situations, relationships, and motivations should drive the story
- Conflict and tension can be mature and nuanced — not resolved too easily
- Tone should fit the adult main character's life and experience
- No graphic violence, gore, or sexual content — keep it PG-13 unless the genre specifies otherwise
"""
    else:
        safety_rules = """
CHILD SAFETY RULES:
- All scenes must be clearly safe, gentle, and age-appropriate
- No violence, fighting, attacks, weapons, threats, kidnapping, or physical danger
- No horror, fear, intimidation, or sinister imagery
- Villains must be mischievous or silly, never scary or threatening
- Any conflict must be playful, light, and non-violent
"""

    npc_bible_block = ""
    if npc_bible:
        lines = []
        for npc in npc_bible:
            lines.append(f"  {npc['name'].upper()}: {npc['description']}")
        npc_bible_block = (
            "\n=== NPC CHARACTER BIBLE (LOCKED — copy these descriptions verbatim into every panel) ===\n"
            "The following named NPCs appear in the story. Their visual appearance is PRE-DEFINED and CANNOT change.\n"
            "Every time one of these characters appears in an image_prompt, copy their description EXACTLY as written below.\n"
            "Do NOT redesign them. Do NOT change their clothing, hair, skin tone, or any physical detail between panels.\n"
            "If you describe them differently across panels, the image model will draw a different person — that is always wrong.\n"
            "EVERY NPC LISTED BELOW MUST APPEAR IN AT LEAST ONE PANEL of your adaptation — they were included in this bible "
            "specifically because they matter to the story. If you find yourself compressing scenes and about to drop one of "
            "them entirely, keep a panel for them instead of cutting it — a reference portrait was already generated for each "
            "of these names, so omitting them wastes it.\n\n"
            + "\n\n".join(lines) + "\n"
        )

    story_section = f"""Story:
- Theme: {theme}
- Logline: {logline}
- Outline: {outline}"""
    if prose_story and not is_preview:
        story_section += f"""

FULL PROSE STORY (canonical source — adapt this faithfully into {page_count} panels):
This complete story is the single source of truth. All character names, plot points, events, key dialogue,
and emotional beats must come from here. Do NOT invent plot elements not in this prose.
Distribute the prose across all {page_count} panels chronologically, giving each panel its own specific moment.

{prose_story}"""

    page_number_range_desc = f"{chunk_range[0]} to {chunk_range[1]}" if chunk_range else f"1 to {page_count}"

    if chunk_range:
        chunk_start, chunk_end = chunk_range
        prior_note = f"A separate earlier request already generated panels 1 through {chunk_start - 1}." if chunk_start > 1 else ""
        later_note = f"A separate later request will generate any panels after {chunk_end}." if chunk_end < page_count else ""
        chunk_continuation_block = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS REQUEST'S PANEL RANGE — READ BEFORE WRITING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Everything above describes the story's FULL {page_count}-panel arc so you understand where this excerpt falls within it — but this single request must produce ONLY panels {chunk_start} through {chunk_end} ({chunk_end - chunk_start + 1} panels), not the whole story. {prior_note} {later_note}
Every page_number in your JSON output must be an integer from {chunk_start} to {chunk_end} inclusive — do NOT restart numbering from 1, and do NOT include any panel outside this range.
{prior_context or ""}
Continue the story in STRICT forward chronology from exactly where the prior panels left off — do not repeat, restate, or re-show any moment, setting, or dialogue beat already covered by an earlier request. If this range includes the climax or resolution per the story arc above, deliver it in full within this range — do not defer it to a later request.
"""
    else:
        chunk_continuation_block = ""

    if locked_outfit:
        outfit_schema_desc = json.dumps(f"the locked outfit already established for the main character: '{locked_outfit}' — copy this exact phrase, do not invent a new one")
    else:
        outfit_schema_desc = '"the single canonical outfit for the main character used in every panel — copy the exact phrase from the first page\'s image_prompt (e.g. \'black jersey with red left sleeve and yellow right sleeve, blue shorts\')"'

    prompt = f"""
You are a creative writer for {writer_desc}

Create exactly {page_count} pages for a comic based on the selected story below.

{purpose_block}

STORY DEPTH (CRITICAL — applies to every genre, every page):
These rules separate a compelling story from a forgettable one. Enforce them on every page you write:

- SPECIFICITY OVER VAGUENESS: Every scene must have concrete, specific details — named objects, specific locations,
  exact discoveries, real consequences. NEVER write placeholder descriptions like "something unsettling," "a strange
  event," or "they learn more about the history." Name the thing. Describe the thing. Make it real.

- STAKES THAT MATTER: The reader must always know what {hero_name} stands to lose. Stakes should feel personal —
  tied to her relationships, her goals, or her safety — not abstract.

- CAUSE AND EFFECT: Every page must logically follow from the last. Characters make decisions based on what just
  happened. No convenient coincidences. No scenes that could be cut without affecting the story.

- CHARACTER REACTIONS MUST BE EARNED: Characters should react with genuine emotion proportional to what just happened.
  A major revelation must land with weight — not be brushed off in one line of dialogue.

- NO FILLER PAGES: Every single page must either advance the plot, reveal character, or raise stakes. A page where
  characters simply walk to the next location and say nothing meaningful is wasted.

- SHOW THE JOURNEY, NOT JUST LANDMARKS: Don't just jump from "they decided to investigate" to "they found the truth."
  Show the actual investigation — what they tried, what failed, what clue unlocked the next step.

STORY CRAFT (CRITICAL — applies to every genre, every page):
These rules determine whether the story feels personal and real or generic and forgettable.

NAMED SIDE CHARACTERS (required for all genres, non-negotiable):
- Every side character or NPC who appears in more than one scene MUST have an actual first name — never a label like "the rival," "the competitive couple," "the mentor," or "the suspicious stranger"
- Each named side character must have ONE DEFINING PERSONALITY TRAIT that affects their behavior throughout the story and creates friction or contrast with {hero_name}
- Their trait must produce a recurring effect across multiple scenes — not just mentioned on introduction and forgotten
- WRONG: "A rival shows up and challenges them." — no name, no personality, no story value
- RIGHT: "Devon, who treats every competition like a personal audition and never acknowledges {hero_name} unless she's losing, shows up at the venue." — named, specific personality, creates ongoing conflict

USE THE CUSTOMER'S PERSONALITY TO DRIVE THE STORY:
- The personality traits listed below are the engine of this story — they shape WHICH situations arise and HOW {hero_name} reacts
- WRONG: Personality says "competitive" → story contains no competitive element
- RIGHT: Personality says "competitive" → {hero_name}'s competitiveness is what starts the problem, makes it worse in the middle, and is the key to solving it
- Every major scene should connect to who this specific person is — their reactions must reflect their traits, not generic "hero" behavior

BANNED GENERIC DIALOGUE (applies to every genre, no exceptions):
- "We did it!" — too generic, always rewrite it
- "I knew you could do it!" — too generic
- "Together we can do anything!" — greeting card, not real dialogue
- "You were right all along." — generic resolution line
- "I believe in you!" — too generic
- Any line that could be said by any character in any story — if it does not reference what is SPECIFICALLY happening on this page, it is wrong and must be rewritten
- Every dialogue line must be grounded in the exact situation occurring on that specific page

Main character:
- Name: {hero_name}
{personalization_block}

Character visual brief:
- {character_brief}

{story_section}

Instructions:
- Each page must include:
  - page_number ({page_number_range_desc})
  - narration: 1–2 sentences, maximum 200 characters total — REQUIRED on EVERY panel, including the final cliffhanger
    NARRATION QUALITY RULE (CRITICAL — applies to every panel, no exceptions):
    Narration MUST be present on every single panel — never leave narration blank or empty.
    Narration must add what the IMAGE CANNOT show on its own. The reader can already see what is happening — do NOT describe it.
    Use narration for: {hero_name}\'s specific inner logic or wrong-headed reasoning, an ironic observation about what just happened, a piece of context that makes the image funnier, or a character personality detail that explains why the situation is about to get worse.
    {'Draw this detail from the prose story provided.' if prose_story else f'Draw this detail from what you know about {hero_name}\'s personality, the logline, and the specific situation in this panel.'}
    WRONG: "Ceiara and Vincent arrive at the pottery studio, excitement buzzing in the air." — describes what the image already shows.
    RIGHT: "Vincent had watched exactly one pottery tutorial, which he felt was more than enough." — adds meaning the image cannot carry.
    CLIFFHANGER PANELS: Even the last panel must have narration — use it to amplify the cliffhanger tension with an internal thought or ironic observation.
  - dialogue: REQUIRED array with 1 to 2 entries — every page MUST have at least one line of dialogue, never leave it empty. Each entry has:
      - speaker (character name)
      - text ({'8–15' if genre == 'comedy' else '5–10'} words the character says out loud, {'specific to this exact situation — no generic lines' if genre == 'comedy' else 'fun and age-appropriate'})

DIALOGUE ADDRESS RULE (CRITICAL — check every line before writing it):
  FREQUENCY: Most dialogue lines should have NO address term — characters in real conversation do not say
  each other's name or nickname constantly. BUT nicknames must appear naturally across the story.
  TARGET: {'at least 1 nickname use across these ' + str(page_count) + ' panels — if a nickname is defined it MUST appear at least once' if is_preview and page_count <= 4 else '3–5 nickname uses total, spread across the story — not clustered, not zero'}.
  Use them at moments that feel genuinely natural: a tense beat, a tender moment, a laugh, calling for
  someone's attention, or a punchline landing with the name attached.
  - WRONG (overuse): "Did you hear that, Babe?" / "Look at this, Babe!" / "What do you think, Babe?" — nickname on every single line
  - WRONG (never used): 0 uses across these panels — if a nickname is defined, it MUST appear
  - RIGHT: Most lines have no address term. One or two feel natural: "Stay close, Babe." in a tense moment or "Babe, you're not going to believe this." when surprised

  When an address term IS used, use the correct name or nickname:
  - A HUSBAND addresses his WIFE by her first name or nickname — NEVER "Mom", "Sis", "Kid"
  - A WIFE addresses her HUSBAND by his first name or nickname — NEVER "Son", "Boy"
  - WRONG: Husband says "Mom, it's glowing!" — this is always wrong
  - RIGHT: Husband says "{hero_name}, it's glowing!" — OR uses their nickname if NICKNAME RULES specify one
  - If NICKNAME RULES define a nickname, use the nickname (not the full name) on those rare occasions an address term is needed.

  CRITICAL — NICKNAME SCOPE: A nickname is ONLY valid between the two characters it belongs to.
  If a character is speaking TO a ghost, spirit, villain, librarian, stranger, or ANY third party,
  the nickname must NEVER appear in that line — not at the end, not anywhere.
  WRONG: Ceiara says to the ghost "We understand your pain, Babe." — 'Babe' cannot appear here
  RIGHT: Ceiara says to the ghost "We understand your pain." — no address term when speaking to a third party

  Before finalising any dialogue line, ask: (1) does this line need an address term? Usually no. (2) WHO is being spoken to? The nickname only applies when speaking directly to the specific person it belongs to.
  - image_prompt (detailed visual scene)

IMAGE PROMPT SYSTEM:

Each page must include an "image_prompt" that follows this structure.

=== CHARACTER LOCK (STRICT) ===
The main character must match this visual brief exactly:
{character_brief}

CRITICAL CONSISTENCY RULES:
- The character must be treated as the SAME exact person in every image
- Face structure must remain identical (eyes, nose, jawline, eyebrows)
- Hair must remain identical (color, length, style)
- Skin tone must remain identical
- Do NOT reinterpret, redesign, or restyle the character
- Include exact eye color if visible
- Be specific about facial features that define identity

OUTFIT LOCK (CRITICAL):
- Choose ONE outfit that fits this specific story and establish it on page 1 — then use that EXACT SAME outfit in every single image_prompt
- Pick the outfit type based on what makes sense for the story:
  * Gymnastics, dance, cheer → leotard
  * Soccer, basketball, sports → jersey and shorts
  * Swimming → swimsuit
  * School, everyday adventure, mystery → their regular described clothes from the character brief
  * Superhero story → a simple heroic outfit
- CRITICAL — read the character brief RIGHT NOW and find the shirt/top description — use those EXACT colors and pattern for the story outfit, do NOT invent a different color
- Example: if the character brief says "green polka dot shirt" → the leotard MUST be "green polka dot leotard" — NOT pink, NOT sparkly pink, NOT any other color
- Do NOT default to pink for gymnastics/dance — pink is NOT the correct color unless the character's own shirt is pink
- Describe the outfit with specific colors, pattern, and sleeve style — e.g. "sleeveless green polka dot leotard with black straps" — choose ONE sleeve style and lock it
- Do NOT change the sleeve style, color, or pattern mid-story
- Once the outfit is chosen, write the EXACT SAME outfit description word-for-word in every image_prompt where the character appears
- Do NOT change the outfit mid-story under any circumstances — one outfit for the entire comic

SPORT/ACTIVITY LOCK (CRITICAL):
- If the story involves a specific sport or activity (gymnastics, soccer, dance, swimming, etc.) every action scene must show THAT EXACT SPORT — never a different one
- Gymnastics story: show beams, floor mats, vaults, uneven bars — NEVER soccer balls, basketball hoops, or any other sport's equipment
- Soccer story: show a soccer field, soccer ball, goalposts — NEVER gymnastics equipment
- Name the specific sport equipment in every image_prompt that involves athletic action — do NOT use vague words like "runs" or "jumps" that the image model could misinterpret as a different sport
- Wrong: "Ariyah runs nervously toward her event" → model draws soccer
- Right: "Ariyah stands nervously at the edge of the gymnastics floor mat, chalk on her hands"

NON-SPORT SETTING RULE (CRITICAL — applies to ALL stories):
- A stadium, sports field, gym, pool, or any sports venue ONLY appears if this story's theme is specifically that sport AND the characters are actively playing or competing in that moment
- If this story is NOT about a sport (e.g. it is about adventure, mystery, science, fire rescue, friendship, school, etc.) — NEVER show a stadium, sports field, bleachers, crowd, or any sports venue in ANY panel — not even in the background
- WRONG: a child running through a burning building scene that shows a soccer stadium with crowd — this is a completely wrong setting
- RIGHT: a child running through a burning building shows smoke-filled hallways, debris, fire exits, stairwells
- The image model tends to default to stadiums and sports fields for ANY running or action scene — you MUST explicitly describe the correct non-sport setting instead
- For non-sport stories: running scenes happen in streets, hallways, parks, forests, schools — NEVER stadiums
- If the story involves a sport venue AND a non-sport scene: the non-sport scene (planning, talking, discovering) still takes place at a home, school, or park — NOT at the sport venue

If any of the above changes, the result is incorrect.

=== GENRE ===
Genre: {genre} — {genre_style['story_tone']}
Visual style for this genre: {genre_style['image_style']}
Every panel must reflect this genre's mood in lighting, color, poses, and environment.

{horror_atmosphere_block}

=== STYLE LOCK ===
All images must use:

{style_lock_block}

=== SCALE CONSISTENCY ===
- Animals and objects must be proportionally accurate to the child
- Do NOT oversize pets or distort scale
- Maintain realistic spatial relationships in every scene
- Do NOT add animals, sidekicks, or extra characters unless explicitly required by the story
- CHILD HEIGHT RULE: when two or more child characters appear together, they must be approximately the same height — do NOT draw the main character significantly taller or larger than their child friends unless the story establishes an age/size difference — children of the same age are the same size

=== IMAGE PROMPT FORMAT (MANDATORY — follow this exact structure for every image_prompt) ===

Every image_prompt MUST be written in this exact order:

1. FRAMING WORD (first word must be one of these two options — nothing else before it):
   "WIDE SHOT:" — character visible head-to-toe, environment fills at least 60% of the frame (USE THIS FOR ALL PANELS WITH 2+ CHARACTERS)
   "MEDIUM SHOT:" — character visible from waist up, environment clearly visible on all sides (ONLY for single-character solo panels where a closer view is essential)

2. SETTING (immediately after the framing word):
   Describe the full environment BEFORE describing the character.
   Be specific: name the location, describe walls/sky/ground, furniture/trees/objects, lighting.
   Example: "EXTERIOR — a sunny neighborhood park with green grass, a red slide, and bright blue sky with white clouds."

3. CHARACTER ACTION (after the setting):
   Describe what the character's BODY is doing — not their face.
   A bare verb is NOT enough on its own — "running," "pointing," "bracing" are starting points, not finished descriptions. Every action verb must be followed by the SPECIFIC body mechanics that make it visible: which way limbs are extended, what the torso/hair/clothing is doing, whether the stance is stable or off-balance. "Ceiara runs on the left" is not a real description of running — "Ceiara sprints with her arms pumping and hair whipping backward, body angled forward mid-stride" is.
   The character's pose and movement must reflect the story moment — see the mandatory ACTION SCENES rule below for anything physical, chaotic, or fast-moving.

4. KEY PROPS (if any):
   If the scene involves a key object that drives the story (envelope, map, trophy, letter, gadget, book, etc.) — describe it explicitly and place it in the foreground or mid-ground where it is clearly visible. Do NOT let important props disappear from the image.

5. MOOD AND LIGHTING (CRITICAL — lighting language directly controls whether the image looks like a comic or a photograph):
{lighting_block}

6. STYLE ANCHOR (final sentence of every image_prompt — MANDATORY, word for word):
   "{style_anchor_line}"

WRONG FORMAT (never write this):
"Vincent's face fills the frame, eyes wide with surprise as he gasps at what he sees..."

RIGHT FORMAT (always write this):
"WIDE SHOT: EXTERIOR — outside a colorful go-kart racing track with red and white barriers, a crowd of kids cheering in the background. Vincent stands at the starting line visible head-to-toe, both arms raised in victory, wearing his blue hoodie and jeans. Bright well-lit comic illustration lighting, vivid saturated colors, triumphant mood. Illustrated comic art — vivid saturated colors, clear crisp linework, characters match their reference photos exactly."

COMPOSITION RULES:
- Every image must show the character actively doing something — running, jumping, crouching, reaching, reacting, pointing, climbing, discovering
- Each page must use a DIFFERENT framing: alternate between WIDE SHOT and MEDIUM SHOT across pages
- The character should occupy roughly 30-50% of the frame height — NOT filling the entire image
- Significant background/environment must be visible in every panel
- Think like a comic book artist: each panel captures a decisive moment in the story, not a posed photo

⚠️ ACTION SCENES MUST SHOW MOTION, NOT A POSE — MANDATORY, NO EXCEPTIONS (check this every time the narration describes chaos, a chase, a struggle, or anything physical happening): Real testing found panels defaulting to characters standing upright, waving an arm, or pointing, even when the narration says a crowd is "thrown into disarray," characters "burst out" of somewhere, or security is "flooding in." This is the single most common way an otherwise well-structured story still fails visually, and it recurred even after this rule already existed in an earlier, weaker form — it needs to actually change what gets written, not just be read and nodded along to.
- CONFIRMED REAL FAILURES from actual delivered books — do not repeat these patterns: "Ceiara runs on the left, looking back anxiously, while Vincent runs on the right, urging her forward." / "Ceiara stands on the left, eyes wide with shock, while Vincent stands on the right, bracing for confrontation" (narration: "the door burst open, security flooding in"). Both of these are bare labels, not descriptions — "runs on the left" and "bracing" tell you nothing about what the body is actually doing. If your own draft reads like either of these, it has already failed this rule — rewrite it before moving to the next panel.
- SELF-CHECK, before finalizing every panel: (1) Does this panel's narration describe something physically happening — running, fleeing, colliding, grabbing, falling, bursting, a crowd reacting — rather than a character just feeling or deciding something? (2) If yes, does the image_prompt's CHARACTER ACTION portion name specific body mechanics (which way limbs/hair/clothing are moving, the angle of the body, contact with an object) rather than a bare verb plus an emotion word? (3) If the action portion would still make sense with the verb swapped for "stands" and nothing else changed, it is NOT specific enough — go back and add the missing mechanics.
- The "blocking" field (who is on which side) only fixes RELATIVE POSITION for consistency — it is not a template for the action language. Reusing a locked blocking string like "Ceiara running on the left, Vincent running on the right" verbatim as the entire action description is exactly the failure this rule exists to prevent. Keep the left/right positions from blocking, but always build the actual motion description around them fresh — do not let the blocking string become the whole sentence.
- Use exaggerated, specific body mechanics: sprinting with limbs extended and hair/clothing flying backward, lunging sideways, ducking beneath something, stumbling, colliding into an object, grabbing at something out of reach — not "stands confidently," "braces," or "gestures towards."
- Show the character interacting with the physical environment, not just standing in front of it: knocking into a table, sending objects flying, a crowd scattering or recoiling, something overturned or shattered, food/papers/debris caught in motion, a door actually bursting open with visible motion rather than security merely present in the frame.
- Use dynamic camera framing for these moments: a tilted or low-angle shot, strong foreshortening, a foreground/background depth split (one figure closer and larger, the action continuing behind them) — not a flat, centered, eye-level two-shot.
- WRONG (static pose masquerading as action): "Vincent confronts the scammer angrily, fists raised, standing his ground."
- RIGHT (motion actually shown): "Vincent lunges sideways as the scammer reaches for him, knocking into a festival table — plates and napkins fly through the air as Ceiara ducks beneath the falling display behind him. Dynamic low-angle action shot, strong foreshortening, bodies captured mid-motion, shocked crowd recoiling in the background."
- This applies with equal force to the climax and any chase, escape, or confrontation beat — these are exactly the panels most likely to default to a static standoff pose, and exactly the panels where showing real motion matters most.

SPEECH BUBBLE POSITIONING RULE:
- When two characters appear in the same panel, ALWAYS place the MAIN CHARACTER on the LEFT side of the frame and the supporting character on the RIGHT
- For each dialogue entry, set "side" to "left" or "right" based on where that speaker physically stands in the image
- The speech bubble renderer uses the "side" field to place each bubble — "left" speakers get a bubble on the left, "right" speakers get a bubble on the right
- In scenes with 3+ characters: the main character is LEFT, the opposing NPC or antagonist is RIGHT (even if they are partially off-frame or pointing from the side)
- WRONG: Lydia is pointing from the right side of the image but her dialogue entry has side:"left"
- RIGHT: Lydia is on the right → side:"right"; Ceiara is on the left → side:"left"

=== CHARACTER PRESENCE (CRITICAL — check EVERY panel) ===
Every page's JSON must include "characters_present": the list of every named character (main character, named support characters, named NPCs) who is PHYSICALLY VISIBLE, standing in the scene, in that panel's image_prompt — and ONLY those characters.
- A character is NOT present if: they left the scene, are elsewhere, are contacted remotely (text message, phone call, radio), are being searched for because they're missing, are captured/restrained off-panel, are hiding out of sight, or are only mentioned/remembered/referenced in dialogue or narration.
- This matters most for horror/mystery "someone vanished" beats: if the narration says a character is gone, missing, vanished, or separated from the group, that character must be OMITTED from characters_present AND from the image_prompt for every panel until the story explicitly reunites them in the same shot. Do not draw a "missing" character standing calmly in the scene — that contradicts the story.
- characters_present and the image_prompt must always agree: never list someone in characters_present who isn't actually described in the image_prompt, and never describe someone in the image_prompt who isn't listed in characters_present.
- The main character is present in almost every panel by default — only omit them if the scene is explicitly from another character's POV without them.

GAZE RULES:
- Characters must NEVER look directly at the camera or stare straight at the viewer
- Characters' eyes should always be directed at something in the scene — each other, an object, the environment, or the direction of action
- In every multi-character panel, characters must be visually engaging with each other — facing each other, reacting to each other, or both reacting to the same thing in the scene

EMOTIONAL EXPRESSION RULES (CRITICAL — check EVERY panel before writing it):
- Every character's facial expression and body language MUST match the emotional tone of that specific scene
- FEAR / HORROR / THREAT: wide eyes, tense body, mouth slightly open, defensive posture — NEVER a smile
- ANGER / CONFRONTATION: jaw set, eyes narrowed, tense stance, pointing or leaning forward — NEVER a smile
- GRIEF / SHOCK / REVELATION: eyes wide or downcast, hand raised to mouth, stiff posture — NEVER a smile
- DETERMINATION / DEFIANCE: eyes forward and intense, chin up, fists clenched or arms crossed — NEVER a passive neutral expression
- CELEBRATION / JOY: broad smile, relaxed body, open arms, light posture — appropriate ONLY in genuinely happy scenes
- ANXIOUS / NERVOUS / UNCERTAIN: eyebrows raised, eyes alert and alive, slight frown or pursed lips — character looks ALIVE and engaged, NOT ill, NOT gaunt, NOT zombie-like
- WRONG: Ceiara is being threatened by a villain → image_prompt describes her smiling warmly
- WRONG: Ceiara feels anxious → image_prompt results in her looking gaunt, dark-faced, or like she is ill
- RIGHT: Ceiara is being threatened → image_prompt says "Ceiara stands with fists clenched, eyes narrowed, jaw set, expression fierce and defiant"
- RIGHT: Ceiara feels nervous → image_prompt says "Ceiara stands with arms crossed, eyebrows raised, eyes wide and alert, expression wary but alive"
- State the emotional expression EXPLICITLY in the CHARACTER ACTION section — do not leave it implied

FRAMING RULES (CRITICAL — these are hard constraints):
- NEVER write a close-up, tight shot, or face-filling shot of any kind
- NEVER describe only the character's face or head — always describe the full body in context
- The character's head must occupy NO MORE than 20% of the total image height
- If a scene naturally calls for an emotional reaction, show the character's FULL BODY reacting — not a face close-up
- If TWO characters are in the same scene, you MUST write a WIDE SHOT that shows BOTH characters full-body — never write a two-shot that would result in both faces filling the frame
- Default to WIDE SHOT for ALL panels — MEDIUM SHOT is only allowed for single-character solo panels
- If TWO OR MORE characters appear in any scene, you MUST use WIDE SHOT — never MEDIUM SHOT for multi-character scenes

SETTING RULES (CRITICAL):
- You MUST explicitly state whether each scene is INDOORS or OUTDOORS at the start of the image_prompt
- For indoor scenes: begin with "INTERIOR — inside [specific room/location]:" and describe walls, ceiling, furniture, and BRIGHT overhead lighting or large windows with daylight streaming in
- For outdoor scenes: begin with "EXTERIOR — outside [specific location]:" and describe bright sky, ground, and outdoor elements
- ALL scenes must specify BRIGHT, WELL-LIT environments — never dim, dark, or shadowy (exception: horror/mystery genre only)
- NEVER write a vague setting — be specific (e.g. "INTERIOR — inside a brightly-lit school gymnasium with wooden floors, colorful banners on the walls, bright overhead gym lights")
- The AI image generator will ignore vague settings and default to outdoors — you must be explicit

LOCATION CONSISTENCY RULE (CRITICAL):
- When a scene spans multiple consecutive pages at the same location, EVERY panel in that scene must use the SAME location description — word for word
- WRONG: Page 5 is "inside the local library" but page 6 (still the same library visit) shows a kitchen or different room
- RIGHT: Both pages 5 and 6 explicitly say "INTERIOR — inside the local library with tall wooden bookshelves, a green reading lamp on the table, and dusty volumes stacked high"
- Before writing each panel's image_prompt, ask: is this panel part of an ongoing scene at a known location? If yes, copy that location description exactly
- When the narration transitions to a NEW location, introduce it fresh with new setting details
- The image generator has no memory between panels — you MUST repeat the full location description every time

SETTING LABEL (CRITICAL — this is a separate field from image_prompt, see JSON shape below):
- Every panel must also include a short "setting" label (3-8 words) naming the physical location, e.g. "the pottery studio workspace" or "outside the go-kart track"
- Use the EXACT SAME label, word-for-word, for every panel in that same physical location — including if the story revisits that location later, even many panels apart, not just consecutive panels
- A different room, a different part of the same building, or moving outside all count as a different location and need a different label
- This label is used downstream to enforce visual consistency in code — it must be precise and reused consistently, not paraphrased panel to panel

NO MID-SCENE TELEPORTING (CRITICAL — this is a common and very visible failure):
- If a panel establishes characters INSIDE an enclosed space (a car, a room, a building), every following panel that continues the SAME conversation/moment must also show them INSIDE that same space
- Do NOT cut to an exterior/roadside/outdoor shot of the same characters mid-conversation unless the narration explicitly says they stopped, pulled over, got out, or otherwise left — a character mid-sentence in a car cannot suddenly be standing outside it in the next panel with no narrated reason
- This applies especially to vehicle scenes: once characters are established as driving/riding, they stay seated inside the vehicle, in motion, until the narration explicitly says the vehicle stopped and they got out

BLOCKING LABEL (CRITICAL — this is a separate field from image_prompt, see JSON shape below):
- Every panel must also include a short "blocking" string stating each present character's specific physical position relative to the others and the setting — e.g. "Vincent driving, hands on wheel; Ceiara in the front passenger seat; Max in the back seat" or "Vincent standing on the left; Ceiara standing on the right"
- For a vehicle scene specifically: state who is driving, who is in the front passenger seat, and who is in the back seat — and every panel that shares this scene's setting label must assign characters to those SAME seats. A character cannot be in the front seat in one panel and the back seat in another within the same continuous scene unless the narration explicitly shows them switching seats
- Use the EXACT SAME blocking string, word-for-word, for every panel that shares the same setting label AND the same characters present — copy it forward exactly like the setting description, don't re-invent it panel to panel
- If a character joins, leaves, or the group's arrangement genuinely changes (someone gets in/out, someone moves seats), write a NEW blocking string for that panel onward and continue reusing that new one
- This label is used downstream to enforce visual consistency in code — it must be precise and reused consistently, not paraphrased panel to panel

DIALOGUE SPEAKER MANDATE (ABSOLUTE RULE — check before finalizing every panel):
Before writing any panel's image_prompt, look at the dialogue array for that panel.
Every character listed as a speaker MUST physically appear in that panel's image_prompt.
- If the speaker is the main character → include them (always required)
- If the speaker is a supporting character → include them (always required)
- If the speaker is a NAMED NPC → copy their EXACT locked description from the NPC Character Bible into this panel's image_prompt
- A speech bubble without a visible speaker is ALWAYS wrong — the image model will assign the bubble to the wrong character
- WRONG: Darnell speaks in the dialogue but only Ceiara and Vincent appear in the image_prompt → Darnell's bubble will incorrectly point to Vincent
- RIGHT: Darnell speaks → Darnell's full locked description from the NPC Bible appears in the image_prompt, positioned on the correct side of the frame

NPC / ANTAGONIST PRESENCE RULE (CRITICAL):
- When the narration or dialogue says a third-party character is present (villain, antagonist, rival, instructor, family member, police, librarian, bystander, etc.), that character MUST physically appear in the image_prompt AS A PROMINENT FIGURE — not a tiny background silhouette
- Do NOT write a confrontation scene where only the hero characters appear — if someone is threatening them, that person must be in the image
- Describe every non-reference character explicitly in the image_prompt with: position in frame, approximate age, gender, build, skin tone, hair style, clothing, and expression
- FEMALE NPC RULE (CRITICAL): The image model defaults to rendering ambiguous figures as male. For any female NPC, you MUST use the word "woman" prominently AND include at least one clearly feminine visual marker — a dress, skirt, feminine hairstyle (long hair, styled updo, bob), or feminine jewelry. Do NOT rely on the word "female" alone buried in a description.
  - WRONG: "Lydia points at them, angry expression" — no gender markers, will render as a man
  - RIGHT: "Lydia, a sharp-dressed woman in her 40s with a sleek black bob and a fitted red evening gown, points accusingly at Ceiara and Vincent"
- NPCs who are interacting with the main characters must appear IN THE FOREGROUND or MID-GROUND — not shrunken into the background
- WRONG: Narration says "rival couple Jenna and Ian smirk at them" but image_prompt only describes Ceiara and Vincent, with "a couple visible in the background"
- RIGHT: image_prompt describes "in the mid-ground, a confident couple — a tall white man with brown hair in a blue shirt with arms crossed, and a Black woman in a yellow top with natural hair and a smug expression — stands facing Ceiara and Vincent"
- Use WIDE SHOT for all scenes with multiple named characters so everyone fits in the frame

OUTFIT RULES (CRITICAL — outfit changes are one of the most common errors, check every panel):
- The character wears the EXACT SAME outfit in every single image — do not change or upgrade clothing between pages
- The outfit is whatever was established on page 1 of this story — it may be their everyday clothes OR a story-appropriate outfit (leotard for gymnastics, jersey for sports, etc.)
- Copy the EXACT outfit description from page 1's image_prompt word-for-word into EVERY panel where the character appears — do not paraphrase, do not simplify
- Do NOT switch between the everyday clothes from the character brief and the story outfit — the story outfit overrides for this book
- WRONG: Page 1 establishes black tank top → page 5 image_prompt says "white shirt"
- RIGHT: Every page image_prompt copies "black tank top" exactly as written on page 1
- Supporting characters also keep the same outfit throughout — they wear the SAME outfit in every single panel
- PROSE STORY OVERRIDE (CRITICAL): Outfit locks override any clothing descriptions in the prose story. If the prose says a character "changed into a tuxedo" or "put on an apron" — IGNORE that clothing change. The image_prompt must still show the locked outfit. The prose drives PLOT EVENTS; visual appearance is controlled by these outfit rules, not the prose narrative.

{safety_rules}

SUPPORTING CHARACTER RULES:
- Supporting characters (friends, sidekicks, helpers) are allowed and encouraged
- Supporting characters must look CLEARLY DIFFERENT from the main character — different hair, different clothing, different build
- Do NOT duplicate the main character's appearance for any background or supporting character
- The main character must always be visually distinct and the clear focal point of the scene
- Supporting characters must NOT have dreadlocks or locs — give them clearly different hair (short curly, short afro, straight hair, etc.)
- Supporting characters must NOT wear glasses of any kind
- Each supporting character must have a unique, distinct look that cannot be confused with the main character
{additional_chars_block}
=== CHARACTER INTRODUCTION RULE (CRITICAL) ===
When a supporting character appears in a panel for the FIRST TIME, their name MUST be stated in that panel's narration OR another character must say their name out loud in the dialogue.
This applies to EVERY named supporting character — sidekick, mentor, anyone. Never let a character appear silently without being named.

=== CHARACTER CONTINUITY RULE (CRITICAL) ===
Once a character has appeared in an earlier panel they are ALREADY PRESENT — do NOT write them as arriving, appearing, or being introduced again in a later panel.
- WRONG: Harmonie is in panel 1, then panel 3 narration says "Harmonie arrives sensing something is off." — she was already there!
- RIGHT: If Harmonie was in panel 1, panel 3 narration should continue her existing presence: "Harmonie notices something strange" or "Harmonie points at the shadow."
Track every character who appears across all 3 panels and keep their presence consistent. A character cannot both be present earlier AND arrive later.

=== SUPPORTING CHARACTER CONSISTENCY (CRITICAL) ===
This applies to EVERY named character in the story — best friend, coach, rival, parent, teacher, anyone who appears more than once.

BEFORE writing any pages, mentally define a locked appearance for every recurring character:
- Best friend / sidekick
- Coach, teacher, or mentor
- Rival or antagonist
- Any other named character who appears in 2+ panels

For EACH recurring character, on their FIRST appearance in an image_prompt write their complete description:
  age, gender, skin tone, exact hair (color + style), exact clothing (every item)

Then copy that EXACT description word-for-word into EVERY subsequent panel where they appear.

NEVER describe the same character differently across panels. If the coach is "tall adult woman, medium brown skin, short natural black hair, red tracksuit" in panel 2, she must be described with those exact words in panels 3, 4, and beyond.

Example — coach introduced in panel 2:
  "...their coach, a tall adult woman with medium brown skin, short natural black hair, wearing a red tracksuit and white sneakers, demonstrates the move..."
Example — same coach in panel 4:
  "...their coach, tall adult woman, medium brown skin, short natural black hair, red tracksuit, white sneakers, watches from the sideline..."
{npc_bible_block}
=== PAGE SPREAD CONTINUITY (CRITICAL) ===
Panels are printed as two-panel page spreads: (1,2), (3,4), etc.
The two panels on the same spread MUST take place in the same location or a direct continuation of it.
- Panels 1 and 2: same setting
- Panels 3 and 4: same setting (can be a new location from panels 1-2)
NEVER jump to a completely different environment between two panels on the same spread.
Think of each spread as one scene, and each new spread as a new scene.
CRITICAL — narration must match the image setting: do NOT write narration that references a location not shown in the image — if panels 3-4 are on a soccer field, the narration must not mention the attic or any other location not in the image.
CRITICAL — if the narration for panel 3 says "gathering friends" or "secret meeting" or "making a plan," panel 4 must ALSO be in that same meeting location (e.g. a garage, a bedroom, a backyard) — do NOT set panel 3 in a garage and panel 4 in a stadium — both panels on the same spread must share the SAME location.

=== TIME OF DAY & WEATHER CONTINUITY (CRITICAL, WHOLE STORY) ===
Whatever time of day and weather are established in the story's opening panel (day/night, sunny/raining/stormy, etc.) MUST stay consistent across EVERY subsequent panel's image_prompt — not just within a spread — unless the prose explicitly narrates time passing (e.g. "the next morning," "hours later") or a change of location.
- If it's established as nighttime and raining in panel 1, every panel remains nighttime and raining (or the weather can only ease/change if the prose says so) until the prose narrates a time skip.
- NEVER let a panel suddenly show bright daylight/sun when the prose has not stated time has passed — this breaks visual continuity and confuses the reader.
- If the story's setting is a single continuous scene (as in the SETTING LOCK), the time of day should typically not change at all across the whole story.
{chunk_continuation_block}
=== OUTPUT RULE ===
The image_prompt must be a single detailed paragraph combining all the above.

General:
- Keep the main character visually consistent across all pages
- Reflect appearance, clothing, and powers in visuals
- Make scenes dynamic, colorful, and cinematic

Return ONLY valid JSON.
Wrap the result in this exact object shape:
{{
  "outfit": {outfit_schema_desc},
  "pages": [
    {{
      "page_number": {chunk_range[0] if chunk_range else 1},
      "narration": "string",
      "dialogue": [
        {{
            "speaker": "string",
            "text": "string",
            "side": "left or right — which side of the panel this speaker appears on"
        }}
      ],
      "characters_present": ["list of every named character actually drawn/visible in this panel's image_prompt — see CHARACTER PRESENCE rules above"],
      "setting": "short 3-8 word location label, reused word-for-word across every panel at this same location — see SETTING LABEL rules above",
      "blocking": "short string stating each present character's physical position relative to the others (e.g. who's driving/front seat/back seat, or who's standing left/right), reused word-for-word across every panel at this same setting with the same characters — see BLOCKING LABEL rules above",
      "image_prompt": "string"
    }}
  ]
}}
"""

    client = get_client()

    try:
        if use_claude:
            content = _call_claude_json(
                prompt=prompt,
                system="You must return only valid JSON.",
                max_tokens=64000,
            )
            print("RAW CLAUDE PREVIEW CONTENT:", repr(content))
        else:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You must return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                timeout=max(120, page_count * 6),
            )
            content = response.choices[0].message.content
            print("RAW OPENAI PREVIEW CONTENT:", repr(content))

        data = json.loads(content)

        if isinstance(data, list):
            pages = data
            outfit = ""
        elif isinstance(data, dict):
            if "pages" not in data:
                raise RuntimeError("OpenAI JSON missing expected key: 'pages'")
            pages = data["pages"]
            outfit = data.get("outfit", "")
        else:
            raise RuntimeError("OpenAI JSON missing expected key: 'pages'")

        if locked_outfit:
            outfit = locked_outfit

        if outfit:
            pages = _enforce_outfit_across_pages(pages, outfit, client, hero_name=hero_name)

        if outfit_out is not None:
            outfit_out["outfit"] = outfit

        return pages

    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI connection failed: {repr(e)}")
    except AuthenticationError as e:
        raise RuntimeError(f"OpenAI authentication failed: {repr(e)}")
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI quota/rate limit failed: {repr(e)}")
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI status error {e.status_code}: {repr(e)}")
    except json.JSONDecodeError as e:
        snippet_start = max(0, e.pos - 200)
        snippet = content[snippet_start:e.pos + 200] if content else "<no content captured>"
        near_end = content is not None and (len(content) - e.pos) < 50
        raise RuntimeError(
            f"OpenAI returned non-JSON content: {e} "
            f"(content length {len(content) if content else 0}, error at char {e.pos}"
            f"{', within 50 chars of the end -- likely truncated by max_tokens' if near_end else ''}) "
            f"context around error: ...{snippet}..."
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI error: {type(e).__name__}: {e}")


def _panel_script_repetition_clusters(pages: list, names: list = None, threshold: float = 0.68, min_len: int = 12, min_cluster: int = 3) -> list:
    """Ground-truth check mirroring tests/regression/checks.py's detector (kept
    as a separate, dependency-free copy here since app/ code must not import
    from tests/) -- used to decide whether dedupe_panel_script_ai needs
    another pass, not just to trust the correction call's own say-so.

    Strips character names before comparing, and the threshold is raised from
    an earlier 0.6. Found empirically on real production output: two short
    lines that just happen to share a character's name and similar length
    ("Ceiara, stay strong!" vs "Ceiara, I feel it too!") scored 0.60-0.71 on
    raw string similarity despite being completely different sentiments --
    almost entirely driven by both containing "Ceiara", not by actual
    repeated content. Genuine repeated-declaration padding ("This is going to
    be epic!" vs "...legendary!" vs "...wild!") scored 0.63-0.88 over the
    same measure. The two distributions overlap too much to separate with a
    threshold alone, which is why name-stripping matters: with character
    names removed first, the false-positive pairs drop to 0.43-0.65 while the
    genuine padding examples stay 0.63-0.88, which 0.68 actually separates.
    """
    import difflib
    import re

    name_pattern = None
    if names:
        escaped = [re.escape(n) for n in names if n]
        if escaped:
            name_pattern = re.compile("|".join(escaped), re.IGNORECASE)

    def _normalize(text: str) -> str:
        if name_pattern:
            text = name_pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def _clusters(lines):
        candidates = [l for l in lines if l and len(l) >= min_len]
        normalized = [_normalize(l) for l in candidates]
        used = set()
        found = []
        for i in range(len(candidates)):
            if i in used:
                continue
            group = [candidates[i]]
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                if difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio() >= threshold:
                    group.append(candidates[j])
                    used.add(j)
            if len(group) >= min_cluster:
                found.append(group)
                used.add(i)
        return found

    dialogue_lines = [
        d.get("text") or d.get("line") or ""
        for p in pages for d in (p.get("dialogue") or [])
    ]
    narration_lines = [p.get("narration") or "" for p in pages]
    return _clusters(dialogue_lines) + _clusters(narration_lines)


def dedupe_panel_script_ai(pages: list, hero_name: str, page_count: int, catchphrase: str = None, other_names: list = None, max_passes: int = 3, warnings_out: list = None) -> list:
    """
    Runs _dedupe_panel_script_pass_ai repeatedly, re-checking for repeated-
    sentiment clusters after each pass, until clean or max_passes is reached.

    One pass alone proved inconsistent in practice -- sometimes fully cleaned
    a script, sometimes left several clusters untouched, since judging "is
    this the same sentiment as an earlier line" is exactly the kind of
    holistic call that doesn't reliably converge in a single shot. Same
    check-then-retry shape as expand_prose_if_short_ai and
    verify_and_fix_mystery_suspect_closure_ai elsewhere in this file -- that
    pattern has proven to work here before.

    If clusters remain after max_passes, returns the best attempt as-is; the
    regression suite's own repetition check is what surfaces that remainder
    for a human to glance at before a book ships. `warnings_out`, if given,
    gets a human-readable note appended when that happens -- same out-param
    pattern as outfit_out elsewhere in this file -- so a caller (e.g. the
    admin notification email) can flag it as something worth a closer look,
    instead of the note only ever reaching a Railway log line.
    """
    names = [hero_name] + list(other_names or [])
    current = pages
    for attempt in range(1, max_passes + 1):
        current = _dedupe_panel_script_pass_ai(current, hero_name=hero_name, page_count=page_count, catchphrase=catchphrase)
        remaining = _panel_script_repetition_clusters(current, names=names)
        if not remaining:
            if attempt > 1:
                print(f"✅ Panel dedup: clean after {attempt} pass(es)")
            return current
        print(f"⚠️  Panel dedup pass {attempt}/{max_passes}: {len(remaining)} repeated-sentiment cluster(s) remain"
              + (" — retrying" if attempt < max_passes else " — leaving for review"))
    if warnings_out is not None:
        warnings_out.append(f"Panel dialogue/narration may still have repeated lines after {max_passes} cleanup passes — worth a scan.")
    return current


def _dedupe_panel_script_pass_ai(pages: list, hero_name: str, page_count: int, catchphrase: str = None) -> list:
    """
    Single dedicated pass over the COMPLETE panel script (all chunks already
    combined) that hunts for repeated-sentiment clusters in dialogue and
    narration -- the failure mode inline instructions and cross-chunk
    prior_context reminders don't reliably prevent on their own. Called in a
    check-then-retry loop by dedupe_panel_script_ai, since a single pass
    doesn't reliably catch everything.

    Found in production even after those inline fixes: a character said
    "This is going to be epic!", then later "...legendary!", then
    "...viral!", then "...wild!", then "...huge!" -- nine separate panels,
    no two lines word-for-word identical, but all the same declaration
    repeated with one word swapped. Asking the generator to self-police this
    live, chunk by chunk, is unreliable no matter how the instruction is
    worded, because judging "is this the same sentiment as an earlier line"
    is a holistic read-the-whole-thing task, not something that fits
    naturally into "generate the next 16 panels." Mirrors punch_up_prose_ai's
    approach for exactly this reason -- that pass proved a dedicated
    read-everything-then-edit call catches cross-scene repetition far more
    reliably than live self-policing during generation.

    Returns the full pages list with only the offending narration/dialogue
    text edited -- panel count, ordering, and every other field (image_prompt,
    setting, blocking, characters_present) are untouched.
    """
    sorted_pages = sorted(pages, key=lambda x: x.get("page_number", 0))
    compact = [
        {
            "page_number": p.get("page_number"),
            "narration": p.get("narration", ""),
            "dialogue": [
                {"speaker": d.get("speaker", ""), "text": d.get("text") or d.get("line") or ""}
                for d in (p.get("dialogue") or [])
            ],
        }
        for p in sorted_pages
    ]

    catchphrase_block = ""
    if catchphrase:
        catchphrase_block = f"""
ONE NAMED EXCEPTION, AND ONLY THIS ONE: the EXACT phrase "{catchphrase}" is {hero_name}'s own established catchphrase, a deliberate character trait, not padding — it is allowed to appear more than once. This exception applies ONLY to this exact phrase (or very close variants of it, still recognizably the same line) and to NOTHING else. It does NOT make other recurring-sounding lines okay — a hype line like "This is going to be legendary!" appearing several times is STILL padding and must still be cut down to one instance even though this one named catchphrase gets to stay. If "{catchphrase}" appears in the script below, keep at least one instance of it said BY {hero_name} — but if it shows up more than twice, trim it down to two instances (the first, and the best-placed later one), never to zero."""

    prompt = f"""You are editing a finished {page_count}-panel comic script for repeated-sentiment padding.

THE FULL SCRIPT (narration and dialogue only, in panel order):
{json.dumps(compact, indent=2)}

YOUR ONLY JOB: find every cluster of 3 or more panels where a character's dialogue OR a narration box states essentially the SAME realization, triumph, or reaction as another panel in the cluster — even when the exact wording differs. This is the single most common padding failure in this kind of adaptation: a character saying "This is going to be epic!", then later "This is going to be legendary!", then later "This is going to be huge!" are the SAME declaration repeated three times, not three different beats, even though no two lines are word-for-word identical.

This applies especially to any "we're famous / this is viral / this is legendary / this is epic / this is huge / look at this" type hype reaction — {hero_name} and every other character are each allowed to get excited about something once per distinct real event in the story, not repeatedly re-announce the same ongoing excitement in new words.

NARRATION IS JUST AS PRONE TO THIS AS DIALOGUE, AND EASIER TO MISS — because each instance swaps in a different specific noun, it can look like variety when it's actually the same beat on repeat. A real production example: "The crowd watches in anticipation, eager for the next spicy challenge," then later "The crowd erupts in cheers as Vincent prepares to unleash the next spicy challenge," then later "The crowd roars in response, eager for the next round of spicy madness" — three narration boxes, three different specific words ("challenge" / "spicy challenge" / "spicy madness"), but all three are just "the crowd is excited for what's next," restated. Likewise "Vincent feels the adrenaline surge as he prepares to unleash the next sauce" and "Vincent feels the adrenaline surge as the crowd clamors for more" are the same internal-state beat twice. Read every narration box as carefully as you read dialogue — don't let a changed noun or verb convince you two boxes are different if the underlying sentiment (crowd is excited / hero feels a rush / this is escalating) is the same one being restated.
{catchphrase_block}
FOR EACH CLUSTER YOU FIND:
- Keep the single best instance unchanged (usually the first, or whichever lands at the most dramatically appropriate point).
- For every OTHER panel in that cluster, replace the repeated line with something that does NOT restate the same sentiment: a different character's reaction, a new specific physical detail of what's actually happening in that panel, or simply remove that panel's dialogue/narration (an empty dialogue array or empty narration string is fine — the image can carry the panel alone).
- Do NOT invent a new plot event, and do NOT touch panels that aren't part of a repeated cluster — those must come back byte-for-byte unchanged.

Return ONLY valid JSON: {{"pages": [{{"page_number": int, "narration": "string", "dialogue": [{{"speaker": "string", "text": "string"}}]}}, ...]}} covering all {page_count} panels in page_number order, including every unchanged one."""

    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a meticulous comic script editor hunting for repeated-sentiment padding. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=16000,
            timeout=180,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            print("⚠️  Panel dedup pass hit the token limit — keeping original panel script untouched")
            return pages
        result = json.loads(choice.message.content)
        edits = {p["page_number"]: p for p in result.get("pages", []) if p.get("page_number") is not None}
        if len(edits) != len(sorted_pages):
            print(f"⚠️  Panel dedup pass returned {len(edits)} panels, expected {len(sorted_pages)} — keeping original")
            return pages

        merged = []
        for p in sorted_pages:
            edit = edits.get(p.get("page_number"))
            if not edit:
                merged.append(p)
                continue
            new_p = dict(p)
            new_p["narration"] = edit.get("narration", p.get("narration", ""))
            original_dialogue = p.get("dialogue") or []
            merged_dialogue = []
            for i, d in enumerate(edit.get("dialogue") or []):
                entry = {"speaker": d.get("speaker", ""), "text": d.get("text", "")}
                if i < len(original_dialogue) and original_dialogue[i].get("side"):
                    entry["side"] = original_dialogue[i]["side"]
                merged_dialogue.append(entry)
            new_p["dialogue"] = merged_dialogue
            merged.append(new_p)
        print("✏️  Panel dedup pass complete")
        return merged
    except Exception as e:
        print(f"⚠️  Panel dedup pass failed — keeping original panel script: {e}")
        return pages


def generate_preview_pages_chunked_ai(
    hero_name: str,
    theme: str,
    logline: str,
    outline,
    character_brief: str,
    page_count: int = 4,
    personalization: dict = None,
    genre: str = "adventure",
    is_preview: bool = True,
    additional_characters: list = None,
    prose_story: str = None,
    npc_bible: list = None,
    use_claude: bool = False,
    chunk_size: int = 16,
    warnings_out: list = None,
) -> list:
    """
    Wraps generate_preview_pages_ai, splitting long books (page_count > 24,
    i.e. 48-panel books) into sequential chunked generation calls instead of
    one massive one-shot request.

    Root cause this fixes: a real 48-panel production book had its full
    panel script generated in a single call, and the model ran out of
    genuinely new content partway through -- panels 24-47 turned out to be
    near word-for-word duplicates of panels 13-36, and the story's actual
    specified climax never made it into any panel at all, in either the
    original attempt or the from-scratch retry. Prose generation already
    avoids this exact failure mode by splitting into 8 scene-by-scene calls
    (see generate_prose_story_ai) -- panel adaptation had no equivalent
    chunking, asking a single call to sustain ~48 panels (10,000+ tokens) of
    coherent, non-repeating structured output, which the 8-scene prose
    approach never has to do.

    24-panel books (and previews, always tiny) stay on the existing
    single-call path entirely unchanged -- proven reliable at that size
    across extensive testing; only page counts above 24 use chunking.
    """
    if page_count <= 24 or is_preview:
        return generate_preview_pages_ai(
            hero_name=hero_name, theme=theme, logline=logline, outline=outline,
            character_brief=character_brief, page_count=page_count,
            personalization=personalization, genre=genre, is_preview=is_preview,
            additional_characters=additional_characters, prose_story=prose_story,
            npc_bible=npc_bible, use_claude=use_claude,
        )

    if chunk_size % 2 != 0:
        chunk_size += 1
    chunks: list = []
    start = 1
    while start <= page_count:
        end = min(start + chunk_size - 1, page_count)
        chunks.append((start, end))
        start = end + 1

    all_pages: list = []
    locked_outfit: Optional[str] = None
    for chunk_start, chunk_end in chunks:
        prior_context = None
        if all_pages:
            recap_lines = [
                f"  Panel {p.get('page_number')}: setting=\"{p.get('setting', '')}\", "
                f"blocking=\"{p.get('blocking', '')}\", characters_present={p.get('characters_present', [])}, "
                f"narration=\"{p.get('narration', '')}\""
                for p in all_pages[-3:]
            ]
            prior_context = (
                "The most recent panels already generated (for continuity — do not repeat these):\n"
                + "\n".join(recap_lines)
            )
            # Every dialogue line spoken so far in the WHOLE book, not just this
            # recap window -- a later chunk can't see an earlier chunk's exact
            # wording otherwise, and independently reaches for the same generic
            # "we're famous / this is legendary" reaction line a different way
            # in each chunk, which reads as a pile of repeated declarations once
            # the chunks are stitched together (found in a real production book).
            # NOTE: dialogue entries use the key "text" (see the schema below,
            # "text": "string") -- an earlier version of this read d.get("line"),
            # which is always empty, so this cross-chunk tracking never actually
            # activated until this fix.
            all_prior_lines = [
                f'{d.get("speaker", "")}: "{d.get("text") or d.get("line") or ""}"'
                for p in all_pages
                for d in (p.get("dialogue") or [])
                if (d.get("text") or d.get("line"))
            ]
            if all_prior_lines:
                prior_context += (
                    "\n\nEvery dialogue line already used earlier in this book — do NOT repeat any of these, "
                    "and do NOT write a new line that just restates the same sentiment in different words "
                    "(especially any \"we're famous / this is viral / this is legendary\" type reaction — "
                    "that beat has already happened if it appears below):\n"
                    + "\n".join(f"  {line}" for line in all_prior_lines)
                )
            # Narration boxes repeat the same sentiment just as easily as dialogue
            # does (found in production: "Vincent realized the chaos was turning
            # into a marketing goldmine" and near-identical restatements across
            # several panels, with no dialogue involved at all) -- track these
            # separately since they're a different field with their own voice.
            all_prior_narration = [p.get("narration") for p in all_pages if p.get("narration")]
            if all_prior_narration:
                prior_context += (
                    "\n\nEvery narration box already used earlier in this book — do NOT repeat the same "
                    "realization or sentiment in a new narration box worded differently (e.g. restating "
                    "\"this is going viral\" or \"the chaos was escalating\" again in different words once "
                    "it's already appeared below):\n"
                    + "\n".join(f"  {line}" for line in all_prior_narration)
                )
        outfit_out: dict = {}
        chunk_pages = generate_preview_pages_ai(
            hero_name=hero_name, theme=theme, logline=logline, outline=outline,
            character_brief=character_brief, page_count=page_count,
            personalization=personalization, genre=genre, is_preview=is_preview,
            additional_characters=additional_characters, prose_story=prose_story,
            npc_bible=npc_bible, use_claude=use_claude,
            chunk_range=(chunk_start, chunk_end), prior_context=prior_context,
            locked_outfit=locked_outfit, outfit_out=outfit_out,
        )
        if locked_outfit is None:
            locked_outfit = outfit_out.get("outfit") or None
        expected = chunk_end - chunk_start + 1
        if len(chunk_pages) != expected:
            print(f"⚠️  Chunked adaptation: panels {chunk_start}-{chunk_end} expected {expected} pages, got {len(chunk_pages)}")
        all_pages.extend(chunk_pages)
        print(f"  [chunked adaptation] panels {chunk_start}-{chunk_end} generated ({len(chunk_pages)} pages)")

    try:
        other_names = [c.get("name") for c in (additional_characters or []) if c.get("name")]
        other_names += [n.get("name") for n in (npc_bible or []) if n.get("name")]
        all_pages = dedupe_panel_script_ai(
            all_pages, hero_name=hero_name, page_count=page_count,
            catchphrase=(personalization or {}).get("catchphrase"),
            other_names=other_names, warnings_out=warnings_out,
        )
    except Exception as e:
        print(f"⚠️  Panel dedup pass crashed — using panel script as generated: {e}")

    return all_pages


def build_character_reference_prompt(character_brief: str) -> str:
    return f"""
Create a highly detailed semi-realistic child character reference image.

This should look like a real person illustrated in a semi-realistic cinematic graphic novel style.


No background story scene.
No action pose.
No extra characters.
No animals.
No props.
No dramatic effects.
No motion blur.
No environmental storytelling.

Character inspiration:
{character_brief}

CHARACTER DESIGN RULES:
- create an original fictional comic character design
- keep the design consistent and recognizable
- clear hairstyle
- clear outfit
- clear color palette
- full outfit visible
- believable human facial expression
- realistic human proportions
- fully clothed
- age-appropriate appearance
- no celebrity likeness
- no photorealism
- no attempt to recreate a real person's face

STYLE:
- semi-realistic (NOT cartoon)
- semi-realistic facial proportions
- natural skin texture and subtle imperfections
- detailed hair strands
- realistic lighting and shadows
- cinematic depth of field
- semi-realistic cinematic graphic novel quality
- detailed comic book page rendering
- believable human anatomy and facial structure

STRICTLY AVOID:
- cartoon style
- Pixar style
- oversized eyes
- rounded toy-like faces
- smooth plastic skin
- exaggerated expressions

POSE RULES:
- standing upright
- relaxed neutral heroic pose
- full body visible
- facing forward or slight 3/4 angle
"""

def generate_character_identity_lock(character_brief: str) -> str:
    client = get_client()

    prompt = f"""
You create reusable visual identity locks for children's comic characters.

Based on this character brief, extract a consistent visual identity.

Character brief:
{character_brief}

Return ONLY valid JSON with this exact shape:
{{
  "hair": "string",
  "face": "string",
  "eyes": "string",
  "outfit": "string",
  "signature_marker": "string",
  "color_palette": "string",
  "style": "polished children's comic book illustration"
}}

Rules:
- Do not invent unsafe details.
- Do not mention real people or photo likeness.
- Make the details reusable across many comic scenes.
- Keep it specific enough to preserve consistency.
- Include exact eye color if visible
- Be specific about facial features that define identity
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=20,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        return json.dumps(data, indent=2)

    except Exception as e:
        raise RuntimeError(f"Character identity lock generation failed: {str(e)}")

def generate_character_reference_image(character_brief: str) -> tuple[str, str]:
    prompt = build_character_reference_prompt(character_brief)
    image_base64 = generate_image_from_prompt(prompt)

    identity_lock = generate_character_identity_lock(character_brief)

    return image_base64, identity_lock
   
def generate_image_from_prompt(prompt: str) -> str:
    client = get_client()

    try:
        response = client.images.generate(
            model="gpt-image-2.5-sunburst",
            prompt=prompt,
            size="1024x1024"
        )

        image_base64 = response.data[0].b64_json
        return image_base64

    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")
    


def decode_base64_image(image_base64: str) -> bytes:
    return base64.b64decode(image_base64)



def generate_image_with_reference(
    prompt: str,
    reference_image_bytes: bytes,
    photo_bytes_list: list[bytes] = None,
    prior_page_bytes: list[bytes] = None,
) -> str:
    client = get_client()

    try:
        comic_ref = BytesIO(reference_image_bytes)
        comic_ref.name = "comic-reference.jpg"
        images = [comic_ref]

        # Add raw uploaded photos as additional face/outfit reference (max 3)
        for i, pb in enumerate((photo_bytes_list or [])[:3]):
            raw_ref = BytesIO(pb)
            raw_ref.name = f"photo-reference-{i+1}.jpg"
            images.append(raw_ref)

        if prior_page_bytes:
            first_page = BytesIO(prior_page_bytes[0])
            first_page.name = "prior-page-1.jpg"
            images.append(first_page)

        response = client.images.edit(
            model="gpt-image-2.5-sunburst",
            image=images,
            prompt=prompt,
            size="1024x1024",
        )

        image_base64 = response.data[0].b64_json
        return image_base64

    except Exception as e:
        raise RuntimeError(f"Image generation with reference failed: {str(e)}")

def generate_book_title_ai(hero_name: str, theme: str, logline: str, audience_age: int = 10) -> str:
    client = get_client()
    if audience_age >= 18:
        age_note = "Bold and dramatic — suitable for an adult audience."
    else:
        age_note = "Exciting and adventurous, age-appropriate for kids 6-12."
    prompt = f"""Generate a short, exciting title for a personalized comic book.

Hero's name: {hero_name}
Theme: {theme}
Story: {logline}

Rules:
- 3 to 6 words maximum
- Must include the hero's name
- {age_note}
- Comic book style (e.g. "Vincent and the Hidden Map")

Return ONLY the title text. No quotes, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=20,
        timeout=15,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


def generate_cover_line_ai(hero_name: str, logline: str, audience_age: int = 10) -> str:
    client = get_client()
    if audience_age >= 18:
        tone_note = "Confident, bold, and punchy — adult tone, not childlike."
    else:
        tone_note = "Energetic, fun, age-appropriate."
    prompt = f"""Write a single short, exciting speech bubble line for the cover of a comic book.

The hero's name is {hero_name}.
Story: {logline}

Rules:
- 4 to 8 words maximum
- {tone_note}
- Something the hero would say at the start of an adventure
- No punctuation at the end except ! or ?
- Examples: "This is going to be epic!", "Adventure starts NOW!", "Let's solve this mystery!"

Return ONLY the line. No quotes, no explanation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=20,
        timeout=15,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


def build_cover_page_prompt(hero_name: str, theme: str, logline: str, character_brief: str, genre_style: dict = None) -> str:
    genre_line = f"\nGENRE VISUAL STYLE: {genre_style['image_style']}\n" if genre_style else ""
    return f"""
Create a dramatic, exciting COVER PAGE illustration for a children's comic book.
{genre_line}

CHARACTER (match the reference character sheet exactly):
{character_brief}

COMPOSITION — MOVIE POSTER STYLE (CRITICAL):
Think of this like a movie poster. The title will be printed in large text across the TOP of the image. The character must be fully below where the title appears.

EXACT CHARACTER PLACEMENT:
- Divide the image into 5 equal horizontal bands from top to bottom
- Bands 1 and 2 (top 40% of image) = BACKGROUND ONLY — sky, clouds, scenery — absolutely NO character, NO limbs, NO hair
- The character's HEAD must appear in band 3 or lower (below the 40% mark from the top)
- The character's FEET must be near the bottom edge of the image
- The character occupies the BOTTOM 55-60% of the image only
- Wide or medium-wide establishing shot — do NOT fill the frame with a close-up

CHARACTER POSE:
- Confident, heroic ground-level pose — wide stance, hands on hips, or arms at sides
- Both feet planted on the ground — NO jumping, leaping, or floating poses
- NO arms raised above the head — keep arms at shoulder level or below
- Character centered or slightly off-center horizontally
- ANGLE: character turned 30-45 degrees to a 3/4 view — NOT facing straight at the camera — the body and face should be angled slightly to one side, like a classic comic book hero pose — this gives depth and a more natural face shape

FACIAL EXPRESSION:
- Natural, confident smile — mouth closed or slightly open showing just a hint of teeth
- NOT an exaggerated wide-open grin, NOT oversized teeth, NOT a cartoon ear-to-ear smile
- The mouth must be proportionate to the face — no wider than the distance between the pupils

BACKGROUND:
- Top 40% of image: open sky, bright clouds, or wide scenery — rich and colorful
- BRIGHT, VIVID lighting — cheerful and energetic, NOT dark, NOT stormy, NOT moody
- Even if the story involves weather or chaos, keep the overall image BRIGHT and colorful

STORY CONTEXT:
- Hero: {hero_name}
- Theme: {theme}
- Story: {logline}

STYLE (LOCKED — match the reference character sheet exactly):
- Premium professional comic book illustration — clean precise ink linework, flat color fills with cel-shading, like a modern DC or Image Comics graphic novel cover
- Clean precise ink lines around the character, hair, clothing, and all elements — medium-weight, sharp, detailed professional linework — NOT overly thick or chunky
- Cel-shading: flat color base with sharp 2-3 tonal shadow zones — graphic and clean
- Skin: flat deep dark brown base with sharp shadow zones — maintain exact skin depth from reference
- Hair: flat base color with precise ink strand lines — detailed and defined
- Vibrant, saturated comic book colors — bright, punchy, high-contrast cover palette
- Richly detailed background — intricate environment with depth behind the character
- Dynamic, dramatic composition befitting a premium comic book cover — character pops off the detailed background
- STRICTLY AVOID: soft gradient blending, 3D render look, oversized eyes, chibi proportions, anime style, watercolor wash, photorealism, overly thick chunky outlines

ABSOLUTELY NO: text, words, letters, signs, speech bubbles, thought bubbles, captions, or any written content anywhere in the image. The image must be 100% illustration only.
"""


def build_cartoon_page_prompt(image_prompt: str, character_brief: str, genre_style: dict = None) -> str:
    genre_line = f"\nGENRE VISUAL STYLE: {genre_style['image_style']}\n" if genre_style else ""
    return f"""
Draw the following comic book scene in exactly the same polished storybook illustration style as the reference character sheet provided.
{genre_line}

CHARACTER IDENTITY (match the reference character sheet exactly):
{character_brief}

IDENTITY RULES — this is the SAME child shown throughout the entire comic. Every panel must be consistent:
- Same face structure as the reference — same face shape, same jaw width, same chin — do NOT change the face shape between panels
- AGE: This is a YOUNG CHILD — eyes large relative to the face, nose short and soft, chin small and rounded, NO sharp angular adult features — do NOT give the character a strong jawline, high cheekbones, or any feature that reads as a teenager or adult — if the face looks older than a child it is WRONG
- Same eyes — do NOT change proportions, do NOT age up
- Same nose — copy the EXACT nose width and shape from the reference image in every single panel — do NOT narrow the nose, do NOT make it more pointed — if the reference shows a broad rounded nose, every panel must show a broad rounded nose
- Same hair: same exact color, length, and texture style (locs stay locs, braids stay braids — NEVER change)
- HEIGHT: This child is the same size as their friends — do NOT draw the main character significantly taller or larger than other child characters in the scene — same-age children are the same height
- Same skin tone — do NOT lighten between scenes — if the reference shows deep dark brown skin, every panel must maintain that same deep dark brown skin
- Same outfit as described in the SCENE PROMPT below — the scene establishes the story outfit (leotard, jersey, everyday clothes, etc.)
- CRITICAL: do NOT revert to the reference image's everyday clothes in any panel — even when the character is watching, reacting, standing still, or not actively doing the sport — the story outfit stays on for the ENTIRE comic with no exceptions
- If the story outfit is a leotard: every single panel shows ONLY the leotard — do NOT add a jacket, cardigan, hoodie, or any outer layer over the leotard unless the scene prompt explicitly describes one — the reference image may show a jacket but that jacket does NOT appear in the panels
- If the story outfit is a sports jersey: every single panel shows ONLY the jersey and matching shorts — do NOT add a jacket, vest, hoodie, or any outer layer over the jersey — the reference image may show a jacket but that jacket does NOT appear in the panels
- Glasses: copy the EXACT frame shape, color, and thickness from the reference — if the reference has white rectangular glasses, every panel must have white rectangular glasses
- Mouth: natural smile — width must NOT exceed the distance between the pupils — NOT an ear-to-ear grin
- The character's face must be RECOGNIZABLE as the same person in every single panel — slim narrow face, not a generic rounded cartoon face

RENDERING STYLE (FIXED — use this style regardless of how the reference looks):
Use the reference image for FACE, HAIR, and SKIN TONE ONLY.
DO NOT copy the clothing from the reference image — the outfit is defined entirely by the scene prompt below.
If the reference image shows a jacket, shirt, or any everyday clothes — IGNORE those completely. Draw the outfit described in the scene prompt instead.

Required rendering style:
- Premium professional comic book illustration — clean precise ink linework, flat color fills with cel-shading, like a modern DC or Image Comics graphic novel for younger readers
- Clean precise ink lines around all characters, hair, clothing, and elements — medium-weight lines, sharp and detailed like a professional inker — NOT overly thick or chunky
- Cel-shading: flat color base fills with sharp 2-3 tonal shadow zones — clean graphic shading, NOT soft gradient blending
- Skin: flat base tone (deep dark brown if reference shows dark skin) with 1-2 sharp shadow zones — maintain exact skin depth from reference
- Hair: flat base color with precise ink strand lines and sharp shadow shapes — detailed and defined
- Eyes: natural proportionate size — NOT enlarged, NOT anime-style
- Mouth: relaxed, slight smile — closed or minimally open — NOT a wide grin, NOT oversized teeth
- Head: natural child proportions — head no larger than 1/4 of total body height — NOT chibi
- Vibrant saturated comic book colors — bright blues, vivid greens, rich purples, warm reds — punchy and high-contrast
- BACKGROUNDS: richly detailed and fully illustrated — intricate environments with depth and texture in every panel
- NEVER: soft gradient blending, 3D render look, chibi proportions, anime style, watercolor, oil-painting texture, photorealism, overly thick chunky outlines

LIGHTING (CRITICAL):
- ALL scenes must be BRIGHT and well-lit — cheerful, vibrant lighting
- Interior scenes: bright overhead lighting, large windows with daylight — NOT dim, NOT warm amber lamp-only lighting
- Exterior scenes: bright daylight or golden hour — NOT overcast, NOT dark
- Exception: horror, horror_adult, and mystery genres may use moody or dim lighting
- The overall image must feel bright and energetic, never dark or oppressive

FRAMING (MANDATORY — these are hard rules, not suggestions):
- WIDE SHOT is the default — character(s) visible head-to-toe, environment fills at least 60% of the frame
- MEDIUM SHOT only when the scene requires it — character visible from mid-thigh up, NOT waist-only
- The character's head must occupy NO MORE than 20% of the total image height — if the head is larger than this, zoom out
- NEVER a close-up, medium close-up, or head-and-shoulders shot of any kind
- PROHIBITED shot types: CU (close-up), MCU (medium close-up), ECU (extreme close-up), over-the-shoulder tight shot, face portrait
- If the scene description implies closeness or reaction, STILL use a wide or medium shot — pull the camera back
- Camera distance: imagine a filmmaker standing 15-20 feet from the characters

MULTI-CHARACTER FRAMING (CRITICAL):
- If TWO OR MORE characters appear in the same panel, ALL characters must be FULLY VISIBLE head-to-toe
- NEVER render a two-shot as a face close-up — this is the most common error to avoid
- Both characters must be small enough that the background environment is clearly visible around and between them
- Do NOT fill the panel with two large faces — show two small full-body figures in a wide environment

SETTING (MANDATORY):
- If the scene description says INTERIOR or mentions an indoor location (gym, room, hallway, building, house, classroom, etc.) — the background MUST show interior architecture: walls, ceiling, furniture, and indoor lighting. Do NOT show sky, trees, or any outdoor elements.
- If the scene description says EXTERIOR or mentions an outdoor location — the background must show outdoor environment.
- READ the scene description carefully and match the background exactly. Do NOT default to outdoors.

SUPPORTING CHARACTER RULES (CRITICAL):
- Supporting characters must look CLEARLY DIFFERENT from the main character
- Supporting characters must NOT have dreadlocks or locs — they must have different hair (short afro, short curly, straight, etc.)
- Supporting characters must NOT wear glasses of any kind
- Render each supporting character with the EXACT appearance described in the scene prompt — same hair, skin tone, and clothing
- Do NOT redesign or reinterpret any supporting character between scenes
SCENE TO ILLUSTRATE:
⚠️ FRAMING OVERRIDE — READ BEFORE RENDERING:
- Use WIDE SHOT. The character(s) must be SMALL FIGURES in a large environment — like dolls in a room, not faces filling the frame.
- Character head(s) must take up NO MORE than 15% of the image height — smaller than you think.
- Character full height should be no more than 60% of the total image height — leave space above and below.
- If the scene has TWO characters, BOTH must be fully visible from head to toe — no exceptions.
- No close-ups. No medium close-ups. No face portraits. No two-shot face frames.
- If the description below implies any kind of tight or close shot — IGNORE that implication and render WIDE SHOT instead.

⚠️ OUTFIT OVERRIDE:
- The character wears EXACTLY the outfit described in the scene below — copy the colors, pattern, and sleeve style precisely.
- Do NOT change or substitute the outfit.
- If the outfit is a leotard — do NOT add a jacket, hoodie, or any outer layer over it.
- If the outfit is a sports jersey — do NOT add a jacket, vest, hoodie, or any outer layer over it. The jersey is the only top garment visible.

{image_prompt}

⚠️ FINAL FRAMING CHECK — verify before finishing:
1. Are ALL characters visible from head to toe? If NO — zoom out.
2. Is the background environment clearly visible around the characters? If NO — zoom out.
3. Does any character's head take up more than 20% of the image height? If YES — zoom out.
4. Are there two characters shown only from the chest up? If YES — this is wrong — zoom out to show both full bodies.

The character in this scene must look identical to the reference illustration — same face, hair, skin, outfit, and art style. Zero variation.
"""

def build_back_cover_prompt(hero_name: str, logline: str, character_brief: str, genre_style: dict = None) -> str:
    genre_line = f"\nGENRE VISUAL STYLE: {genre_style['image_style']}\n" if genre_style else ""
    return f"""
Create a BACK COVER illustration for a children's comic book.
{genre_line}
CHARACTER (match the reference character sheet exactly):
{character_brief}

COMPOSITION:
- The character in a relaxed, satisfied, or triumphant pose — adventure over, mission complete
- Half-body or three-quarter body shot showing face, torso, and upper arms clearly
- Softer, quieter mood than the front cover — reflective, warm, inviting
- Leave the BOTTOM 40% of the image relatively clear and uncluttered for text overlay
- BACKGROUND: Use a soft atmospheric or abstract background — dark trees, moody sky, glowing light, or
  gentle environmental texture that matches the story genre. Do NOT show a specific building, house, or
  architecture — the background must be impressionistic and non-specific so it reads as a mood, not a location.

STORY CONTEXT:
- Hero: {hero_name}
- Story: {logline}

STYLE (LOCKED — match the reference character sheet exactly):
- Traditional American comic book illustration — bold ink outlines, clean flat color fills with cel-shading, like a premium YA graphic novel
- Bold, clean black ink outlines around the character, hair, clothing, and all major elements
- Cel-shading: flat color base with sharp 2-3 tonal shadow zones — graphic and clean
- Skin: flat deep dark brown base with sharp shadow zones — maintain exact skin depth from reference
- Hair: flat base color with bold ink strand lines — graphic and defined
- Vibrant, saturated comic book colors — warm, inviting back-cover palette
- STRICTLY AVOID: soft gradient blending, 3D render look, oversized eyes, chibi proportions, anime style, watercolor wash, photorealism

CRITICAL: NO TEXT, NO WORDS, NO LETTERS, NO SIGNS with writing in the image.
"""


def generate_back_cover_copy_ai(hero_name: str, logline: str, outline: str) -> dict:
    client = get_client()
    prompt = f"""Write back cover copy for a personalized children's comic book.

Hero: {hero_name}
Story: {logline}
Outline: {outline}

Return ONLY valid JSON with this shape:
{{
  "synopsis": "2 SHORT sentences max 160 characters total — exciting hook for a kid reader, present tense",
  "teaser": "1 short exciting sentence teasing the next adventure (8-12 words)"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=200,
        timeout=20,
    )
    return json.loads(response.choices[0].message.content)


def detect_speaker_positions(image_bytes: bytes, speaker_names: list) -> dict:
    """Use GPT-4o-mini vision to find each speaker's head x-position as a 0.0–1.0 fraction."""
    if not speaker_names:
        return {}
    client = get_client()
    names_str = ", ".join(speaker_names)
    b64 = base64.b64encode(image_bytes).decode()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "low",
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Find these characters in this comic panel: {names_str}. "
                            f"For each visible character, estimate the horizontal center of their head "
                            f"as a decimal fraction of the image width (0.0=left edge, 1.0=right edge). "
                            f"Return ONLY valid JSON, e.g. {{\"Ceiara\": 0.28, \"Vincent\": 0.71}}. "
                            f"Omit any character not clearly visible."
                        )
                    }
                ]
            }]
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return {k.lower(): max(0.05, min(0.95, float(v))) for k, v in data.items()}
    except Exception as e:
        print(f"[vision] detect_speaker_positions failed: {e}")
        return {}


def verify_characters_visible_ai(image_bytes: bytes, character_names: list) -> list:
    """
    Vision ground-truth check: which of these named characters, if any, are
    NOT clearly visible in this image? Returns the list of missing names
    (empty list = everyone's there).

    Built as a dedicated, more careful check rather than reusing
    detect_speaker_positions (designed for a different, more tolerant job --
    roughly where to put a speech bubble -- using detail="low" for cost/
    speed). Found in production that reusing it for a hard pass/fail gate
    was unreliable: the SAME generated image, checked twice (once by this
    gate right after generation, once later by detect_speaker_positions
    during compositing for bubble placement), gave DIFFERENT answers --
    present on the first check, missing on the second. That's noise in a
    detail="low" vision call being asked a harder question than it was
    designed for, not a code bug in the retry logic itself. This uses
    detail="high" and asks a single, narrow, explicit question per character
    instead of trying to also estimate head position, since the two
    questions have different reliability needs: approximate bubble placement
    can tolerate noise, a generate-or-retry gate can't.

    Fails OPEN (returns []) on any error -- a vision-check outage must never
    block generation.
    """
    if not character_names:
        return []
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode()
    names_str = ", ".join(character_names)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This comic panel is supposed to show these named characters: {names_str}.\n"
                            f"For EACH name, look carefully at the whole image and decide: is a clearly visible "
                            f"human figure actually present for that character (even if small, partly turned "
                            f"away, or in the background) -- not just a prop, a screen, or an empty scene? "
                            f"Return ONLY valid JSON mapping each name to true or false, e.g. "
                            f'{{"Ceiara": true, "Vincent": false}}. No other text.'
                        ),
                    },
                ],
            }],
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return [n for n in character_names if not data.get(n, True)]
    except Exception as e:
        print(f"[vision] verify_characters_visible_ai failed (failing open): {e}")
        return []


def verify_panel_image_matches_narration_ai(image_bytes: bytes, narration: str, image_prompt: str) -> bool:
    """
    Vision ground-truth check: does the generated panel image actually depict
    what its own narration/image_prompt describe?

    Found in production: panel 1 of an otherwise fully-verified 48-panel book
    showed two people unpacking moving boxes, while its own narration
    described the hero lounging on a chaise reading a novel -- completely
    unrelated scenes. Audited every pairing/ordering path that assigns an
    image to a panel number (the LlamaGen batch response sort, the parallel
    single-panel fallback's future-to-entry mapping, the llamagen_deferred
    construction loop) and found all three correctly paired -- this was not
    a code bug pairing the wrong image to the wrong panel, it was the image
    model itself drawing something unrelated to an otherwise-correct prompt.
    Image generation isn't deterministic; the model can simply miss.

    Used as the ground-truth half of a generate-then-verify-then-retry loop
    for panel 1 specifically (the highest-visibility panel, and the one with
    a documented real incident) -- the same check-then-retry pattern already
    proven for text elsewhere in this file, applied to an image for the
    first time. Fails OPEN (returns True) on any error -- a vision-check
    outage must never block book generation.
    """
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode()
    key_elements = (narration or "").strip() or (image_prompt or "").strip()
    if not key_elements:
        return True
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This comic panel is supposed to show: \"{key_elements}\"\n"
                            f"Does the image actually depict this scene -- matching setting, action, and "
                            f"which characters are present? Answer with exactly one word: YES or NO."
                        ),
                    },
                ],
            }],
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"[vision] verify_panel_image_matches_narration_ai failed (failing open): {e}")
        return True


def build_final_image_prompt(image_prompt: str, character_brief: str) -> str:
    return f"""
CHARACTER IDENTITY (STRICT):
{character_brief}

CRITICAL:
- This is the SAME exact character in every image
- Face, eyes, hair, and proportions must NOT change
- Eye color must remain consistent

STYLE (LOCKED):
- semi-realistic digital painting, premium concept art quality
- VIBRANT saturated colors — rich, punchy, full-spectrum palette
- varied color temperature — cool backgrounds, warm skin tones, colorful clothing
- painterly textures with detailed skin and hair
- soft cinematic lighting with volumetric depth
- natural realistic proportions
- NOT cartoon, NOT cel shading, NOT flat or muted colors
- consistent semi-realistic rendering style

SCENE:
{image_prompt}
"""