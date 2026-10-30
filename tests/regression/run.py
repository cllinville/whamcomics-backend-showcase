"""Regression test runner for story + panel-script generation quality.

Runs the REAL generation pipeline (app/services/ai.py) end-to-end for a
fixed scenario per genre -- blueprint -> prose -> expand -> structural
checks -> genre fidelity check -> punch-up (comedy only) -> NPC bible ->
chunked panel-script adaptation -- the same sequence book_generation.py
uses for a real 48-panel book, minus image generation (no LlamaGen/S3
calls, so this costs an OpenAI bill but no image credits and no 20-minute
wait).

Usage:
    python -m tests.regression.run                  # all genres
    python -m tests.regression.run --genre comedy    # one genre
    python -m tests.regression.run --genre comedy --page-count 24

Requires a local .env with a real OpenAI key (same as the backend itself).
Output (prose, panel script, and a report) is saved under
tests/regression/output/<genre>/ for manual spot-checking -- the automated
checks catch the mechanical failure patterns found in production, but
genuine craft judgment (is this actually funny, does the mystery actually
make sense) still benefits from a human reading the saved files.
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services import ai
from tests.regression import checks as c
from tests.regression.genre_config import GENRE_CONFIG
from tests.regression.scenarios import SCENARIOS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run_scenario(genre: str, scenario: dict, page_count: int) -> dict:
    cfg = GENRE_CONFIG[genre]
    hero_name = scenario["hero_name"]
    support = scenario["support"]
    personalization = scenario["personalization"]
    catchphrase = personalization.get("catchphrase")
    support_names = [s["name"] for s in support]

    results: list = []
    out_dir = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(out_dir, exist_ok=True)

    # --- Blueprint ---
    blueprint = cfg["blueprint_gen"](
        hero_name=hero_name, logline=scenario["logline"], outline=scenario["outline"],
        personalization=personalization, additional_characters=support, page_count=page_count,
    )
    if not blueprint:
        results.append(("FAIL", "blueprint: generation returned empty"))
        return {"results": results, "prose": None, "panels": None}
    blueprint = cfg["blueprint_validate"](blueprint=blueprint, hero_name=hero_name)
    with open(os.path.join(out_dir, "blueprint.json"), "w", encoding="utf-8") as f:
        json.dump(blueprint, f, indent=2)

    # --- Prose ---
    prose_kwargs = {cfg["blueprint_kwarg"]: blueprint}
    prose = ai.generate_prose_story_ai(
        hero_name=hero_name, theme=scenario["theme"], logline=scenario["logline"],
        outline=scenario["outline"], personalization=personalization, genre=genre,
        additional_characters=support, page_count=page_count, **prose_kwargs,
    )
    if not prose:
        results.append(("FAIL", "prose: generation returned empty"))
        return {"results": results, "prose": None, "panels": None}

    target_words = 3000 if page_count > 24 else 2000
    prose = ai.expand_prose_if_short_ai(
        prose_story=prose, target_word_count=target_words, hero_name=hero_name,
        max_passes=3 if page_count > 24 else 2,
    )

    prose = ai.check_story_pacing_ai(prose_story=prose, hero_name=hero_name, genre=genre, page_count=page_count)
    prose = ai.check_story_continuity_ai(prose_story=prose, hero_name=hero_name, page_count=page_count)
    prose = ai.check_story_specificity_ai(prose_story=prose, hero_name=hero_name, page_count=page_count)
    prose = ai.check_story_arc_shape_ai(prose_story=prose, hero_name=hero_name, genre=genre, page_count=page_count)
    prose = cfg["fidelity_check"](prose_story=prose, blueprint=blueprint, hero_name=hero_name, page_count=page_count)

    if cfg["has_punchup"]:
        prose = ai.punch_up_prose_ai(prose_story=prose, hero_name=hero_name, page_count=page_count, additional_characters=support)

    with open(os.path.join(out_dir, "prose.txt"), "w", encoding="utf-8") as f:
        f.write(prose)

    results.append(c.check_word_count("prose", prose, min_words=1800, target_words=target_words))
    results.append(c.check_not_truncated("prose", prose))
    results.append(c.check_catchphrase_in_prose("prose", prose, catchphrase, hero_name))
    results.append(c.check_hazard_content("prose", prose))

    # --- NPC bible + panel adaptation ---
    npc_bible = ai.generate_story_character_bible_ai(prose_story=prose, hero_name=hero_name, support_char_names=support_names)
    character_brief = f"{hero_name}: described per personalization; medium build."
    panels = ai.generate_preview_pages_chunked_ai(
        hero_name=hero_name, theme=scenario["theme"], logline=scenario["logline"],
        outline=scenario["outline"], character_brief=character_brief, page_count=page_count,
        personalization=personalization, genre=genre, is_preview=False,
        additional_characters=support, prose_story=prose, npc_bible=npc_bible or None,
    )
    with open(os.path.join(out_dir, "panels.json"), "w", encoding="utf-8") as f:
        json.dump(panels, f, indent=2)

    results.append(c.check_panel_count("panels", panels, page_count))

    dialogue_lines = [
        d.get("line") or d.get("text") or ""
        for p in panels for d in (p.get("dialogue") or [])
    ]
    narration_lines = [p.get("narration") or "" for p in panels]
    all_names = [hero_name] + support_names + [n.get("name") for n in (npc_bible or []) if n.get("name")]
    results.append(c.check_line_repetition("panel dialogue", dialogue_lines, names=all_names))
    results.append(c.check_line_repetition("panel narration", narration_lines, names=all_names))
    results.append(c.check_catchphrase_in_panels("panels", panels, catchphrase, hero_name))
    results.append(c.check_hazard_content("panels", json.dumps(panels)))

    return {"results": results, "prose": prose, "panels": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genre", help="Run only this genre (default: all)")
    parser.add_argument("--page-count", type=int, default=48)
    args = parser.parse_args()

    genres = [args.genre] if args.genre else list(SCENARIOS.keys())
    overall_ok = True

    for genre in genres:
        if genre not in SCENARIOS:
            print(f"⚠️  No scenario for genre '{genre}' — skipping")
            continue
        scenario = SCENARIOS[genre]
        tag = "validated" if scenario.get("validated") else "UNVALIDATED SCENARIO"
        print(f"\n{'=' * 70}\n{genre}  ({tag})\n{'=' * 70}")
        try:
            outcome = run_scenario(genre, scenario, args.page_count)
        except Exception as e:
            print(f"❌ CRASHED: {e}")
            traceback.print_exc()
            overall_ok = False
            continue

        for status, message in outcome["results"]:
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "⏭️ "}[status]
            print(f"  {icon} [{status}] {message}")

        ok, _ = c.run_checks(outcome["results"])
        overall_ok = overall_ok and ok
        print(f"  --> {'PASS' if ok else 'FAIL'}")

    print(f"\n{'=' * 70}")
    print("OVERALL:", "PASS" if overall_ok else "FAIL — see FAILs above")
    print(f"Output saved under {OUTPUT_DIR}/<genre>/ for manual spot-checking")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
