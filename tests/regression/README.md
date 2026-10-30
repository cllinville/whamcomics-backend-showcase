# Story/panel regression suite

Catches the failure patterns found during live production testing (prose
truncation, repeated "going viral" style dialogue piling up, a catchphrase
said by the wrong character, hazardous-content mix-ups) without needing a
human to manually read every generated story again after every prompt
change.

## What it does

For each genre, runs the real generation pipeline from `app/services/ai.py`
on a fixed scenario (same hero, logline, and outline every time) — blueprint,
prose, expansion, structural checks, genre fidelity check, punch-up (comedy
only), NPC bible, and chunked panel-script adaptation. No images are
generated (no LlamaGen/S3 calls), so this costs an OpenAI bill but no image
credits and takes a few minutes per genre instead of 20+.

## Running it

```
python -m tests.regression.run                  # every genre
python -m tests.regression.run --genre comedy    # one genre
python -m tests.regression.run --genre comedy --page-count 24
```

Requires a local `.env` with a real OpenAI key, same as running the backend
itself.

## Reading the output

Each run prints a PASS/WARN/FAIL line per check and an overall verdict.
`FAIL` means something the automated checks are confident is broken.
`WARN` means "worth a human glance" — these are deliberately noisy (e.g. the
hazard-content check) because a false alarm costs a few seconds and missing
a real one costs a customer-facing incident.

Full prose and the panel script are saved to `output/<genre>/` for every
run, overwriting the previous run's files — read these directly when you
want to judge actual craft (is the escalation funny, does the mystery's
solution make sense), which the automated checks don't attempt to judge.

## What this does NOT catch

- Visual/image problems (an NPC's appearance drifting between panels, a
  panel's image not matching its own narration, speech bubbles covering a
  character's face) — those are rendering-pipeline bugs, not something
  visible in text output. Still needs a human reading the actual rendered
  panels.
- Genuine craft quality (is this actually funny / scary / mysterious) —
  the checks here catch mechanical failure patterns, not taste.
- Anything in a scenario marked `"validated": False` in `scenarios.py` — no
  one has confirmed that scenario's content is actually good, only that
  the pipeline runs on it without crashing. Mystery and adventure currently
  need this pass before they should be treated as covered.

## Extending it

Add a new scenario (a different premise for a genre already in
`genre_config.py`) or a new genre (add both a `GENRE_CONFIG` entry and a
`SCENARIOS` entry) — see the existing entries in `scenarios.py` for the
shape. Add a new automated check as a function in `checks.py` returning
`(status, message)` and call it from `run.py`.
