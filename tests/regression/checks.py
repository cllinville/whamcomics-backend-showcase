"""Automated, cheap checks for story/panel-script output.

These replace the manual "read 3000 words and judge it" step for the things
that actually ARE mechanical to detect -- truncation, repeated lines,
missing/misattributed catchphrases, hazardous-content tripwires. They do NOT
replace a human read for genuine craft judgment (is the escalation actually
funny, does the resolution feel earned) -- that's still a manual spot-check,
but these catch the specific failure patterns found in production this
session without needing a human to re-read the whole thing every time.

Each check returns (status, message) where status is one of
"PASS" / "WARN" / "FAIL" / "SKIP".
"""
import difflib
import re

Result = tuple[str, str]

_QUOTE_NORMALIZE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalize_quotes(text: str) -> str:
    """The model consistently writes curly apostrophes/quotes; scenario
    config is plain ASCII. Without this, every catchphrase substring check
    false-FAILs even when the line is right there in the text."""
    return (text or "").translate(_QUOTE_NORMALIZE)


def check_word_count(label: str, text: str, min_words: int, target_words: int = None) -> Result:
    words = len(text.split())
    if words < min_words:
        return ("FAIL", f"{label}: {words} words — below the {min_words}-word floor (likely truncated or over-cut)")
    if target_words and words < target_words:
        return ("WARN", f"{label}: {words} words — below the {target_words}-word target, worth a look")
    return ("PASS", f"{label}: {words} words")


def check_not_truncated(label: str, text: str) -> Result:
    stripped = (text or "").strip()
    if not stripped:
        return ("FAIL", f"{label}: empty")
    if stripped[-1] not in ".!?\"'”’“":
        return ("WARN", f"{label}: doesn't end on sentence-ending punctuation — possible mid-sentence truncation: ...{stripped[-70:]!r}")
    return ("PASS", f"{label}: ends cleanly")


def check_catchphrase_in_prose(label: str, prose: str, catchphrase: str, hero_name: str) -> Result:
    if not catchphrase:
        return ("SKIP", f"{label}: no catchphrase configured for this scenario")
    normalized_prose = _normalize_quotes(prose)
    normalized_catchphrase = _normalize_quotes(catchphrase)
    idx = normalized_prose.lower().find(normalized_catchphrase.lower())
    if idx == -1:
        return ("FAIL", f"{label}: catchphrase {catchphrase!r} never appears in the prose")
    # Cheap attribution heuristic: look at the ~120 chars before the quote for the hero's name
    # appearing closer than any OTHER capitalized name-like token immediately before the line.
    window = normalized_prose[max(0, idx - 120):idx]
    if hero_name.lower() not in window.lower():
        return ("WARN", f"{label}: catchphrase present but {hero_name} isn't named in the preceding text — check attribution manually: ...{window[-80:]!r}")
    return ("PASS", f"{label}: catchphrase present near {hero_name}'s name")


def check_catchphrase_in_panels(label: str, panels: list, catchphrase: str, hero_name: str) -> Result:
    if not catchphrase:
        return ("SKIP", f"{label}: no catchphrase configured for this scenario")
    normalized_catchphrase = _normalize_quotes(catchphrase).lower()
    hits = []
    for p in panels:
        for d in (p.get("dialogue") or []):
            line = _normalize_quotes(d.get("line") or d.get("text") or "")
            if normalized_catchphrase in line.lower():
                hits.append((p.get("page_number"), d.get("speaker", "")))
    if not hits:
        return ("FAIL", f"{label}: catchphrase {catchphrase!r} never appears in any panel's dialogue")
    wrong = [(pn, spk) for pn, spk in hits if spk.strip().lower() != hero_name.strip().lower()]
    if wrong:
        return ("FAIL", f"{label}: catchphrase said by the wrong character in panel(s) {wrong} — should only be {hero_name}")
    return ("PASS", f"{label}: catchphrase said correctly by {hero_name} ({len(hits)}x, panels {[h[0] for h in hits]})")


def _strip_names(text: str, names: list) -> str:
    if not names:
        return text
    pattern = "|".join(re.escape(n) for n in names if n)
    if not pattern:
        return text
    return re.sub(r"\s+", " ", re.sub(pattern, "", text, flags=re.I)).strip()


def _similarity(a: str, b: str, names: list = None) -> float:
    a, b = _strip_names(a, names or []), _strip_names(b, names or [])
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_line_repetition(label: str, lines: list, names: list = None, threshold: float = 0.68, min_len: int = 12, min_cluster: int = 3) -> Result:
    """Flags clusters of 3+ near-identical lines -- the "going viral 5 different
    ways" pattern found in production. A single echoed callback (2 instances)
    is normal craft, not a cluster; min_cluster=3 avoids flagging that.

    `names` (hero + support/NPC names) are stripped before comparing, and the
    threshold is raised from an earlier 0.6. Found empirically: two short
    lines sharing only a character's name and similar length/punctuation
    ("Ceiara, stay strong!" vs "Ceiara, I feel it too!" -- different
    sentiments entirely) scored 0.60-0.71 on raw similarity, overlapping the
    0.63-0.88 range of genuine repeated-declaration padding ("This is going
    to be epic!" vs "...legendary!"). Stripping names separates them: false
    positives drop to 0.43-0.65, genuine padding stays 0.63-0.88, and 0.68
    sits in the gap between."""
    candidates = [l for l in lines if l and len(l) >= min_len]
    used = set()
    clusters = []
    for i in range(len(candidates)):
        if i in used:
            continue
        group = [candidates[i]]
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            if _similarity(candidates[i], candidates[j], names) >= threshold:
                group.append(candidates[j])
                used.add(j)
        if len(group) >= min_cluster:
            clusters.append(group)
            used.add(i)
    if clusters:
        detail = "; ".join(f"{len(g)}x similar, e.g. {g[0]!r}" for g in clusters)
        return ("FAIL", f"{label}: repeated-sentiment cluster(s) — {detail}")
    return ("PASS", f"{label}: no repeated-sentiment clusters found")


_HAZARD_WORDS = ["clean", "bleach", "detergent", "chemical", "toxic", "poison", "pesticide", "antifreeze", "solvent", "disinfect"]
_CONSUMPTION_WORDS = ["ate", "eat", "eats", "eating", "bit", "bite", "drank", "drink", "drinks", "sip", "swallow", "tast", "serv", "spoonful", "bite"]


def check_hazard_content(label: str, text: str) -> Result:
    """Tripwire for the cleaning-solution-served-to-a-customer failure found
    this session. Heuristic and noisy by design (a false positive here just
    means a human glances at one sentence) -- missing a real one is the
    expensive failure mode, a false alarm is cheap."""
    lower = (text or "").lower()
    hits = []
    for hw in _HAZARD_WORDS:
        for m in re.finditer(re.escape(hw), lower):
            window = lower[max(0, m.start() - 150):m.end() + 150]
            if any(cw in window for cw in _CONSUMPTION_WORDS):
                hits.append(window.strip())
    if hits:
        return ("WARN", f"{label}: possible hazardous-substance-consumed content — check: ...{hits[0][:150]}...")
    return ("PASS", f"{label}: no hazard/consumption overlap detected")


def check_panel_count(label: str, panels: list, expected: int) -> Result:
    if len(panels) != expected:
        return ("FAIL", f"{label}: expected {expected} panels, got {len(panels)}")
    nums = sorted(p.get("page_number") for p in panels if p.get("page_number") is not None)
    if nums != list(range(1, expected + 1)):
        return ("FAIL", f"{label}: panel numbers aren't a clean 1..{expected} sequence: {nums}")
    return ("PASS", f"{label}: {expected} panels, correctly numbered")


def run_checks(checks: list) -> tuple[bool, list]:
    """Takes a list of (status, message) results, returns (ok, results) where
    ok is False if any check FAILed (WARN/SKIP don't block)."""
    ok = all(status != "FAIL" for status, _ in checks)
    return ok, checks
