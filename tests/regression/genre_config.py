"""Per-genre function wiring for the regression suite.

Mirrors the genre-conditional blocks in app/services/book_generation.py --
each genre gets a blueprint generator, a blueprint validator, a prose
fidelity checker, and the keyword argument name generate_prose_story_ai
expects that genre's blueprint under. Keeping this as one table instead of
scattering genre if/elif chains through the test runner is what makes adding
a new genre's scenario a one-line change instead of a new code path.
"""
from app.services import ai

GENRE_CONFIG = {
    "comedy": {
        "blueprint_gen": ai.generate_comedy_blueprint_ai,
        "blueprint_validate": ai.validate_comedy_blueprint_ai,
        "fidelity_check": ai.check_comedy_fidelity_ai,
        "blueprint_kwarg": "comedy_blueprint",
        "has_punchup": True,
    },
    "adventure": {
        "blueprint_gen": ai.generate_adventure_blueprint_ai,
        "blueprint_validate": ai.validate_adventure_blueprint_ai,
        "fidelity_check": ai.check_adventure_fidelity_ai,
        "blueprint_kwarg": "adventure_blueprint",
        "has_punchup": False,
    },
    "mystery": {
        "blueprint_gen": ai.generate_mystery_blueprint_ai,
        "blueprint_validate": ai.validate_mystery_blueprint_ai,
        "fidelity_check": ai.check_mystery_fidelity_ai,
        "blueprint_kwarg": "mystery_blueprint",
        "has_punchup": False,
    },
    "horror_adult": {
        "blueprint_gen": ai.generate_horror_blueprint_ai,
        "blueprint_validate": ai.validate_horror_blueprint_ai,
        "fidelity_check": ai.check_horror_fidelity_ai,
        "blueprint_kwarg": "horror_blueprint",
        "has_punchup": False,
    },
    "horror": {
        "blueprint_gen": ai.generate_horror_kids_blueprint_ai,
        "blueprint_validate": ai.validate_horror_kids_blueprint_ai,
        "fidelity_check": ai.check_horror_kids_fidelity_ai,
        "blueprint_kwarg": "horror_kids_blueprint",
        "has_punchup": False,
    },
    "sci_fi": {
        "blueprint_gen": ai.generate_sci_fi_blueprint_ai,
        "blueprint_validate": ai.validate_sci_fi_blueprint_ai,
        "fidelity_check": ai.check_sci_fi_fidelity_ai,
        "blueprint_kwarg": "sci_fi_blueprint",
        "has_punchup": False,
    },
    "fantasy": {
        "blueprint_gen": ai.generate_fantasy_blueprint_ai,
        "blueprint_validate": ai.validate_fantasy_blueprint_ai,
        "fidelity_check": ai.check_fantasy_fidelity_ai,
        "blueprint_kwarg": "fantasy_blueprint",
        "has_punchup": False,
    },
}
