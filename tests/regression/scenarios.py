"""Fixed test scenarios for the regression suite, one (or more) per genre.

Each scenario is a self-contained set of inputs to the real generation
pipeline (app/services/ai.py) -- the same hero/logline/outline every run, so
a change in output between runs reflects a change in the prompts, not a
different random scenario. `validated` marks whether a human has actually
read this scenario's output and confirmed it's good craft, as opposed to
just "ran without crashing" -- a scenario with validated=False still runs
the automated checks, but its checks haven't been confirmed to catch the
genre's real failure modes yet.

Add a new genre by adding a new dict here and in genre_config.GENRE_CONFIG.
"""

SCENARIOS = {
    "comedy": {
        "validated": True,  # Run repeatedly this session; current prompts confirmed clean on this scenario.
        "hero_name": "Vincent",
        "support": [{"name": "Ceiara", "relationship": "wife"}],
        "personalization": {
            "age": 39,
            "character_gender": "boy",
            "favorite_activities": ["watching TikTok videos", "eating", "napping"],
            "personality_traits": ["funny", "judgemental"],
            "favorite_place": "the bathroom",
            "dream": "to be a millionaire",
            "special_skill": "driving a semi truck, cooking",
            "catchphrase": "Let's not and say we did",
        },
        "theme": "won the lottery",
        "logline": (
            "After Vincent wins the lottery, he opens a food truck with a 'Mystery Sauce Roulette' "
            "gimmick that spirals into way more chaos than he bargained for."
        ),
        "outline": (
            "Vincent and Ceiara convert a semi-truck into a food truck and launch it with a hot-sauce "
            "roulette gimmick to draw a crowd. The sauce is far hotter than anyone expects, and Vincent's "
            "overeager cook Marcus keeps escalating the chaos. What starts as a rocky grand opening spirals "
            "through bigger and bigger disasters before Vincent manages to salvage the day."
        ),
    },
    "sci_fi": {
        "validated": True,  # Confirmed good structure/escalation after the blueprint-wiring fix this session.
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 28, "character_gender": "female"},
        "theme": "something in the walls",
        "logline": (
            "Ceiara finds a cryptic message embedded in the walls of their new home that hints at a "
            "long-lost family heirloom connected to advanced technology."
        ),
        "outline": (
            "Ceiara finds a cryptic message in the walls that hints at a long-lost family heirloom "
            "connected to advanced technology, leading her on a nostalgic journey into her family's "
            "history. Midway through their investigation, they are ambushed by rogue robots programmed "
            "to protect the secrets, threatening to expose their deepest vulnerabilities. With clever "
            "thinking and Vincent's resourcefulness, Ceiara uncovers the truth and reclaims the heirloom, "
            "strengthening their bond and addressing the importance of past legacies."
        ),
    },
    "horror_adult": {
        "validated": True,
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 30, "character_gender": "female"},
        "theme": "something in the walls",
        "logline": (
            "When Ceiara inherits her grandmother's isolated cabin, the whispers in the walls reveal a "
            "decades-old pact that never truly ended."
        ),
        "outline": (
            "Ceiara and Vincent move into the inherited cabin to renovate it, but strange occurrences -- "
            "cold spots, muttering voices, a locked cellar -- escalate into terrifying encounters with a "
            "vengeful presence tied to a betrayal from her grandmother's past. As the entity's attacks grow "
            "violent, Ceiara must uncover the truth of what her grandmother did decades ago and find a way "
            "to break the pact before it claims her and Vincent."
        ),
    },
    "horror": {
        "validated": True,
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 10, "character_gender": "female"},
        "theme": "a friendly ghost",
        "logline": (
            "Ceiara's new tree house comes with a friendly-looking ghost who insists something spooky is "
            "coming for the whole neighborhood."
        ),
        "outline": (
            "Ceiara discovers her new backyard tree house is haunted by a spooky-looking but ultimately "
            "kind spirit who warns her about strange lights in the old oak tree at the edge of the "
            "neighborhood. Together with Vincent, she investigates spooky clues around the neighborhood, "
            "facing mild scares and mysterious happenings, before discovering the 'threat' is actually a "
            "lonely, misunderstood creature that just wants a friend."
        ),
    },
    "fantasy": {
        "validated": True,
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 28, "character_gender": "female"},
        "theme": "a family ring",
        "logline": (
            "On her birthday, Ceiara discovers the family ring she inherited is the key to a hidden realm "
            "collapsing under a curse only she can lift."
        ),
        "outline": (
            "Ceiara is pulled into a magical hidden realm connected to her family's ring, where an ancient "
            "curse is unraveling the world's magic. Guided by Vincent, an exiled guardian of the realm, she "
            "must master her newly awakened powers, navigate rivalries among magical factions, and confront "
            "the truth about who cast the original curse -- and why it targeted her bloodline -- before the "
            "realm and her own world are lost forever."
        ),
    },
    "mystery": {
        "validated": False,  # First draft -- not yet confirmed against mystery's real failure modes.
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 32, "character_gender": "female"},
        "theme": "the neighbor who wasn't home",
        "logline": (
            "When the Hendersons' prize-winning garden is destroyed the same night their reclusive "
            "neighbor claims to have been out of town, Ceiara suspects the alibi is a lie."
        ),
        "outline": (
            "Ceiara notices inconsistencies in her neighbor Mr. Doyle's story about being away the night "
            "the Hendersons' garden was trashed, and starts quietly investigating with Vincent. Several "
            "neighbors each have their own motive and a shaky alibi, and the evidence keeps pointing in "
            "contradictory directions. Ceiara eventually pieces together the real timeline from a detail "
            "everyone else overlooked, confronts the actual culprit, and closes out every suspect's real "
            "story -- including why Mr. Doyle lied about something unrelated to the crime."
        ),
    },
    "adventure": {
        "validated": False,  # First draft -- not yet confirmed against adventure's real failure modes.
        "hero_name": "Ceiara",
        "support": [{"name": "Vincent", "relationship": "husband"}],
        "personalization": {"age": 29, "character_gender": "female"},
        "theme": "a treasure hunt through a jungle temple",
        "logline": (
            "Ceiara inherits her late uncle's research pointing to a lost temple deep in the jungle, and "
            "sets out with Vincent to find it before a rival expedition gets there first."
        ),
        "outline": (
            "Ceiara and Vincent follow her uncle's notes into the jungle, facing natural obstacles -- a "
            "collapsed bridge, a flooded cave system, a trap-laced temple entrance -- while a rival "
            "treasure hunter shadows their progress. At the temple's heart, Ceiara must solve a puzzle "
            "her uncle never cracked, make a real sacrifice to claim the treasure safely, and get out "
            "before the temple's defenses seal the only way out."
        ),
    },
}
