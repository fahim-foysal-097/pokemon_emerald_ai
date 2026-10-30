"""Single source of truth for vanilla Pokémon Emerald curriculum objectives.

Map IDs are the project's observed BPEE map-group/map-number encoding
(map_group << 8 | map_number). Story flags are from the vanilla pokeemerald
source constants and are verified against the expected BPEE revision 0 ROM.
"""
from __future__ import annotations

import math
from numbers import Real

CURRICULUM_PLAN_ID = "emerald_intro_v4"

# BPEE map IDs observed/decoded by this project.
ROUTE_101_MAP_ID = 16
OLDALE_TOWN_MAP_ID = 10
ROUTE_103_MAP_ID = 18
BIRCH_LAB_MAP_ID = 260

# Vanilla Emerald story/system flags.
CLOCK_SET_FLAG_ID = 0x51
BIRCH_RESCUED_FLAG_ID = 0x52
RIVAL_DEFEATED_ROUTE103_FLAG_ID = 0x82
STARTER_OBTAINED_FLAG_ID = 0x860  # FLAG_SYS_POKEMON_GET
POKEDEX_OBTAINED_FLAG_ID = 0x861  # FLAG_SYS_POKEDEX_GET

# Order matters: each stage starts from the immediately preceding verified milestone.
STAGES = (
    "starter",
    "leave_birch_lab",
    "reach_oldale",
    "reach_route103",
    "defeat_rival",
    "return_birch_lab",
    "pokedex",
    "first_badge",
)

OBJECTIVE_DEFINITIONS = {
    "starter": {
        "label": "rescue Birch, win the wild battle and reach the lab",
        "description": "Obtain the starter, finish the opening wild battle, and reach Birch's lab.",
    },
    "leave_birch_lab": {
        "label": "leave Birch's Lab via Route 101",
        "description": "Complete Birch's lab introduction and exit onto Route 101.",
    },
    "reach_oldale": {
        "label": "reach Oldale Town",
        "description": "Leave Littleroot and reach Oldale Town with the starter in the party.",
    },
    "reach_route103": {
        "label": "reach Route 103",
        "description": "Navigate from Oldale Town to Route 103, where the rival waits.",
    },
    "defeat_rival": {
        "label": "defeat May on Route 103",
        "description": "Win the first Route 103 rival battle (vanilla story flag 0x82).",
    },
    "return_birch_lab": {
        "label": "return to Birch's Lab",
        "description": "After the Route 103 rival battle, reach Professor Birch's lab again.",
    },
    "pokedex": {
        "label": "obtain the Pokédex",
        "description": "Obtain the Pokédex after defeating the Route 103 rival and returning to Birch.",
    },
    "first_badge": {
        "label": "first Gym Badge",
        "description": "Obtain at least one Gym Badge after the Pokédex milestone.",
    },
}
OBJECTIVE_LABELS = {name: definition["label"] for name, definition in OBJECTIVE_DEFINITIONS.items()}


def _number(value):
    """Accept only finite real scalars; do not silently accept numeric strings."""
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _state_metrics(state):
    if not isinstance(state, dict):
        return None
    party_count = _number(state.get("party_count", 0))
    party_levels = _number(state.get("party_levels", 0))
    map_id = _number(state.get("map_id", -1))
    if (
        party_count is None or not party_count.is_integer() or not 0 <= party_count <= 6
        or party_levels is None or not party_levels.is_integer() or not 0 <= party_levels <= 600
        or map_id is None or not map_id.is_integer() or map_id < 0
    ):
        return None
    flags = state.get("story_flags")
    if not isinstance(flags, (set, frozenset, list, tuple, range)):
        return None
    try:
        story_flags = {int(flag) for flag in flags}
    except (TypeError, ValueError, OverflowError):
        return None
    badge_count = _number(state.get("badge_count"))
    if badge_count is None:
        badge_flags = _number(state.get("badge_flags", 0))
        if badge_flags is None or not badge_flags.is_integer() or badge_flags < 0:
            return None
        badge_count = float(int(badge_flags).bit_count())
    if not badge_count.is_integer() or not 0 <= badge_count <= 8:
        return None
    return {
        "party_count": int(party_count),
        "party_levels": int(party_levels),
        "map_id": int(map_id),
        "story_flags": story_flags,
        "badge_count": int(badge_count),
    }


def objective_satisfied(objective, state):
    """Evaluate a curriculum milestone from decoded live game state only."""
    objective = str(objective or "").strip().lower()
    if objective not in OBJECTIVE_DEFINITIONS:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of " + ", ".join(OBJECTIVE_DEFINITIONS)
        )
    metrics = _state_metrics(state)
    if metrics is None:
        return False
    party = metrics["party_count"] >= 1 and metrics["party_levels"] >= 1
    map_id = metrics["map_id"]
    flags = metrics["story_flags"]
    if objective == "starter":
        # The starter enters the live party on Route 101 before the mandatory
        # opening battle and lab return are complete. Do not terminate on party_count alone.
        return party and map_id == BIRCH_LAB_MAP_ID
    if objective == "leave_birch_lab":
        return party and map_id == ROUTE_101_MAP_ID
    if objective == "reach_oldale":
        return party and map_id == OLDALE_TOWN_MAP_ID
    if objective == "reach_route103":
        return party and map_id == ROUTE_103_MAP_ID
    if objective == "defeat_rival":
        return party and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags
    if objective == "return_birch_lab":
        return party and map_id == BIRCH_LAB_MAP_ID and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags
    if objective == "pokedex":
        return (
            party
            and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags
            and POKEDEX_OBTAINED_FLAG_ID in flags
        )
    if objective == "first_badge":
        return metrics["badge_count"] >= 1
    return False


def objective_start_error(objective, state, *, route_101_map_id=ROUTE_101_MAP_ID):
    """Return a prerequisite error, or None when the state is valid for a stage.

    Do not reject a state merely because the objective is already satisfied: a
    scripted game transition can finish multiple adjacent milestones automatically.
    The normal success detector will mark that stage complete on its first step.
    """
    objective = str(objective or "").strip().lower()
    metrics = _state_metrics(state)
    if objective not in OBJECTIVE_DEFINITIONS:
        return "unknown objective; choose one of: " + ", ".join(OBJECTIVE_DEFINITIONS)
    if metrics is None:
        return "game state is missing valid party, map, badge, or story-flag telemetry"
    party = metrics["party_count"] >= 1 and metrics["party_levels"] >= 1
    map_id = metrics["map_id"]
    flags = metrics["story_flags"]

    if objective == "starter":
        if map_id != route_101_map_id:
            return f"the starter task must start on Route 101 (map_id={route_101_map_id}); the current map_id is {map_id}"
        if CLOCK_SET_FLAG_ID not in flags:
            return "the bedroom clock-set flag is absent; save after setting the clock"
        if BIRCH_RESCUED_FLAG_ID in flags:
            return "the starter task must start before rescuing Birch and before receiving a starter"
        if metrics["party_count"] != 0 or metrics["party_levels"] != 0:
            return "the starter task must start before receiving a starter, with an empty party and no party levels"
        return None

    if objective == "leave_birch_lab":
        if not party:
            return "the leave-the-lab stage requires a valid starter in the live party"
        if map_id not in {BIRCH_LAB_MAP_ID, ROUTE_101_MAP_ID}:
            return f"the leave-the-lab stage must start in Birch's Lab (or already be on Route 101); current map_id={map_id}"
        return None
    if objective == "reach_oldale":
        if not party:
            return "the Oldale stage requires a valid starter in the live party"
        if map_id not in {ROUTE_101_MAP_ID, OLDALE_TOWN_MAP_ID}:
            return f"the Oldale stage must start on Route 101 (or already be in Oldale); current map_id={map_id}"
        return None
    if objective == "reach_route103":
        if not party:
            return "the Route 103 stage requires a valid starter in the live party"
        if map_id not in {OLDALE_TOWN_MAP_ID, ROUTE_103_MAP_ID}:
            return f"the Route 103 stage must start in Oldale Town (or already be on Route 103); current map_id={map_id}"
        return None
    if objective == "defeat_rival":
        if not party:
            return "the Route 103 rival battle stage requires a valid starter in the live party"
        if RIVAL_DEFEATED_ROUTE103_FLAG_ID not in flags and map_id != ROUTE_103_MAP_ID:
            return f"the rival-battle stage must start on Route 103; current map_id={map_id} and the rival flag is unset"
        return None
    if objective == "return_birch_lab":
        if not party:
            return "the return-to-lab stage requires a valid starter in the live party"
        if RIVAL_DEFEATED_ROUTE103_FLAG_ID not in flags:
            return "the return-to-lab stage requires the Route 103 rival-defeated flag (0x82)"
        if map_id not in {ROUTE_103_MAP_ID, BIRCH_LAB_MAP_ID}:
            return f"the return-to-lab stage must start on Route 103 or in Birch's Lab; current map_id={map_id}"
        return None
    if objective == "pokedex":
        if not party:
            return "the Pokédex stage requires a valid starter in the live party"
        if RIVAL_DEFEATED_ROUTE103_FLAG_ID not in flags:
            return "the Pokédex stage requires the verified Route 103 rival-defeated milestone (flag 0x82)"
        if POKEDEX_OBTAINED_FLAG_ID not in flags and map_id != BIRCH_LAB_MAP_ID:
            return f"the Pokédex stage must start in Birch's Lab; current map_id={map_id}"
        return None
    if objective == "first_badge":
        if not party:
            return "the first-badge stage requires a valid Pokémon in the live party"
        if POKEDEX_OBTAINED_FLAG_ID not in flags:
            return "the first-badge stage requires a verified Pokédex milestone start state"
        return None
    return "no start-state validation is defined for this objective"
