"""Single source of truth for vanilla Pokémon Emerald curriculum objectives.

Map IDs are the project's observed BPEE map-group/map-number encoding
(map_group << 8 | map_number). Story flags are from the vanilla pokeemerald
source constants and are verified against the expected BPEE revision 0 ROM.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

CURRICULUM_PLAN_ID = "emerald_intro_v6"

# BPEE Map Group 0 IDs (Towns, Cities, and Routes in vanilla Emerald).
PETALBURG_CITY_MAP_ID = 0
SLATEPORT_CITY_MAP_ID = 1
MAUVILLE_CITY_MAP_ID = 2
RUSTBORO_CITY_MAP_ID = 3
FORTREE_CITY_MAP_ID = 4
LILYCOVE_CITY_MAP_ID = 5
MOSSDEEP_CITY_MAP_ID = 6
SOOTOPOLIS_CITY_MAP_ID = 7
EVER_GRANDE_CITY_MAP_ID = 8
LITTLEROOT_TOWN_MAP_ID = 9
OLDALE_TOWN_MAP_ID = 10
DEWFORD_TOWN_MAP_ID = 11
LAVARIDGE_TOWN_MAP_ID = 12
FALLARBOR_TOWN_MAP_ID = 13
VERDANTURF_TOWN_MAP_ID = 14
PACIFIDLOG_TOWN_MAP_ID = 15

ROUTE_101_MAP_ID = 16
ROUTE_102_MAP_ID = 17
ROUTE_103_MAP_ID = 18
ROUTE_104_MAP_ID = 19
ROUTE_105_MAP_ID = 20
ROUTE_106_MAP_ID = 21
ROUTE_107_MAP_ID = 22
ROUTE_108_MAP_ID = 23
ROUTE_109_MAP_ID = 24
ROUTE_110_MAP_ID = 25
ROUTE_111_MAP_ID = 26
ROUTE_112_MAP_ID = 27
ROUTE_113_MAP_ID = 28
ROUTE_114_MAP_ID = 29
ROUTE_115_MAP_ID = 30
ROUTE_116_MAP_ID = 31
ROUTE_117_MAP_ID = 32
ROUTE_118_MAP_ID = 33
ROUTE_119_MAP_ID = 34
ROUTE_120_MAP_ID = 35
ROUTE_121_MAP_ID = 36
ROUTE_128_MAP_ID = 43

# Interior & Special Maps
BIRCH_LAB_MAP_ID = 260

# Vanilla Emerald Story & System Flags from pokeemerald constants/flags.h
CLOCK_SET_FLAG_ID = 0x51
BIRCH_RESCUED_FLAG_ID = 0x52
RIVAL_DEFEATED_ROUTE103_FLAG_ID = 0x82

# System Flags
STARTER_OBTAINED_FLAG_ID = 0x860   # FLAG_SYS_POKEMON_GET
POKEDEX_OBTAINED_FLAG_ID = 0x861   # FLAG_SYS_POKEDEX_GET
POKENAV_OBTAINED_FLAG_ID = 0x862   # FLAG_SYS_POKENAV_GET
FLAG_SYS_GAME_CLEAR = 0x864        # Game clear / Hall of Fame Champion defeated

# Badges (0x867 .. 0x86E)
FLAG_BADGE01_GET = 0x867  # Stone Badge (Roxanne)
FLAG_BADGE02_GET = 0x868  # Knuckle Badge (Brawly)
FLAG_BADGE03_GET = 0x869  # Dynamo Badge (Wattson)
FLAG_BADGE04_GET = 0x86A  # Heat Badge (Flannery)
FLAG_BADGE05_GET = 0x86B  # Balance Badge (Norman)
FLAG_BADGE06_GET = 0x86C  # Feather Badge (Winona)
FLAG_BADGE07_GET = 0x86D  # Mind Badge (Tate & Liza)
FLAG_BADGE08_GET = 0x86E  # Rain Badge (Juan)

# Visited Town Flags (0x86F .. 0x87E)
FLAG_VISITED_LITTLEROOT_TOWN = 0x86F
FLAG_VISITED_OLDALE_TOWN     = 0x870
FLAG_VISITED_DEWFORD_TOWN    = 0x871
FLAG_VISITED_LAVARIDGE_TOWN  = 0x872
FLAG_VISITED_FALLARBOR_TOWN  = 0x873
FLAG_VISITED_VERDANTURF_TOWN = 0x874
FLAG_VISITED_PACIFIDLOG_TOWN = 0x875
FLAG_VISITED_PETALBURG_CITY  = 0x876
FLAG_VISITED_SLATEPORT_CITY  = 0x877
FLAG_VISITED_MAUVILLE_CITY   = 0x878
FLAG_VISITED_RUSTBORO_CITY   = 0x879
FLAG_VISITED_FORTREE_CITY    = 0x87A
FLAG_VISITED_LILYCOVE_CITY   = 0x87B
FLAG_VISITED_MOSSDEEP_CITY   = 0x87C
FLAG_VISITED_SOOTOPOLIS_CITY = 0x87D
FLAG_VISITED_EVER_GRANDE_CITY= 0x87E

# Key Story Progression Flags. Values and names are taken from the vanilla
# BPEE definitions in pret/pokeemerald/include/constants/flags.h. Do not reuse
# nearby IDs: several old values were valid flags for unrelated side events.
FLAG_HIDE_SKY_PILLAR_TOP_RAYQUAZA_STILL = 0x50  # Static/still Rayquaza object spawn flag; not story completion.
FLAG_HIDE_SKY_PILLAR_TOP_RAYQUAZA = 0x305  # Rayquaza departs Sky Pillar for Sootopolis.
FLAG_WALLY_TUTORIAL_COMPLETED = 0xC1       # FLAG_WALLY_SPEECH
FLAG_RESCUED_DEVON_RESEARCHER = 0x2D4      # FLAG_HIDE_PETALBURG_WOODS_DEVON_EMPLOYEE
FLAG_RESCUED_DEVON_RESEARCHER_AQUA_GRUNT = 0x2D5  # FLAG_HIDE_PETALBURG_WOODS_AQUA_GRUNT
FLAG_RUSTURF_TUNNEL_PEEKO_RESCUED = 0x370 # FLAG_HIDE_RUSTURF_TUNNEL_PEEKO
FLAG_STEVEN_LETTER_DELIVERED = 0xBD      # FLAG_DELIVERED_STEVEN_LETTER
FLAG_POKENAV_RECEIVED = 0xBC             # FLAG_RECEIVED_POKENAV
FLAG_ADVENTURE_STARTED = 0x74            # FLAG_ADVENTURE_STARTED (Pokédex received)
FLAG_OCEANIC_MUSEUM_GRUNTS_HIDDEN = 0x373
FLAG_OCEANIC_MUSEUM_GRUNT1_HIDDEN = 0x374
FLAG_OCEANIC_MUSEUM_GRUNT2_HIDDEN = 0x375
FLAG_OCEANIC_MUSEUM_ARCHIE_HIDDEN = 0x376
FLAG_OCEANIC_MUSEUM_STERN_HIDDEN = 0x377
VAR_ROUTE110_STATE_ID = 0x4069  # VAR_ROUTE110_STATE; set to 1 after the Route 110 rival battle
FLAG_RUSTURF_TUNNEL_OPENED = 0xC7        # FLAG_RUSTURF_TUNNEL_OPENED
FLAG_METEOR_FALLS_TEAM_SCENE = 0xCF      # FLAG_MET_ARCHIE_METEOR_FALLS
FLAG_MT_CHIMNEY_SUMMIT_DEFEATED = 0x8B   # FLAG_DEFEATED_EVIL_TEAM_MT_CHIMNEY
FLAG_RECEIVED_CASTFORM = 0x97            # FLAG_RECEIVED_CASTFORM
FLAG_RIVAL_DEFEATED_ROUTE119 = 0x353     # FLAG_HIDE_ROUTE_119_RIVAL
FLAG_RECEIVED_DEVON_SCOPE = 0x11D        # FLAG_RECEIVED_DEVON_SCOPE
FLAG_MT_PYRE_MAXIE_HIDDEN = 0x358        # FLAG_HIDE_MT_PYRE_SUMMIT_MAXIE
FLAG_RECEIVED_RED_OR_BLUE_ORB = 0xD4     # FLAG_RECEIVED_RED_OR_BLUE_ORB
FLAG_AWAKENED_GROUDON = 0x6F             # FLAG_GROUDON_AWAKENED_MAGMA_HIDEOUT
FLAG_AQUA_SUBMARINE_ESCAPED = 0x70       # FLAG_TEAM_AQUA_ESCAPED_IN_SUBMARINE
FLAG_SPACE_CENTER_DEFEATED = 0x75        # FLAG_DEFEATED_MAGMA_SPACE_CENTER
FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN = 0x81  # FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN
FLAG_AWAKENED_RAYQUAZA = FLAG_HIDE_SKY_PILLAR_TOP_RAYQUAZA
FLAG_RAYQUAZA_CALMED_WEATHER = 0x9E      # FLAG_SOOTOPOLIS_ARCHIE_MAXIE_LEAVE
FLAG_DEFEATED_WALLY_VICTORY_ROAD = 0x7E  # FLAG_DEFEATED_WALLY_VICTORY_ROAD

# Every persistent flag consulted by the curriculum must be read from RAM.
# This set is deliberately separate from the policy observation features so
# loading older PPO checkpoints keeps exactly the same observation dimension.
CURRICULUM_STORY_FLAG_IDS = tuple(sorted({
    STARTER_OBTAINED_FLAG_ID, POKEDEX_OBTAINED_FLAG_ID, POKENAV_OBTAINED_FLAG_ID,
    FLAG_SYS_GAME_CLEAR, CLOCK_SET_FLAG_ID, BIRCH_RESCUED_FLAG_ID,
    RIVAL_DEFEATED_ROUTE103_FLAG_ID,
    FLAG_WALLY_TUTORIAL_COMPLETED,
    FLAG_RESCUED_DEVON_RESEARCHER, FLAG_RESCUED_DEVON_RESEARCHER_AQUA_GRUNT,
    FLAG_RUSTURF_TUNNEL_PEEKO_RESCUED, FLAG_STEVEN_LETTER_DELIVERED,
    FLAG_POKENAV_RECEIVED, FLAG_ADVENTURE_STARTED,
    FLAG_OCEANIC_MUSEUM_GRUNTS_HIDDEN, FLAG_OCEANIC_MUSEUM_GRUNT1_HIDDEN,
    FLAG_OCEANIC_MUSEUM_GRUNT2_HIDDEN, FLAG_OCEANIC_MUSEUM_ARCHIE_HIDDEN,
    FLAG_OCEANIC_MUSEUM_STERN_HIDDEN,
    FLAG_RUSTURF_TUNNEL_OPENED,
    FLAG_METEOR_FALLS_TEAM_SCENE, FLAG_MT_CHIMNEY_SUMMIT_DEFEATED,
    FLAG_RECEIVED_CASTFORM, FLAG_RIVAL_DEFEATED_ROUTE119, FLAG_RECEIVED_DEVON_SCOPE,
    FLAG_MT_PYRE_MAXIE_HIDDEN, FLAG_RECEIVED_RED_OR_BLUE_ORB,
    FLAG_AWAKENED_GROUDON, FLAG_AQUA_SUBMARINE_ESCAPED, FLAG_SPACE_CENTER_DEFEATED,
    FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN, FLAG_AWAKENED_RAYQUAZA,
    FLAG_RAYQUAZA_CALMED_WEATHER, FLAG_DEFEATED_WALLY_VICTORY_ROAD,
    *range(FLAG_VISITED_LITTLEROOT_TOWN, FLAG_VISITED_EVER_GRANDE_CITY + 1),
}))

# Sequential Stages from Start to Game Clear
STAGES = (
    # Act I: Littleroot to Rustboro (Gym 1: Roxanne)
    "starter",
    "leave_birch_lab",
    "reach_route101",
    "reach_oldale",
    "reach_route103",
    "defeat_rival",
    "return_birch_lab",
    "pokedex",
    "reach_route102",
    "reach_petalburg",
    "wally_tutorial",
    "reach_route104_south",
    "defeat_aqua_woods",
    "reach_rustboro",
    "first_badge",

    # Act II: Dewford Island & Slateport (Gym 2: Brawly)
    "reach_route116",
    "rusturf_tunnel_rescue",
    "receive_pokenav",
    "sail_to_dewford",
    "second_badge",
    "deliver_letter_steven",
    "reach_slateport",
    "oceanic_museum",

    # Act III: Mauville & Cycling Road (Gym 3: Wattson)
    "reach_route110",
    "defeat_rival_route110",
    "reach_mauville",
    "third_badge",
    "smash_rusturf_rock",

    # Act IV: Mt. Chimney, Lavaridge & Meteor Falls (Gym 4: Flannery)
    "reach_route111_north",
    "reach_fallarbor",
    "meteor_falls",
    "mt_chimney_summit",
    "reach_lavaridge",
    "fourth_badge",

    # Act V: Petalburg Gym & Norman (Gym 5: Norman)
    "reach_petalburg_gym",
    "fifth_badge",

    # Act VI: Route 118, Weather Institute & Fortree (Gym 6: Winona)
    "surf_route118",
    "weather_institute",
    "defeat_rival_route119",
    "reach_fortree",
    "devon_scope",
    "sixth_badge",

    # Act VII: Mt. Pyre, Hideouts & Mossdeep (Gym 7: Tate & Liza)
    "reach_route121",
    "mt_pyre",
    "reach_lilycove",
    "magma_hideout",
    "aqua_hideout",
    "reach_mossdeep",
    "seventh_badge",
    "space_center_battle",

    # Act VIII: Seafloor Cavern, Rayquaza & Sootopolis (Gym 8: Juan)
    "seafloor_cavern",
    "reach_sootopolis",
    "sky_pillar_rayquaza",
    "calm_weather",
    "eighth_badge",

    # Act IX: Victory Road & Hall of Fame (Champion Wallace)
    "reach_ever_grande",
    "pass_victory_road",
    "defeat_champion",
)

# Objectives requiring flag telemetry verification
OBJECTIVES_REQUIRING_FLAG_TELEMETRY = frozenset({
    "defeat_rival",
    "return_birch_lab",
    "pokedex",
    "wally_tutorial",
    "defeat_aqua_woods",
    "first_badge",
    "rusturf_tunnel_rescue",
    "receive_pokenav",
    "second_badge",
    "deliver_letter_steven",
    "oceanic_museum",
    "third_badge",
    "smash_rusturf_rock",
    "meteor_falls",
    "mt_chimney_summit",
    "fourth_badge",
    "fifth_badge",
    "weather_institute",
    "defeat_rival_route119",
    "devon_scope",
    "sixth_badge",
    "mt_pyre",
    "magma_hideout",
    "aqua_hideout",
    "seventh_badge",
    "space_center_battle",
    "seafloor_cavern",
    "sky_pillar_rayquaza",
    "calm_weather",
    "eighth_badge",
    "pass_victory_road",
    "defeat_champion",
})

# Objectives whose success condition is primarily reaching a destination. During these
# stages, optional story/battle/party rewards are distractions: the target map is the task.
# Keep combat and story rewards enabled for stages that actually require a battle or flag.
NAVIGATION_OBJECTIVES = frozenset(
    stage for stage in STAGES if stage.startswith("reach_")
) | frozenset({
    "leave_birch_lab",
    "return_birch_lab",
    "sail_to_dewford",
    "surf_route118",
})


def is_navigation_objective(objective: str) -> bool:
    """Return True when a stage is a destination-only navigation objective."""
    return str(objective or "").strip().lower() in NAVIGATION_OBJECTIVES


# Route 110's rival battle is recorded in VAR_ROUTE110_STATE (SaveBlock1 vars),
# not a persistent event flag. Keep this separate from event-flag telemetry.
OBJECTIVES_REQUIRING_VAR_TELEMETRY = frozenset({"defeat_rival_route110"})

OBJECTIVE_DEFINITIONS: dict[str, dict[str, str]] = {
    # Act I
    "starter": {
        "label": "rescue Birch, win the wild battle and reach the lab",
        "description": "Obtain the starter, finish the opening wild battle, and reach Birch's lab.",
    },
    "leave_birch_lab": {
        "label": "exit Birch's Lab into Littleroot Town",
        "description": "Complete Birch's lab introduction and enter Littleroot Town (map 9).",
    },
    "reach_route101": {
        "label": "reach Route 101",
        "description": "Navigate from Littleroot Town onto Route 101 (map 16).",
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
    "reach_route102": {
        "label": "reach Route 102",
        "description": "Head west from Oldale Town onto Route 102 (map 17).",
    },
    "reach_petalburg": {
        "label": "reach Petalburg City",
        "description": "Travel west through Route 102 into Petalburg City (map 0).",
    },
    "wally_tutorial": {
        "label": "complete Wally catching tutorial",
        "description": "Visit Norman at Petalburg Gym and complete Wally's Ralts capture sequence (Wally speech flag 0xC1).",
    },
    "reach_route104_south": {
        "label": "reach Route 104 South",
        "description": "Leave Petalburg City west onto Route 104 South (map 19).",
    },
    "defeat_aqua_woods": {
        "label": "defeat Team Aqua in Petalburg Woods",
        "description": "Rescue the Devon researcher and clear the Aqua grunt from Petalburg Woods (event flags 0x2D4 and 0x2D5).",
    },
    "reach_rustboro": {
        "label": "reach Rustboro City",
        "description": "Pass through Route 104 North and enter Rustboro City (map 3).",
    },
    "first_badge": {
        "label": "first Gym Badge (Stone Badge)",
        "description": "Defeat Gym Leader Roxanne in Rustboro Gym (Stone Badge, flag 0x867).",
    },

    # Act II
    "reach_route116": {
        "label": "reach Route 116",
        "description": "Head east from Rustboro City onto Route 116 (map 31).",
    },
    "rusturf_tunnel_rescue": {
        "label": "rescue Peeko in Rusturf Tunnel",
        "description": "Defeat the Aqua Grunt and rescue Peeko in Rusturf Tunnel (Peeko hide flag 0x370).",
    },
    "receive_pokenav": {
        "label": "receive PokéNav from Devon Corp",
        "description": "Return the Devon Goods and receive the PokéNav (story flag 0xBC / system flag 0x862).",
    },
    "sail_to_dewford": {
        "label": "sail to Dewford Town",
        "description": "Board Mr. Briney's boat at Route 104 and arrive in Dewford Town (map 11).",
    },
    "second_badge": {
        "label": "second Gym Badge (Knuckle Badge)",
        "description": "Defeat Gym Leader Brawly in Dewford Gym (Knuckle Badge, flag 0x868).",
    },
    "deliver_letter_steven": {
        "label": "deliver letter to Steven",
        "description": "Navigate Granite Cave and deliver the Devon Letter to Steven Stone (story flag 0xBD).",
    },
    "reach_slateport": {
        "label": "reach Slateport City",
        "description": "Sail with Mr. Briney to Route 109 and enter Slateport City (map 1).",
    },
    "oceanic_museum": {
        "label": "defeat Aqua at Oceanic Museum",
        "description": "Defeat the Oceanic Museum Aqua group, Archie, and finish the Captain Stern scene (hide flags 0x373–0x377).",
    },

    # Act III
    "reach_route110": {
        "label": "reach Route 110",
        "description": "Travel north from Slateport City onto Route 110 (map 25).",
    },
    "defeat_rival_route110": {
        "label": "defeat Rival on Route 110",
        "description": "Defeat Rival May/Brendan on Route 110 and complete the post-battle exit event (VAR_ROUTE110_STATE=1).",
    },
    "reach_mauville": {
        "label": "reach Mauville City",
        "description": "Follow Route 110 north and enter Mauville City (map 2).",
    },
    "third_badge": {
        "label": "third Gym Badge (Dynamo Badge)",
        "description": "Defeat Gym Leader Wattson in Mauville Gym (Dynamo Badge, flag 0x869).",
    },
    "smash_rusturf_rock": {
        "label": "smash Rusturf Tunnel rocks (HM04)",
        "description": "Open Rusturf Tunnel by smashing the connecting rocks (story flag 0xC7).",
    },

    # Act IV
    "reach_route111_north": {
        "label": "reach Route 111 North via Fiery Path",
        "description": "Travel north from Mauville through Route 111 and Fiery Path onto Route 111 North (map 26).",
    },
    "reach_fallarbor": {
        "label": "reach Fallarbor Town",
        "description": "Travel west through volcanic Route 113 into Fallarbor Town (map 13).",
    },
    "meteor_falls": {
        "label": "confront villainous team in Meteor Falls",
        "description": "Navigate Route 114 to Meteor Falls and complete the Archie encounter (story flag 0xCF).",
    },
    "mt_chimney_summit": {
        "label": "defeat evil team atop Mt. Chimney",
        "description": "Ride Cable Car to Mt. Chimney summit, stop volcanic device and defeat the evil boss (flag 0x8B).",
    },
    "reach_lavaridge": {
        "label": "reach Lavaridge Town",
        "description": "Descend Jagged Pass into Lavaridge Town (map 12).",
    },
    "fourth_badge": {
        "label": "fourth Gym Badge (Heat Badge)",
        "description": "Defeat Gym Leader Flannery in Lavaridge Gym (Heat Badge, flag 0x86A).",
    },

    # Act V
    "reach_petalburg_gym": {
        "label": "return to Petalburg Gym",
        "description": "Travel back to Petalburg City to challenge Norman (map 0).",
    },
    "fifth_badge": {
        "label": "fifth Gym Badge (Balance Badge)",
        "description": "Defeat Gym Leader Norman in Petalburg Gym (Balance Badge, flag 0x86B, receives Surf).",
    },

    # Act VI
    "surf_route118": {
        "label": "surf east across Route 118",
        "description": "Use Surf to cross Route 118 east towards Route 119 (map 33).",
    },
    "weather_institute": {
        "label": "rescue Weather Institute",
        "description": "Defeat the villainous team at the Weather Institute and receive Castform (story flag 0x97).",
    },
    "defeat_rival_route119": {
        "label": "defeat Rival on Route 119",
        "description": "Defeat Rival May/Brendan on Route 119 (rival-hide event flag 0x353).",
    },
    "reach_fortree": {
        "label": "reach Fortree City",
        "description": "Enter the treetop city of Fortree (map 4).",
    },
    "devon_scope": {
        "label": "obtain Devon Scope from Steven",
        "description": "Meet Steven on Route 120 bridge and obtain the Devon Scope (story flag 0x11D).",
    },
    "sixth_badge": {
        "label": "sixth Gym Badge (Feather Badge)",
        "description": "Defeat Gym Leader Winona in Fortree Gym (Feather Badge, flag 0x86C).",
    },

    # Act VII
    "reach_route121": {
        "label": "reach Route 121 past Safari Zone",
        "description": "Travel east along Route 120 and Route 121 (map 36).",
    },
    "mt_pyre": {
        "label": "climb Mt. Pyre summit",
        "description": "Complete the Mt. Pyre summit encounter, after Maxie leaves and the Red/Blue Orb is received (flags 0x358 and 0xD4).",
    },
    "reach_lilycove": {
        "label": "reach Lilycove City",
        "description": "Enter the coastal city of Lilycove (map 5).",
    },
    "magma_hideout": {
        "label": "infiltrate Magma Hideout & awaken Groudon",
        "description": "Infiltrate Magma Hideout in Jagged Pass and awaken Groudon (flag 0x6F).",
    },
    "aqua_hideout": {
        "label": "infiltrate Aqua Hideout in Lilycove",
        "description": "Defeat Aqua Admins in Lilycove Hideout before Submarine Explorer 1 departs (flag 0x70).",
    },
    "reach_mossdeep": {
        "label": "reach Mossdeep City",
        "description": "Surf east across Route 124 into island city Mossdeep (map 6).",
    },
    "seventh_badge": {
        "label": "seventh Gym Badge (Mind Badge)",
        "description": "Defeat Gym Leaders Tate & Liza in Mossdeep Gym (Mind Badge, flag 0x86D).",
    },
    "space_center_battle": {
        "label": "save Mossdeep Space Center with Steven",
        "description": "Win the tag team battle with Steven against Team Magma at the Space Center (flag 0x75, Dive).",
    },

    # Act VIII
    "seafloor_cavern": {
        "label": "infiltrate Seafloor Cavern & awaken Kyogre",
        "description": "Reach the legendary event at Seafloor Cavern and trigger Kyogre escaping (story flag 0x81).",
    },
    "reach_sootopolis": {
        "label": "reach Sootopolis City",
        "description": "Dive into the Sootopolis volcanic crater during the apocalyptic weather crisis (map 7).",
    },
    "sky_pillar_rayquaza": {
        "label": "climb Sky Pillar & awaken Rayquaza",
        "description": "Ascend Sky Pillar and trigger Rayquaza to depart for Sootopolis (hide flag 0x305).",
    },
    "calm_weather": {
        "label": "witness Rayquaza calm the legendaries",
        "description": "Return to Sootopolis and complete the scene where Archie and Maxie leave after Rayquaza intervenes (story flag 0x9E).",
    },
    "eighth_badge": {
        "label": "eighth Gym Badge (Rain Badge)",
        "description": "Defeat Gym Leader Juan in Sootopolis Gym (Rain Badge, flag 0x86E, Waterfall).",
    },

    # Act IX
    "reach_ever_grande": {
        "label": "reach Ever Grande City waterfall",
        "description": "Surf east through Route 128 to Ever Grande City waterfall (map 8).",
    },
    "pass_victory_road": {
        "label": "navigate Victory Road & defeat Wally",
        "description": "Climb waterfall, conquer Victory Road, and defeat rival Wally (story flag 0x7E).",
    },
    "defeat_champion": {
        "label": "conquer Pokémon League & defeat Champion Wallace",
        "description": "Defeat Elite Four (Sidney, Phoebe, Glacia, Drake) and Champion Wallace into Hall of Fame (flag 0x864).",
    },
}

OBJECTIVE_LABELS: dict[str, str] = {name: definition["label"] for name, definition in OBJECTIVE_DEFINITIONS.items()}

OBJECTIVE_TARGET_MAPS: dict[str, int] = {
    "leave_birch_lab": LITTLEROOT_TOWN_MAP_ID,
    "reach_route101": ROUTE_101_MAP_ID,
    "reach_oldale": OLDALE_TOWN_MAP_ID,
    "reach_route103": ROUTE_103_MAP_ID,
    "return_birch_lab": BIRCH_LAB_MAP_ID,
    "reach_route102": ROUTE_102_MAP_ID,
    "reach_petalburg": PETALBURG_CITY_MAP_ID,
    "reach_route104_south": ROUTE_104_MAP_ID,
    "reach_rustboro": RUSTBORO_CITY_MAP_ID,
    "reach_route116": ROUTE_116_MAP_ID,
    "sail_to_dewford": DEWFORD_TOWN_MAP_ID,
    "reach_slateport": SLATEPORT_CITY_MAP_ID,
    "reach_route110": ROUTE_110_MAP_ID,
    "reach_mauville": MAUVILLE_CITY_MAP_ID,
    "reach_route111_north": ROUTE_111_MAP_ID,
    "reach_fallarbor": FALLARBOR_TOWN_MAP_ID,
    "reach_lavaridge": LAVARIDGE_TOWN_MAP_ID,
    "reach_petalburg_gym": PETALBURG_CITY_MAP_ID,
    "surf_route118": ROUTE_118_MAP_ID,
    "reach_fortree": FORTREE_CITY_MAP_ID,
    "reach_route121": ROUTE_121_MAP_ID,
    "reach_lilycove": LILYCOVE_CITY_MAP_ID,
    "reach_mossdeep": MOSSDEEP_CITY_MAP_ID,
    "reach_sootopolis": SOOTOPOLIS_CITY_MAP_ID,
    "reach_ever_grande": EVER_GRANDE_CITY_MAP_ID,
}

# Per-objective episode rollout horizons (1 rollout = PPO_N_STEPS per environment).
# Tailored dynamically to each stage's geographical and event complexity.
STAGE_EPISODE_ROLLOUTS: dict[str, int] = {
    # Short initial events
    "starter": 6,
    "leave_birch_lab": 15,
    "reach_route101": 20,
    "reach_oldale": 20,
    "reach_route103": 24,
    "defeat_rival": 32,
    "return_birch_lab": 40,
    "pokedex": 8,

    # Act I
    "reach_route102": 24,
    "reach_petalburg": 32,
    "wally_tutorial": 16,
    "reach_route104_south": 24,
    "defeat_aqua_woods": 48,
    "reach_rustboro": 32,
    "first_badge": 64,

    # Act II
    "reach_route116": 24,
    "rusturf_tunnel_rescue": 40,
    "receive_pokenav": 24,
    "sail_to_dewford": 24,
    "second_badge": 64,
    "deliver_letter_steven": 48,
    "reach_slateport": 32,
    "oceanic_museum": 48,

    # Act III
    "reach_route110": 32,
    "defeat_rival_route110": 48,
    "reach_mauville": 32,
    "third_badge": 64,
    "smash_rusturf_rock": 32,

    # Act IV
    "reach_route111_north": 40,
    "reach_fallarbor": 40,
    "meteor_falls": 48,
    "mt_chimney_summit": 56,
    "reach_lavaridge": 32,
    "fourth_badge": 64,

    # Act V
    "reach_petalburg_gym": 40,
    "fifth_badge": 64,

    # Act VI
    "surf_route118": 32,
    "weather_institute": 56,
    "defeat_rival_route119": 48,
    "reach_fortree": 32,
    "devon_scope": 32,
    "sixth_badge": 64,

    # Act VII
    "reach_route121": 32,
    "mt_pyre": 56,
    "reach_lilycove": 32,
    "magma_hideout": 64,
    "aqua_hideout": 64,
    "reach_mossdeep": 40,
    "seventh_badge": 64,
    "space_center_battle": 48,

    # Act VIII
    "seafloor_cavern": 64,
    "reach_sootopolis": 40,
    "sky_pillar_rayquaza": 64,
    "calm_weather": 40,
    "eighth_badge": 64,

    # Act IX
    "reach_ever_grande": 48,
    "pass_victory_road": 80,
    "defeat_champion": 128,
}
DEFAULT_STAGE_EPISODE_ROLLOUTS = 32


def stage_episode_rollouts(stage: str) -> int:
    """Return the shared per-stage PPO-rollout horizon."""
    key = str(stage or "").strip().lower()
    if key not in STAGES:
        raise ValueError(f"Unknown stage {key!r}; choose one of: {', '.join(STAGES)}")
    return STAGE_EPISODE_ROLLOUTS.get(key, DEFAULT_STAGE_EPISODE_ROLLOUTS)


def evaluation_gate_unreachable(
    successes: int, completed: int, requested: int, threshold: float,
) -> bool:
    """Return whether even perfect remaining episodes cannot satisfy the gate.

    Work in integer success counts rather than comparing rounded percentages. This
    avoids float-boundary errors for thresholds such as 90% over ten episodes.
    """
    if int(requested) < 1:
        raise ValueError("requested episodes must be positive")
    if int(completed) < 0 or int(completed) > int(requested):
        raise ValueError("completed episodes must be between zero and requested")
    if int(successes) < 0 or int(successes) > int(completed):
        raise ValueError("successes must be between zero and completed episodes")
    if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if int(completed) == int(requested):
        return False
    required_successes = math.ceil(float(threshold) * int(requested) - 1e-12)
    maximum_possible_successes = int(successes) + int(requested) - int(completed)
    return maximum_possible_successes < required_successes


def stage_start_state_path(
    project_dir: str | Path,
    objective: str,
    canonical_start_state: str | Path | None = None,
) -> Path:
    """Return the authoritative savestate path for a curriculum objective.

    The first stage always starts from the pre-starter Route 101 state. Every
    later stage starts from the immediately preceding milestone.
    """
    root = Path(project_dir).expanduser().resolve()
    stage = str(objective or "").strip().lower()
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose one of: {', '.join(STAGES)}")

    index = STAGES.index(stage)
    configured = canonical_start_state or "runtime_data/states/training_start.state"
    candidate = Path(configured).expanduser()
    canonical_path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    milestone_root = (root / "runtime_data" / "states" / "milestones").resolve()
    try:
        canonical_path.relative_to(milestone_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "POKEMON_AI_CANONICAL_START_STATE cannot point inside the completed-milestones directory: "
            f"{canonical_path}. It must identify the pre-starter Route 101 savestate."
        )

    if index == 0:
        return canonical_path

    prior_stage = STAGES[index - 1]
    return (root / "runtime_data" / "states" / "milestones" / f"{prior_stage}.state").resolve()


def resolve_runtime_start_state_path(
    project_dir: str | Path,
    objective: str,
    *,
    mode: str = "train",
    canonical_start_state: str | Path | None = None,
    training_start_state: str | Path | None = None,
    continuous_tour: bool = False,
) -> Path:
    """Resolve and enforce the runtime start-state contract for every entry point.

    Continuous watch intentionally traverses the whole game in one emulator
    session, so it always starts from the canonical pre-starter save and ignores
    the ordinary per-stage POKEMON_AI_TRAINING_START_STATE value.
    """
    root = Path(project_dir).expanduser().resolve()
    stage = str(objective or "").strip().lower()
    runtime_mode = str(mode or "train").strip().lower()
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose one of: {', '.join(STAGES)}")

    milestone_root = (root / "runtime_data" / "states" / "milestones").resolve()

    def resolve(value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    canonical_value = str(canonical_start_state).strip() if canonical_start_state is not None else ""
    training_value = str(training_start_state).strip() if training_start_state is not None else ""
    if not canonical_value and training_value and stage == STAGES[0]:
        legacy_candidate = resolve(training_value)
        try:
            legacy_candidate.relative_to(milestone_root)
        except ValueError:
            canonical_value = training_value

    if not canonical_value:
        canonical_value = "runtime_data/states/training_start.state"

    canonical_path = stage_start_state_path(root, STAGES[0], canonical_value)

    if continuous_tour:
        if runtime_mode != "watch":
            raise ValueError("continuous_tour is only valid with mode='watch'")
        return canonical_path

    expected_stage_path = stage_start_state_path(root, stage, canonical_path)

    if runtime_mode == "setup":
        return canonical_path

    if training_value:
        configured_stage_path = resolve(training_value)
        if configured_stage_path != expected_stage_path:
            raise ValueError(
                f"POKEMON_AI_OBJECTIVE={stage!r} must use its mapped start state "
                f"{expected_stage_path}, but POKEMON_AI_TRAINING_START_STATE resolves to "
                f"{configured_stage_path}. Run configure_training.py for this stage, or unset "
                "the stale shell variable so the shared curriculum mapping can be used."
            )

    return expected_stage_path


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


def objective_satisfied(objective: str, state: dict) -> bool:
    """Evaluate a curriculum milestone from decoded live game state only."""
    objective = str(objective or "").strip().lower()
    if objective not in OBJECTIVE_DEFINITIONS:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of " + ", ".join(OBJECTIVE_DEFINITIONS)
        )
    if isinstance(state, dict) and state.get("saveblock_pointer_stable", True) is False:
        return False
    if (
        objective in OBJECTIVES_REQUIRING_FLAG_TELEMETRY
        and isinstance(state, dict)
        and state.get("flag_telemetry_valid", True) is False
    ):
        return False
    if objective in OBJECTIVES_REQUIRING_VAR_TELEMETRY:
        if not isinstance(state, dict) or state.get("route110_var_telemetry_valid", False) is not True:
            return False
        route110_state = _number(state.get("route110_state"))
        if route110_state is None or not route110_state.is_integer() or int(route110_state) != 1:
            return False
    metrics = _state_metrics(state)
    if metrics is None:
        return False
    party = metrics["party_count"] >= 1 and metrics["party_levels"] >= 1
    map_id = metrics["map_id"]
    flags = metrics["story_flags"]
    badges = metrics["badge_count"]

    # Act I
    if objective == "starter":
        return party and map_id == BIRCH_LAB_MAP_ID
    if objective == "leave_birch_lab":
        return party and map_id == LITTLEROOT_TOWN_MAP_ID
    if objective == "reach_route101":
        return party and map_id == ROUTE_101_MAP_ID
    if objective == "reach_oldale":
        return party and (map_id == OLDALE_TOWN_MAP_ID or FLAG_VISITED_OLDALE_TOWN in flags)
    if objective == "reach_route103":
        return party and map_id == ROUTE_103_MAP_ID
    if objective == "defeat_rival":
        return party and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags
    if objective == "return_birch_lab":
        return party and map_id == BIRCH_LAB_MAP_ID and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags
    if objective == "pokedex":
        return party and RIVAL_DEFEATED_ROUTE103_FLAG_ID in flags and (POKEDEX_OBTAINED_FLAG_ID in flags or FLAG_ADVENTURE_STARTED in flags)
    if objective == "reach_route102":
        return party and map_id == ROUTE_102_MAP_ID
    if objective == "reach_petalburg":
        return party and (map_id == PETALBURG_CITY_MAP_ID or FLAG_VISITED_PETALBURG_CITY in flags)
    if objective == "wally_tutorial":
        return party and FLAG_WALLY_TUTORIAL_COMPLETED in flags
    if objective == "reach_route104_south":
        return party and map_id == ROUTE_104_MAP_ID
    if objective == "defeat_aqua_woods":
        return party and {FLAG_RESCUED_DEVON_RESEARCHER, FLAG_RESCUED_DEVON_RESEARCHER_AQUA_GRUNT}.issubset(flags)
    if objective == "reach_rustboro":
        return party and (map_id == RUSTBORO_CITY_MAP_ID or FLAG_VISITED_RUSTBORO_CITY in flags)
    if objective == "first_badge":
        return badges >= 1 or FLAG_BADGE01_GET in flags

    # Act II
    if objective == "reach_route116":
        return party and map_id == ROUTE_116_MAP_ID
    if objective == "rusturf_tunnel_rescue":
        return party and FLAG_RUSTURF_TUNNEL_PEEKO_RESCUED in flags
    if objective == "receive_pokenav":
        return party and (POKENAV_OBTAINED_FLAG_ID in flags or FLAG_POKENAV_RECEIVED in flags)
    if objective == "sail_to_dewford":
        return party and (map_id == DEWFORD_TOWN_MAP_ID or FLAG_VISITED_DEWFORD_TOWN in flags)
    if objective == "second_badge":
        return badges >= 2 or FLAG_BADGE02_GET in flags
    if objective == "deliver_letter_steven":
        return party and FLAG_STEVEN_LETTER_DELIVERED in flags
    if objective == "reach_slateport":
        return party and (map_id == SLATEPORT_CITY_MAP_ID or FLAG_VISITED_SLATEPORT_CITY in flags)
    if objective == "oceanic_museum":
        museum_completion_flags = {
            FLAG_OCEANIC_MUSEUM_GRUNTS_HIDDEN,
            FLAG_OCEANIC_MUSEUM_GRUNT1_HIDDEN,
            FLAG_OCEANIC_MUSEUM_GRUNT2_HIDDEN,
            FLAG_OCEANIC_MUSEUM_ARCHIE_HIDDEN,
            FLAG_OCEANIC_MUSEUM_STERN_HIDDEN,
        }
        return party and museum_completion_flags.issubset(flags)

    # Act III
    if objective == "reach_route110":
        return party and map_id == ROUTE_110_MAP_ID
    if objective == "defeat_rival_route110":
        # The event script sets VAR_ROUTE110_STATE=1 only after the player
        # defeats May/Brendan and completes the post-battle exit sequence.
        return party and int(route110_state) == 1
    if objective == "reach_mauville":
        return party and (map_id == MAUVILLE_CITY_MAP_ID or FLAG_VISITED_MAUVILLE_CITY in flags)
    if objective == "third_badge":
        return badges >= 3 or FLAG_BADGE03_GET in flags
    if objective == "smash_rusturf_rock":
        return party and FLAG_RUSTURF_TUNNEL_OPENED in flags

    # Act IV
    if objective == "reach_route111_north":
        return party and map_id in {ROUTE_111_MAP_ID, ROUTE_112_MAP_ID}
    if objective == "reach_fallarbor":
        return party and (map_id == FALLARBOR_TOWN_MAP_ID or FLAG_VISITED_FALLARBOR_TOWN in flags)
    if objective == "meteor_falls":
        return party and FLAG_METEOR_FALLS_TEAM_SCENE in flags
    if objective == "mt_chimney_summit":
        return party and FLAG_MT_CHIMNEY_SUMMIT_DEFEATED in flags
    if objective == "reach_lavaridge":
        return party and (map_id == LAVARIDGE_TOWN_MAP_ID or FLAG_VISITED_LAVARIDGE_TOWN in flags)
    if objective == "fourth_badge":
        return badges >= 4 or FLAG_BADGE04_GET in flags

    # Act V
    if objective == "reach_petalburg_gym":
        return party and map_id == PETALBURG_CITY_MAP_ID and badges >= 4
    if objective == "fifth_badge":
        return badges >= 5 or FLAG_BADGE05_GET in flags

    # Act VI
    if objective == "surf_route118":
        return party and map_id == ROUTE_118_MAP_ID and badges >= 5
    if objective == "weather_institute":
        return party and FLAG_RECEIVED_CASTFORM in flags
    if objective == "defeat_rival_route119":
        return party and FLAG_RIVAL_DEFEATED_ROUTE119 in flags
    if objective == "reach_fortree":
        return party and (map_id == FORTREE_CITY_MAP_ID or FLAG_VISITED_FORTREE_CITY in flags)
    if objective == "devon_scope":
        return party and FLAG_RECEIVED_DEVON_SCOPE in flags
    if objective == "sixth_badge":
        return badges >= 6 or FLAG_BADGE06_GET in flags

    # Act VII
    if objective == "reach_route121":
        return party and map_id == ROUTE_121_MAP_ID
    if objective == "mt_pyre":
        return party and FLAG_MT_PYRE_MAXIE_HIDDEN in flags and FLAG_RECEIVED_RED_OR_BLUE_ORB in flags
    if objective == "reach_lilycove":
        return party and (map_id == LILYCOVE_CITY_MAP_ID or FLAG_VISITED_LILYCOVE_CITY in flags)
    if objective == "magma_hideout":
        return party and FLAG_AWAKENED_GROUDON in flags
    if objective == "aqua_hideout":
        return party and FLAG_AQUA_SUBMARINE_ESCAPED in flags
    if objective == "reach_mossdeep":
        return party and (map_id == MOSSDEEP_CITY_MAP_ID or FLAG_VISITED_MOSSDEEP_CITY in flags)
    if objective == "seventh_badge":
        return badges >= 7 or FLAG_BADGE07_GET in flags
    if objective == "space_center_battle":
        return party and FLAG_SPACE_CENTER_DEFEATED in flags

    # Act VIII
    if objective == "seafloor_cavern":
        return party and FLAG_KYOGRE_ESCAPED_SEAFLOOR_CAVERN in flags
    if objective == "reach_sootopolis":
        return party and (map_id == SOOTOPOLIS_CITY_MAP_ID or FLAG_VISITED_SOOTOPOLIS_CITY in flags)
    if objective == "sky_pillar_rayquaza":
        return party and FLAG_AWAKENED_RAYQUAZA in flags
    if objective == "calm_weather":
        return party and FLAG_RAYQUAZA_CALMED_WEATHER in flags
    if objective == "eighth_badge":
        return badges >= 8 or FLAG_BADGE08_GET in flags

    # Act IX
    if objective == "reach_ever_grande":
        return party and (map_id == EVER_GRANDE_CITY_MAP_ID or FLAG_VISITED_EVER_GRANDE_CITY in flags)
    if objective == "pass_victory_road":
        return party and FLAG_DEFEATED_WALLY_VICTORY_ROAD in flags
    if objective == "defeat_champion":
        return FLAG_SYS_GAME_CLEAR in flags

    return False


def objective_start_error(objective: str, state: dict, *, route_101_map_id: int = ROUTE_101_MAP_ID) -> str | None:
    """Return a prerequisite error, or None when the state is valid for a stage."""
    objective = str(objective or "").strip().lower()
    if objective not in OBJECTIVE_DEFINITIONS:
        return "unknown objective; choose one of: " + ", ".join(OBJECTIVE_DEFINITIONS)
    if isinstance(state, dict) and state.get("saveblock_pointer_stable", True) is False:
        return "SaveBlock1 pointer changed during the state read; map/position telemetry is not trustworthy"
    metrics = _state_metrics(state)
    if metrics is None:
        return "game state is missing valid party, map, badge, or story-flag telemetry"
    requires_flags = objective == "starter" or objective in OBJECTIVES_REQUIRING_FLAG_TELEMETRY
    if requires_flags and isinstance(state, dict) and state.get("flag_telemetry_valid", True) is False:
        return "the SaveBlock1 event/badge flag snapshot is unstable; this objective requires verified flags"
    if objective in OBJECTIVES_REQUIRING_VAR_TELEMETRY:
        if not isinstance(state, dict) or state.get("route110_var_telemetry_valid", False) is not True:
            return "the Route 110 progression variable could not be read consistently from SaveBlock1"
    party = metrics["party_count"] >= 1 and metrics["party_levels"] >= 1
    map_id = metrics["map_id"]
    flags = metrics["story_flags"]
    badges = metrics["badge_count"]

    if objective == "starter":
        if map_id != route_101_map_id:
            return f"the starter task must start on Route 101 (map_id={route_101_map_id}); current map_id is {map_id}"
        if CLOCK_SET_FLAG_ID not in flags:
            return "the bedroom clock-set flag is absent; save after setting the clock"
        if BIRCH_RESCUED_FLAG_ID in flags:
            return "the starter task must start before rescuing Birch and before receiving a starter"
        if metrics["party_count"] != 0 or metrics["party_levels"] != 0:
            return "the starter task must start before receiving a starter, with an empty party and no party levels"
        return None

    if not party:
        return f"stage {objective!r} requires a valid starter: a valid Pokémon in the live party"

    if objective == "leave_birch_lab":
        if map_id not in {BIRCH_LAB_MAP_ID, LITTLEROOT_TOWN_MAP_ID}:
            return f"the leave-the-lab stage must start in Birch's Lab or Littleroot Town; current map_id={map_id}"
        return None
    if objective == "reach_route101":
        if map_id not in {LITTLEROOT_TOWN_MAP_ID, ROUTE_101_MAP_ID}:
            return f"the Route 101 stage must start in Littleroot Town or Route 101; current map_id={map_id}"
        return None
    if objective == "return_birch_lab":
        if RIVAL_DEFEATED_ROUTE103_FLAG_ID not in flags:
            return "the return-to-lab stage requires the Route 103 rival-defeated flag (0x82)"
        return None
    if objective == "pokedex":
        if RIVAL_DEFEATED_ROUTE103_FLAG_ID not in flags:
            return "the Pokédex stage requires the verified Route 103 rival-defeated milestone (flag 0x82)"
        return None

    # Gym Badge prerequisites
    if "first_badge" in objective and POKEDEX_OBTAINED_FLAG_ID not in flags:
        return "first_badge requires the Pokédex to be obtained"
    if "second_badge" in objective and badges < 1 and FLAG_BADGE01_GET not in flags:
        return "second_badge requires the Stone Badge (first badge)"
    if "third_badge" in objective and badges < 2 and FLAG_BADGE02_GET not in flags:
        return "third_badge requires the Knuckle Badge (second badge)"
    if "fourth_badge" in objective and badges < 3 and FLAG_BADGE03_GET not in flags:
        return "fourth_badge requires the Dynamo Badge (third badge)"
    if "fifth_badge" in objective and badges < 4 and FLAG_BADGE04_GET not in flags:
        return "fifth_badge requires the Heat Badge (fourth badge)"
    if "sixth_badge" in objective and badges < 5 and FLAG_BADGE05_GET not in flags:
        return "sixth_badge requires the Balance Badge (fifth badge)"
    if "seventh_badge" in objective and badges < 6 and FLAG_BADGE06_GET not in flags:
        return "seventh_badge requires the Feather Badge (sixth badge)"
    if "eighth_badge" in objective and badges < 7 and FLAG_BADGE07_GET not in flags:
        return "eighth_badge requires the Mind Badge (seventh badge)"
    if "defeat_champion" in objective and badges < 8 and FLAG_BADGE08_GET not in flags:
        return "defeat_champion requires all 8 Gym Badges"

    return None
