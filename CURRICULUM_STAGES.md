# Complete Pokémon Emerald Story Curriculum (Start to Hall of Fame)

This curriculum guides the recurrent PPO policy through the complete storyline of vanilla Pokémon Emerald across 58 milestones. Story-event IDs are taken from the vanilla BPEE `pret/pokeemerald` flag definitions; multi-flag objectives require all listed flags. The event telemetry is separate from the PPO observation features, so existing hybrid checkpoints retain the same observation shape.

All stages start from the immediately preceding verified milestone `.state`, starting originally from the canonical post-clock start state (`runtime_data/states/training_start.state` on Route 101, before Birch's bag).

---

## 1. Act I: Littleroot Town to Stone Badge (Rustboro Gym)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `starter` | Rescue Birch & obtain starter | Live party $\ge 1$, map ID 260 (Birch's Lab) | `runtime_data/states/training_start.state` |
| `leave_birch_lab` | Exit Birch's Lab | Map ID 9 (Littleroot Town) | `milestones/starter.state` |
| `reach_route101` | Walk north onto Route 101 | Map ID 16 (Route 101) | `milestones/leave_birch_lab.state` |
| `reach_oldale` | Enter Oldale Town | Map ID 10 or flag `0x870` | `milestones/reach_route101.state` |
| `reach_route103` | Walk north to Route 103 | Map ID 18 (Route 103) | `milestones/reach_oldale.state` |
| `defeat_rival` | Defeat May/Brendan on Route 103 | Story flag `0x82` | `milestones/reach_route103.state` |
| `return_birch_lab` | Return to Birch's Lab | Map ID 260 & flag `0x82` | `milestones/defeat_rival.state` |
| `pokedex` | Receive the Pokédex from Birch | System flag `0x861` or story flag `0x74` | `milestones/return_birch_lab.state` |
| `reach_route102` | Travel west onto Route 102 | Map ID 17 (Route 102) | `milestones/pokedex.state` |
| `reach_petalburg` | Enter Petalburg City | Map ID 0 or flag `0x876` | `milestones/reach_route102.state` |
| `wally_tutorial` | Complete Norman & Wally tutorial | Story flag `0xC1` (`FLAG_WALLY_SPEECH`) | `milestones/reach_petalburg.state` |
| `reach_route104_south` | Exit Petalburg onto Route 104 South | Map ID 19 (Route 104) | `milestones/wally_tutorial.state` |
| `defeat_aqua_woods` | Rescue Devon researcher in Petalburg Woods | Hide flags `0x2D4` + `0x2D5` | `milestones/reach_route104_south.state` |
| `reach_rustboro` | Enter Rustboro City | Map ID 3 or flag `0x879` | `milestones/defeat_aqua_woods.state` |
| `first_badge` | Defeat Gym Leader Roxanne | Stone Badge (`FLAG_BADGE01_GET = 0x867`) | `milestones/reach_rustboro.state` |

---

## 2. Act II: Dewford Island & Slateport (Knuckle Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_route116` | Head east from Rustboro onto Route 116 | Map ID 31 (Route 116) | `milestones/first_badge.state` |
| `rusturf_tunnel_rescue` | Rescue Peeko & recover Devon Goods | Peeko hide flag `0x370` | `milestones/reach_route116.state` |
| `receive_pokenav` | Receive PokéNav from Devon Corp President | Story flag `0xBC` or system flag `0x862` | `milestones/rusturf_tunnel_rescue.state` |
| `sail_to_dewford` | Sail with Mr. Briney to Dewford Town | Map ID 11 or flag `0x871` | `milestones/receive_pokenav.state` |
| `second_badge` | Defeat Gym Leader Brawly in Dewford Gym | Knuckle Badge (`FLAG_BADGE02_GET = 0x868`) | `milestones/sail_to_dewford.state` |
| `deliver_letter_steven` | Deliver Letter to Steven in Granite Cave | Story flag `0xBD` | `milestones/second_badge.state` |
| `reach_slateport` | Sail with Mr. Briney to Slateport City | Map ID 1 or flag `0x877` | `milestones/deliver_letter_steven.state` |
| `oceanic_museum` | Defeat Team Aqua at Oceanic Museum | Hide flags `0x373`–`0x377` (all five, including Captain Stern) | `milestones/reach_slateport.state` |

---

## 3. Act III: Mauville City & Cycling Road (Dynamo Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_route110` | Travel north onto Route 110 | Map ID 25 (Route 110) | `milestones/oceanic_museum.state` |
| `defeat_rival_route110` | Defeat May/Brendan on Route 110 and complete the exit scene | `VAR_ROUTE110_STATE` (`0x4069`) equals `1` (SaveBlock1 offset `0x146E`) | `milestones/reach_route110.state` |
| `reach_mauville` | Enter Mauville City | Map ID 2 or flag `0x878` | `milestones/defeat_rival_route110.state` |
| `third_badge` | Defeat Gym Leader Wattson in Mauville Gym | Dynamo Badge (`FLAG_BADGE03_GET = 0x869`) | `milestones/reach_mauville.state` |
| `smash_rusturf_rock` | Open Rusturf Tunnel for HM04 Strength | Story flag `0xC7` | `milestones/third_badge.state` |

---

## 4. Act IV: Mt. Chimney, Meteor Falls & Lavaridge (Heat Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_route111_north` | Travel north through Fiery Path | Map ID 26 (Route 111) | `milestones/smash_rusturf_rock.state` |
| `reach_fallarbor` | Travel west to Fallarbor Town | Map ID 13 or flag `0x873` | `milestones/reach_route111_north.state` |
| `meteor_falls` | Confront Team Magma/Aqua in Meteor Falls | Story flag `0xCF` | `milestones/reach_fallarbor.state` |
| `mt_chimney_summit` | Defeat evil boss atop Mt. Chimney | Story flag `0x8B` | `milestones/meteor_falls.state` |
| `reach_lavaridge` | Descend Jagged Pass into Lavaridge Town | Map ID 12 or flag `0x872` | `milestones/mt_chimney_summit.state` |
| `fourth_badge` | Defeat Gym Leader Flannery | Heat Badge (`FLAG_BADGE04_GET = 0x86A`) | `milestones/reach_lavaridge.state` |

---

## 5. Act V: Balance Badge & Surf (Petalburg Gym)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_petalburg_gym` | Return to Petalburg Gym to face Norman | Map ID 0 & Badges $\ge 4$ | `milestones/fourth_badge.state` |
| `fifth_badge` | Defeat Gym Leader Norman | Balance Badge (`FLAG_BADGE05_GET = 0x86B`) | `milestones/reach_petalburg_gym.state` |

---

## 6. Act VI: Route 118, Weather Institute & Fortree (Feather Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `surf_route118` | Surf east across Route 118 | Map ID 33 (Route 118) | `milestones/fifth_badge.state` |
| `weather_institute` | Rescue Weather Institute on Route 119 and receive Castform | Story flag `0x97` | `milestones/surf_route118.state` |
| `defeat_rival_route119` | Defeat Rival on Route 119 for HM02 Fly | Hide flag `0x353` | `milestones/weather_institute.state` |
| `reach_fortree` | Enter Fortree City | Map ID 4 or flag `0x87A` | `milestones/defeat_rival_route119.state` |
| `devon_scope` | Obtain Devon Scope from Steven on Route 120 | Story flag `0x11D` | `milestones/reach_fortree.state` |
| `sixth_badge` | Defeat Gym Leader Winona in Fortree Gym | Feather Badge (`FLAG_BADGE06_GET = 0x86C`) | `milestones/devon_scope.state` |

---

## 7. Act VII: Mt. Pyre, Hideouts & Mossdeep (Mind Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_route121` | Travel through Route 120/121 past Safari Zone | Map ID 36 (Route 121) | `milestones/sixth_badge.state` |
| `mt_pyre` | Climb Mt. Pyre summit, complete Maxie encounter and receive Orb | Flags `0x358` + `0xD4` | `milestones/reach_route121.state` |
| `reach_lilycove` | Enter coastal city Lilycove | Map ID 5 or flag `0x87B` | `milestones/mt_pyre.state` |
| `magma_hideout` | Awaken Groudon in Magma Hideout | Story flag `0x6F` | `milestones/reach_lilycove.state` |
| `aqua_hideout` | Infiltrate Aqua Hideout in Lilycove | Story flag `0x70` | `milestones/magma_hideout.state` |
| `reach_mossdeep` | Surf east to island city Mossdeep | Map ID 6 or flag `0x87C` | `milestones/aqua_hideout.state` |
| `seventh_badge` | Defeat Gym Leaders Tate & Liza | Mind Badge (`FLAG_BADGE07_GET = 0x86D`) | `milestones/reach_mossdeep.state` |
| `space_center_battle` | Defeat Team Magma at Space Center with Steven | Story flag `0x75` | `milestones/seventh_badge.state` |

---

## 8. Act VIII: Seafloor Cavern, Rayquaza & Sootopolis (Rain Badge)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `seafloor_cavern` | Complete Seafloor Cavern event; Kyogre escapes | Story flag `0x81` | `milestones/space_center_battle.state` |
| `reach_sootopolis` | Enter Sootopolis crater during weather crisis | Map ID 7 or flag `0x87D` | `milestones/seafloor_cavern.state` |
| `sky_pillar_rayquaza` | Climb Sky Pillar and trigger Rayquaza to depart | Hide flag `0x305` | `milestones/reach_sootopolis.state` |
| `calm_weather` | Rayquaza calms the Sootopolis crisis and Archie/Maxie leave | Story flag `0x9E` | `milestones/sky_pillar_rayquaza.state` |
| `eighth_badge` | Defeat Gym Leader Juan in Sootopolis Gym | Rain Badge (`FLAG_BADGE08_GET = 0x86E`) | `milestones/calm_weather.state` |

---

## 9. Act IX: Victory Road & Hall of Fame (Champion Wallace)

| Stage | Objective | Success condition | Starts from |
|---|---|---|---|
| `reach_ever_grande` | Surf east to Ever Grande City waterfall | Map ID 8 or flag `0x87E` | `milestones/eighth_badge.state` |
| `pass_victory_road` | Conquer Victory Road & defeat Wally | Story flag `0x7E` | `milestones/reach_ever_grande.state` |
| `defeat_champion` | Defeat Elite Four and Champion Wallace | Game Clear (`FLAG_SYS_GAME_CLEAR = 0x864`) | `milestones/pass_victory_road.state` |

---

## Controls & Tooling

- **Autonomous Full-Game Training**:
  ```bash
  ./run.sh --auto
  ```
- **Inspect Specific Stage in GUI**:
  ```bash
  ./run.sh --profile gui_watch --stage reach_route101
  ./run.sh --profile gui_watch --stage first_badge
  ```
- **Continuous Tour from the Beginning**:
  ```bash
  ./run.sh --profile continuous_watch
  ```
- **In-GUI Fast Forward**: Press `F` or `Spacebar`, or click the `[FF: ON/OFF]` button in the top-right overlay to run unthrottled at maximum hardware speed.
