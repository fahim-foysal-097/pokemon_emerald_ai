# Incremental Pokémon Emerald story curriculum

This curriculum breaks the opening story into short, verifiable subgoals while preserving the existing recurrent PPO policy and observation/action spaces. The baseline is the pre-autonomous `Pokemon_Emerald_RL_Log_IO_Update` code. There is intentionally **no alternate-start-state bank** in this update.

## Stage order

| Stage | Success condition | Starts from |
|---|---|---|
| `starter` | A valid Pokémon is in the live party and the player reaches map ID 260 (Birch's Lab), avoiding premature success on Route 101 during the opening battle | `runtime_data/states/training_start.state` (Route 101, before rescuing Birch) |
| `leave_birch_lab` | Valid party and map ID 16 (Route 101), used as the observable proxy for finishing the lab introduction and exiting | `milestones/starter.state` |
| `reach_oldale` | Valid party and map ID 10 (Oldale Town) | `milestones/leave_birch_lab.state` |
| `reach_route103` | Valid party and map ID 18 (Route 103) | `milestones/reach_oldale.state` |
| `defeat_rival` | Valid party and vanilla story flag `0x82` (Route 103 rival defeated) | `milestones/reach_route103.state` |
| `return_birch_lab` | Valid party, flag `0x82`, and map ID 260 | `milestones/defeat_rival.state` |
| `pokedex` | Valid party, rival flag `0x82`, and system Pokédex flag `0x861` | `milestones/return_birch_lab.state` |
| `first_badge` | At least one Gym Badge | `milestones/pokedex.state` |

Map IDs use the project's BPEE map-group/map-number encoding. The shared `curriculum_objectives.py` file is the single source of truth for stage order, names, map IDs, flags, success checks, and start validation. Both `main.py` and `run_curriculum.py` import it.

## Exact in-game sequence behind the stages

The vanilla Emerald opening sequence is: rescue Professor Birch and choose the starter on Route 101; return to Birch's Lab; leave along Route 101 and travel north through Oldale Town; continue to Route 103 and defeat the rival; return to Birch's Lab; receive the Pokédex. The lab scripts use `VAR_BIRCH_LAB_STATE` to track starter selection/receipt, the Route 103 rival win, and Pokédex receipt. The rival-defeated event is vanilla story flag `0x82`; the Pokédex system flag is `0x861`. Because Emerald can execute lab events automatically on entering the building, `return_birch_lab` and `pokedex` remain separate verifiable gates even if the latter becomes satisfied very soon after the former.

Sources:
- Bulbapedia, *Walkthrough: Pokémon Emerald/Part 1*: https://bulbapedia.bulbagarden.net/wiki/Appendix:Emerald_walkthrough/Section_1
- pret/pokeemerald lab event scripts (`VAR_BIRCH_LAB_STATE` and Pokédex receipt): https://github.com/pret/pokeemerald/blob/master/data/maps/LittlerootTown_ProfessorBirchsLab/scripts.inc
- pret/pokeemerald story flag definitions (`FLAG_DEFEATED_RIVAL_ROUTE103 = 0x82`): https://github.com/pret/pokeemerald/blob/master/include/constants/flags.h

## Budget arithmetic

The recommended fast-train setup has four environments and `PPO_N_STEPS=2048`:

- Rollout = `2048 × 4 = 8192` transitions.
- Chunk = `245760 ÷ 8192 = 30` complete rollouts.
- Per environment per chunk = `245760 ÷ 4 = 61440` transitions = 30 rollout lengths per worker.
- Eight chunks per stage = at most `8 × 245760 = 1,966,080` training transitions per stage (before evaluation runs).
- A direct `main.py` run with `TOTAL_TIMESTEPS=2,007,040` targets `245` complete 8192-transition rollouts.

These are training budgets. Evaluation runs add emulator steps and are not counted in a stage's requested training chunk. If a stage passes early, later chunks are skipped.

## Compatibility / resume behavior

The plan ID is `emerald_intro_v4`. Existing checkpoints are preserved and evaluated; they are not retrained from scratch solely because the curriculum has expanded. If `runtime_data/curriculum_state.json` has a different plan ID (for example, the former three-stage sequence), the controller restarts stage selection at `starter`, retains the existing model checkpoint, and evaluates it before training. This prevents an old `pokedex.state` from being treated as proof that each new route/battle/lab stage has been completed.

The policy architecture, observation schema, and action space are unchanged; the existing hybrid checkpoint remains the intended checkpoint. The additional runtime dependency is `curriculum_objectives.py`. This update contains no state-bank capture, alternate start-state sampling, or state-bank environment settings.
