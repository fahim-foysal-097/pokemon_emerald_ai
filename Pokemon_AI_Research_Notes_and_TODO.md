# Pokémon Emerald AI — Research Notes & Prioritized TODO

**Prepared:** 2026-10-28  
**Decision:** Keep the current mGBA + Stable-Baselines3 project. Use other repositories as research references, porting only verified ideas rather than replacing the GUI/runtime.

## How to read the checkboxes

- `[x]` = implemented in the **updated source bundle** and confirmed by static inspection; it does not automatically mean the user's local installation has been updated or that the behavior has passed a live mGBA test.
- `[ ]` = not implemented, or not yet verified to the standard needed for a reliable claim.
- Live integration is tracked separately below. Passing helper tests is not the same as validating a RAM field against a running ROM.

The user confirmed that their ROM SHA-1 matches the vanilla Pokémon Emerald BPEE revision 0 SHA-1 expected by `pret/pokeemerald`: `f3ae088181bf583e55daf962a92bb46f4f1d07b7`. The updated code also enforces that expected SHA-1 at startup by default. The code bundle has not yet completed a live test against the user's mGBA build.

## 1. Current implementation status

### Implemented in the updated source bundle

- [x] Verify ROM game code/revision and exact expected SHA-1 before creating the mGBA bootstrap core; reject a mismatch unless the user explicitly changes the setting.
- [x] Read SaveBlock1 through a validated pointer for position, location, flags and trainer-flag data; reject invalid pointers/ranges.
- [x] Read the **live** `gPlayerPartyCount` and `gPlayerParty` globals for party count/levels instead of relying only on the SaveBlock1 party copy, which may not be synchronized until the game saves. Log both sources for diagnosis.
- [x] Validate canonical start states for the named objectives. `starter` starts on Route 101 after clock setting, before the Birch rescue/starter, with an empty party; `pokedex` requires a starter and an incomplete Pokédex objective; `first_badge` requires a valid party and the Pokédex milestone.
- [x] Hybrid observation space combining resized RGB, a normalized structured state vector, and a visited-tile mask; explicit WAIT action and action-release frames.
- [x] Recurrent PPO policy architecture, under a checkpoint name distinct from the previous RGB-only/eight-action model schema.
- [x] Named objectives: `starter`, `pokedex`, and `first_badge`; objective completion is based on validated party/story/badge telemetry and produces a terminal success signal.
- [x] One-time tile/map/story/trainer/party/level/badge reward deltas, a small step penalty, bounded screen novelty and anti-stagnation penalties, and a distinct task-success bonus.
- [x] Completed-episode reward-component sums. Component means are based on actual completed episodes, not intermediate per-step values mislabelled as final sums.
- [x] Per-worker map/coordinate and progress diagnostics, concise terminal episode outcomes, task-success/timeout tracking, and JSON evaluation reports with checkpoint and ROM identity.
- [x] Deterministic or stochastic fixed-start evaluation (`./run.sh --profile evaluate`), configurable episode count, per-episode outcome and aggregate success/return/length/timeout metrics.
- [x] Train/evaluate time limits; GUI watch/setup do not terminate merely because the training step limit was reached. Watch mode supports a configurable success grace period so the post-success transition can be observed.
- [x] GUI arrow-key regression coverage; lowercase `q` or Esc is the unambiguous quit action, and `K` requests saving a canonical start state in setup mode.
- [x] Configurable failure limit for repeated RAM/state reads, with explicit diagnostic information rather than silently continuing forever.
- [x] Objective-completion savestate plus JSON manifest recording objective, state telemetry, ROM identity and evaluation-checkpoint fingerprint when available.
- [x] Atomic latest-model saving and a bounded archive of distinct, validated model checkpoints.
- [x] Curriculum controller that evaluates the current checkpoint, screens a bounded number of recent archived checkpoints if necessary, fully re-evaluates a promising candidate, backs up the existing model before candidate activation, trains in configurable chunks, and advances only after the success-rate gate and a fresh matching milestone manifest/state are verified.
- [x] Curriculum state persistence and a true `--dry-run` that previews commands without launching the emulator or changing project files.
- [x] Soft mini-episodes for the longer `pokedex` and `first_badge` objectives. They reset short-term exploration memory/loop history without resetting emulator state, overall episode return, the PPO episode or LSTM state. They are disabled for the starter task by design; set `POKEMON_AI_MINI_EPISODE_STEPS=0` to disable them altogether.
- [x] Transactional update installer that backs up changed files, merges only missing environment settings, validates syntax/tests, and attempts rollback if validation fails.

### Implemented only partially / important limitations

- [x] Periodic rolling model checkpoints are implemented. [ ] A dedicated **first-success model checkpoint** saved at the exact first successful transition is not yet implemented; the objective savestate and periodic model archives are not the same thing.
- [x] Objective-completion states are saved with metadata. [ ] The environment does not yet save a distinct savestate for *every* verified story event (clock, each route, each trainer, each healing event, etc.).
- [x] The curriculum advances through `starter → pokedex → first_badge` automatically after its configurable gate. [ ] There is no selection among a diverse pool of independently validated starts or random start-state distribution yet.
- [x] Candidate checkpoint screening is implemented and bounded. [ ] It does not evaluate every historical checkpoint; it considers only a limited number of recent distinct archives.
- [x] A fixed-start evaluation profile is implemented. [ ] The default curriculum gate does not yet use a multi-seed, multi-start-state test suite; repeated trials from one canonical savestate measure repeatability on that start, not generalization.
- [x] Repeated telemetry/read failures can stop a run. [ ] A separate watchdog that automatically halts/skips training solely because no new tiles or story events occur for a long interval is not yet implemented as a comprehensive no-progress detector.
- [x] The Pokédex-obtained flag can be used as a milestone. [ ] Species seen/caught counts and detailed Pokédex rewards remain unimplemented pending verified readers.
- [x] Story, trainer, party, level and badge rewards are logged separately. [ ] The code does not yet provide a standalone first-success PPO model snapshot or a full reward-ablation dashboard.
- [x] One recurrent policy can continue across the three named curriculum objectives. [ ] There are not separate battle/overworld policies or a battle-specific action controller yet.

## 2. Live integration validation — still required

A matching ROM checksum is a strong identity check, but it does not itself validate every reader, savestate restore, or mGBA API interaction. These checks must be run in the user's actual `.venv` with the user's mGBA build and ROM before a long curriculum run.

- [ ] **Cold startup:** confirm the startup preflight prints the expected BPEE revision 0 SHA-1 before the mGBA bootstrap core is created.
- [ ] **Clock-set state:** capture a known post-clock savestate and assert Route 101, valid coordinates, clock-set flag, no Birch-rescued flag and an empty live party.
- [ ] **Starter acquired / first battle:** verify the live party count, first party species/level data, party levels, `FLAG_SYS_POKEMON_GET`, SaveBlock1 party count, task-success event and episode termination. Confirm the apparent GUI battle state and telemetry agree.
- [ ] **Movement:** move exactly one tile and verify coordinates change as expected while map group/number remain stable.
- [ ] **Map transition:** cross a real boundary and verify map group/number, coordinates and SaveBlock1 pointer validation.
- [ ] **Badge:** obtain/use a controlled post-badge state and verify the relevant badge flag and badge count; do not infer success only from a non-zero map ID.
- [ ] **Reset/recovery:** confirm each parallel worker returns to its expected canonical start and that the milestone state/manifest belongs to the currently evaluated checkpoint and the expected ROM.
- [ ] **Watch mode:** confirm the GUI continues for the configured success grace window and reports success without accidentally resetting before the user can observe the transition.
- [ ] **Evaluation report:** confirm each completed episode has internally consistent step count, success/timeout status, return, reward-component sums and checkpoint/ROM fingerprint.
- [ ] **Installer in the project `.venv`:** run the complete test command there. A test that cannot import `gymnasium` is an environment/dependency limitation, not a passing test.

### Validation observed so far

- The user confirmed the local ROM SHA-1 matches the expected standard BPEE revision 0 ROM.
- A prior stochastic evaluation of the existing project reported one successful starter episode at 28,250 steps (`success_rate=100%` for that single episode). That is encouraging but is only one episode and was not a live test of the complete updated bundle.
- In this preparation environment, `python -m py_compile main.py run_curriculum.py tests/test_main_logic.py tests/test_curriculum_manager.py` passes.
- In this preparation environment, the curriculum-controller unit tests pass: 7/7.
- `bash -n install_update.sh` passes.
- The main logic tests cannot import in this preparation environment because `gymnasium` is not installed. They must be run in the target project's `.venv`; do not mark them passed based on syntax compilation.
- Full live mGBA integration has not been executed in this preparation environment.

---

## 3. Executive summary: research decisions

The most useful transferable ideas across the projects below are:

1. **Give PPO a task it can actually complete.** Use curriculum stages, validated savestates and explicit success conditions.
2. **Feed validated state to the policy, not only the reward function.** Combine pixels with coordinates, story/game-state signals and a visited-area mask.
3. **Reward verified state changes.** Use new tile/area, unique story/trainer flags, starter acquisition and other confirmed events. Log each reward component and guard against farming.
4. **Evaluate actual game achievements.** Episode reward is diagnostic; milestone success rate and steps-to-milestone are the outcomes that matter.
5. **Treat battles and overworld navigation as different subproblems** once basic navigation, resets and telemetry are reliable.
6. **Never promote a curriculum stage just because reward increased.** Require successful evaluation and a milestone state/manifest produced by the same ROM and checkpoint being considered.

The 2025 Pokémon Red research is especially useful for reward ablations and documenting how plausible rewards can lead to behavior such as avoiding battles, healing loops or aimless wandering. The Emerald experiments project describes a repeatable Mr. Briney event exploit. Adding reward terms without exploit tests is not sufficient.

---

## 4. Projects worth studying

### A. [dvruette/pokemon-emerald-experiments](https://github.com/dvruette/pokemon-emerald-experiments) — Emerald-specific reward reference

**Read first:**
- [Reward wrapper](https://github.com/dvruette/pokemon-emerald-experiments/blob/master/baselines/emerald_wrapper.py)
- [Emerald environment](https://github.com/dvruette/pokemon-emerald-experiments/blob/master/baselines/emerald_env.py)
- [Training README/checkpoint caveat](https://github.com/dvruette/pokemon-emerald-experiments)
- [PyGBA Gymnasium/mGBA wrapper](https://github.com/dvruette/pygba)

**Reusable ideas:** state extraction for maps/coordinates, party and boxed Pokémon, species caught/seen, experience, flags, trainer progression, money, Pokédex/PokéNav, healing and champion status; coordinate novelty per map; healing tracking that waits for stable readings; saturating experience rewards; reward-component dictionaries; and step rewards based on changes in cumulative score.

**Failure mode:** its README describes a 2023-12-03 checkpoint that reached the second Gym but farmed a repeatable Mr. Briney event between Dewford and Petalburg. This is a concrete example of reward hacking.

**Decision:** port concepts selectively after tests. Do not replace mGBA with PyGBA just to import the wrapper. The backend, game state, frame skip, reset and reward scale differ; do not copy coefficients blindly.

### B. [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments) and [Pokémon Red via Reinforcement Learning (2025)](https://arxiv.org/abs/2502.19920) — research and ablation reference

**Read first:**
- [Repository and V2 discussion](https://github.com/PWhiddy/PokemonRedExperiments)
- [Reward/environment implementation](https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/red_gym_env.py)
- [Paper](https://arxiv.org/html/2502.19920v2)
- [Paper section on exploration and reward-exploitation results](https://arxiv.org/html/2502.19920v2#S4)

**Reusable ideas:** coordinate-based exploration rather than frame novelty alone; hybrid visual and structured observations; visited-coordinate memory; intermediate story flags as breadcrumbs; milestone curves, multiple seeds and steps-to-milestone; longer PPO rollouts as an experiment candidate; and press/release rhythm to make movement reliable.

**Failure modes:** weak navigation progress without navigation rewards; excessive navigation rewards encouraging exploration over story progress; healing rewards encouraging grinding/loops; and reward ablations that change story success even when average reward looks better. Fixed-length, synchronized episodes can also destabilize training.

**Decision:** transfer evaluation discipline and reward-hacking lessons, not exact numeric values. The current project has a visited mask, hybrid observations and explicit WAIT, but action-repeat tests, multiple seeds, and several of the paper's ablations remain TODOs.

### C. [reddheeraj/PokemonRL](https://github.com/reddheeraj/PokemonRL) and [PokeRL: Reinforcement Learning for Pokémon Red (2026)](https://arxiv.org/abs/2604.10812) — curriculum and anti-loop reference

**Read first:**
- [Paper](https://arxiv.org/abs/2604.10812)
- [Repository README/reward structure](https://github.com/reddheeraj/PokemonRL)
- [Stage-specific configs](https://github.com/reddheeraj/PokemonRL/tree/master/config)
- [Tests and environment](https://github.com/reddheeraj/PokemonRL/tree/master/tests)

The project describes a modular early-game curriculum (house exit, reaching tall grass, first rival battle) and identifies button/menu loops and unproductive wandering as environment-level issues.

**Reusable ideas:** separate stage goals and start states; stage-specific reward scales; visited masks, stacked frames and loop detection; explicit movement handling; and separate manual-control/environment tests. Its reported success rates are author-reported and have not been independently reproduced here.

**Decision:** keep one codebase but retain named stages, explicit prerequisites, and automated success gates. Its implementation targets Pokémon Red/PyBoy, not Emerald/mGBA.

### D. [Hrishirt/pokegym](https://github.com/Hrishirt/pokegym) — battle-agent engineering reference

**Read first:**
- [Repository and README](https://github.com/Hrishirt/pokegym)
- [Battle environment](https://github.com/Hrishirt/pokegym/tree/main/python/env)
- [Lua TCP bridge](https://github.com/Hrishirt/pokegym/tree/main/lua)

**Reusable ideas:** a compact battle observation from verified RAM; explicit battle win/loss/faint states; damage dealt versus received; enemy-switch detection; waiting for a valid turn; and a known battle-state reset for each battle episode.

**Caveat/decision:** FireRed battle task, not a complete story-playing PPO. Verify the license before copying code. Use as a design reference for a later battle-specific environment only after navigation and reset correctness are stable.

### E. [AndresTY/Pok-mon-Emerald-RL-Agent](https://github.com/AndresTY/Pok-mon-Emerald-RL-Agent) — secondary Emerald event reference

Useful ideas include event detection, frame/state novelty and RAM-derived game events. Its README/TODO also describes unresolved battle reward and RAM-address problems, so treat it as a source of ideas, not an authoritative address map or proven reward implementation. Verify its license before copying source.

### F. [niccolo-farinella/gbai](https://github.com/niccolo-farinella/gbai) — optional hierarchical planner reference

Uses an mGBA/Lua bridge and separates high-level goals from local navigation. This may inform an optional future planner, but no LLM should be introduced into the per-frame control loop now. If explored later, keep planning outside the low-level control loop and use verified, constrained actions.

### G. [DaysonRegulus/pokemon-firered-ai](https://github.com/DaysonRegulus/pokemon-firered-ai) — lower-priority reference

Its CNN/PPO, RAM/visual reward, parallel training and reward-manager ideas may be worth reading, but it is not a primary baseline for this project. Verify the code license and test any idea independently.

---

## 5. Current-version audit

### Present in the updated source bundle

- [x] mGBA native binding + Gymnasium/Stable-Baselines3 PPO; setup, train, evaluate and watch profiles.
- [x] Canonical `training_start.state` and objective-aware start validation.
- [x] RAM/game-state readers for map group/number, coordinates, live party count/levels, badges, curated story flags and trainer-defeated flags.
- [x] Per-episode unique map/tile/story/trainer bookkeeping and a visited-tile policy observation.
- [x] Named objective success conditions and an explicit terminal reward for successful objective completion.
- [x] Reward-component breakdown and completed-episode accounting.
- [x] Structured episode terminal logs, task-success/timeout rates and per-worker map/coordinate metrics.
- [x] Deterministic/stochastic evaluation profile and JSON reports with checkpoint fingerprint.
- [x] Atomic latest PPO checkpoint plus bounded rolling checkpoint archive.
- [x] Automated curriculum runner with evaluation gates, bounded archive comparison, chunked training and prerequisite milestone validation.
- [x] Objective state manifests that associate a milestone with the ROM and evaluated checkpoint identity.
- [x] Soft mini-episodes for `pokedex`/`first_badge` that reset exploration windows but not the emulator or PPO episode.
- [x] Mocked/helper tests in the source tree for GUI keyboard handling, state parsing, objective validation, observation bounds, mini-episode boundaries, ROM verification, checkpoint and manifest helpers.

### Still missing or not proven

1. [ ] **Live known-state validation:** all key fields must be asserted in controlled clock, starter, movement, map-transition and badge states.
2. [ ] **First-success model archive:** save a dedicated PPO checkpoint at the first task success, independently of emulator milestone states and periodic checkpoints.
3. [ ] **Diverse evaluation set:** multiple validated start states and fixed seeds; repeat metrics across seeds.
4. [ ] **Event-by-event story snapshots:** optionally save selected verified story milestones beyond only the current objective completion.
5. [ ] **Diverse start sampling:** select from a pool of validated story states rather than always relying on one state per stage.
6. [ ] **Game-mode detection:** robustly distinguish overworld, dialogue, menu, battle and transition states from verified RAM/UI signals.
7. [ ] **Battle curriculum/controller:** state-aware battle actions and explicit win/loss detection.
8. [ ] **HP/blackout/healing rewards:** only after the party/HP reader and game-mode transitions are verified.
9. [ ] **Species/Pokédex count rewards:** only after verified parsing and unique-species anti-farming logic.
10. [ ] **Route heatmap and video retention:** automatically retain representative success, timeout and loop evidence with seed/config/checkpoint metadata.
11. [ ] **Ablations:** compare action-repeat choices, reward variants and PPO parameters only after correctness and baseline success are established.
12. [ ] **Comprehensive no-progress watchdog:** stop or flag an episode when verified telemetry stays valid but no meaningful progress occurs for a configured interval.

---

## 6. Prioritized TODO list

Legend: **P0 = blocking correctness**, **P1 = next**, **P2 = after learning is stable**, **P3 = optional research**.

### P0 — Verify state and environment correctness

- [x] Startup ROM identity preflight checks game code/revision and required SHA-1 before bootstrapping mGBA.
- [x] Validate SaveBlock1 pointer range/alignment and reject reads while the pointer is uninitialized or invalid.
- [x] Add configurable state diagnostics for map group/number, coordinates, party, levels, badges and key flags.
- [x] Read live party globals and log the SaveBlock1 party count separately for mismatch diagnosis.
- [x] Reject malformed/invalid party, coordinate, map, badge or story telemetry for objective success.
- [ ] Test known states: post-clock, after receiving starter, after moving one tile, after a real map transition and after obtaining a badge.
- [ ] Confirm a live pointer remains valid through map transitions in the user's mGBA binding.
- [ ] Confirm all parallel workers return to the expected start state during a real run.
- [x] Repeated RAM/state read failures trigger bounded failure handling.
- [ ] Add a separate verified no-progress watchdog and test its false-positive behavior.

### P1 — Make curriculum learning finish verifiable tasks

- [x] Implement the `starter` objective using live party count/levels and award one-time task success.
- [x] Implement named `pokedex` and `first_badge` objectives with prerequisite-state validation.
- [x] Curriculum runner evaluates before training and requires both a success-rate gate and a fresh verified milestone state/manifest to promote.
- [x] Resume curriculum metadata safely; re-evaluate the saved stage before continuing.
- [x] Compare a bounded number of recent archived checkpoints if the current model fails the evaluation gate.
- [ ] Add a dedicated model checkpoint for first-ever success, with model/ROM/config/seed metadata.
- [ ] Validate the full starter → Pokédex → badge process on the real ROM, including successful state restoration at each stage.
- [ ] Decide whether later stages need separate policies/heads; current implementation intentionally continues the same recurrent policy architecture.

### P1 — Upgrade rewards safely

- [x] Centralize named reward components: `step`, `tile`, `map`, `screen`, `story`, `party`, `level`, `badge`, `trainer`, `loop`, `task_success`.
- [x] Reward unique verified state changes rather than re-paying a persistent achievement on every frame.
- [x] Reward new tiles/maps and known story/trainer events once per episode; use a bounded screen-novelty cap.
- [x] Add conservative stagnation/loop penalty and expose its component in terminal logs.
- [ ] Add species/Pokédex seen/caught rewards only after the corresponding data structures are verified.
- [ ] Add saturating experience rewards only after testing whether level grinding impedes the badge objective.
- [ ] Add healing rewards only after HP reads are verified, with cooldown/cap anti-exploit rules.
- [ ] Run short controlled reward ablations; retain changes based on milestone success, not reward alone.

### P1 — Log useful evidence of learning

- [x] Completed-episode reward-component sums and true terminal episode boundaries.
- [x] Task success, timeouts, success rate over recent episodes, episode length/return and exploration counts.
- [x] Per-episode terminal reward breakdown with end map/position and progress summaries.
- [x] Machine-readable evaluation report that includes checkpoint identity and aggregate success metrics.
- [x] TensorBoard scalar metrics and training/checkpoint progress.
- [ ] Add standardized report fields for every desired future milestone and a suite-level multi-start/multi-seed summary.
- [ ] Retain representative screenshots/video for success, timeout and loop episodes with reproducible metadata.
- [ ] Add a separate first-success **model** snapshot (not just a milestone savestate).

### P2 — Improve observations and action semantics

- [x] Hybrid observations: resized RGB, normalized state features and visited-tile mask.
- [x] Explicit WAIT action and release-frame support.
- [x] Soft mini-episodes for long objectives, which clear short-term exploration memory without resetting game/episode/LSTM state.
- [ ] Benchmark action-repeat values (for example 2, 4 and 8) and release cadence on the same controlled task.
- [ ] Detect game mode (`OVERWORLD`, `DIALOG`, `MENU`, `BATTLE`, `TRANSITION`) from verified RAM/UI state.
- [ ] Build a separate battle-stage environment after navigation and state reset are reliable.
- [ ] Add a route heatmap showing visit counts, loops, battle locations and successful paths.

### P2 — Training stability and reproducibility

- [x] Four-environment fast-training profile and configurable PPO rollout/batch/learning parameters.
- [x] Atomic checkpoints, bounded archive and safe candidate activation with a backup.
- [x] Deterministic/stochastic evaluation and checkpoint fingerprinting.
- [ ] Complete a live model-load smoke test in the user's `.venv` and mGBA environment after installation.
- [ ] Benchmark PPO `n_steps`, batch size, `gamma`, learning rate and action cadence only after state/reward correctness is established.
- [ ] Run multiple random seeds and fixed evaluation sets.
- [ ] Consider validated diverse start states and asynchronous/dynamic worker budgets if synchronized resets are later shown to be a problem.
- [ ] Run the full main logic suite inside the project's `.venv`; the preparation environment used here lacks `gymnasium`, so that suite did not complete here.

### P3 — Optional future research

- [ ] Curiosity/RND only if coordinate novelty plateaus and anti-exploit evaluation is in place.
- [ ] High-level goal planner (rule-based first, LLM only if justified), outside the per-frame controller.
- [ ] Battle-specific structured state and action policy.
- [ ] Multi-run reward-ablation and checkpoint-selection studies.

---

## 7. Reward principles

- Reward **verifiable forward progress**.
- Reward persistent story/trainer flags on false-to-true transitions, not repeated dialog frames.
- Treat tile novelty as an exploration aid, not the main objective.
- Use diminishing/capped rewards for long-run quantities such as XP, money and healing.
- Keep the step cost small but non-zero; test that it does not overpower attainable progress rewards.
- Make each reward component observable in logs/TensorBoard and testable independently.
- Evaluate actual objectives (starter, Pokédex, badge). Rising reward alone may indicate reward exploitation.
- Every milestone state must be tied to ROM and checkpoint identity; stale or mismatched states must not unlock a curriculum stage.

## 8. Sources and license notes

- `PWhiddy/PokemonRedExperiments`: repository presents an MIT license.
- `dvruette/pokemon-emerald-experiments`: repository contains an MIT license; README documents the repeatable-event exploit.
- `reddheeraj/PokemonRL`: repository presents an MIT license.
- `AndresTY/Pok-mon-Emerald-RL-Agent`: README presents an MIT license and lists unresolved RAM/reward issues.
- `Hrishirt/pokegym`, `DaysonRegulus/pokemon-firered-ai`, and `niccolo-farinella/gbai`: no clearly visible repository license was confirmed in the earlier review; use as reading/reference only until verified.
- ROMs and game assets are not supplied by these projects; use a ROM you legally own. Preserve copyright/license notices whenever reusing licensed source code.
- `pret/pokeemerald` is used as the reference for BPEE revision 0 symbols and game data structures. The agent runs the user's original ROM through mGBA; it does not run the decompiled C project instead of the ROM.

## 9. Sources

- [pret/pokeemerald](https://github.com/pret/pokeemerald)
- [PyGBA — mGBA/Gymnasium wrapper](https://github.com/dvruette/pygba)
- [Pokémon Emerald experiments README](https://github.com/dvruette/pokemon-emerald-experiments)
- [Emerald reward wrapper](https://github.com/dvruette/pokemon-emerald-experiments/blob/master/baselines/emerald_wrapper.py)
- [PWhiddy Pokémon Red RL](https://github.com/PWhiddy/PokemonRedExperiments)
- [PWhiddy reward/environment implementation](https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/red_gym_env.py)
- [Pleines et al., *Pokémon Red via Reinforcement Learning* (2025)](https://arxiv.org/abs/2502.19920)
- [reddheeraj/PokemonRL](https://github.com/reddheeraj/PokemonRL)
- [Mudireddy & Patibandla, *PokeRL: Reinforcement Learning for Pokémon Red* (2026)](https://arxiv.org/abs/2604.10812)
- [Hrishirt/pokegym](https://github.com/Hrishirt/pokegym)
- [AndresTY/Pokémon Emerald RL Agent](https://github.com/AndresTY/Pok-mon-Emerald-RL-Agent)
- [niccolo-farinella/gbai](https://github.com/niccolo-farinella/gbai)
- [DaysonRegulus/pokemon-firered-ai](https://github.com/DaysonRegulus/pokemon-firered-ai)

---

## 10. Next actions in order

1. Install the updated bundle into the project and run the installer-provided checks in the project's `.venv`.
2. Confirm the startup ROM SHA-1 preflight passes.
3. Run a short stochastic evaluation from the canonical starter state and inspect the party-count source, success condition and JSON report.
4. Run controlled live-state checks for clock, starter/battle, tile movement, map transition and badge flag.
5. Only after those pass, run `run_curriculum.py --dry-run`, inspect the plan, then start the managed curriculum.
6. Add the first-success model checkpoint, wider evaluation suite and deferred game-mode/battle signals before treating the framework as fully validated.
