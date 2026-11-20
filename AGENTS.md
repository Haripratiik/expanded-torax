# AGENTS.md: operating Expanded TORAX

For an AI agent (or a person) running the simulator and its harness. [README.md](README.md) says what each part is and how far to trust it, with every measured error and limitation; this file says how to operate it without breaking anything or fooling yourself. Work from the repository's folder with a Python that has TORAX 1.4.3:

```bash
python xtorax.py <command>
```

## Rules

These hold for every task. Breaking one makes results unusable or harms other work on the machine.

1. **Held-out shots stay held out.** If the workspace keeps `runs/twin/dev_shots.json`, a MAST shot may be run, scored or read only if it is on that list; `xtorax.py run`, `natural.py run` and `natural.hands_off` refuse others. Never pass `--allow_non_dev` except to reproduce a run made before the list existed, and never for a shot held out for a test.
2. **Data are read only.** Never write into the MAST data store (`XTORAX_MAST_STORE`). Set `PYTHONDONTWRITEBYTECODE=1` (the entry point does).
3. **Shared machine.** Stop only processes you started, by PID, together with their children (each MAST run starts a FreeGSNKE server, and with neutrals a KN1D server). Never touch a process you did not start. The harness and front end use port 8801 unless told otherwise.
4. **Memory.** A whole-shot run with FreeGSNKE takes 2 to 3 GB. Run at most three at once, and check free memory first.
5. **Versions are frozen.** Never edit an entry of `natural.VERSIONS`. A physics change becomes a new version only after it has been scored on all the scorable shots, compared with the recommended version on the same starts, and found better; `natural.RECOMMENDED` changes only on that evidence.
6. **No law fitted to MAST in the general code.** Coefficients come from published multi-machine work or from physics; a machine file (`core/machines.py`) holds hardware and programmes only; MAST's measurements may check a model, never set it.
7. **Honest reporting.** Write the prediction before a run. Report each number as measured or inferred, give the command that reproduces it, and lead with the 10% bar (README, "How far to trust it"). Results on ITER, SPARC and MAST-U are beyond the range any shot here checks: say so. When a change does not move the agreed criteria after two or three tries, stop and say so.
8. **Downloads** only when needed: say what, why and how big, and put each in its own new folder and environment.

## The entry point

| Command | What it does | Time |
|---|---|---|
| `env` | checks TORAX's environment, the optional environments (FreeGSNKE, KN1D, zarr), DREAM, the data and port 8801, and says what each enables | 1 min |
| `versions` | every frozen version and the recommended one | seconds |
| `shots [--all]` | MAST shots in the workspace that can be run, and those that can also be scored | seconds |
| `run SHOT ... \| --scorable [--version V] [--tag T] [--seeds N] [--lanes N] [--extra JSON]` | whole MAST shots hands-off from their first frame; `--scorable` all scorable shots; `--lanes` up to 3 processes (tags TAGa, TAGb, TAGc); `--extra` adds override keys to the version's | 1 to 4 min per shot |
| `score TAG [--name N]` | scores runs against MAST's measurements (TAG takes in its lanes, or give a comma list) | 1 to 2 min |
| `compare A B` | two scored sets on the same starts and times | 1 min |
| `ready N` | a scored set against the readiness criteria | 1 min |
| `script SCENARIO FILE.json` | a scenario with a list of commands | minutes |
| `serve SCENARIO [--port 8801]` | the harness as a local server | stays up |
| `call 'JSON' [--port 8801]` | one command to a running server | per command |
| `studio [--scenario S]` | the front end at http://127.0.0.1:8801 | stays up |
| `smoke` | the functional checks, branches included; checks whose optional pieces are missing are skipped | 10 to 12 min with everything |

Long commands (`run` on many shots, `serve`, `studio`, `smoke`) should run in the background with their output in a log; wait for them with a check on a file or the process, not with fixed sleeps.

## Recipes

**Drive a plasma through the server.** Start the server in the background, then send commands; a branch is how to ask what-if:

```bash
python xtorax.py serve studio_iter --port 8801
```

```bash
python xtorax.py call '{"cmd": "step", "until": 2}'
```

```bash
python xtorax.py call '{"cmd": "snapshot", "name": "flat_top"}'
```

```bash
python xtorax.py call '{"cmd": "island", "m": 2, "n": 1, "w_cm": 4}'
```

```bash
python xtorax.py call '{"cmd": "restore", "name": "flat_top"}'
```

Or a whole list at once: `python xtorax.py script studio_iter examples/iter_quickstart.json`. Scenarios are the files in `scenarios/`. The quoting above is for Git Bash or a Linux or macOS shell; in Windows PowerShell, put the commands in a file and use `script`.

**From Python.** Put `core/` on `sys.path`; `twin.Harness(json.load(open("scenarios/studio_iter.json")))` builds a scenario's plasma, `h.apply({...})` takes the same commands, and `h.apply({"cmd": "observe"})` reads the plasma. `examples/run_iter.py` is a complete example. For a MAST shot, `natural.hands_off(shot)` returns the harness with the recommended version and FreeGSNKE on it as `h.free`; close FreeGSNKE with `h.free.close()` when done (`examples/run_shot.py`).

**Run a MAST shot and score it** (needs MAST shot files in the workspace; README, "MAST shots"):

```bash
python xtorax.py run 22786 --tag my22786
```

```bash
python xtorax.py score my22786
```

The score prints, per quantity, the median error against time beside holding the first frame, events (ELMs, crashes, n = 1 mode time, H-mode time) and how the run ended.

**Test a change to the physics.** Put it behind an override (a key in the overrides dict that `core/coupled.py`'s `build_shot` reads, off by default), write your prediction, then run the scorable shots with the recommended version plus that key, score them, and compare with the recommended version's reference set:

```bash
python xtorax.py run --scorable --lanes 3 --tag mynew --extra '{"my_option": true}'
```

```bash
python xtorax.py score mynew
```

```bash
python xtorax.py compare REFERENCE mynew
```

```bash
python xtorax.py ready mynew
```

Record what was predicted, what was measured and whether it was adopted. A change that does better becomes a new entry of `natural.VERSIONS` (never an edit of an old one), and `natural.RECOMMENDED` moves to it.

**Check health** after changing code: `python xtorax.py smoke` (every check that runs must pass).

## The server's protocol

POST a JSON object with `cmd` and its parameters to `http://127.0.0.1:8801/`; the answer is JSON. `GET /observe` returns the plasma now, `GET /help` the command table with parameters. One command runs at a time. The server answers this machine only: a request whose Host is not 127.0.0.1 or localhost on its port, or a browser request from another site, gets 403. `decision` with `"kind": "python"` imports and calls a function by name, so the server refuses it unless started with `--allow-python`; scripts and Python sessions may use it. An unknown command answers `{"error": ..., "commands": [...]}`. Times are in seconds of plasma time (`until` is absolute; `n` counts TORAX steps). `observe` returns scalars (time, fusion power and Q or the MAST scalars, stored energy, line density, temperatures, q, H-mode, ELMs, islands with their width, rotation and lock, the drive), and with `"profiles": true` T_e, T_i, n_e and q against rho.

Commands and their parameters:

| Command | Parameters |
|---|---|
| `step` | `n` steps, or `until` (s) |
| `observe` | `profiles` (true adds the profiles) |
| `island` | `m`, `n`, `w_cm` |
| `profile` | `name` (T_e, T_i, n_e), `factor`, `rho` [a, b] |
| `kick` | `frac` of T_e and T_i lost inside `rho` [a, b] |
| `crash` | `rho` [a, b]: flatten T keeping the energy |
| `elm` | none |
| `pedestal` | `T`, `n` scale of the pedestal top, until changed |
| `pedestal_model` | `name` (EPED1NNmodel, EPED1NNensemble, off), `interval` |
| `couple` | `models`, `H0`, `n_line` (number or {t: value}) |
| `free_boundary` | `machine` (ITER, or MAST on a MAST shot), `mode` (control, coils, off), `interval`, `shape`; needs FreeGSNKE |
| `quench` | MAST: `mode` (static, full), `growth`; elsewhere (DREAM): `material` (none, neon, neon+deuterium), `arrive_ms`, `tpost` |
| `turbulence` | `factor`, `rho` [a, b], `duration`, `channel` (heat, e, i, particles, all) |
| `turbulence_noise` | `sigma`, `tau`, `length`, `rho`, `duration`, `channel`, `seed` |
| `heating` | `factor`, until changed |
| `brake` | `m`, `n`, `factor` |
| `error_field` | `factor`, until changed |
| `physics` | island coefficients: `a_bs`, `a_cd`, `a_ggj`, ... |
| `trip` | `units`, `duration` (gyrotrons lost) |
| `mirror_stuck` | `duration` |
| `sensor_bias` | `surface_cm`, `width_cm`, `duration` |
| `decision` | `spec`, as in a scenario (`kind`: none, rule, phys (control on the island equation), net (a forecaster, with trained weights you supply), python (your own function: `"target": "module:function"` or `"path/to/file.py:function"`; refused over HTTP without `--allow-python`), etched (a separate, unpublished project's chip, not included)); `core/twin.py`'s header lists each kind's settings |
| `drive` | `P` (MW), `aim` (rho) |
| `set`, `get` | `path` of a TORAX runtime parameter (for example `sources.gas_puff.S_total`), `value` |
| `snapshot`, `restore` | `name`. A snapshot holds everything a step reads: TORAX's state, the reactor's carried readings, ELM draws and module arrays, the plant, every model on the step (the coupled plasma's random draws, H-mode, density loop and transport table, EPED-NN's schedule, the free boundary's client) and FreeGSNKE's own state. A null branch repeats the first run exactly (`core/check_branches.py`); the same name can be restored any number of times |
| `save` | `path` (npz of the traces so far) |
| `help` | none |

Actuators and instabilities, each tagged with its level (torax: TORAX's own model; reduced: a published reduced equation; prescribed: an assumed effect, a what-if; margin: a 0D watch that only ends the shot). `{"cmd": "help"}` gives every default and range.

| Command | Level | Parameters |
|---|---|---|
| `pellet` | reduced | `species`, `d_mm` (an injector's size: a cylinder, length equal to diameter), or `r_mm` (a sphere's radius; wins over `d_mm`); with neither, the machine's own injector (MAST 1.1 mm, ITER 5 mm; else a 3 mm radius); `v`, `side` (hfs, lfs), `drift`. The answer gives `pellet_atoms`, `plasma_electrons`, `share_of_plasma`, `size_from`, and a `warning` when a pellet holds more atoms than the plasma has electrons |
| `spi` | reduced | `species`, `atoms`, `hydrogen`, `shards`, `v`, `flight`, `mix`, `mix_at` |
| `gas` | torax | `species`, `rate` (atoms/s), `delay` (s), `duration` (s, 0 until changed) |
| `laser_blowoff` | reduced | `species`, `atoms` |
| `nbi` | prescribed | `P`, `E_keV`, `rho0`, `width`, `f_e` |
| `icrh` | prescribed | `P`, `rho0`, `width`, `f_i` |
| `ecrh` | prescribed | `P`, `rho0`, `width` |
| `rmp` | prescribed | `level` (0 to 1) |
| `ip` | torax | `I_MA`, `rate` (MA/s) |
| `density_control` | torax | `target` (1e20 m^-3, 0 off), `gain`, `rate_max` |
| `radiation_control` | torax | `target_MW`, `species`, `gain`, `rate_max` |
| `mitigation` | margin | `armed`, `width_cm`, `delay`, `species`, `atoms`, `hydrogen` |
| `fishbone` | prescribed | `period`, `expelled`, `lost`, `rho`, `seed_prob`, `seed_cm`, `duration` |
| `alfven` | prescribed | `redistributed`, `lost`, `rho`, `duration` |
| `wall_flake` | reduced | `species`, `atoms` |
| `marfe` | prescribed | `share`, `pedestal`, `duration` |
| `back_transition` | prescribed | `T`, `n`, `ramp`, `duration` |
| `transport_barrier` | prescribed | `factor`, `rho0`, `rho1`, `duration` |
| `rwm` | margin | `seed`, `beta_N_limit`, `tau_wall`, `gain` |
| `vde` | margin | `kick_cm`, `growth`, `gain`, `v_max`, `z_max_cm` |
| `limits` | margin | `on`, `greenwald`, `hl_hold` |

## Reading results

- `runs/twin/mast_natural2/<tag>/results.jsonl`: one JSON line per run: `shot`, `seed`, `t_cut` (the start), `t_end`, `steps`, `sim_error` (`SimError.NO_ERROR` when TORAX did not stop it), `disrupted`, `elm_times`, `h_log` (H-mode changes), `seeds` (island seeds), `collapses` (beta-limit crossings), `overrides`, `free` (FreeGSNKE: `start` forward, inverse or refused; `log` of accepted solves with the shape and the synthetic whmd; `rejected`).
- `<shot>_<seed>.npz`: one value per TORAX step: `t`, `n_line`, `T_e0`, `T_i0`, `W_thermal`, `P_SOL`, `P_rad`, `P_ohmic`, `q95`, `q0`, `q_min`, `li3`, `H`, `P_net`, `P_LH_twin`, `elm`, `sawtooth`, `w_2_1` (island width, m), `locked_2_1`, `kink_n1` (the n = 1 internal mode reading), `beam_W_fast`, `beam_neutrons`, `beam_P_abs`, `beta_N`, `beta_over_limit`, `line_target`, `rad_core`, `rad_sol_c`, `rad_h_main`, `rad_h_div`, `rad_bolo`. `<shot>_<seed>_prof.npz`: `t`, `rho`, `T_e`, `T_i`, `n_e`.
- `runs/twin/whole82/summary_<name>.json`: `effects` (each with `median`, `q25`, `q75`, `held`, `signed` against `grid_s`, `within_bar`, `twin_better`), `per_run`, `events`, `ends`, `profiles`.
- Units are SI (m^-3, W, J, s, m) except temperatures, which are in keV, and the scrape-off layer's and targets' temperatures (`sol_*`, `rad_T_t`), which are in eV.

## Things that will surprise you

- The first step of a run compiles TORAX's step with JAX (about 20 s), and a MAST start waits about 25 s before that for FreeGSNKE's start solve; a run with a different grid compiles again.
- On Windows, call each environment's interpreter by path or through its variable. `python -I` leaves the script's own folder off the import path; the scripts in `core/` add it themselves, so keep them there.
- FreeGSNKE and KN1D run as separate processes in their own environments, talking over pipes. A refused or failed solve keeps the last accepted shape; the start's solve is checked against EFIT's plasma and retried, or the run keeps EFIT's surfaces (`free.start` says which).
- Fuelling by neutrals (`fuelling: kn1d_gauge`) calls KN1D every 10 ms of plasma time at 10 to 20 s each; it reads MAST's measured wall pressure, so it is an oracle, not a prediction.
- ELMs, sawteeth, island seeds and some laws are drawn per run from seeds, and even the same seed is not always repeated exactly: one shot run twice gave q95 errors of 3.0% and 3.3% (floating-point drift, README.md). Compare statistics over shots or seeds, never one run against another.
- Branches are exact from release 1.1: within one process, a restore brings back every model on the step and FreeGSNKE's own state, so a null branch repeats the first run on every record (`core/check_branches.py`). A free boundary built inside a branch is closed on restore; if FreeGSNKE's server stopped after a snapshot, that snapshot cannot be restored.
- Pellets: MAST's injector made 0.6 to 2.4e20 atoms per pellet (1.1 to 1.7 mm), 0.44 to 1.64 of the electrons of a low-density plasma such as 20689's; smaller pellets are a what-if beyond MAST's hardware, so label them. The line density jumps at once and decays over about 12 ms, then sits 1 to 3.5% below the null branch for a few tens of ms while version 12's density loop rebuilds its source: read a pellet's response against its null branch, and the first 10 to 20 ms most. A pellet past the Greenwald density ends the shot only with `limits` on.
- A run that ends early either stopped on TORAX's numerical checks (`sim_error`) or disrupted on its own (`disrupted`); `xtorax.py ready` separates the two.
- A sleeping computer stalls runs; the quench steps have time limits so a run recovers after waking.

## Where to look

[README.md](README.md): what it is, how to install and run it, the scenarios, the harness, the optional pieces, the main measured errors and where things live. [VALIDATION.md](VALIDATION.md): the full measured record and every known limitation. `core/twin.py`'s header: every scenario key and decision model. Each module's own header: its physics, sources and checks. Headers cite tests by code (H34 to H91) and files under `docs/`: they belong to the project's development log, which is not published.
