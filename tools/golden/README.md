# Golden-output harness (non-regression safety net)

Pins the simulation's current behaviour so the refactor can be verified step by
step: after each change the metrics must be **identical**, not merely plausible.

## Usage

```bash
# once, before touching any code (already done — references are committed)
python tools/golden/run_cases.py record --data-root /path/to/data

# after every refactor step
python tools/golden/run_cases.py check --data-root /path/to/data
```

`--data-root` is a directory containing the project's `Data*/`, `Reward_weights/`
and `SVG_model/` folders. It is only ever read from.

Exit code is 0 on match, 1 on any drift. `record` runs each case twice and
refuses to store a reference that is not itself reproducible.

## Files

| file | role |
|---|---|
| `cases.json` | the reference runs (4 cases, ~30 s each) |
| `reference/*.json` | recorded metrics — the golden values |
| `run_cases.py` | driver: `record` / `check` |
| `run_golden.py` | runs one script deterministically, from a contained workspace |
| `make_workspace.py` | builds that workspace out of symlinks to the data |
| `determinism.py` | seeds every RNG the simulation touches |
| `compare.py` | standalone diff of one metrics pickle vs a reference |

## Two bugs this harness caught (both now fixed)

**1. The runs were not reproducible.** `load_environment_variables` sampled the
skill table with no `random_state`. `DataFrame.sample` draws from the *numpy*
global RNG, which the `random.seed(42)` in `constrain_veh` does not control, so
every run used a different set of firefighters. Two identical commands differed
on 10 of 21 metrics (`rupture_ff` 288 vs 305, `v_degraded` 11 vs 13, …).

`--constraint_factor_ff 1` was affected too: the selected *set* is the whole
table, but `sample` returns it in a shuffled *order*, and roles are assigned by
scanning firefighters in order.

Fixed in Étape 1: the loader takes an explicit `seed` (default 42), threaded
through from `--seed` on both simulation scripts. `determinism.py` no longer
patches `sample` — the golden references still match without it, which is the
evidence the fix is real and not propped up by the harness.

**2. Runs wrote outside their working directory.** The scripts navigated with
`os.chdir` and saved to `../Plots` relative to the data folder they last
entered, so a run launched from the wrong place dropped its results into the
*source data* tree. Fixed in Étape 1 by `paths.py`; `run_golden.py` sets
`RL_DATA_ROOT` to the workspace so outputs stay contained.

## Coverage

Four cases cover `simulation_start.py` (the heuristic baseline) and one covers
`agent_run_explainable.py` (FQF). Both sides of the comparison the refactor is
about to merge are therefore pinned.

The agent case runs in `--train` mode from a fresh seeded init rather than from a
checkpoint: no checkpoint exists for the current agent API. Weights are only
saved every 10000 interventions, so a 400-intervention run writes none — the run
does not mutate anything on disk.

### The references are device-specific

`fixtures/hyper_params_fqf.json` sets `device: cuda`. On this machine that is
~5x faster than CPU (52 s vs 4 min 18 s for the agent case) and reproducible run
to run, with `CUBLAS_WORKSPACE_CONFIG` and the deterministic-algorithm flags set
in `determinism.py`.

**Floating-point accumulation order differs between backends, so a reference
recorded on GPU will not match a CPU run** — about half the agent metrics move
(`rupture_ff` 1349 on CPU vs 1327 on GPU). The four heuristic cases are
unaffected either way: they never touch torch.

So on a machine with a different GPU, or with no GPU, `check` will report a
regression on `agent_fqf_train` that is not one. Re-record in that case:

```bash
python tools/golden/run_cases.py record --data-root "$RL_DATA_ROOT"
```

and compare *within* a machine, not across. Set `device: cpu` in the fixture if
you need a reference that is portable rather than fast.

## Known issues this harness had to work around

`agent_run_explainable.py` accepts `--save_metrics_as` but **never writes it**,
so there is no artifact to compare; `run_golden.py --capture-metrics` pickles
`dic_indic` out of the script's globals instead. Giving that flag a real
implementation belongs with the rewrite of the script, not here.

Windows shorter than ~60 interventions crash in `reinforcement_returning` (a
lookup runs past the end of the sliced `df_pc`), so `--end` must stay comfortably
above that. That is a latent bug in the simulation, worth fixing on its own.

The stale `Data/hyper_params.json` in the data tree cannot instantiate the
current `FQFAgent`: it carries `AM`/`N`, renamed to `am`/`n_quantiles`. The
fixture here is the corrected version.
