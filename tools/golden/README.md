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

## Two things this harness works around

**1. The runs were not reproducible.** `load_environment_variables` calls

```python
df_skills = df_skills.sample(len(df_skills) // constraint_factor_ff)
```

([`collective_functions.py:423`](../../collective_functions.py)) with no
`random_state`. `DataFrame.sample` draws from the *numpy* global RNG, which the
`random.seed(42)` in `constrain_veh` does not control, so every run used a
different set of firefighters. Two identical commands differed on 10 of 21
metrics (`rupture_ff` 288 vs 305, `v_degraded` 11 vs 13, …).

`--constraint_factor_ff 1` is affected too: the selected *set* is the whole
table, but `sample` returns it in a shuffled *order*, and roles are assigned by
scanning firefighters in order.

`determinism.py` seeds this from the outside, leaving repo sources untouched at
this stage. **When the refactor threads an explicit seed through the loader, that
shim should be deleted** and the fix made in the loader itself.

**2. Runs write outside their working directory.** The scripts navigate with
`os.chdir` and save to `../Plots` relative to the data folder they last entered —
so a run launched from the wrong place drops its results into the *source data*
tree. `make_workspace.py` therefore builds `Data_environment/` as a real
directory of per-file symlinks (not a symlink to the folder), which keeps `..`
inside the workspace.

## Coverage

Four cases cover `simulation_start.py` (the heuristic baseline) and one covers
`agent_run_explainable.py` (FQF). Both sides of the comparison the refactor is
about to merge are therefore pinned.

The agent case runs in `--train` mode from a fresh seeded init rather than from a
checkpoint: no checkpoint exists for the current agent API. Weights are only
saved every 10000 interventions, so a 400-intervention run writes none — the run
does not mutate anything on disk. `fixtures/hyper_params_fqf.json` sets
`device: cpu`, since GPU kernels are not bit-reproducible across machines.

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
