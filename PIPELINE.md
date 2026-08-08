# Running the full pipeline

From the raw CSVs to a trained agent, in order, with the reasoning behind each
parameter. The short form of every command is in [ReadMe.md](ReadMe.md); this
document is for actually running the thing end to end and knowing why the
numbers are what they are.

Every step writes into the data tree and is read by the next one, so the order
matters. Times below are measured on an RTX 4090 with 16 GB of RAM, on the
2018 Haute-Garonne dataset (53 088 real interventions).

```bash
export RL_DATA_ROOT=/path/to/data     # holds Data/, Data_preprocessed/, ...
```

`paths.py` resolves everything from that root, so the scripts run from any
directory. Without it they look next to the repo.

For what that tree has to contain before any of this runs — and what is
generated rather than shipped — see **[DATA_FILES.md](DATA_FILES.md)**.

---

## 0. Before and after any change

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest          # ~2 s, no data
python tools/golden/run_cases.py check --data-root "$RL_DATA_ROOT"   # ~3 min
```

The golden cases pin whole-run behaviour against recorded metrics. **Steps 4 and
onward overwrite files the golden cases read** (`df_pc_real_prob.pkl`,
`planning.pkl`, `df_skills.pkl`, `df_stations.pkl`, `df_v.pkl`, `df_roles.pkl`,
`df_vehicles_history.pkl`). Back them up first if you intend to run `check`
again afterwards:

```bash
mkdir -p "$RL_DATA_ROOT/Data_environment/.golden_backup"
cd "$RL_DATA_ROOT/Data_environment"
cp df_pc_real_prob.pkl planning.pkl df_skills.pkl df_stations.pkl \
   df_v.pkl df_roles.pkl df_vehicles_history.pkl \
   df_pc_prob_rare_skills_merged.pkl .golden_backup/
(cd .golden_backup && md5sum *.pkl > .manifest.md5)
```

Otherwise the references have to be re-recorded (`run_cases.py record`) against
the new environment, and comparisons with earlier runs stop being meaningful.

---

## 1. Preprocess — raw CSVs to a clean event table

```bash
python preprocess.py --inters inters.csv --sorties sorties.csv \
                     --pdd pdd.geojson --materiel materiel.csv
```

~2 min. Reads `Data/`, writes `Data_preprocessed/`. Creates the seven data
folders on first run.

The four inputs are the raw exports: interventions, vehicle departures, the
station coverage polygons (`pdd` = *plus proche destination*), and the vehicle
inventory. Filenames are passed rather than hardcoded because the yearly
exports carry the year in their name (`sorties_2018.csv` and friends).

The reference dataset covers **one calendar year** — the shipped `inters.csv`
runs from 2018-01-01 to 2018-12-31, 53 088 interventions. Everything downstream
inherits that: the generative model learns a single year's seasonality, and the
synthetic years are replays of it under a different label (see `--start_year`
in step 4).

**Check the output before going further.** `Duration` silently went to zero on
older pandas, and every later step depends on it:

```python
import pandas as pd
d = pd.read_pickle("Data_preprocessed/df_real.pkl")
print(d["Duration"].describe())   # expect mean ~87, min 11, max ~1160
```

A mean of 0 means the datetime cleaning failed; see the note on
`_clean_datetimes` in `preprocess.py`.

---

## 2. Train the generative model

```bash
python train.py --lr 0.0025 --layers 1024 --num_timesteps 1000 \
                --is_y_cond --save_as agent_name
```

~45 s on GPU. Reads `Data_preprocessed/`, writes `Data_trained/` and
`SVG_model/model_agent_name_{diffusion,ema}.pt`.

A tabular DDPM over 9 features (coordinates, duration, incident type, and the
sin/cos encodings of month/day/hour).

| parameter | value | why |
|---|---|---|
| `--num_timesteps` | 1000 | Diffusion steps. The standard TabDDPM setting; fewer degrades the tails of the duration distribution, more costs sampling time for no visible gain. |
| `--layers` | 1024 | Width of the two MLP layers. The feature space is small (9 inputs, 56 incident classes), so depth buys nothing — this is already generous. |
| `--lr` | 0.0025 | With `--epochs 20000` (the default) and batch 4096, this converges well inside the step budget. |
| `--is_y_cond` | on | Conditions generation on the incident type. Without it the model mixes incident classes and the sampled mix drifts from the real one. |
| `--save_as` | `agent_name` | Names the checkpoint; `sample.py --load_as` must match. |

`--epochs` is misleadingly named: it counts optimisation *steps*, not passes
over the data.

---

## 3. Sample synthetic years

One command per year. For a ten-year training set plus a held-out test year:

```bash
for i in 01 02 03 04 05 06 07 08 09 10; do
  python sample.py --load_as agent_name --save_sample_as df_fake_y$i.pkl \
                   --to_keep 40 --value_span 100 --pressure 1.2
done
python sample.py --load_as agent_name --save_sample_as df_fake_test.pkl \
                 --to_keep 40 --value_span 100 --pressure 1.2
```

~2 min each, so ~22 min for eleven. Writes `Data_sampled/`.

There is no `--seed`: sampling is stochastic, which is the point — the eleven
draws are independent years, not copies. Verify with `md5sum Data_sampled/*.pkl`.

| parameter | value | why |
|---|---|---|
| `--pressure` | 1.2 | Interventions to draw, as a multiple of the real year. 1.2 puts the system under 20 % more demand than 2018 actually saw — the scenario the agent is meant to handle. **1.0 reproduces the real volume.** |
| `--to_keep` | 40 | Harmonisation keeps the 40 highest-volume days when flattening the day-of-year distribution. The default 10 leaves visible spikes at this volume. |
| `--value_span` | 100 | Interventions permuted per harmonisation move. Larger smooths faster; 100 suits ~64 k interventions. |
| `--variability` | 0.02 (default) | Tolerated deviation from the target per-sector distribution. |

**`--pressure` decides your test set.** At 1.2 a synthetic year holds 63 696
interventions against 53 088 real ones — exactly the 1.2 ratio. Training on one
density and testing on another confounds the comparison, so:

- `--pressure 1.2` → test on a **sampled** year at the same pressure
  (`df_fake_test.pkl`, generated above);
- `--pressure 1.0` → test on the **real** stream (`df_pc_real_prob.pkl`).

---

## 4. Build the environments

Training stream, ten years chained into one continuous timeline:

```bash
python generate_environment.py --prob_dep \
  --sample_list df_fake_y01.pkl df_fake_y02.pkl df_fake_y03.pkl df_fake_y04.pkl \
                df_fake_y05.pkl df_fake_y06.pkl df_fake_y07.pkl df_fake_y08.pkl \
                df_fake_y09.pkl df_fake_y10.pkl \
  --save_as df_pc_fake_10y_prob_p12.pkl
```

~50 min (~5 min per year). Then the test stream:

```bash
python generate_environment.py --prob_dep --sample_list df_fake_test.pkl \
  --save_as df_pc_fake_test_prob_p12.pkl
```

~10 min. Both write `Data_environment/`.

This turns a table of interventions into an event stream: it assigns each one
its nearest stations, its zone, its required vehicles, and inserts a matching
RETURN event — hence output twice the input length.

| parameter | value | why |
|---|---|---|
| `--prob_dep` | on | Draws the required vehicles from the historical distribution for that incident type and area, instead of the single most frequent departure. Keeps the rare, heavy departures the agent has to cope with; without it the mix is deterministic and too easy. |
| `--sample_list` | 10 files | Concatenated in order, one calendar year each. `num_inter` continues across years, so the decade is one timeline rather than ten overlapping ones. |
| `--start_year` | current year | Calendar year stamped on the first generated year; each further sample takes the next one. **This only labels the timeline — it does not change what was sampled.** It does decide which years are leap, and therefore which get a 366th day. Pass `--start_year 2018` to align the labels with the year the source data actually covers. |
| `--save_as` | — | Output name in `Data_environment/`. |

Two things this step does that are worth knowing:

- **It regenerates the real stream too**, unconditionally — `df_pc_real_prob.pkl`
  is rewritten even when only `--sample_list` fake years are requested. This is
  what clobbers the golden inputs (see step 0).
- **Leap years get their 366th day.** The generative model only ever saw a
  365-day year, so a leap year would otherwise stop on 30 December. A day drawn
  from the turn of the year is replayed onto day 366 — the donor comes from late
  December or early January because activity is seasonal (~162 interventions/day
  in December against ~202 in June), so an arbitrary day would land a summer
  profile on a winter date. Which years are affected follows from
  `--start_year`: two of them in any ten-year span.

Sanity check:

```python
import pandas as pd
d = pd.read_pickle("Data_environment/df_pc_fake_10y_prob_p12.pkl")
dep = d[d["departure"] != {0: "RETURN"}]
print(len(d), dep["date"].dt.normalize().nunique())   # ~1274488, 3652 days
```

3652 = 8×365 + 2×366 — a ten-year span always contains two leap years. Anything
less means a calendar hole. The row count moves slightly with `--start_year`,
since the replayed leap days differ.

---

## 5. Rare skills — the agent's 20th column

```bash
python explainability.py --dataset df_pc_fake_10y_prob_p12.pkl \
  --from_dir environment --merge_into df_pc_fake_10y_prob_p12.pkl \
  --save_as df_pc_fake_10y_rare_skills_merged.pkl \
  --save_rare_as df_pc_fake_10y_rare_skills.pkl --rarity 80
```

~50 min for ten years, ~5 min for one. Same again for the test stream:

```bash
python explainability.py --dataset df_pc_fake_test_prob_p12.pkl \
  --from_dir environment --merge_into df_pc_fake_test_prob_p12.pkl \
  --save_as df_pc_fake_test_rare_skills_merged.pkl \
  --save_rare_as df_pc_fake_test_rare_skills.pkl --rarity 80
```

**This step is mandatory before training.** The agent unpacks 20 values per
event row, the 20th being `rare_skills_required`; a generated environment
carries 19. Skipping it fails on the first row with a tuple-unpacking error.

| parameter | value | why |
|---|---|---|
| `--rarity` | 80 | A skill is "rare" at a given hour when fewer than 80 of the firefighters then on duty hold it. Lower flags almost nothing; higher flags nearly everything and the signal stops discriminating. |
| `--from_dir` | `environment` | Where `--dataset` lives. Defaults to `sampled`, which is right for a raw sample but not for a generated stream. |
| `--merge_into` | same file | The stream that carries the skills. Rare skills are computed on departures only; the merge puts the RETURN rows back with empty arrays. Defaults to the *real* stream, so it must be set explicitly when working on a synthetic one. |

Check the width:

```python
d = pd.read_pickle("Data_environment/df_pc_fake_10y_rare_skills_merged.pkl")
print(len(next(d.itertuples(index=True, name=None))))   # must be 20
```

---

## 6. Heuristic baseline

```bash
python simulation_start.py --dataset df_pc_real_prob.pkl --start 1 --end 53088 \
  --is_best --constraint_factor_veh 1 --constraint_factor_ff 3 \
  --save_metrics_as metrics_heuristic
```

The comparison point for the agent: same environment, same constraints, rules
instead of a policy.

| parameter | value | why |
|---|---|---|
| `--is_best` | on | Picks the best-matching firefighter for each role. Omit it for a random feasible pick — the weaker baseline. |
| `--constraint_factor_veh` | 1 | Vehicles kept in the Z1 stations: 1 = all, 3 = a third. |
| `--constraint_factor_ff` | 3 | Firefighters kept: 3 keeps a third, which is where crewing starts to bite and the metrics separate. At 1 almost everything is satisfiable. |
| `--seed` | 42 (default) | Fixes both downsampling draws. Same seed, same environment; vary it across replicates. |

Windows shorter than ~60 interventions crash in `reinforcement_returning`, so
keep `--end` well above that.

---

## 7. Train the agent

```bash
python agent_run_explainable.py --train \
  --model_name agent_10y.pt --agent_model fqf \
  --hyper_params hyper_params.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_10y_rare_skills_merged.pkl \
  --start 1 --end 637244 --n_hours 2 --top_n 5 \
  --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --save_metrics_as metrics_agent_train
```

Then evaluate on the held-out year — no `--train`, so weights are loaded and
frozen:

```bash
python agent_run_explainable.py \
  --model_name agent_10y.pt --agent_model fqf \
  --hyper_params hyper_params.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_test_rare_skills_merged.pkl \
  --start 1 --end 63696 --n_hours 2 --top_n 5 \
  --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --save_metrics_as metrics_agent_test
```

| parameter | value | why |
|---|---|---|
| `--agent_model` | `fqf` | Fully-parameterised Quantile Function. `dqn`, `ppo` and `dt` also exist; FQF learns the whole return distribution, which suits a reward driven by rare, costly failures. |
| `--end` | 637244 / 63696 | The departure count of each stream — the full decade for training, the full held-out year for testing. |
| `--n_hours` | 2 | Look-ahead for upcoming rare skills. The horizon over which committing a scarce specialist now may cost a later intervention. |
| `--top_n` | 5 | Nearby stations considered in that look-ahead: the ones that could realistically cover the same call. |
| `--reward_weights` | `rw_rupture_ff.json` | Which failure the reward punishes. `rw_rupture_ff` targets crewing shortfalls; the other `rw_*.json` files target degraded departures, first-station misses, and so on. Pick the one matching the question being asked. |
| `--constraint_factor_ff` | 1 | Unconstrained here, unlike the baseline: the agent should first learn the task, then be re-run under constraint. |
| `--eps_start` | 1.0 (default) | Starts fully exploratory and decays to a 0.05 floor across the run, in ~23 steps keyed to `--end`. |

Weights are checkpointed every 10 000 interventions, so a long run survives an
interruption; resume with `--load`.

---

## Where it can go wrong

| symptom | cause |
|---|---|
| `Duration` all zeros after step 1 | datetime cleaning lost precision; check `_clean_datetimes` |
| `KeyError: 'duree'` in step 3 | column named `Duration` here, `duree` in `preprocess.py` |
| `ValueError: not enough values to unpack (expected 3, got 2)` | `act()` must return `potential_actions` alongside the choice |
| `TypeError: unexpected keyword argument 'AM'` | stale `hyper_params.json`: `AM`/`N` are now `am`/`n_quantiles` |
| `rwd_mean: nan` throughout training | no reward is being computed — the training loop is not running |
| golden `check` fails on 4 heuristic cases | step 4 overwrote `df_pc_real_prob.pkl`; restore from `.golden_backup/` |

## Total runtime

| step | time |
|---|---|
| 1. preprocess | ~2 min |
| 2. train DDPM | ~45 s |
| 3. sample × 11 | ~22 min |
| 4. environments (10 y + test) | ~60 min |
| 5. rare skills (10 y + test) | ~55 min |
| 6. baseline | ~5 min |
| **before training** | **~2 h 25** |
