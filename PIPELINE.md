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
  --save_rare_as df_pc_fake_10y_rare_skills.pkl --rarity 10
```

~50 min for ten years, ~5 min for one. Same again for the test stream:

```bash
python explainability.py --dataset df_pc_fake_test_prob_p12.pkl \
  --from_dir environment --merge_into df_pc_fake_test_prob_p12.pkl \
  --save_as df_pc_fake_test_rare_skills_merged.pkl \
  --save_rare_as df_pc_fake_test_rare_skills.pkl --rarity 10
```

**This step is mandatory before training.** The agent unpacks 20 values per
event row, the 20th being `rare_skills_required`; a generated environment
carries 19. Skipping it fails on the first row with a tuple-unpacking error.

| parameter | value | why |
|---|---|---|
| `--rarity` | 10 | A skill is "rare" at a given hour when **at least one but fewer than 10** of the firefighters then on duty hold it. The lower bound matters: on a typical 262-strong roster, 50 of the 134 skills have no holder at all, and counting those as rare made the flag fire on 87% of skills at `--rarity 80` — and still 61% at `--rarity 5`. Excluding them, 10 flags ~32% and 5 flags ~24%. A skill nobody on duty holds is a gap in the roster, not a trade-off an assignment can make. |
| `--top_k` | unset | Keep at most the K rarest skills of the hour, **ties included**, among those `--rarity` already admitted. A holder count is not comparable across hours — a day roster and a night one make "fewer than 10" mean different things — so the threshold alone lets the number of flagged skills swing with the shift; a rank does not. Ties are admitted whole (asking 20 may return 23) because cutting through equal counts would have to order them by skill index, and membership would then flip between neighbouring hours on nothing real — noise that the two-hour window in `get_related_rows_in_time` spreads and `rare_skills_given_up` reports per decision. Bites only where the threshold is loose: at `--rarity 80`, `--top_k 20` takes ~54% of skills down to ~15%; at `--rarity 10` it is inert. |
| `--from_dir` | `environment` | Where `--dataset` lives. Defaults to `sampled`, which is right for a raw sample but not for a generated stream. |
| `--merge_into` | same file | The stream that carries the skills. Rare skills are computed on departures only; the merge puts the RETURN rows back with empty arrays. Defaults to the *real* stream, so it must be set explicitly when working on a synthetic one. |

### Scoped rarity (decision-time, no pipeline step)

The column above pools all 34 stations, but the agent picks a firefighter from
one station's crew. Measured on the reference year, the pooled count **misses
89%** of the skills that are scarce where the choice is made: a station holds
~39 of the 134 skills with a median crew of 17, so a skill carried by 40 people
department-wide can still be its only one.

`explainability.rare_skills_for_step(st)` computes the two scoped levels at
decision time instead — no precompute, because the scope depends on the step:
`current_station` is known only once the departure has walked down the PDD, and
a reinforcement's sender is chosen when it is sent.

| level | rule | why |
|---|---|---|
| `LOCAL_RARITY` | 2 | The **sole holder** on this station's shift. Assigning them to a subordinate role removes the skill for the shift; from two holders on, one remains. The only threshold here resting on a mechanism rather than a calibration — and the one `modelisation.tex` already describes ("le seul détenteur"). Flags ~9.9 skills per decision, 26% of those the station holds. |
| `NEIGH_RARITY` | 6 | Also scarce across the stations called next, so no neighbour can cover the gap. Splits the sole-holder skills ~48/52 into irreversible and absorbable — the other half are locally scarce but abundant next door, and the deployment order handles them. |

Both are **absolute, not ranked**. A `top_k` here would flatten the signal: the
number of irreversible skills *should* rise when the neighbourhood is thin.
Absolute thresholds proved stable anyway — 50%/47%/49% irreversible across a
2.2x range of neighbourhood sizes — because a scarce skill's holder count does
not grow with the pool.

The neighbourhood is always that of the station **losing** the firefighter,
ordered by the PDD when there is one and by distance otherwise. Reinforcements
take the second path: the crew leaves the sending station for the travel time
plus 20 minutes with no local incident, and senders (the 11 Z2/Z3 stations) have
thinner rosters than first-due stations (13 vs 17) and thinner neighbourhoods
(80 vs 116). Same thresholds, same split (47% vs 48%).

Recorded per decision in the log as `rare_skills_local`,
`rare_skills_irreversible` and `irreversible_spent`.

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
