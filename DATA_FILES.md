# Minimum data files

Everything the pipeline needs that it cannot produce itself. The list was
derived by tracing the reads in each script, then checked by building a tree
containing only these files and running the whole chain against it — preprocess,
train, sample, generate_environment, explainability, the heuristic baseline and
the agent all complete from this set and nothing else.

Everything not listed here is **generated**. A fresh clone plus these inputs is
enough; the other six folders are created on first run.

```
$RL_DATA_ROOT/
├── Data/                  <- 14 raw files + Planning/ + one hyper-parameter JSON
│   └── Planning/          <- 12 monthly CSVs
└── Reward_weights/        <- 1 JSON minimum
```

---

## `Data/` — raw inputs (14 files)

| file | read by | what it is |
|---|---|---|
| `inters.csv` | preprocess | Interventions: timestamps, coordinates, incident type. The source of truth — one calendar year. |
| `sorties.csv` | preprocess | Vehicle departures per intervention. |
| `materiel.csv` | preprocess, generate_environment | Vehicle inventory and station assignment. |
| `pdd.geojson` | preprocess, sample, generate_environment | Station coverage polygons (*plus proche destination*). Drives the nearest-station ordering. |
| `X-Y-lieu.csv` | generate_environment | Coordinates to place names. |
| `dbo.LIEU.csv` | generate_environment | Place reference table. |
| `dbo.COMMUNE.csv` | generate_environment | Municipality reference. |
| `dbo.NOM_COMMUNE.csv` | generate_environment | Municipality names. |
| `dbo.SECTEUR.csv` | generate_environment | Sector reference. |
| `firestations.csv` | generate_environment | Station list with coordinates. |
| `comp.csv` | generate_environment | Firefighter skills and their validity windows. |
| `roles_competences.csv` | generate_environment | Role → required-skill mapping. |
| `responses_by_incident.csv` | generate_environment | Vehicles historically sent per incident type — the basis for `--prob_dep`. |
| `df_vehicles_history.csv` | generate_environment | Vehicle availability history. |

Sizes on the reference dataset: ~125 MB total, dominated by `dbo.LIEU.csv`
(13 MB), `inters.csv` (13 MB), `materiel.csv` (15 MB), `pdd.geojson` (19 MB).

## `Data/Planning/` — 12 files

`planning_<year>_<month>.csv`, one per month (`planning_2018_01.csv` …
`planning_2018_012.csv` — note the inconsistent zero-padding, which is how they
ship). ~189 MB total.

Read by `generate_environment.py`, which folds them into `planning.pkl`: who is
on duty at each station, for each month/day/hour. Every month must be present —
a missing one leaves holes the simulation will index into.

## `Data/*.json` — agent hyper-parameters (1 file)

Named by `agent_run_explainable.py --hyper_params`. A working FQF config is
committed at `tools/golden/fixtures/hyper_params_fqf.json`.

The `hyper_params.json` shipped in the data tree **cannot instantiate the
current agent**: it carries `AM` and `N`, renamed to `am` and `n_quantiles`.
Either use the fixture or rename those two keys.

## `Reward_weights/*.json` — at least 1 file

Named by `agent_run_explainable.py --reward_weights`; picks which failure the
reward punishes. Six ship with the project (`rw_rupture_ff.json`,
`rw_v_degraded.json`, `rw_v1_not_sent_from_s1.json`,
`rw_v3_not_sent_from_s3.json`, `rw_v_not_found_in_last_station.json`,
`rw_z1_VSAV_sent.json`). Only the one being used is needed.

---

## Generated — do not ship these

Produced by the pipeline, in this order:

| folder | contents | produced by |
|---|---|---|
| `Data_preprocessed/` | `df_real.pkl`, `df_prob_dep.pkl`, `df_rank_incident.pkl` | preprocess |
| `Data_trained/` | `df_real.pkl`, `df_quantile.pkl`, `dataset.pkl`, `normalizer_ddpm.pkl` | train |
| `SVG_model/` | `model_<name>_{diffusion,ema}.pt`, agent checkpoints | train, agent |
| `Data_sampled/` | one pickle per sampled year | sample |
| `Data_environment/` | `df_pc_*.pkl`, `planning.pkl`, `df_skills.pkl`, `df_stations.pkl`, `df_v.pkl`, `df_roles.pkl`, `df_vehicles_history.pkl` | generate_environment, explainability |
| `Plots/` | metrics pickles | simulation, agent |

Note `Data_environment/` holds both generated *streams* and generated
*reference tables* — the tables come from the raw CSVs above, so they are
rebuilt whenever `generate_environment.py` runs, whatever it was asked for.

## Present in the reference tree but never read

Leftovers from earlier work; the pipeline runs without them:

`admin-departement.{shp,dbf,prj,shx}`, `pdd_back.geojson`, `rank_incident.csv`,
`roles_competences_associees.csv`, `df_pc_pons.csv`, `df_oversampled.csv`,
`df_fake.csv`, `df_quantile.csv`, `df_real.csv`, `df_pc_real.csv`,
`df_pc_fake.csv`, `comp_2018.csv`, `sorties_2018.csv`, `materiel_2018.csv`,
`inters.csv` duplicates, and the `.pkl`/`.pt` copies sitting directly in
`Data/` rather than in their generated folders.

`hyper_params_dt.json`, `dt_hyper_params.json` and `hyper_params_pomo.json` are
only needed for the DT and POMO agents.

---

## Checking a tree before a long run

```bash
cd "$RL_DATA_ROOT"
for f in inters.csv sorties.csv materiel.csv pdd.geojson X-Y-lieu.csv \
         dbo.LIEU.csv dbo.COMMUNE.csv dbo.NOM_COMMUNE.csv dbo.SECTEUR.csv \
         firestations.csv comp.csv roles_competences.csv \
         responses_by_incident.csv df_vehicles_history.csv; do
  [ -f "Data/$f" ] || echo "MANQUE Data/$f"
done
n=$(ls Data/Planning/planning_*.csv 2>/dev/null | wc -l)
[ "$n" -eq 12 ] || echo "Planning: $n fichiers au lieu de 12"
ls Reward_weights/*.json >/dev/null 2>&1 || echo "MANQUE Reward_weights/*.json"
```

Silence means the tree is complete.
