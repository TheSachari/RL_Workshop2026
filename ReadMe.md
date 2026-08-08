# Setup

```bash
pip install -r requirements.txt
```

The commands below are the reference for each script taken on its own. To run
the whole chain — raw CSVs to a trained agent, in order, with the reasoning
behind each parameter and the measured runtimes — see
**[PIPELINE.md](PIPELINE.md)**.

Scripts resolve their paths through `paths.py`, so they can be run from any
directory. By default the data folders are expected next to the repo; point
`RL_DATA_ROOT` elsewhere to keep the data outside it:

```bash
export RL_DATA_ROOT=/path/to/data
```

Both simulation scripts take `--seed` (default 42), which fixes the vehicle and
firefighter downsampling draws. The same seed always yields the same
environment; vary it across replicates.

Two levels of checking, both worth running before and after a change:

```bash
# unit tests on the pure helpers — ~1s, no data needed
python -m pytest

# whole-run behaviour against recorded references — ~3min, needs the data
python tools/golden/run_cases.py check --data-root "$RL_DATA_ROOT"
```

If pytest fails during collection with `No module named 'yaml'`, a system ROS
install is injecting its plugins; run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m
pytest` instead.

See [tools/golden/README.md](tools/golden/README.md).

# Preprocessing

```python3 preprocess.py --inters inters.csv --sorties sorties.csv --pdd pdd.geojson --materiel materiel.csv```

Running it for the first time will create the following folders :

["Data_preprocessed", "Data_trained", "Data_sampled", "Data_environment", "SVG_model", "Plots", "Reward_weights"]

Preprocesses the raw data from the ```./Data``` folder.

# Train

```train.py --lr 0.0025 --layers 1024 --num_timesteps 1000 --is_y_cond --save_as agent_name ```

Trains the model. 

# Sample

```sample.py --load_as agent_name --save_sample_as df_fake_sample_p12.pkl --to_keep 40 --value_span 100 --pressure 1.2 ```

Samples new interventions from a trained model. 

# Generate environment

```generate_environment.py --sample_list df_fake_sample_p12.pkl  --prob_dep --save_as df_pc_fake_10y_prob_p12.pkl ```

# Explainability

```explainability.py -- dataset df_fake_sample_p12.pkl --rarity 80 ```

Precomputes rare skills for explainability.

# Simulation


```simulation_start.py --dataset df_pc_real_prob.pkl --start 1 --end 202 --is_best --constraint_factor_veh 1 --constraint_factor_ff 3 --save_metrics_as metrics_name```

Runs the simulation.

# Agent

```agent_run_explainable.py --model_name agent_name --agent_model fqf --hyper_params hyper_params.json --reward_weights rw_rupture_ff.json --dataset df_pc_prob_rare_skills_merged.pkl --start 1 --end 63696 --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 --save_metrics_as metrics --train```

Trains an agent with ```--train```. \
Loads training weights in training mode with ```--load```. \
Otherwise, runs the agent ```agent_name``` in test mode.










