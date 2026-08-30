#!/bin/bash
# Pipeline déterministe complet : environnements -> rare_skills -> trainings -> évals.
# Séquentiel : un seul run lourd à la fois (contrainte mémoire du projet).
set -u

REPO="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/RL_Workshop2026"
PY="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

run() {  # run <label> <logfile> <cmd...>
  local label=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] >>> $label"
  if ! env -u PYTHONPATH "$@" > "$log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC $label — voir $log"; tail -20 "$log"; exit 1
  fi
  if grep -qi traceback "$log"; then
    echo "[$(date +%H:%M)] !!! TRACEBACK $label — voir $log"; grep -A5 -i traceback "$log" | head -20; exit 1
  fi
  echo "[$(date +%H:%M)] <<< $label OK"
}

# --- 1. environnement 10 ans p1.2 déterministe (le p1.0 tourne déjà à côté) ---
while pgrep -f "[g]enerate_environment.py" >/dev/null; do sleep 30; done
run "env 10y p1.2 det" "$LOGS/gen_det_10y_p12.log" \
  "$PY" generate_environment.py \
  --sample_list df_fake_y01.pkl df_fake_y02.pkl df_fake_y03.pkl df_fake_y04.pkl \
                df_fake_y05.pkl df_fake_y06.pkl df_fake_y07.pkl df_fake_y08.pkl \
                df_fake_y09.pkl df_fake_y10.pkl \
  --start_year 2018 --save_as df_pc_fake_10y_p12_det.pkl

# --- 2. rare_skills sur les quatre flux déterministes ---
for pair in \
  "df_pc_fake_10y_p10_det.pkl:df_pc_fake_10y_p10_det_rs.pkl" \
  "df_pc_fake_10y_p12_det.pkl:df_pc_fake_10y_p12_det_rs.pkl" \
  "df_pc_fake_test_det.pkl:df_pc_fake_test_det_rs.pkl" \
  "df_pc_real.pkl:df_pc_real_det_rs.pkl"
do
  src="${pair%%:*}"; dst="${pair##*:}"
  run "rare_skills $src" "$LOGS/rs_${dst%.pkl}.log" \
    "$PY" explainability.py --dataset "$src" --from_dir environment \
    --merge_into "$src" --save_as "$dst" \
    --save_rare_as "rare_${dst}" --rarity 10
done

# --- 3. bornes réelles (nb d'interventions) de chaque flux ---
read -r END_TRAIN_P10 END_TRAIN_P12 END_TEST_P12 END_REAL <<< "$(
  "$PY" - <<'EOF'
import pandas as pd
def n(f):
    d = pd.read_pickle(f"run_full/Data_environment/{f}")
    return len(d) // 2
print(n("df_pc_fake_10y_p10_det_rs.pkl"), n("df_pc_fake_10y_p12_det_rs.pkl"),
      n("df_pc_fake_test_det_rs.pkl"), n("df_pc_real_det_rs.pkl"))
EOF
)"
echo "bornes: train p10=$END_TRAIN_P10 train p12=$END_TRAIN_P12 test p12=$END_TEST_P12 real=$END_REAL"

# --- 4. trainings ---
run "train agent p1.0 det" "$LOGS/train_det_p10.log" \
  "$PY" agent_run_explainable.py --train \
  --model_name agent_10y_p10_det.pt --agent_model fqf \
  --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_10y_p10_det_rs.pkl --start 1 --end "$END_TRAIN_P10" \
  --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --checkpoint_every 10000 --save_metrics_as metrics_agent_det_p10_train

run "train agent p1.2 det" "$LOGS/train_det_p12.log" \
  "$PY" agent_run_explainable.py --train \
  --model_name agent_10y_p12_det.pt --agent_model fqf \
  --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_10y_p12_det_rs.pkl --start 1 --end "$END_TRAIN_P12" \
  --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --checkpoint_every 10000 --save_metrics_as metrics_agent_det_p12_train

# --- 5. évals agent, avec ventilation du gaspillage ---
run "eval agent p1.0 det" "$LOGS/eval_det_p10.log" \
  "$PY" tools/waste_agent.py -- \
  --model_name agent_10y_p10_det.pt --agent_model fqf \
  --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_real_det_rs.pkl --start 1 --end "$END_REAL" \
  --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --save_metrics_as metrics_agent_det_p10_test

run "eval agent p1.2 det" "$LOGS/eval_det_p12.log" \
  "$PY" tools/waste_agent.py -- \
  --model_name agent_10y_p12_det.pt --agent_model fqf \
  --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_test_det_rs.pkl --start 1 --end "$END_TEST_P12" \
  --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --save_metrics_as metrics_agent_det_p12_test

# --- 6. baselines, avec ventilation du gaspillage ---
run "best det p1.0"   "$LOGS/base_det_best_p10.log" \
  "$PY" tools/waste_baseline.py --dataset df_pc_real_det_rs.pkl --end "$END_REAL" \
  --is_best --tag best_det_p10
run "random det p1.0" "$LOGS/base_det_random_p10.log" \
  "$PY" tools/waste_baseline.py --dataset df_pc_real_det_rs.pkl --end "$END_REAL" \
  --tag random_det_p10
run "best det p1.2"   "$LOGS/base_det_best_p12.log" \
  "$PY" tools/waste_baseline.py --dataset df_pc_fake_test_det_rs.pkl --end "$END_TEST_P12" \
  --is_best --tag best_det_p12
run "random det p1.2" "$LOGS/base_det_random_p12.log" \
  "$PY" tools/waste_baseline.py --dataset df_pc_fake_test_det_rs.pkl --end "$END_TEST_P12" \
  --tag random_det_p12

# --- 7. complément manquant : agent p1.2 probabiliste ---
run "eval agent p1.2 prob (gaspillage)" "$LOGS/eval_prob_p12_waste.log" \
  "$PY" tools/waste_agent.py -- \
  --model_name agent_10y_p12_rupture_ff.pt --agent_model fqf \
  --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
  --dataset df_pc_fake_test_rs_v2.pkl --start 1 --end 63696 \
  --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
  --save_metrics_as tmp_waste_agent_p12

echo "[$(date +%H:%M)] ===== PIPELINE TERMINÉ ====="
