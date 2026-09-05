#!/bin/bash
# Réplicats restants : reprise de det_p10_s303 (interrompu à 92 %) puis det_p12.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

echo "########## DÉBUT $(date +%H:%M) ##########"

# --- 1. reprise du run interrompu -----------------------------------
TAG=det_p10_s303
echo "[$(date +%H:%M)] >>> REPRISE $TAG (checkpoint à ~490k/531k)"
if env -u PYTHONPATH "$PY" agent_run_explainable.py --train --resume \
    --model_name agent_${TAG}.pt --agent_model fqf \
    --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
    --dataset df_pc_fake_10y_p10_det_rs.pkl --start 1 --end 531168 \
    --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
    --checkpoint_every 10000 --seed 303 \
    --save_metrics_as metrics_agent_${TAG}_train \
    >> "$LOGS/train_${TAG}.log" 2>&1; then
  echo "[$(date +%H:%M)] <<< entraînement $TAG OK"
  if env -u PYTHONPATH "$PY" agent_run_explainable.py \
      --model_name agent_${TAG}.pt --agent_model fqf \
      --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
      --dataset df_pc_real_det_rs.pkl --start 1 --end 53088 \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --seed 303 --save_metrics_as metrics_agent_${TAG}_test \
      > "$LOGS/eval_${TAG}.log" 2>&1; then
    env -u PYTHONPATH "$PY" -c "
import pickle; d=pickle.load(open('run_full/Plots/metrics_agent_${TAG}_test.pkl','rb'))
print(f'[$(date +%H:%M)] <<< ${TAG} : rupture_ff={d[\"rupture_ff\"]}   (publié 3942 / best 4057)')"
  else
    echo "[$(date +%H:%M)] !!! ÉCHEC évaluation $TAG"; tail -20 "$LOGS/eval_${TAG}.log"
  fi
else
  echo "[$(date +%H:%M)] !!! ÉCHEC reprise $TAG — voir $LOGS/train_${TAG}.log"
  tail -20 "$LOGS/train_${TAG}.log"
fi

# --- 2. dernière configuration --------------------------------------
bash tools/seed_replicates.sh det_p12 101 202 303 || echo "!!! det_p12 interrompu"

echo "########## TOUT TERMINÉ $(date +%H:%M) ##########"
