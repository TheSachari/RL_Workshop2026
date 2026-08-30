#!/bin/bash
# Réplicats de graine d'entraînement — configuration probabiliste p=1,0.
# Séquentiel : un seul entraînement à la fois (contrainte mémoire).
# `--seed` pilote l'initialisation des poids autant que les tirages de
# l'environnement : la config de base suffit, une seule par configuration.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

END_TRAIN=531168
END_REAL=53088

for S in "$@"; do
  echo "[$(date +%H:%M)] >>> ENTRAÎNEMENT graine $S (prob p1.0)"
  if ! env -u PYTHONPATH "$PY" agent_run_explainable.py --train \
      --model_name agent_10y_p10_s${S}.pt --agent_model fqf \
      --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
      --dataset df_pc_fake_10y_p10_rs.pkl --start 1 --end "$END_TRAIN" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --checkpoint_every 10000 --seed "$S" \
      --save_metrics_as metrics_agent_p10_s${S}_train \
      > "$LOGS/train_p10_s${S}.log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC entraînement $S"; tail -25 "$LOGS/train_p10_s${S}.log"; exit 1
  fi
  echo "[$(date +%H:%M)] <<< entraînement $S OK"

  echo "[$(date +%H:%M)] >>> ÉVALUATION graine $S (flux réel)"
  if ! env -u PYTHONPATH "$PY" agent_run_explainable.py \
      --model_name agent_10y_p10_s${S}.pt --agent_model fqf \
      --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
      --dataset df_pc_real_rs.pkl --start 1 --end "$END_REAL" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --seed "$S" --save_metrics_as metrics_agent_p10_s${S}_test \
      > "$LOGS/eval_p10_s${S}.log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC évaluation $S"; tail -25 "$LOGS/eval_p10_s${S}.log"; exit 1
  fi
  echo "[$(date +%H:%M)] <<< évaluation $S OK"
  env -u PYTHONPATH "$PY" -c "
import pickle; d=pickle.load(open('run_full/Plots/metrics_agent_p10_s${S}_test.pkl','rb'))
print(f'    graine ${S} : rupture_ff={d[\"rupture_ff\"]}  (référence agent 6742, best 7544)')"
done
echo "[$(date +%H:%M)] ===== TERMINE ====="
