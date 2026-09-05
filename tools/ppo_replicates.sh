#!/bin/bash
# Réplicats PPO, calqués sur `seed_replicates.sh` : mêmes flux, mêmes bornes,
# même récompense, même protocole train puis eval. Seuls changent
# `--agent_model` et le fichier d'hyper-paramètres, pour que la comparaison
# avec les chiffres FQF publiés porte sur l'algorithme et rien d'autre.
#
# Séquentiel : un seul entraînement à la fois (contrainte mémoire).
# `--seed` pilote l'initialisation des poids autant que les tirages de
# l'environnement : une seule config d'hyper-paramètres suffit.
#
# Usage : tools/ppo_replicates.sh <config> <graine> [graine...]
#   config ∈ prob_p10 | prob_p12 | det_p10 | det_p12
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

CFG="${1:?config manquante : prob_p10|prob_p12|det_p10|det_p12}"; shift

case "$CFG" in
  prob_p10) TRAIN=df_pc_fake_10y_p10_rs.pkl;     N_TRAIN=531168
            TEST=df_pc_real_rs.pkl;              N_TEST=53088  ; REF="fqf 6742 / best 7544" ;;
  prob_p12) TRAIN=df_pc_fake_10y_rs_v2.pkl;      N_TRAIN=637331
            TEST=df_pc_fake_test_rs_v2.pkl;      N_TEST=63696  ; REF="fqf 8901 / best 9989" ;;
  det_p10)  TRAIN=df_pc_fake_10y_p10_det_rs.pkl; N_TRAIN=531168
            TEST=df_pc_real_det_rs.pkl;          N_TEST=53088  ; REF="fqf 3942 / best 4057" ;;
  det_p12)  TRAIN=df_pc_fake_10y_p12_det_rs.pkl; N_TRAIN=637332
            TEST=df_pc_fake_test_det_rs.pkl;     N_TEST=63696  ; REF="fqf 5076 / best 5342" ;;
  *) echo "config inconnue : $CFG"; exit 1 ;;
esac

echo "=== PPO $CFG : entraînement sur $TRAIN, test sur $TEST (référence $REF)"

for S in "$@"; do
  TAG="ppo_${CFG}_s${S}"
  echo "[$(date +%H:%M)] >>> ENTRAÎNEMENT $TAG"
  if ! env -u PYTHONPATH "$PY" agent_run_explainable.py --train \
      --model_name agent_${TAG}.pt --agent_model ppo \
      --hyper_params hp_ppo_am.json --reward_weights rw_rupture_ff.json \
      --dataset "$TRAIN" --start 1 --end "$N_TRAIN" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --checkpoint_every 10000 --seed "$S" \
      --save_metrics_as metrics_agent_${TAG}_train \
      > "$LOGS/train_${TAG}.log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC entraînement $TAG"; tail -25 "$LOGS/train_${TAG}.log"; exit 1
  fi
  echo "[$(date +%H:%M)] <<< entraînement $TAG OK"

  echo "[$(date +%H:%M)] >>> ÉVALUATION $TAG"
  if ! env -u PYTHONPATH "$PY" agent_run_explainable.py \
      --model_name agent_${TAG}.pt --agent_model ppo \
      --hyper_params hp_ppo_am.json --reward_weights rw_rupture_ff.json \
      --dataset "$TEST" --start 1 --end "$N_TEST" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --seed "$S" --save_metrics_as metrics_agent_${TAG}_test \
      > "$LOGS/eval_${TAG}.log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC évaluation $TAG"; tail -25 "$LOGS/eval_${TAG}.log"; exit 1
  fi
  env -u PYTHONPATH "$PY" -c "
import pickle; d=pickle.load(open('run_full/Plots/metrics_agent_${TAG}_test.pkl','rb'))
print(f'[$(date +%H:%M)] <<< ${TAG} : rupture_ff={d[\"rupture_ff\"]}   (référence $REF)')"
done
echo "[$(date +%H:%M)] ===== PPO $CFG TERMINE ====="
