#!/bin/bash
# Ventilation gaspillées/justifiées pour les 12 réplicats de graine.
# `waste_agent.py` enveloppe l'évaluation standard : même chemin de code,
# avec en plus la décomposition de irreversible_spent.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

run() {  # run <modèle> <flux_test> <fin> <tag>
  local model=$1 test=$2 end=$3 tag=$4
  if [ ! -f "run_full/SVG_model/$model" ]; then
    echo "[$(date +%H:%M)] !! modèle absent : $model"; return
  fi
  echo "[$(date +%H:%M)] >>> $tag"
  if env -u PYTHONPATH "$PY" tools/waste_agent.py -- \
      --model_name "$model" --agent_model fqf \
      --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
      --dataset "$test" --start 1 --end "$end" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --save_metrics_as waste_${tag} > "$LOGS/waste_${tag}.log" 2>&1; then
    grep -E "exigées|gaspillage|total" "$LOGS/waste_${tag}.log" | tail -3
  else
    echo "[$(date +%H:%M)] !!! ÉCHEC $tag"; tail -15 "$LOGS/waste_${tag}.log"
  fi
}

echo "########## VENTILATION DES RÉPLICATS $(date +%H:%M) ##########"

# --- probabiliste p1.0 : test sur flux réel
for s in 101 202; do run "agent_10y_p10_s${s}.pt"  df_pc_real_rs.pkl         53088 "prob_p10_s${s}"; done
run "agent_prob_p10_s303.pt" df_pc_real_rs.pkl 53088 "prob_p10_s303"

# --- probabiliste p1.2 : test sur flux synthétique
for s in 101 202 303; do run "agent_prob_p12_s${s}.pt" df_pc_fake_test_rs_v2.pkl 63696 "prob_p12_s${s}"; done

# --- déterministe p1.0
for s in 101 202 303; do run "agent_det_p10_s${s}.pt"  df_pc_real_det_rs.pkl      53088 "det_p10_s${s}"; done

# --- déterministe p1.2
for s in 101 202 303; do run "agent_det_p12_s${s}.pt"  df_pc_fake_test_det_rs.pkl 63696 "det_p12_s${s}"; done

echo "########## TERMINÉ $(date +%H:%M) ##########"
