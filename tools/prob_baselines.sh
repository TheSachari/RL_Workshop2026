#!/bin/bash
# Ventilation justifiées/gaspillées des 4 baselines probabilistes.
# Logs écrits dans run_full/logs/ pour conserver la piste d'audit.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

run() {
  local tag=$1; shift
  echo "[$(date +%H:%M)] >>> $tag"
  if env -u PYTHONPATH "$PY" tools/waste_baseline.py "$@" --tag "$tag" \
       > "$LOGS/waste_${tag}.log" 2>&1; then
    echo "[$(date +%H:%M)] <<< $tag OK"
    grep -E "rupture_ff|irreversible_spent \(total\)|exigées|gaspillage" "$LOGS/waste_${tag}.log"
  else
    echo "[$(date +%H:%M)] !!! ECHEC $tag"; tail -20 "$LOGS/waste_${tag}.log"; exit 1
  fi
}

run best_prob_real   --dataset df_pc_real_rs.pkl         --end 53088 --is_best
run random_prob_real --dataset df_pc_real_rs.pkl         --end 53088
run best_prob_fake   --dataset df_pc_fake_test_rs_v2.pkl --end 63696 --is_best
run random_prob_fake --dataset df_pc_fake_test_rs_v2.pkl --end 63696
echo "===== TERMINE ====="
