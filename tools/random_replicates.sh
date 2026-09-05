#!/bin/bash
# Réplicats de la baseline `random` : 4 graines x 4 configurations.
# Seule `random` a une variance de graine : `best` prend le minimum de niveau
# de compétence et ne lit jamais le générateur (vérifié : sorties identiques
# bit à bit sur deux graines). La graine d'environnement reste 42 partout,
# pour que toutes les politiques voient le même flux d'interventions.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

run() {  # run <flux_test> <fin> <graine> <tag>
  local test=$1 end=$2 seed=$3 tag=$4
  echo "[$(date +%H:%M)] >>> $tag"
  if env -u PYTHONPATH "$PY" tools/waste_baseline.py \
      --dataset "$test" --end "$end" --seed "$seed" \
      --tag "$tag" > "$LOGS/rnd_${tag}.log" 2>&1; then
    grep -E "rupture_ff|gaspillage|exigées par" "$LOGS/rnd_${tag}.log" | tail -3
  else
    echo "[$(date +%H:%M)] !!! ÉCHEC $tag"; tail -15 "$LOGS/rnd_${tag}.log"
  fi
}

echo "########## RÉPLICATS RANDOM $(date +%H:%M) ##########"
for s in 41 101 202 303; do run df_pc_real_rs.pkl          53088 $s "prob_p10_s${s}"; done
for s in 41 101 202 303; do run df_pc_fake_test_rs_v2.pkl  63696 $s "prob_p12_s${s}"; done
for s in 41 101 202 303; do run df_pc_real_det_rs.pkl      53088 $s "det_p10_s${s}";  done
for s in 41 101 202 303; do run df_pc_fake_test_det_rs.pkl 63696 $s "det_p12_s${s}";  done
echo "########## TERMINÉ $(date +%H:%M) ##########"
