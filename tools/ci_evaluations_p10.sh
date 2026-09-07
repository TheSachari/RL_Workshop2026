#!/bin/bash
# Évaluations p=1,0, pour porter les trois nouvelles métriques et la
# ventilation par compétence sur ces configurations.
#
# Plan différent de p=1,2 : le test porte sur le flux RÉEL, unique par nature.
# Seule la graine varie donc (4 graines), et `best`, déterministe, n'a qu'une
# exécution par régime.
#
# 2 régimes x (1 best + 4 random + 4 agent) = 18 évaluations.
set -u

REPO="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/RL_Workshop2026"
PY="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

N=53088
JOBS_BASE=${JOBS_BASE:-4}
JOBS_AGENT=${JOBS_AGENT:-2}

dataset() { [ "$1" = "prob" ] && echo df_pc_real_rs.pkl || echo df_pc_real_det_rs.pkl; }

model() {  # model <regime> <graine>
  local reg=$1 s=$2
  case "$reg:$s" in
    prob:41)  echo agent_10y_p10_rupture_ff.pt ;;
    prob:101) echo agent_10y_p10_s101.pt ;;
    prob:202) echo agent_10y_p10_s202.pt ;;
    prob:303) echo agent_prob_p10_s303.pt ;;
    det:41)   echo agent_10y_p10_det.pt ;;
    det:*)    echo "agent_det_p10_s${s}.pt" ;;
  esac
}

done_ok() { grep -q "gaspillage" "$1" 2>/dev/null; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do wait -n 2>/dev/null || sleep 5; done; }

echo "########## ÉVALUATIONS p=1,0 $(date +%H:%M) ##########"

# ---------- Phase 1 : baselines ----------
echo "--- phase 1 : baselines ---"
for reg in prob det; do
  DS=$(dataset "$reg")
  TAG="best_${reg}_p10"
  if done_ok "$LOGS/waste_${TAG}.log"; then echo "[skip] $TAG"; else
    throttle "$JOBS_BASE"
    ( env -u PYTHONPATH "$PY" tools/waste_baseline.py --dataset "$DS" \
        --end $N --is_best --tag "$TAG" > "$LOGS/waste_${TAG}.log" 2>&1 \
      && echo "[$(date +%H:%M)] OK  $TAG  $(grep -oP 'rupture_ff\s*:\s*\K\d+' "$LOGS/waste_${TAG}.log" | head -1)" \
      || echo "[$(date +%H:%M)] !!! ÉCHEC $TAG" ) &
  fi
  for s in 41 101 202 303; do
    TAG="${reg}_p10_s${s}"
    done_ok "$LOGS/rnd_${TAG}.log" && { echo "[skip] rnd_$TAG"; continue; }
    throttle "$JOBS_BASE"
    ( env -u PYTHONPATH "$PY" tools/waste_baseline.py --dataset "$DS" \
        --end $N --seed "$s" --tag "rnd_$TAG" > "$LOGS/rnd_${TAG}.log" 2>&1 \
      && echo "[$(date +%H:%M)] OK  rnd_$TAG  $(grep -oP 'rupture_ff\s*:\s*\K\d+' "$LOGS/rnd_${TAG}.log" | head -1)" \
      || echo "[$(date +%H:%M)] !!! ÉCHEC rnd_$TAG" ) &
  done
done
wait
echo "--- phase 1 terminée $(date +%H:%M) ---"

# ---------- Phase 2 : agent ----------
echo "--- phase 2 : agent ---"
for reg in prob det; do
  DS=$(dataset "$reg")
  for s in 41 101 202 303; do
    TAG="${reg}_p10_s${s}"
    done_ok "$LOGS/waste_${TAG}.log" && { echo "[skip] waste_$TAG"; continue; }
    MODEL=$(model "$reg" "$s")
    [ -f "run_full/SVG_model/$MODEL" ] || { echo "!! modèle absent : $MODEL"; continue; }
    throttle "$JOBS_AGENT"
    ( env -u PYTHONPATH "$PY" tools/waste_agent.py -- \
        --model_name "$MODEL" --agent_model fqf \
        --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
        --dataset "$DS" --start 1 --end $N \
        --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
        --save_metrics_as "waste_${TAG}" > "$LOGS/waste_${TAG}.log" 2>&1 \
      && echo "[$(date +%H:%M)] OK  waste_$TAG  $(grep -oP 'gaspillage\s*:\s*\K\d+' "$LOGS/waste_${TAG}.log" | head -1)" \
      || echo "[$(date +%H:%M)] !!! ÉCHEC waste_$TAG" ) &
  done
done
wait
echo "########## TERMINÉ $(date +%H:%M) ##########"
