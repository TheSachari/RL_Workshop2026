#!/bin/bash
# Évaluations manquantes pour mettre des IC sur toutes les métriques des
# tableaux, configurations p=1,2 (probabiliste et déterministe).
#
# Deux sources de variance croisées :
#   - le tirage du flux : 4 années de test (v2 existante + 02, 03, 04)
#   - la graine         : 4 graines pour `random` et pour l'agent
# `best` est déterministe (min du niveau de compétence, aucun tirage) : une
# seule exécution par année, son écart-type est nul par construction.
#
# On ne relance QUE ce qui manque. `random` et l'agent sur l'année v2 sont déjà
# mesurés ; `best` est repris partout, ses anciens logs datant d'avant le dump
# des métriques d'observation.
#
# PARALLÉLISME. Les runs sont indépendants : processus séparés, logs et .pkl
# distincts, et depuis le semis de `prob_departure` chaque tirage part d'un
# générateur propre — rien n'est partagé, donc rien ne dépend de l'ordre.
# Les tailles diffèrent d'un ordre de grandeur (baseline 0,5 Go / 2,7 min,
# agent 1,7 Go / 15 min), d'où deux pools distincts : la RAM est la contrainte,
# pas les 12 cœurs.
set -u

REPO="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/RL_Workshop2026"
PY="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

N=63696
JOBS_BASE=${JOBS_BASE:-4}     # 4 x 0,5 Go = 2 Go
JOBS_AGENT=${JOBS_AGENT:-2}   # 2 x 1,7 Go = 3,4 Go
# Chaque processus sature un cœur ; brider BLAS évite qu'ils se disputent les 12.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

dataset() {
  local reg=$1 y=$2
  if [ "$y" = "v2" ]; then
    [ "$reg" = "prob" ] && echo "df_pc_fake_test_rs_v2.pkl" || echo "df_pc_fake_test_det_rs.pkl"
  else
    [ "$reg" = "prob" ] && echo "df_pc_fake_test_p12_${y}_rs.pkl" \
                        || echo "df_pc_fake_test_p12_${y}_det_rs.pkl"
  fi
}
model() {
  local reg=$1 s=$2
  case "$reg:$s" in
    prob:41) echo agent_10y_p12_rupture_ff.pt ;;
    det:41)  echo agent_10y_p12_det.pt ;;
    prob:*)  echo "agent_prob_p12_s${s}.pt" ;;
    det:*)   echo "agent_det_p12_s${s}.pt" ;;
  esac
}
done_ok() { grep -q "gaspillage" "$1" 2>/dev/null; }

# Attend qu'il reste moins de $1 jobs du motif $2 avant de rendre la main.
throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do wait -n 2>/dev/null || sleep 5; done
}

echo "########## ÉVALUATIONS IC $(date +%H:%M) ##########"
echo "pools : $JOBS_BASE baselines, $JOBS_AGENT agents"

# ---------- Phase 1 : baselines (best + random), pool de $JOBS_BASE ----------
echo "--- phase 1 : baselines ---"
for reg in prob det; do
  for y in v2 02 03 04; do
    DS=$(dataset "$reg" "$y")
    [ -f "run_full/Data_environment/$DS" ] || { echo "!! flux absent : $DS"; continue; }

    TAG="best_${reg}_p12_${y}"
    if done_ok "$LOGS/waste_${TAG}.log"; then echo "[skip] $TAG"; else
      throttle "$JOBS_BASE"
      ( env -u PYTHONPATH "$PY" tools/waste_baseline.py --dataset "$DS" \
          --end $N --is_best --tag "$TAG" > "$LOGS/waste_${TAG}.log" 2>&1 \
        && echo "[$(date +%H:%M)] OK  $TAG  $(grep -oP 'rupture_ff\s*:\s*\K\d+' "$LOGS/waste_${TAG}.log" | head -1)" \
        || echo "[$(date +%H:%M)] !!! ÉCHEC $TAG" ) &
    fi

    for s in 41 101 202 303; do
      if [ "$y" = "v2" ] && done_ok "$LOGS/rnd_${reg}_p12_s${s}.log"; then
        echo "[skip] rnd_${reg}_p12_v2_s${s}"; continue; fi
      TAG="${reg}_p12_${y}_s${s}"
      done_ok "$LOGS/rnd_${TAG}.log" && { echo "[skip] rnd_$TAG"; continue; }
      throttle "$JOBS_BASE"
      ( env -u PYTHONPATH "$PY" tools/waste_baseline.py --dataset "$DS" \
          --end $N --seed "$s" --tag "rnd_$TAG" > "$LOGS/rnd_${TAG}.log" 2>&1 \
        && echo "[$(date +%H:%M)] OK  rnd_$TAG  $(grep -oP 'rupture_ff\s*:\s*\K\d+' "$LOGS/rnd_${TAG}.log" | head -1)" \
        || echo "[$(date +%H:%M)] !!! ÉCHEC rnd_$TAG" ) &
    done
  done
done
wait
echo "--- phase 1 terminée $(date +%H:%M) ---"

# ---------- Phase 2 : agent, pool de $JOBS_AGENT ----------
echo "--- phase 2 : agent ---"
for reg in prob det; do
  for y in v2 02 03 04; do
    DS=$(dataset "$reg" "$y")
    [ -f "run_full/Data_environment/$DS" ] || continue
    for s in 41 101 202 303; do
      if [ "$y" = "v2" ] && done_ok "$LOGS/waste_${reg}_p12_s${s}.log"; then
        echo "[skip] waste_${reg}_p12_v2_s${s}"; continue; fi
      TAG="${reg}_p12_${y}_s${s}"
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
done
wait
echo "########## TERMINÉ $(date +%H:%M) ##########"
