#!/bin/bash
# PPO en déterministe p=1,2 : 4 graines x 4 années de test.
#
# But : donner à l'exclusion de PPO du papier une base mesurée plutôt qu'un
# run unique. Même plan que FQF dans cette configuration -- mêmes flux, mêmes
# bornes, même récompense -- pour que l'écart porte sur l'algorithme seul.
#
# Attend la fin des évaluations FQF en cours : un entraînement PPO occupe
# ~2 Go et sature un cœur, et la machine n'a que ~6 Go libres.
#
# Coût : ~3 h par entraînement (4 graines, séquentiel) + 16 évaluations.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TRAIN=df_pc_fake_10y_p12_det_rs.pkl; N_TRAIN=637332
N_TEST=63696
SEEDS="41 101 202 303"
YEARS="v2 02 03 04"

year_dataset() {
  [ "$1" = "v2" ] && echo df_pc_fake_test_det_rs.pkl \
                  || echo "df_pc_fake_test_p12_$1_det_rs.pkl"
}

echo "########## PPO det p=1,2 : IC $(date +%H:%M) ##########"

# --- attendre que les évaluations FQF libèrent la mémoire ---
while pgrep -f "tools/waste_(agent|baseline)\.py" >/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] mémoire libre, démarrage"

# --- 1. entraînements, un à la fois ---
for S in $SEEDS; do
  TAG="ppo_det_p12_s${S}"
  if [ -f "run_full/SVG_model/agent_${TAG}.pt" ]; then
    echo "[$(date +%H:%M)] [skip] entraînement $TAG"; continue
  fi
  echo "[$(date +%H:%M)] >>> ENTRAÎNEMENT $TAG"
  if ! env -u PYTHONPATH "$PY" agent_run_explainable.py --train \
      --model_name "agent_${TAG}.pt" --agent_model ppo \
      --hyper_params hp_ppo_am.json --reward_weights rw_rupture_ff.json \
      --dataset "$TRAIN" --start 1 --end "$N_TRAIN" \
      --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
      --checkpoint_every 10000 --seed "$S" \
      --save_metrics_as "metrics_${TAG}_train" \
      > "$LOGS/train_${TAG}.log" 2>&1; then
    echo "[$(date +%H:%M)] !!! ÉCHEC entraînement $TAG"; tail -25 "$LOGS/train_${TAG}.log"; exit 1
  fi
  echo "[$(date +%H:%M)] <<< entraînement $TAG OK"
done

# --- 2. évaluations : 4 graines x 4 années, 2 en parallèle ---
echo "--- évaluations ---"
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$1" ]; do wait -n 2>/dev/null || sleep 5; done; }

for S in $SEEDS; do
  MODEL="agent_ppo_det_p12_s${S}.pt"
  [ -f "run_full/SVG_model/$MODEL" ] || { echo "!! modèle absent : $MODEL"; continue; }
  for Y in $YEARS; do
    DS=$(year_dataset "$Y")
    [ -f "run_full/Data_environment/$DS" ] || { echo "!! flux absent : $DS"; continue; }
    TAG="ppo_det_p12_${Y}_s${S}"
    if grep -q "gaspillage" "$LOGS/waste_${TAG}.log" 2>/dev/null; then
      echo "[skip] $TAG"; continue
    fi
    throttle 2
    ( env -u PYTHONPATH "$PY" tools/waste_agent.py -- \
        --model_name "$MODEL" --agent_model ppo \
        --hyper_params hp_ppo_am.json --reward_weights rw_rupture_ff.json \
        --dataset "$DS" --start 1 --end $N_TEST \
        --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
        --save_metrics_as "waste_${TAG}" > "$LOGS/waste_${TAG}.log" 2>&1 \
      && echo "[$(date +%H:%M)] OK  $TAG  ruptures=$(grep -oP 'rupture_ff: \K\d+' "$LOGS/waste_${TAG}.log" | tail -1)" \
      || echo "[$(date +%H:%M)] !!! ÉCHEC $TAG" ) &
  done
done
wait
echo "########## TERMINÉ $(date +%H:%M) ##########"
