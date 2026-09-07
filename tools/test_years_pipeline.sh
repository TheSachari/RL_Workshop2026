#!/bin/bash
# Trois années de test supplémentaires à +20 %, dans les deux régimes.
#
# Avec `df_fake_test.pkl` déjà en place, cela donne quatre jeux de test par
# régime : la variance mesurée cesse d'être seulement celle de la graine
# d'entraînement et inclut celle du tirage du flux.
#
# Régime probabiliste = `--prob_dep` (départs tirés de la distribution
# historique) ; déterministe = sans (départ modal). `--only fake` laisse
# intactes les entrées des cas golden -- df_pc_real*, planning, df_skills,
# df_stations, df_v, df_roles, df_vehicles_history.
#
# Séquentiel : un seul run lourd à la fois (contrainte mémoire du projet).
# ~10 min par environnement, ~5 min par rare_skills, soit ~1 h 30 au total.
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
    echo "[$(date +%H:%M)] !!! TRACEBACK $label — voir $log"
    grep -A5 -i traceback "$log" | head -20; exit 1
  fi
  echo "[$(date +%H:%M)] <<< $label OK"
}

echo "########## ANNÉES DE TEST p=1,2, DEUX RÉGIMES $(date +%H:%M) ##########"

# Empreintes des entrées golden : `--only fake` ne doit en toucher aucune.
GOLDEN="df_pc_real_prob.pkl planning.pkl df_skills.pkl df_stations.pkl
        df_v.pkl df_roles.pkl df_vehicles_history.pkl"
( cd run_full/Data_environment && md5sum $GOLDEN > /tmp/golden_before.md5 )

for i in 02 03 04; do
  SAMPLE="df_fake_test_p12_${i}.pkl"

  # --- probabiliste ---
  run "env prob $i" "$LOGS/gen_prob_test_p12_${i}.log" \
    "$PY" generate_environment.py --prob_dep --only fake \
    --sample_list "$SAMPLE" --start_year 2018 \
    --save_as "df_pc_fake_test_p12_${i}.pkl"

  run "rare_skills prob $i" "$LOGS/rs_prob_test_p12_${i}.log" \
    "$PY" explainability.py --dataset "df_pc_fake_test_p12_${i}.pkl" \
    --from_dir environment --merge_into "df_pc_fake_test_p12_${i}.pkl" \
    --save_as "df_pc_fake_test_p12_${i}_rs.pkl" \
    --save_rare_as "rare_df_pc_fake_test_p12_${i}.pkl" --rarity 10

  # --- déterministe : même échantillon, départ modal ---
  run "env det $i" "$LOGS/gen_det_test_p12_${i}.log" \
    "$PY" generate_environment.py --only fake \
    --sample_list "$SAMPLE" --start_year 2018 \
    --save_as "df_pc_fake_test_p12_${i}_det.pkl"

  run "rare_skills det $i" "$LOGS/rs_det_test_p12_${i}.log" \
    "$PY" explainability.py --dataset "df_pc_fake_test_p12_${i}_det.pkl" \
    --from_dir environment --merge_into "df_pc_fake_test_p12_${i}_det.pkl" \
    --save_as "df_pc_fake_test_p12_${i}_det_rs.pkl" \
    --save_rare_as "rare_df_pc_fake_test_p12_${i}_det.pkl" --rarity 10
done

echo "--- contrôle des entrées golden ---"
( cd run_full/Data_environment && md5sum -c /tmp/golden_before.md5 ) || {
  echo "!!! une entrée golden a été modifiée"; exit 1; }

echo "--- bornes des flux produits ---"
"$PY" - <<'PYEOF'
import pandas as pd
from pathlib import Path
root = Path("run_full/Data_environment")
for i in ("02", "03", "04"):
    for tag in (f"df_pc_fake_test_p12_{i}_rs.pkl",
                f"df_pc_fake_test_p12_{i}_det_rs.pkl"):
        p = root / tag
        if p.exists():
            print(f"  {tag:42s} {len(pd.read_pickle(p)) // 2} interventions")
PYEOF

echo "########## TERMINÉ $(date +%H:%M) ##########"
