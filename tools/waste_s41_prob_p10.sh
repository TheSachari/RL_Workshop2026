#!/bin/bash
# Ventilation de la graine 41 (modèle publié) en probabiliste p=1.0.
# Le log d'origine a disparu : on refait la mesure pour que les 4 graines
# de la ligne "gaspillées" viennent toutes du même chemin de code.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

echo "########## VENTILATION s41 prob_p10 $(date +%H:%M) ##########"
env -u PYTHONPATH "$PY" tools/waste_agent.py -- \
    --model_name agent_10y_p10_rupture_ff.pt --agent_model fqf \
    --hyper_params hp_fqf.json --reward_weights rw_rupture_ff.json \
    --dataset df_pc_real_rs.pkl --start 1 --end 53088 \
    --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
    --save_metrics_as waste_prob_p10_s41 > "$LOGS/waste_prob_p10_s41.log" 2>&1
grep -E "exigées|gaspillage|total" "$LOGS/waste_prob_p10_s41.log" | tail -3
echo "########## TERMINÉ $(date +%H:%M) ##########"
