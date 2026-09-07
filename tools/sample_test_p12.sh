#!/bin/bash
# Trois années de test supplémentaires à +20 % d'interventions.
#
# Avec `df_fake_test.pkl` (déjà générée) cela fait quatre jeux de test à la même
# pression, ce qui permet de mesurer la variance due au tirage du flux et non
# plus seulement celle due à la graine d'entraînement.
#
# Mêmes paramètres que la génération d'origine (PIPELINE.md §3) : seul le nom de
# sortie change. `sample.py` n'a pas de `--seed` : les tirages sont indépendants
# par construction, ce qui est précisément l'effet recherché ici.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"
PY="$ROOT/.venv/bin/python"
LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"

echo "########## ÉCHANTILLONNAGE TEST p=1,2 $(date +%H:%M) ##########"
for i in 03 04; do
  OUT="df_fake_test_p12_${i}.pkl"
  echo "[$(date +%H:%M)] >>> $OUT"
  if env -u PYTHONPATH "$PY" sample.py --load_as ddpm \
      --save_sample_as "$OUT" \
      --to_keep 40 --value_span 100 --pressure 1.2 \
      > "$LOGS/sample_test_p12_${i}.log" 2>&1; then
    "$PY" -c "
import pandas as pd, hashlib
d = pd.read_pickle('run_full/Data_sampled/$OUT')
print(f'    {len(d)} interventions')
"
  else
    echo "[$(date +%H:%M)] !!! ÉCHEC $OUT"; tail -20 "$LOGS/sample_test_p12_${i}.log"
  fi
done
echo "########## TERMINÉ $(date +%H:%M) ##########"
