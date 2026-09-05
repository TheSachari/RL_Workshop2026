#!/bin/bash
# Les 10 réplicats restants : 3 graines x 3 configurations, plus la 3e graine
# de prob_p10. Séquentiel, ~21 h. Chaque configuration est indépendante :
# un échec n'interrompt pas les suivantes.
set -u
REPO="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips/RL_Workshop2026"
cd "$REPO" || exit 1

echo "########## DÉBUT $(date +%H:%M) ##########"
bash tools/seed_replicates.sh prob_p10 303          || echo "!!! prob_p10 interrompu"
bash tools/seed_replicates.sh prob_p12 101 202 303  || echo "!!! prob_p12 interrompu"
bash tools/seed_replicates.sh det_p10  101 202 303  || echo "!!! det_p10 interrompu"
bash tools/seed_replicates.sh det_p12  101 202 303  || echo "!!! det_p12 interrompu"
echo "########## TOUT TERMINÉ $(date +%H:%M) ##########"
