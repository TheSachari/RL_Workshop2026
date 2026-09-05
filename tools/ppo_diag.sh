#!/bin/bash
# Diagnostic de la divergence PPO : run court (100k interventions) par hypothèse,
# séquentiel (4,2 Go par run, ~4 Go libres). On ne cherche pas un chiffre final
# mais le SIGNE de la pente : le taux de ruptures par tranche doit décroître.
set -u
ROOT="/media/michael/Fichiers/Fac/Post_thèse/RL_neurips"
REPO="$ROOT/RL_Workshop2026"; PY="$ROOT/.venv/bin/python"; LOGS="$REPO/run_full/logs"
cd "$REPO" || exit 1
export RL_DATA_ROOT="$REPO/run_full"
N=${N:-100000}

for HP in "$@"; do
  TAG="diag_$(basename "$HP" .json)"
  echo "[$(date +%H:%M)] >>> $TAG (N=$N)"
  env -u PYTHONPATH "$PY" agent_run_explainable.py --train \
    --model_name ${TAG}.pt --agent_model ppo \
    --hyper_params "$HP" --reward_weights rw_rupture_ff.json \
    --dataset df_pc_fake_10y_p10_det_rs.pkl --start 1 --end "$N" \
    --n_hours 2 --top_n 5 --constraint_factor_veh 1 --constraint_factor_ff 1 \
    --checkpoint_every 999999 --seed 41 \
    --save_metrics_as metrics_${TAG} > "$LOGS/${TAG}.log" 2>&1 \
    || { echo "!!! ECHEC $TAG"; tail -20 "$LOGS/${TAG}.log"; continue; }
  "$PY" - "$LOGS/${TAG}.log" <<'PYEOF'
import re, sys
rows=[]
for line in open(sys.argv[1], errors="ignore"):
    m=re.match(r"^(\d+) v_out:.*rupture_ff: (\d+)", line)
    if m: rows.append((int(m.group(1)), int(m.group(2))))
if len(rows)<2: print("   (pas assez de points)"); sys.exit()
step=max(rows)[0]//5
print("   taux ruptures/1000 par tranche:")
vals=[]
for i in range(5):
    a,b=i*step,(i+1)*step
    xs=[(x,v) for x,v in rows if a<=x<=b]
    if len(xs)>=2:
        (i0,v0),(i1,v1)=xs[0],xs[-1]
        r=(v1-v0)/(i1-i0)*1000; vals.append(r)
        print(f"     {a:>7,}-{b:<7,}: {r:6.1f}")
if len(vals)>=2:
    verdict = "DECROIT (piste valide)" if vals[-1] < vals[0] else "CROIT ENCORE"
    print(f"   => premiere {vals[0]:.1f} -> derniere {vals[-1]:.1f} : {verdict}")
PYEOF
done
echo "[$(date +%H:%M)] ===== DIAGNOSTIC TERMINE ====="
