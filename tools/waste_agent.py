"""Ventile `irreversible_spent` d'un run d'AGENT entre compétences exigées par
le rôle pourvu et gaspillage réel.

Enveloppe `agent_run_explainable.py` sans le modifier : remplace
`explainability.irreversible_spent_by` par une version qui compte la même chose
et ventile en plus, puis exécute le script d'évaluation tel quel — garantissant
le même chemin de code que les évaluations publiées.

Usage :
  python tools/waste_agent.py -- <arguments habituels de agent_run_explainable.py>
"""
import json
import os
import runpy
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

import explainability as ex
import simulator as sim

_orig_spent_by = ex.irreversible_spent_by
_orig_gen_state = sim.gen_state

live = {"num_role": None}
_veh = {"key": None, "base": 0}
tally = {"total": 0, "exigees": 0, "gaspillees": 0,
         "decisions_avec_irrev": 0, "role_introuvable": 0}

# Ventilation par identifiant de compétence, en plus des totaux. `held` porte
# déjà les identifiants : les compter par clé ne change rien au comptage global.
per_skill = {"exigees": Counter(), "gaspillees": Counter()}


def gen_state_tracked(st, *a, **kw):
    """Reconstitue le numéro de rôle courant, local à `_fill_roles`."""
    key = (st.num_inter, st.num_d, st.v_mat)
    if key != _veh["key"]:
        _veh["key"] = key
        _veh["base"] = st.idx_role
    live["num_role"] = st.idx_role - _veh["base"] + 1
    return _orig_gen_state(st, *a, **kw)


def spent_by_tracked(st, ff_array, ff_existing, action, n_following=5, cache=None):
    total = _orig_spent_by(st, ff_array, ff_existing, action,
                           n_following=n_following, cache=cache)
    if not total:
        return total

    tally["decisions_avec_irrev"] += 1

    _local, irreversible = ex.rare_skills_for_step(
        st, n_following=n_following, cache=cache)
    idx = np.asarray(irreversible, dtype=int)
    idx = idx[idx < ff_array.shape[1]]
    held = idx[ff_array[action, idx] == 1]

    roles = st.dic_roles.get(st.vehicle_to_find, {})
    role_name = roles.get(live.get("num_role"))
    if role_name is None:
        tally["role_introuvable"] += 1
    required = None
    if role_name is not None:
        vec = st.dic_roles_skills.get(role_name)
        if vec is not None:
            required = np.flatnonzero((np.atleast_2d(vec) == 1).any(axis=0))

    exigees = 0 if required is None or required.size == 0 \
        else int(np.isin(held, required).sum())

    tally["total"] += int(held.size)
    tally["exigees"] += exigees
    tally["gaspillees"] += int(held.size) - exigees

    req_set = set() if required is None else set(required.tolist())
    for skill in held.tolist():
        key = "exigees" if skill in req_set else "gaspillees"
        per_skill[key][int(skill)] += 1
    return total


sim.gen_state = gen_state_tracked
ex.irreversible_spent_by = spent_by_tracked
# `agent_run_explainable` importe le symbole directement : le patch doit donc
# précéder son import, ce que garantit `runpy` ci-dessous.

sep = sys.argv.index("--")
sys.argv = ["agent_run_explainable.py"] + sys.argv[sep + 1:]

try:
    runpy.run_path(os.path.join(REPO, "agent_run_explainable.py"),
                   run_name="__main__")
finally:
    t = tally
    print("\n=== ventilation irreversible_spent (agent) ===")
    print(f"décisions avec dépense         : {t['decisions_avec_irrev']}")
    print(f"total                          : {t['total']}")
    print(f"  exigées par le rôle pourvu   : {t['exigees']:6d}"
          f"  ({100*t['exigees']/max(t['total'],1):5.1f} %)")
    print(f"  NON exigées -> gaspillage    : {t['gaspillees']:6d}"
          f"  ({100*t['gaspillees']/max(t['total'],1):5.1f} %)")

    # Ventilation par compétence, à côté du .pkl de métriques. Le tag de sortie
    # est celui de `--save_metrics_as`, pour que les deux fichiers s'apparient.
    tag = "run"
    if "--save_metrics_as" in sys.argv:
        tag = sys.argv[sys.argv.index("--save_metrics_as") + 1]
    out = os.path.join(REPO, "run_full", "Plots", f"per_skill_{tag}.json")
    with open(out, "w") as f:
        json.dump({k: {str(i): n for i, n in c.most_common()}
                   for k, c in per_skill.items()}, f, indent=1)
    print(f"ventilation par compétence     : {out}")
    print(f"rôle non résolu (contrôle)     : {t['role_introuvable']}")
