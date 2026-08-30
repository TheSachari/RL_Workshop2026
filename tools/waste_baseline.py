"""Ventile `irreversible_spent` d'une baseline entre compétences exigées par le
rôle pourvu et gaspillage réel (compétences rares engagées sans que le rôle les
demande).

Usage :
  python tools/waste_baseline.py --dataset <flux>_rs.pkl --end N [--is_best] --tag NOM
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import simulator as sim
from collective_functions import (load_environment, apply_logic,
                                  get_potential_actions)
from explainability import rare_skills_for_step
from sim_state import Fleet

live = {}
cache = {}
tally = {"total": 0, "exigees": 0, "gaspillees": 0,
         "decisions_avec_irrev": 0, "decisions": 0, "role_introuvable": 0}

_orig_gen_state = sim.gen_state
_veh = {"key": None, "base": 0}


def on_state_ready(st):
    live["st"] = st


def gen_state_tracked(st, *a, **kw):
    """Reconstitue le numéro de rôle courant, local à `_fill_roles`.

    `irreversible_fn` est appelé AVANT `step`, donc intercepter `step` donnerait
    le rôle de la décision précédente. `gen_state` s'exécute juste avant la
    décision, ce qui donne le bon.
    """
    key = (st.num_inter, st.num_d, st.v_mat)
    if key != _veh["key"]:
        _veh["key"] = key
        _veh["base"] = st.idx_role
    live["num_role"] = st.idx_role - _veh["base"] + 1
    return _orig_gen_state(st, *a, **kw)


sim.gen_state = gen_state_tracked


def irreversible_split(st, ff_array, ff_existing, action):
    """Même compte que `irreversible_spent_by`, ventilé en deux."""
    if action >= 79 or not len(ff_existing) or action >= len(ff_existing):
        return 0

    _local, irreversible = rare_skills_for_step(st, n_following=5, cache=cache)
    if len(irreversible) == 0:
        return 0
    idx = np.asarray(irreversible, dtype=int)
    idx = idx[idx < ff_array.shape[1]]
    if idx.size == 0:
        return 0

    held = idx[ff_array[action, idx] == 1]
    total = int(held.size)
    if total == 0:
        return 0

    tally["decisions_avec_irrev"] += 1

    # `dic_roles[fonction]` est un dict {num_role: nom_du_role}, indexé à 1.
    roles = st.dic_roles.get(st.vehicle_to_find, {})
    role_name = roles.get(live.get("num_role"))
    if role_name is None:
        tally["role_introuvable"] += 1
    required = None
    if role_name is not None:
        vec = st.dic_roles_skills.get(role_name)
        if vec is not None:
            # (niveaux, 134) : exigée si un niveau quelconque la demande.
            required = np.flatnonzero((np.atleast_2d(vec) == 1).any(axis=0))

    exigees = 0 if required is None or required.size == 0 \
        else int(np.isin(held, required).sum())

    tally["total"] += total
    tally["exigees"] += exigees
    tally["gaspillees"] += total - exigees
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="df_pc_real_rs.pkl")
    ap.add_argument("--end", type=int, default=53088)
    ap.add_argument("--is_best", action="store_true")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    env = load_environment(1, 1, args.dataset, 1, args.end, 42)

    def decide(state, all_ff_waiting, ff_array, inter_done):
        tally["decisions"] += 1
        pa, ps = get_potential_actions(state, all_ff_waiting)
        action, skill_lvl = apply_logic(pa, ps, args.is_best)
        return action, skill_lvl, pa

    def on_interval(num_inter, vehicle_out, fleet):
        if num_inter % 10000 == 0:
            print(f"  {num_inter}...", flush=True)

    sim.run_simulation(env, Fleet(), decide, action_size=80,
                       on_interval=on_interval,
                       on_state_ready=on_state_ready,
                       irreversible_fn=irreversible_split)

    t = tally
    print(f"\n=== {args.tag} | {args.dataset} | is_best={args.is_best} ===")
    print(f"v_required                       : {env.dic_indic['v_required']}")
    print(f"rupture_ff                       : {env.dic_indic['rupture_ff']}")
    print(f"décisions                        : {t['decisions']}")
    print(f"  dont dépense irrécupérable     : {t['decisions_avec_irrev']}"
          f"  ({100*t['decisions_avec_irrev']/max(t['decisions'],1):.1f} %)")
    print(f"\nirreversible_spent (total)       : {t['total']}")
    print(f"  exigées par le rôle pourvu     : {t['exigees']:6d}"
          f"  ({100*t['exigees']/max(t['total'],1):5.1f} %)")
    print(f"  NON exigées -> gaspillage      : {t['gaspillees']:6d}"
          f"  ({100*t['gaspillees']/max(t['total'],1):5.1f} %)")
    print(f"\nrôle non résolu (contrôle)       : {t['role_introuvable']}")
    print(f"contrôle dic_indic               : {env.dic_indic['irreversible_spent']}")


if __name__ == "__main__":
    main()
