"""Ventile `irreversible_spent` d'une baseline entre compétences exigées par le
rôle pourvu et gaspillage réel (compétences rares engagées sans que le rôle les
demande).

Usage :
  python tools/waste_baseline.py --dataset <flux>_rs.pkl --end N [--is_best] --tag NOM

`--seed` ne concerne que la baseline `random` : `apply_logic` y tire via le
module `random`, seul point d'entropie du chemin baseline. `best` prend le
minimum de niveau de compétence et ne lit jamais le générateur, donc son
resultat est identique quelle que soit la graine.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import simulator as sim
from collective_functions import (load_environment, apply_logic,
                                  get_potential_actions)
from explainability import rare_skills_for_step
from paths import PLOTS, resolve
from sim_state import Fleet

live = {}
cache = {}
tally = {"total": 0, "exigees": 0, "gaspillees": 0,
         "decisions_avec_irrev": 0, "decisions": 0, "role_introuvable": 0}

# Ventilation par identifiant de compétence, en plus des totaux. `held` porte
# déjà les identifiants : les compter par clé ne change rien au comptage global,
# qui reste la somme de ces compteurs.
per_skill = {"exigees": Counter(), "gaspillees": Counter()}

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

    req_set = set() if required is None else set(required.tolist())
    for skill in held.tolist():
        key = "exigees" if skill in req_set else "gaspillees"
        per_skill[key][int(skill)] += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="df_pc_real_rs.pkl")
    ap.add_argument("--end", type=int, default=53088)
    ap.add_argument("--is_best", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=42,
                    help="graine du tirage de `random` (sans effet si --is_best)")
    args = ap.parse_args()

    # La graine d'environnement reste 42 : le flux d'évaluation doit être
    # identique pour toutes les politiques. Seul le tirage de décision varie.
    env = load_environment(1, 1, args.dataset, 1, args.end, 42)

    # Après `load_environment` : `constrain_veh` y appelle `random.seed(42)`,
    # qui écraserait une graine posée plus tôt.
    random.seed(args.seed)
    np.random.seed(args.seed)

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
    print(f"\n=== {args.tag} | {args.dataset} | is_best={args.is_best}"
          f" | seed={args.seed} ===")
    # Tous les indicateurs d'observation, pour que le tableau récapitulatif
    # sorte d'un seul passage plutôt que d'un run par ligne.
    for k in ("v_required", "v_sent", "v_sent_initial",
              "v_sent_full", "v_degraded",
              "function_not_found", "v1_not_sent_from_s1",
              "v3_not_sent_from_s3", "v_not_found_in_last_station",
              "skill_lvl", "ff_sent", "rupture_ff"):
        print(f"{k:33s}: {env.dic_indic.get(k)}")
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

    # Ventilation par compétence : un JSON à côté du .pkl de métriques, trop
    # volumineux pour le log et de toute façon plus commode à relire ainsi.
    out = resolve(f"per_skill_{args.tag}.json", PLOTS)
    with open(out, "w") as f:
        json.dump({k: {str(i): n for i, n in c.most_common()}
                   for k, c in per_skill.items()}, f, indent=1)
    print(f"ventilation par compétence       : {out}")
    top = per_skill["gaspillees"].most_common(5)
    print("  5 compétences les plus gaspillées : "
          + ", ".join(f"{i}({n})" for i, n in top))


if __name__ == "__main__":
    main()
