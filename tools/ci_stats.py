"""Agrège les évaluations p=1,2 croisant années de test et graines.

Le plan est à deux facteurs :

- 4 **années de test** tirées du même modèle de diffusion à la même pression ;
- 4 **graines** d'entraînement (agent) ou de tirage de décision (`random`).

Ce sont deux sources de variance distinctes, et les confondre donnerait un
intervalle faux. On agrège donc en deux temps : moyenne sur les graines à
l'intérieur de chaque année, puis dispersion de ces moyennes entre années. Les
`n = 4` valeurs qui portent l'intervalle sont ainsi indépendantes -- quatre
flux distincts -- alors que les 16 mesures brutes ne le sont pas : quatre
d'entre elles partagent une année, donc les mêmes interventions.

`best` échappe au plan : `apply_logic` prend `min(potential_skills)` sans
jamais tirer au sort, ce que deux graines confirment au bit près. Une seule
exécution par année, et l'intervalle qu'on en tire ne mesure que le flux.

    python tools/ci_stats.py            # tableau lisible
    python tools/ci_stats.py --json out.json
"""
import argparse
import json
import math
import os
import pickle
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "run_full", "logs")
PLOTS = os.path.join(REPO, "run_full", "Plots")

YEARS = ("v2", "02", "03", "04")
SEEDS = (41, 101, 202, 303)
REGIMES = ("prob", "det")

# Student bilatéral à 95 %, ddl = n-1.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# Compteurs bruts lus dans les logs et les .pkl.
RAW_KEYS = ("rupture_ff", "v_required", "v_sent", "v_sent_initial",
            "v_sent_full", "v_degraded", "function_not_found",
            "v1_not_sent_from_s1", "v3_not_sent_from_s3",
            "v_not_found_in_last_station", "skill_lvl", "ff_sent",
            "irreversible_spent", "wasted", "warranted")

# Métriques dérivées, calculées run par run avant toute moyenne : une moyenne
# de ratios n'est pas le ratio des moyennes, et c'est la première qui porte la
# dispersion réelle entre réplicats.
DERIVED = ("fit_inverse", "initial_service_rate")

KEYS = RAW_KEYS + DERIVED

# Les métriques d'observation présentes dans un log de baseline.
_LOG_KEYS = ("v_required", "v_sent", "v_sent_initial", "v_sent_full",
             "v_degraded", "function_not_found", "v1_not_sent_from_s1",
             "v3_not_sent_from_s3", "v_not_found_in_last_station",
             "skill_lvl", "ff_sent", "rupture_ff")


def _add_derived(run):
    """Ajoute les métriques dérivées à un run.

    `fit_inverse` = pompiers envoyés / somme des niveaux : l'inverse du niveau
    moyen par affectation. Le niveau étant meilleur quand il est bas, son
    inverse se lit comme les autres colonnes -- plus haut vaut mieux.

    `initial_service_rate` = véhicules du train initial effectivement partis /
    véhicules requis. `v_required` ne compte que le train propre à
    l'intervention ; `v_sent_initial` en exclut les renforts Z1, que `v_sent`
    inclut.
    """
    lvl = run.get("skill_lvl") or 0
    run["fit_inverse"] = (run["ff_sent"] / lvl) if lvl else 0.0
    req = run.get("v_required") or 0
    run["initial_service_rate"] = (
        100 * run["v_sent_initial"] / req) if req else 0.0
    return run


def _ventilation(path):
    """(justifiées, gaspillées) depuis un log de ventilation."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    w = re.search(r"pourvu\s*:\s*(\d+)", txt)
    g = re.search(r"gaspillage\s*:\s*(\d+)", txt)
    if not (w and g):
        return None
    return int(w.group(1)), int(g.group(1))


def _from_log(path):
    """Toutes les métriques depuis un log de baseline."""
    vent = _ventilation(path)
    if vent is None:
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    run = {}
    for k in _LOG_KEYS:
        m = re.search(rf"^{k}\s*:\s*([\d.]+)", txt, re.M)
        if m:
            run[k] = float(m.group(1))
    m = re.search(r"irreversible_spent \(total\)\s*:\s*(\d+)", txt)
    if not m or len(run) != len(_LOG_KEYS):
        return None
    run["irreversible_spent"] = int(m.group(1))
    run["warranted"], run["wasted"] = vent
    return _add_derived(run)


def _from_pkl(pkl, log):
    """Métriques d'observation depuis le .pkl, ventilation depuis le log."""
    if not os.path.exists(pkl):
        return None
    vent = _ventilation(log)
    if vent is None:
        return None
    with open(pkl, "rb") as f:
        run = dict(pickle.load(f))
    run["warranted"], run["wasted"] = vent
    return _add_derived(run)


def read_best(reg, year):
    return _from_log(f"{LOGS}/waste_best_{reg}_p12_{year}.log")


def read_random(reg, year, seed):
    # L'année v2 a été mesurée avant l'introduction du suffixe d'année.
    for tag in (f"rnd_{reg}_p12_{year}_s{seed}",
                f"rnd_{reg}_p12_s{seed}" if year == "v2" else None):
        if tag and (run := _from_log(f"{LOGS}/{tag}.log")):
            return run
    return None


def read_agent(reg, year, seed):
    for tag in (f"waste_{reg}_p12_{year}_s{seed}",
                f"waste_{reg}_p12_s{seed}" if year == "v2" else None):
        if tag and (run := _from_pkl(f"{PLOTS}/{tag}.pkl", f"{LOGS}/{tag}.log")):
            return run
    return None


def summarise(values):
    """(moyenne, écart-type, demi-IC 95 %) sur des observations indépendantes."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd, T_95.get(n - 1, 1.96) * sd / math.sqrt(n)


def collect(strict=True):
    """{régime: {politique: {métrique: (moyenne, sd, demi-IC, n_années)}}}.

    Pour l'agent et `random`, chaque année fournit d'abord la moyenne de ses
    graines ; la dispersion est ensuite calculée sur ces moyennes annuelles.
    """
    out, missing = {}, []
    for reg in REGIMES:
        entry = {}

        # --- best : une valeur exacte par année ---
        per_year = {}
        for y in YEARS:
            run = read_best(reg, y)
            if run is None:
                missing.append(f"best_{reg}_p12_{y}")
            else:
                per_year[y] = run
        if per_year:
            entry["best"] = {
                k: summarise([r[k] for r in per_year.values()]) + (len(per_year),)
                for k in KEYS
            }

        # --- random et agent : moyenne par année, puis dispersion inter-années ---
        for pol, reader in (("random", read_random), ("agent", read_agent)):
            year_means = {}
            for y in YEARS:
                runs = []
                for s in SEEDS:
                    run = reader(reg, y, s)
                    if run is None:
                        missing.append(f"{pol}_{reg}_p12_{y}_s{s}")
                    else:
                        runs.append(run)
                if runs:
                    year_means[y] = {k: sum(r[k] for r in runs) / len(runs)
                                     for k in KEYS}
            if year_means:
                entry[pol] = {
                    k: summarise([m[k] for m in year_means.values()]) + (len(year_means),)
                    for k in KEYS
                }

        # part des justifiées dans le total
        for pol in entry:
            w = entry[pol]["warranted"][0]
            t = entry[pol]["irreversible_spent"][0]
            entry[pol]["share"] = (100 * w / t, 0.0, 0.0, entry[pol]["warranted"][3])
        out[reg] = entry

    if missing and strict:
        print(f"!! {len(missing)} mesures manquantes :", file=sys.stderr)
        for m in missing[:12]:
            print(f"   {m}", file=sys.stderr)
        if len(missing) > 12:
            print(f"   ... et {len(missing) - 12} autres", file=sys.stderr)
    return out, missing


def paired_gap(reg, key="rupture_ff"):
    """IC 95 % de l'écart agent/`best`, apparié par année.

    Les deux politiques sont évaluées sur les mêmes quatre années : leurs
    erreurs sont corrélées, et les comparer comme deux échantillons
    indépendants gonfle l'intervalle jusqu'à lui faire contenir zéro alors que
    l'agent devance `best` sur chacune des quatre années. On calcule donc
    l'écart dans chaque année, puis l'intervalle sur ces quatre écarts.
    """
    gaps = []
    for y in YEARS:
        b = read_best(reg, y)
        runs = [read_agent(reg, y, s) for s in SEEDS]
        runs = [r for r in runs if r]
        if b is None or not runs:
            continue
        mean = sum(r[key] for r in runs) / len(runs)
        gaps.append(100 * (mean - b[key]) / b[key])
    if len(gaps) < 2:
        return None
    m, _sd, ci = summarise(gaps)
    return m - ci, m + ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="écrit les statistiques dans ce fichier")
    ap.add_argument("--allow-missing", action="store_true",
                    help="agrège malgré des mesures absentes (n réduit)")
    args = ap.parse_args()

    stats, missing = collect()
    if missing and not args.allow_missing:
        raise SystemExit("mesures incomplètes ; relancer les runs "
                         "ou passer --allow-missing")

    for reg, entry in stats.items():
        print(f"\n########## p=1,2 {reg} ##########")
        for pol in ("best", "random", "agent"):
            if pol not in entry:
                continue
            n = entry[pol]["rupture_ff"][3]
            print(f"-- {pol}  (n={n} années)")
            for k in KEYS:
                m, sd, ci, _ = entry[pol][k]
                print(f"   {k:30s} {m:11.0f} +/- {sd:7.0f}   IC95 +/-{ci:8.0f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(stats, f, indent=1)
        print(f"\nécrit : {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
