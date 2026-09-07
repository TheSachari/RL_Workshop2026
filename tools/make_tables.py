"""Régénère les tableaux LaTeX de l'article à partir des mesures brutes.

Aucun chiffre n'est saisi à la main : les métriques d'observation viennent des
`.pkl` écrits par `--save_metrics_as`, la ventilation rare/gaspillée des logs
correspondants. Le script sort les cinq tableaux de `resultats.tex` (quatre
récapitulatifs + l'arbitrage) et le tableau d'arbitrage seul.

    python tools/make_tables.py --out-dir <dossier de l'article>

Trois politiques, quatre configurations :

- `agent` et `random` sont moyennés sur quatre graines (41, 101, 202, 303), avec
  l'écart-type et l'IC de Student à 95 % (n=4, ddl=3, t=3,182).
- `best` est déterministe — `apply_logic` prend `min(potential_skills)` sans
  jamais tirer au sort, ce que deux graines confirment au bit près. Il n'a pas
  de distribution d'échantillonnage, donc pas d'intervalle : une valeur exacte.

Les valeurs de la graine 41 pour trois configurations, ainsi que celles de
`best`, sont figées ci-dessous : ces runs sont antérieurs à la sauvegarde
systématique des `.pkl`, et leurs logs de ventilation ont été purgés. Elles
proviennent des tableaux publiés. Le seul cas revérifiable l'a été
(`waste_s41_prob_p10.sh` : 67 543 contre 67 546 publié, soit 0,004 %).
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

SEEDS = (41, 101, 202, 303)
CONFIGS = ("prob_p10", "prob_p12", "det_p10", "det_p12")

# Student à 95 %, ddl = n-1.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182}

KEYS = ("rupture_ff", "v_required", "v_sent", "v_sent_full", "v_degraded",
        "function_not_found", "v1_not_sent_from_s1", "v3_not_sent_from_s3",
        "v_not_found_in_last_station", "skill_lvl", "ff_sent",
        "irreversible_spent", "wasted", "warranted",
        "fit_inverse", "initial_service_rate")

# Les configurations p=1,0 datent d'avant le compteur `v_sent_initial` : leur
# taux de service ne peut pas être recalculé sans réexécuter ces runs. La
# cellule reste vide plutôt que d'afficher un ratio bâti sur `v_sent`, qui
# inclut les renforts et dépasserait 100 %.
MISSING = float("nan")

# Compteurs qu'un log de baseline imprime réellement (v_sent_initial n'existe
# que depuis la campagne p=1,2 ; absent des logs p=1,0, il reste None).
_LOG_KEYS = ("v_required", "v_sent", "v_sent_initial", "v_sent_full", "v_degraded",
             "function_not_found", "v1_not_sent_from_s1",
             "v3_not_sent_from_s3", "v_not_found_in_last_station",
             "skill_lvl", "ff_sent", "rupture_ff")

# Graine 41 de l'agent : `.pkl` absents pour ces trois configurations.
AGENT_S41 = {
    "prob_p12": dict(rupture_ff=8916, v_sent=86752, v_sent_full=85490,
                     v_degraded=1262, function_not_found=25763,
                     v1_not_sent_from_s1=14585, v3_not_sent_from_s3=5707,
                     v_not_found_in_last_station=466, skill_lvl=440223,
                     irreversible_spent=87406, warranted=5063, wasted=82343),
    "det_p10": dict(rupture_ff=3948, v_sent=63718, v_sent_full=62912,
                    v_degraded=806, function_not_found=6498,
                    v1_not_sent_from_s1=6709, v3_not_sent_from_s3=1054,
                    v_not_found_in_last_station=73, skill_lvl=319545,
                    irreversible_spent=60816, warranted=2887, wasted=57929),
    "det_p12": dict(rupture_ff=5096, v_sent=76206, v_sent_full=75200,
                    v_degraded=1006, function_not_found=8430,
                    v1_not_sent_from_s1=8906, v3_not_sent_from_s3=1512,
                    v_not_found_in_last_station=124, skill_lvl=381172,
                    irreversible_spent=73974, warranted=3714, wasted=70260),
}

# `best` : déterministe, une exécution suffit et fait foi.
BEST = {
    "prob_p10": dict(rupture_ff=7544, v_sent=72545, v_sent_full=71575,
                     v_degraded=970, function_not_found=20372,
                     v1_not_sent_from_s1=11232, v3_not_sent_from_s3=4409,
                     v_not_found_in_last_station=384, skill_lvl=268047,
                     irreversible_spent=75832, warranted=2862, wasted=72970),
    "prob_p12": dict(rupture_ff=9989, v_sent=86641, v_sent_full=85446,
                     v_degraded=1195, function_not_found=26232,
                     v1_not_sent_from_s1=14840, v3_not_sent_from_s3=6211,
                     v_not_found_in_last_station=509, skill_lvl=322679,
                     irreversible_spent=91880, warranted=3704, wasted=88176),
    "det_p10": dict(rupture_ff=4057, v_sent=63668, v_sent_full=62908,
                    v_degraded=760, function_not_found=6484,
                    v1_not_sent_from_s1=6774, v3_not_sent_from_s3=952,
                    v_not_found_in_last_station=77, skill_lvl=231020,
                    irreversible_spent=62908, warranted=1796, wasted=61112),
    "det_p12": dict(rupture_ff=5342, v_sent=76156, v_sent_full=75186,
                    v_degraded=970, function_not_found=8485,
                    v1_not_sent_from_s1=9070, v3_not_sent_from_s3=1485,
                    v_not_found_in_last_station=138, skill_lvl=278609,
                    irreversible_spent=76403, warranted=2376, wasted=74027),
}


# --------------------------------------------------------------------------
# Lecture des mesures
# --------------------------------------------------------------------------
def _ventilation(tag):
    """(justifiées, gaspillées) depuis le log de ventilation `tag`.

    Renvoie None si le log est absent : les configurations p=1,2 passent
    désormais par `collect_p12`, avec un suffixe d'année, et leurs anciens logs
    sans suffixe n'existent plus. `collect` les traverse quand même avant d'être
    écrasé, d'où cette tolérance.
    """
    path = os.path.join(LOGS, f"{tag}.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    warranted = re.search(r"pourvu\s*:\s*(\d+)", txt)
    wasted = re.search(r"gaspillage\s*:\s*(\d+)", txt)
    if not (warranted and wasted):
        raise ValueError(f"ventilation illisible dans {path}")
    return int(warranted.group(1)), int(wasted.group(1))


def _from_log(path):
    """Toutes les métriques d'un log de baseline, ou None s'il est absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    run = {}
    for key in _LOG_KEYS:
        m = re.search(rf"^{key}\s*:\s*([\d.]+)", txt, re.M)
        if m:
            run[key] = float(m.group(1))
    m = re.search(r"irreversible_spent \(total\)\s*:\s*(\d+)", txt)
    if not m:
        return None
    run["irreversible_spent"] = int(m.group(1))
    w = re.search(r"pourvu\s*:\s*(\d+)", txt)
    g = re.search(r"gaspillage\s*:\s*(\d+)", txt)
    if not (w and g):
        return None
    run["warranted"], run["wasted"] = int(w.group(1)), int(g.group(1))
    return _derive(run)


def _derive(run):
    """Ajoute les deux métriques dérivées à un run p=1,0."""
    lvl = run.get("skill_lvl") or 0
    ff = run.get("ff_sent") or 0
    run["fit_inverse"] = (ff / lvl) if (lvl and ff) else MISSING
    req = run.get("v_required") or 0
    init = run.get("v_sent_initial")
    run["initial_service_rate"] = (
        100 * init / req) if (init is not None and req) else MISSING
    return run


def read_agent(cfg, seed):
    pkl = os.path.join(PLOTS, f"waste_{cfg}_s{seed}.pkl")
    # Les littéraux de la graine 41 ne servent que faute de mesure : ces runs
    # sont antérieurs à la sauvegarde des `.pkl`. Dès qu'un `.pkl` existe, il
    # fait foi -- lui seul porte les compteurs ajoutés depuis.
    if seed == 41 and cfg in AGENT_S41 and not os.path.exists(pkl):
        return _derive(dict(AGENT_S41[cfg]))
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        run = dict(pickle.load(f))
    vent = _ventilation(f"waste_{cfg}_s{seed}")
    if vent is None:
        return None
    run["warranted"], run["wasted"] = vent
    return _derive(run)


def read_random(cfg, seed):
    path = os.path.join(LOGS, f"rnd_{cfg}_s{seed}.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    run = {}
    # Nommées explicitement : une tranche de KEYS dépendait de l'ordre des clés
    # et cassait dès qu'on en insérait une.
    for key in _LOG_KEYS:
        m = re.search(rf"^{key}\s*:\s*([\d.]+)", txt, re.M)
        if m:
            run[key] = float(m.group(1))
    m = re.search(r"irreversible_spent \(total\)\s*:\s*(\d+)", txt)
    run["irreversible_spent"] = int(m.group(1))
    vent = _ventilation(f"rnd_{cfg}_s{seed}")
    if vent is None:
        return None
    run["warranted"], run["wasted"] = vent
    return _derive(run)


def summarise(values):
    """(moyenne, écart-type, demi-IC 95 %). Écart-type nul si une seule valeur."""
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd, T_95.get(n - 1, 1.96) * sd / math.sqrt(n)


def collect_p12():
    """Statistiques p=1,2 : 4 années de test x 4 graines, agrégées à deux
    niveaux par `ci_stats` (moyenne des graines dans l'année, puis dispersion
    entre années). Remplace les valeurs à graines seules pour ces deux
    configurations."""
    from ci_stats import collect as _collect
    stats, missing = _collect(strict=False)
    if missing:
        raise SystemExit(
            f"{len(missing)} mesures p=1,2 absentes ; lancer "
            "tools/ci_evaluations.sh avant de régénérer les tableaux.")
    # ci_stats renvoie (moyenne, sd, demi-IC, n) ; on garde les trois premiers.
    return {f"{'prob' if reg == 'prob' else 'det'}_p12":
            {pol: {k: v[:3] for k, v in metrics.items()}
             for pol, metrics in entry.items()}
            for reg, entry in stats.items()}


def collect():
    """Statistiques à graines seules. Les p=1,2 sont ensuite remplacées par
    `collect_p12` (plan à deux facteurs), donc inutile de les lire ici : leurs
    logs portent désormais un suffixe d'année."""
    stats = {}
    for cfg in CONFIGS:
        if cfg.endswith("_p12"):
            continue
        entry = {}
        for name, reader in (("agent", read_agent), ("random", read_random)):
            runs = [r for r in (reader(cfg, s) for s in SEEDS) if r]
            if len(runs) != len(SEEDS):
                got = [s for s in SEEDS if reader(cfg, s)]
                raise SystemExit(
                    f"{cfg}/{name} : {len(runs)} réplicats sur {len(SEEDS)} "
                    f"(graines trouvées : {got}). Relancer les runs manquants.")
            # `.get` : les littéraux figés (graine 41) ne portent qu'un
            # sous-ensemble des compteurs ; la cellule vaut alors « --- ».
            entry[name] = {
                k: (summarise([r[k] for r in runs])
                    if all(r.get(k) is not None for r in runs)
                    else (MISSING, 0.0, 0.0))
                for k in KEYS
            }
        # `best` est déterministe : un seul run fait foi. On lit le log mesuré
        # quand il existe -- il porte les compteurs récents -- et on retombe sur
        # les valeurs publiées sinon.
        reg = "prob" if cfg.startswith("prob") else "det"
        measured = _from_log(os.path.join(LOGS, f"waste_best_{reg}_p10.log"))
        best_run = measured if measured else _derive(dict(BEST[cfg]))
        entry["best"] = {k: (best_run.get(k, MISSING), 0.0, 0.0) for k in KEYS}
        for pol in ("best", "random", "agent"):
            warranted = entry[pol]["warranted"][0]
            total = entry[pol]["irreversible_spent"][0]
            entry[pol]["share"] = (100 * warranted / total, 0.0, 0.0)
        stats[cfg] = entry
    return stats


# --------------------------------------------------------------------------
# Rendu LaTeX
# --------------------------------------------------------------------------
def num(x, dec=0):
    """Format français : séparateur de milliers fin, virgule décimale."""
    s = f"{x:,.{dec}f}".replace(",", "\\,")
    return s.replace(".", "{,}") if dec else s


def cell(stat, bold=False, dec=0, suffix=""):
    mean, sd, _ = stat
    if isinstance(mean, float) and math.isnan(mean):
        return "---"
    body = num(mean, dec) if sd == 0 else f"{num(mean, dec)} \\pm {num(sd, dec)}"
    body += suffix
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def gap(agent, best):
    if (isinstance(agent, float) and math.isnan(agent)) or \
       (isinstance(best, float) and math.isnan(best)) or not best:
        return "---"
    """Écart relatif agent/best. Deux décimales sous 0,1 %, sinon '+0,0 %'
    masque le signe des métriques quasi identiques."""
    d = 100 * (agent - best) / best
    dec = 2 if abs(d) < 0.1 else 1
    sign = "+" if d >= 0 else "-"
    return f"${sign}{f'{abs(d):.{dec}f}'.replace('.', '{,}')}\\,\\%$"


def ci_interval(entry, cfg=None):
    """IC 95 % de l'écart de ruptures agent/`best`, en pourcentage.

    À $pres=1{,}2$ les deux politiques voient les mêmes quatre années : l'écart
    est calculé par année puis moyenné (test apparié). Les traiter comme deux
    échantillons indépendants élargirait l'intervalle jusqu'à contenir zéro,
    alors que l'agent devance `best` sur chacune des années.
    """
    if cfg and cfg.endswith("_p12"):
        from ci_stats import paired_gap
        bounds = paired_gap("prob" if cfg.startswith("prob") else "det")
        if bounds:
            f = lambda v: f"{v:.1f}".replace(".", "{,}")
            return f"$[{f(bounds[0])}\\,\\%, {f(bounds[1])}\\,\\%]$"
    mean, _, half = entry["agent"]["rupture_ff"]
    best = entry["best"]["rupture_ff"][0]
    centre, width = 100 * (mean - best) / best, 100 * half / best
    f = lambda v: f"{v:.1f}".replace(".", "{,}")
    return f"$[{f(centre - width)}\\,\\%, {f(centre + width)}\\,\\%]$"


# Décimales d'affichage, par métrique. Les compteurs restent entiers ; les deux
# ratios ne veulent rien dire arrondis (0,61 deviendrait « 1 »).
DECIMALS = {"fit_inverse": 3, "initial_service_rate": 1}

ROWS = (
    ("rupture_ff", "\\textbf{Shortfalls}", True, "min"),
    ("v_sent", "Vehicles sent", False, None),
    ("v_sent_full", "\\quad of which fully crewed", False, None),
    ("v_degraded", "Undermanned departures", False, None),
    ("function_not_found", "Vehicle type not found", False, None),
    ("v1_not_sent_from_s1", "$v_1$ not sent from $S_1$", False, None),
    ("v3_not_sent_from_s3", "$v_3$ not sent from $S_3$", False, None),
    ("v_not_found_in_last_station", "$v$ not found, last station", False, None),
    ("skill_lvl", "Cumulative skill-match level", False, None),
    ("irreversible_spent", "Scarce skills", False, None),
    ("wasted", "\\quad wasted", False, "min"),
    ("warranted", "\\quad warranted", False, "max"),
    ("fit_inverse", "Inverse mean skill level", False, "max"),
    ("v_required", "Vehicles required", False, None),
    ("initial_service_rate", "Initial service rate (\\%)", False, "max"),
)

CAPTIONS = {
    "det_p10": ("Deterministic composition, real test set.", "tab:recap-det-real"),
    "det_p12": ("Deterministic composition, synthetic test set.", "tab:recap-det-fake"),
    "prob_p10": ("Probabilistic composition, real test set.", "tab:recap-prob-real"),
    "prob_p12": ("Probabilistic composition, synthetic test set.", "tab:recap-prob-fake"),
}

# Deux plans distincts, donc deux notes : à $pres=1{,}0$ le test porte sur le
# flux réel, unique, et seule la graine varie ; à $pres=1{,}2$ quatre années de
# test synthétiques s'ajoutent, et c'est leur dispersion qui porte l'intervalle.
SEED_NOTE = ("Agent and \\textsc{random}: mean over four training seeds $\\pm$ one\n"
             "standard deviation, on the single real test stream. "
             "\\textsc{best} is deterministic\n(it selects the minimum skill "
             "level and never draws at random), so it has a\nsingle exact value.")

YEAR_NOTE = ("All three policies: mean over four synthetic test years $\\pm$ one\n"
             "standard deviation. Agent and \\textsc{random} are first averaged "
             "over four\nseeds within each year, so the interval carries the "
             "dispersion between\nindependent years rather than between "
             "correlated runs. \\textsc{best} draws at\nrandom on neither "
             "count and is run once per year, so its interval reflects the\n"
             "test stream alone.")


def recap_table(cfg, entry):
    caption, label = CAPTIONS[cfg]
    note = YEAR_NOTE if cfg.endswith("_p12") else SEED_NOTE
    out = [f"""\\begin{{table}}[htbp]
\\caption{{{caption} {note}
$95\\,\\%$ CI on the agent's shortfall gap to \\textsc{{best}}:
{ci_interval(entry, cfg)}.}}
\\label{{{label}}}
\\centering
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
Indicator & \\textsc{{best}} & \\textsc{{random}} & Agent & Agent vs \\textsc{{best}} \\\\

\\midrule"""]
    for key, label_txt, bold, better in ROWS:
        means = {p: entry[p][key][0] for p in ("best", "random", "agent")}
        usable = {k: v for k, v in means.items()
                  if not (isinstance(v, float) and math.isnan(v))}
        winner = None
        if better and usable:
            winner = (min if better == "min" else max)(usable, key=usable.get)
        dec = DECIMALS.get(key, 0)
        cells = [cell(entry[p][key], bold=(bold or winner == p), dec=dec)
                 for p in ("best", "random", "agent")]
        g = gap(entry["agent"][key][0], entry["best"][key][0])
        if bold:
            g = f"$\\mathbf{{{g[1:-1]}}}$"
        out.append(f"{label_txt} & {cells[0]} & {cells[1]} & {cells[2]} & {g} \\\\")
    out.append("\n\\bottomrule\n\\end{tabular}\n\n\\end{table}\n")
    return "\n".join(out)


def arbitrage_table(stats):
    order = ("prob_p10", "prob_p12", "det_p10", "det_p12")
    names = ("probabilistic $pres=1{,}0$", "$pres=1{,}2$",
             "deterministic $pres=1{,}0$", "$pres=1{,}2$")
    cis = ", ".join(f"{n} {ci_interval(stats[c], c)}" for n, c in zip(names, order))

    def block(title, key, better, dec=0, share=False, suffix=""):
        rows = [f"\\multirow{{3}}{{*}}{{{title}}}"]
        winners = {}
        for cfg in order:
            means = {p: stats[cfg][p][key][0] for p in ("best", "random", "agent")}
            winners[cfg] = (min if better == "min" else max)(means, key=means.get)
        for pol, label in (("best", "\\textsc{best}  "),
                           ("random", "\\textsc{random}"),
                           ("agent", "agent          ")):
            cells = []
            for cfg in order:
                stat = stats[cfg][pol][key]
                if share:
                    stat = (stat[0], 0.0, 0.0)
                cells.append(cell(stat, bold=(winners[cfg] == pol),
                                  dec=dec, suffix=suffix))
            rows.append(f" & {label} & " + " & ".join(cells) + " \\\\")
        return rows

    out = ["""\\begin{table}[t]
\\caption{Shortfalls, wasted scarce skills and warranted scarce skills, for the
3 policies and the 4 evaluation environments. Agent and \\textsc{random}: mean
over four seeds $\\pm$ one standard deviation. \\textsc{best} selects the
minimum skill level and never draws at random, so it is deterministic and has a
single exact value. $95\\,\\%$ CI on the agent's shortfall gap to \\textsc{best}:
""" + cis + """.}
\\label{tab:explic-arbitrage}
\\centering
\\small
\\begin{tabular}{llrrrr}
\\toprule
& & \\multicolumn{2}{c}{Probabilistic} & \\multicolumn{2}{c}{Deterministic} \\\\
\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}
Quantity & Policy & $pres = 1{,}0$ & $pres = 1{,}2$ & $pres = 1{,}0$ & $pres = 1{,}2$ \\\\
\\midrule"""]
    out += block("Crew shortfalls $\\downarrow$", "rupture_ff", "min")
    out.append("\\midrule")
    out += block("Wasted skills", "wasted", "min")
    out.append("\\midrule")
    out += block("Warranted skills", "warranted", "max")
    out.append("\\addlinespace")
    out += block("\\quad \\emph{share of total}", "share", "max",
                 dec=1, share=True, suffix="\\,\\%")
    out.append("\\bottomrule\n\\end{tabular}\n\n\\end{table}")
    return "\n".join(out)


# Le tableau de synthèse de l'article : les mêmes mesures que les
# récapitulatifs, restreintes aux lignes qui portent le propos et aux deux
# politiques comparées. Les quatre configurations tiennent côte à côte, d'où
# l'usage de `table*`.
SUMMARY_ROWS = (
    ("rupture_ff", "Shortfalls", 0, True),
    ("v_sent", "Vehicles sent", 0, False),
    ("v_required", "Vehicles required", 0, False),
    ("initial_service_rate", "Initial service rate (\\%)", 1, False),
    ("v1_not_sent_from_s1", "$v_1$ not sent from $S_1$", 0, False),
    ("v3_not_sent_from_s3", "$v_3$ not sent from $S_3$", 0, False),
    ("skill_lvl", "Cum. skill-match level", 0, False),
    ("fit_inverse", "Inverse mean skill level", 3, False),
    ("wasted", "Scarce skills wasted", 0, False),
    ("warranted", "Scarce skills warranted", 0, False),
    ("v_degraded", "Undermanned departures", 0, False),
)

SUMMARY_ORDER = ("det_p10", "det_p12", "prob_p10", "prob_p12")


def _bold(cell_tex):
    """Passe `$x$` en `$\\mathbf{x}$` ; laisse « --- » intact."""
    if not cell_tex.startswith("$"):
        return cell_tex
    return f"$\\mathbf{{{cell_tex[1:-1]}}}$"


def _plain(x, dec):
    """Moyenne seule, sans écart-type : le tableau de synthèse est trop serré
    pour les porter, et les intervalles figurent en appendice. Le mode
    mathématique est obligatoire -- `num` produit `\\,` et `{,}`, qui ne
    rendent rien en mode texte."""
    if isinstance(x, float) and math.isnan(x):
        return "---"
    return f"${num(x, dec)}$"


def summary_table(stats):
    """Le tableau `tab:summary-all` de `resultats.tex`, mesures à l'appui."""
    lines = ["""\\begin{table*}[htbp]
\\caption{Summary of average results across composition strategies and test
         sets. The agent is trained to minimise shortfalls; all other metrics
         are observational. Gaps are relative to \\textsc{best}. Confidence
         intervals and the full set of indicators are given in
         Appendix~\\ref{sec:details_results}.}
\\label{tab:summary-all}
\\centering
\\footnotesize
\\setlength{\\tabcolsep}{1.2pt}
\\begin{tabular}{lrrrrrrrrrrrr}
\\toprule
& \\multicolumn{3}{c}{Det.~real} & \\multicolumn{3}{c}{Det.~synthetic}
& \\multicolumn{3}{c}{Prob.~real} & \\multicolumn{3}{c}{Prob.~synthetic} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}
Indicator & Best & Agent & Gap & Best & Agent & Gap
          & Best & Agent & Gap & Best & Agent & Gap \\\\
\\midrule"""]
    for key, label, dec, bold in SUMMARY_ROWS:
        cells = []
        for cfg in SUMMARY_ORDER:
            b = stats[cfg]["best"][key][0]
            a = stats[cfg]["agent"][key][0]
            g = gap(a, b)
            if bold:
                cells += [_bold(_plain(b, dec)), _bold(_plain(a, dec)),
                          _bold(g)]
            else:
                cells += [_plain(b, dec), _plain(a, dec), g]
        name = f"\\textbf{{{label}}}" if bold else label
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    return "\n".join(lines)



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="dossier de l'article ; sinon, stdout")
    ap.add_argument("--json", help="écrit aussi les statistiques brutes ici")
    args = ap.parse_args()

    stats = collect()
    # p=1,2 : remplacé par le plan à deux facteurs (années x graines).
    stats.update(collect_p12())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(stats, f, indent=1)

    recaps = "\n".join(recap_table(c, stats[c])
                       for c in ("det_p10", "det_p12", "prob_p10", "prob_p12"))
    summary = summary_table(stats)
    arbitrage = arbitrage_table(stats)

    if not args.out_dir:
        print(summary)
        print(recaps)
        print(arbitrage)
        return

    # `resultats.tex` = le texte de la section, le tableau de synthèse puis
    # l'arbitrage. Le préambule (labels et paragraphes) précède le premier
    # \begin{table} et est conservé tel quel.
    path = os.path.join(args.out_dir, "resultats.tex")
    with open(path, encoding="utf-8") as f:
        current = f.read()
    head = current[:current.index("\\begin{table")]
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + summary + "\n\n" + arbitrage + "\n")
    print(f"écrit : {path}", file=sys.stderr)

    # Les quatre récapitulatifs vivent en appendice. Ils y étaient maintenus à
    # la main et avaient dérivé des mesures ; ils sont désormais générés, et
    # ne sont plus dupliqués dans `resultats.tex` (labels en double).
    path = os.path.join(args.out_dir, "detailsResults.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\label{sec:details_results}\n" + recaps + "\n")
    print(f"écrit : {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
