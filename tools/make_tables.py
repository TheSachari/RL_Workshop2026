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

KEYS = ("rupture_ff", "v_sent", "v_sent_full", "v_degraded",
        "function_not_found", "v1_not_sent_from_s1", "v3_not_sent_from_s3",
        "v_not_found_in_last_station", "skill_lvl", "irreversible_spent",
        "wasted", "warranted")

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
    """(justifiées, gaspillées) depuis le log de ventilation `tag`."""
    path = os.path.join(LOGS, f"{tag}.log")
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    warranted = re.search(r"pourvu\s*:\s*(\d+)", txt)
    wasted = re.search(r"gaspillage\s*:\s*(\d+)", txt)
    if not (warranted and wasted):
        raise ValueError(f"ventilation illisible dans {path}")
    return int(warranted.group(1)), int(wasted.group(1))


def read_agent(cfg, seed):
    if seed == 41 and cfg in AGENT_S41:
        return dict(AGENT_S41[cfg])
    pkl = os.path.join(PLOTS, f"waste_{cfg}_s{seed}.pkl")
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        run = dict(pickle.load(f))
    run["warranted"], run["wasted"] = _ventilation(f"waste_{cfg}_s{seed}")
    return run


def read_random(cfg, seed):
    path = os.path.join(LOGS, f"rnd_{cfg}_s{seed}.log")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    run = {}
    for key in KEYS[:9]:
        m = re.search(rf"^{key}\s*:\s*([\d.]+)", txt, re.M)
        if m:
            run[key] = float(m.group(1))
    m = re.search(r"irreversible_spent \(total\)\s*:\s*(\d+)", txt)
    run["irreversible_spent"] = int(m.group(1))
    run["warranted"], run["wasted"] = _ventilation(f"rnd_{cfg}_s{seed}")
    return run


def summarise(values):
    """(moyenne, écart-type, demi-IC 95 %). Écart-type nul si une seule valeur."""
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, sd, T_95.get(n - 1, 1.96) * sd / math.sqrt(n)


def collect():
    stats = {}
    for cfg in CONFIGS:
        entry = {}
        for name, reader in (("agent", read_agent), ("random", read_random)):
            runs = [r for r in (reader(cfg, s) for s in SEEDS) if r]
            if len(runs) != len(SEEDS):
                got = [s for s in SEEDS if reader(cfg, s)]
                raise SystemExit(
                    f"{cfg}/{name} : {len(runs)} réplicats sur {len(SEEDS)} "
                    f"(graines trouvées : {got}). Relancer les runs manquants.")
            entry[name] = {k: summarise([r[k] for r in runs]) for k in KEYS}
        entry["best"] = {k: (BEST[cfg][k], 0.0, 0.0) for k in KEYS}
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
    body = num(mean, dec) if sd == 0 else f"{num(mean, dec)} \\pm {num(sd, dec)}"
    body += suffix
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def gap(agent, best):
    """Écart relatif agent/best. Deux décimales sous 0,1 %, sinon '+0,0 %'
    masque le signe des métriques quasi identiques."""
    d = 100 * (agent - best) / best
    dec = 2 if abs(d) < 0.1 else 1
    sign = "+" if d >= 0 else "-"
    return f"${sign}{f'{abs(d):.{dec}f}'.replace('.', '{,}')}\\,\\%$"


def ci_interval(entry):
    """IC 95 % de l'écart de ruptures agent/best, en pourcentage."""
    mean, _, half = entry["agent"]["rupture_ff"]
    best = entry["best"]["rupture_ff"][0]
    centre, width = 100 * (mean - best) / best, 100 * half / best
    f = lambda v: f"{v:.1f}".replace(".", "{,}")
    return f"$[{f(centre - width)}\\,\\%, {f(centre + width)}\\,\\%]$"


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
)

CAPTIONS = {
    "det_p10": ("Deterministic composition, real test set.", "tab:recap-det-real"),
    "det_p12": ("Deterministic composition, synthetic test set.", "tab:recap-det-fake"),
    "prob_p10": ("Probabilistic composition, real test set.", "tab:recap-prob-real"),
    "prob_p12": ("Probabilistic composition, synthetic test set.", "tab:recap-prob-fake"),
}

SEED_NOTE = ("Agent and \\textsc{random}: mean over four seeds $\\pm$ one\n"
             "standard deviation. \\textsc{best} is deterministic (it selects "
             "the minimum\nskill level and never draws at random), so it has a "
             "single exact value.")


def recap_table(cfg, entry):
    caption, label = CAPTIONS[cfg]
    out = [f"""\\begin{{table}}[htbp]
\\caption{{{caption} {SEED_NOTE}
$95\\,\\%$ CI on the agent's shortfall gap to \\textsc{{best}}:
{ci_interval(entry)}.}}
\\label{{{label}}}
\\centering
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
Indicator & \\textsc{{best}} & \\textsc{{random}} & Agent & Agent vs \\textsc{{best}} \\\\

\\midrule"""]
    for key, label_txt, bold, better in ROWS:
        means = {p: entry[p][key][0] for p in ("best", "random", "agent")}
        winner = None
        if better:
            winner = (min if better == "min" else max)(means, key=means.get)
        cells = [cell(entry[p][key], bold=(bold or winner == p))
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
    cis = ", ".join(f"{n} {ci_interval(stats[c])}" for n, c in zip(names, order))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", help="dossier de l'article ; sinon, stdout")
    ap.add_argument("--json", help="écrit aussi les statistiques brutes ici")
    args = ap.parse_args()

    stats = collect()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(stats, f, indent=1)

    recaps = "\n".join(recap_table(c, stats[c])
                       for c in ("det_p10", "det_p12", "prob_p10", "prob_p12"))
    arbitrage = arbitrage_table(stats)

    if not args.out_dir:
        print(recaps)
        print(arbitrage)
        return

    # `resultats.tex` = les quatre récapitulatifs puis l'arbitrage. Le préambule
    # (\label de section) précède le premier \begin{table} et est conservé.
    path = os.path.join(args.out_dir, "resultats.tex")
    with open(path, encoding="utf-8") as f:
        current = f.read()
    head = current[:current.index("\\begin{table}")]
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + recaps + "\n\n" + arbitrage + "\n")
    print(f"écrit : {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
