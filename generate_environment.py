"""
generate_environment.py

Builds all artifacts required to run the emergency-response simulation environment
(e.g., stations, vehicles, firefighter skills, dispatch rules, planning) and produces
precomputed "event streams" for both real and synthetic interventions.

What this script does
---------------------
1) Loads raw SDIS/dispatch data from `./Data/` (CSV + GeoJSON)
2) Generates core environment tables:
   - fire stations (filtered + normalized names + numeric coordinates)
   - vehicles inventory per station
   - firefighter skills availability windows (pivoted table)
   - roles/competences table
   - vehicle history table
3) Loads preprocessed artifacts (`df_prob_dep.pkl`, `df_rank_incident.pkl`) and builds
   departure rules, either:
   - deterministic from `responses_by_incident.csv` (default), or
   - probabilistic from historical `df_prob_dep.pkl` when `--prob_dep` is set
4) For REAL interventions (`./Data_trained/df_real.pkl`):
   - assigns PDD (list of stations in the polygon containing the incident point)
   - computes zone (Z_1/Z_2/Z_3/Z_4) from PDD
   - maps incident rank -> incident label
   - computes area_type (land-use / sector type) by nearest neighbor lookup
   - computes departures (deterministic or probabilistic)
   - builds an event stream including RETURN events
   - normalizes selected continuous columns into [0,1]
   - saves `df_pc_real.pkl` to `./Data_environment/`
5) For FAKE interventions (one or more pickle files under `./Data_sampled/`):
   - repeats the same pipeline per sample file and concatenates results
   - normalizes selected continuous columns into [0,1]
   - saves the combined fake stream to `./Data_environment/<save_as>`
6) Builds a `planning.pkl` dictionary from `./Data/Planning/` (CSV files)

Inputs expected on disk
-----------------------
- ./Data/ (multiple CSV files + pdd.geojson + responses_by_incident.csv + Planning/*.csv)
- ./Data_preprocessed/df_prob_dep.pkl
- ./Data_preprocessed/df_rank_incident.pkl
- ./Data_trained/df_real.pkl
- ./Data_sampled/<sample pickle files> (provided via --sample_list)

Outputs
-------
Grouped by `--only` target; without the flag every group is written, which is
the historical behaviour.

- tables   : df_stations.pkl, df_v.pkl, df_skills.pkl, df_roles.pkl,
             df_vehicles_history.pkl
- real     : df_pc_real.pkl (or df_pc_real_prob.pkl under --prob_dep)
- fake     : <save_as>, the combined synthetic event stream
- planning : planning.pkl

Notes
-----
- Paths come from `paths.py`, so the script can be run from any directory.
- Every group except `fake` is an input to the golden cases, so regenerating a
  synthetic stream used to overwrite the references it is compared against.
  `--only fake` adds one and leaves the rest alone:

      python generate_environment.py --prob_dep --only fake \
          --sample_list df_fake_test_p12_02.pkl \
          --save_as df_pc_fake_test_p12_02.pkl

  `--prob_dep` still derives its departure distribution from the real stream,
  so that stream is recomputed in memory; it simply is not written back.
"""

import argparse
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import KDTree

from paths import (
    DATA,
    DATA_ENVIRONMENT,
    DATA_PREPROCESSED,
    DATA_SAMPLED,
    DATA_TRAINED,
    resolve,
)

# Même valeur que `collective_functions.DEFAULT_SEED`, redéfinie ici pour ne pas
# importer la simulation dans un script de préparation de données.
DEFAULT_SEED = 42


def reorg_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort an event stream so that RETURN events come before departures at the same timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least columns ["date", "departure"]. Departure is expected
        to be a dict, where RETURN events are encoded as `{0: 'RETURN'}`.

    Returns
    -------
    pd.DataFrame
        Reordered DataFrame, sorted by date and a computed `departure_sort` key.
    """
    df["departure_sort"] = df["departure"].apply(lambda x: 0 if x == {0: "RETURN"} else 1)
    return df.sort_values(by=["date", "departure_sort"]).drop(columns="departure_sort")


def generate_stations(df_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and normalize the fire stations table.

    Filters:
    - Type starts with 'CS'
    - Name does not start with X/Z
    - Etat == 'Disponible'
    - Unique center ID starts with 'B'

    Also normalizes some Toulouse station names and converts coordinates to numeric.

    Parameters
    ----------
    df_stations : pd.DataFrame
        Raw stations CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned stations table with columns ["Type", "Nom", "Coordonnée X", "Coordonnée Y"].
    """
    df_stations["Type"] = df_stations["Type"].fillna("")
    df_stations_u = df_stations[
        df_stations["Type"].str.startswith("CS")
        & ~df_stations["Nom"].str.startswith(("X", "Z"))
        & (df_stations["Etat"] == "Disponible")
        & (df_stations["Identifiant unique du centre"].str.startswith("B"))
    ]
    df_stations_u = df_stations_u[["Type", "Nom", "Coordonnée X", "Coordonnée Y"]].reset_index(drop=True)

    df_stations_u.loc[df_stations_u["Nom"] == "VION", "Nom"] = "TOULOUSE - VION"
    df_stations_u.loc[df_stations_u["Nom"] == "LOUGNON", "Nom"] = "TOULOUSE - LOUGNON"
    df_stations_u.loc[df_stations_u["Nom"] == "BUCHENS", "Nom"] = "RAMONVILLE - BUCHENS"
    df_stations_u.loc[df_stations_u["Nom"] == "MURET", "Nom"] = "MURET - MASSAT"
    # NOTE: the original code has an incomplete assignment for "ST LYS"; kept as-is.

    df_stations_u["Coordonnée X"] = df_stations_u["Coordonnée X"].str.replace(",", ".", regex=False)
    df_stations_u["Coordonnée Y"] = df_stations_u["Coordonnée Y"].str.replace(",", ".", regex=False)
    df_stations_u["Coordonnée X"] = pd.to_numeric(df_stations_u["Coordonnée X"], errors="coerce")
    df_stations_u["Coordonnée Y"] = pd.to_numeric(df_stations_u["Coordonnée Y"], errors="coerce")
    return df_stations_u


def generate_vehicles(df_vehicles: pd.DataFrame, df_stations_u: pd.DataFrame) -> pd.DataFrame:
    """
    Build the vehicles inventory per station.

    Parameters
    ----------
    df_vehicles : pd.DataFrame
        Raw material/vehicles CSV.
    df_stations_u : pd.DataFrame
        Clean stations table produced by `generate_stations`.

    Returns
    -------
    pd.DataFrame
        Grouped vehicles table by ["Nom du Centre", "Type materiel", "IU Materiel"],
        aggregating "Fonction materiel" into unique lists.
    """
    df_veh_u = df_vehicles[df_vehicles["Nom du Centre"].isin(df_stations_u["Nom"])]
    return (
        df_veh_u[["Nom du Centre", "IU Materiel", "Type materiel", "Fonction materiel"]]
        .groupby(["Nom du Centre", "Type materiel", "IU Materiel"])
        .agg(lambda x: list(set(x)))
        .reset_index()
    )


def generate_firefighters(df_firefighters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute firefighter skills validity windows and pivot them into a per-firefighter table.

    Parameters
    ----------
    df_firefighters : pd.DataFrame
        Raw skills CSV (one row per firefighter-skill record).

    Returns
    -------
    df_comp : pd.DataFrame
        Pivoted table with multi-index columns (skill, {Début, Fin}) and rows by Matricule.
    df_firefighters_clean : pd.DataFrame
        Cleaned long-format table (unique Matricule/Compétence with min start and max end).
    """
    df_firefighters = df_firefighters.rename(
        columns={
            "Compétence - Centre": "Centre",
            "Compétence - Nom": "Compétence",
            "Compétence - Date et heure de début": "Début",
            "Compétence -Date et heure de fin": "Fin",
        }
    )
    df_firefighters["Début"] = df_firefighters["Début"].apply(lambda x: x[:10])
    df_firefighters["Début"] = pd.to_datetime(df_firefighters["Début"].str.split().str[0], format="%d/%m/%Y")
    df_firefighters["Début"] = df_firefighters.groupby(["Matricule", "Compétence"])["Début"].transform("min")

    end_fallback_date = "01/01/2100"
    df_firefighters["Fin"] = df_firefighters["Fin"].fillna(end_fallback_date)
    df_firefighters["Fin"] = pd.to_datetime(df_firefighters["Fin"].str.split().str[0], format="%d/%m/%Y")
    df_firefighters["Fin"] = df_firefighters.groupby(["Matricule", "Compétence"])["Fin"].transform("max")

    df_firefighters = df_firefighters.drop_duplicates()
    df_firefighters = df_firefighters.drop_duplicates(subset=["Matricule", "Compétence"], keep="first")

    df_comp = df_firefighters.pivot_table(
        index=["Matricule"], columns="Compétence", values=["Début", "Fin"], aggfunc=lambda x: x
    )
    df_comp = df_comp.swaplevel(0, 1, axis=1).sort_index(axis=1)
    df_comp[("TOUTES", "Début")] = df_comp[[col for col in df_comp.columns if "Début" in col]].min(axis=1)
    df_comp[("TOUTES", "Fin")] = np.datetime64("2100-01-01T00:00:00")
    df_comp = df_comp.apply(pd.to_datetime)
    return df_comp, df_firefighters


def get_stations(x: float, y: float, df_pdd: gpd.GeoDataFrame, stations_u: list[str]) -> list[str]:
    """
    Return the list of station names (CIS) whose polygon contains a point.

    Parameters
    ----------
    x, y : float
        Point coordinates (expected CRS consistent with df_pdd).
    df_pdd : geopandas.GeoDataFrame
        GeoDataFrame with geometry and CIS columns (cis1, cis2, ...).
    stations_u : list[str]
        Allowed station names.

    Returns
    -------
    list[str]
        List of CIS names covering the point and present in `stations_u`.
    """
    point = Point([x, y])
    pdd = df_pdd.loc[:, df_pdd.columns.str.startswith("cis")].iloc[np.where(df_pdd.geometry.contains(point))].values[0]
    return [v for v in pdd if v in stations_u]


def distance_euclidienne(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def trier_villes_par_distance(df: pd.DataFrame, ville_z1: str, villes_z2: list[str]) -> dict[str, int]:
    """
    Build an ordered distance map from one station to a list of stations (in km).

    Returns
    -------
    dict[str, int]
        Mapping station -> rounded distance in kilometers, sorted by distance.
    """
    x1, y1 = df.loc[df["Nom"] == ville_z1, ["Coordonnée X", "Coordonnée Y"]].values[0]
    distances = []
    for ville_z2 in villes_z2:
        x2, y2 = df.loc[df["Nom"] == ville_z2, ["Coordonnée X", "Coordonnée Y"]].values[0]
        dist = distance_euclidienne(x1, y1, x2, y2) / 1000.0
        distances.append((ville_z2, int(dist)))
    distances.sort(key=lambda x: x[1])
    return {ville: dist for ville, dist in distances}


def get_zone(pdd: list[str], Z_1: list[str], Z_2: list[str], Z_3: list[str]) -> str:
    """
    Assign a zone label Z_1/Z_2/Z_3/Z_4 based on the first CIS in PDD.
    """
    if pdd[0] in Z_1:
        return "Z_1"
    if pdd[0] in Z_2:
        return "Z_2"
    if pdd[0] in Z_3:
        return "Z_3"
    return "Z_4"


def precompute_pdd(df_pdd: gpd.GeoDataFrame, df_sample: pd.DataFrame, stations_u: list[str]) -> pd.DataFrame:
    """
    Add a "PDD" column containing the list of candidate stations per intervention.
    """
    cols = [col for col in df_pdd.columns if col.startswith("cis")] + ["geometry"]
    df_pdd = df_pdd[cols]
    df_sample["PDD"] = df_sample.apply(
        lambda row: get_stations(row["Coord X"], row["Coord Y"], df_pdd, stations_u),
        axis=1,
    )
    return df_sample


def precompute_zone(df_stations: pd.DataFrame, df_sample: pd.DataFrame, Z_1: list[str], Z_2: list[str], Z_3: list[str]) -> pd.DataFrame:
    """
    Add a "zone" column and (optionally) precompute distance maps (Z_1 -> Z_2/Z_3).
    """
    _ = {ville_z1: trier_villes_par_distance(df_stations, ville_z1, Z_2 + Z_3) for ville_z1 in Z_1}
    df_sample["zone"] = df_sample["PDD"].apply(get_zone, args=(Z_1, Z_2, Z_3))
    return df_sample


def precompute_incident(df_rank_incident: pd.DataFrame, df_sample: pd.DataFrame) -> pd.DataFrame:
    """
    Map incident rank -> incident label and store it in `incident_name`.
    """
    dict_result = df_rank_incident.set_index("rank")["sin"].to_dict()
    df_sample["incident_name"] = df_sample["Incident"].map(dict_result)
    return df_sample


def get_area_type(
    x: float,
    y: float,
    df_xy: pd.DataFrame,
    df_lieu: pd.DataFrame,
    df_nom_commune: pd.DataFrame,
    df_commune: pd.DataFrame,
    df_secteur: pd.DataFrame,
    data: np.ndarray,
    tree: KDTree,
) -> str:
    """
    Infer a semantic 'area_type' (sector type) for a coordinate using nearest-neighbor lookup.

    The method finds the nearest known "lieu" point (KDTree on X/Y), then traverses
    location hierarchies to retrieve the sector type.
    """
    point = np.array([[x, y]])
    _, ind = tree.query(point, k=1)
    nearest_point = data[ind[0][0]]
    X, Y = nearest_point
    id_lieu = df_xy[(df_xy["Coordonnées X"] == X) & (df_xy["Coordonnées Y"] == Y)]["Identifiant unique du lieu"].iloc[0]

    if id_lieu in df_lieu:
        while id_lieu:
            part_of = df_lieu[df_lieu["NUMERO_LIEU"] == id_lieu]["EST_SITUE_SUR_NUMERO_LIEU"].iloc[0]
            num_secteur = df_lieu[df_lieu["NUMERO_LIEU"] == id_lieu]["NUMERO_TYPE_SECTEUR"].iloc[0]
            id_lieu = int(part_of)
    else:
        nom_com = df_xy[df_xy["Identifiant unique du lieu"] == id_lieu]["Nom de la commune"].iloc[0]
        num_com = df_nom_commune[df_nom_commune["NOM_COMMUNE"] == nom_com]["NUMERO_COMMUNE"].iloc[0]
        num_secteur = df_commune[df_commune["NUMERO_COMMUNE"] == num_com]["NUMERO_TYPE_SECTEUR"].iloc[0]

    return df_secteur[df_secteur["NUMERO_TYPE_SECTEUR"] == num_secteur]["TYPE_SECTEUR"].iloc[0]


def precompute_area_type(
    df_xy: pd.DataFrame,
    df_lieu: pd.DataFrame,
    df_nom_commune: pd.DataFrame,
    df_commune: pd.DataFrame,
    df_secteur: pd.DataFrame,
    df_sample: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add an `area_type` column to df_sample using nearest-neighbor lookup via KDTree.
    """
    df_xy = df_xy.dropna(subset=["Coordonnées X", "Coordonnées Y"])
    data = df_xy[["Coordonnées X", "Coordonnées Y"]].to_numpy()
    tree = KDTree(data, leaf_size=40)

    df_lieu["EST_SITUE_SUR_NUMERO_LIEU"] = df_lieu["EST_SITUE_SUR_NUMERO_LIEU"].fillna(0)

    df_sample["area_type"] = df_sample.apply(
        lambda row: get_area_type(
            row["Coord X"],
            row["Coord Y"],
            df_xy,
            df_lieu,
            df_nom_commune,
            df_commune,
            df_secteur,
            data,
            tree,
        ),
        axis=1,
    )
    return df_sample


def precompute_prob_dict(df_inter_clean: pd.DataFrame) -> dict:
    """
    Compute a nested probability dictionary P(departure | incident_name, area_type).

    Returns
    -------
    dict
        prob_dict[incident_name][area_type][tuple(real_func)] = probability
    """
    nested_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for _, row in df_inter_clean.iterrows():
        incident = row["incident_name"]
        area = row["area_type"]
        list_of_strings = tuple(row["real_func"])
        nested_dict[incident][area][list_of_strings] += 1

    nested_dict = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in nested_dict.items()}
    prob_dict = defaultdict(lambda: defaultdict(dict))

    for incident, areas in nested_dict.items():
        for area, string_counts in areas.items():
            total = sum(string_counts.values())
            prob_dict[incident][area] = {strings: count / total for strings, count in string_counts.items()}

    return {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in prob_dict.items()}


def prob_departure(area_type: str, inc_name: str, prob_dict: dict,
                   rng: random.Random | None = None) -> dict:
    """
    Sample a departure dictionary from probabilities.

    Fallback: if area_type not found, tries "*" (global) area key.

    Returns
    -------
    dict
        Departure mapping like {1: [veh1], 2: [veh2], ...} or {0: "RETURN"} (elsewhere).
    """
    draw = rng if rng is not None else random
    if area_type in prob_dict[inc_name]:
        choices = list(prob_dict[inc_name][area_type].keys())
        weights = list(prob_dict[inc_name][area_type].values())
        dep_tuple = draw.choices(choices, weights=weights, k=1)[0]
    elif "*" in prob_dict[inc_name]:
        choices = list(prob_dict[inc_name]["*"].keys())
        weights = list(prob_dict[inc_name]["*"].values())
        dep_tuple = draw.choices(choices, weights=weights, k=1)[0]
    else:
        # Unspecified behavior in original code: keep it explicit
        dep_tuple = tuple()

    return {i + 1: [val] for i, val in enumerate(dep_tuple)}


def precompute_prob_departure(df_sample: pd.DataFrame, prob_dict: dict,
                              seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Add a 'departure' column by sampling from the probability dictionary.

    The draw is seeded per call, on a generator of its own rather than the
    global `random`. Sharing the global one made the result depend on how much
    sampling had happened earlier in the process: building the real stream
    before a synthetic one shifted the latter's draws by 53 088 positions, and
    changed 17 255 of its 127 392 departures. Each stream now draws the same
    way whatever ran before it.

    Passing an explicit `seed` gives independent draws for successive years --
    that is how `build_fake` keeps a decade from being ten copies of one year.
    """
    rng = random.Random(seed)
    df_sample["departure"] = df_sample.apply(
        lambda row: prob_departure(row["area_type"], row["incident_name"],
                                   prob_dict, rng),
        axis=1,
    )
    return df_sample


def get_departure(area_type: str, inc_name: str, dic_inc_ar_mat: dict) -> dict:
    """
    Deterministic departure lookup from the incident/area/material rules dictionary.
    """
    if area_type in dic_inc_ar_mat[inc_name].keys():
        return dic_inc_ar_mat[inc_name][area_type]
    if "*" in dic_inc_ar_mat[inc_name].keys():
        return dic_inc_ar_mat[inc_name]["*"]
    new_area_dic = {k: v for k, v in dic_inc_ar_mat[inc_name].items() if v != ""}.values()
    return next(iter(new_area_dic))


def precompute_departure(df_sample: pd.DataFrame, dic_inc_ar_mat: dict) -> pd.DataFrame:
    """
    Add a 'departure' column using deterministic rules.
    """
    df_sample["departure"] = df_sample.apply(
        lambda row: get_departure(row["area_type"], row["incident_name"], dic_inc_ar_mat),
        axis=1,
    )
    return df_sample


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def add_leap_day(
    df_sample: pd.DataFrame, start_year: int, seed: int = 42
) -> pd.DataFrame:
    """Give a leap year its 366th day, by replaying a nearby sampled day.

    The generator draws each synthetic year from a model fitted on a single
    365-day year, so `Day` never exceeds 365. Laid onto a leap year that leaves
    the calendar a day short: the year would run to 30 December and skip the
    31st, putting a one-day hole in an otherwise continuous stream. Which years
    are affected follows from `--start_year`.

    Day 366 falls on 31 December, so the donor is drawn from the turn of the
    year — late December or early January — rather than from the year at large.
    Activity is strongly seasonal, roughly 162 interventions a day in December
    against 202 in June and with a different incident mix, so replaying an
    arbitrary day would land a summer profile on a winter date and distort both
    the monthly totals and the incident distribution. Days on either side of
    31 December share its volume, hour-of-day profile, incident types and
    spatial spread, and 1 January is as close a neighbour as 30 December.

    The cost is that day 366 is not an independent draw. That is deliberate: one
    replayed day in 366 is a smaller distortion than a missing day, and doing it
    here rather than inside `precompute_date` keeps it visible.
    """
    if not is_leap_year(start_year):
        return df_sample

    # `Day` is still day-of-year here: 335+ is December, 1-15 is early January.
    # Both flank 31 December, so either is a plausible stand-in for it.
    window = df_sample["Day"].between(335, 365) | df_sample["Day"].between(1, 15)
    candidates = df_sample.loc[window, "Day"].unique()
    if not len(candidates):  # neither end present: fall back to the whole year
        candidates = df_sample["Day"].unique()

    donor_day = np.random.default_rng(seed + start_year).choice(np.sort(candidates))
    leap_day = df_sample[df_sample["Day"] == donor_day].copy()
    leap_day["Day"] = 366

    return pd.concat([df_sample, leap_day], ignore_index=True)


def precompute_date(df_sample: pd.DataFrame, start_year: int, seed: int = 42) -> pd.DataFrame:
    """
    Build an absolute datetime from Day/Hour/Minute and overwrite Month/Day/Hour/Minute from it.
    """
    np.random.seed(seed)
    if "Minute" not in df_sample.columns:
        df_sample["Minute"] = np.random.randint(0, 60, size=len(df_sample))
    df_sample["date"] = (
        datetime(start_year, 1, 1)
        + pd.to_timedelta(df_sample["Day"] - 1, unit="D")
        + pd.to_timedelta(df_sample["Hour"], unit="h")
        + pd.to_timedelta(df_sample["Minute"], unit="m")
    )
    df_sample["Month"] = df_sample["date"].dt.month
    df_sample["Day"] = df_sample["date"].dt.day
    df_sample["Hour"] = df_sample["date"].dt.hour
    df_sample["Minute"] = df_sample["date"].dt.floor("s").dt.minute
    return df_sample


def precompute_returns(df_sample: pd.DataFrame, start_inter: int, end_inter: int, is_fake: bool) -> pd.DataFrame:
    """
    Create RETURN events and merge them with departure events, sorted stably.

    Returns
    -------
    pd.DataFrame
        Combined event stream containing both returns and departures.
    """
    df_sample["delta"] = pd.to_timedelta(df_sample["Duration"], unit="m")
    df_sample["date_return"] = df_sample["date"] + df_sample["delta"]
    df_sample_sorted = df_sample.sort_values(by="date").reset_index(drop=True)
    df_sample_sorted["num_inter"] = range(start_inter, end_inter + 1)

    df_sample_short = df_sample_sorted[
        ["num_inter", "date", "PDD", "departure", "zone", "Duration", "Month", "Day", "Hour", "Minute"]
    ].copy()

    df_sample_short.loc[
        :, ["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]
    ] = df_sample_sorted[
        ["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]
    ]

    df_return = pd.DataFrame()
    df_return["num_inter"] = df_sample_sorted["num_inter"]
    df_return["date"] = df_sample_sorted["date_return"]
    df_return["PDD"] = [[] for _ in range(len(df_sample_sorted))]
    df_return["departure"] = [{0: "RETURN"} for _ in range(len(df_sample_sorted))]
    df_return["zone"] = ""
    df_return["Duration"] = 0
    df_return[["Month", "Day", "Hour", "Minute"]] = df_return["date"].apply(lambda x: pd.Series([x.month, x.day, x.hour, x.minute]))
    df_return[["Coord X", "Coord Y", "Month_sin", "Month_cos", "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]] = (
        0, 0, 0, 0, 0, 0, 0, 0
    )

    # Put returns first to handle same-timestamp return/departure ordering.
    df_combined = pd.concat([df_return, df_sample_short], ignore_index=True)
    return df_combined.sort_values(by="date", kind="mergesort").reset_index(drop=True)


def extract_bracket_number_and_clean(materiel: str) -> dict[int, list[str]]:
    """
    Parse a serialized materials string into a dict: step_number -> list of materials.

    The parser expects patterns like "[1] MAT1 |-- ou MAT2 [2] MAT3 ...".
    """
    result: dict[int, list[str]] = {}
    split_by_number = re.split(r"(\[\d+\])", materiel)

    current_num = None
    for part in split_by_number:
        if re.match(r"\[\d+\]", part):
            current_num = int(part.strip("[]"))
            result[current_num] = []
        elif current_num is not None:
            materials = re.split(r" \|-- ou ", part)
            for material in materials:
                combined_words = " ".join(re.findall(r"[A-Z\[\]_\- ]+\d*", material))
                if combined_words.strip():
                    result[current_num].append(combined_words.strip())
    return result


def create_responses(df_responses: pd.DataFrame, df_rank_incident: pd.DataFrame) -> dict:
    """
    Build deterministic departure rules from the responses table.

    Returns
    -------
    dict
        dic_inc_ar_mat[incident_name][sector][step_number] = list_of_materials
    """
    dic_replace = {
        "CCF DEGRAD": "CCF",
        "XCOMPL": "COMPL",
        "VPL": "VSN",
        "EMB": "VEMB",
        "CEIN": "VSAV",
        "EPA DEGRAD": "EPC18",
        "VL[X-GPT OPERATION]": "VL",
        "VSN[TOULOUSE - LOUGNON]": "VSN",
        "VFT[ST GAUDENS]": "VFT",
        "VSR[COLOMIERS]": "VSR",
        "VL[X-CTA CODIS]": "VL",
        "CESDMF": "CESD",
        "FPT   MPR": "FPT",
        "VLHR[Z CRS LUCHON]": "VLHR",
        "PSECINC    MPR": "PSECINC",
        "PSECINC2    MPR": "PSECINC2",
    }
    df_responses["Materiel"] = df_responses["Materiel"].replace(dic_replace)
    df_responses = df_responses.dropna(subset=["Materiel"]).reset_index(drop=True)

    dict_result = df_rank_incident.set_index("rank")["sin"].to_dict()
    allowed_incidents = set(dict_result.values())

    df_responses_short = df_responses[df_responses["Nom"].isin(allowed_incidents)].reset_index(drop=True)
    dic_inc_ar_mat: dict = {}

    for _, row in df_responses_short.iterrows():
        nom = row["Nom"]
        secteur = row["Secteur"]
        materiel = row["Materiel"]
        materiel_dict = extract_bracket_number_and_clean(materiel)

        dic_inc_ar_mat.setdefault(nom, {})
        dic_inc_ar_mat[nom].setdefault(secteur, {})

        for num, materials in materiel_dict.items():
            dic_inc_ar_mat[nom][secteur][num] = materials

    return dic_inc_ar_mat


def create_dic_planning(chemin_dossier: str) -> dict:
    """
    Build the hierarchical planning dictionary from CSV files.

    Returns
    -------
    dict
        planning[centre][month][day][hour] = {"planned": [...], "available": [...], "standby": [...]}
    """
    fichiers_csv = [f for f in os.listdir(chemin_dossier) if f.endswith(".csv")]
    liste_df = []
    for fichier in fichiers_csv:
        chemin_fichier = os.path.join(chemin_dossier, fichier)
        liste_df.append(pd.read_csv(chemin_fichier, sep=";"))

    df_plan = pd.concat(liste_df, ignore_index=True)
    df_plan["Date Heure de début de tranche planning"] = pd.to_datetime(
        df_plan["Date Heure de début de tranche planning"], format="%d/%m/%Y %H:%M:%S"
    )
    df_plan["Mois"] = df_plan["Date Heure de début de tranche planning"].dt.month
    df_plan["Jour"] = df_plan["Date Heure de début de tranche planning"].dt.day

    list_ff_not_in_df_skills = [
        np.int64(9977),
        np.int64(9282),
        np.int64(8227),
        np.int64(9755),
        np.int64(9953),
        np.int64(9439),
        np.int64(10009),
        np.int64(10049),
        np.int64(10067),
        np.int64(10066),
        np.int64(10053),
    ]

    planning: dict = {}
    for (centre, mois, jour, heure), group in df_plan.groupby(["Nom Centre", "Mois", "Jour", "Heure"]):
        planning.setdefault(centre, {}).setdefault(mois, {}).setdefault(jour, {})
        mat_filtered = [m for m in group["Matricule"].unique().tolist() if m not in list_ff_not_in_df_skills]
        planning[centre][mois][jour][heure] = {"planned": mat_filtered, "available": mat_filtered, "standby": []}

    for centre in planning.keys():
        for mois in planning[centre]:
            for jour in range(1, 32):
                planning[centre][mois].setdefault(jour, {})
                for heure in range(0, 24):
                    planning[centre][mois][jour].setdefault(heure, {"planned": [], "available": [], "standby": []})

    return planning


# Zones de rattachement, du plus dense au plus rural. Elles pilotent la
# reconstruction du départ et ne dépendent d'aucune donnée d'entrée.
Z_1 = ["TOULOUSE - LOUGNON", "TOULOUSE - VION"]
Z_2 = ["ST JORY", "ROUFFIAC", "RAMONVILLE - BUCHENS", "COLOMIERS", "MURET - MASSAT"]
Z_3 = ["AUTERIVE", "ST LYS", "GRENADE", "FRONTON", "VERFEIL", "CARAMAN"]

# Colonnes ramenées dans [0, 1] avant écriture d'un flux.
COLS_TO_NORM = ["Coord X", "Coord Y", "Month_sin", "Month_cos",
                "Day_sin", "Day_cos", "Hour_sin", "Hour_cos"]

ALL_TARGETS = ("tables", "real", "fake", "planning")


@dataclass
class Sources:
    """Tables brutes et référentiels, partagés par les deux flux."""

    df_pdd: gpd.GeoDataFrame
    df_stations: pd.DataFrame
    stations_u: list[str]
    df_v: pd.DataFrame
    df_skills: pd.DataFrame
    df_roles: pd.DataFrame
    df_vehicles_history: pd.DataFrame
    df_xy: pd.DataFrame
    df_lieu: pd.DataFrame
    df_commune: pd.DataFrame
    df_nom_commune: pd.DataFrame
    df_secteur: pd.DataFrame
    df_prob_dep: pd.DataFrame
    df_rank_incident: pd.DataFrame
    dic_inc_ar_mat: dict


def load_sources() -> Sources:
    """Lit les CSV/GeoJSON d'entrée et construit les tables dérivées."""
    df_firestations = pd.read_csv(DATA / "firestations.csv", sep=";")
    df_materiel = pd.read_csv(DATA / "materiel.csv", sep=";")
    df_comp = pd.read_csv(DATA / "comp.csv", sep=";")
    df_roles = pd.read_csv(DATA / "roles_competences.csv", sep=";")
    df_vehicles_history = pd.read_csv(DATA / "df_vehicles_history.csv", sep=";")
    df_responses = pd.read_csv(DATA / "responses_by_incident.csv", sep=";")

    df_stations = generate_stations(df_firestations)
    stations_u = sorted(
        x for x in df_materiel["Nom du Centre"].unique()
        if not x.startswith("X") and not x.startswith("Z")
    )
    df_skills, _df_firefighters = generate_firefighters(df_comp)
    df_rank_incident = pd.read_pickle(DATA_PREPROCESSED / "df_rank_incident.pkl")

    return Sources(
        df_pdd=gpd.read_file(DATA / "pdd.geojson"),
        df_stations=df_stations,
        stations_u=stations_u,
        df_v=generate_vehicles(df_materiel, df_stations),
        df_skills=df_skills,
        df_roles=df_roles,
        df_vehicles_history=df_vehicles_history,
        df_xy=pd.read_csv(DATA / "X-Y-lieu.csv", sep=";"),
        df_lieu=pd.read_csv(DATA / "dbo.LIEU.csv", sep=";"),
        df_commune=pd.read_csv(DATA / "dbo.COMMUNE.csv", sep=";"),
        df_nom_commune=pd.read_csv(DATA / "dbo.NOM_COMMUNE.csv", sep=";"),
        df_secteur=pd.read_csv(DATA / "dbo.SECTEUR.csv", sep=";"),
        df_prob_dep=pd.read_pickle(DATA_PREPROCESSED / "df_prob_dep.pkl"),
        df_rank_incident=df_rank_incident,
        dic_inc_ar_mat=create_responses(df_responses, df_rank_incident),
    )


def write_tables(src: Sources) -> None:
    """Écrit les référentiels que lisent la simulation et les cas golden."""
    src.df_stations.to_pickle(DATA_ENVIRONMENT / "df_stations.pkl")
    src.df_v.to_pickle(DATA_ENVIRONMENT / "df_v.pkl")
    src.df_skills.to_pickle(DATA_ENVIRONMENT / "df_skills.pkl")
    src.df_roles.to_pickle(DATA_ENVIRONMENT / "df_roles.pkl")
    src.df_vehicles_history.to_pickle(DATA_ENVIRONMENT / "df_vehicles_history.pkl")


def locate(df: pd.DataFrame, src: Sources) -> pd.DataFrame:
    """Situe chaque intervention : casernes, zone, type d'incident, secteur.

    Identique pour le flux réel et le flux synthétique -- c'est la seule
    géographie que le reste du script suppose déjà calculée.
    """
    df = precompute_pdd(src.df_pdd, df, src.stations_u)
    df = precompute_zone(src.df_stations, df, Z_1, Z_2, Z_3)
    df = precompute_incident(src.df_rank_incident, df)
    return precompute_area_type(src.df_xy, src.df_lieu, src.df_nom_commune,
                                src.df_commune, src.df_secteur, df)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ramène les colonnes continues dans [0, 1]. Une colonne constante vaut 0."""
    for col in COLS_TO_NORM:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo) if lo != hi else 0.0
    return df


def build_real(src: Sources, prob_dep: bool, start_year: int,
               write: bool = True,
               seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame | None, str, dict | None]:
    """Construit le flux réel, et la table de probabilités qu'il porte.

    Renvoie aussi `prob_dict` parce que le flux synthétique probabiliste en
    dépend : les départs y sont tirés de la distribution observée sur le réel.

    `write=False` s'arrête dès que `prob_dict` est disponible. Ce n'est pas
    qu'une économie : `prob_departure` tire dans le `random` global, non semé,
    donc chaque tirage supplémentaire décale ceux du flux synthétique. Aller
    jusqu'au bout du flux réel alors qu'on ne le garde pas changerait les
    départs tirés ensuite -- 17 255 lignes sur 127 392 lors du refactor.
    """
    df = pd.read_pickle(DATA_TRAINED / "df_real.pkl")
    window = len(df)
    print("window real:", window)

    df = locate(df, src)

    prob_dict = None
    if prob_dep:
        df = pd.concat([df, src.df_prob_dep], axis=1)
        prob_dict = precompute_prob_dict(df)
        print("prob_dict computed")
        if not write:
            return None, "_prob", prob_dict
        df = precompute_prob_departure(df, prob_dict, seed)
        suffix = "_prob"
    else:
        if not write:
            return None, "", None
        df = precompute_departure(df, src.dic_inc_ar_mat)
        suffix = ""

    print("start_year", start_year, "start_inter", 1, "end_inter", window)
    df = precompute_date(df, start_year)
    df = precompute_returns(df, 1, window, False)
    print("real", len(df), "done")

    return normalise(df), suffix, prob_dict


def build_fake(src: Sources, sample_list: list[str], prob_dep: bool,
               start_year: int, prob_dict: dict | None,
               seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Enchaîne les années échantillonnées en une seule frise continue.

    `num_inter` se poursuit d'une année sur l'autre, de sorte que dix années
    forment une timeline et non dix qui se recouvrent.
    """
    frames = []
    start_inter, end_inter, window = 1, 0, 0

    for sample_file in sample_list:
        start_inter += window
        df = pd.read_pickle(resolve(sample_file, DATA_SAMPLED))
        # Le 29 février doit être ajouté avant le comptage : `window` fixe la
        # plage de num_inter attribuée par precompute_returns, et des lignes
        # ajoutées après laisseraient la numérotation trop courte.
        df = add_leap_day(df, start_year)

        window = len(df)
        end_inter += window
        print("window fake:", window)
        print("start_year", start_year, "start_inter", start_inter, "end_inter", end_inter)

        df = locate(df, src)
        if prob_dep:
            # Décalée par l'année : sans cela, dix années tireraient les mêmes
            # départs et la décennie ne serait qu'une seule année répétée.
            df = precompute_prob_departure(df, prob_dict, seed + start_year)
        else:
            df = precompute_departure(df, src.dic_inc_ar_mat)

        df = precompute_date(df, start_year)
        df = precompute_returns(df, start_inter, end_inter, True)

        print(sample_file, len(df), "done")
        frames.append(df)
        start_year += 1

    if not frames:
        return pd.DataFrame()
    return normalise(reorg_dates(pd.concat(frames, ignore_index=True)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment params")
    parser.add_argument("--prob_dep", action="store_true", help="if departure is computed from probabilities")
    parser.add_argument("--sample_list", nargs="+", help="List of samples to use")
    parser.add_argument("--save_as", type=str, help="Output pickle name for the combined fake stream")
    parser.add_argument(
        "--start_year",
        type=int,
        default=datetime.now().year,
        help="calendar year the first generated year is stamped with; each "
             "further sample takes the next year (default: the current year)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for the probabilistic departure draw (default: {DEFAULT_SEED}). "
             "Each synthetic year offsets it by its own calendar year, so a "
             "decade stays ten distinct draws.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_TARGETS,
        help="artefacts to (re)generate; default: all of them. Generating only "
             "the fake stream leaves the golden inputs -- df_pc_real*.pkl, "
             "planning.pkl, df_skills.pkl, df_stations.pkl, df_v.pkl, "
             "df_roles.pkl, df_vehicles_history.pkl -- untouched.",
    )
    args = parser.parse_args(argv)

    # Sans `--only`, tout est régénéré : le comportement historique, où une
    # invocation sans `--sample_list` produisait simplement les tables, le flux
    # réel et le planning. Ne valider que si `fake` a été demandé
    # explicitement, sinon cet appel-là cesserait de fonctionner.
    args.targets = set(args.only) if args.only else set(ALL_TARGETS)
    if args.only and "fake" in args.targets:
        if not args.sample_list:
            parser.error("--only fake demande --sample_list")
        if not args.save_as:
            parser.error("--only fake demande --save_as")
    return args


def real_stream_plan(targets: set[str], prob_dep: bool) -> tuple[bool, bool]:
    """(construire, écrire) le flux réel.

    Le flux synthétique probabiliste tire ses départs de `prob_dict`, qui se
    déduit du réel : `--only fake --prob_dep` doit donc le construire. Mais
    c'est l'écriture, et elle seule, qui écrase les entrées des cas golden --
    d'où les deux booléens plutôt qu'un.
    """
    write = "real" in targets
    build = write or (prob_dep and "fake" in targets)
    return build, write


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée : construit les artefacts demandés par `--only`."""
    args = parse_args(argv)
    targets = args.targets

    print("is probabilistic departure", args.prob_dep)
    print("targets:", " ".join(sorted(targets)))

    src = load_sources()

    DATA_ENVIRONMENT.mkdir(parents=True, exist_ok=True)
    if "tables" in targets:
        write_tables(src)
        print("tables done")
    else:
        print("tables skipped")

    # Le flux synthétique probabiliste tire ses départs de `prob_dict`, qui se
    # déduit du flux réel. `--only fake --prob_dep` doit donc quand même
    # construire le réel, mais sans l'écrire : c'est l'écriture, et elle seule,
    # qui écrase les entrées des cas golden.
    prob_dict = None
    build, write = real_stream_plan(targets, args.prob_dep)
    if build:
        df_real, suffix, prob_dict = build_real(src, args.prob_dep,
                                                args.start_year, write=write,
                                                seed=args.seed)
        if write:
            df_real.to_pickle(DATA_ENVIRONMENT / f"df_pc_real{suffix}.pkl")
        else:
            print("real stream stopped at prob_dict, not written")
    else:
        print("real skipped")

    if "fake" in targets:
        df_fake = build_fake(src, args.sample_list or [], args.prob_dep,
                             args.start_year, prob_dict, seed=args.seed)
        if len(df_fake):
            df_fake.to_pickle(resolve(args.save_as, DATA_ENVIRONMENT))
            print("global fake done", len(df_fake))
        else:
            print("No fake samples provided; skipping fake stream generation.")
    else:
        print("fake skipped")

    if "planning" in targets:
        planning = create_dic_planning(str(DATA / "Planning") + "/")
        with open(DATA_ENVIRONMENT / "planning.pkl", "wb") as f:
            pickle.dump(planning, f)
        print("Planning done")
    else:
        print("planning skipped")


if __name__ == "__main__":
    main()
