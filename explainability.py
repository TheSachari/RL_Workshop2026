import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from collective_functions import update_skills
from paths import DATA_ENVIRONMENT, DATA_SAMPLED, resolve

tqdm.pandas()



def get_dic_rare_skills(rare_skills, ff_array):
    if rare_skills.size == 0:
        return {i: [] for i in range(ff_array.shape[0])}
    sub = ff_array[:, rare_skills]   

    return {i: rare_skills[np.flatnonzero(sub[i])].astype(int) for i in range(ff_array.shape[0])}

def get_related_rows_in_time(idx, date, pdd, df, top_n = 5, n_hours=2):

    limit_time = date + timedelta(hours=n_hours)
    target_villes = pdd[:top_n+1]
    selected_skills = []

    for j in range(idx + 1, len(df)):
        t = df.at[j, "date"]
        if t > limit_time:  # on sort dès qu'on dépasse la fenêtre
            break
        if df.at[j, "PDD"]:
            first_city = df.at[j, "PDD"][0]
            if first_city in target_villes:
                selected_skills.extend(df.at[j, "rare_skills_required"])

    return np.unique(np.concatenate(selected_skills, dtype=int)) if selected_skills else np.array([], dtype=int)

def get_skill_counts(date, df_skills):

    skills_updated = update_skills(df_skills, date)
    counts = np.count_nonzero(skills_updated, axis=0)
    return counts

def get_all(date, required_departure, df_skills, rarity, dic_roles, dic_roles_skills):
    counts = np.count_nonzero(update_skills(df_skills, date), axis=0)
    return get_rare_skills_from_dep(required_departure, counts, rarity, dic_roles, dic_roles_skills)

def get_rare_skills_from_dep(required_departure, counts, rarity, dic_roles, dic_roles_skills):
    list_veh = [val for lst in required_departure.values() for val in lst]
    required_roles = set(val for d in [dic_roles[v] for v in list_veh] for val in d.values())
    return np.unique(np.concatenate([get_rare_skills(counts, dic_roles_skills[role]) for role in required_roles]))

def get_rare_skills(rare_skills, ff_skills, rarity = 50, skill_lvl_gt = 0):
    pos_in_b = ff_skills > skill_lvl_gt                   
    a_lt50   = rare_skills < rarity                    
    match    = pos_in_b & a_lt50[None,:]
    return np.unique(np.concatenate([np.where(row)[0] for row in match]))

def get_all_planed_ff(planning, month, day, hour, df_stations):
    return np.concatenate([planning[c][month][day][hour]["planned"] for c in df_stations["Nom"].values])


# --- Scoped rarity -----------------------------------------------------------
#
# Rarity is a property of the roster that loses the firefighter, not of the
# department. Pooling all 34 stations, as `get_rare_skills_from_planed_ff` does,
# misses 89% of the skills that are scarce where the decision is actually made:
# a station holds ~39 of the 134 skills with a median crew of 17, so a skill
# carried by 40 people department-wide can still be the only one here.
#
# Two scales, because "scarce" and "irreplaceable" are different questions:
#
#   local        -- the sole holder on this station's shift. Assigning them to a
#                   subordinate role removes the skill from the station for the
#                   shift; from two holders on, one remains. This is the only
#                   threshold resting on a mechanism rather than a calibration.
#   neighbourhood-- also scarce across the stations that would be called next,
#                   so no nearby station can cover the gap. Roughly half of the
#                   sole-holder skills fall here; the other half are locally
#                   scarce but abundant next door, and the deployment order
#                   absorbs them.
#
# Measured on the reference year, the split is stable across both kinds of step
# (48% irreversible on departures, 47% on reinforcements) and across a 2.2x
# range of neighbourhood sizes, which is why both thresholds are absolute. A
# rank (`top_k`) is deliberately *not* applied here: the number of irreversible
# skills should rise when the neighbourhood is thin, and a rank would flatten
# exactly that signal.

LOCAL_RARITY = 2
NEIGH_RARITY = 6


def scoped_rare_skills(local_counts, neigh_counts,
                       local_rarity=LOCAL_RARITY, neigh_rarity=NEIGH_RARITY):
    """Split a station's scarce skills into absorbable and irreversible.

    Returns `(local, irreversible)`, both arrays of skill ids, with
    `irreversible` a subset of `local`. Callers that want the absorbable ones
    take the difference.
    """
    local = np.flatnonzero((local_counts > 0) & (local_counts < local_rarity))
    if local.size == 0:
        return local, local
    scarce_around = (neigh_counts[local] > 0) & (neigh_counts[local] < neigh_rarity)
    return local, local[scarce_around]


def station_neighbourhood(station, pdd, dic_station_distance, n_following=5):
    """The stations that could absorb a gap opened at `station`.

    Ordered by the PDD when the step belongs to an intervention -- that is the
    order the departure will actually work through, so it beats geometric
    distance -- and by distance when it does not.

    Reinforcement steps are the second case. They have no PDD, and
    `get_neighborhood_availability` zeroes its features for them, but the
    scarcity question still applies and applies to the *sender*: the crew leaves
    that station and stays away for the travel time plus 20 minutes, with no
    local incident to justify it. Measured on the 11 Z2/Z3 stations that can
    send, a sender's roster is thinner than a typical first-due station's (13 vs
    17) and so is its neighbourhood (80 vs 116), so these are the steps where
    spending the last holder costs most -- not the ones to skip.
    """
    if pdd and station in pdd:
        index = pdd.index(station)
        return pdd[index + 1:index + 1 + n_following]

    ordering = dic_station_distance.get(station) if dic_station_distance else None
    if not ordering:
        return []
    return [s for s in ordering if s != station][:n_following]


def rare_skills_for_step(st, n_following=5, cache=None,
                         local_rarity=LOCAL_RARITY, neigh_rarity=NEIGH_RARITY):
    """`(local, irreversible)` skill ids for the decision `st` is about to make.

    Computed here rather than precomputed into the event stream because the
    scope depends on the step, not only on the hour: `st.current_station` is
    known only once the departure has walked down the PDD, and a reinforcement's
    sender is chosen at the moment it is sent. A column on the event row cannot
    carry that without storing one entry per station per event.

    `cache` keys on (station, neighbourhood, calendar day, hour). The crew comes
    from `planning[station][month][day][hour]`, and `update_skills` compares
    against validity windows that start and end at midnight, so the counts are
    constant within an hour for a given scope. Every role of every vehicle at a
    station shares one entry.
    """
    station = st.current_station
    pdd = st.pdd if getattr(st, "pdd", None) else []
    neighbourhood = tuple(
        station_neighbourhood(station, pdd, st.dic_station_distance, n_following)
    )

    if cache is not None:
        key = (station, neighbourhood, st.date.normalize(), st.hour)
        hit = cache.get(key)
        if hit is not None:
            return hit

    args = (st.planning, st.month, st.day, st.hour, st.df_skills, st.date)
    local_counts = crew_skill_counts([station], *args)
    neigh_counts = crew_skill_counts([station, *neighbourhood], *args)
    result = scoped_rare_skills(local_counts, neigh_counts, local_rarity, neigh_rarity)

    if cache is not None:
        cache[key] = result
    return result


def crew_skill_counts(stations, planning, month, day, hour, df_skills, date):
    """How many of the crew on duty across `stations` hold each skill."""
    matricules = []
    for station in stations:
        try:
            matricules.extend(planning[station][month][day][hour]["planned"])
        except (KeyError, TypeError):
            continue

    known = df_skills.index
    matricules = [m for m in matricules if m in known]
    if not matricules:
        return np.zeros(df_skills.shape[1] // 2, dtype=int)
    return np.count_nonzero(update_skills(df_skills.loc[matricules], date), axis=0)

def select_rarest(counts, rarity, top_k=None):
    """The scarce skills among `counts`, as a sorted array of skill ids.

    Two cuts, and each answers a different objection to the other.

    The **threshold** (`0 < count < rarity`) is a floor on what scarcity means.
    Its lower bound excludes zero deliberately: on a typical 262-strong duty
    roster, 50 of the 134 skills have no holder at all, and counting those as
    rare made the flag fire on 87% of skills. A skill nobody on duty holds
    cannot be preserved or spent by an assignment -- there is no firefighter
    carrying it to choose between -- so it describes a gap in the roster, not a
    trade-off the agent can make. With the zeros in, even `rarity 5` flagged 61%
    of skills; with them out, `rarity 10` flags 31%.

    The **rank** (`top_k`) fixes what the threshold alone cannot: a count is not
    comparable across hours. A roster of 262 in the day and a fraction of that
    at night make "fewer than 10 holders" mean different things, so the number
    of skills flagged swings with the shift rather than with scarcity. A rank is
    invariant to roster size and hands the agent a near-constant number of
    flagged skills per step.

    Ties are kept rather than broken. Taking exactly `top_k` would have to
    separate skills on equal counts by index order, which is arbitrary: two
    skills held by 12 people are equally rare, and letting one in while the
    other stays out makes membership flip between neighbouring hours on nothing
    real. That instability would propagate -- `get_related_rows_in_time`
    aggregates this column over a two-hour window, and `rare_skills_given_up`
    reports it per decision -- so a skill would drift in and out of the
    explanation for no reason a reader could follow. Admitting the whole tie
    band returns slightly more than `top_k` (23 where 20 was asked) and is
    stable under counts that barely move.

    `top_k=None` keeps the threshold alone, which is the pre-rank behaviour.
    """
    eligible = np.flatnonzero((counts > 0) & (counts < rarity))
    if top_k is None or eligible.size <= top_k:
        return eligible

    # The count of the k-th rarest; every skill at or below it stays, which is
    # what admits the whole tie band instead of cutting through it.
    cutoff = np.partition(counts[eligible], top_k - 1)[top_k - 1]
    return eligible[counts[eligible] <= cutoff]


def get_rare_skills_from_planed_ff(date, month, day, hour, planning, df_stations, df_skills,
                                   rarity, top_k=None, cache=None):
    """Which skills are scarce among the firefighters on duty at this hour.

    Counts the holders on the hour's roster and passes them to `select_rarest`,
    which decides what "scarce" means; see it for the threshold-and-rank rule.

    `cache` keys on (calendar day, month, day, hour), which is exact rather than
    an approximation: the crew comes from `planning[station][month][day][hour]`,
    and `update_skills` compares the date against validity windows that start
    and end at midnight, so two events in the same hour of the same day always
    produce the same answer. On a ten-year stream that collapses 637k calls into
    87k -- the difference between roughly six hours and forty minutes.
    """
    if cache is not None:
        key = (date.normalize(), month, day, hour)
        hit = cache.get(key)
        if hit is not None:
            return hit

    array_of_mats = get_all_planed_ff(planning, month, day, hour, df_stations)
    df_skills_filtered = df_skills.loc[array_of_mats]
    updated_skills_filtered = update_skills(df_skills_filtered, date)
    all_current_ff_skills = np.count_nonzero(updated_skills_filtered, axis=0)
    # A 1-tuple, because callers index `[0]`: `np.where` returned one per axis.
    result = (select_rarest(all_current_ff_skills, rarity, top_k),)

    if cache is not None:
        cache[key] = result
    return result

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Environment params")
    parser.add_argument("--dataset", type=str, help="name of dataset")
    parser.add_argument("--rarity", type=int, help="rarity threshold for skills")
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=(
            "keep at most the K rarest skills per hour, ties included, among those "
            "passing --rarity. Omit for the threshold alone."
        ),
    )
    parser.add_argument(
        "--from_dir",
        type=str,
        choices=("sampled", "environment"),
        default="sampled",
        help="folder holding --dataset: Data_sampled (default) or Data_environment",
    )
    parser.add_argument(
        "--merge_into",
        type=str,
        default="df_pc_real_prob.pkl",
        help="event stream in Data_environment to carry the rare skills",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="df_pc_prob_rare_skills_merged.pkl",
        help="output name in Data_environment",
    )
    parser.add_argument(
        "--save_rare_as",
        type=str,
        default="df_pc_rare_skills.pkl",
        help="intermediate (departures only) name in Data_environment",
    )
    args = parser.parse_args()

    source = DATA_SAMPLED if args.from_dir == "sampled" else DATA_ENVIRONMENT
    df_pc = pd.read_pickle(resolve(args.dataset, source))

    df_stations = pd.read_pickle(DATA_ENVIRONMENT / "df_stations.pkl")
    df_skills = pd.read_pickle(DATA_ENVIRONMENT / "df_skills.pkl")
    planning = pd.read_pickle(DATA_ENVIRONMENT / "planning.pkl")

    # Rare skills are a property of the crew on duty when a departure is
    # requested, so RETURN rows are dropped here and given empty arrays by the
    # merge below, which puts them back.
    df_pc = df_pc[df_pc["departure"] != {0: 'RETURN'}]

    rare_skills_cache = {}
    df_pc["rare_skills_required"] = df_pc.progress_apply(lambda row: get_rare_skills_from_planed_ff(row["date"],
                                                                                                    row["Month"],
                                                                                                    row["Day"],
                                                                                                    row["Hour"],
                                                                                                    planning,
                                                                                                    df_stations,
                                                                                                    df_skills,
                                                                                                    args.rarity,
                                                                                                    top_k=args.top_k,
                                                                                                    cache=rare_skills_cache),
                                                         axis=1)

    df_pc.to_pickle(DATA_ENVIRONMENT / args.save_rare_as)

    df_pc_rs = pd.read_pickle(DATA_ENVIRONMENT / args.save_rare_as)
    df_pc_full = pd.read_pickle(DATA_ENVIRONMENT / args.merge_into)

    df_pc_full["rare_skills_required"] = df_pc_full.index.map(lambda _: np.array([], dtype=int))
    df_pc_full.update(df_pc_rs["rare_skills_required"])
    df_pc_full.to_pickle(DATA_ENVIRONMENT / args.save_as)

if __name__ == "__main__":

    main()

