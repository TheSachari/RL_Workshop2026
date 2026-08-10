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

def get_rare_skills_from_planed_ff(date, month, day, hour, planning, df_stations, df_skills,
                                   rarity, cache=None):
    """Which skills are scarce among the firefighters on duty at this hour.

    Scarce means *held by few of them*, not *held by none*. The test excludes
    zero deliberately: on a typical 262-strong duty roster, 50 of the 134 skills
    have no holder at all, and counting those as rare made the flag fire on 87%
    of skills. A skill nobody on duty holds cannot be preserved or spent by an
    assignment -- there is no firefighter carrying it to choose between -- so it
    describes a gap in the roster, not a trade-off the agent can make.

    Excluding them is what gives `rarity` its meaning back: with the zeros in,
    even `rarity 5` flagged 61% of skills; with them out, `rarity 10` flags 31%.

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
    result = np.where((all_current_ff_skills > 0) & (all_current_ff_skills < rarity))

    if cache is not None:
        cache[key] = result
    return result

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Environment params")
    parser.add_argument("--dataset", type=str, help="name of dataset")
    parser.add_argument("--rarity", type=int, help="rarity threshold for skills")
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
                                                                                                    rare_skills_cache),
                                                         axis=1)

    df_pc.to_pickle(DATA_ENVIRONMENT / args.save_rare_as)

    df_pc_rs = pd.read_pickle(DATA_ENVIRONMENT / args.save_rare_as)
    df_pc_full = pd.read_pickle(DATA_ENVIRONMENT / args.merge_into)

    df_pc_full["rare_skills_required"] = df_pc_full.index.map(lambda _: np.array([], dtype=int))
    df_pc_full.update(df_pc_rs["rare_skills_required"])
    df_pc_full.to_pickle(DATA_ENVIRONMENT / args.save_as)

if __name__ == "__main__":

    main()

