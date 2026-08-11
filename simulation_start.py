import argparse
import pickle
import random

import numpy as np

# from tqdm.notebook import trange, tqdm
from collective_functions import (
    DEFAULT_SEED,
    apply_logic,
    get_potential_actions,
    load_environment,
)
from decision_log import DecisionLog
from explainability import get_dic_rare_skills, get_related_rows_in_time, rare_skills_for_step
from paths import PLOTS, resolve
from sim_state import Fleet
from simulator import run_simulation

# Shared empty result, for decisions where scoped rarity is not computed.
_EMPTY = np.array([], dtype=int)

#########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulation_params")
    parser.add_argument("--dataset", type=str, help="Name of dataset")
    parser.add_argument("--is_best", action='store_true', help="Best logic or random logic")
    parser.add_argument("--start", type=int, help="start from num_inter")
    parser.add_argument("--end", type=int, help="end after num_inter")
    parser.add_argument("--save_metrics_as", type=str, help="save dic_indic as")
    parser.add_argument("--constraint_factor_veh", type=int, default=1, help="size of available vehicles in Z1. factor 1 is 100%%, factor 3 is 33%%")
    parser.add_argument("--constraint_factor_ff", type=int, default=1, help="size of available firefighters. factor 1 is 100%%, factor 3 is 33%%")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the environment downsampling draws")
    # Mirrors the agent runner's flags, so a baseline log is comparable to an
    # agent log field for field. `--n_hours`/`--top_n` only matter when logging.
    parser.add_argument("--decision_log", type=str, default=None, help="Record per-decision explanations to this file in Plots/")
    parser.add_argument("--decision_log_rate", type=float, default=0.01, help="Sampling rate for decisions that are neither close nor skill-relevant")
    parser.add_argument("--decision_log_max", type=int, default=200000, help="Stop collecting past this many records")
    parser.add_argument("--n_hours", type=int, default=2, help="Look-ahead window for upcoming rare skills")
    parser.add_argument("--top_n", type=int, default=5, help="Nearby stations considered for scoped rarity and the look-ahead")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.is_best:
        print("logic is best")
    else:
        print("logic is random")

    ### LOAD ENVIRONMENT VARIABLES ###

    env = load_environment(args.constraint_factor_veh, args.constraint_factor_ff,
                           args.dataset, args.start, args.end, args.seed)

    # Reinforcement bookkeeping for VSAV/FPT/EPA. Field defaults match the
    # flat initialisation this replaces ("" / False / 0).
    fleet = Fleet()

    action_size = 80

    # Off unless asked for. Present so a baseline can be compared to the agent
    # on the same per-decision quantities, not only on aggregate metrics.
    decision_log = None
    if args.decision_log:
        decision_log = DecisionLog(
            resolve(args.decision_log, PLOTS),
            rate=args.decision_log_rate,
            # A heuristic has no Q-values, so no decision can be "close". Set
            # the margin to 0 to say so explicitly rather than let the default
            # suggest a criterion that never fires here.
            margin=0.0,
            max_records=args.decision_log_max,
        )
        print("Decision log ->", decision_log.path, flush=True)

    live = {"st": None}
    upcoming = {"skills": np.array([], dtype=int)}
    scoped_cache = {}

    def on_state_ready(st):
        live["st"] = st

    def on_row(idx, date, pdd, df_pc):
        if pdd:
            upcoming["skills"] = get_related_rows_in_time(
                idx, date, pdd, df_pc, args.top_n, args.n_hours
            )

    def decide(state, all_ff_waiting, ff_array, inter_done):
        """Heuristic choice: best available skill match, or a random feasible one."""
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        action, skill_lvl = apply_logic(potential_actions, potential_skills, args.is_best)

        if decision_log is not None and live["st"] is not None:
            st = live["st"]
            local, irreversible = rare_skills_for_step(
                st, n_following=args.top_n, cache=scoped_cache
            )
            decision_log.consider(
                num_inter=st.num_inter, idx_role=st.idx_role, num_d=st.num_d,
                current_station=st.current_station, v_mat=st.v_mat, date=st.date,
                action=action, potential_actions=potential_actions,
                ff_existing=st.ff_existing,
                dic_rare_skills=get_dic_rare_skills(upcoming["skills"], ff_array),
                upcoming_skills=upcoming["skills"], skill_lvl=skill_lvl,
                # No Q-values behind a heuristic choice; `explain` stays None so
                # the log records the absence rather than inventing a margin.
                explain=None,
                local_rare_skills=local,
                irreversible_rare_skills=irreversible,
            )

        return action, skill_lvl, potential_actions

    def log_interval(num_inter, vehicle_out, fleet):
        print(f"{num_inter} v_out: {vehicle_out} | v1notfroms1: {env.dic_indic['v1_not_sent_from_s1']} | v3notfroms3: {env.dic_indic['v3_not_sent_from_s3']} | v_not_found_ls: {env.dic_indic['v_not_found_in_last_station']} | deg: {env.dic_indic['v_degraded']} | rupture_ff: {env.dic_indic['rupture_ff']}", flush=True)
        print(f"{num_inter} z1_VSAV_sent: {env.dic_indic['z1_VSAV_sent']} | z1_FPT_sent: {env.dic_indic['z1_FPT_sent']} | z1_EPA_sent: {env.dic_indic['z1_EPA_sent']} | VSAV_disp: {fleet.VSAV.disp} | FPT_disp: {fleet.FPT.disp} | EPA_disp: {fleet.EPA.disp} |", flush=True)

    run_simulation(
        env, fleet, decide, action_size=action_size, on_interval=log_interval,
        on_row=on_row if decision_log is not None else None,
        on_state_ready=on_state_ready,
    )

    dic_indic = env.dic_indic


    print("Simulation done")

    # np.save(args.save_metrics_as + "_vehicle_" + args.dataset[6:10] +".npy", vehicle_evo)

    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve(args.save_metrics_as + ".pkl", PLOTS)
    with open(metrics_path, "wb") as f:
        pickle.dump(dic_indic, f)
    print("Metrics saved to", metrics_path)

    if decision_log is not None:
        print("Decision log saved to", decision_log.flush(), flush=True)
        print("  ", decision_log.summary(), flush=True)

