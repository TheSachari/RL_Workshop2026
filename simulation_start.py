import pandas as pd
import numpy as np
import random
import pickle
import argparse
import re
import ast
import json
from IPython.display import clear_output
# from tqdm.notebook import trange, tqdm
from tqdm.auto import tqdm

from collective_functions import *
from paths import PLOTS, resolve
from sim_state import Fleet
from simulator import run_simulation

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

    def decide(state, all_ff_waiting, ff_array, inter_done):
        """Heuristic choice: best available skill match, or a random feasible one."""
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        action, skill_lvl = apply_logic(potential_actions, potential_skills, args.is_best)
        return action, skill_lvl, potential_actions

    def log_interval(num_inter, vehicle_out, fleet):
        print(f"{num_inter} v_out: {vehicle_out} | v1notfroms1: {env.dic_indic['v1_not_sent_from_s1']} | v3notfroms3: {env.dic_indic['v3_not_sent_from_s3']} | v_not_found_ls: {env.dic_indic['v_not_found_in_last_station']} | deg: {env.dic_indic['v_degraded']} | rupture_ff: {env.dic_indic['rupture_ff']}", flush=True)
        print(f"{num_inter} z1_VSAV_sent: {env.dic_indic['z1_VSAV_sent']} | z1_FPT_sent: {env.dic_indic['z1_FPT_sent']} | z1_EPA_sent: {env.dic_indic['z1_EPA_sent']} | VSAV_disp: {fleet.VSAV.disp} | FPT_disp: {fleet.FPT.disp} | EPA_disp: {fleet.EPA.disp} |", flush=True)

    run_simulation(env, fleet, decide, action_size=action_size, on_interval=log_interval)

    dic_indic = env.dic_indic


    print("Simulation done")

    # np.save(args.save_metrics_as + "_vehicle_" + args.dataset[6:10] +".npy", vehicle_evo)

    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve(args.save_metrics_as + ".pkl", PLOTS)
    with open(metrics_path, "wb") as f:
        pickle.dump(dic_indic, f)
    print("Metrics saved to", metrics_path)

