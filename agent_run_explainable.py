"""
agent_run_explainable.py

Run an "explainable" RL agent (e.g., FQF) inside the emergency-response simulation loop.

This script is primarily intended for:
- Evaluating a trained agent on an event stream (`df_pc_real.pkl` or `df_pc_fake.pkl`)
- Optionally training online (when `--train` is provided)
- Tracking and saving simulation metrics and reward evolution

High-level workflow
-------------------
1) Parse CLI arguments (hyper-parameters JSON, dataset, constraints, model paths, reward weights)
2) Load agent hyper-parameters and instantiate the requested agent (FQF currently supported here)
3) Load model weights and switch the policy network to eval mode (this file currently runs eval-only by default)
4) Load reward weights (tariffs) and environment variables/event stream
5) Iterate over events and:
   - handle RETURN events (restore resources)
   - handle departure events by selecting vehicles, assigning firefighters and applying the agent policy
   - compute reward and optionally train via agent.step(...)
6) Periodically log metrics and save model checkpoints

Import safety
-------------
The executable portion is wrapped in `main()` and guarded by `if __name__ == "__main__":`,
so importing this module does not start a run.

Notes
-----
- This code mutates environment dictionaries in-place (vehicles, planning, etc.).
- The original file contained a syntax error in the tuple unpacking of `row` (extra commas).
  This refactor fixes that unpacking so the script can run.
"""

import pandas as pd
import numpy as np
from agent_explainable import *
from collective_functions import *
from explainability import *
from paths import DATA, PLOTS, REWARD_WEIGHTS, SVG_MODEL, resolve
from sim_state import Fleet
from simulator import run_simulation

from datetime import datetime, timedelta
import torch
import json
import argparse
import re
import ast
import pickle
from IPython.display import clear_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent parameters")
    parser.add_argument("--hyper_params", type=str, help="Agent hyper parameters")
    parser.add_argument("--dataset", type=str, default="df_pc_fake.pkl", help="Name of dataset")
    parser.add_argument("--eps_start", type=float, help="epsilon", default=1.0)
    parser.add_argument("--train", action='store_true', help="Train mode")
    parser.add_argument("--load", action='store_true', help="Load weights")
    parser.add_argument("--agent_model", type=str, help="Model name")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--start", type=int, default=1, help="start from num_inter")
    parser.add_argument("--end", type=int, default=159264, help="end after num_inter")
    parser.add_argument("--n_hours", type=int, help="time window to consider")
    parser.add_argument("--top_n", type=int, help="nearest stations to consider")
    parser.add_argument("--reward_weights", type=str, help="JSON file with reward weights")
    parser.add_argument("--save_metrics_as", type=str, default="dic_indic_agent", help="save metrics as")
    parser.add_argument("--constraint_factor_veh", type=int, default=1, help="size of available vehicles in Z1. factor 1 is 100%%, factor 3 is 33%%")
    parser.add_argument("--constraint_factor_ff", type=int, default=1, help="size of available firefighters. factor 1 is 100%%, factor 3 is 33%%")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the environment downsampling draws")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(resolve(args.hyper_params, DATA), "r") as f:
        hyper_params = json.load(f)

    device = torch.device(hyper_params["device"])
    torch.autograd.set_detect_anomaly(True)
    hyper_params["max_train_steps"] = (args.end-args.start) * 5 # (approx. 5 actions by intervention)
    print("max_train_steps", hyper_params["max_train_steps"])
    if args.agent_model == "dqn":
        agent = DQN_Agent(**hyper_params)
    elif args.agent_model == "fqf":
        agent = FQF_Agent(**hyper_params)
    elif args.agent_model == "ppo":
        agent = PPO_Agent(**hyper_params)

    model_path = resolve(args.model_name, SVG_MODEL)

    if args.train:
        if args.load:
            agent.qnetwork_local.load_state_dict(torch.load(model_path, weights_only=True))
            print("Weights loaded")
        agent.qnetwork_local.train()
        print("Train mode", flush=True)

    else:
        agent.qnetwork_local.load_state_dict(torch.load(model_path, weights_only=True))
        agent.qnetwork_local.eval()

        print("Eval mode - weights loaded", flush=True)

    with open(resolve(args.reward_weights, REWARD_WEIGHTS)) as f:
        dic_tarif = json.load(f)
    # print("Reward weights", dic_tarif)

    ### LOAD ENVIRONMENT VARIABLES ###

    env = load_environment(args.constraint_factor_veh, args.constraint_factor_ff,
                           args.dataset, args.start, args.end, args.seed)

    # Reinforcement bookkeeping for VSAV/FPT/EPA. Field defaults match the
    # flat initialisation this replaces ("" / False / 0).
    fleet = Fleet()

    action_size = hyper_params["action_size"] # idx role + rl infos

    SVG_MODEL.mkdir(parents=True, exist_ok=True)

    # Mutable bits the callbacks share across calls.
    # Eval mode starts at 0, but the decay's max(0.05, ...) floor then lifts it
    # to 0.05 on the first tick. Preserved as-is: changing it changes results.
    rl = {"eps": args.eps_start if args.train else 0.0, "d": 1,
          "compute": False, "old_state": None, "reward": 0.0, "loss": 0,
          "score": 0.0, "action_num": 0}
    reward_evo = []
    dic_saved_skills = {k: 0 for k in range(0, 134)}
    upcoming = {"skills": np.array([], dtype=int)}
    eps_update = (args.end - args.start) // 23  # ~23 steps to reach 5% of eps
    print("eps_start", rl["eps"], "eps_update", eps_update, flush=True)

    def on_row(idx, date, pdd, df_pc):
        if pdd:
            upcoming["skills"] = get_related_rows_in_time(
                idx, date, pdd, df_pc, args.top_n, args.n_hours
            )

    def decide(state, all_ff_waiting, ff_array, inter_done):
        """Agent choice, learning from the previous transition first."""
        dic_rare_skills = get_dic_rare_skills(upcoming["skills"], ff_array)

        # `compute` guards the first action, where there is no old_state yet.
        if rl["compute"] and args.train:
            l0 = agent.step(rl["old_state"], rl["action"], rl["reward"], state, inter_done)
            if l0 is not None:
                rl["loss"] = l0
        else:
            rl["loss"] = 0

        action, skill_lvl, potential_actions = agent.act(state, all_ff_waiting, rl["eps"])

        # Explainability: did this choice preserve a rare skill that some other
        # feasible firefighter also carried?
        others = [a for a in potential_actions if a != action]
        if (action < 79) and not all_ff_waiting and upcoming["skills"].size > 0:
            for urs in upcoming["skills"]:
                lol_skills_ff = [dic_rare_skills[pa] for pa in others]
                flat_list = list({e for l1 in lol_skills_ff for e in l1})
                if (urs not in dic_rare_skills[action]) and (urs in flat_list):
                    dic_saved_skills[urs] += 1

        rl["action"] = action
        rl["old_state"] = state
        return action, skill_lvl, potential_actions

    def on_action(ctx):
        """Reward is the indicator delta, read before dic_indic_old is refreshed."""
        rl["reward"] = compute_reward(ctx.dic_indic, ctx.dic_indic_old, ctx.num_d, dic_tarif)
        rl["score"] += rl["reward"]
        rl["action_num"] += 1
        reward_evo.append([rl["action_num"], rl["reward"]])
        rl["compute"] = True

    def log_interval(num_inter, vehicle_out, fleet):
        rwd_mean = np.mean([r[1] for r in reward_evo[-100:]]) if reward_evo else float("nan")
        dic_indic = env.dic_indic
        print(f"{num_inter} v_out: {vehicle_out} | rwd_mean: {rwd_mean:.2f} | v1notfroms1: {dic_indic['v1_not_sent_from_s1']} | v3notfroms3: {dic_indic['v3_not_sent_from_s3']} | v_not_found_ls: {dic_indic['v_not_found_in_last_station']} | deg: {dic_indic['v_degraded']} | rupture_ff: {dic_indic['rupture_ff']}", flush=True)
        print(f"{num_inter} z1_VSAV_sent: {dic_indic['z1_VSAV_sent']} | z1_FPT_sent: {dic_indic['z1_FPT_sent']} | z1_EPA_sent: {dic_indic['z1_EPA_sent']} | VSAV_disp: {fleet.VSAV.disp} | FPT_disp: {fleet.FPT.disp} | EPA_disp: {fleet.EPA.disp} |", flush=True)

    def on_return(num_inter):
        """Runs on every RETURN event — not on the 100-tick logging schedule."""
        if args.eps_start > 0 and eps_update and num_inter % eps_update == 0:
            rl["eps"] = max(0.05, rl["eps"] * 0.99 ** rl["d"])
            rl["d"] += 1

        if num_inter % 10000 == 0 and args.train:
            torch.save(agent.qnetwork_local.state_dict(), model_path)
            print(num_inter, "Agent saved as", args.model_name, flush=True)

    run_simulation(env, fleet, decide, action_size=action_size,
                   on_row=on_row, on_action=on_action,
                   on_interval=log_interval, on_return=on_return)

    dic_indic = env.dic_indic
    print("Simulation done")

    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve(args.save_metrics_as + ".pkl", PLOTS)
    with open(metrics_path, "wb") as f:
        pickle.dump(dic_indic, f)
    print("Metrics saved to", metrics_path)
