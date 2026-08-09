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

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch

import checkpoint as ckpt
from agent_explainable import DQNAgent, FQFAgent, PPOAgent
from collective_functions import DEFAULT_SEED, compute_reward, load_environment
from decision_log import DecisionLog
from explainability import get_dic_rare_skills, get_related_rows_in_time
from paths import DATA, PLOTS, REWARD_WEIGHTS, SVG_MODEL, resolve
from sim_state import Fleet
from simulator import run_simulation

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
    parser.add_argument("--resume", action='store_true', help="Resume from the checkpoint written by --checkpoint_name")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Resumable checkpoint file (default: <model_name>.ckpt)")
    parser.add_argument("--checkpoint_every", type=int, default=10000, help="Write a resumable checkpoint every N interventions")
    parser.add_argument("--no_save_buffer", action='store_true', help="Omit the replay buffer from checkpoints (smaller files, colder resume)")
    parser.add_argument("--detect_anomaly", action='store_true', help="Enable torch autograd anomaly detection (slow; for debugging NaNs)")
    parser.add_argument("--decision_log", type=str, default=None, help="Record per-decision explanations to this file in Plots/ (eval runs)")
    parser.add_argument("--decision_log_rate", type=float, default=0.01, help="Sampling rate for decisions that are neither close nor skill-relevant")
    parser.add_argument("--decision_log_margin", type=float, default=0.5, help="Q-value gap below which a decision counts as close")
    parser.add_argument("--decision_log_quantiles", action='store_true', help="Also store the per-action return distribution (larger file)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(resolve(args.hyper_params, DATA), "r") as f:
        hyper_params = json.load(f)

    device = torch.device(hyper_params["device"])
    # Off by default: anomaly mode records a stack trace for every autograd op
    # to attribute a future NaN, which costs real time on a run measured in
    # hours. Turn it on with --detect_anomaly when a NaN actually appears.
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    hyper_params["max_train_steps"] = (args.end-args.start) * 5 # (approx. 5 actions by intervention)
    print("max_train_steps", hyper_params["max_train_steps"])
    if args.agent_model == "dqn":
        agent = DQNAgent(**hyper_params)
    elif args.agent_model == "fqf":
        agent = FQFAgent(**hyper_params)
    elif args.agent_model == "ppo":
        agent = PPOAgent(**hyper_params)
    else:
        raise ValueError(f"Unknown --agent_model: {args.agent_model!r}")

    model_path = resolve(args.model_name, SVG_MODEL)
    ckpt_path = resolve(args.checkpoint_name or (args.model_name + ".ckpt"), SVG_MODEL)

    # Loaded before the environment so a settings mismatch is refused up front,
    # rather than after several minutes of environment building.
    resume_payload = None
    if args.resume:
        if not args.train:
            raise SystemExit("--resume only applies to training runs (use --train).")
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt_path}")
        resume_payload = ckpt.load(ckpt_path, map_location=hyper_params.get("device", "cpu"))
        ckpt.check_compatible(resume_payload, args)
        print(f"Resuming from {ckpt_path} at intervention {resume_payload['num_inter']}", flush=True)

    if args.train:
        if args.load and not args.resume:
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

    # Applied after `load_environment` so the checkpoint's containers overwrite
    # the freshly built ones, and after `rl` exists so epsilon and its decay
    # counter come back together — restoring eps without `d` would put the decay
    # back on the wrong rung.
    resume_from = None
    loop_seed = None
    if resume_payload is not None:
        restored = ckpt.restore(resume_payload, agent=agent, env=env, rl=rl)
        fleet = restored["fleet"]
        resume_from = restored["row_index"]
        loop_seed = restored["loop"]
        reward_evo = restored["reward_evo"]
        dic_saved_skills = restored["dic_saved_skills"]
        print("Resumed: eps", rl["eps"], "d", rl["d"], "t_step", agent.t_step, flush=True)

    print("eps_start", rl["eps"], "eps_update", eps_update, flush=True)

    # Off unless asked for. Intended for evaluation runs: it explains a settled
    # policy, and during training the thing being explained changes underneath
    # you as the weights move.
    decision_log = None
    if args.decision_log:
        decision_log = DecisionLog(
            resolve(args.decision_log, PLOTS),
            rate=args.decision_log_rate,
            margin=args.decision_log_margin,
            keep_quantiles=args.decision_log_quantiles,
        )
        print("Decision log ->", decision_log.path, flush=True)

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

        # Only ask for the Q-values when something will read them: filling the
        # dict copies a value per feasible action on every decision.
        explain = {} if decision_log is not None else None
        action, skill_lvl, potential_actions = agent.act(
            state, all_ff_waiting, rl["eps"], explain=explain
        )

        # Explainability: did this choice preserve a rare skill that some other
        # feasible firefighter also carried?
        others = [a for a in potential_actions if a != action]
        if (action < 79) and not all_ff_waiting and upcoming["skills"].size > 0:
            for urs in upcoming["skills"]:
                lol_skills_ff = [dic_rare_skills[pa] for pa in others]
                flat_list = list({e for l1 in lol_skills_ff for e in l1})
                if (urs not in dic_rare_skills[action]) and (urs in flat_list):
                    dic_saved_skills[urs] += 1

        if decision_log is not None:
            st = live["st"]
            decision_log.consider(
                num_inter=st.num_inter, idx_role=st.idx_role, num_d=st.num_d,
                current_station=st.current_station, v_mat=st.v_mat, date=st.date,
                action=action, potential_actions=potential_actions,
                ff_existing=st.ff_existing, dic_rare_skills=dic_rare_skills,
                upcoming_skills=upcoming["skills"], skill_lvl=skill_lvl,
                explain=explain,
            )

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

    # `run_simulation` builds `_LoopState` internally; this hands it back so a
    # checkpoint can capture the bookkeeping that lives only inside the loop.
    live = {"st": None}

    def on_state_ready(st):
        live["st"] = st

    metrics_path = resolve(args.save_metrics_as + ".pkl", PLOTS)
    curves_path = resolve(args.save_metrics_as + "_curves.pkl", PLOTS)

    def save_metrics(num_inter):
        """Write the run's metrics, atomically, so an interrupted run keeps them.

        `dic_indic` keeps its own file and its exact previous shape — a bare
        dict of the 21 counters — because that is what the plotting code and the
        heuristic baseline already read.

        `reward_evo` and `dic_saved_skills` were tracked for the whole run and
        then dropped on the floor: the reward curve and the rare-skill
        explainability counts only ever existed in memory, and a run that did
        not reach its final line left nothing behind. They go in a second file
        alongside, with the intervention they were written at.
        """
        PLOTS.mkdir(parents=True, exist_ok=True)
        for path, payload in (
            (metrics_path, env.dic_indic),
            (curves_path, {"num_inter": num_inter,
                           "reward_evo": reward_evo,
                           "dic_saved_skills": dic_saved_skills}),
        ):
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        if decision_log is not None:
            decision_log.flush()

    def on_return(num_inter):
        """Runs on every RETURN event — not on the 100-tick logging schedule."""
        if args.eps_start > 0 and eps_update and num_inter % eps_update == 0:
            rl["eps"] = max(0.05, rl["eps"] * 0.99 ** rl["d"])
            rl["d"] += 1

        if args.checkpoint_every and num_inter % args.checkpoint_every == 0 and args.train:
            st = live["st"]
            # The weights file stays a plain state_dict so eval runs (--load
            # without --resume) keep working unchanged.
            torch.save(agent.qnetwork_local.state_dict(), model_path)
            ckpt.save(
                ckpt_path,
                agent=agent, env=env, fleet=fleet, loop=st, rl=rl,
                num_inter=num_inter,
                row_index=st.idx if st is not None else None,
                old_date=env.old_date,
                date_reference=env.date_reference,
                reward_evo=reward_evo,
                dic_saved_skills=dic_saved_skills,
                args=args,
                include_buffer=not args.no_save_buffer,
            )
            save_metrics(num_inter)
            print(num_inter, "Agent saved as", args.model_name,
                  "| checkpoint", ckpt_path.name,
                  "| metrics", metrics_path.name, flush=True)

    run_simulation(env, fleet, decide, action_size=action_size,
                   on_row=on_row, on_action=on_action,
                   on_interval=log_interval, on_return=on_return,
                   resume_from=resume_from, loop_state=loop_seed,
                   on_state_ready=on_state_ready)

    dic_indic = env.dic_indic
    print("Simulation done")

    save_metrics(live["st"].num_inter if live["st"] is not None else None)
    print("Metrics saved to", metrics_path, "and", curves_path)

    if decision_log is not None:
        print("Decision log saved to", decision_log.flush())
        print("  ", decision_log.summary())
