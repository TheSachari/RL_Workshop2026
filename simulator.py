"""The simulation loop, shared by the heuristic baseline and the RL agent.

`simulation_start.py` and `agent_run_explainable.py` used to hold two copies of
this loop — 275 identical lines that differed only in how the action is chosen.
Fixing a bug in one meant remembering to fix the other, and any drift silently
invalidated the baseline/agent comparison, since the two arms would no longer be
running the same environment.

They now call `run_simulation` with different callbacks:

    decide(state, all_ff_waiting, ff_array, inter_done)
        -> (action, skill_lvl, potential_actions)

        Chooses one action. The heuristic wraps `apply_logic`; the agent wraps
        `agent.act`. Returning the feasible actions lets the agent runner do its
        explainability bookkeeping without recomputing them. `ff_array` and
        `inter_done` are passed because the agent needs them for its rare-skill
        accounting and to mark episode boundaries when it learns.

    on_action(ctx) -> None            (optional)

        Called after the environment has been mutated by `step`, with the
        surrounding state. The agent runner uses it to compute the reward and
        train; the heuristic passes nothing.

Everything else — resource lookup, reinforcement logistics, role assignment — is
run once, from one place, for both.

The body below is the original loop, lifted verbatim apart from the decision
call. It is long and mutates its locals throughout; that is deliberate at this
stage. Grouping the rest of the state is a separate step, and doing it in the
same commit as the merge would have made the golden diff impossible to read.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from collective_functions import (
    adding_lent_ff,
    are_all_ff_waiting,
    gen_ff_array,
    gen_state,
    get_mandatory_max,
    get_neighborhood_availability,
    get_potential_actions,
    get_potential_veh,
    reinforcement_arriving,
    reinforcement_returning,
    reinforcement_sending,
    returning,
    step,
    update_dep,
    update_dict,
    update_duration,
    update_skills,
    v_to_return_managing,
    veh_management,
)


@dataclass
class ActionContext:
    """State handed to `on_action` after each decision has been applied."""

    state: Any
    action: int
    skill_lvl: int
    num_d: int
    inter_done: bool
    dic_indic: dict
    dic_indic_old: dict
    potential_actions: list


def run_simulation(
    env,
    fleet,
    decide: Callable,
    *,
    action_size: int,
    on_row=None,
    on_action: Optional[Callable[[ActionContext], None]] = None,
    on_interval=None,
    on_return=None,
):
    """Run the event stream, delegating each action choice to `decide`.

    `env` is the `Environment` from `collective_functions.load_environment`;
    `fleet` the `Fleet` reinforcement state. Both are mutated in place, as the
    original loop did — the caller reads its metrics back off `env.dic_indic`.

    `on_row(row_fields)` runs once per event before it is handled (the agent
    runner uses it to precompute upcoming rare skills). `on_interval(num_inter)`
    runs on the loop's periodic logging tick.
    """
    # Unpack the environment into the local names the loop body uses.
    dic_vehicles = env.dic_vehicles
    dic_functions = env.dic_functions
    df_skills = env.df_skills
    dic_roles_skills = env.dic_roles_skills
    dic_roles = env.dic_roles
    planning = env.planning
    dic_inter = env.dic_inter
    dic_ff = env.dic_ff
    dic_indic = env.dic_indic
    dic_indic_old = env.dic_indic_old
    Z_1 = env.Z_1
    Z_4 = env.Z_4
    dic_lent = env.dic_lent
    dic_station_distance = env.dic_station_distance
    df_pc = env.df_pc
    old_date = env.old_date
    date_reference = env.date_reference
    skills_updated = env.skills_updated

    vehicle_out, num_d, action_num = 0, 42, 0
    all_ff_waiting, v_waiting, following_depart = (False,) * 3

    vehicle_evo, current_ff_inter = [], []
    dic_log, dic_back, dic_start_time, dic_veh_typ = {}, {}, {}, {}

    max_duration = df_pc["Duration"].max()

    for row in df_pc.itertuples(index=True, name=None):

        # The agent's event stream carries a 20th column (rare skills required);
        # the baseline's does not. Take the common prefix either way.
        idx, num_inter, date, pdd, required_departure, zone, duration, month, day, hour, minute, \
        coord_x, coord_y, month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos = row[:19]

        if on_row is not None:
            on_row(idx, date, pdd, df_pc)

        dic_ff = update_duration(date, old_date, current_ff_inter, dic_ff)

        if date >= date_reference:
            date_reference = date
            skills_updated = update_skills(df_skills, date_reference)
                
        if (fleet.VSAV.sent) and (num_inter == fleet.VSAV.arrival_num) :  # ARRIVEE DES RENFORTS VSAV
            dic_vehicles, fleet.VSAV.sent, fleet.VSAV.lent, fleet.VSAV.returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            fleet.VSAV.to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, fleet.VSAV.from_station, fleet.VSAV.to_station, fleet.VSAV.sent, \
                                                    fleet.VSAV.returning, fleet.VSAV.lent, fleet.VSAV.to_return, month, day, hour, dic_start_time, "VSAV")

        if (fleet.FPT.sent) and (num_inter == fleet.FPT.arrival_num) :  # ARRIVEE DES RENFORTS FPT
            dic_vehicles, fleet.FPT.sent, fleet.FPT.lent, fleet.FPT.returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            fleet.FPT.to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, fleet.FPT.from_station, fleet.FPT.to_station, fleet.FPT.sent, \
                                                    fleet.FPT.returning, fleet.FPT.lent, fleet.FPT.to_return, month, day, hour, dic_start_time, "FPT")

        if (fleet.EPA.sent) and (num_inter == fleet.EPA.arrival_num) :  # ARRIVEE DES RENFORTS EPA
            dic_vehicles, fleet.EPA.sent, fleet.EPA.lent, fleet.EPA.returning, \
            dic_ff, planning, dic_back, dic_lent, dic_log, \
            fleet.EPA.to_return = reinforcement_arriving(num_inter, dic_vehicles, dic_back, dic_lent, \
                                                    dic_ff, dic_log, planning, fleet.EPA.from_station, fleet.EPA.to_station, fleet.EPA.sent, \
                                                    fleet.EPA.returning, fleet.EPA.lent, fleet.EPA.to_return, month, day, hour, dic_start_time, "EPA")
    
        if (required_departure == {0:"RETURN"}):  # RETOUR D'INTERVENTION    
            vehicle_out, dic_vehicles, dic_ff, current_ff_inter, planning, \
            dic_inter = returning(df_pc, dic_inter, num_inter, vehicle_out, dic_vehicles, \
                                  dic_ff, current_ff_inter, planning, month, day, hour)
                     
        else: # INTERVENTION      
            veh_depart = [v[0] for k, v in sorted(required_departure.items())]

            dic_indic['v_required'] += len(required_departure)

            if len(veh_depart) > 5:
                new_veh_depart = veh_depart[5:].copy()
                veh_depart = veh_depart[:5]  
                required_departure = {i + 1: [v] for i, v in enumerate(veh_depart)}
                following_depart = True
                # print("following_depart", veh_depart)

            new_required_departure = {}
            stations = iter(pdd)            
            inter_done = False
            fleet.VSAV.to_return, fleet.FPT.to_return, fleet.EPA.to_return = (False,) * 3
            station_lvl = 0
            idx_role = 0
            v_mat_to_return = 0
            
            vehicle_evo.append([num_inter, vehicle_out])
    
            while not inter_done: # Tant que l'intervention n'est pas finie   
                
                current_station = next(stations, False) # On va dans la plus proche caserne
                if (num_d < 99) and (current_station not in dic_inter[num_inter]):
                    # Il s'agit d'une intervention et non d'un renfort
                    dic_inter[num_inter][current_station] = {}
                    
                station_lvl += 1
    
                if not current_station: # S'il n'y a plus de plus proche caserne, l'intervention est terminée
                    inter_done = True
                    departure_done = True         
                    dic_indic['v_not_found_in_last_station'] += len(required_departure)
                    # print("v_not_found_in_last_station")
    
                else: # Sinon on cherche les véhicules requis
                    departure_done = False
                    required_vehicles = iter(sorted(required_departure.items()))

                # print(num_inter, veh_depart)
        
                while not departure_done: # Tant que tous les véhicules n'ont pas été envoyés
                    
                    num_d, list_v = next(required_vehicles, (0, [])) # On cherche les véhicules requis dans le train initial
                    # print("start", "num_d", num_d, "list_v", list_v)
                    if station_lvl == 2 and num_d == 1:
                        dic_indic['v1_not_sent_from_s1'] += 1
                        # print("v1_not_sent_from_1st_station", station_lvl)
                    if station_lvl > 3 and num_d <= 3 and current_station in Z_4:
                        dic_indic['v3_not_sent_from_s3'] += 1
                    
                    if list_v:
                        mandatory, team_max = get_mandatory_max(list_v[0])
                    
                    if not list_v: # S'il n'y a plus de véhicules requis dans le train initial    
                        departure_done = True # On a envoyés tous les véhicules possibles depuis cette caserne 
                        # print("no more v to send from this station")
                        if new_required_departure: # S'il reste des véhicules à faire partir dans le nouveau train  
                            required_departure = new_required_departure
                            required_departure = update_dep(required_departure) # to test
                            veh_depart = [v[0] for k, v in sorted(required_departure.items())]
                            new_required_departure = {}
                            idx_role = 0
                            # print("new_required_departure", veh_depart)

                        else: # S'il ne reste pas de véhicules à faire partir
                            inter_done = True # L'intervention est terminée
                            if following_depart:
                                inter_done = False
                                stations = iter(pdd)
                                station_lvl = 0
                                required_departure = {i + 1: [v] for i, v in enumerate(new_veh_depart)}
                                required_departure = update_dep(required_departure)
                                veh_depart = new_veh_depart

                                if len(veh_depart) > 5:
                                    new_veh_depart = veh_depart[5:].copy()
                                    veh_depart = veh_depart[:5]  
                                    required_departure = {i + 1: [v] for i, v in enumerate(veh_depart)}
                                    following_depart = True
                                    # print("following_depart", veh_depart)
                                
                                new_required_departure = {}
                                idx_role, num_d = 0, 1
                                following_depart = False
                                      
                    else: # S'il y a des véhicules requis dans le train initial   
    
                        if (num_d == 99):
                            current_station = next(stations_VSAV, False)
                        if (num_d == 100):
                            current_station = next(stations_FPT, False)
                        if (num_d == 101):
                            current_station = next(stations_EPA, False)
                        # print(num_inter, "current_station", current_station, "num_d", num_d)
                        # print("to return", fleet.VSAV.to_return, fleet.FPT.to_return, fleet.EPA.to_return)
                            
                        vehicles_to_find = iter(list_v)   
                        vehicle_found = False
                        vehicle_lvl = 1
    
                        if not current_station:
                            vehicle_found = True
                            # v_waiting = False
                            # print('no station')
                            all_ff_waiting = False
                        
                        while not vehicle_found: # Tant qu'on a pas trouvé le véhicule requis
    
                            vehicle_to_find = next(vehicles_to_find, False) # On cherche la prochaine fonction à faire partir  
                            
                            if not vehicle_to_find: # S'il n'y a plus de fonction à faire partir
                                
                                idx_role += team_max 
                                vehicle_found = True
                                new_required_departure[num_d] = list_v # Le véhicule requis est ajouté au nouveau train 
                                dic_indic['function_not_found'] += 1
                                # print("function not found")

                            else: # S'il y a une fonction à faire partir depuis cette caserne
                                # On cherche les véhicules disponibles dans la caserne actuelle correspondant à la fonction requise  

                                # print(num_inter, "v_mat before li_mat_veh", v_mat_to_return)
                                
                                v_mats = dic_vehicles[current_station]["available"].copy()                             
                                li_mat_veh = [v_m for v_m in v_mats if vehicle_to_find in dic_functions[v_m]]
                                # li_mat_veh = [v_mat for v_mat in v_mats if any(func.startswith(vehicle_to_find) for func in dic_functions[v_mat])]
    
                                if li_mat_veh: # Si des véhicules ont la fonction requise   
    
                                    if fleet.VSAV.to_return and (current_station == fleet.VSAV.to_station) and (num_d == 99):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "VSAV", v_mat_to_return)
                                        # print("VSAV_to_return", v_mat, v_waiting)
                                        
                                    elif fleet.FPT.to_return and (current_station == fleet.FPT.to_station) and (num_d == 100):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "FPT", v_mat_to_return)
                                        # print("FPT_to_return", v_mat, v_waiting)
                                        
                                    elif fleet.EPA.to_return and (current_station == fleet.EPA.to_station) and (num_d == 101):
                                        v_mat, v_waiting = v_to_return_managing(dic_log, li_mat_veh, v_waiting, vehicle_to_find, \
                                                                                   current_station, dic_vehicles, "EPA", v_mat_to_return) 
                                        # print("EPA_to_return", v_mat, v_waiting)
                                        
                                    else: 
                                        v_mat = li_mat_veh[0] 
                                        
                                    if num_d < 99:
                                        veh_depart[num_d-1] = vehicle_to_find

                                    # print(num_inter, "veh_depart", veh_depart, "num_d", num_d, vehicle_to_find, v_mat, "found in", current_station)
    
                                    mandatory, team_max = get_mandatory_max(vehicle_to_find)
                                    
                                    # On met le véhicule en standby
                                    dic_vehicles[current_station]["available"].remove(v_mat)
                                    dic_vehicles[current_station]["standby"].append(v_mat) 
                                    vehicle_found = True
                                    # print(num_inter, "vehicle_found", dic_functions[v_mat],  v_mat, "in",  current_station, "num_d", num_d)
    
                                    # On cherche les rôles à pourvoir
                                    roles = dic_roles[vehicle_to_find]
                                    # print(roles)
                                    role_number = iter(range(1, len(roles) + 1))
                                    all_roles_found = False
                                    degraded = False
    
                                    while not all_roles_found: # Tant que tous les rôles ne sont pas pourvus
    
                                        num_role = next(role_number, (0))  
    
                                        if num_role > team_max: # Pour limiter les rôles
                                            num_role = 0
                                                           
                                        if not num_role: # S'il n'y a plus de rôles à pourvoir   
                                            # print("no more role to fill", v_waiting)
                                            all_roles_found = True
                                            ff_to_send = planning[current_station][month][day][hour]['standby'].copy()
                                            planning[current_station][month][day][hour]['standby'] = []    
                                            dic_vehicles[current_station]["standby"].remove(v_mat)

                                            if fleet.VSAV.needed and (num_d == 99): # DEPART VSAV EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                fleet.VSAV.from_station, dic_vehicles, fleet.VSAV.arrival_num, dic_lent, \
                                                dic_log, new_required_departure, fleet.VSAV.needed, \
                                                fleet.VSAV.sent  = reinforcement_sending(num_inter, current_station, fleet.VSAV.from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   fleet.VSAV.to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, fleet.VSAV.needed, \
                                                                                   fleet.VSAV.sent, required_departure, \
                                                                                   new_required_departure, num_d, "VSAV")
                                                dic_indic['z1_VSAV_sent'] += 1

                                            elif fleet.FPT.needed and (num_d == 100): # DEPART FPT EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                fleet.FPT.from_station, dic_vehicles, fleet.FPT.arrival_num, dic_lent, \
                                                dic_log, new_required_departure, fleet.FPT.needed, \
                                                fleet.FPT.sent  = reinforcement_sending(num_inter, current_station, fleet.FPT.from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   fleet.FPT.to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, fleet.FPT.needed, \
                                                                                   fleet.FPT.sent, required_departure, \
                                                                                   new_required_departure, num_d, "FPT")
                                                dic_indic['z1_FPT_sent'] += 1

                                            elif fleet.EPA.needed and (num_d == 101): # DEPART EPA EN RENFORT
                                                dic_start_time[v_mat] = (month, day, hour)
                                                fleet.EPA.from_station, dic_vehicles, fleet.EPA.arrival_num, dic_lent, \
                                                dic_log, new_required_departure, fleet.EPA.needed, \
                                                fleet.EPA.sent  = reinforcement_sending(num_inter, current_station, fleet.EPA.from_station, v_mat, \
                                                                                   dic_vehicles, dic_station_distance, \
                                                                                   fleet.EPA.to_station, date, df_pc, idx, \
                                                                                   dic_lent, ff_to_send, dic_log, fleet.EPA.needed, \
                                                                                   fleet.EPA.sent, required_departure, \
                                                                                   new_required_departure, num_d, "EPA")
                                                dic_indic['z1_EPA_sent'] += 1
                                                # print(num_inter, "EPA", fleet.EPA.sent)

                                            elif v_waiting: # RENFORT A RETOURNER

                                                # print("v_waiting, all_ff_waiting", all_ff_waiting)

                                                if all_ff_waiting: # RETOUR DES RENFORTS

                                                    if fleet.VSAV.to_return and (num_d == 99):  # RETOUR DU VSAV
                                                        fleet.VSAV.from_station, fleet.VSAV.to_station, fleet.VSAV.arrival_num, dic_back, dic_log, \
                                                        fleet.VSAV.needed, fleet.VSAV.sent, all_ff_waiting, v_waiting, fleet.VSAV.to_return, \
                                                        fleet.VSAV.returning = reinforcement_returning(num_inter, fleet.VSAV.to_station, fleet.VSAV.from_station, \
                                                                                                 dic_log, \
                                                                                                 v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, fleet.VSAV.needed, \
                                                                                                 fleet.VSAV.sent, all_ff_waiting, v_waiting, \
                                                                                                 fleet.VSAV.returning, "VSAV")

                                                    elif fleet.FPT.to_return and (num_d == 100): # RETOUR DU FPT
                                                        fleet.FPT.from_station, fleet.FPT.to_station, fleet.FPT.arrival_num, dic_back, dic_log, \
                                                        fleet.FPT.needed, fleet.FPT.sent, all_ff_waiting, v_waiting, fleet.FPT.to_return, \
                                                        fleet.FPT.returning = reinforcement_returning(num_inter, fleet.FPT.to_station, fleet.FPT.from_station, \
                                                                                                dic_log, \
                                                                                                v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, fleet.FPT.needed, \
                                                                                                 fleet.FPT.sent, all_ff_waiting, v_waiting, \
                                                                                                 fleet.FPT.returning, "FPT")

                                                    elif fleet.EPA.to_return and (num_d == 101):  # RETOUR DE L'EPA
                                                        fleet.EPA.from_station, fleet.EPA.to_station, fleet.EPA.arrival_num, dic_back, dic_log, \
                                                        fleet.EPA.needed, fleet.EPA.sent, all_ff_waiting, v_waiting, fleet.EPA.to_return, \
                                                        fleet.EPA.returning = reinforcement_returning(num_inter, fleet.EPA.to_station, fleet.EPA.from_station, \
                                                                                                dic_log, \
                                                                                                v_mat, dic_vehicles, dic_station_distance, date,\
                                                                                                 df_pc, idx, dic_back, ff_to_send, fleet.EPA.needed, \
                                                                                                 fleet.EPA.sent, all_ff_waiting, v_waiting, \
                                                                                                 fleet.EPA.returning, "EPA")
                                                  
                                                else: # retour impossible, pompiers indisponibles

                                                    # print("vehicle", vehicle_to_find, v_mat, 'available again in', current_station)
                                                    v_waiting = False
                                                    dic_vehicles[current_station]["available"].append(v_mat)
                                                    planning[current_station][month][day][hour]['available'] += ff_to_send
    
                                            else: # départ de véhicule en inter

                                                dic_inter[num_inter][current_station][v_mat] = ff_to_send
                                                dic_vehicles[current_station]["inter"].append(v_mat)
                                                
                                                current_ff_inter += ff_to_send
                                                for f in ff_to_send:
                                                    dic_ff[f] = duration
                                                
                                                vehicle_out += 1

                                                # print(num_inter, "vehicle out", current_station, v_mat, ff_to_send, vehicle_out, "|", fleet.VSAV.lent,"/", fleet.VSAV.disp, "|", fleet.FPT.lent,"/",fleet.FPT.disp, "|", fleet.EPA.lent,"/",fleet.EPA.disp)

                                                dic_indic['v_sent'] += 1
                                                if degraded: 
                                                    dic_indic['v_degraded'] += 1 
                                                else:
                                                    dic_indic['v_sent_full'] += 1 
                                                dic_indic['ff_sent'] += len(ff_to_send)

                                                dic_veh_typ = update_dict(dic_veh_typ, vehicle_to_find) # metrique

                                                if (current_station in Z_1): # GESTION DES RENFORTS
    
                                                    if not fleet.VSAV.sent: # s'il n'y a pas de renfort en route
                                                        fleet.VSAV.disp, fleet.VSAV.to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "VSAV") 
                                                        
                                                        stations_VSAV, fleet.VSAV.needed, fleet.VSAV.to_return, new_required_departure, fleet.VSAV.to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(fleet.VSAV.disp, fleet.VSAV.needed, fleet.VSAV.to_return, fleet.VSAV.lent, fleet.VSAV.to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 2, "VSAV", 99) 
                                                        
                                                        dic_indic['VSAV_needed'] += int(fleet.VSAV.needed)
                                                        dic_indic['VSAV_disp'] = int(fleet.VSAV.disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)
    
                                                    elif not fleet.FPT.sent:
                                                        fleet.FPT.disp, fleet.FPT.to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "FPT")
                                                        
                                                        stations_FPT, fleet.FPT.needed, fleet.FPT.to_return, new_required_departure, fleet.FPT.to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(fleet.FPT.disp, fleet.FPT.needed, fleet.FPT.to_return, fleet.FPT.lent, fleet.FPT.to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 2, "FPT", 100) 
    
                                                        dic_indic['FPT_needed'] += int(fleet.FPT.needed)
                                                        dic_indic['FPT_disp'] = int(fleet.FPT.disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)
    
                                                    elif not fleet.EPA.sent:
                                                        fleet.EPA.disp, fleet.EPA.to_station = get_potential_veh(Z_1, dic_vehicles, dic_functions, "EPA")
                                                        
                                                        stations_EPA, fleet.EPA.needed, fleet.EPA.to_return, new_required_departure, fleet.EPA.to_station, \
                                                        dic_ff, v_mat_to_return = veh_management(fleet.EPA.disp, fleet.EPA.needed, fleet.EPA.to_return, fleet.EPA.lent, fleet.EPA.to_station, \
                                                                                new_required_departure, dic_station_distance, num_inter, dic_lent, \
                                                                                dic_vehicles, dic_functions, dic_ff, 1, "EPA", 101)   
    
                                                        dic_indic['EPA_needed'] += int(fleet.EPA.needed)
                                                        dic_indic['EPA_disp'] = int(fleet.EPA.disp)
                                                        # print(num_inter, "v_mat", v_mat_to_return)

                                        
                                        else: # S'il y a un rôle à pourvoir

                                            info_avail = get_neighborhood_availability(pdd, current_station, num_d, dic_vehicles, \
                                                                                       planning, month, day, hour, 5) 
    
                                            ff_mats = planning[current_station][month][day][hour]["planned"].copy()

                                            ff_existing = adding_lent_ff(fleet.VSAV.lent, fleet.FPT.lent, fleet.EPA.lent, \
                                                                         current_station, Z_1, dic_lent, ff_mats, dic_ff)  
                                            
                                            if v_waiting:

                                                # print("role to fill and v_waiting")

                                                v_mat = v_mat_to_return

                                                if fleet.VSAV.to_return and (current_station==fleet.VSAV.to_station) and "VSAV" in dic_functions[v_mat]:    
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    # print("VSAV_to_return", fleet.VSAV.to_return, "all_ff_waiting", all_ff_waiting)
    
                                                if not all_ff_waiting and fleet.FPT.to_return and (current_station==fleet.FPT.to_station) and \
                                                "FPT" in dic_functions[v_mat]:    
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    fleet.VSAV.to_return = False
                                                    # print("FPT_to_return", fleet.FPT.to_return, "all_ff_waiting", all_ff_waiting)
                                                    
                                                if not all_ff_waiting and fleet.EPA.to_return and (current_station==fleet.EPA.to_station) and \
                                                "EPA" in dic_functions[v_mat]:   
                                                    all_ff_waiting = are_all_ff_waiting(ff_existing, current_station, \
                                                                                                   dic_lent, dic_ff, v_mat)
                                                    fleet.FPT.to_return = False
                                                    # print("EPA_to_return", fleet.EPA.to_return, "all_ff_waiting", all_ff_waiting)

                                                if all_ff_waiting:
                                                    # print("all_ff_waiting", v_mat, current_station)
                                                    ff_existing = dic_lent[current_station][v_mat]
                                                    # print(ff_existing)

                                                else: 
                                                    # print("not all ff waiting")
                                                    ff_existing = [f for f in ff_existing if dic_ff[f] > -1].copy()
                                                    fleet.EPA.to_return = False
                                                    
                                            else: # no vehicle waiting
                                                lent_ff = [ff for v_m in dic_lent.values() for ff_lent in v_m.values() for ff in ff_lent]
                                                ff_not_lent = [num for num in ff_existing if num not in lent_ff]
                                                ff_existing = [f for f in ff_not_lent if dic_ff[f] > -1].copy()

                                            ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)

                                            state = gen_state(veh_depart, idx_role, ff_array, ff_existing, \
                                                              dic_roles, dic_roles_skills, dic_ff, df_skills, \
                                                              coord_x, coord_y, month_sin, month_cos, day_sin, \
                                                              day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size)

                                            action, skill_lvl, potential_actions = decide(
                                                state, all_ff_waiting, ff_array, inter_done
                                            )
                                            
                                            dic_indic, dic_lent, all_roles_found, vehicle_found, planning, dic_vehicles, dic_ff, idx_role, \
                                            degraded = step(action, idx_role, ff_existing, all_ff_waiting, current_station, Z_1, dic_lent, \
                                                            v_mat, dic_ff, fleet.VSAV.lent, fleet.FPT.lent, fleet.EPA.lent, planning, month, day, hour, num_inter, \
                                                            new_required_departure, num_d, list_v, num_role, mandatory, degraded, team_max, \
                                                            all_roles_found, vehicle_found, dic_vehicles, dic_indic, \
                                                            skill_lvl, station_lvl)

                                            # Before dic_indic_old is refreshed: the reward is the
                                            # delta between the two, so the agent must read it here.
                                            if on_action is not None:
                                                on_action(ActionContext(
                                                    state=state, action=action, skill_lvl=skill_lvl,
                                                    num_d=num_d, inter_done=inter_done,
                                                    dic_indic=dic_indic, dic_indic_old=dic_indic_old,
                                                    potential_actions=potential_actions,
                                                ))

                                            dic_indic_old = dic_indic.copy()
                                            action_num += 1 # for metrics


                                # else: # si aucun véhicule n'a la fonction requise
                                #     print(num_inter, veh_depart, vehicle_to_find, "no vehicule found")
                                    

        old_date = date

        # Two separate ticks, both keyed on RETURN events. The agent's epsilon
        # decay and checkpointing run on their own schedules, NOT on the logging
        # one — folding them together silently stops epsilon from decaying.
        if required_departure == {0: "RETURN"}:
            if on_interval is not None and num_inter % 100 == 0:
                on_interval(num_inter, vehicle_out, fleet)
            if on_return is not None:
                on_return(num_inter)

    # Hand back what the callers report on.
    env.dic_indic = dic_indic
    env.dic_vehicles = dic_vehicles
    env.planning = planning
    env.dic_ff = dic_ff
    return {
        "vehicle_out": vehicle_out,
        "action_num": action_num,
        "vehicle_evo": vehicle_evo,
    }
