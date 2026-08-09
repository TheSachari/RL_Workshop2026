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

## Structure

The loop body used to be one 470-line `for` statement nested up to eighteen
blocks deep, because every level of the dispatch problem was expressed inline:
per event, per station, per required vehicle, per candidate function, per role.
The nesting was not incidental — it is what hid three `UnboundLocalError`s
(`stations_VSAV/FPT/EPA`, read ~200 lines above the only branch that assigns
them) behind a wall of indentation where no reader was going to find them.

It is now split along the seams the loop already had, outermost to innermost:

    run_simulation          per event row: arrivals, then RETURN or intervention
      _handle_intervention  per intervention: walks the stations
        _fill_station       per station: walks the vehicles still required
          _find_vehicle     per required vehicle: finds a machine with the function
            _fill_roles     per vehicle found: assigns crew, one role at a time

Each helper takes and returns a `_LoopState`, which holds exactly the locals the
original body mutated in place, under the same names. That is what keeps the
translation honest: the helpers are the original blocks, moved, not rewritten.

The three reinforceable vehicle types (VSAV, FPT, EPA) had their arrival,
dispatch, return and availability logic written out three times each — twelve
blocks that a token-level diff shows to be identical apart from the type name,
the `dic_indic` key, and one threshold (EPA uses 1 where the others use 2). They
are now loops over `VEHICLE_TYPES`, with the threshold carried as data in
`_REINFORCEMENT_THRESHOLD`. `fleet[v_type]` and `ReinforcementState.num_d`
already existed in `sim_state.py` for this.

Behaviour is unchanged, deliberately: same order of operations, same mutations,
same metrics. The one intended difference is the `stations_*` fix described in
`_LoopState`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from collective_functions import (
    adding_lent_ff,
    are_all_ff_waiting,
    gen_ff_array,
    gen_state,
    get_mandatory_max,
    get_neighborhood_availability,
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
from sim_state import VEHICLE_TYPES

# `veh_management`'s `threshold` argument. EPA keeps fewer in reserve than the
# other two; this was the only real difference between the three copies of the
# reinforcement-availability block.
_REINFORCEMENT_THRESHOLD = {"VSAV": 2, "FPT": 2, "EPA": 1}

# Departures numbered at or above this are reinforcement movements rather than
# vehicles belonging to the intervention's own train (99/100/101 per type; see
# `sim_state.SENTINEL`).
_REINFORCEMENT_NUM_D = 99

# Longest initial train dispatched in one pass; anything beyond spills into a
# follow-up departure.
_MAX_TRAIN = 5


def _slot(st):
    """The planning slot for the station and hour currently being served."""
    return st.planning[st.current_station][st.month][st.day][st.hour]


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


@dataclass
class _LoopState:
    """The locals the original loop body mutated in place, under one name.

    Passing this between the extracted helpers is what lets them be the original
    blocks rather than rewrites of them: a helper reads and assigns the same
    fields the inline code did, in the same order.

    `stations` holds the per-type station iterators consulted when a
    reinforcement sentinel comes up. In the original these were three bare
    locals (`stations_VSAV`, `stations_FPT`, `stations_EPA`) assigned only
    inside the `current_station in Z_1` branch but read from the top of the
    departure loop, so an event that hit a sentinel before any reinforcement had
    been arranged raised `UnboundLocalError` — ruff flagged all three as F821.
    Starting them as empty iterators preserves the intended reading (no
    candidate stations yet, so `next(..., False)` yields False and the departure
    is treated as unfulfillable) without the crash.
    """

    # --- environment handles (mutated in place) ---
    dic_vehicles: dict
    dic_functions: dict
    df_skills: Any
    dic_roles_skills: dict
    dic_roles: dict
    planning: dict
    dic_inter: dict
    dic_ff: dict
    dic_indic: dict
    dic_indic_old: dict
    Z_1: Any
    Z_4: Any
    dic_lent: dict
    dic_station_distance: dict
    df_pc: Any
    skills_updated: Any
    max_duration: Any

    # --- per-run bookkeeping ---
    fleet: Any
    action_size: int
    decide: Callable
    on_action: Optional[Callable[[ActionContext], None]] = None

    vehicle_out: int = 0
    action_num: int = 0
    num_d: int = 42
    all_ff_waiting: bool = False
    v_waiting: bool = False
    following_depart: bool = False

    vehicle_evo: list = field(default_factory=list)
    current_ff_inter: list = field(default_factory=list)
    dic_log: dict = field(default_factory=dict)
    dic_back: dict = field(default_factory=dict)
    dic_start_time: dict = field(default_factory=dict)
    dic_veh_typ: dict = field(default_factory=dict)

    # Per-type station iterators; see the class docstring for why these are
    # seeded rather than left unbound.
    stations: dict = field(
        default_factory=lambda: {v: iter(()) for v in VEHICLE_TYPES}
    )

    # --- per-event row fields, refreshed by `_unpack_row` ---
    idx: Any = None
    num_inter: Any = None
    date: Any = None
    duration: Any = None
    month: Any = None
    day: Any = None
    hour: Any = None
    coord_x: Any = None
    coord_y: Any = None
    month_sin: Any = None
    month_cos: Any = None
    day_sin: Any = None
    day_cos: Any = None
    hour_sin: Any = None
    hour_cos: Any = None

    # --- per-intervention scratch, reset by `_handle_intervention` ---
    pdd: Any = None
    required_departure: dict = field(default_factory=dict)
    new_required_departure: dict = field(default_factory=dict)
    veh_depart: list = field(default_factory=list)
    new_veh_depart: list = field(default_factory=list)
    station_iter: Optional[Iterator] = None
    inter_done: bool = False
    departure_done: bool = False
    station_lvl: int = 0
    idx_role: int = 0
    v_mat_to_return: int = 0
    current_station: Any = None
    required_vehicles: Optional[Iterator] = None
    list_v: list = field(default_factory=list)
    mandatory: Any = None
    team_max: Any = None
    v_mat: Any = None
    vehicle_to_find: Any = None
    vehicle_found: bool = False
    degraded: bool = False


def _unpack_row(st: _LoopState, row) -> None:
    """Copy one event row's fields onto the loop state.

    The agent's event stream carries a 20th column (rare skills required); the
    baseline's does not. Take the common prefix either way.
    """
    (
        st.idx, st.num_inter, st.date, pdd, required_departure, _zone, st.duration,
        st.month, st.day, st.hour, _minute, st.coord_x, st.coord_y,
        st.month_sin, st.month_cos, st.day_sin, st.day_cos, st.hour_sin, st.hour_cos,
    ) = row[:19]
    st.pdd = pdd
    st.required_departure = required_departure


def _handle_arrivals(st: _LoopState) -> None:
    """Land any reinforcement whose arrival event is this intervention."""
    for v_type in VEHICLE_TYPES:
        r = st.fleet[v_type]
        if not (r.sent and st.num_inter == r.arrival_num):
            continue

        reinforcement_arriving(st, r, v_type)


def _split_train(st: _LoopState) -> None:
    """Cap the departure at `_MAX_TRAIN`, spilling the rest into a follow-up."""
    if len(st.veh_depart) > _MAX_TRAIN:
        st.new_veh_depart = st.veh_depart[_MAX_TRAIN:].copy()
        st.veh_depart = st.veh_depart[:_MAX_TRAIN]
        st.required_departure = {i + 1: [v] for i, v in enumerate(st.veh_depart)}
        st.following_depart = True


def _manage_reinforcements(st: _LoopState) -> None:
    """Reassess reinforcement needs after a vehicle left a Z_1 station.

    Only the first type with no reinforcement already in flight is considered,
    which is what the original `if not VSAV.sent / elif not FPT.sent / elif not
    EPA.sent` chain expressed.
    """
    for v_type in VEHICLE_TYPES:
        r = st.fleet[v_type]
        if r.sent:
            continue

        r.disp, r.to_station = get_potential_veh(
            st.Z_1, st.dic_vehicles, st.dic_functions, v_type
        )

        veh_management(st, r, _REINFORCEMENT_THRESHOLD[v_type], v_type)

        st.dic_indic[f"{v_type}_needed"] += int(r.needed)
        st.dic_indic[f"{v_type}_disp"] = int(r.disp)
        return


def _send_reinforcement(st: _LoopState, ff_to_send: list) -> bool:
    """Dispatch this vehicle as a reinforcement, if that is what it is."""
    for v_type in VEHICLE_TYPES:
        r = st.fleet[v_type]
        if not (r.needed and st.num_d == r.num_d):
            continue

        st.dic_start_time[st.v_mat] = (st.month, st.day, st.hour)
        reinforcement_sending(st, r, ff_to_send, v_type)
        st.dic_indic[f"z1_{v_type}_sent"] += 1
        return True
    return False


def _return_reinforcement(st: _LoopState, ff_to_send: list) -> None:
    """Send a borrowed vehicle home, or release it if its crew is unavailable."""
    if st.all_ff_waiting:
        for v_type in VEHICLE_TYPES:
            r = st.fleet[v_type]
            if not (r.to_return and st.num_d == r.num_d):
                continue

            reinforcement_returning(st, r, ff_to_send, v_type)
            return
    else:
        # Return impossible, firefighters unavailable: make it available again.
        st.v_waiting = False
        st.dic_vehicles[st.current_station]["available"].append(st.v_mat)
        _slot(st)["available"] += ff_to_send


def _depart_on_intervention(st: _LoopState, ff_to_send: list) -> None:
    """Commit the vehicle and its crew to the intervention, and update metrics."""
    st.dic_inter[st.num_inter][st.current_station][st.v_mat] = ff_to_send
    st.dic_vehicles[st.current_station]["inter"].append(st.v_mat)

    st.current_ff_inter += ff_to_send
    for f in ff_to_send:
        st.dic_ff[f] = st.duration

    st.vehicle_out += 1

    st.dic_indic["v_sent"] += 1
    if st.degraded:
        st.dic_indic["v_degraded"] += 1
    else:
        st.dic_indic["v_sent_full"] += 1
    st.dic_indic["ff_sent"] += len(ff_to_send)

    st.dic_veh_typ = update_dict(st.dic_veh_typ, st.vehicle_to_find)

    if st.current_station in st.Z_1:
        _manage_reinforcements(st)


def _resolve_crew(st: _LoopState) -> list:
    """Work out which firefighters are available to crew `st.v_mat`.

    When a borrowed vehicle is waiting to go home, its own lent crew is the
    candidate pool and `all_ff_waiting` records whether they are all free;
    otherwise the pool is the station's planned crew minus anyone lent out or
    already committed.
    """
    ff_mats = _slot(st)["planned"].copy()
    ff_existing = adding_lent_ff(
        st.fleet.VSAV.lent, st.fleet.FPT.lent, st.fleet.EPA.lent,
        st.current_station, st.Z_1, st.dic_lent, ff_mats, st.dic_ff,
    )

    if not st.v_waiting:
        lent_ff = [
            ff
            for v_m in st.dic_lent.values()
            for ff_lent in v_m.values()
            for ff in ff_lent
        ]
        ff_not_lent = [num for num in ff_existing if num not in lent_ff]
        return [f for f in ff_not_lent if st.dic_ff[f] > -1].copy()

    st.v_mat = st.v_mat_to_return

    # Checked in fleet order, and each type clears its predecessor's flag once
    # it takes over the return — the original wrote this as three guarded ifs.
    previous = None
    for v_type in VEHICLE_TYPES:
        r = st.fleet[v_type]
        if st.all_ff_waiting:
            break
        if r.to_return and st.current_station == r.to_station and \
                v_type in st.dic_functions[st.v_mat]:
            st.all_ff_waiting = are_all_ff_waiting(
                ff_existing, st.current_station, st.dic_lent, st.dic_ff, st.v_mat
            )
            if previous is not None:
                st.fleet[previous].to_return = False
        previous = v_type

    if st.all_ff_waiting:
        return st.dic_lent[st.current_station][st.v_mat]

    st.fleet.EPA.to_return = False
    return [f for f in ff_existing if st.dic_ff[f] > -1].copy()


def _fill_roles(st: _LoopState) -> None:
    """Assign crew to `st.v_mat`, one role at a time, until the vehicle leaves.

    Each pass either fills one role (delegating the choice to `decide`) or, when
    no role is left, releases the vehicle — as a reinforcement being sent, a
    reinforcement going home, or a departure on the intervention.
    """
    roles = st.dic_roles[st.vehicle_to_find]
    role_number = iter(range(1, len(roles) + 1))
    all_roles_found = False
    st.degraded = False

    while not all_roles_found:

        num_role = next(role_number, 0)
        if num_role > st.team_max:  # Pour limiter les rôles
            num_role = 0

        if not num_role:  # S'il n'y a plus de rôles à pourvoir
            all_roles_found = True
            ff_to_send = _slot(st)["standby"].copy()
            _slot(st)["standby"] = []
            st.dic_vehicles[st.current_station]["standby"].remove(st.v_mat)

            if _send_reinforcement(st, ff_to_send):
                continue
            if st.v_waiting:
                _return_reinforcement(st, ff_to_send)
            else:
                _depart_on_intervention(st, ff_to_send)
            continue

        # S'il y a un rôle à pourvoir
        info_avail = get_neighborhood_availability(
            st.pdd, st.current_station, st.num_d, st.dic_vehicles,
            st.planning, st.month, st.day, st.hour, 5,
        )

        ff_existing = _resolve_crew(st)
        ff_array = gen_ff_array(st.df_skills, st.skills_updated, ff_existing)

        state = gen_state(st, ff_array, ff_existing, info_avail)

        action, skill_lvl, potential_actions = st.decide(
            state, st.all_ff_waiting, ff_array, st.inter_done
        )

        all_roles_found, st.vehicle_found = step(
            st, action, ff_existing, num_role, all_roles_found, skill_lvl,
        )

        # Before dic_indic_old is refreshed: the reward is the delta between the
        # two, so the agent must read it here.
        if st.on_action is not None:
            st.on_action(ActionContext(
                state=state, action=action, skill_lvl=skill_lvl,
                num_d=st.num_d, inter_done=st.inter_done,
                dic_indic=st.dic_indic, dic_indic_old=st.dic_indic_old,
                potential_actions=potential_actions,
            ))

        st.dic_indic_old = st.dic_indic.copy()
        st.action_num += 1  # for metrics


def _find_vehicle(st: _LoopState) -> None:
    """Find a machine at `st.current_station` providing one required function.

    Walks the functions this departure will accept. The first one the station
    can serve wins and goes on to crewing; if none can be served, the departure
    is deferred to the next station via `new_required_departure`.
    """
    if st.num_d >= _REINFORCEMENT_NUM_D:
        for v_type in VEHICLE_TYPES:
            if st.num_d == st.fleet[v_type].num_d:
                st.current_station = next(st.stations[v_type], False)
                break

    vehicles_to_find = iter(st.list_v)
    st.vehicle_found = False

    if not st.current_station:
        st.vehicle_found = True
        st.all_ff_waiting = False

    while not st.vehicle_found:

        st.vehicle_to_find = next(vehicles_to_find, False)

        if not st.vehicle_to_find:  # S'il n'y a plus de fonction à faire partir
            st.idx_role += st.team_max
            st.vehicle_found = True
            st.new_required_departure[st.num_d] = st.list_v
            st.dic_indic["function_not_found"] += 1
            continue

        # On cherche les véhicules disponibles dans la caserne actuelle
        # correspondant à la fonction requise.
        v_mats = st.dic_vehicles[st.current_station]["available"].copy()
        li_mat_veh = [
            v_m for v_m in v_mats if st.vehicle_to_find in st.dic_functions[v_m]
        ]

        if not li_mat_veh:  # si aucun véhicule n'a la fonction requise
            continue

        st.v_mat = li_mat_veh[0]
        for v_type in VEHICLE_TYPES:
            r = st.fleet[v_type]
            if r.to_return and st.current_station == r.to_station \
                    and st.num_d == r.num_d:
                st.v_mat, st.v_waiting = v_to_return_managing(
                    st.dic_log, li_mat_veh, st.v_waiting, st.vehicle_to_find,
                    st.current_station, st.dic_vehicles, v_type, st.v_mat_to_return,
                )
                break

        if st.num_d < _REINFORCEMENT_NUM_D:
            st.veh_depart[st.num_d - 1] = st.vehicle_to_find

        st.mandatory, st.team_max = get_mandatory_max(st.vehicle_to_find)

        # On met le véhicule en standby
        st.dic_vehicles[st.current_station]["available"].remove(st.v_mat)
        st.dic_vehicles[st.current_station]["standby"].append(st.v_mat)
        st.vehicle_found = True

        _fill_roles(st)


def _advance_train(st: _LoopState) -> None:
    """No vehicle left to place from this station: move to the next train.

    Promotes the deferred departures to the active train, or ends the
    intervention — unless a follow-up train is pending, which restarts the
    station walk from the top.
    """
    st.departure_done = True

    if st.new_required_departure:
        st.required_departure = update_dep(st.new_required_departure)
        st.veh_depart = [v[0] for k, v in sorted(st.required_departure.items())]
        st.new_required_departure = {}
        st.idx_role = 0
        return

    st.inter_done = True
    if not st.following_depart:
        return

    # A follow-up departure is pending: restart from the nearest station.
    st.inter_done = False
    st.station_iter = iter(st.pdd)
    st.station_lvl = 0
    st.veh_depart = st.new_veh_depart
    st.required_departure = update_dep(
        {i + 1: [v] for i, v in enumerate(st.new_veh_depart)}
    )
    _split_train(st)
    st.new_required_departure = {}
    st.idx_role, st.num_d = 0, 1
    st.following_depart = False


def _fill_station(st: _LoopState) -> None:
    """Place as many of the still-required vehicles as this station can supply."""
    while not st.departure_done:

        # On cherche les véhicules requis dans le train initial.
        st.num_d, st.list_v = next(st.required_vehicles, (0, []))

        if st.station_lvl == 2 and st.num_d == 1:
            st.dic_indic["v1_not_sent_from_s1"] += 1
        if st.station_lvl > 3 and st.num_d <= 3 and st.current_station in st.Z_4:
            st.dic_indic["v3_not_sent_from_s3"] += 1

        if not st.list_v:
            _advance_train(st)
            continue

        st.mandatory, st.team_max = get_mandatory_max(st.list_v[0])
        _find_vehicle(st)


def _handle_intervention(st: _LoopState) -> None:
    """Dispatch one intervention, walking outward through the nearest stations."""
    st.veh_depart = [v[0] for k, v in sorted(st.required_departure.items())]
    st.dic_indic["v_required"] += len(st.required_departure)
    _split_train(st)

    st.new_required_departure = {}
    st.station_iter = iter(st.pdd)
    st.inter_done = False
    for v_type in VEHICLE_TYPES:
        st.fleet[v_type].to_return = False
    st.station_lvl = 0
    st.idx_role = 0
    st.v_mat_to_return = 0

    st.vehicle_evo.append([st.num_inter, st.vehicle_out])

    while not st.inter_done:  # Tant que l'intervention n'est pas finie

        # On va dans la plus proche caserne.
        st.current_station = next(st.station_iter, False)
        if st.num_d < _REINFORCEMENT_NUM_D \
                and st.current_station not in st.dic_inter[st.num_inter]:
            # Il s'agit d'une intervention et non d'un renfort.
            st.dic_inter[st.num_inter][st.current_station] = {}

        st.station_lvl += 1

        if not st.current_station:
            # Plus de caserne à visiter: l'intervention est terminée.
            st.inter_done = True
            st.departure_done = True
            st.dic_indic["v_not_found_in_last_station"] += len(st.required_departure)
        else:
            st.departure_done = False
            st.required_vehicles = iter(sorted(st.required_departure.items()))

        _fill_station(st)


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
    resume_from=None,
    loop_state=None,
    on_state_ready=None,
):
    """Run the event stream, delegating each action choice to `decide`.

    `env` is the `Environment` from `collective_functions.load_environment`;
    `fleet` the `Fleet` reinforcement state. Both are mutated in place, as the
    original loop did — the caller reads its metrics back off `env.dic_indic`.

    `on_row(row_fields)` runs once per event before it is handled (the agent
    runner uses it to precompute upcoming rare skills). `on_interval(num_inter)`
    runs on the loop's periodic logging tick.

    Resuming
    --------
    `resume_from` is the event index of the last row a checkpoint covered; rows
    up to and including it are skipped, so the stream continues *after* it
    rather than replaying it. `loop_state` seeds the cross-event `_LoopState`
    fields from that checkpoint, and `on_state_ready(st)` hands the caller the
    live state object so a checkpoint can capture it mid-run. All three default
    to None, leaving a normal run byte-for-byte on its original path.
    """
    st = _LoopState(
        dic_vehicles=env.dic_vehicles,
        dic_functions=env.dic_functions,
        df_skills=env.df_skills,
        dic_roles_skills=env.dic_roles_skills,
        dic_roles=env.dic_roles,
        planning=env.planning,
        dic_inter=env.dic_inter,
        dic_ff=env.dic_ff,
        dic_indic=env.dic_indic,
        dic_indic_old=env.dic_indic_old,
        Z_1=env.Z_1,
        Z_4=env.Z_4,
        dic_lent=env.dic_lent,
        dic_station_distance=env.dic_station_distance,
        df_pc=env.df_pc,
        skills_updated=env.skills_updated,
        max_duration=env.df_pc["Duration"].max(),
        fleet=fleet,
        action_size=action_size,
        decide=decide,
        on_action=on_action,
    )

    # Seed the cross-event fields from a checkpoint before the loop starts, so
    # the first resumed event sees the same bookkeeping the interrupted run had.
    if loop_state:
        for key, value in loop_state.items():
            setattr(st, key, value)

    if on_state_ready is not None:
        on_state_ready(st)

    old_date = env.old_date
    date_reference = env.date_reference

    for row in st.df_pc.itertuples(index=True, name=None):

        # Skip everything the checkpoint already accounted for. Comparing on the
        # row index (not a counter) keeps this correct even though `num_inter`
        # repeats across the departure/RETURN pair of a single intervention.
        if resume_from is not None and row[0] <= resume_from:
            continue

        _unpack_row(st, row)

        if on_row is not None:
            on_row(st.idx, st.date, st.pdd, st.df_pc)

        st.dic_ff = update_duration(st.date, old_date, st.current_ff_inter, st.dic_ff)

        if st.date >= date_reference:
            date_reference = st.date
            st.skills_updated = update_skills(st.df_skills, date_reference)

        # Both dates are loop locals, so a checkpoint taken from a callback
        # would otherwise read whatever `load_environment` left on `env` rather
        # than where the run has actually got to. Mirror them as we go.
        env.old_date = old_date
        env.date_reference = date_reference

        _handle_arrivals(st)

        if st.required_departure == {0: "RETURN"}:  # RETOUR D'INTERVENTION
            (
                st.vehicle_out, st.dic_vehicles, st.dic_ff, st.current_ff_inter,
                st.planning, st.dic_inter,
            ) = returning(
                st.df_pc, st.dic_inter, st.num_inter, st.vehicle_out,
                st.dic_vehicles, st.dic_ff, st.current_ff_inter, st.planning,
                st.month, st.day, st.hour,
            )
        else:
            _handle_intervention(st)

        old_date = st.date

        # Two separate ticks, both keyed on RETURN events. The agent's epsilon
        # decay and checkpointing run on their own schedules, NOT on the logging
        # one — folding them together silently stops epsilon from decaying.
        if st.required_departure == {0: "RETURN"}:
            if on_interval is not None and st.num_inter % 100 == 0:
                on_interval(st.num_inter, st.vehicle_out, fleet)
            if on_return is not None:
                on_return(st.num_inter)

    # Hand back what the callers report on.
    env.dic_indic = st.dic_indic
    env.dic_vehicles = st.dic_vehicles
    env.planning = st.planning
    env.dic_ff = st.dic_ff
    return {
        "vehicle_out": st.vehicle_out,
        "action_num": st.action_num,
        "vehicle_evo": st.vehicle_evo,
    }
