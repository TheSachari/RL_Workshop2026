"""The per-type dispatch branches in `simulator`, pinned against the originals.

`simulator` used to write the VSAV/FPT/EPA logic out three times in each of
several places; those blocks are now loops over `VEHICLE_TYPES`. The collapse is
only correct if the loop picks the same type, and leaves the same `to_return`
flags behind, as the original `if / elif / elif` chains did — including the
non-obvious part, where each type clears its *predecessor's* flag once it takes
over the return.

The whole-run golden cases would catch a mistake here, but they need the
dataset. These compare the branch logic directly against a transcription of the
original code, over every combination of the flags that select a branch, so a
regression is caught in a second with no data.

The transcriptions below are the pre-refactor code, kept verbatim on purpose:
they are the reference, so they must not be "cleaned up" to match the new shape.
"""

import itertools

import pytest

from sim_state import VEHICLE_TYPES, Fleet


def _fleet(to_return_flags, to_stations):
    f = Fleet()
    for v_type, flag, station in zip(VEHICLE_TYPES, to_return_flags, to_stations):
        f[v_type].to_return = flag
        f[v_type].to_station = station
    return f


def _flags(f):
    return tuple((r.v_type, r.to_return) for r in f)


# --- transcriptions of the original blocks -------------------------------

def _original_resolve(fleet, current_station, functions, all_ff_waiting):
    """Original `_resolve_crew` v_waiting branch (simulator.py:507-534, pre-refactor).

    `are_all_ff_waiting(...)` is stubbed to True: the question here is which
    branch runs and what it leaves behind, not what that helper computes.
    """
    if fleet.VSAV.to_return and (current_station == fleet.VSAV.to_station) and \
            "VSAV" in functions:
        all_ff_waiting = True

    if not all_ff_waiting and fleet.FPT.to_return and \
            (current_station == fleet.FPT.to_station) and "FPT" in functions:
        all_ff_waiting = True
        fleet.VSAV.to_return = False

    if not all_ff_waiting and fleet.EPA.to_return and \
            (current_station == fleet.EPA.to_station) and "EPA" in functions:
        all_ff_waiting = True
        fleet.FPT.to_return = False

    if not all_ff_waiting:
        fleet.EPA.to_return = False

    return all_ff_waiting, _flags(fleet)


def _new_resolve(fleet, current_station, functions, all_ff_waiting):
    """The loop as written in `simulator._resolve_crew`."""
    previous = None
    for v_type in VEHICLE_TYPES:
        r = fleet[v_type]
        if all_ff_waiting:
            break
        if r.to_return and current_station == r.to_station and v_type in functions:
            all_ff_waiting = True
            if previous is not None:
                fleet[previous].to_return = False
        previous = v_type

    if not all_ff_waiting:
        fleet.EPA.to_return = False

    return all_ff_waiting, _flags(fleet)


def _original_pick(fleet, current_station, num_d):
    """Original `_find_vehicle` to_return selection (simulator.py:298-314)."""
    if fleet.VSAV.to_return and (current_station == fleet.VSAV.to_station) \
            and (num_d == 99):
        return "VSAV"
    elif fleet.FPT.to_return and (current_station == fleet.FPT.to_station) \
            and (num_d == 100):
        return "FPT"
    elif fleet.EPA.to_return and (current_station == fleet.EPA.to_station) \
            and (num_d == 101):
        return "EPA"
    return None


def _new_pick(fleet, current_station, num_d):
    """The loop as written in `simulator._find_vehicle`."""
    for v_type in VEHICLE_TYPES:
        r = fleet[v_type]
        if r.to_return and current_station == r.to_station and num_d == r.num_d:
            return v_type
    return None


def _original_return(fleet, num_d):
    """Original `_return_reinforcement` all_ff_waiting branch (simulator.py:391-421)."""
    if fleet.VSAV.to_return and (num_d == 99):
        return "VSAV"
    elif fleet.FPT.to_return and (num_d == 100):
        return "FPT"
    elif fleet.EPA.to_return and (num_d == 101):
        return "EPA"
    return None


def _new_return(fleet, num_d):
    """The loop as written in `simulator._return_reinforcement`."""
    for v_type in VEHICLE_TYPES:
        r = fleet[v_type]
        if r.to_return and num_d == r.num_d:
            return v_type
    return None


# --- the cases ------------------------------------------------------------

STATIONS = ("S1", "S2")
FLAG_SETS = list(itertools.product([False, True], repeat=3))
STATION_SETS = list(itertools.product(STATIONS, repeat=3))
FUNCTION_SETS = [
    set(c)
    for k in range(len(VEHICLE_TYPES) + 1)
    for c in itertools.combinations(VEHICLE_TYPES, k)
]


@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("stations", STATION_SETS)
@pytest.mark.parametrize("current", STATIONS)
def test_resolve_crew_matches_original(flags, stations, current):
    """Same all_ff_waiting, and the same to_return flags left behind."""
    for functions in FUNCTION_SETS:
        for all_ff_waiting in (False, True):
            old = _original_resolve(
                _fleet(flags, stations), current, functions, all_ff_waiting
            )
            new = _new_resolve(
                _fleet(flags, stations), current, functions, all_ff_waiting
            )
            assert old == new, (
                f"flags={flags} stations={stations} current={current} "
                f"functions={sorted(functions)} all_ff_waiting={all_ff_waiting}"
            )


@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("stations", STATION_SETS)
@pytest.mark.parametrize("current", STATIONS)
@pytest.mark.parametrize("num_d", [1, 5, 99, 100, 101])
def test_find_vehicle_pick_matches_original(flags, stations, current, num_d):
    f = _fleet(flags, stations)
    assert _original_pick(f, current, num_d) == _new_pick(f, current, num_d)


@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("num_d", [1, 5, 99, 100, 101])
def test_return_reinforcement_matches_original(flags, num_d):
    f = _fleet(flags, ("S1", "S1", "S1"))
    assert _original_return(f, num_d) == _new_return(f, num_d)


def test_sentinels_line_up_with_the_num_d_constants():
    """The loops key off `ReinforcementState.num_d`; the originals hardcoded these."""
    assert [Fleet()[v].num_d for v in VEHICLE_TYPES] == [99, 100, 101]
