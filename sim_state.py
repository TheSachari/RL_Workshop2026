"""Grouped simulation state.

The simulation loop carried ~116 loose locals. This module gives the two
largest, most regular groups a name, so the loop can be extracted into a shared
function without threading dozens of arguments through eleven nesting levels.

Deliberately *not* a rewrite of the logic: these are containers holding exactly
the values the loop already kept, under the same names. The behaviour of a run
is unchanged — the golden cases verify that.

`ReinforcementState` is the big win. The loop tracked nine fields for each of
the three reinforceable vehicle types (VSAV, FPT, EPA) as 27 separate locals:

    VSAV_sent, VSAV_lent, VSAV_needed, VSAV_to_return, VSAV_returning,
    VSAV_to_station, VSAV_from_station, VSAV_arrival_num, VSAV_disp
    ... and the same nine again for FPT, and again for EPA

The three sets are perfectly symmetric (no field exists for one type and not
another), but the *logic* around them is not interchangeable — each type has its
own `num_d` sentinel, and some branches carry extra guards. So this groups the
state without merging the code paths.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Departure sentinel identifying a reinforcement request per vehicle type.
# Used as `num_d == 99` for VSAV, 100 for FPT, 101 for EPA.
SENTINEL = {"VSAV": 99, "FPT": 100, "EPA": 101}

VEHICLE_TYPES = ("VSAV", "FPT", "EPA")


@dataclass
class ReinforcementState:
    """Reinforcement bookkeeping for one vehicle type (VSAV, FPT or EPA)."""

    v_type: str

    sent: bool = False
    lent: bool = False
    needed: bool = False
    to_return: bool = False
    returning: bool = False

    to_station: str = ""
    from_station: str = ""

    arrival_num: int = 0
    disp: int = 0

    @property
    def num_d(self) -> int:
        """The departure sentinel that identifies this type's reinforcement."""
        return SENTINEL[self.v_type]


@dataclass
class Fleet:
    """The three reinforcement states, addressable by name or by type string."""

    VSAV: ReinforcementState = field(
        default_factory=lambda: ReinforcementState("VSAV")
    )
    FPT: ReinforcementState = field(default_factory=lambda: ReinforcementState("FPT"))
    EPA: ReinforcementState = field(default_factory=lambda: ReinforcementState("EPA"))

    def __getitem__(self, v_type: str) -> ReinforcementState:
        return getattr(self, v_type)

    def __iter__(self):
        return iter((self.VSAV, self.FPT, self.EPA))


@dataclass
class Environment:
    """Everything `load_environment_variables` returns, under one name.

    The loader previously returned an 18-tuple that every caller unpacked by
    position — adding or reordering a field silently broke callers.
    """

    dic_vehicles: dict
    dic_functions: dict
    df_skills: object
    dic_roles_skills: dict
    dic_roles: dict
    planning: dict
    dic_inter: dict
    dic_ff: dict
    dic_indic: dict
    dic_indic_old: dict
    Z_1: List[str]
    Z_4: List[str]
    dic_lent: dict
    dic_station_distance: dict
    df_pc: object
    old_date: object
    date_reference: object
    skills_updated: object

    def as_tuple(self):
        """The original 18-tuple, for callers not yet migrated."""
        return (
            self.dic_vehicles, self.dic_functions, self.df_skills,
            self.dic_roles_skills, self.dic_roles, self.planning, self.dic_inter,
            self.dic_ff, self.dic_indic, self.dic_indic_old, self.Z_1, self.Z_4,
            self.dic_lent, self.dic_station_distance, self.df_pc, self.old_date,
            self.date_reference, self.skills_updated,
        )


@dataclass
class RunLog:
    """Per-run bookkeeping that outlives a single event row."""

    dic_log: Dict = field(default_factory=dict)
    dic_back: Dict = field(default_factory=dict)
    dic_start_time: Dict = field(default_factory=dict)
    dic_veh_typ: Dict = field(default_factory=dict)

    vehicle_evo: List = field(default_factory=list)
    reward_evo: List = field(default_factory=list)
    current_ff_inter: List = field(default_factory=list)

    vehicle_out: int = 0
    action_num: int = 0
    score: float = 0.0
