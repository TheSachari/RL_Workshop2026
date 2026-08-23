"""Tests for the copy-on-write planning store.

Three properties this pins, each of which broke a golden case while the store
was being written:

* list order survives, including re-append after removal;
* `planned`/`available` aliasing from the generated pickle is reproduced;
* a copy is independent of its base, in both directions.
"""

import pytest

from planning_state import PlanningView, SlotStore


def nested(planned, available=None, standby=None, alias=False):
    """One-slot planning dict at ('S', 1, 1, 0)."""
    if alias:
        shared = list(planned)
        slot = {"planned": shared, "available": shared, "standby": standby or []}
    else:
        slot = {
            "planned": list(planned),
            "available": list(planned if available is None else available),
            "standby": list(standby or []),
        }
    return {"S": {1: {1: {0: slot}}}}


def view(planning):
    return PlanningView(SlotStore.from_planning(planning))[ "S"][1][1][0]


class TestListSemantics:
    def test_initial_contents_match(self):
        slot = view(nested([3, 1, 2]))
        assert list(slot["planned"]) == [3, 1, 2]
        assert list(slot["available"]) == [3, 1, 2]
        assert list(slot["standby"]) == []

    def test_readd_after_remove_goes_to_the_end(self):
        """A roster-ordered structure would put it back in place. 36% of the
        appends in a reference run are this case."""
        slot = view(nested([10, 20, 30]))
        slot["available"].remove(10)
        slot["available"].append(10)
        assert list(slot["available"]) == [20, 30, 10]

    def test_off_roster_append_is_kept(self):
        """Two call sites append without checking the roster."""
        slot = view(nested([1, 2]))
        slot["available"].append(999)
        assert list(slot["available"]) == [1, 2, 999]

    def test_remove_of_absent_member_raises(self):
        slot = view(nested([1, 2]))
        with pytest.raises(ValueError):
            slot["available"].remove(77)

    def test_standby_can_be_cleared_by_assignment(self):
        slot = view(nested([1, 2], standby=[1]))
        slot["standby"] = []
        assert list(slot["standby"]) == []

    def test_len_and_contains(self):
        slot = view(nested([1, 2, 3]))
        assert len(slot["available"]) == 3
        assert 2 in slot["available"]
        assert 9 not in slot["available"]

    def test_copy_returns_a_detached_list(self):
        slot = view(nested([1, 2]))
        got = slot["available"].copy()
        got.append(3)
        assert list(slot["available"]) == [1, 2]


class TestRoster:
    def test_planned_cannot_be_assigned(self):
        slot = view(nested([1, 2]))
        with pytest.raises(TypeError):
            slot["planned"] = [3]

    def test_planned_is_not_mutable_in_place(self):
        """Rosters are shared between copies, so an in-place edit would reach
        every one of them. The nested dicts gave each planning its own list."""
        slot = view(nested([1, 2]))
        assert not hasattr(slot["planned"], "append")

    def test_planned_copy_gives_a_real_list(self):
        slot = view(nested([1, 2]))
        got = slot["planned"].copy()
        got.append(3)
        assert list(slot["planned"]) == [1, 2]


class TestAliasing:
    """`generate_environment` puts one list under both `planned` and
    `available` for 308,811 of 334,056 slots, so removing from one removes
    from the other. Splitting them moved every golden metric."""

    def test_alias_is_reproduced(self):
        slot = view(nested([1, 2, 3], alias=True))
        slot["available"].remove(2)
        assert list(slot["planned"]) == [1, 3]

    def test_unaliased_slots_stay_independent(self):
        slot = view(nested([1, 2, 3], available=[1, 2, 3]))
        slot["available"].remove(2)
        assert list(slot["planned"]) == [1, 2, 3]

    def test_alias_survives_a_copy(self):
        store = SlotStore.from_planning(nested([1, 2, 3], alias=True))
        clone = PlanningView(store.copy())["S"][1][1][0]
        clone["available"].remove(2)
        assert list(clone["planned"]) == [1, 3]


class TestCopyOnWrite:
    def test_copy_starts_equal(self):
        store = SlotStore.from_planning(nested([1, 2, 3]))
        clone = PlanningView(store.copy())["S"][1][1][0]
        assert list(clone["available"]) == [1, 2, 3]

    def test_writing_the_copy_leaves_the_base_alone(self):
        store = SlotStore.from_planning(nested([1, 2, 3]))
        base = PlanningView(store)["S"][1][1][0]
        clone = PlanningView(store.copy())["S"][1][1][0]

        clone["available"].remove(1)

        assert list(clone["available"]) == [2, 3]
        assert list(base["available"]) == [1, 2, 3]

    def test_writing_the_base_leaves_an_untouched_copy_alone(self):
        """The copy reads through to the base until it writes, so a base edit
        after the copy must not leak into it."""
        store = SlotStore.from_planning(nested([1, 2, 3]))
        copy_store = store.copy()
        base = PlanningView(store)["S"][1][1][0]
        clone = PlanningView(copy_store)["S"][1][1][0]

        clone["available"].copy()          # read, no write
        base["available"].append(4)

        assert list(clone["available"]) == [1, 2, 3]

    def test_copies_stay_independent_in_both_directions(self):
        """Two forks of one environment must not share a firefighter."""
        store = SlotStore.from_planning(nested([1, 2, 3]))
        a = PlanningView(store)["S"][1][1][0]
        b = PlanningView(store.copy())["S"][1][1][0]

        a["available"].append(7)
        b["available"].append(8)

        assert list(a["available"]) == [1, 2, 3, 7]
        assert list(b["available"]) == [1, 2, 3, 8]

    def test_a_copy_of_a_copy_is_independent(self):
        store = SlotStore.from_planning(nested([1, 2]))
        first = store.copy()
        second = first.copy()

        PlanningView(store)["S"][1][1][0]["available"].append(10)
        PlanningView(first)["S"][1][1][0]["available"].append(20)
        PlanningView(second)["S"][1][1][0]["available"].append(30)

        assert list(PlanningView(store)["S"][1][1][0]["available"]) == [1, 2, 10]
        assert list(PlanningView(first)["S"][1][1][0]["available"]) == [1, 2, 20]
        assert list(PlanningView(second)["S"][1][1][0]["available"]) == [1, 2, 30]

    def test_a_slot_is_converted_only_when_touched(self):
        """Converting all 334,056 up front cost a second of startup, and a run
        reaches only a few thousand."""
        planning = nested([1, 2])
        planning["S"][1][1][1] = {"planned": [9], "available": [9], "standby": []}
        store = SlotStore.from_planning(planning)

        assert store._owned == {}
        PlanningView(store)["S"][1][1][0]["available"].copy()
        assert list(store._owned) == [("S", 1, 1, 0)]

    def test_a_lazily_converted_slot_still_reflects_its_alias(self):
        store = SlotStore.from_planning(nested([1, 2, 3], alias=True))
        slot = PlanningView(store)["S"][1][1][0]   # first touch converts it
        slot["available"].remove(2)
        assert list(slot["planned"]) == [1, 3]

    def test_a_copy_can_reach_a_slot_neither_store_has_touched(self):
        store = SlotStore.from_planning(nested([1, 2]))
        clone = PlanningView(store.copy())["S"][1][1][0]
        assert list(clone["available"]) == [1, 2]

    def test_missing_slot_raises_keyerror(self):
        store = SlotStore.from_planning(nested([1, 2]))
        with pytest.raises(KeyError):
            PlanningView(store)["NOPE"][1][1][0]["available"].copy()

    def test_aliased_copies_are_independent(self):
        store = SlotStore.from_planning(nested([1, 2, 3], alias=True))
        a = PlanningView(store)["S"][1][1][0]
        b = PlanningView(store.copy())["S"][1][1][0]

        b["available"].remove(2)

        assert list(b["planned"]) == [1, 3]
        assert list(a["planned"]) == [1, 2, 3]
