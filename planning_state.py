"""A copyable `planning`, without changing what a slot's lists mean.

`planning` is the simulation's largest mutable structure: 334,056 hour-slots,
each a dict of three Python lists (`planned`, `available`, `standby`). Copying
it with `deepcopy` takes 2.6 s, which is what makes running several
environments at once impractical -- an actor cannot cheaply fork the world it
is in.

Two measurements shape what replaces it.

**Order is load-bearing.** An action index is a *position* in the candidate
list that `_resolve_crew` builds out of these slots, so any structure that
returns members in a different order silently changes which firefighter an
action refers to. A bitmask over each slot's roster looks like the obvious
encoding -- membership is a subset of `planned`, average length 10 -- but it
reorders: a list re-appended after a removal goes to the *end*, while a mask
puts it back in roster position. That happens in 1996 of 5521 appends in a
reference run, and every golden metric moved. The lists are kept as lists.

**Most slots are never touched**: they are hours no event in the window falls
in. So neither building nor copying should walk all 334,056 -- converting them
eagerly cost a full second of startup, which on a short run swamped the saving.

`SlotStore` therefore flattens the four-level nesting to one dict keyed
`(station, month, day, hour)` and fills it lazily: a slot is converted out of
the nested dict on first touch, and a copy shares its base's slots until either
side writes to one. Building is O(1), copying is O(touched).

There is one wrinkle in the generated data that this has to reproduce rather
than fix: `generate_environment` writes 308,811 of the slots as
`{"planned": mat_filtered, "available": mat_filtered, ...}` -- one list under
two keys -- so removing a firefighter from `available` also removes them from
`planned`. Storing the two independently changes which candidates
`_resolve_crew` sees and moved every golden metric.

`PlanningView` keeps the old
`planning[station][month][day][hour]["available"]` syntax working, with
`.append`, `.remove`, `.copy` and `len`, so the call sites in
`collective_functions` and `simulator` do not have to change at once.
"""

from typing import Dict, Iterator, List, Tuple

SlotKey = Tuple[str, int, int, int]

# Index into a slot's `[available, standby]` pair.
AVAILABLE, STANDBY = 0, 1
_KINDS = {"available": AVAILABLE, "standby": STANDBY}


class SlotStore:
    """Flattened planning slots with copy-on-write membership.

    Attributes
    ----------
    rosters : dict[SlotKey, list[int]]
        Each slot's `planned`, shared by every copy. Never mutated: the
        simulation only reads it, and `_SlotView` refuses to assign it.
    """

    __slots__ = ("rosters", "_owned", "_base", "_aliased", "_source")

    def __init__(self, rosters, owned=None, base=None, aliased=None,
                 source=None):
        self.rosters = rosters
        # Slots this instance has written to: key -> [available, standby].
        self._owned: Dict[SlotKey, List[List[int]]] = {} if owned is None else owned
        # The store this one was copied from, read through for untouched slots.
        self._base = base
        # Slots whose `planned` and `available` are one list; see
        # `from_planning`. Copying has to recreate the alias, not two lists.
        self._aliased = set() if aliased is None else aliased
        # The nested dict a root store converts from, lazily. None on copies,
        # which read through `_base` instead.
        self._source = source

    def _materialise(self, key: SlotKey) -> List[List[int]]:
        """Convert one slot out of the source dict, on first touch."""
        station, month, day, hour = key
        try:
            slot = self._source[station][month][day][hour]
        except (KeyError, TypeError):
            raise KeyError(key) from None

        planned = slot["planned"]
        raw_available = slot["available"]
        available = list(raw_available)
        if planned is raw_available:
            # One list under both names; see `from_planning`.
            self._aliased.add(key)
            self.rosters[key] = available
        else:
            self.rosters[key] = list(planned)

        pair = [available, list(slot["standby"])]
        self._owned[key] = pair
        return pair

    @classmethod
    def from_planning(cls, planning: dict) -> "SlotStore":
        """Build from the nested-dict planning, preserving every slot's order.

        Slots where `planned` and `available` are the *same list object* keep
        that link. `generate_environment` builds 308,811 of the 334,056 slots
        as `{"planned": mat_filtered, "available": mat_filtered, ...}` -- one
        list under two keys -- so removing a firefighter from `available` also
        removes them from `planned`. Storing the two independently changes the
        simulation: `_resolve_crew` draws its candidates from `planned`, and
        with the alias that pool shrinks as the shift is spent.

        The aliasing also makes the guard in `reinforcement_arriving` and
        `returning` -- `f in planned and f not in available` -- unsatisfiable
        in those slots, so that branch is dead for 92% of the planning. Both
        are properties of the generated data, which this reproduces rather
        than corrects.
        """
        # Slots are converted on first touch, not up front. Copying every one
        # of the 334,056 eagerly cost a full second of startup, and a run only
        # ever reaches a few thousand of them -- the rest are hours no event in
        # the window falls in.
        return cls({}, {}, source=planning)

    def copy(self) -> "SlotStore":
        """An independent copy, sharing slots until either side writes.

        The new store takes over this one's owned slots and both are given a
        shared, frozen base, so a later write on *either* side materialises its
        own copy of that slot. Handing the copy a read-through reference to a
        still-mutable base would be cheaper by one dict move, but then editing
        the base after copying would leak into the copy -- copy-on-write in one
        direction only, which is the kind of thing that shows up much later as
        two environments quietly sharing a firefighter.
        """
        shared = SlotStore(self.rosters, self._owned, self._base,
                           self._aliased, self._source)
        self._owned = {}
        self._base = shared
        self._source = None          # reached through `shared` from now on
        return SlotStore(dict(self.rosters), {}, shared, self._aliased)

    # --- membership -----------------------------------------------------

    def _read(self, key: SlotKey) -> List[List[int]]:
        """The slot's `[available, standby]`, without claiming ownership."""
        store = self
        while store is not None:
            pair = store._owned.get(key)
            if pair is not None:
                return pair
            if store._source is not None:
                return store._materialise(key)
            store = store._base
        raise KeyError(key)

    def _write(self, key: SlotKey) -> List[List[int]]:
        """The slot's `[available, standby]`, copied into this store first.

        For an aliased slot the fresh `available` also becomes this store's
        `planned`, so the two stay one list here as they were in the pickle.
        """
        pair = self._owned.get(key)
        if pair is None:
            source = self._read(key)
            available = list(source[0])
            pair = [available, list(source[1])]
            self._owned[key] = pair
            if key in self._aliased:
                self.rosters[key] = available
        return pair

    def members(self, kind: int, key: SlotKey) -> List[int]:
        return self._read(key)[kind]

    def mutable(self, kind: int, key: SlotKey) -> List[int]:
        return self._write(key)[kind]

    def roster(self, key: SlotKey) -> List[int]:
        """The slot's `planned` -- the same object as `available` when aliased."""
        store = self
        while store is not None:
            if key in store._aliased:
                # Read through the copy-on-write chain so a copied-but-unwritten
                # slot sees its base's list rather than a stale roster entry.
                return self._read(key)[AVAILABLE]
            roster = store.rosters.get(key)
            if roster is not None:
                return roster
            if store._source is not None:
                self._read(key)          # materialises, filling `rosters`
                return store.roster(key)
            store = store._base
        raise KeyError(key)




class _Roster(tuple):
    """A slot's `planned`: everything the readers do, minus the mutators.

    `.copy()` returns a real list, because that is what the callers go on to
    mutate.
    """

    __slots__ = ()

    def copy(self) -> List[int]:
        return list(self)


class _SlotView:
    """One slot, presenting the original `{"planned", "available", "standby"}`.

    Reads hand back the underlying list, so `len`, `in`, iteration, `.copy()`
    and indexing behave exactly as they did. Writes go through
    `SlotStore.mutable`, which materialises the slot first -- that is what
    makes the copy lazy without the call sites knowing.
    """

    __slots__ = ("_store", "_key", "_available", "_standby")

    def __init__(self, store: SlotStore, key: SlotKey):
        self._store = store
        self._key = key
        # Built once for the same reason `PlanningView` caches its children:
        # a `_ListView` holds nothing but the store and the key, so handing
        # back the same one avoids an allocation on every read.
        self._available = _ListView(store, AVAILABLE, key)
        self._standby = _ListView(store, STANDBY, key)

    def __getitem__(self, kind: str):
        if kind == "available":
            return self._available
        if kind == "standby":
            return self._standby
        if kind == "planned":
            # A tuple, not the stored list. The nested dicts handed out their
            # own list here and each planning owned it, so a caller that
            # mutated it in place damaged only that run; the rosters here are
            # shared by every copy, so the same mutation would corrupt all of
            # them at once. `_resolve_crew` reads this and passes it to
            # `adding_lent_ff`, which does `ff_mats += ff_lent` in place -- it
            # calls `.copy()` first, but returning something unmutable means a
            # caller that forgets fails loudly rather than silently rewriting
            # the roster for the rest of the run.
            #
            # Tuples support `len`, `in`, iteration, indexing and `+`, which is
            # everything the readers do. `.copy()` is the one list method they
            # use, so `_Roster` adds it back.
            return _Roster(self._store.roster(self._key))
        raise KeyError(kind)

    def __setitem__(self, kind: str, value) -> None:
        # The one assignment the simulation makes is `slot["standby"] = []`.
        if kind == "planned":
            raise TypeError("`planned` is fixed once the environment is loaded")
        self._store.mutable(_KINDS[kind], self._key)[:] = value

    def keys(self):
        return ("planned", "available", "standby")

    def __iter__(self):
        return iter(self.keys())


class _ListView:
    """A slot's `available` or `standby`, with list semantics preserved.

    Every read delegates to the stored list and every write materialises the
    slot first. Duplicates and position are therefore whatever a plain list
    would have given -- which matters, because re-appending a removed member
    puts it at the end, and 36% of appends in a reference run do exactly that.
    """

    __slots__ = ("_store", "_kind", "_key")

    def __init__(self, store: SlotStore, kind: int, key: SlotKey):
        self._store = store
        self._kind = kind
        self._key = key

    def _read(self) -> List[int]:
        return self._store.members(self._kind, self._key)

    def _write(self) -> List[int]:
        return self._store.mutable(self._kind, self._key)

    def __len__(self) -> int:
        return len(self._read())

    def __iter__(self) -> Iterator[int]:
        return iter(self._read())

    def __contains__(self, matricule) -> bool:
        return matricule in self._read()

    def __getitem__(self, i):
        return self._read()[i]

    def __eq__(self, other) -> bool:
        return list(self._read()) == list(other)

    def __repr__(self) -> str:
        return repr(self._read())

    def copy(self) -> List[int]:
        return list(self._read())

    def append(self, matricule) -> None:
        self._write().append(matricule)

    def remove(self, matricule) -> None:
        self._write().remove(matricule)

    def extend(self, matricules) -> None:
        self._write().extend(matricules)


class PlanningView:
    """`planning[station][month][day][hour]` over a `SlotStore`.

    Exists so the flattened store can be introduced without rewriting the 31
    call sites that index the nested dict. Each `__getitem__` narrows the key
    by one level; the fourth returns a `_SlotView`.
    """

    __slots__ = ("_store", "_prefix", "_children")

    def __init__(self, store: SlotStore, prefix: tuple = ()):
        self._store = store
        self._prefix = prefix
        # The narrowing levels and slot views are immutable wrappers over a
        # key, so they can be built once and handed out again. Without this,
        # `planning[station][month][day][hour]["available"]` allocated three
        # intermediate views plus a list view on every access -- 0.92 us
        # against the nested dict's 0.10 -- and the simulation does it in its
        # innermost loop. Caching brings the two within noise; the dict of
        # children is bounded by the slots a run actually visits.
        self._children = {}

    def __getitem__(self, key):
        cached = self._children.get(key)
        if cached is not None:
            return cached

        prefix = self._prefix + (key,)
        child = (_SlotView(self._store, prefix) if len(prefix) == 4
                 else PlanningView(self._store, prefix))
        self._children[key] = child
        return child

    def __contains__(self, key) -> bool:
        prefix = self._prefix + (key,)
        if len(prefix) == 4:
            return prefix in self._store.rosters
        return any(k[:len(prefix)] == prefix for k in self._store.rosters)

    @property
    def store(self) -> SlotStore:
        return self._store

    def copy(self) -> "PlanningView":
        """A view onto a copy-on-write clone of the slots."""
        return PlanningView(self._store.copy())
