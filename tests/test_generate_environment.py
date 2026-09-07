"""Tests for the environment generator's pure helpers and its target selection.

Two things here are worth pinning.

`--only` decides what gets written, and every group except `fake` is an input to
the golden cases. Adding a synthetic stream used to overwrite the references it
would later be compared against, so the selection logic — and in particular the
fact that `--only fake --prob_dep` still *builds* the real stream (it carries
`prob_dict`) without writing it — is the part that must not regress.

The rest covers the helpers the refactor extracted from `main()`: `normalise`
and `locate` were duplicated between the real and the synthetic path, and a
divergence between the two copies would have shifted one stream's features
relative to the other's without failing anything.

No test reads the data tree: `Sources` is assembled from small frames, in
keeping with conftest pointing `RL_DATA_ROOT` at a temp dir.
"""

import numpy as np
import pandas as pd
import pytest

import generate_environment as ge


class TestIsLeapYear:
    """Drives whether a synthetic year gets its 366th day."""

    @pytest.mark.parametrize("year", [2016, 2020, 2024, 2000, 1600])
    def test_leap_years(self, year):
        assert ge.is_leap_year(year)

    @pytest.mark.parametrize("year", [2018, 2019, 2021, 1900, 2100])
    def test_common_years(self, year):
        """1900 and 2100 are the century rule: divisible by 4 but not leap."""
        assert not ge.is_leap_year(year)


class TestNormalise:
    def test_maps_to_unit_interval(self):
        df = pd.DataFrame({c: [1.0, 3.0, 5.0] for c in ge.COLS_TO_NORM})
        out = ge.normalise(df)
        for col in ge.COLS_TO_NORM:
            assert out[col].tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_constant_column_becomes_zero(self):
        """A constant column would divide by zero; it is pinned to 0 instead."""
        df = pd.DataFrame({c: [7.0, 7.0, 7.0] for c in ge.COLS_TO_NORM})
        out = ge.normalise(df)
        for col in ge.COLS_TO_NORM:
            assert (out[col] == 0.0).all()
        assert not out[ge.COLS_TO_NORM].isna().any().any()

    def test_negative_values_are_shifted_not_clipped(self):
        df = pd.DataFrame({c: [-10.0, 0.0, 10.0] for c in ge.COLS_TO_NORM})
        out = ge.normalise(df)
        assert out["Coord X"].tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_leaves_other_columns_alone(self):
        df = pd.DataFrame({c: [1.0, 2.0] for c in ge.COLS_TO_NORM})
        df["Duration"] = [30.0, 90.0]
        out = ge.normalise(df)
        assert out["Duration"].tolist() == [30.0, 90.0]

    def test_covers_exactly_the_documented_columns(self):
        """The stream carries other numeric columns that must stay untouched."""
        assert set(ge.COLS_TO_NORM) == {
            "Coord X", "Coord Y", "Month_sin", "Month_cos",
            "Day_sin", "Day_cos", "Hour_sin", "Hour_cos",
        }


class TestAddLeapDay:
    """Day 366 is replayed from a neighbouring day, not drawn afresh."""

    @staticmethod
    def _year(days):
        return pd.DataFrame({"Day": days, "Duration": [10.0] * len(days)})

    def test_common_year_is_returned_unchanged(self):
        df = self._year([1, 200, 365])
        out = ge.add_leap_day(df, 2018)
        assert out.equals(df)

    def test_leap_year_gains_day_366(self):
        df = self._year([1, 2, 340, 360, 365])
        out = ge.add_leap_day(df, 2020)
        assert (out["Day"] == 366).sum() >= 1
        assert len(out) > len(df)

    @pytest.mark.parametrize("year", [2020, 2024, 2028, 2032, 2016, 2000])
    def test_donor_comes_from_the_turn_of_the_year(self, year):
        """A summer donor would put a June profile on 31 December.

        Swept over several leap years because the donor is drawn with a
        year-dependent seed: a single year could pick an edge day by luck even
        if the window were wrong. Summer dominates the frame 20 to 4, so an
        unrestricted draw lands outside the window far more often than not.
        """
        summer = list(range(150, 170))
        df = pd.DataFrame({
            "Day": [1, 10, 340, 365] + summer,
            "Duration": [10.0] * (4 + len(summer)),
            "tag": ["jan", "jan", "dec", "dec"] + ["summer"] * len(summer),
        })
        out = ge.add_leap_day(df, year)
        assert set(out.loc[out["Day"] == 366, "tag"]) <= {"jan", "dec"}

    def test_falls_back_to_the_whole_year_when_no_edge_day(self):
        """Neither December nor early January present: any day beats none."""
        df = self._year([150, 160, 170])
        out = ge.add_leap_day(df, 2020)
        assert (out["Day"] == 366).sum() >= 1

    def test_is_deterministic_for_a_given_year(self):
        df = self._year([1, 10, 340, 365])
        a = ge.add_leap_day(df.copy(), 2020)
        b = ge.add_leap_day(df.copy(), 2020)
        pd.testing.assert_frame_equal(a, b)


class TestParseArgs:
    """`--only` gates every write; the default must stay all-inclusive."""

    def test_default_selects_every_target(self):
        args = ge.parse_args(["--prob_dep"])
        assert args.targets == set(ge.ALL_TARGETS)

    def test_default_allows_omitting_sample_list(self):
        """Historically a bare run built tables, real and planning."""
        args = ge.parse_args([])
        assert args.sample_list is None
        assert "fake" in args.targets

    def test_only_fake_selects_just_that(self):
        args = ge.parse_args(
            ["--only", "fake", "--sample_list", "y.pkl", "--save_as", "o.pkl"]
        )
        assert args.targets == {"fake"}

    def test_only_accepts_several_targets(self):
        args = ge.parse_args(["--only", "tables", "planning"])
        assert args.targets == {"tables", "planning"}

    def test_only_fake_requires_a_sample_list(self):
        with pytest.raises(SystemExit):
            ge.parse_args(["--only", "fake", "--save_as", "o.pkl"])

    def test_only_fake_requires_an_output_name(self):
        with pytest.raises(SystemExit):
            ge.parse_args(["--only", "fake", "--sample_list", "y.pkl"])

    def test_unknown_target_is_rejected(self):
        with pytest.raises(SystemExit):
            ge.parse_args(["--only", "everything"])

    def test_start_year_defaults_to_the_current_year(self):
        from datetime import datetime
        assert ge.parse_args([]).start_year == datetime.now().year


class TestRealStreamIsBuiltForProbDict:
    """`--only fake --prob_dep` must build the real stream without writing it.

    The synthetic stream draws its departures from a distribution derived from
    the real one, so skipping the real build would break `--prob_dep`; writing
    it would clobber the golden inputs. Both halves matter.
    """

    @staticmethod
    def _needs_real(argv):
        args = ge.parse_args(argv)
        return ge.real_stream_plan(args.targets, args.prob_dep)

    def test_prob_dep_fake_builds_real_but_does_not_write_it(self):
        build, write = self._needs_real(
            ["--prob_dep", "--only", "fake",
             "--sample_list", "y.pkl", "--save_as", "o.pkl"]
        )
        assert build and not write

    def test_deterministic_fake_needs_no_real_stream(self):
        """Without --prob_dep the departures come from the response table."""
        build, write = self._needs_real(
            ["--only", "fake", "--sample_list", "y.pkl", "--save_as", "o.pkl"]
        )
        assert not build and not write

    def test_real_target_writes_it(self):
        build, write = self._needs_real(["--only", "real"])
        assert build and write

    def test_tables_only_touches_neither(self):
        build, write = self._needs_real(["--only", "tables"])
        assert not build and not write


class TestBuildRealStopsAtProbDict:
    """`write=False` must stop as soon as `prob_dict` exists.

    `prob_departure` draws from the unseeded global `random`, so every extra
    draw shifts the synthetic stream's own draws. Running the real stream to
    completion when it is only wanted for `prob_dict` changed 17 255 of 127 392
    departures -- a silent divergence no unit test would have caught.
    """

    def test_stops_before_drawing_real_departures(self, monkeypatch):
        drawn = []
        monkeypatch.setattr(ge.pd, "read_pickle",
                            lambda *a, **k: pd.DataFrame({"Day": [1]}))
        monkeypatch.setattr(ge, "locate", lambda df, src: df)
        monkeypatch.setattr(ge, "precompute_prob_dict", lambda df: {"p": 1})
        monkeypatch.setattr(ge, "precompute_prob_departure",
                            lambda df, pd_, seed=42: drawn.append("drew") or df)

        src = ge.Sources(**{f: None for f in ge.Sources.__dataclass_fields__})
        df, suffix, prob_dict = ge.build_real(src, True, 2018, write=False)

        assert prob_dict == {"p": 1}
        assert df is None
        assert suffix == "_prob"
        assert drawn == [], "the real stream must not consume the RNG here"

    def test_still_draws_when_the_stream_is_kept(self, monkeypatch):
        drawn = []
        monkeypatch.setattr(ge.pd, "read_pickle",
                            lambda *a, **k: pd.DataFrame({"Day": [1]}))
        monkeypatch.setattr(ge, "locate", lambda df, src: df)
        monkeypatch.setattr(ge, "precompute_prob_dict", lambda df: {"p": 1})
        monkeypatch.setattr(ge, "precompute_prob_departure",
                            lambda df, pd_, seed=42: drawn.append("drew") or df)
        monkeypatch.setattr(ge, "precompute_date", lambda df, y: df)
        monkeypatch.setattr(ge, "precompute_returns", lambda df, a, b, c: df)
        monkeypatch.setattr(ge, "normalise", lambda df: df)

        src = ge.Sources(**{f: None for f in ge.Sources.__dataclass_fields__})
        df, _suffix, _prob = ge.build_real(src, True, 2018, write=True)

        assert drawn == ["drew"]
        assert df is not None


class TestProbDepartureSeeding:
    """The departure draw is seeded per call, on its own generator.

    Sharing the global `random` made the output depend on how much sampling had
    already happened in the process: building the real stream first shifted the
    synthetic stream's draws and changed 17 255 of its 127 392 departures.
    """

    PROB_DICT = {
        "fire": {"urban": {("VSAV",): 0.5, ("FPT",): 0.5}},
        "rescue": {"*": {("VSAV",): 0.5, ("EPA",): 0.5}},
    }

    @staticmethod
    def _frame(n=200):
        return pd.DataFrame({
            "area_type": ["urban"] * n,
            "incident_name": ["fire"] * n,
        })

    def test_same_seed_gives_the_same_draw(self):
        a = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=7)
        b = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=7)
        assert a["departure"].tolist() == b["departure"].tolist()

    def test_different_seeds_give_different_draws(self):
        a = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=1)
        b = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=2)
        assert a["departure"].tolist() != b["departure"].tolist()

    def test_draw_is_immune_to_prior_global_rng_use(self):
        """The regression: earlier sampling must not shift this stream."""
        import random as _random

        _random.seed(0)
        a = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=7)

        _random.seed(0)
        for _ in range(53088):          # as if a real stream had been drawn
            _random.random()
        b = ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=7)

        assert a["departure"].tolist() == b["departure"].tolist()

    def test_does_not_disturb_the_global_generator(self):
        """A data-prep step should leave the caller's RNG where it found it."""
        import random as _random

        _random.seed(123)
        expected = [_random.random() for _ in range(3)]

        _random.seed(123)
        ge.precompute_prob_departure(self._frame(), self.PROB_DICT, seed=7)
        after = [_random.random() for _ in range(3)]

        assert after == expected

    def test_falls_back_to_the_star_key(self):
        df = pd.DataFrame({"area_type": ["rural"], "incident_name": ["rescue"]})
        out = ge.precompute_prob_departure(df, self.PROB_DICT, seed=7)
        assert out["departure"].iloc[0] in ({1: ["VSAV"]}, {1: ["EPA"]})

    def test_unknown_incident_and_area_yields_no_departure(self):
        df = pd.DataFrame({"area_type": ["rural"], "incident_name": ["fire"]})
        out = ge.precompute_prob_departure(df, self.PROB_DICT, seed=7)
        assert out["departure"].iloc[0] == {}

    def test_years_of_a_decade_draw_differently(self, monkeypatch):
        """Ten years sharing one seed would be one year repeated ten times."""
        seeds = []
        monkeypatch.setattr(ge, "locate", lambda df, src: df)
        monkeypatch.setattr(ge, "add_leap_day", lambda df, y: df)
        monkeypatch.setattr(ge, "precompute_date", lambda df, y: df)
        monkeypatch.setattr(ge, "precompute_returns", lambda df, a, b, c: df)
        monkeypatch.setattr(ge, "reorg_dates", lambda df: df)
        monkeypatch.setattr(ge, "normalise", lambda df: df)
        monkeypatch.setattr(ge.pd, "read_pickle",
                            lambda *a, **k: pd.DataFrame({"Day": [1]}))
        monkeypatch.setattr(
            ge, "precompute_prob_departure",
            lambda df, pdict, seed=42: seeds.append(seed) or df)

        src = ge.Sources(**{f: None for f in ge.Sources.__dataclass_fields__})
        ge.build_fake(src, ["y1.pkl", "y2.pkl", "y3.pkl"], True, 2018, {}, seed=42)

        assert seeds == [42 + 2018, 42 + 2019, 42 + 2020]
        assert len(set(seeds)) == 3


class TestLocate:
    """The four positioning steps, shared by both streams since the refactor."""

    def test_applies_the_steps_in_order(self, monkeypatch):
        calls = []

        def spy(name, result_col):
            def _fn(*args, **kwargs):
                calls.append(name)
                df = next(a for a in args if isinstance(a, pd.DataFrame))
                df = df.copy()
                df[result_col] = name
                return df
            return _fn

        monkeypatch.setattr(ge, "precompute_pdd", spy("pdd", "PDD"))
        monkeypatch.setattr(ge, "precompute_zone", spy("zone", "zone"))
        monkeypatch.setattr(ge, "precompute_incident", spy("inc", "incident_name"))
        monkeypatch.setattr(ge, "precompute_area_type", spy("area", "area_type"))

        src = ge.Sources(**{f: None for f in ge.Sources.__dataclass_fields__})
        out = ge.locate(pd.DataFrame({"Day": [1]}), src)

        assert calls == ["pdd", "zone", "inc", "area"]
        # area_type depends on the zone and incident set before it.
        assert set(out.columns) >= {"PDD", "zone", "incident_name", "area_type"}

    def test_passes_area_type_its_tables_in_order(self, monkeypatch):
        """precompute_area_type takes five tables positionally before the frame;
        swapping two of them silently mislabels every sector."""
        seen = {}

        def fake_area_type(df_xy, df_lieu, df_nom_commune, df_commune, df_secteur, df):
            seen.update(xy=df_xy, lieu=df_lieu, nom=df_nom_commune,
                        commune=df_commune, secteur=df_secteur)
            return df

        monkeypatch.setattr(ge, "precompute_pdd", lambda *a, **k: a[1])
        monkeypatch.setattr(ge, "precompute_zone", lambda *a, **k: a[1])
        monkeypatch.setattr(ge, "precompute_incident", lambda *a, **k: a[1])
        monkeypatch.setattr(ge, "precompute_area_type", fake_area_type)

        fields = {f: None for f in ge.Sources.__dataclass_fields__}
        marks = {k: pd.DataFrame({k: [1]}) for k in
                 ("df_xy", "df_lieu", "df_nom_commune", "df_commune", "df_secteur")}
        fields.update(marks)
        ge.locate(pd.DataFrame({"Day": [1]}), ge.Sources(**fields))

        assert seen["xy"] is marks["df_xy"]
        assert seen["lieu"] is marks["df_lieu"]
        assert seen["nom"] is marks["df_nom_commune"]
        assert seen["commune"] is marks["df_commune"]
        assert seen["secteur"] is marks["df_secteur"]


class TestZones:
    def test_zones_are_disjoint(self):
        """A station in two zones would make get_zone order-dependent."""
        assert not (set(ge.Z_1) & set(ge.Z_2))
        assert not (set(ge.Z_1) & set(ge.Z_3))
        assert not (set(ge.Z_2) & set(ge.Z_3))

    def test_z1_is_the_pair_the_constraint_factor_targets(self):
        assert set(ge.Z_1) == {"TOULOUSE - LOUGNON", "TOULOUSE - VION"}


class TestPrecomputeReturns:
    """Every departure gets a matching RETURN, and returns sort first on ties."""

    @staticmethod
    def _frame(n=3):
        base = pd.Timestamp("2018-01-01 08:00")
        df = pd.DataFrame({
            "date": [base + pd.Timedelta(hours=i) for i in range(n)],
            "Duration": [30.0] * n,
            "PDD": [["A"] for _ in range(n)],
            "departure": [{1: "VSAV"} for _ in range(n)],
            "zone": ["Z_1"] * n,
            "Month": [1] * n, "Day": [1] * n, "Hour": [8] * n, "Minute": [0] * n,
        })
        for col in ge.COLS_TO_NORM:
            df[col] = 0.5
        return df

    def test_output_holds_two_rows_per_intervention(self):
        out = ge.precompute_returns(self._frame(3), 1, 3, True)
        assert len(out) == 6

    def test_num_inter_spans_the_requested_range(self):
        out = ge.precompute_returns(self._frame(3), 10, 12, True)
        assert sorted(out["num_inter"].unique()) == [10, 11, 12]

    def test_every_intervention_has_one_return(self):
        out = ge.precompute_returns(self._frame(4), 1, 4, True)
        returns = out[out["departure"].apply(lambda d: d == {0: "RETURN"})]
        assert len(returns) == 4
        assert sorted(returns["num_inter"]) == [1, 2, 3, 4]

    def test_stream_is_sorted_by_date(self):
        out = ge.precompute_returns(self._frame(5), 1, 5, True)
        assert out["date"].is_monotonic_increasing

    def test_return_carries_no_zone_and_no_duration(self):
        out = ge.precompute_returns(self._frame(2), 1, 2, True)
        returns = out[out["departure"].apply(lambda d: d == {0: "RETURN"})]
        assert (returns["zone"] == "").all()
        assert (returns["Duration"] == 0).all()

    def test_return_precedes_a_departure_at_the_same_instant(self):
        """A vehicle returning at t must be free for a departure at t."""
        df = self._frame(1)
        df.loc[0, "Duration"] = 0.0  # returns the instant it leaves
        out = ge.precompute_returns(df, 1, 1, True)
        first = out.iloc[0]["departure"]
        assert first == {0: "RETURN"}
