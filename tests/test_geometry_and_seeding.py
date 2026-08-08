"""Tests for the distance helpers and the seeded downsampling.

`constrain_veh` is covered because the environment's reproducibility rests on it
and on the seeded `df_skills.sample` beside it — an unseeded draw there is what
made two identical runs disagree on 10 of 21 metrics before Étape 1.
"""

import random

import numpy as np
import pandas as pd
import pytest

from collective_functions import (
    constrain_veh,
    distance_euclidienne,
    trier_villes_par_distance,
)


class TestDistanceEuclidienne:
    def test_known_triangle(self):
        assert distance_euclidienne(0, 0, 3, 4) == pytest.approx(5.0)

    def test_zero_for_identical_points(self):
        assert distance_euclidienne(2.5, -1.5, 2.5, -1.5) == pytest.approx(0.0)

    def test_symmetric(self):
        a = distance_euclidienne(1, 2, 4, 6)
        b = distance_euclidienne(4, 6, 1, 2)
        assert a == pytest.approx(b)

    def test_never_negative(self):
        assert distance_euclidienne(5, 5, 1, 1) > 0

    def test_vectorised_over_arrays(self):
        """Called with columns in trier_villes_par_distance, so it must broadcast."""
        xs = np.array([3.0, 6.0])
        ys = np.array([4.0, 8.0])
        out = distance_euclidienne(0.0, 0.0, xs, ys)
        assert out.tolist() == pytest.approx([5.0, 10.0])


class TestTrierVillesParDistance:
    def stations(self):
        return pd.DataFrame({
            "Nom": ["BASE", "NEAR", "MID", "FAR"],
            "Coordonnée X": [0.0, 1000.0, 5000.0, 20000.0],
            "Coordonnée Y": [0.0, 0.0, 0.0, 0.0],
        })

    def test_orders_by_increasing_distance(self):
        out = trier_villes_par_distance(self.stations(), "BASE", ["FAR", "NEAR", "MID"])
        assert list(out.keys()) == ["NEAR", "MID", "FAR"]

    def test_only_returns_the_requested_candidates(self):
        out = trier_villes_par_distance(self.stations(), "BASE", ["FAR", "NEAR"])
        assert set(out) == {"FAR", "NEAR"}

    def test_distances_increase_with_the_ordering(self):
        out = trier_villes_par_distance(self.stations(), "BASE", ["FAR", "NEAR", "MID"])
        assert list(out.values()) == sorted(out.values())

    def test_empty_candidate_list(self):
        assert trier_villes_par_distance(self.stations(), "BASE", []) == {}


class TestConstrainVeh:
    """Downsamples a station's vehicles by a factor; the seed keeps runs comparable."""

    def vehicles(self):
        return [f"V{i}" for i in range(12)]

    def test_factor_divides_the_fleet(self):
        assert len(constrain_veh(self.vehicles(), factor=3, seed=42)) == 4
        assert len(constrain_veh(self.vehicles(), factor=2, seed=42)) == 6

    def test_factor_one_keeps_everything(self):
        assert len(constrain_veh(self.vehicles(), factor=1, seed=42)) == 12

    def test_same_seed_gives_the_same_selection(self):
        first = constrain_veh(self.vehicles(), factor=3, seed=42)
        second = constrain_veh(self.vehicles(), factor=3, seed=42)
        assert first == second

    def test_different_seeds_give_different_selections(self):
        by_seed = {
            tuple(constrain_veh(self.vehicles(), factor=3, seed=s)) for s in range(8)
        }
        assert len(by_seed) > 1

    def test_selection_is_a_subset_of_the_input(self):
        original = self.vehicles()
        kept = constrain_veh(list(original), factor=3, seed=42)
        assert set(kept) <= set(original)

    def test_it_shuffles_the_caller_s_list_in_place(self):
        """Documented quirk: the argument is reordered, not just read."""
        original = self.vehicles()
        passed = list(original)
        constrain_veh(passed, factor=3, seed=42)
        assert passed != original
        assert sorted(passed) == sorted(original)

    def test_it_reseeds_the_global_rng(self):
        """Also a quirk: `random.seed` here perturbs any later `random` draw.

        Pinned so the side effect is visible; removing it would change results.
        """
        random.seed(123)
        baseline = [random.random() for _ in range(3)]

        random.seed(123)
        constrain_veh(self.vehicles(), factor=3, seed=999)
        after = [random.random() for _ in range(3)]

        assert after != baseline
