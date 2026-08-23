"""Tests for the set-pointer action head.

The flatten path concatenates the attention embeddings in slot order and reads
them with `Linear(layer_size, action_size)`, so "the candidate in position 3"
is representable -- when the positions come from whatever order `_resolve_crew`
happened to build `ff_existing` in. These pin the two properties the pointer is
for: permutation equivariance by construction, and a candidate count that is
not fixed at build time.
"""

import pytest
import torch

from collective_functions import N_HEADER_ROWS, state_size_for, state_width
from networks import QVN, PointerHead

ACTION_SIZE = 80


def qvn(pointer, action_size=ACTION_SIZE, layer_size=128):
    return QVN(
        state_size=state_size_for(action_size), action_size=action_size,
        layer_size=layer_size, AM=True, n_steps=1, device="cpu", seed=0,
        N=8, num_layers=2, layer_type="ff", use_batchnorm=False,
        pointer=pointer,
    ).eval()


def state(batch, n_candidates):
    return torch.randn(batch, n_candidates + N_HEADER_ROWS, state_width())


def taus(batch, num_tau=8):
    return torch.rand(batch, num_tau).sort(dim=1).values


class TestPointerHead:
    def test_scores_one_per_candidate(self):
        head = PointerHead(d_model=64, context_size=32)
        scores = head(torch.randn(3, 7, 64), torch.randn(3, 32))
        assert scores.shape == (3, 7)

    def test_permutation_equivariant(self):
        head = PointerHead(d_model=64, context_size=32).eval()
        entities, context = torch.randn(2, 9, 64), torch.randn(2, 32)
        perm = torch.randperm(9)

        with torch.no_grad():
            scores = head(entities, context)
            permuted = head(entities[:, perm, :], context)

        assert torch.allclose(scores[:, perm], permuted, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 5, 80, 200])
    def test_same_parameters_serve_any_candidate_count(self, n):
        head = PointerHead(d_model=64, context_size=32).eval()
        with torch.no_grad():
            assert head(torch.randn(1, n, 64), torch.randn(1, 32)).shape == (1, n)

    def test_parameter_count_is_independent_of_candidates(self):
        """A slot-indexed head grows with action_size; this one does not."""
        head = PointerHead(d_model=64, context_size=32)
        before = sum(p.numel() for p in head.parameters())
        head(torch.randn(1, 500, 64), torch.randn(1, 32))
        assert sum(p.numel() for p in head.parameters()) == before


class TestQVNPointer:
    def test_forward_returns_context_and_entities(self):
        net = qvn(pointer=True)
        with torch.no_grad():
            context, entities = net.forward(state(4, ACTION_SIZE))
        assert context.shape[0] == 4
        assert entities.shape[:2] == (4, ACTION_SIZE)

    def test_quantiles_have_one_column_per_candidate(self):
        net = qvn(pointer=True)
        st, tau = state(4, ACTION_SIZE), taus(4)
        with torch.no_grad():
            context, entities = net.forward(st)
            q = net.get_quantiles(st, tau, embedding=context, entities=entities)
        assert q.shape == (4, 8, ACTION_SIZE)

    def test_end_to_end_permutation_equivariance(self):
        """Reordering the candidates must reorder the Q-values, nothing else."""
        net = qvn(pointer=True)
        st, tau = state(2, ACTION_SIZE), taus(2)
        perm = torch.randperm(ACTION_SIZE)
        shuffled = st.clone()
        shuffled[:, N_HEADER_ROWS:, :] = st[:, N_HEADER_ROWS:, :][:, perm, :]

        with torch.no_grad():
            c1, e1 = net.forward(st)
            q1 = net.get_quantiles(st, tau, embedding=c1, entities=e1)
            c2, e2 = net.forward(shuffled)
            q2 = net.get_quantiles(shuffled, tau, embedding=c2, entities=e2)

        assert torch.allclose(q1[:, :, perm], q2, atol=1e-5)

    @pytest.mark.parametrize("n", [20, 80, 150])
    def test_runs_a_roster_size_it_was_not_built_for(self, n):
        """Built for 80; the same weights take any count. The flatten path
        raises a shape error -- see the test below."""
        net = qvn(pointer=True)
        st, tau = state(2, n), taus(2)
        with torch.no_grad():
            context, entities = net.forward(st)
            q = net.get_quantiles(st, tau, embedding=context, entities=entities)
        assert q.shape == (2, 8, n)

    def test_missing_entities_is_an_error(self):
        """Silently falling back to the slot-indexed head would train the
        thing the pointer exists to avoid."""
        net = qvn(pointer=True)
        st, tau = state(2, ACTION_SIZE), taus(2)
        with torch.no_grad():
            context, _ = net.forward(st)
            with pytest.raises(ValueError, match="entities"):
                net.get_quantiles(st, tau, embedding=context)

    def test_pointer_requires_attention(self):
        with pytest.raises(ValueError, match="AM"):
            QVN(state_size=state_size_for(ACTION_SIZE), action_size=ACTION_SIZE,
                layer_size=128, AM=False, n_steps=1, device="cpu", seed=0, N=8,
                num_layers=2, layer_type="ff", use_batchnorm=False, pointer=True)


class TestFlattenPathUnchanged:
    def test_default_is_the_flatten_path(self):
        assert qvn(pointer=False).pointer is False

    def test_flatten_path_still_produces_action_size_quantiles(self):
        net = qvn(pointer=False)
        st, tau = state(4, ACTION_SIZE), taus(4)
        with torch.no_grad():
            q = net.get_quantiles(st, tau, embedding=net.forward(st))
        assert q.shape == (4, 8, ACTION_SIZE)

    def test_flatten_path_is_fixed_to_its_roster_size(self):
        net = qvn(pointer=False)
        with torch.no_grad(), pytest.raises(RuntimeError):
            net.forward(state(2, 20))
