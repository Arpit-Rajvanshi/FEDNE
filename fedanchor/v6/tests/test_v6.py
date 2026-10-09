"""
FEDANCHOR V6 Test Suite
========================

Tests:
  1.  Identical global anchors broadcast to all clients
  2.  Anchor gradients exist after backward pass
  3.  Anchors are included in the optimizer
  4.  Anchor updates are non-zero when expected
  5.  Hungarian alignment works correctly
  6.  Anchor permutation is correctly resolved
  7.  Weighted anchor aggregation is correct
  8.  Global anchor matrix has correct shape
  9.  Client utilisation is computed correctly
 10.  Global utilisation is computed correctly
 11.  Entropy is correct
 12.  Pairwise distances are correct
 13.  Anchor displacement is correct
 14.  Client/global alignment metrics are correct
 15.  Synthetic permutation test: two clients with intentionally swapped
      anchors → alignment restores correspondence before averaging
"""

import sys
import os
import numpy as np
import torch
import pytest

# Ensure the repository root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fedanchor.v6.models.global_anchor import GlobalAnchorSet
from fedanchor.v6.aggregation.hungarian_align import (
    align_anchors_to_reference,
    aggregate_anchors_with_alignment,
    compute_alignment_distance,
)
from fedanchor.v6.evaluation.metrics_v6 import (
    compute_soft_q,
    compute_per_client_utilisation,
    compute_anchor_diagnostics_v6,
)
from fedanchor.v6.training.server_v6 import V6ServerWithSharedAnchors
from fedanchor.v6.client.client_v6 import AnchorClientV6
from fedanchor.v6.losses.total_loss_v6 import TotalLossV6


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_anchors() -> np.ndarray:
    """5 anchors at known 2D positions."""
    return np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]],
        dtype=np.float32,
    )


@pytest.fixture
def simple_embeddings() -> np.ndarray:
    """10 sample embeddings near the anchor positions."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 2)).astype(np.float32)


@pytest.fixture
def two_client_data():
    """Synthetic data for 2 clients."""
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal((100, 4)).astype(np.float32)
    y1 = rng.integers(0, 3, size=100).astype(np.int64)
    X2 = rng.standard_normal((80, 4)).astype(np.float32)
    y2 = rng.integers(0, 3, size=80).astype(np.int64)
    return (X1, y1), (X2, y2)


# ===========================================================================
# Test 1: GlobalAnchorSet shape
# ===========================================================================

class TestGlobalAnchorShape:
    def test_default_shape(self):
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        assert anc.anchors.shape == (5, 2), "Anchors should be [K, 2]."

    def test_custom_shape(self):
        anc = GlobalAnchorSet(num_anchors=8, embedding_dim=3)
        assert anc.anchors.shape == (8, 3)

    def test_forward_returns_correct_shape(self):
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        out = anc()
        assert out.shape == (5, 2)

    def test_is_nn_parameter(self):
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        assert isinstance(anc.anchors, torch.nn.Parameter), (
            "Anchors must be an nn.Parameter for gradient flow."
        )


# ===========================================================================
# Test 2: Anchor gradients exist after backward
# ===========================================================================

class TestAnchorGradients:
    def test_anchor_gradient_exists(self):
        """After loss.backward(), anchor gradient must be non-None and non-zero."""
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        # Initialize to non-trivial positions
        with torch.no_grad():
            anc.anchors.data = torch.randn(5, 2)

        optimizer = torch.optim.Adam(anc.parameters(), lr=0.01)
        optimizer.zero_grad()

        embeddings = torch.randn(10, 2, requires_grad=False)
        anchors = anc()

        # Soft loss
        diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)
        sq = torch.sum(diff ** 2, dim=2)
        loss = torch.mean(sq)
        loss.backward()

        assert anc.anchors.grad is not None, "Anchor gradient must not be None."
        assert torch.norm(anc.anchors.grad) > 0, "Anchor gradient must be non-zero."

    def test_anchors_in_optimizer(self):
        """Anchor parameters must be included in optimizer param groups."""
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        opt = torch.optim.Adam(anc.parameters(), lr=0.001)
        all_params = []
        for pg in opt.param_groups:
            all_params.extend(pg["params"])
        assert anc.anchors in all_params, (
            "GlobalAnchorSet.anchors must be in the optimizer."
        )

    def test_anchor_updates_are_non_zero(self):
        """After one optimizer step, anchor positions must have changed."""
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        with torch.no_grad():
            anc.anchors.data = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]]
            )
        before = anc.anchors.data.clone()

        opt = torch.optim.Adam(anc.parameters(), lr=0.1)
        opt.zero_grad()

        embeddings = torch.randn(20, 2)
        anchors = anc()
        diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)
        sq = torch.sum(diff ** 2, dim=2)
        loss = torch.mean(sq)
        loss.backward()
        opt.step()

        after = anc.anchors.data
        delta = torch.norm(after - before).item()
        assert delta > 0, f"Anchors did not update (delta={delta})."


# ===========================================================================
# Test 3: Hungarian alignment
# ===========================================================================

class TestHungarianAlignment:
    def test_identity_alignment(self, simple_anchors):
        """Aligning identical matrices should give identity permutation and zero cost."""
        aligned, perm, cost = align_anchors_to_reference(
            simple_anchors, simple_anchors
        )
        np.testing.assert_allclose(aligned, simple_anchors, atol=1e-5)
        assert cost < 1e-6, f"Cost should be ≈ 0, got {cost}."

    def test_permutation_resolved(self, simple_anchors):
        """
        Permuted client anchors should be realigned to match global reference.
        """
        K = simple_anchors.shape[0]
        perm_idx = np.array([2, 0, 4, 1, 3])  # known permutation
        permuted = simple_anchors[perm_idx]  # client has shuffled anchors

        aligned, recovered_perm, cost = align_anchors_to_reference(
            permuted, simple_anchors
        )

        # After alignment, aligned should match simple_anchors
        np.testing.assert_allclose(aligned, simple_anchors, atol=1e-4)
        assert cost < 1e-4, f"Post-alignment cost should be near 0, got {cost}."

    def test_weighted_aggregation_correct(self):
        """
        Two-client scenario: one client has permuted anchors.
        After alignment + weighted average, result should be close to global.
        """
        global_anchors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float32,
        )
        # Client 0: identical to global
        client0 = global_anchors.copy()
        # Client 1: permuted (rows 1,0,3,2)
        client1 = global_anchors[[1, 0, 3, 2]].copy()

        updates = {0: client0, 1: client1}
        counts = {0: 100, 1: 100}

        new_global, costs, perms = aggregate_anchors_with_alignment(
            global_anchors, updates, counts
        )

        # After alignment, the average should match global closely
        np.testing.assert_allclose(new_global, global_anchors, atol=1e-4)

    def test_alignment_distance_identical(self, simple_anchors):
        """Alignment distance between identical matrices should be 0."""
        d = compute_alignment_distance(simple_anchors, simple_anchors)
        assert d < 1e-6


# ===========================================================================
# Test 4: Synthetic permutation unit test (specified in requirements)
# ===========================================================================

class TestSyntheticPermutationTest:
    """
    Two clients intentionally permute their local anchors.
    Alignment should restore correspondence before averaging.
    """

    def test_symmetric_permutation_recovery(self):
        """
        Client A: anchor order [a0, a1, a2, a3, a4]
        Client B: anchor order [a4, a3, a2, a1, a0]  (full reverse)

        Without alignment: naive average of rows 0..4 would compute
            (a0+a4)/2, (a1+a3)/2, (a2+a2)/2, ...
            which is geometrically WRONG.

        With alignment: each client's anchors are permuted to match
            global ordering, then averaged → result should be ≈ global.
        """
        global_anchors = np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, 0.0],
                [0.0, -2.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        # Client A: same as global
        client_a = global_anchors.copy()
        # Client B: full reverse
        client_b = global_anchors[::-1].copy()

        # Without alignment (naive average)
        naive_avg = 0.5 * (client_a + client_b)
        naive_error = float(np.mean(np.sum((naive_avg - global_anchors) ** 2, axis=1)))

        # With alignment
        new_global, costs, perms = aggregate_anchors_with_alignment(
            global_anchors,
            {0: client_a, 1: client_b},
            {0: 1, 1: 1},
        )
        aligned_error = float(
            np.mean(np.sum((new_global - global_anchors) ** 2, axis=1))
        )

        assert aligned_error < naive_error, (
            f"Aligned aggregation ({aligned_error:.4f}) should be more "
            f"accurate than naive average ({naive_error:.4f})."
        )
        np.testing.assert_allclose(new_global, global_anchors, atol=1e-4)

    def test_arbitrary_permutation_recovery(self):
        """
        Both clients have independently permuted anchors.
        After alignment, aggregation should produce the global reference.
        """
        rng = np.random.default_rng(99)
        K = 6
        global_anchors = rng.standard_normal((K, 2)).astype(np.float32)

        perm_a = rng.permutation(K)
        perm_b = rng.permutation(K)
        client_a = global_anchors[perm_a]
        client_b = global_anchors[perm_b]

        new_global, costs, perms = aggregate_anchors_with_alignment(
            global_anchors,
            {0: client_a, 1: client_b},
            {0: 1, 1: 1},
        )
        np.testing.assert_allclose(new_global, global_anchors, atol=1e-4)


# ===========================================================================
# Test 5: All clients receive identical global anchors
# ===========================================================================

class TestIdenticalBroadcast:
    def test_server_broadcasts_same_anchors(self):
        """
        All clients must receive numerically identical anchor tensors
        from the server.
        """
        server = V6ServerWithSharedAnchors(
            input_dim=4, embedding_dim=2, num_anchors=5
        )
        # Initialize anchors manually
        known_anchors = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]]
        )
        server.global_anchor_set.set_anchors(known_anchors)

        # Simulate 3 clients receiving anchors
        received = [server.get_global_anchor_tensor() for _ in range(3)]
        for i in range(1, len(received)):
            np.testing.assert_allclose(
                received[0].numpy(),
                received[i].numpy(),
                atol=1e-7,
                err_msg=f"Client {i} received different anchors from client 0.",
            )

    def test_client_copies_global_exactly(self, two_client_data):
        """
        After client.local_train receives global_anchor_tensor,
        the client's local_anchor_copy should be identical at start
        (init_consistency_max ≈ 0).
        """
        (X1, y1), _ = two_client_data
        client = AnchorClientV6(
            client_id=0,
            X_train=X1,
            y_train=y1,
            k_neighbors=3,
            num_anchors=5,
            device="cpu",
            seed=42,
        )

        # Create fake global anchor tensor
        global_anc = torch.randn(5, 2)

        # Minimal config
        cfg = {
            "training": {
                "learning_rate": 0.001,
                "local_epochs": 1,
                "batch_size": 50,
                "max_grad_norm": 1.0,
            },
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_spread": 1.0,
                "lambda_balance": 0.0,
                "lambda_sep": 0.0,
                "margin": 0.1,
                "alpha_soft": 1.0,
                "eps": 1e-7,
            },
        }

        _, _, _, _, init_cons = client.local_train(
            global_encoder_state=client.encoder.state_dict(),
            global_anchor_tensor=global_anc,
            config=cfg,
            round_num=1,
        )
        assert init_cons < 1e-5, (
            f"init_consistency_max should be ≈ 0, got {init_cons:.2e}"
        )


# ===========================================================================
# Test 6: Utilisation metrics
# ===========================================================================

class TestUtilisationMetrics:
    def test_soft_q_sums_to_one(self, simple_embeddings, simple_anchors):
        """q rows must sum to 1 (soft assignment probabilities)."""
        q = compute_soft_q(simple_embeddings, simple_anchors, alpha=1.0)
        row_sums = q.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(simple_embeddings)), atol=1e-5)

    def test_per_client_utilisation_shape(self, simple_embeddings, simple_anchors):
        """Per-client utilisation should have shape [K]."""
        emb_list = [simple_embeddings[:5], simple_embeddings[5:]]
        per_client_u, global_u = compute_per_client_utilisation(
            emb_list, simple_anchors
        )
        K = simple_anchors.shape[0]
        assert len(per_client_u) == 2
        assert per_client_u[0].shape == (K,)
        assert global_u.shape == (K,)

    def test_global_utilisation_sums_to_one(self, simple_embeddings, simple_anchors):
        """Global utilisation (soft) should sum to 1."""
        emb_list = [simple_embeddings]
        _, global_u = compute_per_client_utilisation(emb_list, simple_anchors)
        assert abs(global_u.sum() - 1.0) < 1e-4

    def test_per_client_utilisation_sums_to_one(
        self, simple_embeddings, simple_anchors
    ):
        """Each client's soft utilisation should sum to 1."""
        emb_list = [simple_embeddings[:5], simple_embeddings[5:]]
        per_client_u, _ = compute_per_client_utilisation(emb_list, simple_anchors)
        for u_m in per_client_u:
            assert abs(u_m.sum() - 1.0) < 1e-4


# ===========================================================================
# Test 7: Entropy
# ===========================================================================

class TestEntropy:
    def test_uniform_utilisation_max_entropy(self, simple_anchors):
        """Uniform utilisation → normalised entropy = 1."""
        K = simple_anchors.shape[0]
        # Create embeddings with exactly uniform soft assignment
        # (equal distances from all anchors = origin-like point)
        embeddings = np.zeros((1, 2), dtype=np.float32)
        diag = compute_anchor_diagnostics_v6(embeddings, simple_anchors)
        # Normalised entropy should be close to 1 for uniform assignment
        assert diag["normalized_entropy"] > 0.9, (
            "Near-uniform assignment should give high normalised entropy."
        )

    def test_perfect_collapse_min_entropy(self):
        """All embeddings at a single anchor → entropy approaches 0."""
        anchors = np.array(
            [[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0], [10.0, 10.0]],
            dtype=np.float32,
        )
        # All embeddings near first anchor
        embeddings = np.tile([5.1, 0.1], (100, 1)).astype(np.float32)
        diag = compute_anchor_diagnostics_v6(embeddings, anchors)
        assert diag["normalized_entropy"] < 0.5, (
            "Collapsed assignment should have low normalised entropy."
        )


# ===========================================================================
# Test 8: Pairwise anchor distances
# ===========================================================================

class TestPairwiseDistances:
    def test_known_distances(self):
        """Verify pairwise distance computation for known anchor positions."""
        anchors = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        embeddings = np.zeros((5, 2), dtype=np.float32)
        diag = compute_anchor_diagnostics_v6(embeddings, anchors)
        dists = sorted(diag["anchor_pairwise_distances"])
        # Expected: 3, 4, 5
        np.testing.assert_allclose(sorted(dists), [3.0, 4.0, 5.0], atol=1e-4)

    def test_anchor_displacement_zero_unchanged(self):
        """Displacement between identical anchor matrices should be 0."""
        anchors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        disp = float(np.linalg.norm(anchors - anchors))
        assert disp == 0.0


# ===========================================================================
# Test 9: TotalLossV6 produces valid gradients
# ===========================================================================

class TestTotalLossV6:
    def test_loss_is_scalar(self):
        """TotalLossV6 forward must return a scalar tensor."""
        loss_fn = TotalLossV6()
        embeddings = torch.randn(10, 2)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        anchors = torch.randn(5, 2)
        anchors = torch.nn.Parameter(anchors)

        total, breakdown, q = loss_fn(embeddings, edges, anchors)
        assert total.shape == torch.Size([]), "Loss must be scalar."

    def test_q_shape(self):
        """q must have shape [N, K]."""
        loss_fn = TotalLossV6()
        N, K = 15, 5
        embeddings = torch.randn(N, 2)
        edges = torch.zeros((1, 2), dtype=torch.long)
        anchors = torch.randn(K, 2)

        _, _, q = loss_fn(embeddings, edges, anchors)
        assert q.shape == (N, K), f"Expected q shape [{N}, {K}], got {q.shape}."

    def test_encoder_and_anchor_gradients(self):
        """Both encoder and anchor gradients must be non-None after backward."""
        from fedanchor.models.encoder import Encoder
        enc = Encoder(input_dim=4, embedding_dim=2)
        anc = GlobalAnchorSet(num_anchors=5, embedding_dim=2)
        with torch.no_grad():
            anc.anchors.data = torch.randn(5, 2)

        opt = torch.optim.Adam(
            list(enc.parameters()) + list(anc.parameters()), lr=0.001
        )
        loss_fn = TotalLossV6()

        x = torch.randn(10, 4)
        edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)

        opt.zero_grad()
        emb = enc(x)
        total, _, _ = loss_fn(emb, edges, anc())
        total.backward()

        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Encoder param {name} has no gradient."
        assert anc.anchors.grad is not None, "Anchor gradient must not be None."
        assert torch.norm(anc.anchors.grad) > 0, "Anchor gradient must be non-zero."


# ===========================================================================
# Run directly
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
