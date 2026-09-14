import sys
import os
import unittest
import torch
import numpy as np

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import AnchorSet
from fedanchor.graph.knn import construct_knn_graph
from fedanchor.losses.attraction import AttractionLoss
from fedanchor.losses.anchor_loss import AnchorCoverageLoss
from fedanchor.losses.anchor_repulsion import AnchorRepulsionLoss
from fedanchor.losses.total_loss import TotalLoss

class TestAnchorComponents(unittest.TestCase):
    
    def test_encoder_output_shape(self):
        encoder = Encoder(input_dim=784, embedding_dim=2)
        batch = torch.randn(32, 784)
        output = encoder(batch)
        self.assertEqual(output.shape, (32, 2))

    def test_anchor_initialization_strategies(self):
        # Test kmeans, farthest_point, representative_points
        for strategy in ["kmeans", "farthest_point", "representative_points"]:
            anchor_set = AnchorSet(num_anchors=5, embedding_dim=2)
            embeddings = torch.randn(20, 2)
            anchor_set.init_from_embeddings(embeddings, strategy=strategy, seed=42)
            anchors = anchor_set()
            self.assertEqual(anchors.shape, (5, 2))
            self.assertTrue(anchor_set.initialized)

    def test_knn_graph_shape(self):
        X = np.random.randn(50, 784).astype(np.float32)
        k = 5
        edges = construct_knn_graph(X, k=k)
        self.assertEqual(edges.shape, (50 * k, 2))

    def test_attraction_loss_finite(self):
        att_loss_fn = AttractionLoss()
        embeddings = torch.randn(20, 2, requires_grad=True)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
        loss = att_loss_fn(embeddings, edges)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_anchor_coverage_loss_finite(self):
        cov_loss_fn = AnchorCoverageLoss()
        embeddings = torch.randn(20, 2, requires_grad=True)
        anchors = torch.randn(5, 2, requires_grad=True)
        loss = cov_loss_fn(embeddings, anchors)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_anchor_repulsion_loss_finite(self):
        rep_loss_fn = AnchorRepulsionLoss(normalization="scale", scale=0.01)
        embeddings = torch.randn(20, 2, requires_grad=True)
        other_anchors = torch.randn(10, 2)
        loss, breakdown = rep_loss_fn(embeddings, other_anchors)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertIn("repulsion_raw", breakdown)
        self.assertIn("repulsion_scaled", breakdown)

    def test_total_loss_computation(self):
        total_loss_fn = TotalLoss()
        embeddings = torch.randn(20, 2, requires_grad=True)
        edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        local_anchors = torch.randn(5, 2, requires_grad=True)
        other_anchors = torch.randn(5, 2)
        
        loss, breakdown = total_loss_fn(embeddings, edges, local_anchors, other_anchors)
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("total_loss", breakdown)
        self.assertIn("attraction_loss", breakdown)
        self.assertIn("anchor_loss", breakdown)
        self.assertIn("anchor_repulsion_raw", breakdown)
        self.assertIn("anchor_repulsion_scaled", breakdown)
        self.assertIn("repulsion_to_attraction_ratio", breakdown)

    # REQUIRED AUDIT TEST 4: Anchor coverage is approx zero when anchors equal cluster centroids
    def test_anchor_coverage_zero_on_exact_centroids(self):
        cov_loss_fn = AnchorCoverageLoss()
        # 4 distinct points placed directly on 4 anchor coordinates
        anchors = torch.tensor([[1.0, 1.0], [5.0, 5.0], [-2.0, 3.0], [0.0, -4.0]], dtype=torch.float32)
        embeddings = anchors.clone()  # exact match
        
        loss = cov_loss_fn(embeddings, anchors)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
