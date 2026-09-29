import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fedanchor.client.client import AnchorClient
from fedanchor.server.server import AnchorServer
from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import summarize_embeddings_as_anchors
from fedanchor.losses.anchor_repulsion import AnchorRepulsionLoss
from fedanchor.losses.local_repulsion import LocalNegativeRepulsionLoss

V2_CONFIG = {
    "anchors": {"num_anchors": 8},
    "training": {"mode": "neighbor_embedding", "learning_rate": 0.001, "local_epochs": 1, "batch_size": 32},
    "graph": {"k": 5, "negative_samples": 5},
    "loss": {"lambda_attraction": 1.0, "lambda_local_repulsion": 1.0, "lambda_anchor_repulsion": 1.0,
             "anchor_repulsion": {"normalization": "none", "scale": 1.0, "spread_scale": 1.0}},
}


class TestAnchorV2(unittest.TestCase):

    def test_anchor_summary_masses_and_spreads(self):
        Z = np.random.RandomState(0).randn(500, 2).astype(np.float32)
        centers, counts, spreads = summarize_embeddings_as_anchors(Z, 8, seed=0)
        self.assertEqual(centers.shape, (8, 2))
        self.assertEqual(int(counts.sum()), 500)
        self.assertTrue(np.all(spreads >= 0))

    def test_weighted_repulsion_gradients_and_decay(self):
        fn = AnchorRepulsionLoss(num_negatives=5)
        anchors = torch.zeros(1, 2)
        w, s = torch.tensor([0.5]), torch.tensor([0.1])
        near = torch.tensor([[0.1, 0.0]], requires_grad=True)
        far = torch.tensor([[10.0, 0.0]])
        l_near, _ = fn(near, anchors, weights=w, spreads=s)
        l_far, _ = fn(far, anchors, weights=w, spreads=s)
        self.assertGreater(l_near.item(), l_far.item())
        l_near.backward()
        self.assertLess(near.grad[0, 0].item(), 0.0)  # descent step pushes z away from the anchor

    def test_local_repulsion_positive(self):
        fn = LocalNegativeRepulsionLoss()
        self.assertGreater(fn(torch.zeros(4, 2), torch.randn(4, 5, 2)).item(), 0.0)

    def test_v2_federated_round(self):
        rng = np.random.RandomState(1)
        server = AnchorServer(input_dim=784, embedding_dim=2)
        clients = [AnchorClient(i, rng.rand(60, 784).astype(np.float32), rng.randint(0, 10, 60), num_anchors=8)
                   for i in range(2)]
        for c in clients:
            c.total_samples = 120
            server.client_sizes[c.client_id] = c.num_samples
        for r in (1, 2):
            states = []
            for c in clients:
                st, anc, m = c.local_train(server.get_global_encoder_state(), server.get_other_anchors(c.client_id),
                                           V2_CONFIG, round_num=r,
                                           other_anchor_meta=server.get_other_anchor_meta(c.client_id))
                self.assertEqual(anc.shape, (8, 2))
                self.assertFalse(np.isnan(m["total_loss"]))
                server.update_client_anchors(c.client_id, anc, meta=c.anchor_meta)
                states.append(st)
            server.aggregate_encoders(states, [60, 60])
        meta = server.get_other_anchor_meta(0)
        # masses of the other client's anchors sum to |D_1| / |D|
        self.assertAlmostEqual(meta["weights"].sum().item(), 0.5, places=5)
        self.assertGreater(m["anchor_repulsion_raw"], 0.0)


if __name__ == "__main__":
    unittest.main()
