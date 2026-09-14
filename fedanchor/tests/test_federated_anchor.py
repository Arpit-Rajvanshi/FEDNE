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

from fedanchor.server.server import AnchorServer
from fedanchor.training.trainer import FederatedAnchorTrainer

class TestFederatedAnchor(unittest.TestCase):
    
    def test_fedavg_weighted_aggregation(self):
        server = AnchorServer(input_dim=784, embedding_dim=2, device="cpu")
        state1 = server.get_global_encoder_state()
        state2 = server.get_global_encoder_state()
        
        for k in state1.keys():
            state1[k] = torch.ones_like(state1[k]) * 1.0
            state2[k] = torch.ones_like(state2[k]) * 3.0
            
        server.aggregate_encoders([state1, state2], [100, 300])
        
        agg_state = server.get_global_encoder_state()
        first_param_val = list(agg_state.values())[0][0, 0].item()
        self.assertAlmostEqual(first_param_val, 2.5, places=4)

    def test_anchor_storage_independent(self):
        server = AnchorServer(input_dim=784, embedding_dim=2, device="cpu")
        anc_client0 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        anc_client1 = torch.tensor([[5.0, 5.0], [6.0, 6.0]])
        
        server.update_client_anchors(0, anc_client0)
        server.update_client_anchors(1, anc_client1)
        
        anc_0 = server.client_anchors[0]
        anc_1 = server.client_anchors[1]
        self.assertFalse(torch.equal(anc_0, anc_1))
        
        other_0 = server.get_other_anchors(0)
        self.assertTrue(torch.equal(other_0, anc_client1))

    def test_end_to_end_toy_experiment(self):
        trainer = FederatedAnchorTrainer(config_path="fedanchor/configs/test_config.yaml")
        results = trainer.train()
        
        self.assertIn("history", results)
        self.assertEqual(len(results["history"]), 2)
        self.assertIn("communication", results)
        self.assertTrue(os.path.exists(results["checkpoint_path"]))
        self.assertTrue(os.path.exists(results["visualizations"]["embeddings_with_anchors"]))

if __name__ == "__main__":
    unittest.main()
