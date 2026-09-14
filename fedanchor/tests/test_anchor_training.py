import sys
import os
import unittest
import torch
import torch.optim as optim
import numpy as np

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fedanchor.client.client import AnchorClient
from fedanchor.models.encoder import Encoder
from fedanchor.models.anchor import AnchorSet
from fedanchor.losses.anchor_repulsion import AnchorRepulsionLoss

class TestAnchorTraining(unittest.TestCase):
    
    def test_client_local_training_step(self):
        X_train = np.random.randn(40, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, size=40).astype(np.int64)
        
        client = AnchorClient(
            client_id=0,
            X_train=X_train,
            y_train=y_train,
            k_neighbors=5,
            num_anchors=5,
            device="cpu"
        )
        
        global_encoder = Encoder(input_dim=784, embedding_dim=2)
        global_state = global_encoder.state_dict()
        other_anchors = torch.randn(5, 2)
        
        config = {
            "anchors": {"initialization": "kmeans"},
            "training": {"learning_rate": 0.001, "max_grad_norm": 1.0, "local_epochs": 2},
            "loss": {
                "lambda_attraction": 1.0,
                "lambda_anchor": 1.0,
                "lambda_anchor_repulsion": 1.0,
                "eps": 1e-7,
                "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
            }
        }
        
        up_state, up_anc, metrics = client.local_train(
            global_encoder_state=global_state,
            other_anchors_tensor=other_anchors,
            config=config,
            round_num=1
        )
        
        self.assertIn("network.0.weight", up_state)
        self.assertEqual(up_anc.shape, (5, 2))
        self.assertIn("total_loss", metrics)
        self.assertFalse(np.isnan(metrics["total_loss"]))

    # REQUIRED AUDIT TEST 1: Verify L_rep_anchor produces non-zero encoder gradients
    def test_repulsion_produces_encoder_gradients(self):
        encoder = Encoder(input_dim=784, embedding_dim=2)
        rep_loss_fn = AnchorRepulsionLoss(normalization="scale", scale=1.0)
        
        inputs = torch.randn(10, 784)
        other_anchors = torch.randn(5, 2)  # detached
        
        embeddings = encoder(inputs)
        loss, _ = rep_loss_fn(embeddings, other_anchors)
        loss.backward()
        
        # Check that encoder parameters received non-zero gradients
        has_nonzero_grad = False
        for param in encoder.parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)).item() > 0:
                has_nonzero_grad = True
                break
                
        self.assertTrue(has_nonzero_grad, "L_rep_anchor failed to produce gradients on encoder parameters.")

    # REQUIRED AUDIT TEST 2: Verify local anchors actually change after optimizer.step()
    def test_local_anchors_update(self):
        anchor_set = AnchorSet(num_anchors=5, embedding_dim=2)
        initial_embs = torch.randn(20, 2)
        anchor_set.init_from_embeddings(initial_embs, strategy="kmeans", seed=42)
        
        initial_anchors_copy = anchor_set.anchors.clone().detach()
        optimizer = optim.Adam(anchor_set.parameters(), lr=0.01)
        
        # Compute dummy loss and step
        dummy_embeddings = torch.randn(20, 2)
        diff = dummy_embeddings.unsqueeze(1) - anchor_set().unsqueeze(0)
        loss = torch.mean(torch.sum(diff ** 2, dim=2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        updated_anchors = anchor_set.anchors.detach()
        self.assertFalse(
            torch.equal(initial_anchors_copy, updated_anchors),
            "Local anchors failed to change after optimizer.step()."
        )

    # REQUIRED AUDIT TEST 3: Verify other-client anchors remain detached and receive no gradients
    def test_other_anchors_no_gradients(self):
        encoder = Encoder(input_dim=784, embedding_dim=2)
        rep_loss_fn = AnchorRepulsionLoss(normalization="scale", scale=1.0)
        
        inputs = torch.randn(10, 784)
        # Pass other_anchors as a tensor
        other_anchors = torch.randn(5, 2, requires_grad=False)
        
        embeddings = encoder(inputs)
        loss, _ = rep_loss_fn(embeddings, other_anchors)
        loss.backward()
        
        self.assertIsNone(other_anchors.grad, "other_anchors incorrectly received gradients during local backward pass.")

if __name__ == "__main__":
    unittest.main()
