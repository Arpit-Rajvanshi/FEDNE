import unittest
import torch
import numpy as np

from fedanchor.v2.losses.anchor_soft_loss import AnchorSoftLoss
from fedanchor.v2.losses.anchor_spread_loss import AnchorSpreadLoss

class TestV2Math(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # N=10 embeddings, K=3 anchors
        self.embeddings = torch.randn(10, 2, requires_grad=True)
        self.anchors = torch.tensor([
            [1.0, 1.0],
            [-1.0, -1.0],
            [2.0, -2.0]
        ], requires_grad=True)

    def test_soft_assignment_q_properties(self):
        """
        Verify that q_ij sums to 1 for every embedding, all values are finite/non-negative.
        """
        loss_fn = AnchorSoftLoss(alpha=1.0)
        loss_val, q = loss_fn(self.embeddings, self.anchors)
        
        # 1. Finite and non-negative
        self.assertTrue(torch.all(torch.isfinite(q)), "q contains non-finite values.")
        self.assertTrue(torch.all(q >= 0), "q contains negative values.")
        
        # 2. Sums to 1 over dimension 1 (the K anchors)
        row_sums = torch.sum(q, dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "q_ij does not sum to 1 over anchors.")

    def test_soft_assignment_gradients(self):
        """
        Verify that L_soft_anchor produces gradients with respect to every anchor when assignments are nonzero.
        """
        loss_fn = AnchorSoftLoss(alpha=1.0)
        loss_val, q = loss_fn(self.embeddings, self.anchors)
        
        # Ensure loss is finite
        self.assertTrue(torch.isfinite(loss_val))
        
        # Backpropagate
        loss_val.backward()
        
        # Check gradients on anchors
        self.assertIsNotNone(self.anchors.grad)
        
        # All anchors should receive some non-zero gradient since q_ij > 0 everywhere
        grad_norms = torch.norm(self.anchors.grad, dim=1)
        for i, norm in enumerate(grad_norms):
            self.assertTrue(norm.item() > 1e-6, f"Anchor {i} received zero/negligible gradient: {norm.item()}")

    def test_anchor_spread_increases_when_closer(self):
        """
        Verify that L_spread increases when two anchors are moved closer together.
        """
        loss_fn = AnchorSpreadLoss(eps=1e-7)
        
        # Baseline anchors
        anchors_base = torch.tensor([[0.0, 0.0], [1.0, 0.0]], requires_grad=True)
        loss_base = loss_fn(anchors_base)
        
        # Move closer
        anchors_close = torch.tensor([[0.0, 0.0], [0.5, 0.0]], requires_grad=True)
        loss_close = loss_fn(anchors_close)
        
        # Loss should be strictly larger
        self.assertTrue(loss_close.item() > loss_base.item(), "Spread loss did not increase when anchors moved closer.")
        
        # Move further
        anchors_far = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
        loss_far = loss_fn(anchors_far)
        
        # Loss should be strictly smaller
        self.assertTrue(loss_far.item() < loss_base.item(), "Spread loss did not decrease when anchors moved apart.")

    def test_anchor_spread_gradients(self):
        """
        Verify that L_spread is differentiable and finite.
        """
        loss_fn = AnchorSpreadLoss(eps=1e-7)
        loss_val = loss_fn(self.anchors)
        
        self.assertTrue(torch.isfinite(loss_val))
        
        loss_val.backward()
        
        self.assertIsNotNone(self.anchors.grad)
        self.assertTrue(torch.all(torch.isfinite(self.anchors.grad)))
        
        # The gradients should push anchors apart.
        # Check the gradient of anchor 0 ([1, 1]) w.r.t anchor 1 ([-1, -1]) and anchor 2 ([2, -2]).
        # The sum of gradients should not be identically zero across all dims unless perfectly symmetric.
        self.assertTrue(torch.norm(self.anchors.grad) > 1e-6)

if __name__ == '__main__':
    unittest.main()
