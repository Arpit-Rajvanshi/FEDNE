import unittest
import torch
import math
from fedanchor.v3.losses.anchor_balance_loss import AnchorBalanceLoss
from fedanchor.v3.losses.anchor_soft_loss import AnchorSoftLoss

class TestAnchorBalanceLoss(unittest.TestCase):
    def setUp(self):
        self.balance_fn = AnchorBalanceLoss()
        self.soft_fn = AnchorSoftLoss()
        torch.manual_seed(42)

    def test_balance_loss_finite_and_differentiable(self):
        N, K = 100, 5
        q = torch.softmax(torch.randn(N, K), dim=1)
        q.requires_grad = True
        
        loss, u = self.balance_fn(q)
        
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(u.shape, (K,))
        
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertFalse(torch.isnan(q.grad).any())

    def test_collapsed_configuration(self):
        # A deliberately collapsed configuration where all anchors are at origin (0,0)
        N, K = 10, 5
        embeddings = torch.randn(N, 2)
        anchors = torch.zeros(K, 2, requires_grad=True)
        
        # Soft assignments when anchors are identical will be exactly 1/K for all
        _, q = self.soft_fn(embeddings, anchors)
        loss, u = self.balance_fn(q)
        
        # Because all q_ij are 1/K, utilization should be perfectly uniform
        expected_u = torch.ones(K) / K
        self.assertTrue(torch.allclose(u, expected_u))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        
        # Now let's artificially move one anchor closer to the points so it dominates
        anchors2 = torch.zeros(K, 2, requires_grad=True)
        with torch.no_grad():
            anchors2[0] = embeddings[0] # Make anchor 0 exactly match embedding 0
            
        _, q2 = self.soft_fn(embeddings, anchors2)
        loss2, u2 = self.balance_fn(q2)
        
        loss2.backward()
        
        self.assertGreater(loss2.item(), 0.0)
        self.assertIsNotNone(anchors2.grad)
        
    def test_q_ij_properties(self):
        N, K = 10, 5
        embeddings = torch.randn(N, 2)
        anchors = torch.randn(K, 2)
        
        _, q = self.soft_fn(embeddings, anchors)
        
        # q_ij should be >= 0
        self.assertTrue(torch.all(q >= 0))
        
        # assignments should sum to 1
        q_sum = torch.sum(q, dim=1)
        self.assertTrue(torch.allclose(q_sum, torch.ones(N)))
        
        # Finite
        self.assertTrue(torch.all(torch.isfinite(q)))

if __name__ == '__main__':
    unittest.main()
