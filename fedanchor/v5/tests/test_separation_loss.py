import unittest
import torch
from fedanchor.v4.losses.anchor_separation_loss import AnchorSeparationLoss

class TestAnchorSeparationLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.margin = 0.5
        self.loss_fn = AnchorSeparationLoss(margin=self.margin)
        
    def test_completely_collapsed(self):
        anchors = torch.tensor([[0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0]], requires_grad=True)
        # K = 3, pairs = 6
        # Distances are 0.
        # penalty per pair = (margin - 0)^2 = margin^2 = 0.25
        # sum = 6 * 0.25 = 1.5
        # loss = 1.5 / 6 = 0.25
        
        loss_val = self.loss_fn(anchors)
        self.assertAlmostEqual(loss_val.item(), 0.25, places=3)
        
        # Test gradient
        loss_val.backward()
        self.assertIsNotNone(anchors.grad)
        
    def test_completely_separated(self):
        anchors = torch.tensor([[0.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 1.0]], requires_grad=True)
        # Distances are 1.0, 1.0, 1.414. All >= 0.5
        loss_val = self.loss_fn(anchors)
        self.assertAlmostEqual(loss_val.item(), 0.0, places=5)
        
        loss_val.backward()
        self.assertTrue(torch.all(anchors.grad == 0))
        
    def test_partial_violation(self):
        # A1 and A2 are close, A3 is far
        anchors = torch.tensor([[0.0, 0.0],
                                [0.1, 0.0],
                                [2.0, 2.0]], requires_grad=True)
        
        # dist(A1, A2) = 0.1
        # violation = max(0, 0.5 - 0.1) = 0.4. sq = 0.16. (happens twice)
        # other distances > 0.5
        # sum = 2 * 0.16 = 0.32
        # loss = 0.32 / 6 = 0.05333333333333334
        
        loss_val = self.loss_fn(anchors)
        self.assertAlmostEqual(loss_val.item(), 0.32 / 6, places=3)
        
        loss_val.backward()
        
        # Gradient should push A1 to left, A2 to right
        # For A1, diff with A2 is [0.1, 0]
        # grad shouldn't be zero.
        self.assertTrue(anchors.grad[0][0].item() > 0) # pushes A1 to negative x? No, wait, if we want to minimize loss, we want A1 to move away from A2. So grad for A1 should be positive?
        # A1 is at 0, A2 at 0.1. A1 should move to negative, so A1 -= lr * grad -> grad must be positive!
        self.assertTrue(anchors.grad[0][0].item() > 0)
        self.assertTrue(anchors.grad[1][0].item() < 0)
        
    def test_no_nan_on_identical(self):
        # Test that exactly identical points do not produce NaN gradients when using torch.sqrt
        anchors = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        loss_val = self.loss_fn(anchors)
        loss_val.backward()
        self.assertFalse(torch.isnan(anchors.grad).any())

if __name__ == '__main__':
    unittest.main()
