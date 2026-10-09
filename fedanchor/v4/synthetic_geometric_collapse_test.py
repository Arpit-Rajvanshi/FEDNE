import torch
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fedanchor.v4.losses.anchor_separation_loss import AnchorSeparationLoss

def run_synthetic_test():
    print("==================================================")
    print("V4 SYNTHETIC GEOMETRIC COLLAPSE TEST")
    print("==================================================")
    
    # 1. Deliberately collapsed configuration (all anchors at origin)
    # K=5 anchors perfectly collapsed (with tiny noise to break exact 0 symmetry).
    import torch.nn as nn
    torch.manual_seed(42)
    anchors = nn.Parameter(torch.randn((5, 2)) * 1e-4)
    margin = 0.5
    
    loss_fn = AnchorSeparationLoss(margin=margin)
    optimizer = optim.SGD([anchors], lr=0.1)
    
    print(f"Initial Anchors:\n{anchors.detach().numpy()}")
    
    # Forward pass
    loss = loss_fn(anchors)
    print(f"Initial L_sep: {loss.item():.4f}")
    
    # Verifications
    assert loss.item() > 0, "L_sep should be positive for collapsed configuration."
    assert not torch.isinf(loss) and not torch.isnan(loss), "L_sep must be finite."
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradients should be non-zero
    grad_norm = torch.norm(anchors.grad)
    print(f"Gradient Norm: {grad_norm.item():.4f}")
    assert grad_norm > 0, "Gradients should be non-zero to spread anchors."
    
    # Step
    optimizer.step()
    
    # Re-evaluate
    new_loss = loss_fn(anchors)
    print(f"L_sep after 1 step: {new_loss.item():.4f}")
    assert new_loss.item() < loss.item(), "A gradient step must reduce the separation violation."
    print(f"Anchors after 1 step:\n{anchors.detach().numpy()}")
    print("Collapsed verification PASSED.\n")
    
    # 2. Separated configuration
    print("Testing Separated Configuration...")
    separated_anchors = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ], requires_grad=True)
    
    # All pairs are separated by at least 1.0 > margin (0.5)
    sep_loss = loss_fn(separated_anchors)
    print(f"L_sep for separated configuration: {sep_loss.item():.4f}")
    assert sep_loss.item() < 1e-5, "L_sep should be approx 0 for separated anchors."
    print("Separated verification PASSED.\n")
    
    print("ALL V4 SYNTHETIC TESTS PASSED.")

if __name__ == "__main__":
    run_synthetic_test()
