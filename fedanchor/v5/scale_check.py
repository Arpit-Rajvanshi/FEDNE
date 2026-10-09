import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fedanchor.v4.training.trainer_v4 import AnchorTrainerV4

def run_scale_check():
    print("==================================================")
    print("V4 CRITICAL SCALE CHECK")
    print("==================================================")
    
    override = {
        "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 1, "max_samples_per_client": 100},
        "anchors": {"num_anchors": 5, "initialization": "kmeans", "num_anchors_per_client": 5},
        "training": {"rounds": 1, "local_epochs": 1},
        "loss": {
            "lambda_attraction": 1.0,
            "lambda_anchor": 1.0,
            "lambda_anchor_repulsion": 1.0,
            "lambda_spread": 1.0,  
            "lambda_balance": 0.1,
            "lambda_sep": 0.1,
            "margin": 0.10,
            "anchor_repulsion": {"normalization": "scale", "scale": 0.01}
        },
        "evaluation": {
            "output_dir": "./fedanchor/v4/outputs/temp",
            "checkpoint_dir": "./fedanchor/v4/outputs/temp"
        }
    }
    
    trainer = AnchorTrainerV4(config_path="fedanchor/configs/mnist_default.yaml", override_config=override)
    
    # We will just run 1 round and look at the losses
    # To get gradients, we will mock a tiny training loop
    client = trainer.clients[0]
    
    client.encoder.train()
    client.anchor_set.train()
    client.initialize_anchors()
    
    loss_cfg = trainer.config["loss"]
    from fedanchor.v4.losses.total_loss_v4 import TotalLossV4
    loss_fn = TotalLossV4(
        lambda_attraction=1.0,
        lambda_anchor=1.0,
        lambda_anchor_repulsion=1.0,
        lambda_spread=1.0,
        lambda_balance=0.1,
        lambda_sep=0.1,
        margin=0.10,
        use_spread=True,
        use_attraction=True,
        use_anchor=True,
        use_anchor_repulsion=True
    )
    
    other_anchors = trainer.server.get_other_anchors(client.client_id).to(client.device)
    embeddings = client.encoder(client.X_tensor)
    local_anchors = client.anchor_set()
    
    total_loss, breakdown = loss_fn(
        embeddings=embeddings,
        edges=client.edges_tensor,
        local_anchors=local_anchors,
        other_anchors=other_anchors
    )
    
    print(f"LOSS MAGNITUDES:")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.6f}")
        
    print("\nScaling lambda_sep to 100.0 to see if it causes issues...")
    loss_fn.lambda_sep = 100.0
    total_loss2, breakdown2 = loss_fn(
        embeddings=embeddings,
        edges=client.edges_tensor,
        local_anchors=local_anchors,
        other_anchors=other_anchors
    )
    print(f"LOSS MAGNITUDES (lambda_sep=100.0):")
    for k, v in breakdown2.items():
        print(f"  {k}: {v:.6f}")
        
    total_loss2.backward()
    
    print("\nGRADIENT NORMS:")
    enc_grad = torch.norm(torch.cat([p.grad.flatten() for p in client.encoder.parameters() if p.grad is not None]))
    anc_grad = torch.norm(client.anchor_set.anchors.grad)
    print(f"  Encoder Grad Norm: {enc_grad.item():.6f}")
    print(f"  Anchor Grad Norm: {anc_grad.item():.6f}")

if __name__ == "__main__":
    run_scale_check()
