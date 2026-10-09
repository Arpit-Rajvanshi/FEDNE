"""
FEDANCHOR V6 - Controlled IID Experiment
=========================================

Controlled experiment comparing V5 (client-specific anchors) vs
V6 (shared global anchors).

Protocol (as specified):
  Dataset:     MNIST (60,000 train / 10,000 test)
  Clients:     2
  Partition:   IID
  Seed:        42
  Rounds:      20
  Local epochs: 1
  Encoder:     784 → 512 → 256 → 128 → 2
  Optimizer:   Adam
  LR:          0.001
  Batch size:  1280 (same as V5 controlled experiment)
  Anchors:     K = 5
  Evaluation:  same protocol as V5 (kNN k=7, T&C k=7, subsample 2000)

Run from the D:\FEDNE directory:
    python -m fedanchor.v6.experiments.run_v6_iid_experiment
"""

import os
import sys
import json
import time
import csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fedanchor.v6.training.trainer_v6 import AnchorTrainerV6

# ─────────────────────────────────────────────────────────────
# Experiment configuration
# Mirrors V5 controlled ablation settings exactly, except the
# anchor design changes from client-specific → shared global.
# ─────────────────────────────────────────────────────────────

V6_IID_CONFIG = {
    "seed": 42,
    "dataset": {
        "name": "MNIST",
        "data_dir": "./data",
        "partition": "iid",
        "num_clients": 2,
        "alpha": 0.5,
        "max_samples_per_client": None,  # full 30,000 per client
    },
    "model": {
        "input_dim": 784,
        "embedding_dim": 2,
    },
    "anchors": {
        "num_anchors": 5,
        "initialization": "kmeans",
        "max_init_samples": 10000,
    },
    "graph": {
        "k_neighbors": 5,
    },
    "training": {
        "rounds": 20,
        "local_epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 1280,
        "max_grad_norm": 1.0,
    },
    "loss": {
        # Preserve V5 loss weights exactly
        "lambda_attraction": 1.0,
        "lambda_anchor": 1.0,
        "lambda_spread": 1.0,
        "lambda_balance": 1.0,
        "lambda_sep": 0.1,
        "lambda_embedding_reg": 0.0,
        "margin": 0.1,
        "alpha_soft": 1.0,
        "eps": 1e-7,
    },
    "ablation": {
        "use_attraction": True,
        "use_anchor": True,
        "use_spread": True,
    },
}


def run_v6_iid_experiment(
    output_dir: str = "./fedanchor/v6/outputs/controlled_shared_anchor",
) -> None:
    """
    Run the primary V6 IID experiment.
    Results are saved to output_dir/v6_results.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("FEDANCHOR V6 - SHARED GLOBAL ANCHORS")
    print("Primary Controlled IID Experiment")
    print("=" * 70)
    print()
    print("Research Question:")
    print(
        "  Can a shared/global anchor set provide a common low-dimensional"
        " geometric reference frame across federated clients and improve"
        " dimensionality-reduction quality compared with independently"
        " learned client-specific anchors?"
    )
    print()
    print("Key change: LOCAL CLIENT-SPECIFIC ANCHORS -> SHARED GLOBAL ANCHORS")
    print()

    trainer = AnchorTrainerV6(config=V6_IID_CONFIG, output_dir=output_dir)
    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    final = results["final_metrics"]
    print(f"  kNN Accuracy:      {final.get('knn_accuracy', 0):.4f}")
    print(f"  Trustworthiness:   {final.get('trustworthiness', 0):.4f}")
    print(f"  Continuity:        {final.get('continuity', 0):.4f}")
    print(f"  Normalised Entropy:{final.get('normalized_entropy', 0):.4f}")
    print(f"  Effective Anchors: {final.get('effective_anchors', 0)}")
    print(
        f"  Min Anchor Dist:   "
        f"{final.get('min_pairwise_anchor_distance', 0):.4f}"
    )
    print(f"  Collapse State:    {final.get('collapse_state', 'UNKNOWN')}")
    print(f"  Total Time:        {elapsed:.1f}s")

    # ── CSV summary ───────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "v6_iid_results.csv")
    rows = []
    for r_info in results["history"]:
        ev = r_info["eval_metrics"]
        rows.append(
            {
                "round": r_info["round"],
                "loss": r_info["loss"],
                "attraction_loss": r_info.get("attraction_loss", 0),
                "anchor_loss": r_info.get("anchor_loss", 0),
                "anchor_spread_loss": r_info.get("anchor_spread_loss", 0),
                "separation_loss": r_info.get("separation_loss", 0),
                "balance_loss": r_info.get("balance_loss", 0),
                "knn_accuracy": ev.get("knn_accuracy", 0),
                "trustworthiness": ev.get("trustworthiness", 0),
                "continuity": ev.get("continuity", 0),
                "normalized_entropy": ev.get("normalized_entropy", 0),
                "effective_anchors": ev.get("effective_anchors", 0),
                "min_pairwise_anchor_distance": ev.get(
                    "min_pairwise_anchor_distance", 0
                ),
                "mean_pairwise_anchor_distance": ev.get(
                    "mean_pairwise_anchor_distance", 0
                ),
                "collapse_state": ev.get("collapse_state", ""),
                "anchor_displacement_global": r_info.get(
                    "anchor_displacement_global", 0
                ),
                "mean_alignment_cost": r_info.get("mean_alignment_cost", 0),
                "anchor_gradient_norm": r_info.get("anchor_gradient_norm", 0),
                "fraction_nonzero_update": r_info.get("fraction_nonzero_update", 0),
                "failure_anchor_collapse": r_info.get("failure_modes", {}).get(
                    "anchor_collapse", False
                ),
                "failure_anchor_drift": r_info.get("failure_modes", {}).get(
                    "anchor_drift", False
                ),
                "failure_client_domination": r_info.get("failure_modes", {}).get(
                    "client_domination", False
                ),
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV results: {csv_path}")
    print(f"Full results: {os.path.join(output_dir, 'v6_results.json')}")


if __name__ == "__main__":
    run_v6_iid_experiment()
