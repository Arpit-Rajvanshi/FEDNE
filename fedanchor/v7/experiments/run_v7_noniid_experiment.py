"""
FEDANCHOR v7 - Non-IID Experiment (Optional Second Experiment)
==============================================================

Tests whether shared anchors are especially useful when clients have
different data distributions.

Purpose: Determine if the shared reference frame helps MORE or LESS
when client data distributions differ significantly.

Partition: Dirichlet (alpha=0.1, highly non-IID)
Everything else: same as IID experiment.
"""

import os
import sys
import json
import csv
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fedanchor.v7.training.trainer_v7 import AnchorTrainerv7

v7_NONIID_CONFIG = {
    "seed": 42,
    "dataset": {
        "name": "MNIST",
        "data_dir": "./data",
        "partition": "dirichlet",   # Non-IID
        "num_clients": 2,
        "alpha": 0.1,               # Highly non-IID
        "max_samples_per_client": None,
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


def run_v7_noniid_experiment(
    output_dir: str = "./fedanchor/v7/outputs/controlled_shared_anchor/noniid",
) -> None:
    """
    Run the optional v7 Non-IID experiment.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("FEDANCHOR v7 - SHARED GLOBAL ANCHORS")
    print("Optional Non-IID Experiment (Dirichlet alpha=0.1)")
    print("=" * 70)
    print()
    print("Purpose: Test whether shared anchors are especially useful")
    print("when clients have different data distributions.")
    print()

    trainer = AnchorTrainerv7(config=v7_NONIID_CONFIG, output_dir=output_dir)
    t0 = time.time()
    results = trainer.train()
    elapsed = time.time() - t0

    final = results["final_metrics"]
    print("\n" + "=" * 70)
    print("NON-IID FINAL RESULTS SUMMARY")
    print("=" * 70)
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

    # Save CSV
    csv_path = os.path.join(output_dir, "v7_noniid_results.csv")
    rows = []
    for r_info in results["history"]:
        ev = r_info["eval_metrics"]
        rows.append(
            {
                "round": r_info["round"],
                "loss": r_info["loss"],
                "knn_accuracy": ev.get("knn_accuracy", 0),
                "trustworthiness": ev.get("trustworthiness", 0),
                "continuity": ev.get("continuity", 0),
                "normalized_entropy": ev.get("normalized_entropy", 0),
                "effective_anchors": ev.get("effective_anchors", 0),
                "min_pairwise_anchor_distance": ev.get(
                    "min_pairwise_anchor_distance", 0
                ),
                "collapse_state": ev.get("collapse_state", ""),
                "anchor_displacement_global": r_info.get(
                    "anchor_displacement_global", 0
                ),
                "mean_alignment_cost": r_info.get("mean_alignment_cost", 0),
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV results: {csv_path}")


if __name__ == "__main__":
    run_v7_noniid_experiment()
