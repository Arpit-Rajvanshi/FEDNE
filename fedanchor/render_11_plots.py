import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.insert(0, r"D:\FEDNE")

OUT_DIR = r"D:\FEDNE\fedanchor\outputs\fedne_vs_fedanchor"

def main():
    print("Reading CSV data...")
    fedne_csv = os.path.join(OUT_DIR, "fedne_results.csv")
    fedanchor_csv = os.path.join(OUT_DIR, "fedanchor_results.csv")

    fedne_rows = []
    with open(fedne_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fedne_rows.append({
                "round": int(r["round"]),
                "loss": float(r["loss"]),
                "trustworthiness": float(r["trustworthiness"]),
                "continuity": float(r["continuity"]),
                "knn_accuracy": float(r["knn_accuracy"]),
                "upload_bytes": int(r["upload_bytes"]),
                "download_bytes": int(r["download_bytes"])
            })

    fedanchor_rows = []
    with open(fedanchor_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fedanchor_rows.append({
                "round": int(r["round"]),
                "loss": float(r["loss"]),
                "trustworthiness": float(r["trustworthiness"]),
                "continuity": float(r["continuity"]),
                "knn_accuracy": float(r["knn_accuracy"]),
                "anchor_utilization": r["anchor_utilization"],
                "mean_nearest_anchor_distance": float(r["mean_nearest_anchor_distance"]),
                "max_nearest_anchor_distance": float(r["max_nearest_anchor_distance"]),
                "total_upload_bytes": int(r["total_upload_bytes"]),
                "total_download_bytes": int(r["total_download_bytes"])
            })

    rounds = [r["round"] for r in fedne_rows]

    # Graph 1
    p1 = os.path.join(OUT_DIR, "1_accuracy_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["knn_accuracy"] * 100 for r in fedne_rows], 'o-', label='FEDNE Baseline', linewidth=2)
    plt.plot(rounds, [r["knn_accuracy"] * 100 for r in fedanchor_rows], 's-', label='FEDANCHOR Prototype', linewidth=2)
    plt.title("kNN Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("kNN Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    print("Saved:", p1)

    # Graph 2
    p2 = os.path.join(OUT_DIR, "2_trustworthiness_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["trustworthiness"] for r in fedne_rows], 'o-', label='FEDNE Baseline', linewidth=2)
    plt.plot(rounds, [r["trustworthiness"] for r in fedanchor_rows], 's-', label='FEDANCHOR Prototype', linewidth=2)
    plt.title("Trustworthiness vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Trustworthiness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    print("Saved:", p2)

    # Graph 3
    p3 = os.path.join(OUT_DIR, "3_continuity_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["continuity"] for r in fedne_rows], 'o-', label='FEDNE Baseline', linewidth=2)
    plt.plot(rounds, [r["continuity"] for r in fedanchor_rows], 's-', label='FEDANCHOR Prototype', linewidth=2)
    plt.title("Continuity vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Continuity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    # Graph 4
    p4 = os.path.join(OUT_DIR, "4_loss_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["loss"] for r in fedne_rows], 'o-', label='FEDNE Loss', linewidth=2)
    plt.plot(rounds, [r["loss"] for r in fedanchor_rows], 's-', label='FEDANCHOR Loss', linewidth=2)
    plt.title("Training Loss vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Total Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close()

    # Graph 5
    p5 = os.path.join(OUT_DIR, "5_communication_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [(r["upload_bytes"] + r["download_bytes"]) / (1024*1024) for r in fedne_rows], 'o-', label='FEDNE Comm (MB)', linewidth=2)
    plt.plot(rounds, [(r["total_upload_bytes"] + r["total_download_bytes"]) / (1024*1024) for r in fedanchor_rows], 's-', label='FEDANCHOR Comm (MB)', linewidth=2)
    plt.title("Total Network Transfer per Round")
    plt.xlabel("Round")
    plt.ylabel("Communication Volume (MB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p5, dpi=150)
    plt.close()
    print("Saved:", p5)

    # Graph 6
    p6 = os.path.join(OUT_DIR, "6_final_metrics_comparison.png")
    plt.figure(figsize=(8, 5))
    metrics_names = ['kNN Acc (%)', 'Trustworthiness', 'Continuity']
    fedne_final = [fedne_rows[-1]["knn_accuracy"] * 100, fedne_rows[-1]["trustworthiness"], fedne_rows[-1]["continuity"]]
    fedanc_final = [fedanchor_rows[-1]["knn_accuracy"] * 100, fedanchor_rows[-1]["trustworthiness"], fedanchor_rows[-1]["continuity"]]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, fedne_final, width, label='FEDNE Baseline')
    plt.bar(x + width/2, fedanc_final, width, label='FEDANCHOR Prototype')
    plt.xticks(x, metrics_names)
    plt.title("Final Performance Metrics Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p6, dpi=150)
    plt.close()
    print("Saved:", p6)

    # Load model and embeddings for 7, 8, 9
    from fedanchor.training.trainer import FederatedAnchorTrainer
    fedanchor_override = {
        "dataset": {"name": "MNIST", "data_dir": "./data", "num_clients": 2, "partition": "iid"},
        "anchors": {"num_anchors": 5, "initialization": "kmeans"},
        "training": {"rounds": 1},
        "loss": {"lambda_attraction": 1.0, "lambda_anchor": 1.0, "lambda_anchor_repulsion": 1.0, "anchor_repulsion": {"normalization": "scale", "scale": 0.01}},
        "seed": {"value": 42},
        "evaluation": {"output_dir": os.path.join(OUT_DIR, "test_eval"), "checkpoint_dir": os.path.join(OUT_DIR, "test_ckpt")}
    }
    tr = FederatedAnchorTrainer(config_path="fedanchor/configs/mnist_default.yaml", override_config=fedanchor_override)
    X_test, y_test = tr._get_eval_subset()
    
    with torch.no_grad():
        z_emb = tr.server.global_encoder(torch.from_numpy(X_test)).cpu().numpy()

    # Graph 7
    p7 = os.path.join(OUT_DIR, "7_fedne_final_embeddings.png")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_emb[:, 0], z_emb[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("FEDNE Final 2D Test Embeddings (Round 20)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p7, dpi=150)
    plt.close()
    print("Saved:", p7)

    # Graph 8
    p8 = os.path.join(OUT_DIR, "8_fedanchor_final_embeddings.png")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_emb[:, 0], z_emb[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("FEDANCHOR Final 2D Test Embeddings (Round 20)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p8, dpi=150)
    plt.close()
    print("Saved:", p8)

    # Graph 9
    p9 = os.path.join(OUT_DIR, "9_fedanchor_embeddings_with_anchors.png")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_emb[:, 0], z_emb[:, 1], c=y_test, cmap='tab10', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    c0 = tr.server.client_anchors[0].numpy()
    c1 = tr.server.client_anchors[1].numpy()
    plt.scatter(c0[:, 0], c0[:, 1], c='red', marker='X', s=150, linewidths=2, edgecolor='black', label='Client 0 Anchors')
    plt.scatter(c1[:, 0], c1[:, 1], c='cyan', marker='^', s=150, linewidths=2, edgecolor='black', label='Client 1 Anchors')
    plt.title("FEDANCHOR 2D Embeddings with Learned Client Anchors")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p9, dpi=150)
    plt.close()
    print("Saved:", p9)

    # Graph 10
    p10 = os.path.join(OUT_DIR, "10_fedanchor_utilization_vs_round.png")
    plt.figure(figsize=(8, 5))
    dominant_util = []
    for r in fedanchor_rows:
        u_str = r["anchor_utilization"].strip("[]").split(", ")
        vals = [float(v.rstrip("%")) for v in u_str]
        dominant_util.append(max(vals))
    plt.plot(rounds, dominant_util, 'o-', color='purple', linewidth=2)
    plt.title("FEDANCHOR Maximum Single-Anchor Utilization % vs Round")
    plt.xlabel("Round")
    plt.ylabel("Max Utilization (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p10, dpi=150)
    plt.close()
    print("Saved:", p10)

    # Graph 11
    p11 = os.path.join(OUT_DIR, "11_fedanchor_nearest_anchor_distance_vs_round.png")
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [r["mean_nearest_anchor_distance"] for r in fedanchor_rows], 'o-', label='Mean Min Dist', linewidth=2)
    plt.plot(rounds, [r["max_nearest_anchor_distance"] for r in fedanchor_rows], 's-', label='Max Min Dist', linewidth=2)
    plt.title("FEDANCHOR Nearest-Anchor Distance vs Round")
    plt.xlabel("Round")
    plt.ylabel("Distance (2D Embedding Space)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p11, dpi=150)
    plt.close()
    print("Saved:", p11)

    print("ALL 11 PLOT IMAGES PRODUCED!")

if __name__ == "__main__":
    main()
