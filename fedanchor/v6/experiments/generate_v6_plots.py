"""
FEDANCHOR V6 - Visualization Generator
========================================

Generates all 15 required plots from V6 experiment results.

Requires:
  - fedanchor/v6/outputs/controlled_shared_anchor/v6_results.json  (V6)
  - Optionally: V5 results CSV for comparison plots

Plots generated:
  1.  kNN accuracy vs round
  2.  Trustworthiness vs round
  3.  Continuity vs round
  4.  Global anchor positions over rounds (heatmap / scatter)
  5.  Anchor trajectories in 2D
  6.  Client embeddings + shared anchors (final round)
  7.  Per-client anchor utilisation (bar, final round)
  8.  Global anchor utilisation (bar, all rounds)
  9.  Anchor entropy vs round
 10.  Minimum pairwise anchor distance vs round
 11.  Anchor displacement per round (global)
 12.  Client-to-global anchor alignment distance vs round
 13.  Comparison V5 vs V6 kNN (if V5 data available)
 14.  Comparison V5 vs V6 Trustworthiness
 15.  Comparison V5 vs V6 Continuity
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Any, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _rounds(history: List[Dict]) -> List[int]:
    return [r["round"] for r in history]


# ──────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_knn(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [r["eval_metrics"]["knn_accuracy"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, vals, marker="o", color="steelblue", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("kNN Accuracy")
    ax.set_title(f"kNN Accuracy vs Round — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "01_knn_vs_round.png"))


def plot_trustworthiness(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [r["eval_metrics"]["trustworthiness"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, vals, marker="s", color="darkorange", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Trustworthiness")
    ax.set_title(f"Trustworthiness vs Round — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "02_trustworthiness_vs_round.png"))


def plot_continuity(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [r["eval_metrics"]["continuity"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, vals, marker="^", color="green", label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Continuity")
    ax.set_title(f"Continuity vs Round — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "03_continuity_vs_round.png"))


def plot_anchor_positions_over_rounds(anchor_history, out_dir):
    """Scatter plot of global anchor positions at each round (coloured by round)."""
    if not anchor_history:
        return
    K = len(anchor_history[0])
    R = len(anchor_history)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = cm.viridis
    for r_idx, anchors in enumerate(anchor_history):
        alpha = 0.3 + 0.7 * (r_idx / max(R - 1, 1))
        color = cmap(r_idx / max(R - 1, 1))
        xs = [a[0] for a in anchors]
        ys = [a[1] for a in anchors]
        ax.scatter(xs, ys, color=color, alpha=alpha, s=60, zorder=2)
    # Mark final round with larger markers
    final = anchor_history[-1]
    for j, a in enumerate(final):
        ax.scatter(a[0], a[1], color="red", s=150, zorder=3, edgecolors="black")
        ax.annotate(f"a{j}", (a[0], a[1]), fontsize=9, ha="left", va="bottom")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, R - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Round")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title("Global Anchor Positions Over Rounds")
    ax.grid(True, alpha=0.2)
    _save(fig, os.path.join(out_dir, "04_anchor_positions_over_rounds.png"))


def plot_anchor_trajectories(anchor_history, out_dir):
    """Draw trajectories of each anchor's movement across rounds."""
    if not anchor_history:
        return
    K = len(anchor_history[0])
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    fig, ax = plt.subplots(figsize=(8, 8))
    for j in range(K):
        xs = [anchor_history[r][j][0] for r in range(len(anchor_history))]
        ys = [anchor_history[r][j][1] for r in range(len(anchor_history))]
        ax.plot(xs, ys, color=colors[j], alpha=0.8, linewidth=1.5, label=f"Anchor {j}")
        ax.scatter(xs[0], ys[0], color=colors[j], marker="o", s=80, zorder=3)
        ax.scatter(xs[-1], ys[-1], color=colors[j], marker="*", s=200, zorder=4)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title("Anchor Trajectories in 2D\n(○ = start, ★ = end)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    _save(fig, os.path.join(out_dir, "05_anchor_trajectories.png"))


def plot_per_client_utilisation(round_diags, out_dir):
    """Bar chart of per-client anchor utilisation at the final round."""
    if not round_diags:
        return
    last = round_diags[-1]
    per_client = last.get("per_client_soft_utilization", [])
    if not per_client:
        return
    K = len(per_client[0])
    n_clients = len(per_client)
    x = np.arange(K)
    width = 0.8 / n_clients
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_clients))
    for m, u_m in enumerate(per_client):
        ax.bar(x + m * width, u_m, width=width, color=colors[m],
               label=f"Client {m}", alpha=0.85)
    ax.set_xticks(x + (n_clients - 1) * width / 2)
    ax.set_xticklabels([f"a{j}" for j in range(K)])
    ax.set_ylabel("Soft Utilisation")
    ax.set_title("Per-Client Anchor Utilisation (Final Round)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "07_per_client_utilisation.png"))


def plot_global_utilisation(round_diags, out_dir):
    """Stacked area chart of global anchor utilisation across rounds."""
    if not round_diags:
        return
    rounds = [d["round"] for d in round_diags]
    K = len(round_diags[0].get("global_soft_utilization", [0]))
    if K == 0:
        return
    utils = np.array(
        [d.get("global_soft_utilization", [1 / K] * K) for d in round_diags]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(rounds, utils.T, labels=[f"a{j}" for j in range(K)], alpha=0.85)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Soft Utilisation (stacked)")
    ax.set_title("Global Anchor Utilisation Over Rounds")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    _save(fig, os.path.join(out_dir, "08_global_utilisation.png"))


def plot_entropy(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [r["eval_metrics"].get("normalized_entropy", 0) for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, vals, marker="D", color="purple", label=f"Global Entropy ({label})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Max (uniform)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Normalised Entropy")
    ax.set_title(f"Anchor Entropy vs Round — {label}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "09_anchor_entropy_vs_round.png"))


def plot_min_anchor_dist(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [
        r["eval_metrics"].get("min_pairwise_anchor_distance", 0) for r in history
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, vals, marker="x", color="brown", label=f"Min Anchor Dist ({label})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Min Pairwise Anchor Distance")
    ax.set_title(f"Minimum Pairwise Anchor Distance vs Round — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "10_min_anchor_dist_vs_round.png"))


def plot_anchor_displacement(history, out_dir, label="V6"):
    rs = _rounds(history)
    vals = [r.get("anchor_displacement_global", 0) for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rs, vals, color="teal", alpha=0.8, label=f"Displacement ({label})")
    ax.set_xlabel("Round")
    ax.set_ylabel("||A_t - A_{t-1}||")
    ax.set_title(f"Global Anchor Displacement per Round — {label}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "11_anchor_displacement_per_round.png"))


def plot_alignment_cost(history, out_dir, label="V6"):
    rs = _rounds(history)
    mean_costs = [r.get("mean_alignment_cost", 0) for r in history]
    max_costs = [r.get("max_alignment_cost", 0) for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, mean_costs, marker="o", color="navy", label="Mean Alignment Cost")
    ax.plot(rs, max_costs, marker="s", color="crimson", linestyle="--",
            label="Max Alignment Cost", alpha=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Hungarian Alignment Cost (MSE)")
    ax.set_title(f"Client-to-Global Anchor Alignment Distance — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "12_alignment_distance_vs_round.png"))


def plot_v5_vs_v6_comparison(
    v5_history: Optional[List[Dict]],
    v6_history: List[Dict],
    out_dir: str,
) -> None:
    """Plots 13, 14, 15: V5 vs V6 comparison of kNN, T, C."""
    if v5_history is None:
        print("  [SKIP] No V5 history provided — comparison plots skipped.")
        return

    rs_v6 = _rounds(v6_history)
    rs_v5 = _rounds(v5_history)

    metrics = [
        ("knn_accuracy", "kNN Accuracy", "13_comparison_knn.png"),
        ("trustworthiness", "Trustworthiness", "14_comparison_trustworthiness.png"),
        ("continuity", "Continuity", "15_comparison_continuity.png"),
    ]

    for key, ylabel, fname in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        v5_vals = [r["eval_metrics"].get(key, 0) for r in v5_history]
        v6_vals = [r["eval_metrics"].get(key, 0) for r in v6_history]
        ax.plot(rs_v5, v5_vals, marker="o", color="gray", label="V5 (client-specific anchors)")
        ax.plot(rs_v6, v6_vals, marker="s", color="royalblue", label="V6 (shared global anchors)")
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"V5 vs V6 — {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(out_dir, fname))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    results_json: str = "./fedanchor/v6/outputs/controlled_shared_anchor/v6_results.json",
    plots_dir: str = "./fedanchor/v6/outputs/plots",
    v5_csv: Optional[str] = None,
) -> None:
    """
    Generate all V6 plots from saved results JSON.

    Args:
        results_json: path to v6_results.json
        plots_dir:    output directory for plots
        v5_csv:       optional path to V5 CSV for comparison plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(results_json):
        print(f"ERROR: Results file not found: {results_json}")
        print("Run the experiment first.")
        return

    with open(results_json, "r") as f:
        results = json.load(f)

    history = results["history"]
    anchor_history = results.get("anchor_history", [])
    round_diags = results.get("round_diagnostics", [])

    # Convert anchor_history from list-of-lists to list-of-numpy
    anc_hist_np = [np.array(a) for a in anchor_history]

    print(f"Generating {15} plots to: {plots_dir}")

    plot_knn(history, plots_dir)
    plot_trustworthiness(history, plots_dir)
    plot_continuity(history, plots_dir)
    plot_anchor_positions_over_rounds(anc_hist_np, plots_dir)
    plot_anchor_trajectories(anc_hist_np, plots_dir)

    # Plot 6: client embeddings + anchors (needs actual embeddings — skip if not available)
    print("  [Plot 6] Client embeddings + shared anchors: requires running trainer.")

    plot_per_client_utilisation(round_diags, plots_dir)
    plot_global_utilisation(round_diags, plots_dir)
    plot_entropy(history, plots_dir)
    plot_min_anchor_dist(history, plots_dir)
    plot_anchor_displacement(history, plots_dir)
    plot_alignment_cost(history, plots_dir)

    # V5 comparison plots
    v5_history = None
    if v5_csv and os.path.exists(v5_csv):
        v5_history = _load_v5_history_from_csv(v5_csv)

    plot_v5_vs_v6_comparison(v5_history, history, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


def _load_v5_history_from_csv(csv_path: str) -> List[Dict]:
    """Load V5 results from CSV into a history-like structure."""
    history = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(
                {
                    "round": int(row.get("round", 0)),
                    "eval_metrics": {
                        "knn_accuracy": float(row.get("knn_accuracy", 0)),
                        "trustworthiness": float(row.get("trustworthiness", 0)),
                        "continuity": float(row.get("continuity", 0)),
                        "normalized_entropy": float(
                            row.get("normalized_entropy", 0)
                        ),
                        "min_pairwise_anchor_distance": float(
                            row.get("min_pairwise_anchor_distance", 0)
                        ),
                        "effective_anchors": int(
                            row.get("effective_anchors", 0)
                        ),
                    },
                }
            )
    # De-duplicate by mode if needed — take only the last mode
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate V6 plots")
    parser.add_argument(
        "--results",
        default="./fedanchor/v6/outputs/controlled_shared_anchor/v6_results.json",
        help="Path to v6_results.json",
    )
    parser.add_argument(
        "--plots_dir",
        default="./fedanchor/v6/outputs/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--v5_csv",
        default=None,
        help="Optional V5 results CSV for comparison plots",
    )
    args = parser.parse_args()
    generate_all_plots(args.results, args.plots_dir, args.v5_csv)
