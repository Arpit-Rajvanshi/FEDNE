"""
AnchorTrainerV6 - FEDANCHOR V6 Training Orchestrator
=====================================================

Federated training loop for the Shared Global Anchors design.

Architectural difference from V5:
  - ONE global anchor matrix A ∈ R^(K × 2) maintained on server.
  - All clients receive the SAME A before each round.
  - After local training, Hungarian alignment is applied before
    weighted averaging of client anchor updates.
  - The new A_global is broadcast to all clients next round.

Round structure:
  1. Server broadcasts (encoder_state, global_anchors) to all clients.
  2. Each client:
       a. Loads encoder from global state.
       b. Copies global anchors into local_anchor_copy.
       c. Runs local_epochs of training (encoder + anchors optimised jointly).
       d. Returns (encoder_update, anchor_update, metrics, diagnostics).
  3. Server performs:
       a. FedAvg on encoder updates.
       b. Hungarian alignment of each client's anchor update to current global.
       c. Weighted average → new global anchors.
  4. Server evaluates global model on test set.
  5. Collect and log all diagnostics.
  6. Repeat for `rounds` rounds.

Initialization:
  Before round 1, the server runs one forward pass through the initial
  global encoder on each client's data (or a combined subset) to get
  initial 2D embeddings, then fits KMeans(K) to initialize global anchors.
"""

import os
import time
import json
import yaml
import copy
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional

from fedanchor.utils.seed import set_seed
from fedanchor.datasets.loader import load_dataset
from fedanchor.v6.training.server_v6 import V6ServerWithSharedAnchors
from fedanchor.v6.client.client_v6 import AnchorClientV6
from fedanchor.v6.evaluation.metrics_v6 import compute_metrics_v6
from fedanchor.v6.diagnostics.round_diagnostics import (
    build_round_diagnostics,
    print_round_summary,
)


class AnchorTrainerV6:
    """
    Orchestrates FEDANCHOR V6 federated training.

    Usage:
        trainer = AnchorTrainerV6(config)
        results = trainer.train()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "./fedanchor/v6/outputs/controlled_shared_anchor",
    ) -> None:
        """
        Args:
            config:     Full training configuration dict.
            output_dir: Directory for saving results / checkpoints.
        """
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        seed_val = config.get("seed", 42)
        set_seed(seed_val)
        self.seed = seed_val

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ----- Load & partition dataset -----
        ds_cfg = config.get("dataset", {})
        self.client_data, self.test_data = load_dataset(
            name=ds_cfg.get("name", "MNIST"),
            data_dir=ds_cfg.get("data_dir", "./data"),
            partition_type=ds_cfg.get("partition", "iid"),
            num_clients=ds_cfg.get("num_clients", 2),
            alpha=ds_cfg.get("alpha", 0.5),
            max_samples_per_client=ds_cfg.get("max_samples_per_client", None),
        )
        self.num_clients = len(self.client_data)

        # ----- Build server -----
        model_cfg = config.get("model", {})
        anc_cfg = config.get("anchors", {})
        self.num_anchors = anc_cfg.get("num_anchors", 5)

        self.server = V6ServerWithSharedAnchors(
            input_dim=model_cfg.get("input_dim", 784),
            embedding_dim=model_cfg.get("embedding_dim", 2),
            num_anchors=self.num_anchors,
            device=self.device,
        )

        # ----- Build clients -----
        graph_cfg = config.get("graph", {})
        self.clients = [
            AnchorClientV6(
                client_id=i,
                X_train=X_c,
                y_train=y_c,
                k_neighbors=graph_cfg.get("k_neighbors", 5),
                num_anchors=self.num_anchors,
                device=self.device,
                seed=self.seed,
            )
            for i, (X_c, y_c) in enumerate(self.client_data)
        ]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_global_anchors(self) -> None:
        """
        Initialize the global anchor matrix ONCE before federated training.

        Strategy:
          1. Run the initial (untrained) global encoder on each client's
             full training set to get initial 2D embeddings.
          2. Concatenate all embeddings.
          3. Run KMeans(K) on the combined embeddings.
          4. Store result as A_0 (initial global anchor matrix).

        This ensures label-free, data-driven initialization without bias
        towards any single client.

        Records initialization details for provenance.
        """
        init_cfg = self.config.get("anchors", {})
        strategy = init_cfg.get("initialization", "kmeans")
        max_init_samples = init_cfg.get("max_init_samples", 10000)

        self.server.global_encoder.eval()
        all_embeddings = []

        with torch.no_grad():
            for client in self.clients:
                x_t = client.X_tensor.to(self.device)
                if max_init_samples and len(x_t) > max_init_samples // self.num_clients:
                    idx = torch.randperm(len(x_t))[
                        : max_init_samples // self.num_clients
                    ]
                    x_t = x_t[idx]
                emb = (
                    self.server.global_encoder(x_t).cpu().numpy()
                )
                all_embeddings.append(emb)

        combined = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        combined_tensor = torch.tensor(combined, dtype=torch.float32)

        self.server.initialize_global_anchors(
            combined_embeddings=combined_tensor,
            strategy=strategy,
            seed=self.seed,
        )

        print(
            f"[INIT] Global anchors initialized using '{strategy}' "
            f"on {combined.shape[0]} combined embeddings from "
            f"{self.num_clients} clients."
        )
        A0 = self.server.get_global_anchor_tensor().numpy()
        print(f"[INIT] Initial anchor coordinates:\n{A0}")

        # Save initialization provenance
        init_record = {
            "strategy": strategy,
            "num_combined_samples": int(combined.shape[0]),
            "num_clients": self.num_clients,
            "seed": self.seed,
            "initial_anchor_coordinates": A0.tolist(),
        }
        path = os.path.join(self.output_dir, "anchor_initialization.json")
        with open(path, "w") as f:
            json.dump(init_record, f, indent=2)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run federated training and return full results dict.

        Returns:
            {
                "history":          list of per-round result dicts
                "round_diagnostics": list of per-round V6 diagnostic dicts
                "communication":    communication analysis
                "total_time":       float (seconds)
                "final_metrics":    final round eval metrics
                "anchor_history":   list of [K, 2] anchor matrices per round
            }
        """
        train_cfg = self.config.get("training", {})
        rounds = train_cfg.get("rounds", 20)
        k = self.config.get("graph", {}).get("k_neighbors", 5)

        # Step 0: Initialize global anchors
        self._initialize_global_anchors()

        comm_stats = self.server.compute_communication_bytes(
            num_clients=self.num_clients
        )

        history: List[Dict[str, Any]] = []
        round_diagnostics_list: List[Dict[str, Any]] = []
        start_time = time.time()

        prev_global_anchors: Optional[np.ndarray] = None

        for r in range(1, rounds + 1):
            round_start = time.time()

            # -------- Broadcast global state to all clients --------
            global_enc_state = self.server.get_global_encoder_state()
            global_anchor_tensor = self.server.get_global_anchor_tensor()  # [K, 2]
            current_global_np = global_anchor_tensor.numpy().copy()

            # -------- Local training --------
            client_enc_updates: List[Dict[str, torch.Tensor]] = []
            client_anc_updates: Dict[int, torch.Tensor] = {}
            client_counts: List[int] = []
            round_metrics_list: List[Dict[str, float]] = []
            all_step_diagnostics: List[Dict[str, float]] = []
            init_consistency: Dict[int, float] = {}

            for client in self.clients:
                (
                    upd_enc,
                    upd_anc,
                    metrics,
                    step_diags,
                    init_cons_max,
                ) = client.local_train(
                    global_encoder_state=global_enc_state,
                    global_anchor_tensor=global_anchor_tensor,
                    config=self.config,
                    round_num=r,
                )
                client_enc_updates.append(upd_enc)
                client_anc_updates[client.client_id] = upd_anc
                client_counts.append(client.num_samples)
                round_metrics_list.append(metrics)
                all_step_diagnostics.extend(step_diags)
                init_consistency[client.client_id] = init_cons_max

            # -------- Aggregate encoders (FedAvg) --------
            self.server.aggregate_encoders(client_enc_updates, client_counts)

            # -------- Aggregate anchors (Hungarian + weighted average) --------
            counts_dict = {
                c.client_id: c.num_samples for c in self.clients
            }
            alignment_costs, permutations = self.server.aggregate_anchors(
                client_anc_updates, counts_dict
            )

            new_global_np = self.server.get_global_anchor_tensor().numpy().copy()

            # -------- Evaluate --------
            X_test, y_test = self.test_data
            self.server.global_encoder.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(
                    X_test, dtype=torch.float32
                ).to(self.device)
                X_test_low = (
                    self.server.global_encoder(X_test_t).cpu().numpy()
                )

                client_embs = []
                X_train_low_list = []
                y_train_list = []
                for client in self.clients:
                    emb = client.get_embeddings_eval()
                    client_embs.append(emb)
                    X_train_low_list.append(emb)
                    y_train_list.append(client.y_train)

                X_train_low = np.concatenate(X_train_low_list, axis=0)
                y_train_cat = np.concatenate(y_train_list, axis=0)
                X_high_train = np.concatenate(
                    [c.X_train for c in self.clients], axis=0
                )

            eval_metrics = compute_metrics_v6(
                X_high_train=X_high_train,
                X_low_train=X_train_low,
                y_train=y_train_cat,
                X_high_test=X_test,
                X_low_test=X_test_low,
                y_test=y_test,
                global_anchors=new_global_np,
                client_embeddings=client_embs,
                margin=self.config.get("loss", {}).get("margin", 0.1),
                k=k,
            )

            # -------- Build diagnostics --------
            diag = build_round_diagnostics(
                round_num=r,
                global_anchors=new_global_np,
                prev_global_anchors=prev_global_anchors,
                eval_metrics=eval_metrics,
                alignment_costs=alignment_costs,
                permutations=permutations,
                client_sample_counts=counts_dict,
                init_consistency_max_per_client=init_consistency,
            )
            round_diagnostics_list.append(diag)
            print_round_summary(diag)

            # -------- Log losses --------
            avg_loss = float(
                np.mean([m.get("total_loss", 0.0) for m in round_metrics_list])
            )
            avg_att = float(
                np.mean([m.get("attraction_loss", 0.0) for m in round_metrics_list])
            )
            avg_anc = float(
                np.mean([m.get("anchor_loss", 0.0) for m in round_metrics_list])
            )
            avg_spread = float(
                np.mean([m.get("anchor_spread_loss", 0.0) for m in round_metrics_list])
            )
            avg_sep = float(
                np.mean([m.get("separation_loss", 0.0) for m in round_metrics_list])
            )
            avg_bal = float(
                np.mean([m.get("balance_loss", 0.0) for m in round_metrics_list])
            )

            round_elapsed = time.time() - round_start
            print(f"  Time: {round_elapsed:.2f}s | Loss: {avg_loss:.4f}")

            # Anchor opt stats
            if all_step_diagnostics:
                avg_grad = float(
                    np.mean([d["pre_clip_grad_norm"] for d in all_step_diagnostics])
                )
                avg_disp = float(
                    np.mean([d["displacement"] for d in all_step_diagnostics])
                )
                frac_nz = float(
                    np.mean([d["nonzero_update"] for d in all_step_diagnostics])
                )
            else:
                avg_grad = avg_disp = frac_nz = 0.0

            round_info = {
                "round": r,
                "loss": avg_loss,
                "attraction_loss": avg_att,
                "anchor_loss": avg_anc,
                "anchor_spread_loss": avg_spread,
                "separation_loss": avg_sep,
                "balance_loss": avg_bal,
                "eval_metrics": eval_metrics,
                "round_time": round_elapsed,
                "optimizer_steps_per_round": len(all_step_diagnostics),
                "anchor_gradient_norm": avg_grad,
                "anchor_displacement_local": avg_disp,
                "fraction_nonzero_update": frac_nz,
                "anchor_displacement_global": diag["anchor_displacement"],
                "mean_alignment_cost": diag["mean_alignment_cost"],
                "max_alignment_cost": diag["max_alignment_cost"],
                "failure_modes": diag["failure_modes"],
            }
            history.append(round_info)

            prev_global_anchors = new_global_np.copy()

        total_time = time.time() - start_time

        # -------- Save results --------
        results = {
            "history": history,
            "round_diagnostics": round_diagnostics_list,
            "communication": comm_stats,
            "total_time": total_time,
            "final_metrics": history[-1]["eval_metrics"] if history else {},
            "anchor_history": [
                a.tolist() for a in self.server.anchor_history
            ],
            "config": self.config,
        }

        results_path = os.path.join(self.output_dir, "v6_results.json")
        with open(results_path, "w") as f:
            # Convert non-serializable types
            json.dump(_make_serializable(results), f, indent=2)

        print(f"\nResults saved to: {results_path}")

        return results


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj
