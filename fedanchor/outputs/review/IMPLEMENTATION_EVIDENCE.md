# EVIDENCE INVENTORY — FEDANCHOR

> Verified against actual code files. All file paths confirmed to exist.

## Evidence Table

| Component | Implemented | File | Tested | Experimental Evidence | Review Ready |
|---|---|---|---|---|---|
| Data loading (MNIST) | YES | fedanchor/datasets/loader.py | Via trainer | fedne_vs_fedanchor/dataset_manifest.json | YES |
| Client partitioning (IID) | YES | fedanchor/datasets/loader.py | Via trainer | experiment_metadata.json: 30k per client | YES |
| Encoder (784→2) | YES | fedanchor/models/encoder.py | test_encoder_output_shape PASS | All experiment runs use encoder | YES |
| Dimensionality reduction | YES | encoder.py + trainer.py | End-to-end test PASS | Embedding PNGs in all exp dirs | YES |
| kNN graph | YES | fedanchor/graph/knn.py | test_knn_graph_shape PASS | Static, used every training run | YES |
| Attraction loss | YES | fedanchor/losses/attraction.py | test_attraction_loss_finite PASS | CSV: attraction column (very small) | YES |
| Anchor initialization (3 modes) | YES | fedanchor/models/anchor.py | test_anchor_initialization_strategies PASS | initial_anchors.json: 3 strategies | YES |
| Anchor learning | YES | fedanchor/models/anchor.py | test_local_anchors_update PASS | anchor_positions.png shows learned positions | YES |
| Anchor coverage loss | YES | fedanchor/losses/anchor_loss.py | test_anchor_coverage_loss_finite + zero_centroids PASS | CSV: anchor_loss column | YES |
| Cross-client anchor repulsion | YES | fedanchor/losses/anchor_repulsion.py | test_anchor_repulsion_loss_finite + gradients PASS | CSV: repulsion_raw/scaled columns | YES |
| Federated aggregation (FedAvg) | YES | fedanchor/server/server.py | test_fedavg_weighted_aggregation PASS (2.5 exact) | All multi-round experiments | YES |
| Client/server communication | YES | server.py + client.py + trainer.py | End-to-end PASS | CSV: upload/download bytes columns | YES |
| Anchor gradient isolation | YES | server.py + client.py | test_other_anchors_no_gradients PASS | — | YES |
| Evaluation: Trustworthiness | YES | fedanchor/evaluation/metrics.py | Numerically verified | CSV: trustworthiness column | YES |
| Evaluation: Continuity | YES | fedanchor/evaluation/metrics.py | Numerically verified | CSV: continuity column | YES |
| Evaluation: kNN accuracy | YES | fedanchor/evaluation/metrics.py | — | CSV: knn_accuracy column | PARTIAL (protocol differs from FEDNE) |
| Anchor diagnostics | YES | fedanchor/evaluation/metrics.py | — | CSV: anchor_utilization, mean/max dist | YES |
| Visualization | YES | fedanchor/evaluation/visualization.py | Confirmed (PNGs exist) | 4 PNG files per experiment dir | YES |
| Checkpointing | YES | fedanchor/training/trainer.py | test_end_to_end confirms ckpt exists | fedanchor_checkpoint.pt (2.27 MB) | YES |
| Reproducibility | YES | fedanchor/utils/seed.py + configs/mnist_default.yaml | seed=42 hardcoded | experiment_metadata.json | YES |
| Ablation study | YES (2 rounds) | run_fedne_vs_fedanchor.py + outputs/ablation_* | 4 configs run | ablation PNG dirs | PARTIAL (only 2 rounds) |
| FEDNE baseline | YES | fedne/ (READ-ONLY) | — | fedne_results.csv (20 rounds) | YES (as comparison) |
| FEDNE vs FEDANCHOR comparison | YES | run_fedne_vs_fedanchor.py | — | fedne_vs_fedanchor/ all files | PARTIAL (kNN protocol differs) |
| Communication accounting | YES | server.py compute_communication_bytes | — | CSV: upload/download columns | YES |
| Full batch encoder | YES | client.py train_epoch | test_client_local_training_step PASS | All runs | YES |
