# CONTROLLED EVALUATION PROTOCOL

## Motivation
Previously, FEDNE and FEDANCHOR were evaluated using different protocols for kNN accuracy:
- **FEDNE**: Evaluated by extracting 2,000 test embeddings, splitting them 80/20 (1600 train, 400 test), and computing kNN accuracy.
- **FEDANCHOR**: Evaluated by using all 60,000 training embeddings to fit the kNN classifier, and evaluating on all 10,000 test embeddings.

This discrepancy made direct numerical comparison invalid.

## Unified Protocol
To establish a scientifically sound, apples-to-apples comparison, we define the following controlled evaluation protocol to be applied identically to both models at the end of every federated round.

### Dataset & Partitioning
- **Dataset**: MNIST
- **Training Samples**: 60,000 (partitioned IID)
- **Test Samples**: 10,000
- **Clients**: 2
- **Data Partition**: IID (30,000 per client)
- **Random Seed**: 42 (applied to numpy, random, and PyTorch)

### Model Configuration
- **Encoder Architecture**: 784 -> 512 -> 256 -> 128 -> 2
- **Activation**: ReLU
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Local Epochs**: 1
- **Rounds**: 20

### Evaluation Configuration
At the end of each round, the global encoder is evaluated using:
1. **kNN Classifier**:
   - `k=5`
   - Fitted on a fixed, seeded subset of 2,000 training embeddings.
   - Evaluated on a fixed, seeded subset of 2,000 test embeddings.
2. **Trustworthiness (T) & Continuity (C)**:
   - `k=5`
   - Evaluated on the same fixed subset of 2,000 test embeddings.

By evaluating both models with this identical unified protocol, any differences in T, C, or kNN Accuracy are strictly attributable to the training methodology (Surrogate vs Anchor) rather than evaluation artifacts.
