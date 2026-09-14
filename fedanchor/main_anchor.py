import os
import argparse
import sys

# Ensure sys.get_int_max_str_digits exists for Python 3.11 pre-releases
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits = lambda maxdigits: None

# Add workspace parent directory to sys.path to enable fedanchor package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fedanchor.training.trainer import FederatedAnchorTrainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Anchor (FEDANCHOR) Prototype Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="fedanchor/configs/mnist_default.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    print("==================================================")
    print("      FEDERATED ANCHOR (FEDANCHOR) PROTOTYPE      ")
    print("==================================================")
    print(f"Loading Configuration from: {args.config}\n")
    
    trainer = FederatedAnchorTrainer(config_path=args.config)
    results = trainer.train()
    
    print("==================================================")
    print("               EXPERIMENT COMPLETE                ")
    print("==================================================")
    print(f"Total Elapsed Time: {results['total_time']:.2f} seconds")
    print(f"Checkpoint Saved To: {results['checkpoint_path']}")
    print("\nVisualizations Created:")
    for k, p in results['visualizations'].items():
        print(f"  - {k}: {p}")
        
    print("\nCommunication Breakdown:")
    comm = results['communication']
    print(f"  - Encoder Parameter Count: {comm['encoder_parameter_count']} scalars ({comm['encoder_bytes_per_client']} bytes)")
    print(f"  - Anchor Scalar Count per Client: {comm['anchor_scalar_count_per_client']} scalars ({comm['anchor_bytes_per_client']} bytes)")
    print(f"  - Anchor-to-Encoder Parameter Ratio: {comm['anchor_to_encoder_scalar_ratio'] * 100:.4f}%")
    print(f"  - Total Round Transmitted Volume: {comm['total_round_bytes']} bytes ({comm['total_round_bytes'] / 1024:.2f} KB)")
    print("==================================================")

if __name__ == "__main__":
    main()
