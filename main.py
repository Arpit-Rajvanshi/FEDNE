import fedne

import argparse
import os
import yaml
from fedne.utils.seed import set_seed
from fedne.training.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="FEDNE: Surrogate-Assisted Federated Neighbor Embedding")
    parser.add_argument(
        "--config",
        type=str,
        default="fedne/configs/mnist_default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Enforce reproducibility settings
    seed = config.get("logging", {}).get("seed", 42)
    set_seed(seed=seed, deterministic=True)
    
    # Initialize Trainer and start training
    trainer = Trainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    trainer.fit()

if __name__ == "__main__":
    main()
