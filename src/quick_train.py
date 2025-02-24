from .training_manager import TrainingManager
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='dev', choices=['full', 'dev', 'test', 'custom'])
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--samples', type=int, help='Number of samples to use for training')
    args = parser.parse_args()

    if args.mode == 'custom' and not args.samples:
        parser.error("--samples required when using custom mode")

    # Initialize training manager with sample size
    manager = TrainingManager(
        config_path='config/config.yaml',
        mode=args.mode,
        num_samples=args.samples
    )
    
    print(f"\nRunning {args.mode} training for {args.stages} stage(s)")
    for stage in range(args.stages):
        metrics = manager.run_training_stage()
        if metrics:
            print(f"\nStage {stage + 1} Results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main() 