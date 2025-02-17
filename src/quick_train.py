from .training_manager import TrainingManager
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'dev', 'test'], default='dev',
                       help='Training mode: full, dev, or test')
    parser.add_argument('--stages', type=int, default=1,
                       help='Number of training stages')
    args = parser.parse_args()
    
    # Initialize manager with specified mode
    manager = TrainingManager(mode=args.mode)
    
    print(f"\nRunning {args.mode} training for {args.stages} stage(s)")
    for stage in range(args.stages):
        metrics = manager.run_training_stage()
        if metrics:
            print(f"\nStage {stage + 1} Results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main() 