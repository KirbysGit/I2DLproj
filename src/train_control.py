from .training_manager import TrainingManager
import argparse
from colorama import Fore, Style

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default=1,
                       help='Number of training stages to run')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    args = parser.parse_args()
    
    manager = TrainingManager()
    
    print(f"\n{Fore.CYAN}Starting Training Pipeline - {args.stages} stage(s){Style.RESET_ALL}")
    
    for stage in range(args.stages):
        try:
            metrics = manager.run_training_stage()
            
            if metrics is not None:
                print(f"\n{Fore.GREEN}Stage {stage + 1} Results:{Style.RESET_ALL}")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
            else:
                print(f"\n{Fore.RED}Stage {stage + 1} failed - no metrics returned{Style.RESET_ALL}")
            
            # Ask user if they want to continue
            if stage < args.stages - 1:  # Don't ask after last stage
                if input(f"\n{Fore.YELLOW}Continue to next stage? (y/n): {Style.RESET_ALL}").lower() != 'y':
                    print(f"{Fore.YELLOW}Training stopped by user{Style.RESET_ALL}")
                    break
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Training interrupted by user{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Error in stage {stage + 1}: {str(e)}{Style.RESET_ALL}")
            break
    
    if args.evaluate:
        try:
            from .evaluate import main as evaluate_main
            print(f"\n{Fore.CYAN}Running Evaluation{Style.RESET_ALL}")
            evaluate_main()
        except Exception as e:
            print(f"{Fore.RED}Evaluation failed: {str(e)}{Style.RESET_ALL}")

if __name__ == '__main__':
    main() 