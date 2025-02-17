import json
from pathlib import Path
from .utils import load_config
from .utils import ColorLogger
from colorama import Fore
import logging
from .train_main import main as train_main

class TrainingManager:
    def __init__(self, config_path='config/config.yaml', mode='full'):
        self.config_path = config_path
        self.mode = mode  # 'full', 'dev', or 'test'
        self.history_path = Path('results/training_history.json')
        
        # Reset all training state
        self.reset_training_state()
        
        # Single logger instance
        self.logger = ColorLogger()
        
        # Load and modify config based on mode
        self.config = load_config(config_path)
        if mode == 'dev':
            self._setup_dev_mode()
        elif mode == 'test':
            self._setup_test_mode()

    def _setup_dev_mode(self):
        """Setup faster development training"""
        self.config['dataset'].update({
            'dev_mode': True,
            'dev_samples': {'train': 100, 'val': 20}
        })
        self.config['training'].update({
            'epochs_per_stage': 2,
            'batch_size': 4,
            'save_frequency': 1
        })
        
    def _setup_test_mode(self):
        """Setup minimal test training"""
        self.config['dataset'].update({
            'test_mode': True,
            'test_samples': 10
        })
        self.config['training'].update({
            'epochs_per_stage': 3,
            'batch_size': 2,
            'save_frequency': 1
        })

    def reset_training_state(self):
        """Reset all training state including history and checkpoints"""
        # Remove old checkpoint
        checkpoint_path = Path('models/latest_checkpoint.pt')
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Reset history
        self.history = {
            'stages_completed': 0,
            'total_epochs': 0,
            'best_metrics': {'f1_score': 0.0, 'epoch': 0}
        }
        self.save_history()

    def load_history(self):
        """Load training history if exists"""
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'stages_completed': 0,
                'total_epochs': 0,
                'best_metrics': {
                    'f1_score': 0.0,
                    'epoch': 0
                }
            }

    def save_history(self):
        """Save training history"""
        self.history_path.parent.mkdir(exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def reset_history(self):
        """Reset training history"""
        self.history = {
            'stages_completed': 0,
            'total_epochs': 0,
            'best_metrics': {
                'f1_score': 0.0,
                'epoch': 0
            }
        }
        self.save_history()

    def run_training_stage(self):
        try:
            stage = self.history['stages_completed'] + 1
            config = self.config
            
            # Configure epochs for this stage
            epochs_per_stage = config['training']['epochs_per_stage']
            start_epoch = self.history['total_epochs']
            end_epoch = start_epoch + epochs_per_stage
            
            self.logger.info(f"\nStage {stage}: Training epochs {start_epoch} -> {end_epoch}")
            
            # Verify checkpoint if resuming
            if stage > 1:
                checkpoint_path = Path('models/latest_checkpoint.pt')
                if not checkpoint_path.exists():
                    self.logger.warning("No checkpoint found - starting from scratch")
                    start_epoch = 0
                    self.history['total_epochs'] = 0
            
            # Update config for this stage
            config['training'].update({
                'start_epoch': start_epoch,
                'epochs': end_epoch,
                'current_stage': stage,
                'resume_from_checkpoint': stage > 1
            })
            
            # Run training
            metrics = train_main(config)
            if metrics is None:
                self.logger.error("Training failed - no metrics returned")
                return None
            
            # Update history
            self.history['stages_completed'] = stage
            self.history['total_epochs'] = end_epoch
            
            # Update best metrics if improved
            if metrics['f1_score'] > self.history['best_metrics']['f1_score']:
                self.history['best_metrics'].update({
                    'f1_score': metrics['f1_score'],
                    'epoch': end_epoch
                })
                self.logger.success(f"New best F1: {metrics['f1_score']:.4f}")
            
            self.save_history()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Stage {stage} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 