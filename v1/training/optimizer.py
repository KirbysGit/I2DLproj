# training/optimizer.py

# -----

# Creates an Optimizer that helps train Model by Updating Parameters.

# -----

# Imports.
import torch
import torch.optim as optim

# Optimizer Builder Class.
class OptimizerBuilder:
    """Builds and configures Optimizer w/ Proper Parameters & Weight Decay."""
    
    @staticmethod
    def build(model, config):
        """
        Build Optimizer w/ Parameter Groups & Weight Decay.
        
        Args:
            model: The model to optimize.
            config: Dictionary containing optimizer configuration.
                - optimizer_type: 'adam', 'sgd', etc.
                - learning_rate: Base learning rate.
                - weight_decay: Weight decay factor.
                - momentum: Momentum factor (for SGD).
        """
        # Separate Parameters into Groups.
        decay = set()
        no_decay = set()
        
        # Iterate Over Parameters.
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip Batch Norm & Biases for Weight Decay.
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.add(name)
            else:
                decay.add(name)
        
        # Validate Parameters.
        param_dict = {name: param for name, param in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated into decay/no_decay set!"
        
        # Create Optimizer Groups.
        optim_groups = [
            {
                "params": [param_dict[name] for name in sorted(decay)],
                "weight_decay": config['weight_decay']
            },
            {
                "params": [param_dict[name] for name in sorted(no_decay)],
                "weight_decay": 0.0
            }
        ]
        
        # Initialize Optimizer.
        optimizer_type = config.get('optimizer_type', 'adam').lower()
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                optim_groups,
                lr=config['learning_rate'],
                betas=config.get('adam_betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                optim_groups,
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Return Optimizer.
        return optimizer

def build_optimizer(model, config):
    """Build optimizer for training.
    
    Args:
        model: The model to optimize
        config: Dictionary containing optimizer configuration
            - learning_rate: Learning rate
            - weight_decay: Weight decay factor
            - optimizer_type: Type of optimizer (default: 'adamw')
    
    Returns:
        torch.optim.Optimizer: The configured optimizer
    """
    # Get optimizer parameters
    lr = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 1e-4))
    optimizer_type = config.get('optimizer_type', 'adamw').lower()
    
    # Filter parameters that require gradients
    parameters = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer based on type
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer 