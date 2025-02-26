import torch.optim as optim

class OptimizerBuilder:
    """Builds and configures optimizer with proper parameters and weight decay."""
    
    @staticmethod
    def build(model, config):
        """
        Build optimizer with parameter groups and weight decay.
        
        Args:
            model: The model to optimize
            config: Dictionary containing optimizer configuration
                - optimizer_type: 'adam', 'sgd', etc.
                - learning_rate: Base learning rate
                - weight_decay: Weight decay factor
                - momentum: Momentum factor (for SGD)
        """
        # Separate parameters into with and without weight decay
        decay = set()
        no_decay = set()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip batch norm and biases for weight decay
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.add(name)
            else:
                decay.add(name)
        
        # Validate parameters
        param_dict = {name: param for name, param in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated into decay/no_decay set!"
        
        # Create optimizer groups
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
        
        # Initialize optimizer
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
        
        return optimizer 