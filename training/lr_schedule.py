import torch
import torch.nn as nn

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Get learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)