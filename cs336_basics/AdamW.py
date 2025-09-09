from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        # Simple printing of param_groups
        for i, group in enumerate(self.param_groups):
            print(f"\n=== Param Group {i} ===")
            print(f"Number of parameters: {len(group['params'])}")
            print(f"Learning rate: {group['lr']}")
            print(f"Betas: {group['betas']}")
            print(f"Eps: {group['eps']}")
            print(f"Weight decay: {group['weight_decay']}")
            
            # Print info about each parameter
            for j, param in enumerate(group['params']):
                print(f"  Param {j}: shape={param.shape}, dtype={param.dtype}")
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)  # Initialize once
                    state['v'] = torch.zeros_like(p.data)  # Initialize once
                
                state['t'] += 1
                t = state['t']
                m = state['m']
                v = state['v']

                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                m = beta1 * m + (1 - beta1)*p.grad
                v = beta2 * v + (1 - beta2)*p.grad*p.grad
                state['m'] = m
                state['v'] = v
                old_lr = group['lr']
                new_lr = old_lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p.data - new_lr*m /(torch.sqrt(v) + group['eps'])
                p.data = p.data - old_lr*group['weight_decay']*p.data
        
        return loss
        

