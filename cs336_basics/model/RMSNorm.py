import torch
from torch import nn
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.mean(torch.square(x),dim=-1,keepdim=True)
        rms = torch.sqrt(output + self.eps)
        rms_norm = x / rms

        return rms_norm * self.weights
