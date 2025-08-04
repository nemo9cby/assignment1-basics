import torch
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # dff = round((d_model * 8/3) / 64) * 64 # i want this number to be the closest to multiples of 64
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        # how should i init these weights?
    
    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T