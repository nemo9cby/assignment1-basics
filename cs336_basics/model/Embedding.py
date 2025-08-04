import torch
from torch import nn
import math

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor):
        return self.weights[token_ids]