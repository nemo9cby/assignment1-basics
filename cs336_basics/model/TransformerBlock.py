import torch
from torch import nn
from cs336_basics.model.MultiHeadAttentionWithRoPE import MultiHeadAttentionWithRoPE
from cs336_basics.model.RMSNorm import RMSNorm
from cs336_basics.model.SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.mharope = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.SwiGLU = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.RMSNorm_1 = RMSNorm(d_model=d_model)
        self.RMSNorm_2 = RMSNorm(d_model=d_model)


    def forward(self, in_features):
        pre_norm = self.RMSNorm_1(in_features)
        add1 = in_features + self.mharope(pre_norm)
        pre_norm_2 = self.RMSNorm_2(add1)
        output = add1 + self.SwiGLU(pre_norm_2)
        return output
        
        