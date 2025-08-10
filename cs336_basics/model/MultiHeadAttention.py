import torch
from torch import nn

from cs336_basics.nn_utils import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model, device=None, dtype=None))  # Query projection
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model, device=None, dtype=None))  # Key projection
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model, device=None, dtype=None))  # Value projection
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model, device=None, dtype=None))


    def forward(self, in_features):
        # write all the shapes of the inputs 
        # in_features: (batch_size, seq_len, d_model)
        # q_proj: (d_k, d_model)
        # k_proj: (d_k, d_model)
        # v_proj: (d_v, d_model)
        # o_proj: (d_model, d_v)
        
        Q = in_features @ self.q_proj.T # shape batch_size, seq_len, dk
        K = in_features @ self.k_proj.T # shape batch_size, seq_len, dk
        V = in_features @ self.v_proj.T # shape batch_size, seq_len, dv
        
        mask = torch.tril(torch.ones(Q.shape[1], Q.shape[1]), diagonal=0).bool()  # Upper triangular mask

        # Reshape Q, K, V for multi-head attention
        
        Q = Q.reshape(Q.shape[0],Q.shape[1], self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.reshape(K.shape[0],K.shape[1], self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.reshape(V.shape[0],V.shape[1], self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_v)
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask) # (batch_size, num_heads, seq_len, d_v)

        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(attn_output.shape[0], -1, self.d_model)
        # Apply output projection
        output = attn_output @ self.o_proj.T  # (batch_size, seq_len, d_model) 
        return output