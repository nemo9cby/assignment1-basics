import torch
from torch import nn

from cs336_basics.nn_utils import scaled_dot_product_attention
from cs336_basics.model.RoPE import RoPE

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Initialize projection matrices
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        
        # Initialize RoPE for queries and keys
        # RoPE is applied per head, so d_k = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=None)
    
    def forward(self, in_features, token_positions=None):
        batch_size, seq_len, _ = in_features.shape
        
        # If token_positions not provided, assume sequential positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=in_features.device)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        
        # Project to Q, K, V
        Q = in_features @ self.q_proj.T  # (batch_size, seq_len, d_model)
        K = in_features @ self.k_proj.T  # (batch_size, seq_len, d_model)
        V = in_features @ self.v_proj.T  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE to Q and K for all heads in parallel
        # Current shape: (batch_size, num_heads, seq_len, d_k)
        # RoPE expects: (..., seq_len, d_k)
        # We can reshape to apply RoPE to all heads at once
        
        # Merge batch and head dimensions for vectorized RoPE application
        batch_size_heads = batch_size * self.num_heads
        Q_merged = Q.reshape(batch_size_heads, seq_len, self.d_k)
        K_merged = K.reshape(batch_size_heads, seq_len, self.d_k)
        
        # Expand token_positions to match the merged batch dimension
        # Original: (batch_size, seq_len)
        # Need: (batch_size * num_heads, seq_len)
        token_positions_expanded = token_positions.unsqueeze(1).expand(
            batch_size, self.num_heads, seq_len
        ).reshape(batch_size_heads, seq_len)
        
        # Apply RoPE to all heads at once
        Q_rotated = self.rope(Q_merged, token_positions_expanded)
        K_rotated = self.rope(K_merged, token_positions_expanded)
        
        # Reshape back to separate head dimension
        Q = Q_rotated.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        K = K_rotated.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0).bool()
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        output = attn_output @ self.o_proj.T
        
        return output