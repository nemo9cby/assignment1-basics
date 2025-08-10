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
        
        # Apply RoPE to Q and K for each head
        # RoPE expects shape: (..., seq_len, d_k)
        # We need to apply it per head, so reshape accordingly
        Q_rotated = []
        K_rotated = []
        
        for head_idx in range(self.num_heads):
            # Extract head: (batch_size, seq_len, d_k)
            Q_head = Q[:, head_idx, :, :]
            K_head = K[:, head_idx, :, :]
            
            # Apply RoPE
            Q_head_rotated = self.rope(Q_head, token_positions)
            K_head_rotated = self.rope(K_head, token_positions)
            
            Q_rotated.append(Q_head_rotated)
            K_rotated.append(K_head_rotated)
        
        # Stack rotated heads back
        Q = torch.stack(Q_rotated, dim=1)  # (batch_size, num_heads, seq_len, d_k)
        K = torch.stack(K_rotated, dim=1)  # (batch_size, num_heads, seq_len, d_k)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0).bool()
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        output = attn_output @ self.o_proj.T
        
        return output