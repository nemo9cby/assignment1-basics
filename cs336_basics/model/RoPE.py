import torch
from torch import nn
import math

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # how to store Ri for each position i?
        # maybe a list of tensors?
        R_M = [] # this will be a block diagonal matrix of all Ri
        for i in range(max_seq_len):
            R_list = [] # list of tensors, each tensor is 2x2
            for k in range(d_k//2):
                angle = i / pow(theta, 2*k/d_k)
                # now i need to use the angle to create the Ri
                Ri = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                   [math.sin(angle), math.cos(angle)]], device=device)
                R_list.append(Ri)
            R_M.append(torch.block_diag(*R_list))
        R_M = torch.stack(R_M, dim=0)
        
        # Register as a buffer (non-trainable parameter that moves with model.to(device))
        self.register_buffer('R_M', R_M)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
                # Save original shape for reshaping later
        # original_shape = x.shape
        # batch_dims, seq_len, d_k = original_shape
        
        # # Flatten batch dimensions for easier manipulation
        # # New shape: (batch_size, seq_len, d_k) where batch_size = product of all batch dims
        # x_flat = x.reshape(-1, seq_len, d_k)
        # token_positions_flat = token_positions.reshape(-1, seq_len)
        
        # batch_size = x_flat.shape[0]
        
        # # Gather rotation matrices for the specified positions
        # # token_positions_flat has shape (batch_size, seq_len)
        # # We need to index into R_M which has shape (max_seq_len, d_k, d_k)
        
        # # First, expand token_positions to index into the d_k x d_k matrices
        # # Shape: (batch_size, seq_len, 1, 1)
        # positions_expanded = token_positions_flat.unsqueeze(-1).unsqueeze(-1)
        
        # # Expand positions to match the matrix dimensions
        # # Shape: (batch_size, seq_len, d_k, d_k)
        # positions_expanded = positions_expanded.expand(batch_size, seq_len, d_k, d_k)
        
        # # Gather the rotation matrices for each position
        # # We need to add batch dimension to R_M and expand it
        # # Shape of R_M: (max_seq_len, d_k, d_k)
        # # We want: (batch_size, seq_len, d_k, d_k)
        # R_M_expanded = self.R_M.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # # Gather rotation matrices based on positions
        # # This selects the rotation matrix at each position index
        # rotation_matrices = torch.gather(R_M_expanded, 1, positions_expanded)
        
        # # Apply rotation matrices to input vectors
        # # x_flat shape: (batch_size, seq_len, d_k)
        # # rotation_matrices shape: (batch_size, seq_len, d_k, d_k)
        # # We want to multiply each vector by its corresponding rotation matrix
        
        # # Add dimension for matrix multiplication: (batch_size, seq_len, d_k, 1)
        # x_expanded = x_flat.unsqueeze(-1)
        
        # # Perform batched matrix multiplication
        # # (batch_size, seq_len, d_k, d_k) @ (batch_size, seq_len, d_k, 1)
        # # Result shape: (batch_size, seq_len, d_k, 1)
        # rotated = torch.matmul(rotation_matrices, x_expanded)
        
        # # Remove the extra dimension: (batch_size, seq_len, d_k)
        # rotated = rotated.squeeze(-1)
        
        # # Reshape back to original shape
        # rotated = rotated.reshape(original_shape)
        
        # return rotated
        rotated = self.R_M[token_positions] @ x.unsqueeze(-1)
        return rotated.squeeze(-1)
               
        
