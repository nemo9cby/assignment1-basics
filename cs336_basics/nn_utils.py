"""Neural network utility functions."""

import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np

def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Apply softmax to the given dimension of the input tensor.
    
    Args:
        in_features: Input features to softmax. Shape is arbitrary.
        dim: Dimension of the input to apply softmax to.
        
    Returns:
        Tensor with the same shape as input with softmax applied to the specified dimension.
    """

    x_max = torch.max(in_features, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(in_features - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    softmax_result = x_exp / x_sum
    return softmax_result

    

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], 
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute the average cross-entropy loss across examples.
    
    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size).
        targets: Indices of correct classes of shape (batch_size,).
            Each value must be between 0 and vocab_size - 1.
            
    Returns:
        The average cross-entropy loss across examples.
    """
    raise NotImplementedError

# def scaled_dot_product_attention(q, k, v, mask):
#     q = q.reshape(-1, q.shape[-2], q.shape[-1])
#     k = k.reshape(-1, k.shape[-2], k.shape[-1])
#     v = v.reshape(-1, v.shape[-2], v.shape[-1])
#     results = softmax(torch.tensor.masked_fill(q@k.transpose(-2,-1)/ np.sqrt(q.shape[-1]), mask == False, -float('inf'))) @ v
    

def scaled_dot_product_attention(q, k, v, mask):
    # Reshape inputs
    original_shape = q.shape
    q = q.reshape(-1, q.shape[-2], q.shape[-1])
    k = k.reshape(-1, k.shape[-2], k.shape[-1])
    v = v.reshape(-1, v.shape[-2], v.shape[-1])
    
    # Compute attention scores
    scores = q @ k.transpose(-2, -1) / np.sqrt(q.shape[-1])
    
    # Apply mask (if provided)
    if mask is not None:
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        scores = scores.masked_fill(mask == False, -float('inf'))
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention to values
    output = attention_weights @ v
    
    # Reshape output to match the original input shape
    output = output.reshape(*original_shape[:-2], -1, original_shape[-1])
    return output