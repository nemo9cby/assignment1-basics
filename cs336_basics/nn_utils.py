"""Neural network utility functions."""

import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np
import numpy.typing as npt
from math import cos, pi


def gradient_clipping(params, max_norm, epsilon=1e-6):
    total_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None)
    )

    if total_norm > max_norm:
    # Scale factor to bring norm down to max_norm
        scale_factor = max_norm / float(total_norm + epsilon)
    
    # Apply to ALL parameters at once (vectorized)
    for param in params:
        if param.grad is not None:
            param.grad.mul_(scale_factor)  # In-place multiplication


        


def cosine_lr_schedule_with_warmup(cur_step: int, a_max: float, a_min: float, t_w: int, t_c: int):

    a_t = 0
    if cur_step < t_w:
        a_t =  (a_max * cur_step) / t_w
    elif  cur_step <= t_c:
        a_t = a_min + 0.5*(1 + cos(pi*float(cur_step-t_w)/(t_c-t_w)))*(a_max - a_min)
    else:
        a_t = a_min
    
    return a_t

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
    x_max = torch.max(inputs, dim=-1, keepdim=True)[0]
    x_exp = torch.exp(inputs - x_max)
    x_sum = torch.sum(x_exp, dim=-1, keepdim=True)
    log_x_sum = x_sum.log()
    log_logits = inputs - x_max - log_x_sum
    target_log_probs = torch.gather(log_logits, dim=1, index=targets.unsqueeze(1)).squeeze(1)
    loss = -target_log_probs.mean()
    return loss

    
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


def dataloader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    
    pass