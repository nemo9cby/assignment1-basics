#!/usr/bin/env python3
"""
Basic training loop validation test.
This script tests the fundamental components of a training loop including:
- Forward pass
- Loss computation
- Backward pass
- Parameter updates
- Loss decreasing over iterations
"""

import torch
import torch.nn as nn
import numpy as np
from cs336_basics.model.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import cross_entropy, gradient_clipping, dataloader
import matplotlib.pyplot as plt

def test_basic_training_loop():
    """Test that a basic training loop reduces loss on a small dataset."""
    print("=" * 60)
    print("BASIC TRAINING LOOP TEST")
    print("=" * 60)

    # Set reproducible seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Small model configuration for fast testing
    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 256,
        'theta': 10000,
        'vocab_size': 100,
        'context_length': 32,
        'num_layers': 2
    }

    print(f"\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(**config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Create synthetic dataset (repetitive pattern for easy learning)
    dataset_size = 10000
    pattern = np.array([1, 2, 3, 4, 5] * (dataset_size // 5))
    dataset = pattern[:dataset_size]

    # Training parameters
    batch_size = 8
    num_iterations = 100

    losses = []
    print(f"\nTraining for {num_iterations} iterations...")
    print("-" * 40)

    for iteration in range(num_iterations):
        # Get batch
        inputs, targets = dataloader(dataset, batch_size, config['context_length'], device)

        # Forward pass
        logits = model(inputs)  # Shape: [batch_size, context_length, vocab_size]

        # Reshape for loss computation
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)

        # Compute loss
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optional: Apply gradient clipping
        gradient_clipping(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        losses.append(loss.item())

        if iteration % 20 == 0:
            print(f"Iteration {iteration:3d}: Loss = {loss.item():.4f}")

    print("-" * 40)

    # Validation checks
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Check 1: Loss should decrease
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n1. Loss Reduction Check:")
    print(f"   Initial loss (avg first 10): {initial_loss:.4f}")
    print(f"   Final loss (avg last 10):    {final_loss:.4f}")
    print(f"   Reduction:                    {loss_reduction:.1f}%")

    if final_loss < initial_loss:
        print("   ✓ PASSED: Loss decreased during training")
    else:
        print("   ✗ FAILED: Loss did not decrease")

    # Check 2: Loss should be reasonable (not NaN or extreme)
    has_nan = any(np.isnan(loss) for loss in losses)
    has_extreme = any(loss > 100 for loss in losses)

    print(f"\n2. Loss Stability Check:")
    print(f"   Contains NaN:     {has_nan}")
    print(f"   Contains extreme: {has_extreme}")

    if not has_nan and not has_extreme:
        print("   ✓ PASSED: Loss values are stable")
    else:
        print("   ✗ FAILED: Loss values are unstable")

    # Check 3: Gradient flow (check that gradients exist and are reasonable)
    inputs, targets = dataloader(dataset, batch_size, config['context_length'], device)
    logits = model(inputs)
    logits_flat = logits.view(-1, config['vocab_size'])
    targets_flat = targets.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    optimizer.zero_grad()
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    avg_grad_norm = np.mean(grad_norms)
    print(f"\n3. Gradient Flow Check:")
    print(f"   Average gradient norm: {avg_grad_norm:.6f}")
    print(f"   Min gradient norm:     {min(grad_norms):.6f}")
    print(f"   Max gradient norm:     {max(grad_norms):.6f}")

    if avg_grad_norm > 1e-6 and avg_grad_norm < 100:
        print("   ✓ PASSED: Gradients are flowing properly")
    else:
        print("   ✗ FAILED: Gradient flow issues detected")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add smoothed curve
    window_size = 10
    if len(losses) >= window_size:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(losses)), smoothed,
                label=f'Smoothed (window={window_size})', alpha=0.7)
        plt.legend()

    plt.savefig('training_loss_basic.png')
    print(f"\n4. Loss curve saved to 'training_loss_basic.png'")

    # Overall result
    all_passed = (final_loss < initial_loss and
                  not has_nan and not has_extreme and
                  avg_grad_norm > 1e-6 and avg_grad_norm < 100)

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ✓ ALL TESTS PASSED")
    else:
        print("OVERALL: ✗ SOME TESTS FAILED")
    print("=" * 60)

    return losses, all_passed


if __name__ == "__main__":
    losses, success = test_basic_training_loop()