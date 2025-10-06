#!/usr/bin/env python3
"""
Gradient flow verification test.
This script thoroughly tests gradient propagation through the model,
checking for vanishing/exploding gradients and dead neurons.
"""

import torch
import torch.nn as nn
import numpy as np
from cs336_basics.model.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
import matplotlib.pyplot as plt
from collections import defaultdict

def test_gradient_flow():
    """
    Comprehensive gradient flow test that checks:
    1. Gradients exist for all parameters
    2. No vanishing gradients (too small)
    3. No exploding gradients (too large)
    4. Gradient distribution across layers
    5. Effect of gradient clipping
    """
    print("=" * 60)
    print("GRADIENT FLOW VERIFICATION TEST")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Standard model configuration
    config = {
        'd_model': 256,
        'num_heads': 8,
        'd_ff': 512,
        'theta': 10000,
        'vocab_size': 1000,
        'context_length': 128,
        'num_layers': 4  # Multiple layers to test gradient flow
    }

    print(f"\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(**config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create random input
    batch_size = 16
    inputs = torch.randint(0, config['vocab_size'],
                          (batch_size, config['context_length']), device=device)
    targets = torch.randint(0, config['vocab_size'],
                           (batch_size, config['context_length']), device=device)

    print("\n" + "=" * 60)
    print("TEST 1: BASIC GRADIENT EXISTENCE")
    print("=" * 60)

    # Forward and backward pass
    logits = model(inputs)
    logits_flat = logits.view(-1, config['vocab_size'])
    targets_flat = targets.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    model.zero_grad()
    loss.backward()

    # Check gradient existence
    params_with_grad = 0
    params_without_grad = 0
    zero_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
                if torch.all(param.grad == 0):
                    zero_grad_params.append(name)
            else:
                params_without_grad += 1

    print(f"\nGradient Existence:")
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")
    print(f"  Parameters with zero gradients: {len(zero_grad_params)}")

    if params_without_grad > 0:
        print("  ✗ WARNING: Some parameters have no gradients!")
    elif len(zero_grad_params) > 0:
        print(f"  ✗ WARNING: {len(zero_grad_params)} parameters have all-zero gradients!")
        for param_name in zero_grad_params[:5]:  # Show first 5
            print(f"    - {param_name}")
    else:
        print("  ✓ PASSED: All parameters have non-zero gradients")

    print("\n" + "=" * 60)
    print("TEST 2: GRADIENT MAGNITUDE ANALYSIS")
    print("=" * 60)

    # Analyze gradient magnitudes
    grad_stats = defaultdict(list)
    layer_grad_norms = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            grad_stats['norms'].append(grad_norm)
            grad_stats['means'].append(abs(grad_mean))
            grad_stats['stds'].append(grad_std)

            # Group by layer type
            if 'embeddings' in name:
                layer_type = 'embeddings'
            elif 'transformer_layers.0' in name:
                layer_type = 'layer_0'
            elif 'transformer_layers.1' in name:
                layer_type = 'layer_1'
            elif 'transformer_layers.2' in name:
                layer_type = 'layer_2'
            elif 'transformer_layers.3' in name:
                layer_type = 'layer_3'
            elif 'norm' in name:
                layer_type = 'final_norm'
            elif 'output_embedding' in name:
                layer_type = 'output'
            else:
                layer_type = 'other'

            layer_grad_norms[layer_type].append(grad_norm)

    # Print statistics
    print(f"\nOverall Gradient Statistics:")
    print(f"  Mean gradient norm: {np.mean(grad_stats['norms']):.6f}")
    print(f"  Std gradient norm:  {np.std(grad_stats['norms']):.6f}")
    print(f"  Min gradient norm:  {np.min(grad_stats['norms']):.6f}")
    print(f"  Max gradient norm:  {np.max(grad_stats['norms']):.6f}")

    # Check for vanishing/exploding gradients
    vanishing_threshold = 1e-8
    exploding_threshold = 100
    vanishing_count = sum(1 for norm in grad_stats['norms'] if norm < vanishing_threshold)
    exploding_count = sum(1 for norm in grad_stats['norms'] if norm > exploding_threshold)

    print(f"\nGradient Health Check:")
    print(f"  Vanishing gradients (<{vanishing_threshold}): {vanishing_count}")
    print(f"  Exploding gradients (>{exploding_threshold}): {exploding_count}")

    if vanishing_count > len(grad_stats['norms']) * 0.1:
        print("  ✗ WARNING: >10% parameters have vanishing gradients!")
    elif exploding_count > 0:
        print("  ✗ WARNING: Some parameters have exploding gradients!")
    else:
        print("  ✓ PASSED: Gradient magnitudes are healthy")

    # Print layer-wise statistics
    print(f"\nLayer-wise Gradient Norms:")
    for layer_type in ['embeddings', 'layer_0', 'layer_1', 'layer_2', 'layer_3', 'final_norm', 'output']:
        if layer_type in layer_grad_norms:
            norms = layer_grad_norms[layer_type]
            print(f"  {layer_type:15s}: mean={np.mean(norms):.6f}, std={np.std(norms):.6f}")

    print("\n" + "=" * 60)
    print("TEST 3: GRADIENT CLIPPING EFFECT")
    print("=" * 60)

    # Test gradient clipping
    model.zero_grad()
    loss.backward()

    # Measure before clipping
    total_norm_before = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)
    ).item()

    print(f"\nBefore Gradient Clipping:")
    print(f"  Total gradient norm: {total_norm_before:.6f}")

    # Apply gradient clipping
    max_norm = 1.0
    gradient_clipping(model.parameters(), max_norm=max_norm)

    # Measure after clipping
    total_norm_after = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)
    ).item()

    print(f"\nAfter Gradient Clipping (max_norm={max_norm}):")
    print(f"  Total gradient norm: {total_norm_after:.6f}")
    print(f"  Norm reduction: {total_norm_before - total_norm_after:.6f}")

    if total_norm_before > max_norm and abs(total_norm_after - max_norm) < 0.01:
        print("  ✓ PASSED: Gradient clipping working correctly")
    elif total_norm_before <= max_norm and abs(total_norm_after - total_norm_before) < 0.01:
        print("  ✓ PASSED: Gradient clipping correctly preserves small gradients")
    else:
        print("  ✗ WARNING: Gradient clipping may not be working correctly")

    print("\n" + "=" * 60)
    print("TEST 4: GRADIENT FLOW OVER TRAINING")
    print("=" * 60)

    # Monitor gradient flow over multiple iterations
    optimizer = AdamW(model.parameters(), lr=1e-3)
    num_iterations = 50
    gradient_history = defaultdict(list)

    print(f"\nMonitoring gradients over {num_iterations} iterations...")

    for iteration in range(num_iterations):
        # Random batch
        inputs = torch.randint(0, config['vocab_size'],
                              (batch_size, config['context_length']), device=device)
        targets = torch.randint(0, config['vocab_size'],
                               (batch_size, config['context_length']), device=device)

        # Forward and backward
        logits = model(inputs)
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()

        # Record gradient norms for each layer type
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'transformer_layers.0' in name and 'attention' in name:
                    key = 'Layer 0 Attention'
                elif 'transformer_layers.3' in name and 'attention' in name:
                    key = 'Layer 3 Attention'
                elif 'embeddings' in name:
                    key = 'Embeddings'
                elif 'output_embedding' in name:
                    key = 'Output'
                else:
                    continue

                gradient_history[key].append(param.grad.norm().item())

        optimizer.step()

    # Plot gradient flow over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Flow Over Training Iterations', fontsize=16)

    for idx, (key, values) in enumerate(gradient_history.items()):
        if idx < 4:
            ax = axes[idx // 2, idx % 2]
            ax.plot(values, label=key)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Gradient Norm')
            ax.set_title(key)
            ax.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(range(len(values)), values, 1)
            p = np.poly1d(z)
            ax.plot(range(len(values)), p(range(len(values))),
                   "--", alpha=0.5, label='Trend')
            ax.legend()

    plt.tight_layout()
    plt.savefig('gradient_flow_analysis.png')
    print(f"\nGradient flow plots saved to 'gradient_flow_analysis.png'")

    # Check gradient stability
    print(f"\nGradient Stability Analysis:")
    for key, values in gradient_history.items():
        if len(values) > 0:
            stability = np.std(values) / np.mean(values)  # Coefficient of variation
            print(f"  {key:20s}: CoV = {stability:.3f} {'(stable)' if stability < 1.0 else '(unstable)'}")

    print("\n" + "=" * 60)
    print("TEST 5: DEAD NEURONS CHECK")
    print("=" * 60)

    # Check for dead neurons (neurons with very small gradients consistently)
    dead_neuron_threshold = 1e-10
    num_checks = 10
    dead_neurons = defaultdict(int)

    print(f"\nChecking for dead neurons over {num_checks} random inputs...")

    for _ in range(num_checks):
        inputs = torch.randint(0, config['vocab_size'],
                              (batch_size, config['context_length']), device=device)
        targets = torch.randint(0, config['vocab_size'],
                               (batch_size, config['context_length']), device=device)

        logits = model(inputs)
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for neurons with very small gradients
                if len(param.grad.shape) >= 2:  # Weight matrices
                    neuron_grad_norms = param.grad.norm(dim=1)
                    dead_count = (neuron_grad_norms < dead_neuron_threshold).sum().item()
                    if dead_count > 0:
                        dead_neurons[name] += dead_count

    print("\nDead Neurons Summary:")
    if len(dead_neurons) == 0:
        print("  ✓ PASSED: No dead neurons detected")
    else:
        print("  ✗ WARNING: Dead neurons detected in:")
        for name, count in list(dead_neurons.items())[:5]:  # Show top 5
            print(f"    - {name}: {count} neurons")

    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL GRADIENT FLOW ASSESSMENT")
    print("=" * 60)

    all_checks_passed = (
        params_without_grad == 0 and
        len(zero_grad_params) == 0 and
        vanishing_count <= len(grad_stats['norms']) * 0.1 and
        exploding_count == 0 and
        len(dead_neurons) == 0
    )

    if all_checks_passed:
        print("✓ ALL GRADIENT FLOW TESTS PASSED")
        print("Your training loop has healthy gradient flow!")
    else:
        print("✗ SOME GRADIENT FLOW ISSUES DETECTED")
        print("Review the warnings above and check your implementation")

    return gradient_history, all_checks_passed


if __name__ == "__main__":
    gradient_history, success = test_gradient_flow()