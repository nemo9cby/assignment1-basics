#!/usr/bin/env python3
"""
Learning rate schedule test for training loop.
Tests cosine LR schedule with warmup and its effect on training dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from cs336_basics.model.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import (
    cross_entropy, gradient_clipping,
    cosine_lr_schedule_with_warmup, dataloader
)
import matplotlib.pyplot as plt
from typing import List, Tuple

def test_lr_schedule_behavior():
    """Test that the learning rate schedule function works correctly."""
    print("=" * 60)
    print("LEARNING RATE SCHEDULE FUNCTION TEST")
    print("=" * 60)

    # Schedule parameters
    a_max = 1e-3  # Maximum learning rate
    a_min = 1e-5  # Minimum learning rate
    t_w = 100     # Warmup steps
    t_c = 1000    # Cosine cycle steps

    print(f"\nSchedule Parameters:")
    print(f"  Max LR (a_max): {a_max}")
    print(f"  Min LR (a_min): {a_min}")
    print(f"  Warmup steps (t_w): {t_w}")
    print(f"  Cosine cycle steps (t_c): {t_c}")

    # Test the schedule
    steps = list(range(0, 1200, 10))
    learning_rates = [cosine_lr_schedule_with_warmup(step, a_max, a_min, t_w, t_c)
                      for step in steps]

    # Validate key points
    print(f"\nKey Points Validation:")

    # Test warmup phase
    lr_at_0 = cosine_lr_schedule_with_warmup(0, a_max, a_min, t_w, t_c)
    lr_at_half_warmup = cosine_lr_schedule_with_warmup(t_w // 2, a_max, a_min, t_w, t_c)
    lr_at_warmup = cosine_lr_schedule_with_warmup(t_w, a_max, a_min, t_w, t_c)

    print(f"  LR at step 0: {lr_at_0:.6f} (should be 0)")
    print(f"  LR at step {t_w//2}: {lr_at_half_warmup:.6f} (should be ~{a_max/2:.6f})")
    print(f"  LR at step {t_w}: {lr_at_warmup:.6f} (should be ~{a_max:.6f})")

    # Test cosine phase
    lr_at_mid_cosine = cosine_lr_schedule_with_warmup((t_w + t_c) // 2, a_max, a_min, t_w, t_c)
    lr_at_end_cosine = cosine_lr_schedule_with_warmup(t_c, a_max, a_min, t_w, t_c)

    print(f"  LR at step {(t_w + t_c)//2}: {lr_at_mid_cosine:.6f} (should be between min and max)")
    print(f"  LR at step {t_c}: {lr_at_end_cosine:.6f} (should be ~{a_min:.6f})")

    # Test post-cosine phase
    lr_after_cosine = cosine_lr_schedule_with_warmup(t_c + 100, a_max, a_min, t_w, t_c)
    print(f"  LR at step {t_c + 100}: {lr_after_cosine:.6f} (should be {a_min:.6f})")

    # Validation checks
    print(f"\nValidation Checks:")

    warmup_correct = abs(lr_at_0) < 1e-8 and abs(lr_at_warmup - a_max) < a_max * 0.1
    cosine_correct = lr_at_mid_cosine < a_max and lr_at_mid_cosine > a_min
    end_correct = abs(lr_at_end_cosine - a_min) < a_min * 0.1
    post_correct = abs(lr_after_cosine - a_min) < 1e-8

    if warmup_correct:
        print("  ✓ Warmup phase correct")
    else:
        print("  ✗ Warmup phase incorrect")

    if cosine_correct:
        print("  ✓ Cosine phase correct")
    else:
        print("  ✗ Cosine phase incorrect")

    if end_correct and post_correct:
        print("  ✓ Final LR correct")
    else:
        print("  ✗ Final LR incorrect")

    # Plot the schedule
    plt.figure(figsize=(12, 6))
    plt.plot(steps, learning_rates, linewidth=2, label='Learning Rate')

    # Mark key phases
    plt.axvline(x=t_w, color='red', linestyle='--', alpha=0.5, label='End of Warmup')
    plt.axvline(x=t_c, color='blue', linestyle='--', alpha=0.5, label='End of Cosine')
    plt.axhline(y=a_max, color='green', linestyle=':', alpha=0.5, label='Max LR')
    plt.axhline(y=a_min, color='orange', linestyle=':', alpha=0.5, label='Min LR')

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Learning Rate Schedule with Warmup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_schedule_visualization.png')
    print(f"\nSchedule plot saved to 'lr_schedule_visualization.png'")

    return steps, learning_rates


def test_lr_schedule_in_training():
    """Test learning rate schedule integration with training loop."""
    print("\n" + "=" * 60)
    print("LEARNING RATE SCHEDULE IN TRAINING TEST")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Model configuration
    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 256,
        'theta': 10000,
        'vocab_size': 100,
        'context_length': 32,
        'num_layers': 2
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(**config).to(device)

    # Create optimizer with initial learning rate
    initial_lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # Schedule parameters
    warmup_steps = 50
    total_steps = 300
    min_lr = 1e-5

    print(f"\nTraining Configuration:")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Min LR: {min_lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")

    # Create dataset
    dataset_size = 10000
    dataset = np.random.randint(0, config['vocab_size'], size=dataset_size)

    # Training with LR schedule
    losses = []
    learning_rates = []
    batch_size = 8

    print(f"\nTraining with LR schedule...")
    print("-" * 40)

    for step in range(total_steps):
        # Update learning rate
        current_lr = cosine_lr_schedule_with_warmup(
            step, initial_lr, min_lr, warmup_steps, total_steps
        )

        # Update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        learning_rates.append(current_lr)

        # Training step
        inputs, targets = dataloader(dataset, batch_size, config['context_length'], device)
        logits = model(inputs)
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")

    print("-" * 40)

    # Analyze results
    print(f"\n" + "=" * 60)
    print("TRAINING DYNAMICS ANALYSIS")
    print("=" * 60)

    # Split results into phases
    warmup_losses = losses[:warmup_steps]
    cosine_losses = losses[warmup_steps:total_steps]

    print(f"\nLoss Statistics by Phase:")
    print(f"  Warmup phase:")
    print(f"    Mean loss: {np.mean(warmup_losses):.4f}")
    print(f"    Std loss:  {np.std(warmup_losses):.4f}")
    print(f"  Cosine phase:")
    print(f"    Mean loss: {np.mean(cosine_losses):.4f}")
    print(f"    Std loss:  {np.std(cosine_losses):.4f}")

    # Check if warmup helps stability
    early_loss_variance = np.var(losses[:20])
    mid_loss_variance = np.var(losses[warmup_steps:warmup_steps+20])

    print(f"\nStability Analysis:")
    print(f"  Early training variance: {early_loss_variance:.6f}")
    print(f"  Post-warmup variance: {mid_loss_variance:.6f}")

    if early_loss_variance < mid_loss_variance * 2:
        print("  ✓ Warmup provides training stability")
    else:
        print("  Note: Warmup effect on stability is minimal on this task")

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Learning rate schedule
    ax1.plot(learning_rates, color='blue', linewidth=2)
    ax1.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='End of Warmup')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Schedule During Training')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Training loss
    ax2.plot(losses, color='green', linewidth=1, alpha=0.7, label='Training Loss')
    # Add smoothed loss
    window = 20
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), smoothed,
                color='darkgreen', linewidth=2, label='Smoothed')
    ax2.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='End of Warmup')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Loss vs Learning Rate (scatter plot)
    ax3.scatter(learning_rates[::5], losses[::5], alpha=0.5, s=20)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss vs Learning Rate Relationship')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lr_schedule_training_analysis.png')
    print(f"\nTraining analysis plots saved to 'lr_schedule_training_analysis.png'")

    return losses, learning_rates


def compare_with_without_schedule():
    """Compare training with and without LR schedule."""
    print("\n" + "=" * 60)
    print("COMPARISON: WITH vs WITHOUT LR SCHEDULE")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 256,
        'theta': 10000,
        'vocab_size': 100,
        'context_length': 32,
        'num_layers': 2
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataset
    dataset_size = 10000
    dataset = np.random.randint(0, config['vocab_size'], size=dataset_size)

    batch_size = 8
    num_steps = 200

    # Train WITHOUT schedule (constant LR)
    print(f"\nTraining WITHOUT schedule (constant LR)...")
    torch.manual_seed(42)
    model1 = TransformerLM(**config).to(device)
    optimizer1 = AdamW(model1.parameters(), lr=1e-3, weight_decay=0.01)

    losses_no_schedule = []
    for step in range(num_steps):
        inputs, targets = dataloader(dataset, batch_size, config['context_length'], device)
        logits = model1(inputs)
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        optimizer1.zero_grad()
        loss.backward()
        gradient_clipping(model1.parameters(), max_norm=1.0)
        optimizer1.step()

        losses_no_schedule.append(loss.item())

    # Train WITH schedule
    print(f"Training WITH schedule (cosine + warmup)...")
    torch.manual_seed(42)
    model2 = TransformerLM(**config).to(device)
    optimizer2 = AdamW(model2.parameters(), lr=1e-3, weight_decay=0.01)

    warmup_steps = 20
    losses_with_schedule = []

    for step in range(num_steps):
        # Apply schedule
        current_lr = cosine_lr_schedule_with_warmup(
            step, 1e-3, 1e-5, warmup_steps, num_steps
        )
        for param_group in optimizer2.param_groups:
            param_group['lr'] = current_lr

        inputs, targets = dataloader(dataset, batch_size, config['context_length'], device)
        logits = model2(inputs)
        logits_flat = logits.view(-1, config['vocab_size'])
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        optimizer2.zero_grad()
        loss.backward()
        gradient_clipping(model2.parameters(), max_norm=1.0)
        optimizer2.step()

        losses_with_schedule.append(loss.item())

    # Compare results
    print(f"\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    final_loss_no_schedule = np.mean(losses_no_schedule[-10:])
    final_loss_with_schedule = np.mean(losses_with_schedule[-10:])

    loss_variance_no_schedule = np.var(losses_no_schedule)
    loss_variance_with_schedule = np.var(losses_with_schedule)

    print(f"\nFinal Loss (avg last 10 steps):")
    print(f"  Without schedule: {final_loss_no_schedule:.4f}")
    print(f"  With schedule:    {final_loss_with_schedule:.4f}")

    print(f"\nLoss Variance (training stability):")
    print(f"  Without schedule: {loss_variance_no_schedule:.6f}")
    print(f"  With schedule:    {loss_variance_with_schedule:.6f}")

    if loss_variance_with_schedule < loss_variance_no_schedule:
        print(f"\n✓ LR schedule improves training stability")
    else:
        print(f"\nNote: LR schedule effect is task-dependent")

    # Plot comparison
    plt.figure(figsize=(12, 6))

    # Smooth the losses for better visualization
    window = 10
    if len(losses_no_schedule) >= window:
        smoothed_no_schedule = np.convolve(losses_no_schedule,
                                          np.ones(window)/window, mode='valid')
        smoothed_with_schedule = np.convolve(losses_with_schedule,
                                            np.ones(window)/window, mode='valid')

        plt.plot(range(window-1, len(losses_no_schedule)), smoothed_no_schedule,
                label='Without Schedule', linewidth=2, color='red')
        plt.plot(range(window-1, len(losses_with_schedule)), smoothed_with_schedule,
                label='With Schedule', linewidth=2, color='blue')
    else:
        plt.plot(losses_no_schedule, label='Without Schedule', linewidth=2, color='red')
        plt.plot(losses_with_schedule, label='With Schedule', linewidth=2, color='blue')

    plt.axvline(x=warmup_steps, color='green', linestyle='--',
                alpha=0.5, label='End of Warmup')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss: With vs Without LR Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lr_schedule_comparison.png')
    print(f"\nComparison plot saved to 'lr_schedule_comparison.png'")

    return losses_no_schedule, losses_with_schedule


if __name__ == "__main__":
    print("LEARNING RATE SCHEDULE TEST SUITE")
    print("=" * 60)

    # Test 1: Verify schedule function
    steps, lrs = test_lr_schedule_behavior()

    # Test 2: Test schedule in training
    losses, learning_rates = test_lr_schedule_in_training()

    # Test 3: Compare with and without schedule
    losses_no_sched, losses_with_sched = compare_with_without_schedule()

    print("\n" + "=" * 60)
    print("ALL LEARNING RATE SCHEDULE TESTS COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - lr_schedule_visualization.png")
    print("  - lr_schedule_training_analysis.png")
    print("  - lr_schedule_comparison.png")