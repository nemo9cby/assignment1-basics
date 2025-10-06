#!/usr/bin/env python3
"""
Overfitting test for training loop.
This tests whether the model can perfectly memorize a tiny dataset,
which is a crucial sanity check for any training implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from cs336_basics.model.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import cross_entropy
import matplotlib.pyplot as plt

def test_overfitting_tiny_dataset():
    """
    Test that the model can overfit on a single batch.
    This is a critical test - if a model can't overfit a tiny dataset,
    there's definitely something wrong with the training loop.
    """
    print("=" * 60)
    print("OVERFITTING TEST ON TINY DATASET")
    print("=" * 60)

    # Set reproducible seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Very small model for quick overfitting
    config = {
        'd_model': 64,
        'num_heads': 2,
        'd_ff': 128,
        'theta': 10000,
        'vocab_size': 50,
        'context_length': 16,
        'num_layers': 2
    }

    print(f"\nModel Configuration (intentionally small):")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(**config).to(device)

    # Use higher learning rate for faster overfitting
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)  # No weight decay for overfitting

    # Create a tiny fixed batch (same batch every iteration)
    batch_size = 4
    context_length = config['context_length']
    vocab_size = config['vocab_size']

    # Create a simple pattern that's easy to memorize
    # Pattern: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
    pattern = torch.tensor([i % 5 for i in range(context_length + 1)])

    # Create batch by repeating pattern with small variations
    inputs = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)

    for i in range(batch_size):
        offset = i  # Slight offset for each sample
        for j in range(context_length):
            inputs[i, j] = (j + offset) % 5
            targets[i, j] = (j + offset + 1) % 5

    print(f"\nTraining Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Total tokens: {batch_size * context_length}")
    print(f"  Example input sequence:  {inputs[0, :8].tolist()}...")
    print(f"  Example target sequence: {targets[0, :8].tolist()}...")

    # Training
    num_iterations = 500
    losses = []
    accuracies = []

    print(f"\nTraining for {num_iterations} iterations on the SAME batch...")
    print("-" * 40)

    for iteration in range(num_iterations):
        # Forward pass
        logits = model(inputs)

        # Compute loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        # Compute accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            accuracies.append(accuracy)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 50 == 0 or iteration == num_iterations - 1:
            print(f"Iteration {iteration:3d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2%}")

    print("-" * 40)

    # Validation
    print("\n" + "=" * 60)
    print("OVERFITTING VALIDATION RESULTS")
    print("=" * 60)

    # Check 1: Final loss should be very low (near zero)
    final_loss = losses[-1]
    print(f"\n1. Final Loss Check:")
    print(f"   Final loss: {final_loss:.6f}")

    loss_threshold = 0.1  # Should get below this for successful overfitting
    if final_loss < loss_threshold:
        print(f"   ✓ PASSED: Loss below {loss_threshold} (model memorized data)")
    else:
        print(f"   ✗ FAILED: Loss still high (model failed to memorize)")

    # Check 2: Accuracy should be near 100%
    final_accuracy = accuracies[-1]
    print(f"\n2. Final Accuracy Check:")
    print(f"   Final accuracy: {final_accuracy:.2%}")

    accuracy_threshold = 0.95  # Should achieve at least 95% accuracy
    if final_accuracy > accuracy_threshold:
        print(f"   ✓ PASSED: Accuracy above {accuracy_threshold:.0%}")
    else:
        print(f"   ✗ FAILED: Accuracy below {accuracy_threshold:.0%}")

    # Check 3: Loss should decrease monotonically (mostly)
    loss_decreases = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    decrease_ratio = loss_decreases / (len(losses) - 1)

    print(f"\n3. Loss Decrease Pattern:")
    print(f"   Iterations where loss decreased: {loss_decreases}/{len(losses)-1}")
    print(f"   Decrease ratio: {decrease_ratio:.2%}")

    if decrease_ratio > 0.8:  # At least 80% of steps should decrease loss
        print(f"   ✓ PASSED: Loss decreasing consistently")
    else:
        print(f"   ✗ FAILED: Loss not decreasing consistently")

    # Check 4: Test exact predictions on training data
    print(f"\n4. Exact Prediction Test:")
    model.eval()
    with torch.no_grad():
        test_logits = model(inputs)
        test_predictions = test_logits.argmax(dim=-1)

        # Check first sample in detail
        print(f"   Sample 0 predictions:")
        for i in range(min(8, context_length)):
            pred = test_predictions[0, i].item()
            target = targets[0, i].item()
            match = "✓" if pred == target else "✗"
            print(f"     Position {i}: Pred={pred}, Target={target} {match}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss curve
    ax1.plot(losses, label='Training Loss', color='blue')
    ax1.axhline(y=loss_threshold, color='r', linestyle='--',
                label=f'Target: {loss_threshold}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Overfitting Test: Loss Curve')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    ax2.plot(accuracies, label='Training Accuracy', color='green')
    ax2.axhline(y=accuracy_threshold, color='r', linestyle='--',
                label=f'Target: {accuracy_threshold:.0%}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Overfitting Test: Accuracy Curve')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_overfit_test.png')
    print(f"\n5. Plots saved to 'training_overfit_test.png'")

    # Overall result
    all_passed = (final_loss < loss_threshold and
                  final_accuracy > accuracy_threshold and
                  decrease_ratio > 0.8)

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ✓ MODEL CAN OVERFIT (Training loop works!)")
    else:
        print("OVERALL: ✗ MODEL CANNOT OVERFIT (Check training implementation)")
    print("=" * 60)

    return losses, accuracies, all_passed


def test_generalization_gap():
    """
    Test that shows overfitting vs generalization by comparing
    training and validation performance.
    """
    print("\n" + "=" * 60)
    print("GENERALIZATION GAP TEST")
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
    model = TransformerLM(**config).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    # Create train and validation datasets
    def create_sequence_data(size, seed):
        np.random.seed(seed)
        # Create sequences with patterns
        data = []
        for _ in range(size):
            # Pattern: increasing numbers with wrap-around
            start = np.random.randint(0, 10)
            seq = [(start + i) % 20 for i in range(config['context_length'] + 1)]
            data.append(seq)
        return torch.tensor(data, dtype=torch.long, device=device)

    train_data = create_sequence_data(10, seed=42)  # Small training set
    val_data = create_sequence_data(50, seed=123)   # Larger validation set

    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")

    num_epochs = 50
    train_losses = []
    val_losses = []

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for i in range(len(train_data)):
            inputs = train_data[i, :-1].unsqueeze(0)
            targets = train_data[i, 1:].unsqueeze(0)

            logits = model(inputs)
            logits_flat = logits.view(-1, config['vocab_size'])
            targets_flat = targets.view(-1)
            loss = cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for i in range(len(val_data)):
                inputs = val_data[i, :-1].unsqueeze(0)
                targets = val_data[i, 1:].unsqueeze(0)

                logits = model(inputs)
                logits_flat = logits.view(-1, config['vocab_size'])
                targets_flat = targets.view(-1)
                loss = cross_entropy(logits_flat, targets_flat)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_data)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    print("-" * 40)

    # Analysis
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    generalization_gap = final_val_loss - final_train_loss

    print(f"\nGeneralization Analysis:")
    print(f"  Final training loss:   {final_train_loss:.4f}")
    print(f"  Final validation loss: {final_val_loss:.4f}")
    print(f"  Generalization gap:    {generalization_gap:.4f}")

    if generalization_gap > 0.5:
        print("  → Model is OVERFITTING (large gap between train and val)")
    elif generalization_gap < 0.1:
        print("  → Model is NOT overfitting (similar train and val performance)")
    else:
        print("  → Model shows moderate overfitting")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Generalization Gap)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Shade the generalization gap
    plt.fill_between(range(len(train_losses)), train_losses, val_losses,
                     alpha=0.3, color='red', label='Generalization Gap')
    plt.legend()

    plt.savefig('training_generalization_gap.png')
    print(f"\nPlot saved to 'training_generalization_gap.png'")

    return train_losses, val_losses


if __name__ == "__main__":
    # Run overfitting test
    losses, accuracies, success = test_overfitting_tiny_dataset()

    # Run generalization gap test
    print("\n" + "=" * 60 + "\n")
    train_losses, val_losses = test_generalization_gap()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)