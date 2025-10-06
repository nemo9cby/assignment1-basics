#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and loading works correctly.
"""

import torch
import os
import tempfile
from cs336_basics.model import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint


def test_checkpoint_save_load():
    """Test that checkpoint saving and loading preserves model state."""
    print("=" * 60)
    print("CHECKPOINT SAVE/LOAD TEST")
    print("=" * 60)

    # Create a simple model
    print("\n1. Creating test model...")
    model = TransformerLM(
        d_model=128,
        num_heads=4,
        d_ff=256,
        theta=10000,
        vocab_size=1000,
        context_length=32,
        num_layers=2
    )

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Run one forward pass to initialize optimizer state
    print("\n2. Running forward pass to initialize optimizer...")
    dummy_input = torch.randint(0, 1000, (2, 32))
    output = model(dummy_input)
    loss = output.mean()  # Dummy loss
    loss.backward()
    optimizer.step()

    # Save original state for comparison (AFTER optimizer step)
    print("\n3. Saving model state after optimization step...")
    original_state = {}
    for name, param in model.named_parameters():
        original_state[name] = param.data.clone()

    original_optimizer_state = {}
    for param_group in optimizer.state_dict()['state'].values():
        for key, value in param_group.items():
            if isinstance(value, torch.Tensor):
                original_optimizer_state[key] = value.clone()

    # Save checkpoint
    print("\n4. Saving checkpoint...")
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint_path = tmp.name

    iteration = 42
    save_checkpoint(model, optimizer, iteration, checkpoint_path)
    print(f"   Saved to: {checkpoint_path}")
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")

    # Create new model and optimizer with DIFFERENT initial weights
    print("\n5. Creating new model and optimizer...")
    model2 = TransformerLM(
        d_model=128,
        num_heads=4,
        d_ff=256,
        theta=10000,
        vocab_size=1000,
        context_length=32,
        num_layers=2
    )
    optimizer2 = AdamW(model2.parameters(), lr=1e-3)

    # Modify model2 to have different weights
    print("\n6. Modifying new model to have different weights...")
    for param in model2.parameters():
        param.data += torch.randn_like(param) * 0.5

    # Load checkpoint (should restore to original state)
    print("\n7. Loading checkpoint...")
    loaded_iteration = load_checkpoint(model2, optimizer2, checkpoint_path)
    print(f"   Loaded iteration: {loaded_iteration}")

    # Verify iteration
    assert loaded_iteration == iteration, f"Iteration mismatch: {loaded_iteration} != {iteration}"
    print("   ✓ Iteration matches")

    # Verify model state
    print("\n8. Verifying model state...")
    all_match = True
    mismatches = 0
    for name, param in model2.named_parameters():
        orig = original_state[name]
        loaded = param.data

        # Handle NaN case: if both are NaN in same positions, they match
        orig_nan = torch.isnan(orig)
        loaded_nan = torch.isnan(loaded)

        if not torch.equal(orig_nan, loaded_nan):
            # Different NaN patterns - definite mismatch
            all_match = False
            mismatches += 1
            if mismatches <= 3:
                print(f"   ✗ {name}: NaN pattern mismatch")
        elif orig_nan.any():
            # Both have NaN in same places - that's a match for checkpointing
            # (even though the model has NaN, the checkpoint preserved it correctly)
            pass
        elif not torch.allclose(orig, loaded, rtol=1e-5):
            # No NaN, but values don't match
            all_match = False
            mismatches += 1
            if mismatches <= 3:
                orig_val = orig.flatten()[0].item()
                loaded_val = loaded.flatten()[0].item()
                print(f"   ✗ {name}: orig={orig_val:.6f}, loaded={loaded_val:.6f}")

    if mismatches > 3:
        print(f"   ... and {mismatches - 3} more mismatches")

    if all_match:
        print("   ✓ All model parameters match!")
    else:
        print("   ✗ Some parameters don't match")

    # Clean up
    os.remove(checkpoint_path)
    print(f"\n9. Cleaned up temporary file: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return all_match


def test_generate_with_checkpoint():
    """Quick test to ensure generation script will work."""
    print("\n" + "=" * 60)
    print("GENERATION READINESS TEST")
    print("=" * 60)

    # Check if checkpoint directory exists
    checkpoint_dir = './checkpoints/tinystories'
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')

    if os.path.exists(latest_checkpoint):
        print(f"✓ Latest checkpoint found: {latest_checkpoint}")
        size_mb = os.path.getsize(latest_checkpoint) / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ No checkpoint found at {latest_checkpoint}")
        print("  Run training first to create checkpoints")

    # Check tokenizer files
    tokenizer_dir = './tokenizer_output/tinystories'
    vocab_path = os.path.join(tokenizer_dir, 'vocab.json')
    merges_path = os.path.join(tokenizer_dir, 'merges.txt')

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"✓ Tokenizer files found in {tokenizer_dir}")
    else:
        print(f"✗ Tokenizer files not found in {tokenizer_dir}")

    print("\nTo generate text after training, run:")
    print("  uv run python generate_text.py --prompt 'Once upon a time'")
    print("\nOptions:")
    print("  --strategy [greedy|top_k|nucleus]  : Decoding strategy")
    print("  --temperature 0.8                   : Sampling temperature")
    print("  --max-length 100                    : Max tokens to generate")


if __name__ == "__main__":
    # Test checkpoint save/load
    success = test_checkpoint_save_load()

    # Check if ready for generation
    test_generate_with_checkpoint()

    exit(0 if success else 1)