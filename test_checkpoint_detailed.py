#!/usr/bin/env python3
"""
Detailed test of checkpoint loading.
"""

import torch
import tempfile
import os
from cs336_basics.model import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint

# Create model 1
print("Creating model 1...")
model1 = TransformerLM(
    d_model=128, num_heads=4, d_ff=256, theta=10000,
    vocab_size=1000, context_length=32, num_layers=2
)
optimizer1 = AdamW(model1.parameters())

# Get a specific parameter value before saving
param_name = 'embeddings.weights'
original_value = model1.state_dict()[param_name][0, 0].item()
print(f"Original {param_name}[0,0] = {original_value:.6f}")

# Save checkpoint
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
    checkpoint_path = tmp.name

save_checkpoint(model1, optimizer1, 42, checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")

# Create model 2 with same architecture
print("\nCreating model 2...")
model2 = TransformerLM(
    d_model=128, num_heads=4, d_ff=256, theta=10000,
    vocab_size=1000, context_length=32, num_layers=2
)
optimizer2 = AdamW(model2.parameters())

# Check value before loading
before_load = model2.state_dict()[param_name][0, 0].item()
print(f"Model2 {param_name}[0,0] before load = {before_load:.6f}")

# Load checkpoint
print("\nLoading checkpoint...")
iteration = load_checkpoint(model2, optimizer2, checkpoint_path)
print(f"Loaded iteration {iteration}")

# Check value after loading
after_load = model2.state_dict()[param_name][0, 0].item()
print(f"Model2 {param_name}[0,0] after load = {after_load:.6f}")

# Compare
if abs(original_value - after_load) < 1e-6:
    print("✓ Values match! Checkpoint loading works.")
else:
    print(f"✗ Values don't match: {original_value:.6f} != {after_load:.6f}")

    # Let's check what's in the checkpoint file
    print("\nInspecting checkpoint file...")
    checkpoint = torch.load(checkpoint_path)
    print(f"Checkpoint keys: {checkpoint.keys()}")

    if 'model_state_dict' in checkpoint:
        saved_value = checkpoint['model_state_dict'][param_name][0, 0].item()
        print(f"Value in checkpoint file: {saved_value:.6f}")

        # Try loading manually
        print("\nTrying manual load...")
        model2.load_state_dict(checkpoint['model_state_dict'])
        manual_load = model2.state_dict()[param_name][0, 0].item()
        print(f"After manual load: {manual_load:.6f}")

# Clean up
os.remove(checkpoint_path)