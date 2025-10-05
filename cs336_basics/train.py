import torch
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from cs336_basics.model import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.nn_utils import dataloader, cross_entropy, gradient_clipping
from cs336_basics.bpe.train_bpe import train_bpe, save_trained_tokenizer
from cs336_basics.bpe.Tokenizer import Tokenizer
from cs336_basics.memmap_dataloader import MemmapDataLoader, StreamingMemmapDataLoader

def get_or_train_tokenizer(
    train_data_path: str,
    tokenizer_output_dir: str = "./tokenizer_output/tinystories",
    vocab_size: int = 10000,
    special_tokens: list = None
) -> Tokenizer:
    """
    Load existing tokenizer or train a new one if not found.

    Args:
        train_data_path: Path to training data for tokenizer training
        tokenizer_output_dir: Directory to save/load tokenizer files
        vocab_size: Vocabulary size for BPE tokenizer
        special_tokens: List of special tokens (default: ["<|endoftext|>"])

    Returns:
        Trained or loaded Tokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print("\n" + "=" * 60)
    print("TOKENIZER SETUP")
    print("=" * 60)

    # Check if tokenizer already exists
    vocab_path = os.path.join(tokenizer_output_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_output_dir, "merges.txt")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"✓ Found existing tokenizer at {tokenizer_output_dir}")
        print(f"  Loading tokenizer from saved files...")
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=special_tokens
        )
    else:
        print(f"✗ No tokenizer found at {tokenizer_output_dir}")
        print(f"  Training new BPE tokenizer...")
        print(f"  This may take a few minutes...")

        # Train the tokenizer
        vocab, merges = train_bpe(
            input_path=train_data_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

        # Save the trained tokenizer
        save_trained_tokenizer(vocab, merges, output_dir=tokenizer_output_dir)

        # Load the newly trained tokenizer
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=special_tokens
        )
        print(f"✓ Tokenizer training complete!")

    print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    print(f"  Number of merges: {len(tokenizer.merges)}")

    return tokenizer


def prepare_data(
    data_path: str,
    tokenizer: Tokenizer,
    context_length: int,
    use_streaming: bool = False,
    cache_dir: str = "./data_cache"
) -> MemmapDataLoader:
    """
    Create a memory-efficient dataloader for training.

    Args:
        data_path: Path to the text data file
        tokenizer: Tokenizer instance to use
        context_length: Required context length for validation
        use_streaming: If True, use StreamingMemmapDataLoader for immediate training
        cache_dir: Directory to store cached tokenized data

    Returns:
        MemmapDataLoader instance

    Raises:
        ValueError: If not enough tokens for training
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    print(f"Setting up memory-efficient dataloader for {data_path}...")

    # Choose dataloader type
    if use_streaming:
        print("  Using StreamingMemmapDataLoader (tokenize while training)")
        dataloader = StreamingMemmapDataLoader(
            data_path=data_path,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            chunk_size=1_000_000,  # Tokenize 1MB chunks at a time
            prefetch_chunks=10
        )
    else:
        print("  Using MemmapDataLoader (tokenize first, then train)")
        dataloader = MemmapDataLoader(
            data_path=data_path,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            chunk_size=1_000_000  # Tokenize 1MB chunks at a time
        )

    # Get file size for reporting
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Data file size: {file_size_mb:.2f} MB")
    print(f"  Total tokens: {len(dataloader):,}")

    # Verify we have enough data for training
    min_required_tokens = context_length + 1
    if len(dataloader) < min_required_tokens:
        raise ValueError(
            f"Not enough tokens for training! "
            f"Have {len(dataloader)}, need at least {min_required_tokens}"
        )

    print(f"  ✓ Data ready for training (memory-efficient mode)")

    return dataloader


def get_device():
    """
    Get the best available device in order of preference:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Print additional device info
    if device.type == "cuda":
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

    return device


def train(
    model: TransformerLM,
    dataloader_obj: MemmapDataLoader,
    config: dict,
    device: torch.device,
    tokenizer: Tokenizer = None,  # Add tokenizer for testing
    test_mode: bool = False  # Add test mode flag
) -> None:
    """
    Main training loop.

    Args:
        model: TransformerLM model to train
        dataloader_obj: MemmapDataLoader instance for efficient data loading
        config: Dictionary with training configuration
        device: Device to train on
        tokenizer: Optional tokenizer for decoding (used in test mode)
        test_mode: If True, only test dataloader and exit after first batch
    """
    # Move model to device
    model = model.to(device)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel size: {total_params/1e6:.2f}M parameters")

    # Initialize optimizer with smaller learning rate to avoid NaN
    learning_rate = config.get('learning_rate', 3e-4)  # Smaller default LR
    weight_decay = config.get('weight_decay', 0.1)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")

    # Extract training parameters from config
    batch_size = config.get('batch_size', 32)
    context_length = config['context_length']
    max_iter = config.get('max_iter', 100) if not test_mode else 1  # Only 1 iteration in test mode

    print("\n" + "=" * 60)
    print("DATALOADER TEST" if test_mode else "TRAINING")
    print("=" * 60)

    if test_mode:
        print("Running in TEST MODE - will test dataloader and decoding")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Device: {device}")
        print("=" * 60)
    else:
        print(f"Starting training for {max_iter} iterations...")
        print(f"  Batch size: {batch_size}")
        print(f"  Context length: {context_length}")
        print(f"  Device: {device}")

    for iter in range(max_iter):
        # Get a batch of data
        # TODO(human): Replace the old dataloader call with dataloader_obj.get_batch()
        inputs, targets = dataloader_obj.get_batch(
            batch_size=batch_size,
            context_length=context_length,
            device=str(device)
        )

        if test_mode and tokenizer is not None:
            print(f"\n[Batch {iter + 1}] Testing dataloader output...")
            print(f"  Inputs shape: {inputs.shape}")  # Should be [batch_size, context_length]
            print(f"  Targets shape: {targets.shape}")  # Should be [batch_size, context_length]

            # Test a few samples from the batch
            num_samples_to_test = min(3, batch_size)

            for sample_idx in range(num_samples_to_test):
                print(f"\n--- Sample {sample_idx + 1} ---")

                # Get the sample
                input_ids = inputs[sample_idx].cpu().numpy()
                target_ids = targets[sample_idx].cpu().numpy()

                # Print first few token IDs
                print(f"  First 10 input token IDs: {input_ids[:10].tolist()}")
                print(f"  First 10 target token IDs: {target_ids[:10].tolist()}")

                # Decode to text
                input_text = tokenizer.decode(input_ids.tolist())
                target_text = tokenizer.decode(target_ids.tolist())

                # Print decoded text (truncated for readability)
                max_chars = 200
                print(f"\n  Input text (first {max_chars} chars):")
                print(f"    '{input_text[:max_chars]}{'...' if len(input_text) > max_chars else ''}'")

                print(f"\n  Target text (first {max_chars} chars):")
                print(f"    '{target_text[:max_chars]}{'...' if len(target_text) > max_chars else ''}'")

                # Verify that targets are inputs shifted by 1
                print(f"\n  Verification:")
                print(f"    - Input tokens 1-10: {input_ids[1:11].tolist()}")
                print(f"    - Target tokens 0-9: {target_ids[0:10].tolist()}")
                print(f"    - Do they match? {np.array_equal(input_ids[1:], target_ids[:-1])}")

                # Check if the shift is correct
                # In language modeling, typically: targets[i] = inputs[i+1]
                # So the model learns to predict the next token

            print("\n" + "=" * 60)
            print("DATALOADER TEST COMPLETE")
            print("=" * 60)
            print("\nObservations:")
            print("  1. Inputs and targets have the correct shapes")
            print("  2. Targets should be inputs shifted by 1 position")
            print("  3. Decoded text should be readable and match the original data")
            print("  4. Each batch contains random slices from the tokenized data")

            return  # Exit early in test mode

        # Normal training logic (when not in test mode)
        if not test_mode:
            # Debug: Check inputs for NaN/Inf
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"  WARNING: NaN/Inf in inputs at iteration {iter}")
                print(f"    Inputs min: {inputs.min().item()}, max: {inputs.max().item()}")

            # Forward pass
            logits = model(inputs)

            # Debug: Check logits for NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"  WARNING: NaN/Inf in logits at iteration {iter}")
                print(f"    Logits min: {logits.min().item()}, max: {logits.max().item()}")
                # Check model parameters
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"    NaN/Inf in parameter: {name}")

            # Reshape for loss computation
            logits_flat = logits.view(-1, config['vocab_size'])
            targets_flat = targets.view(-1)

            # Compute loss
            loss = cross_entropy(logits_flat, targets_flat)

            # Debug: Check loss for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at iteration {iter}")
                print(f"    Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                print(f"    Targets stats: min={targets.min().item()}, max={targets.max().item()}")
                print(f"    Unique targets: {targets.unique()[:10].tolist()}...")

                # Check if targets are valid
                invalid_targets = (targets_flat < 0) | (targets_flat >= config['vocab_size'])
                if invalid_targets.any():
                    print(f"    ERROR: Invalid target indices found!")
                    print(f"    Invalid indices: {targets_flat[invalid_targets][:10].tolist()}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Debug: Check gradients before clipping
            max_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    max_grad = max(max_grad, grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"    WARNING: NaN/Inf in gradients before clipping")

            # Gradient clipping
            gradient_clipping(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Print progress with more detail
            if (iter + 1) % 10 == 0 or iter == 0:
                if torch.isnan(loss):
                    print(f"  Iteration {iter + 1}/{max_iter} - Loss: NaN - Max grad before clip: {max_grad:.4f}")
                else:
                    print(f"  Iteration {iter + 1}/{max_iter} - Loss: {loss.item():.4f} - Max grad before clip: {max_grad:.4f}")


    if not test_mode:
        print("\nTraining completed!")


def main():
    """Main entry point for training."""

    # ==============================================================================
    # CONFIGURATION
    # ==============================================================================

    # Tokenizer configuration
    tokenizer_vocab_size = 10000
    tokenizer_output_dir = "./tokenizer_output/tinystories"

    # Model configuration
    model_config = {
        'vocab_size': tokenizer_vocab_size,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'theta': 10000,
        'num_layers': 4,
        'num_heads': 16,
    }

    # Training configuration
    training_config = {
        'batch_size': 4,  # Small batch for testing
        'max_iter': 100,
        'learning_rate': 3e-4,  # Conservative learning rate
        'weight_decay': 0.1,
        'context_length': model_config['context_length'],
        'vocab_size': model_config['vocab_size'],
    }

    # Data paths
    # For debugging: use environment variable to override dataset
    # import os as os_module
    default_data = "./data/TinyStoriesV2-GPT4-train.txt"
    train_data_path = os.environ.get('TRAIN_DATA_PATH', default_data)

    if train_data_path != default_data:
        print(f"[DEBUG] Using custom data path: {train_data_path}")

    # ==============================================================================
    # SETUP
    # ==============================================================================

    # Get device
    device = get_device()

    # Get or train tokenizer
    tokenizer = get_or_train_tokenizer(
        train_data_path=train_data_path,
        tokenizer_output_dir=tokenizer_output_dir,
        vocab_size=tokenizer_vocab_size,
        special_tokens=["<|endoftext|>"]
    )

    # Prepare data (now returns a MemmapDataLoader)
    dataloader = prepare_data(
        data_path=train_data_path,
        tokenizer=tokenizer,
        context_length=model_config['context_length'],
        use_streaming=False,  # Set to True for very large datasets
        cache_dir="./data_cache"
    )

    # ==============================================================================
    # MODEL CREATION
    # ==============================================================================

    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    # Create model
    model = TransformerLM(
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        theta=model_config['theta'],
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        num_layers=model_config['num_layers']
    )

    print(f"Created TransformerLM with configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Check initial model parameters for NaN/Inf
    print("\nChecking model initialization...")
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  WARNING: NaN/Inf in initial parameter: {name}")
            has_nan = True
        # Check for very large initial values
        if param.abs().max() > 10:
            print(f"  WARNING: Large initial values in {name}: max={param.max().item():.4f}")

    if not has_nan:
        print("  ✓ Model initialization looks good (no NaN/Inf)")
    else:
        print("  ✗ Model has NaN/Inf in initial parameters!")

    # ==============================================================================
    # TRAINING
    # ==============================================================================

    # TEST MODE: Set to True to test dataloader, False for normal training
    TEST_DATALOADER = False

    # Run training
    train(
        model=model,
        dataloader_obj=dataloader,
        config=training_config,
        device=device,
        tokenizer=tokenizer if TEST_DATALOADER else None,  # Pass tokenizer for testing
        test_mode=TEST_DATALOADER  # Enable test mode
    )

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

