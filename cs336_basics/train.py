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
    context_length: int
) -> np.ndarray:
    """
    Load and tokenize text data for training.

    Args:
        data_path: Path to the text data file
        tokenizer: Tokenizer instance to use
        context_length: Required context length for validation

    Returns:
        Numpy array of tokenized data

    Raises:
        ValueError: If not enough tokens for training
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    print(f"Loading and tokenizing data from {data_path}...")

    # Read the training data
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
        # For testing, use only first 1MB of data
        if len(text_data) > 1_000_000:
            print(f"  Original text size: {len(text_data):,} characters")
            text_data = text_data[:1_000_000]
            print(f"  Using first 1,000,000 characters for testing")

    # Check data size
    print(f"  Text data size: {len(text_data):,} characters")

    # Tokenize the entire text
    token_ids = tokenizer.encode(text_data)

    # Convert to numpy array for use with dataloader
    tokenized_data = np.array(token_ids, dtype=np.int64)

    print(f"  Tokenized data size: {len(tokenized_data):,} tokens")
    print(f"  Compression ratio: {len(text_data) / len(tokenized_data):.2f} chars/token")

    # Verify we have enough data for training
    min_required_tokens = context_length + 1
    if len(tokenized_data) < min_required_tokens:
        raise ValueError(
            f"Not enough tokens for training! "
            f"Have {len(tokenized_data)}, need at least {min_required_tokens}"
        )

    print(f"  ✓ Data ready for training")

    return tokenized_data


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
    tokenized_data: np.ndarray,
    config: dict,
    device: torch.device,
    tokenizer: Tokenizer = None,  # Add tokenizer for testing
    test_mode: bool = False  # Add test mode flag
) -> None:
    """
    Main training loop.

    Args:
        model: TransformerLM model to train
        tokenized_data: Numpy array of tokenized training data
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

    # Initialize optimizer
    optimizer = AdamW(model.parameters())

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
        inputs, targets = dataloader(
            dataset=tokenized_data,
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
            # TODO: Add forward pass, loss computation, backward pass, optimizer step
            # This is where you would implement the actual training logic
            # Example structure:
            # logits = model(inputs)
            # logits_flat = logits.view(-1, config['vocab_size'])
            # targets_flat = targets.view(-1)
            # loss = cross_entropy(logits_flat, targets_flat)
            # optimizer.zero_grad()
            # loss.backward()
            # gradient_clipping(model.parameters(), max_norm=1.0)
            # optimizer.step()

            if (iter + 1) % 10 == 0:
                print(f"  Iteration {iter + 1}/{max_iter} completed")

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
        'context_length': model_config['context_length'],
        'vocab_size': model_config['vocab_size'],
    }

    # Data paths
    train_data_path = "./data/TinyStoriesV2-GPT4-train.txt"

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

    # Prepare data
    tokenized_data = prepare_data(
        data_path=train_data_path,
        tokenizer=tokenizer,
        context_length=model_config['context_length']
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

    # ==============================================================================
    # TRAINING
    # ==============================================================================

    # TEST MODE: Set to True to test dataloader, False for normal training
    TEST_DATALOADER = True

    # Run training
    train(
        model=model,
        tokenized_data=tokenized_data,
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

