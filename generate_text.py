#!/usr/bin/env python3
"""
Text generation script using trained checkpoint.
Implements decoding strategies for language models.
"""

import torch
import torch.nn.functional as F
import os
import argparse
from pathlib import Path
import numpy as np

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import load_checkpoint
from cs336_basics.AdamW import AdamW
from cs336_basics.bpe.Tokenizer import Tokenizer


def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens whose
    cumulative probability exceeds p.

    Args:
        logits: Logits for next token (vocab_size,)
        p: Cumulative probability threshold
        temperature: Temperature for sampling

    Returns:
        Sampled token index
    """
    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff index where cumulative probability exceeds p
    cutoff_index = torch.searchsorted(cumulative_probs, p) + 1

    # Keep only top-p tokens
    sorted_probs = sorted_probs[:cutoff_index]
    sorted_indices = sorted_indices[:cutoff_index]

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample from the filtered distribution
    sampled_idx = torch.multinomial(sorted_probs, 1)
    sampled_token = sorted_indices[sampled_idx]

    return sampled_token.item()


def top_k_sampling(logits, k=50, temperature=1.0):
    """
    Top-k sampling: sample from the top k most probable tokens.

    Args:
        logits: Logits for next token (vocab_size,)
        k: Number of top tokens to consider
        temperature: Temperature for sampling

    Returns:
        Sampled token index
    """
    # Apply temperature
    logits = logits / temperature

    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

    # Apply softmax to top-k values
    top_k_probs = F.softmax(top_k_values, dim=-1)

    # Sample from top-k distribution
    sampled_idx = torch.multinomial(top_k_probs, 1)
    sampled_token = top_k_indices[sampled_idx]

    return sampled_token.item()


def greedy_decoding(logits):
    """
    Greedy decoding: always pick the most probable token.

    Args:
        logits: Logits for next token (vocab_size,)

    Returns:
        Most probable token index
    """
    return torch.argmax(logits, dim=-1).item()


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    decoding_strategy='nucleus',
    device='cpu'
):
    """
    Generate text from a prompt using the specified decoding strategy.

    Args:
        model: Trained TransformerLM model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to continue from
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: k value for top-k sampling
        top_p: p value for nucleus sampling
        decoding_strategy: 'greedy', 'top_k', or 'nucleus'
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()
    model = model.to(device)

    # Encode the prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Start with a random token if no prompt
        input_ids = torch.randint(0, len(tokenizer.vocab), (1, 1), device=device)

    print(f"Starting generation with {decoding_strategy} decoding...")
    print(f"Prompt: '{prompt}'")
    print(f"Initial tokens: {input_ids.tolist()[0][:10]}...")
    print("-" * 60)

    generated_tokens = input_ids[0].tolist()

    for _ in range(max_length):
        # Ensure we don't exceed model's context length
        current_input = input_ids
        if input_ids.shape[1] > model.context_length:
            current_input = input_ids[:, -model.context_length:]

        # Get model predictions
        logits = model(current_input)
        next_token_logits = logits[0, -1, :]  # Get logits for next token

        # Apply decoding strategy
        if decoding_strategy == 'greedy':
            next_token = greedy_decoding(next_token_logits)
        elif decoding_strategy == 'top_k':
            next_token = top_k_sampling(next_token_logits, k=top_k, temperature=temperature)
        elif decoding_strategy == 'nucleus':
            next_token = nucleus_sampling(next_token_logits, p=top_p, temperature=temperature)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

        # Add to generated tokens
        generated_tokens.append(next_token)

        # Update input_ids for next iteration
        next_token_tensor = torch.tensor([[next_token]], device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

        # Check for end-of-text token (if defined)
        if hasattr(tokenizer, 'special_tokens') and next_token in tokenizer.special_tokens.values():
            print("\n[Reached end-of-text token]")
            break

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/tinystories/checkpoint_latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer-dir', type=str, default='./tokenizer_output/tinystories',
                        help='Directory containing tokenizer files')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt to continue from')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='k value for top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='p value for nucleus sampling')
    parser.add_argument('--strategy', type=str, default='nucleus',
                        choices=['greedy', 'top_k', 'nucleus'],
                        help='Decoding strategy to use')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu, cuda, mps)')

    args = parser.parse_args()

    print("=" * 60)
    print("TEXT GENERATION WITH TRAINED MODEL")
    print("=" * 60)

    # Load tokenizer
    print(f"\n1. Loading tokenizer from {args.tokenizer_dir}...")
    vocab_path = os.path.join(args.tokenizer_dir, 'vocab.json')
    merges_path = os.path.join(args.tokenizer_dir, 'merges.txt')
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")

    # Create model with same configuration as training
    print("\n2. Creating model...")
    model_config = {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'theta': 10000,
        'num_layers': 4,
        'num_heads': 16,
    }

    model = TransformerLM(
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        theta=model_config['theta'],
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        num_layers=model_config['num_layers']
    )

    # Create dummy optimizer (needed for checkpoint loading)
    optimizer = AdamW(model.parameters())

    # Load checkpoint
    print(f"\n3. Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"   ERROR: Checkpoint not found at {args.checkpoint}")
        print("   Please train the model first using train.py")
        return

    iteration = load_checkpoint(model, optimizer, args.checkpoint)
    print(f"   Loaded checkpoint from iteration {iteration}")

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n4. Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("\n4. Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("\n4. Using CPU device")

    # Generate text
    print(f"\n5. Generating text...")
    print(f"   Strategy: {args.strategy}")
    print(f"   Temperature: {args.temperature}")
    if args.strategy == 'top_k':
        print(f"   Top-k: {args.top_k}")
    elif args.strategy == 'nucleus':
        print(f"   Top-p: {args.top_p}")
    print()

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        decoding_strategy=args.strategy,
        device=device
    )

    # Display results
    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()