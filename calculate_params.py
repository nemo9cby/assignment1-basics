#!/usr/bin/env python3
"""
Calculate parameter count for TransformerLM model.
Shows breakdown by component.
"""

def calculate_transformer_params(config):
    """Calculate parameters with detailed breakdown."""

    vocab_size = config['vocab_size']
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']

    print("=" * 60)
    print("TRANSFORMER PARAMETER COUNT")
    print("=" * 60)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("PARAMETER BREAKDOWN")
    print("=" * 60)

    total_params = 0

    # 1. Input Embedding
    embed_params = vocab_size * d_model
    print(f"\n1. Input Embedding (vocab_size × d_model):")
    print(f"   {vocab_size:,} × {d_model} = {embed_params:,}")
    total_params += embed_params

    # 2. Per Transformer Layer
    print(f"\n2. Per Transformer Layer:")

    # Multi-head attention
    print("   a) Multi-Head Attention:")
    q_params = d_model * d_model
    k_params = d_model * d_model
    v_params = d_model * d_model
    o_params = d_model * d_model
    mha_params = q_params + k_params + v_params + o_params
    print(f"      Q projection: {d_model} × {d_model} = {q_params:,}")
    print(f"      K projection: {d_model} × {d_model} = {k_params:,}")
    print(f"      V projection: {d_model} × {d_model} = {v_params:,}")
    print(f"      O projection: {d_model} × {d_model} = {o_params:,}")
    print(f"      Subtotal MHA: {mha_params:,}")

    # SwiGLU feedforward
    print("   b) SwiGLU Feedforward:")
    w1_params = d_model * d_ff  # Linear transformation for gate
    w2_params = d_ff * d_model  # Output projection
    w3_params = d_model * d_ff  # Linear transformation for value
    ff_params = w1_params + w2_params + w3_params
    print(f"      w1 (gate): {d_model} × {d_ff} = {w1_params:,}")
    print(f"      w3 (value): {d_model} × {d_ff} = {w3_params:,}")
    print(f"      w2 (output): {d_ff} × {d_model} = {w2_params:,}")
    print(f"      Subtotal FF: {ff_params:,}")

    # RMSNorm
    print("   c) RMSNorm layers:")
    norm1_params = d_model  # After attention
    norm2_params = d_model  # After feedforward
    norm_params = norm1_params + norm2_params
    print(f"      RMSNorm 1: {norm1_params:,}")
    print(f"      RMSNorm 2: {norm2_params:,}")
    print(f"      Subtotal Norm: {norm_params:,}")

    # Total per layer
    layer_params = mha_params + ff_params + norm_params
    print(f"\n   Total per layer: {layer_params:,}")

    # 3. All layers
    all_layers_params = layer_params * num_layers
    print(f"\n3. All {num_layers} Transformer Layers:")
    print(f"   {layer_params:,} × {num_layers} = {all_layers_params:,}")
    total_params += all_layers_params

    # 4. Final RMSNorm
    final_norm_params = d_model
    print(f"\n4. Final RMSNorm: {final_norm_params:,}")
    total_params += final_norm_params

    # 5. Output projection (unembedding)
    output_params = d_model * vocab_size
    print(f"\n5. Output Projection (d_model × vocab_size):")
    print(f"   {d_model} × {vocab_size:,} = {output_params:,}")
    total_params += output_params

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTotal Parameters (all components):")
    print(f"  {total_params:,} ({total_params/1e6:.2f}M)")

    # Without input embedding (as per handout)
    total_without_input_embed = total_params - embed_params
    print(f"\nTotal WITHOUT Input Embedding:")
    print(f"  {total_without_input_embed:,} ({total_without_input_embed/1e6:.2f}M)")
    print(f"  = Layers + Final Norm + Output Projection")
    print(f"  = {all_layers_params:,} + {final_norm_params:,} + {output_params:,}")

    # Without both embeddings
    total_without_both_embeds = total_params - embed_params - output_params
    print(f"\nTotal WITHOUT Both Embeddings:")
    print(f"  {total_without_both_embeds:,} ({total_without_both_embeds/1e6:.2f}M)")
    print(f"  = Just the transformer layers + final norm")

    # Memory usage estimate (float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"\nMemory Usage (float32):")
    print(f"  {memory_mb:.2f} MB for parameters only")
    print(f"  (excluding gradients, optimizer states, activations)")

    return total_params, total_without_input_embed


if __name__ == "__main__":
    # Your current configuration from train.py
    model_config = {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'theta': 10000,
        'num_layers': 4,
        'num_heads': 16,
    }

    total, without_embed = calculate_transformer_params(model_config)

    print("\n" + "=" * 60)
    print("HANDOUT COMPARISON")
    print("=" * 60)
    print(f"\nHandout says: 17M parameters without embedding")
    print(f"Our calculation: {without_embed/1e6:.2f}M parameters without input embedding")
    print(f"Match: {'✓ YES' if abs(without_embed/1e6 - 17) < 1 else '✗ NO'}")

    # Show the formula
    print("\n" + "=" * 60)
    print("PARAMETER FORMULA")
    print("=" * 60)
    print("\nPer transformer layer:")
    print("  params = 4*d_model² + 3*d_model*d_ff + 2*d_model")
    print("         = 4*512² + 3*512*1344 + 2*512")
    print(f"         = {4*512**2 + 3*512*1344 + 2*512:,}")

    print("\nTotal (without input embedding):")
    print("  = num_layers * layer_params + d_model + d_model*vocab_size")
    print(f"  = 4 * {4*512**2 + 3*512*1344 + 2*512:,} + 512 + 512*10000")
    print(f"  = {without_embed:,}")