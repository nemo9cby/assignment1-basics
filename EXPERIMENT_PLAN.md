# CS336 Section 7 - Experimentation Framework Plan

## Overview
This document outlines the comprehensive experimentation framework designed for Section 7 of CS336 Assignment 1. The framework enables smooth hyperparameter tuning, systematic ablations, and efficient experiment tracking for training models on TinyStories and OpenWebText datasets.

## Key Requirements from Section 7
- Train models on TinyStories and OpenWebText datasets
- Perform hyperparameter tuning (learning rate, batch size, etc.)
- Conduct architecture ablations:
  - Normalization: RMSNorm vs LayerNorm
  - Position embeddings: RoPE on/off
  - Activation functions: SwiGLU vs SiLU
- Generate text samples and evaluate perplexity
- Track and compare multiple experiments efficiently

## Framework Components

### 1. Configuration Management System (`cs336_basics/config.py`)
- **YAML-based config files** for easy editing without code changes
- **Hierarchical configs** with inheritance (base → dataset-specific → experiment-specific)
- **Command-line override** capability for quick parameter sweeps
- **Config validation** and auto-documentation
- **Example usage**: `python train_experiment.py --config configs/tinystories.yaml --lr 1e-4`

### 2. Enhanced Training Script (`cs336_basics/train_experiment.py`)
- Builds on existing `train.py` with experiment-focused features
- **Automatic experiment naming** with timestamps and key hyperparameters
- **Resume capability** from any checkpoint
- **Multi-GPU support** detection and setup
- **Real-time metric visualization**
- **Graceful interruption handling** with checkpoint saving

### 3. Experiment Tracker (`cs336_basics/experiment_tracker.py`)
- **Weights & Biases integration** for cloud tracking
- **Local CSV/JSON logging** as backup
- **Automatic metric comparison** across runs
- **Hyperparameter importance analysis**
- **Best model tracking** and automatic saving
- **Git commit tracking** for reproducibility

### 4. Ablation Runner (`cs336_basics/ablation_runner.py`)
- **Systematic ablation studies** with single command
- **Parallel experiment execution**
- **Component swapping**:
  - RMSNorm ↔ LayerNorm
  - SwiGLU ↔ SiLU
  - RoPE on/off
- **Automatic report generation**
- **Statistical significance testing**

### 5. Results Analysis Tools (`cs336_basics/analyze_results.py`)
- **Compare multiple runs** side-by-side
- **Generate performance plots**:
  - Loss curves
  - Learning rate schedules
  - Gradient norms
  - Perplexity trends
- **Calculate statistical significance**
- **Export tables** for papers/reports
- **Interactive dashboard** for exploration

### 6. Config Templates (`configs/`)

#### Base Configurations
- `base.yaml` - Default hyperparameters shared across all experiments
- `tinystories_small.yaml` - Quick iteration config (reduced size for testing)
- `tinystories_full.yaml` - Section 7.2 recommended settings
- `owt_small.yaml` - OpenWebText test config
- `owt_full.yaml` - Full OpenWebText training

#### Ablation Configurations (`configs/ablations/`)
- `no_rope.yaml` - Disable rotary position embeddings
- `layernorm.yaml` - Use LayerNorm instead of RMSNorm
- `silu.yaml` - Use SiLU instead of SwiGLU
- `combined_ablations.yaml` - Test multiple changes

### 7. Utility Scripts

#### `run_experiment.sh`
Wrapper script with common configurations:
```bash
# Quick test run
./run_experiment.sh test

# Full TinyStories training
./run_experiment.sh tinystories

# OpenWebText with custom learning rate
./run_experiment.sh owt --lr 3e-4
```

#### `sweep_hyperparams.py`
Grid/random search over hyperparameter space:
```python
# Define search space
search_space = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'weight_decay': [0.01, 0.1]
}
```

#### `compare_runs.py`
Interactive comparison tool:
```bash
# Compare two specific runs
python compare_runs.py --runs exp1,exp2

# Compare all runs with same base config
python compare_runs.py --filter config=tinystories
```

#### `generate_report.py`
Create markdown/PDF reports:
```bash
# Generate comprehensive report
python generate_report.py --format md --output results_report.md

# Create PDF with plots
python generate_report.py --format pdf --include-plots
```

## Key Features

### One-Line Experiments
```bash
# Basic experiment
python train_experiment.py --config configs/tinystories_full.yaml

# With overrides
python train_experiment.py --config configs/base.yaml --lr 1e-4 --batch_size 64 --name "lr_test"

# Resume from checkpoint
python train_experiment.py --resume checkpoints/exp_001/latest.pt
```

### Live Monitoring
- Real-time loss plots in terminal
- Gradient norm tracking
- Learning rate schedule visualization
- Memory usage monitoring
- ETA and iteration speed

### Automatic Checkpointing
- Save best models based on validation loss
- Regular checkpoint intervals
- Resume from any failure point
- Checkpoint averaging for ensemble

### Reproducibility
- Automatic seed management
- Config versioning and storage
- Git commit tracking
- Environment snapshot
- Exact command logging

### Parallel Sweeps
```bash
# Run multiple experiments simultaneously
python sweep_hyperparams.py --parallel 4 --gpu-per-exp 1

# Queue experiments with resource management
python experiment_queue.py --add "configs/exp1.yaml"
python experiment_queue.py --add "configs/exp2.yaml"
python experiment_queue.py --start-workers 2
```

### Smart Scheduling
- GPU resource management
- Experiment queueing
- Priority-based scheduling
- Automatic retry on failure

## Workflow Example

### 1. Initial Setup
```bash
# Install dependencies
pip install wandb tensorboard matplotlib pandas

# Initialize wandb (optional)
wandb login

# Create base configs
python setup_configs.py
```

### 2. Quick Iteration
```bash
# Test on small subset
python train_experiment.py --config configs/tinystories_small.yaml --max_iter 100

# Check results
python analyze_results.py --last
```

### 3. Full Training
```bash
# Launch main experiment
python train_experiment.py --config configs/tinystories_full.yaml --name "baseline"

# Monitor in real-time
python monitor.py --run baseline

# Or use wandb dashboard
wandb dashboard
```

### 4. Hyperparameter Search
```bash
# Define search in YAML
cat > configs/sweep.yaml << EOF
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
  batch_size:
    values: [32, 64, 128]
EOF

# Run sweep
python sweep_hyperparams.py --config configs/sweep.yaml --count 20
```

### 5. Ablation Study
```bash
# Run all ablations
python ablation_runner.py --baseline configs/tinystories_full.yaml --components all

# View results
python analyze_results.py --ablation-report
```

### 6. Generate Report
```bash
# Create comprehensive analysis
python generate_report.py \
  --experiments "baseline,ablation_*" \
  --metrics "train_loss,val_loss,perplexity" \
  --format md \
  --output section7_results.md
```

## Expected Outcomes

### Improved Workflow Efficiency
- **10x faster** experiment iteration
- **Zero** manual tracking required
- **Instant** comparison of results
- **Automatic** best model selection

### Better Experiment Management
- All experiments tracked and searchable
- No lost results or forgotten configurations
- Easy reproduction of any past run
- Clear documentation of what was tried

### Publication-Ready Results
- Professional plots and tables
- Statistical significance testing
- Comprehensive ablation analysis
- Reproducible findings

## Implementation Priority

1. **Phase 1** (Immediate Value):
   - Config management system
   - Enhanced training script with logging
   - Basic W&B integration

2. **Phase 2** (Core Features):
   - Experiment tracker with comparisons
   - Ablation runner
   - Result analysis tools

3. **Phase 3** (Advanced):
   - Hyperparameter sweep automation
   - Parallel execution
   - Interactive dashboard

## Next Steps

1. Create the configuration management system
2. Enhance the training script with experiment tracking
3. Set up Weights & Biases integration
4. Implement the ablation runner
5. Build analysis and visualization tools

This framework will transform your experimentation workflow from manual and error-prone to automated and systematic, enabling you to focus on insights rather than logistics.