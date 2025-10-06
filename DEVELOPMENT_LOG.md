# Development Log

This file tracks development progress, bug fixes, and experiment work for CS336 Assignment 1.

---

## [2025-01-05] - Section 7 Experiment Planning Session

### Context
Entering Section 7 of the assignment, which requires extensive hyperparameter experiments on TinyStories and OpenWebText datasets. Previous work included implementing the training loop, fixing critical tokenizer bugs, and establishing a working checkpoint system.

### Added
- **EXPERIMENT_PLAN.md** - Comprehensive documentation for Section 7 experimentation framework including:
  - Configuration management system design
  - Experiment tracking infrastructure
  - Ablation runner specifications
  - Results analysis tools
  - Workflow examples and best practices

### Planning & Design
- **Experiment Framework Architecture** - Designed multi-component system for smooth experimentation:
  - YAML-based hierarchical configuration system
  - Enhanced training script with experiment tracking
  - Weights & Biases integration for cloud monitoring
  - Automated ablation studies (RMSNorm vs LayerNorm, SwiGLU vs SiLU, RoPE on/off)
  - Parallel hyperparameter sweep capabilities
  - Statistical analysis and report generation tools

### Analyzed
- **Current Codebase State** for Section 7 readiness:
  - ✅ Training loop implemented and working (lines 290-308 in train.py)
  - ✅ Checkpoint system functional
  - ✅ Memory-efficient dataloader (MemmapDataLoader) ready
  - ✅ Model architecture supports required ablations
  - ✅ Parameter count verified (17.58M matches assignment spec)
  - 🔄 Needs: Systematic experiment tracking and comparison tools

### Key Decisions
1. **Phased Implementation Approach**:
   - Phase 1: Config management + basic tracking (immediate value)
   - Phase 2: Full experiment tracking + ablation automation
   - Phase 3: Advanced features (parallel sweeps, dashboards)

2. **Tool Choices**:
   - Weights & Biases for cloud tracking (with local CSV backup)
   - YAML for configuration (human-readable, git-friendly)
   - Markdown for reports (easy to version control)

### Previous Session Summary (Reference)
For context, the work leading up to Section 7 included:

#### Major Bug Fixes
- **Tokenizer Loading Bug** - Fixed critical vocab.json parsing issue where token_id and token_str were swapped
- **NaN Loss Issue** - Resolved by reducing learning rate from 1e-3 to 3e-4
- **Debugger Freezing** - Worked around debugpy module execution issue by using direct file approach

#### Infrastructure Improvements
- **Memory-Efficient Dataloader** - Implemented MemmapDataLoader reducing memory usage by 82.6%
- **Checkpoint System** - Integrated saving/loading into training loop
- **Text Generation** - Created generation script with nucleus sampling, top-k, and temperature control
- **Parameter Calculation** - Verified model has 17.58M parameters (excluding input embedding)

### Next Steps
1. Implement configuration management system (cs336_basics/config.py)
2. Create enhanced training script with experiment tracking
3. Set up Weights & Biases integration
4. Build ablation runner for systematic studies
5. Develop comparison and visualization tools

### Notes
- Section 7 focuses on extensive experiments requiring smooth workflow for:
  - Training on TinyStories and OpenWebText
  - Hyperparameter tuning (learning rate, batch size)
  - Architecture ablations (normalization, position embeddings, activation functions)
  - Perplexity evaluation and text generation
  - Leaderboard submission

### Commands for Reference
```bash
# Current training command
python cs336_basics/train.py

# Planned experiment command (after implementation)
python train_experiment.py --config configs/tinystories_full.yaml --name "baseline"

# Planned ablation command
python ablation_runner.py --baseline configs/tinystories_full.yaml --components all

# Planned analysis command
python analyze_results.py --experiments "baseline,ablation_*" --format md
```

### Metrics & Performance
- Model Parameters: 17.58M (without input embedding, matching assignment spec)
- Stable Learning Rate: 3e-4 (reduced from 1e-3 to prevent NaN)
- Memory Efficiency: 82.6% reduction with MemmapDataLoader
- Current Config: 4 layers, 16 heads, d_model=512, context_length=256

---

## Future Session Template

### [Date] - Session Title

### Context
Brief description of what stage of the assignment and what needs to be done.

### Added
- New files and features

### Changed
- Modifications to existing code

### Fixed
- Bug fixes with descriptions

### Experiments
- Config changes and results
- Performance metrics
- Issues encountered

### Next Steps
- Immediate tasks
- Future work items

---