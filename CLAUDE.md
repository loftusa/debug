# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a modular framework for testing language models on code understanding tasks. The core purpose is to rapidly create and run experiments testing how well language models can trace through code execution, track variables, and perform logical reasoning.

## Project Structure

The framework has a clear separation between core functionality and experiments:

- `src/debug/`: Core framework with reusable components
  - `core.py`: Configuration classes and parser functions  
  - `runner.py`: ExperimentRunner class for running experiments and visualization
  - `generators.py`: Program generators for different task types
  - `prompts.py`: Prompt templates for various experiment types
- `experiments/`: Individual experiment scripts and results
- `results/`: Experiment outputs with JSON data and visualizations

## Development Commands

### Installation
```bash
uv sync
uv pip install -e .
```

### Running Experiments
```bash
# Run all experiments
cd experiments && ./run_experiments.sh all

# Run specific experiment type
./run_experiments.sh range -m "Qwen/Qwen3-0.6B" -n 50
python _04_boolean_simple.py --num-seqs 100
```

### Dependencies
The project uses `uv` for dependency management. Key dependencies include PyTorch, Transformers, NumPy, Pandas, and Matplotlib.

## Core Architecture

### ExperimentConfig
Central configuration class that defines:
- Experiment name and prompt template with `{code}` placeholder
- Program generator function: `(seq_len, rng) -> (program, expected_answer[, metadata])`
- Answer parser function for model responses
- Models to test and experiment parameters

### ExperimentRunner
Main execution engine with features:
- Model caching for interactive use (`preload_models()`, `unload_model()`)
- Batch inference with configurable batch sizes
- Automatic result saving to timestamped directories
- Built-in visualization (`plot()`, `analyze_errors()`)
- Memory management for GPU resources

### Program Generators
Functions that create test programs for different tasks:
- Variable tracking (`make_range_program`, `make_variable_increments`)
- Variable binding with referential depth (`make_variable_binding_program`)
- Exception handling (`make_exception_program`)
- Counterfactual pairs (`make_counterfactual_pair`)

### Results Structure
All experiments save to `results/debug_experiments/{experiment_name}_{timestamp}/`:
- `results.json`: Detailed per-sample results
- `summary.json`: Aggregated statistics
- `accuracy_plot.png`: Auto-generated visualizations

## Key Patterns

### Quick Experiment Pattern
```python
from debug import quick_experiment, ExperimentRunner, generators, prompts

config = quick_experiment(
    name="experiment_name",
    prompt_template=prompts.TEMPLATE_NAME,
    program_generator=generators.generator_function,
    models=["model_id"],
    num_seqs=10,
    seq_lens=[2, 3, 4, 5]
)

runner = ExperimentRunner()
result = runner.run(config)
runner.plot("experiment_name")
```

### Interactive Model Caching
```python
runner = ExperimentRunner()
runner.preload_models(["model1", "model2"])  # Load once
# Run multiple experiments using cached models
runner.unload_all_models()  # Clean up GPU memory
```

### Generator Functions
Generator functions must return `(program_str, expected_answer)` or `(program_str, expected_answer, metadata_dict)`. The runner handles both formats automatically.

### Memory Management
Use `no_cache=True` parameter in `runner.run()` to unload models after each use for memory-constrained environments.