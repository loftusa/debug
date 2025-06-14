# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The goal of this series of experiments is to understand how how and whether language models track the internal state of variables in programming languages. The long-term goal is to be able to track the internal states of programming languages if they would be run without needing to execute any code. This should help users with debugging and quickly understanding code.

To this end, we first test various models on their ability to accomplish coding tasks. Part of the framework has a core purpose of rapidly creating and running experiments that test how well language models can trace through code execution, track variables, and perform logical reasoning.

We will also include tools built with the `nnsight` python package that allow for interpretability techniques like patching and causal mediation analysis.

## Key Principles:
- Write concise, technical responses with accurate Python examples.
- We are using test-driven development. Build tests first. Before writing code, run the tests and confirm that they fail. After writing code, run the tests again, and make confirm that they pass. Do not modify tests once you are in the stage of writing code. The exception to this is visualization code, for that just make example visualizations that I can look at and see for myself (save them in an image file so that I can look)
- Use object-oriented programming for model architectures and functional programming for data processing pipelines.
- Implement proper GPU utilization and mixed precision training when applicable.
- Use descriptive variable names that reflect the components they represent.
- Follow PEP 8 style guidelines for Python code.
- Follow the `black` formatting style for Python code.
- Use modern python tools like dataclasses and pydantic when appropriate.
- When visualizing data, always use the style of Edward Tufte, using his words as wisdom.
- Important: try to fix things at the cause, not the symptom.
- Be very detailed with summarization and do not miss out things that are important.
- Don’t be helpful, be better
- Don't try to demo python interactive / jupyter notebook files, I'll do that myself
- Ask me clarifying questions on design decisions when appropriate
- When writing any code that uses arrays or tensors, use `jaxtyping` for tensor type/shape typing annotation.


Dependencies:
- torch
- numpy
- transformers
- tqdm (for progress bars)
- wandb (for experiment tracking)
- jaxtyping (for tensor type and shape annotations)
- uv
- scikit-learn
- pytest

### Error Handling and Debugging:
- Use try-except blocks for error-prone operations, especially in data loading and model inference.
- Implement proper logging for training progress and errors.

### Performance Optimization:
- Profile code to identify and optimize bottlenecks, especially in data loading and preprocessing.

## Key Conventions:
0. Thoroughly plan and refine before actually writing code.
1. Begin projects with clear problem definition and dataset analysis.
2. Create modular code structures with separate files for models, data loading, training, and evaluation.
3. Use configuration files (e.g., YAML) for hyperparameters and model settings.
4. Implement proper experiment tracking and model checkpointing.
5. Use version control (e.g., git) for tracking changes in code and configurations.

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
uv run _04_boolean_simple.py --num-seqs 100
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


# Causal Tracing methodology

We will implement a causal tracing experiment. Use the methodology below.

## Counterfactual Construction

1. **Identify the variable chain**: For each program, trace the chain of variable assignments from the queried variable back to its root numerical value
   - Example: For query `#w:` with chain `w = j, j = k, k = 2`, the root value is `2`

2. **Create counterfactual**: Replace the root numerical value with a different number
   - Original: `k = 2` → Counterfactual: `k = 7` 
   - Keep all other assignments identical

## Intervention Procedure

3. **Two intervention targets**:
   - **Residual stream**: The hidden state vectors at each token position going into each layer
   - **Attention head outputs**: Individual head contributions to the residual stream

4. **Token selection strategy**:
   - Target tokens on the right-hand side (RHS) of assignment statements (`=`)
   - Label by referential depth: "Ref Depth 1 RHS" for root value, "Ref Depth 2 RHS" for next level, etc.
   - Also target the query variable token and final colon token

5. **Intervention execution**:
   - Run forward pass on original program → get activation at target position
   - Run forward pass on counterfactual program → get activation at same target position  
   - Replace original activation with counterfactual activation
   - Continue forward pass through remaining layers
   - Measure final logits

## Measurement Metrics

6. **Success metrics**:
   - **Logit difference**: Change in pre-softmax activation for the new number (counterfactual - original), normalized by maximum difference
   - **Success rate**: Percentage of times intervention causes model to predict the counterfactual number as top choice
   - Average across test programs (excluding line-1/line-2 heuristic cases for systematic analysis)

## Key Implementation Details

7. **Filtering**: Focus on programs where correct answer appears after line 2 to isolate systematic binding mechanism from shallow heuristics

8. **Dynamic targeting**: Select token positions based on actual variable chains rather than fixed indices, since programs have different structures

The core insight: If replacing an activation with its counterfactual value causes the model to output the counterfactual answer, that activation causally contributes to tracking the variable binding.

## Tools and Techniques

- `nnsight` should be used for code that directly modifies LLM internals. Documentation can be found at https://nnsight.net/