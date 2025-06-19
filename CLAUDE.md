# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The goal of this series of experiments is to understand how and whether language models track the internal state of variables in programming languages. The long-term goal is to be able to track the internal states of programming languages if they would be run without needing to execute any code. This should help users with debugging and quickly understanding code.

To this end, we first test various models on their ability to accomplish coding tasks. Part of the framework has a core purpose of rapidly creating and running experiments that test how well language models can trace through code execution, track variables, and perform logical reasoning.

We will also include tools built with the `nnsight` python package that allow for interpretability techniques like patching and causal mediation analysis. Many of our experiments recreate and extend the methodology from ["Tracing Knowledge in Language Models Back to the Training Data"](https://arxiv.org/abs/2505.20896), particularly their causal tracing approach for understanding variable binding mechanisms in language models.

## Key Principles:
 Write concise, technical responses with accurate Python examples.
- Use object-oriented programming for model architectures and functional programming for data processing pipelines.
- Use concise, descriptive variable names that reflect the components they represent.
- Follow PEP 8 style guidelines for Python code.
- Follow the `black` formatting style for Python code.
- Use useful functions from the standard library when appropriate. You know itertools, functools, pathlib, collections, bisect, shutil, and other tools well; but tend towards simple, easy-to-understand code and don't overengineer.
- When visualizing data, always use the style of Edward Tufte, using his words as wisdom.
- Important: try to fix things at the cause, not the symptom.
- Be very detailed with summarization and do not miss out things that are important.
- Don't be helpful, be better
- Don't try to demo python interactive / jupyter notebook files, I'll do that myself
- Ask me clarifying questions on design decisions when appropriate
- When writing any code that uses arrays or tensors, use `jaxtyping` for tensor type/shape typing annotation.
- Be clear with type annotation in general
- Include a lot of assert statements to make sure the code is doing what you think it's doing.


Dependencies:
- torch
- numpy
- transformers 
- accelerate
- bitsandbytes
- datasets
- huggingface-hub
- matplotlib
- seaborn
- nnsight (for interpretability)
- jaxtyping (for tensor type and shape annotations)
- torchtyping
- einops
- lovely-tensors
- pytest
- ruff (for linting)
- gradio
- jupyter/jupyterlab
- uv (package manager)
- vllm

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
  - `causal_tracing.py`: Causal intervention and patching functionality
  - `causal_experiment_runner.py`: Specialized runner for causal experiments
  - `causal_visualization.py`: Visualization tools for causal analysis
  - `counterfactual.py`: Counterfactual generation utilities
  - `token_analyzer.py`: Token-level analysis tools
- `experiments/`: Individual experiment scripts and results
  - Multiple experiment types: boolean, integer, range tracking, variable binding
  - Causal tracing and patching experiments
  - Shell scripts for batch experiment execution
- `results/`: Experiment outputs with JSON data and visualizations
  - `debug_experiments/`: Core experiment results
  - `full_token_layer_patching/`: Causal intervention results
  - Various dated experiment runs with detailed data

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

# Run causal tracing experiments
uv run _07_binding_patching.py
uv run _08_full_layer_token_patching.py
```

### Linting and Code Quality
```bash
# Run ruff for linting and formatting
ruff check src/
ruff format src/
```

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

Causal tracing experiments save to `results/full_token_layer_patching/{timestamp}/`:
- `intervention_results.json`: Detailed causal intervention results
- `patch_results.json`: Aggregated patching statistics

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

### Causal Tracing Pattern
```python
from debug.causal_tracing import run_causal_tracing_experiment
from debug.causal_experiment_runner import CausalExperimentRunner

# Run causal intervention experiment
runner = CausalExperimentRunner()
results = runner.run_intervention_experiment(
    model_id="model_name",
    programs=program_list,
    intervention_type="residual_stream"  # or "attention_heads"
)
```

### Generator Functions
Generator functions must return `(program_str, expected_answer)` or `(program_str, expected_answer, metadata_dict)`. The runner handles both formats automatically.

### Memory Management
Use `no_cache=True` parameter in `runner.run()` to unload models after each use for memory-constrained environments.


# Causal Tracing Methodology

We implement causal tracing experiments based on the methodology from ["Tracing Knowledge in Language Models Back to the Training Data"](https://arxiv.org/abs/2505.20896). This paper introduces a systematic approach to understanding how language models track variable bindings through causal interventions on internal activations.

## Paper Overview

The referenced paper investigates how language models internally track variable bindings in code by using causal tracing - a technique that patches activations from counterfactual inputs to identify which internal states causally contribute to the model's predictions. Our implementation extends their approach with additional analysis and visualization tools.

## Our Implementation

We will implement a causal tracing experiment using the methodology below.

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

## Experiment Notes
- We are copying experiments from https://arxiv.org/abs/2505.20896

# INFO FOR CONTEXT

### RNG SEED EXPERIMENT
calling `uv run experiments/11_test_rng_seeds.py` returns:

==================================================
🎉 Found a robust program (all models correct)!
RNG Seed: 5
Expected Answer: 6
--- Program ---
d = 6
x = d
i = x
s = 1
z = i
p = 4
w = 9
m = z
j = 5
a = m
o = a
g = a
e = g
y = o
v = e
h = 1
u = 0
#d:
==================================================

### QWEN MODEL ARCHITECTURE
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
  (generator): Generator(
    (streamer): Streamer()
  )
)