# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The goal of this series of experiments is to understand how and whether language models track the internal state of variables in programming languages. The long-term goal is to be able to track the internal states of programming languages if they would be run without needing to execute any code. This should help users with debugging and quickly understanding code.

To this end, we first test various models on their ability to accomplish coding tasks. Part of the framework has a core purpose of rapidly creating and running experiments that test how well language models can trace through code execution, track variables, and perform logical reasoning.

We will also include tools built with the `nnsight` python package that allow for interpretability techniques like patching and causal mediation analysis. Many of our experiments recreate and extend the methodology from ["Tracing Knowledge in Language Models Back to the Training Data"](https://arxiv.org/abs/2505.20896), particularly their causal tracing approach for understanding variable binding mechanisms in language models.

Your focus is on fast, high-information experiments and iteration. You try to embody "You and your Research" by Richard Hamming. You don't spend time yak-shaving, but you are careful with your experimental design and with making sure your code has no bugs in it. Your goal with writing experiments is maximum information-per-line-of-code: you write simple, efficient code that can be iterated on easily, and the point of the code is answering questions. Your code is designed to make it easy to go back and see what you did, or make minor experimental adjustments.

## Dependencies:
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
- hypothesis
- ruff (for linting)
- gradio
- jupyter/jupyterlab
- uv (package manager)
- vllm


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
below are random seeds that work for 1, 2, and 3 hops for sequence lengths of 5 and 17, generated by running `experiments/11_test_rng_seeds.py` multiple times with `experiments/find_all_seeds.sh`. Prioritize the 17-length 3-hop case.

Configuration: 1 hop(s), sequence length 5
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 11
Expected Answer: 4
Number of Hops: 1
--- Program ---
z = 1
x = z
s = 0
a = 4
d = 5
#a: 
✅ Completed at: Thu 26 Jun 2025 12:18:27 PM EDT


Configuration: 2 hop(s), sequence length 5
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 3
Expected Answer: 3
Number of Hops: 2
--- Program ---
k = 3
z = 5
u = k
l = 6
a = z
#u: 
✅ Completed at: Thu 26 Jun 2025 12:20:30 PM EDT


Configuration: 3 hop(s), sequence length 5
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 2
Expected Answer: 8
Number of Hops: 3
--- Program ---
i = 8
x = i
h = x
y = 4
w = x
#h: 
✅ Completed at: Thu 26 Jun 2025 12:22:17 PM EDT


Configuration: 1 hop(s), sequence length 17
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 5
Expected Answer: 6
Number of Hops: 1
--- Program ---
d = 6
x = d
i = x
s = 1
z = i
p = 4
✅ Completed at: Thu 26 Jun 2025 12:24:05 PM EDT


Configuration: 2 hop(s), sequence length 17
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 14
Expected Answer: 6
Number of Hops: 2
--- Program ---
l = 6
h = l
m = h
z = 7
e = m
y = m
✅ Completed at: Thu 26 Jun 2025 12:26:23 PM EDT


Configuration: 3 hop(s), sequence length 17
----------------------------------------
🎉 Found a robust program (all models correct)!
RNG Seed: 12
Expected Answer: 1
Number of Hops: 3
--- Program ---
l = 1
c = l
y = 5
p = 6
m = 8
q = p
f = m
a = c
j = 9
v = 0
x = f
o = q
r = a
w = 5
g = r
b = r
i = r
#a: 
✅ Completed at: Thu 26 Jun 2025 12:28:26 PM EDT

=========================================
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