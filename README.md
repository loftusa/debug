# Debug Experiment Framework

A comprehensive framework for testing language models on code understanding tasks with advanced causal analysis capabilities. Designed for rapid experimentation and interpretability research on how language models track variable states and perform logical reasoning in code.

## Overview

This framework enables systematic investigation of language model capabilities through:
- **Core Experiments**: Variable tracking, boolean logic, arithmetic sequences
- **Causal Tracing**: Understanding internal mechanisms via activation patching
- **Interpretability Tools**: Built on `nnsight` for detailed model analysis
- **Batch Execution**: Efficient experiment running across multiple models and configurations

The framework recreates and extends methodology from ["How Do Transformers Learn Variable Binding in Symbolic Programs?
"](https://arxiv.org/abs/2505.20896), focusing on variable binding mechanisms through causal interventions.

## Quick Start

### Basic Experiments

```python
from debug import quick_experiment, ExperimentRunner, generators, prompts

# Create an experiment
config = quick_experiment(
    name="range_tracking", 
    prompt_template=prompts.RANGE_TRACKING,
    program_generator=generators.make_range_program,
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=10,
    seq_lens=[2, 3, 4, 5, 6]
)

# Run it
runner = ExperimentRunner()
result = runner.run(config)

# Analyze
runner.plot("range_tracking")
runner.analyze_errors("range_tracking")
```

### Causal Tracing

```python
from debug.causal_experiment_runner import CausalExperimentRunner
from debug.causal_tracing import run_causal_tracing_experiment

# Initialize causal experiment runner
runner = CausalExperimentRunner()

# Run intervention experiment
results = runner.run_intervention_experiment(
    model_id="Qwen/Qwen3-0.6B",
    programs=program_list,
    intervention_type="residual_stream",  # or "attention_heads"
    num_hops=2,
    seq_len=5
)

# Visualize results
runner.plot_intervention_results(results)
```

### Interactive Use (Jupyter/IPython)

For interactive sessions, preload models to avoid reloading:

```python
runner = ExperimentRunner()

# Preload models once (takes time but only done once)
runner.preload_models(["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])

# Now experiments run much faster!
result1 = runner.run(config1)  # Uses cached model
result2 = runner.run(config2)  # Uses cached model

# Manage models
runner.list_loaded_models()           # See what's loaded
runner.unload_model("Qwen/Qwen3-0.6B")  # Free specific model
runner.unload_all_models()            # Free all GPU memory
```

## Architecture

### Core Framework

```
src/debug/
├── __init__.py                    # Main exports and quick_experiment helper
├── core.py                        # ExperimentConfig and parser functions  
├── runner.py                      # ExperimentRunner class for basic experiments
├── generators.py                  # Program generators for different tasks
├── prompts.py                     # Prompt templates
├── causal_tracing.py             # Causal intervention and patching functionality
├── causal_experiment_runner.py   # Specialized runner for causal experiments
├── causal_visualization.py       # Visualization tools for causal analysis
├── counterfactual.py             # Counterfactual generation utilities
└── token_analyzer.py             # Token-level analysis tools
```

### Experiment Types

```
experiments/
├── _01_boolean_simple.py          # Boolean logic experiments
├── _02_integer_simple.py          # Integer tracking experiments  
├── _03_range_simple.py            # Range/loop tracking experiments
├── _04_boolean_simple.py          # Advanced boolean experiments
├── _05_variable_binding.py        # Variable binding experiments
├── _06_code_tracking.py           # Interactive code tracking notebook
├── _07_binding_patching.py        # Basic causal tracing
├── _08_full_layer_token_patching.py # Comprehensive causal analysis
├── _09_visualize_patching_auto.py # Automated visualization
├── 11_test_rng_seeds.py          # RNG seed testing for robust programs
└── run_experiments*.sh           # Batch execution scripts
```

### Results Structure

```
results/
├── debug_experiments/            # Core experiment results
│   └── {experiment_name}_{timestamp}/
│       ├── results.json          # Detailed per-sample results
│       ├── summary.json          # Aggregated statistics
│       └── accuracy_plot.png     # Auto-generated visualizations
└── full_token_layer_patching/    # Causal intervention results
    └── {timestamp}/
        ├── intervention_results.json # Detailed causal intervention results
        └── patch_results.json       # Aggregated patching statistics
```

## Causal Tracing Methodology

Our causal tracing implementation follows the methodology from ["Tracing Knowledge in Language Models Back to the Training Data"](https://arxiv.org/abs/2505.20896):

### Core Concept

1. **Counterfactual Construction**: For each program, identify the variable binding chain and create a counterfactual version by changing the root numerical value
2. **Activation Patching**: Replace activations from the original program with activations from the counterfactual at specific layers/positions
3. **Causal Measurement**: Measure how much the intervention changes the model's output toward the counterfactual answer

### Implementation

```python
from debug.causal_tracing import run_causal_tracing_experiment

# Define your programs (original and counterfactual pairs)
programs = [
    {
        'original': 'x = 2\ny = x\n#y:',
        'counterfactual': 'x = 7\ny = x\n#y:', 
        'expected_original': 2,
        'expected_counterfactual': 7,
        'variable_chain': ['y', 'x']
    }
]

# Run causal tracing
results = run_causal_tracing_experiment(
    model_id="Qwen/Qwen3-0.6B",
    programs=programs,
    intervention_layers=list(range(28)),  # All layers for Qwen3-0.6B
    intervention_type="residual_stream"   # or "attention_heads"
)
```

### Intervention Types

- **Residual Stream**: Patch hidden states going into each layer
- **Attention Heads**: Patch individual attention head outputs

### Token Targeting Strategy

- **RHS Tokens**: Target tokens on right-hand side of assignments (`=`)
- **Referential Depth**: Label by chain position (Ref Depth 1, 2, 3...)
- **Query Tokens**: Target the final query variable and colon

### Robust Program Generation

Use validated RNG seeds for consistent, robust programs:

```python
# Known working seeds for different configurations
ROBUST_SEEDS = {
    (1, 5): 11,   # 1 hop, sequence length 5
    (2, 5): 3,    # 2 hops, sequence length 5  
    (3, 5): 2,    # 3 hops, sequence length 5
    (1, 17): 5,   # 1 hop, sequence length 17
    (2, 17): 14,  # 2 hops, sequence length 17
    (3, 17): 12   # 3 hops, sequence length 17
}

# Generate robust program
rng = np.random.RandomState(ROBUST_SEEDS[(num_hops, seq_len)])
program, expected = generators.make_variable_binding_program(seq_len, rng, num_hops)
```

## Components

### Program Generators
- `make_range_program()`: Simple for-loop with fixed increment
- `make_range_program_lines()`: Explicit line-by-line arithmetic
- `make_variable_increments()`: Different increment each step
- `make_arithmetic_sequence()`: Arithmetic progressions
- `make_counter_program()`: Basic counting
- `make_fibonacci_program()`: Fibonacci sequences
- `make_variable_binding_program()`: Variable binding chains with configurable hops
- `make_counterfactual_pair()`: Generate program pairs that diverge
- `make_exception_program()`: Exception handling scenarios

### Prompt Templates
- `RANGE_TRACKING`: Basic arithmetic tracking
- `BOOLEAN_LOGIC`: Boolean operations
- `STEP_BY_STEP`: Guided reasoning
- `CHAIN_OF_THOUGHT`: Open-ended reasoning
- `MINIMAL`: Minimal prompt
- `ENTITY_TRACKING`: Generic tracking
- `VARIABLE_BINDING`: Variable binding tasks

### Parser Functions
- `parse_integer()`: Extract integers from model responses
- `parse_boolean()`: Extract boolean values
- `parse_variable_binding()`: Extract variable binding results

### Causal Analysis Tools
- `CausalExperimentRunner`: Specialized runner for causal experiments
- `run_causal_tracing_experiment()`: Core causal tracing functionality
- `TokenAnalyzer`: Token-level analysis and targeting
- `create_counterfactual_program()`: Systematic counterfactual generation
- Visualization tools for intervention results

## Installation

```bash
# Clone and install
git clone <repo>
cd debug
uv sync

# Install in development mode
uv pip install -e .
```

### Dependencies

- **Core**: torch, transformers, accelerate, numpy, matplotlib
- **Interpretability**: nnsight, jaxtyping, torchtyping, einops
- **Development**: pytest, ruff, jupyter, gradio
- **Performance**: bitsandbytes, vllm
- **Package Management**: uv

## Development Commands

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

# Run with specific seeds for reproducibility
uv run _08_full_layer_token_patching.py --rng-seed 11 --num-hops 1 --seq-len 5
```

### Code Quality

```bash
# Run ruff for linting and formatting
ruff check src/
ruff format src/

# Run tests
pytest tests/
```

## Creating Custom Experiments

### Custom Generator
```python
def my_generator(seq_len: int, rng: np.random.RandomState):
    # Generate your program
    program = f"x = 0\nfor i in range({seq_len}):\n    x += 2"
    expected = seq_len * 2
    metadata = {"operation": "addition", "increment": 2}
    return program, expected, metadata

config = quick_experiment("my_test", prompts.MINIMAL, my_generator)
```

### Custom Causal Experiment
```python
from debug.causal_experiment_runner import CausalExperimentRunner

runner = CausalExperimentRunner()

# Create custom program pairs
custom_programs = [
    {
        'original': 'x = 5\ny = x * 2\n#y:',
        'counterfactual': 'x = 3\ny = x * 2\n#y:',
        'expected_original': 10,
        'expected_counterfactual': 6,
        'variable_chain': ['y', 'x']
    }
]

results = runner.run_intervention_experiment(
    model_id="Qwen/Qwen3-0.6B",
    programs=custom_programs,
    intervention_type="residual_stream"
)
```

### Custom Prompt
```python
MY_PROMPT = """Trace this code step by step:
{code}
Final answer: """

config = quick_experiment("my_test", MY_PROMPT, generators.make_range_program)
```

## Key Patterns

### Memory Management
```python
# For memory-constrained environments
result = runner.run(config, no_cache=True)  # Unload models after each use

# For interactive use
runner.preload_models(["model1", "model2"])  # Load once, reuse many times
```

### Batch Processing
```python
# Run multiple configurations efficiently
configs = [config1, config2, config3]
for config in configs:
    result = runner.run(config)  # Uses cached models automatically
    runner.plot(config.name)
```

### Error Analysis
```python
# Analyze failure cases
runner.analyze_errors("experiment_name", show_programs=True)

# Filter by specific criteria
runner.analyze_errors("experiment_name", min_seq_len=5, max_seq_len=10)
```

## Research Applications

This framework supports research into:

- **Variable Binding Mechanisms**: How models track variable assignments across different referential depths
- **Causal Intervention Analysis**: Which layers and positions are critical for variable tracking
- **Model Comparison**: Systematic comparison of different model architectures on code understanding
- **Failure Mode Analysis**: Understanding when and why models fail at code tracing
- **Attention Head Analysis**: Role of individual attention heads in variable binding

The framework is designed for maximum information-per-line-of-code, following principles from Richard Hamming's "You and Your Research" - efficient, iterative experimentation with clear experimental design and easy reproducibility.
