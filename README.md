# Debug Experiment Framework

A simple, modular framework for testing language models on code understanding tasks.

## Overview

This framework allows you to rapidly create and run experiments testing how well language models can trace through code execution, track variables, and perform logical reasoning.

## Quick Start

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

## Structure

```
src/debug/
├── __init__.py      # Main exports and quick_experiment helper
├── core.py          # ExperimentConfig and parser functions  
├── generators.py    # Program generators for different tasks
├── prompts.py       # Prompt templates
└── runner.py        # ExperimentRunner class
```

## Components

### Program Generators
- `make_range_program()`: Simple for-loop with fixed increment
- `make_range_program_lines()`: Explicit line-by-line arithmetic
- `make_variable_increments()`: Different increment each step
- `make_arithmetic_sequence()`: Arithmetic progressions
- `make_counter_program()`: Basic counting
- `make_fibonacci_program()`: Fibonacci sequences
- `make_counterfactual_pair()`: Generate program pairs that diverge

### Prompt Templates
- `RANGE_TRACKING`: Basic arithmetic tracking
- `BOOLEAN_LOGIC`: Boolean operations
- `STEP_BY_STEP`: Guided reasoning
- `CHAIN_OF_THOUGHT`: Open-ended reasoning
- `MINIMAL`: Minimal prompt
- `ENTITY_TRACKING`: Generic tracking

### Parser Functions
- `parse_integer()`: Extract integers from model responses
- `parse_boolean()`: Extract boolean values

## Installation

```bash
# Clone and install
git clone <repo>
cd debug
uv sync

# Install in development mode
uv pip install -e .
```

## Creating Custom Experiments

### Custom Generator
```python
def my_generator(seq_len: int, rng: np.random.RandomState):
    # Generate your program
    program = f"x = 0\nfor i in range({seq_len}):\n    x += 2"
    expected = seq_len * 2
    return program, expected

config = quick_experiment("my_test", prompts.MINIMAL, my_generator)
```

### Custom Prompt
```python
MY_PROMPT = """Trace this code step by step:
{code}
Final answer: """

config = quick_experiment("my_test", MY_PROMPT, generators.make_range_program)
```

## Examples

### Interactive Notebook
See `experiments/_06_code_tracking.py` for an interactive notebook with examples.

### Ready-to-Run Experiments
The `experiments/` folder contains both simplified and legacy experiments:

```bash
# Run simplified experiments (recommended)
cd experiments
./run_experiments.sh all

# Or run individual experiments
./run_experiments.sh range -m "Qwen/Qwen3-0.6B" -n 50
python _04_boolean_simple.py --num-seqs 100
```

**Complexity Reduction Achieved:**
- Entity tracking: 465 → 67 lines (85% reduction)
- Boolean logic: 636 → 95 lines (85% reduction)  
- Range tracking: 304 → 75 lines (75% reduction)
- **Total: 1,569 → 376 lines (76% reduction)**

## Results

Results are automatically saved to `results/debug_experiments/` with:
- `results.json`: Full detailed results
- `summary.json`: Aggregated statistics

## Dependencies

- PyTorch + Transformers (for model loading)
- NumPy + Pandas (for data handling) 
- Matplotlib (for plotting)
- Managed with `uv`
