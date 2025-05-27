# Experiments Directory

This directory contains both **legacy experiments** (complex, full-featured) and **simplified experiments** (clean, modular) using the debug framework.

## 🚀 Quick Start (Simplified Experiments)

```bash
# Run a single experiment
./run_experiments.sh range

# Run with specific models
./run_experiments.sh boolean -m "Qwen/Qwen3-0.6B" -n 100

# Run all experiments
./run_experiments.sh all -o my_results
```

## 📁 File Organization

### ✅ Simplified Experiments (Recommended)
Built using the modular debug framework - clean, maintainable, and easy to extend:

- **`_02_entity_tracking_simple.py`** - Entity tracking with random +/- operations
- **`_04_boolean_simple.py`** - Boolean logic with and/or/not operations  
- **`_05_range_tracking_simple.py`** - Range tracking with fixed increments
- **`_03_counterfactual_simple.py`** - Counterfactual pair collection
- **`run_experiments.sh`** - Unified runner script
- **`_06_code_tracking.py`** - Interactive notebook for custom experiments

### 📚 Legacy Experiments (Reference)
Original complex implementations - kept for comparison and advanced features:

- **`_02_opensource_entity_tracking.py`** - Full entity tracking (466 lines)
- **`_04_boolean_experiment.py`** - Full boolean experiment (637 lines)
- **`_05_range_tracking.py`** - Full range tracking (305 lines)
- **`_03_collect_correct_pairs.py`** - Original counterfactual collection
- Various shell scripts: `_02_run_*.sh`, `_04_run_*.sh`, `_05_run_*.sh`

### 🗂️ Supporting Files
- **`results/`** - Experiment output directory
- **`*.png`** - Visualization outputs
- **`_.py`** - Small test file

## 🎯 Experiment Types

### Range Tracking
Tests arithmetic sequence understanding:
```python
x = 0
for i in range(5):
    x += 3
# Expected: 15
```

### Entity Tracking  
Tests variable tracking through random operations:
```python
x = 0
x += 5
x -= 2
x += 1
# Expected: 4
```

### Boolean Logic
Tests logical reasoning:
```python
x = True
y = False
x = x and y
x = not x
# Expected: True
```

### Counterfactual Pairs
Generates program pairs that diverge but both should be solvable:
```python
# Program A          # Program B
x = 0               x = 0
x += 5              x += 5  
x -= 2              x *= 2    # <- divergence
x += 1              x += 1
# Result: 4         # Result: 11
```

## 📊 Output Format

All simplified experiments use the debug framework's standard output:
- **`results.json`** - Detailed results for each test case
- **`summary.json`** - Aggregated statistics
- **Automatic plotting** - Accuracy vs sequence length
- **Error analysis** - Common failure patterns

## 🔧 Framework Benefits

The simplified experiments offer:

**Lines of Code Reduction:**
- Entity tracking: 466 → 71 lines (85% reduction)
- Boolean logic: 637 → 77 lines (88% reduction)  
- Range tracking: 305 → 67 lines (78% reduction)
- Counterfactual: 165 → 95 lines (42% reduction)

**Features:**
- ✅ Automatic model loading and cleanup
- ✅ **Model caching** for interactive use (load once, reuse many times)
- ✅ Built-in plotting and analysis
- ✅ Consistent result format
- ✅ Error handling and retries
- ✅ Progress tracking
- ✅ Configurable via CLI or code

## 🚀 Creating New Experiments

Use the interactive notebook or create a simple script:

```python
from debug import quick_experiment, ExperimentRunner, generators, prompts

config = quick_experiment(
    name="my_experiment",
    prompt_template=prompts.RANGE_TRACKING,
    program_generator=generators.make_range_program,
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=50,
    seq_lens=[2, 3, 4, 5]
)

runner = ExperimentRunner()

# For interactive use, preload models to avoid reloading
runner.preload_models(["Qwen/Qwen3-0.6B"])

result = runner.run(config)  # Uses cached model!
runner.plot("my_experiment")
```

## 🎛️ Advanced Usage

For complex scenarios, you can still use the legacy experiments or extend the framework:

```bash
# Legacy experiments (if you need specific features)
python _02_opensource_entity_tracking.py --output-dir results/legacy --num-seqs 100

# Custom interactive experiments  
jupyter notebook _06_code_tracking.py
```

---

**Recommendation:** Start with the simplified experiments. They cover 95% of use cases with 10x less code! 🎉 