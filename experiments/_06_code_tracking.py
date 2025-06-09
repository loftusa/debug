#%%
%load_ext autoreload
%autoreload

import sys
sys.path.append('../src')

from debug import ExperimentConfig, ExperimentRunner, generators, prompts, quick_experiment
import numpy as np

print('Debug framework loaded!')

#%% [markdown]
# # Interactive Code Tracking Experiments
# 
# This notebook provides an easy interface for testing different code understanding experiments.
# The heavy lifting is now done by the `debug` package.
#%% Initialize Runner

runner = ExperimentRunner()

#%% Quick Experiment Examples
%autoreload
from debug import generators, prompts, quick_experiment

# Basic range tracking
basic_config = quick_experiment(
    name="basic_range", 
    prompt_template=prompts.RANGE_TRACKING,
    program_generator=generators.make_range_program,
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=5,
    seq_lens=[2, 3, 4]
)

# Step-by-step reasoning
reasoning_config = quick_experiment(
    name="step_by_step",
    prompt_template=prompts.STEP_BY_STEP, 
    program_generator=generators.make_range_program,
    models=["Qwen/Qwen3-1.7B"],
    num_seqs=5,
    seq_lens=[2, 3, 4]
)

# Variable increments test
variable_config = quick_experiment(
    name="variable_increments",
    prompt_template=prompts.RANGE_TRACKING,
    program_generator=generators.make_variable_increments,
    models=["Qwen/Qwen3-1.7B"],
    num_seqs=5,
    seq_lens=[2, 3, 4]
)

exception_config = quick_experiment(
    name="exception_handling",
    prompt_template=prompts.EXCEPTION_HANDLING,
    program_generator=generators.make_exception_program,
    models=["Qwen/Qwen3-1.7B"],
    num_seqs=5,
    seq_lens=[1, 2]
)

variable_binding_config = quick_experiment(
    name="variable_binding",
    prompt_template=prompts.VARIABLE_BINDING,
    program_generator=generators.make_variable_binding_program,
    models=["Qwen/Qwen3-1.7B"],
    num_seqs=100,
    seq_lens=list(range(1, 17))
)

print("Configurations ready!")
print("Available configs: basic_config, reasoning_config, variable_config")

#%%
%autoreload
import debug
from debug.generators import make_exception_program


runner.run(variable_binding_config)

#%% Run Experiments

# Uncomment to run:
result = runner.run(basic_config)

#%% Custom Experiment Template

def my_custom_generator(seq_len: int, rng: np.random.RandomState):
    """Define your custom program generator here."""
    # Example: multiplication tables
    base = rng.randint(2, 5)
    program = f"""x = 0
for i in range(1, {seq_len + 1}):
    x += {base} * i"""
    
    # Sum of base * 1 + base * 2 + ... + base * seq_len
    # = base * (1 + 2 + ... + seq_len) = base * seq_len * (seq_len + 1) / 2
    expected = base * seq_len * (seq_len + 1) // 2
    return program, expected

custom_config = quick_experiment(
    name="multiplication_table",
    prompt_template=prompts.CHAIN_OF_THOUGHT,
    program_generator=my_custom_generator,
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=3,
    seq_lens=[2, 3]
)

# Uncomment to run:
# custom_result = runner.run(custom_config)

#%% Analysis Tools

# View results
def show_results():
    """Show all experiment results."""
    if not runner.results:
        print("No results yet! Run some experiments first.")
        return
    
    import pandas as pd
    df = pd.DataFrame(runner.results)
    
    print("=== Experiment Summary ===")
    summary = df.groupby(['experiment', 'model_id', 'seq_len'])['correct'].agg(['count', 'sum', 'mean'])
    summary.columns = ['total', 'correct', 'accuracy']
    print(summary)
    
    print(f"\nOverall accuracy: {df['correct'].mean():.1%}")

# Quick plotting
def plot_results(experiment_name=None):
    """Plot results for all or specific experiment."""
    runner.plot(experiment_name)

# Error analysis  
def analyze_errors(experiment_name):
    """Analyze errors for an experiment."""
    runner.analyze_errors(experiment_name)

#%% Available Components

print("\n=== Available Components ===")
print("\nPrompt templates (prompts.*):")
for name in dir(prompts):
    if not name.startswith('_'):
        print(f"  - prompts.{name}")

print("\nProgram generators (generators.*):")  
for name in dir(generators):
    if not name.startswith('_') and callable(getattr(generators, name)):
        print(f"  - generators.{name}")

print("\nUsage:")
print("1. Create config: quick_experiment(name, prompt_template, program_generator)")
print("2. Run: runner.run(config)")
print("3. Analyze: show_results(), plot_results(), analyze_errors()")

#%% Model Management for Interactive Use

print("\n=== Model Management ===")
print("For interactive sessions, preload models to avoid reloading:")
print("")
print("# Preload models once (takes time but only done once)")
print("runner.preload_models(['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B'])")
print("")
print("# Check loaded models")
print("runner.list_loaded_models()")
print("")
print("# Unload specific model")
print("runner.unload_model('Qwen/Qwen3-0.6B')")
print("")
print("# Free all GPU memory")
print("runner.unload_all_models()")
print("")
print("# For multiple models: avoid GPU memory issues")
print("runner.run(config, no_cache=True)  # Unloads each model after use")
print("")
print("# Batch inference for efficiency (much faster!)")
print("runner.run(config, batch_size=16)  # Process 16 prompts at once")
print("runner.run(config)  # Default: batch all prompts for each seq_len")

# Uncomment to preload models for faster experimentation:
# runner.preload_models(["Qwen/Qwen3-0.6B"])

#%% Interactive Testing Area

# Use this cell for quick testing
print("\n=== Ready for Interactive Testing ===")
print("Example:")
print("config = quick_experiment('test', prompts.MINIMAL, generators.make_counter_program)")
print("result = runner.run(config)  # Models cached after first run!")
print("plot_results('test')")
print("")
print("# Multiple models without GPU memory issues:")
print("multi_model_config = quick_experiment('multi_test', prompts.MINIMAL, generators.make_range_program,")
print("                                      models=['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B'])")
print("result = runner.run(multi_model_config, no_cache=True)  # Safe for GPU memory!")
print("")
print("# Batch inference for speed:")
print("large_config = quick_experiment('speed_test', prompts.VARIABLE_BINDING,")
print("                                generators.make_variable_binding_program,")
print("                                num_seqs=100, seq_lens=[4, 8, 12])")
print("result = runner.run(large_config, batch_size=32)  # Much faster than sequential!") 