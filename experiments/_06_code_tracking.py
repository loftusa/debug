#%%
import sys
sys.path.append('../src')

from debug import ExperimentConfig, ExperimentRunner, generators, prompts, quick_experiment
import numpy as np

print('Debug framework loaded!')

#%% [markdown]
# # Interactive Code Tracking Experiments
# 
# This notebook provides an easy interface for testing different code understanding experiments.
# The heavy lifting is now done by the `debug` package - just focus on experiment design!

#%% Initialize Runner

runner = ExperimentRunner()

#%% Quick Experiment Examples

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
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=5,
    seq_lens=[2, 3, 4]
)

# Variable increments test
variable_config = quick_experiment(
    name="variable_increments",
    prompt_template=prompts.RANGE_TRACKING,
    program_generator=generators.make_variable_increments,
    models=["Qwen/Qwen3-0.6B"],
    num_seqs=5,
    seq_lens=[2, 3, 4]
)

print("Configurations ready!")
print("Available configs: basic_config, reasoning_config, variable_config")

#%% Run Experiments

# Uncomment to run:
# result = runner.run(basic_config)

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

#%% Interactive Testing Area

# Use this cell for quick testing
print("\n=== Ready for Interactive Testing ===")
print("Example:")
print("config = quick_experiment('test', prompts.MINIMAL, generators.make_counter_program)")
print("result = runner.run(config)")
print("plot_results('test')") 