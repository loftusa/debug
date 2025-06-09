#%%
"""
Query Hops Analysis - Interactive Notebook
==========================================
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from debug import ExperimentRunner, generators, prompts, quick_experiment

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("🔬 Query Hops Analysis Notebook")
print("================================")

#%%
# Test the query hops functionality with a small example
print("Testing query hops tracking...")

rng = np.random.RandomState(42)
for i in range(3):
    program, answer, query_hops = generators.make_variable_binding_program(10, rng)
    print(f"\nExample {i+1}:")
    print(f"Query hops: {query_hops}")
    print(f"Answer: {answer}")
    print("Program:")
    print(program)
    print("-" * 50)

#%%
# Set up a small experiment configuration for testing
print("Setting up experiment configuration...")

# Use smaller models for faster testing - you can modify this list
TEST_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B", 
]

# Create experiment config
config = quick_experiment(
    name="variable_binding_hops_test",
    prompt_template=prompts.VARIABLE_BINDING,
    program_generator=generators.make_variable_binding_program,
    models=TEST_MODELS,
    num_seqs=20,  # Small number for testing
    seq_lens=list(range(5, 16, 2))  # [5, 7, 9, 11, 13, 15]
)

print(f"Models: {config.models}")
print(f"Sequence lengths: {config.seq_lens}")
print(f"Samples per condition: {config.num_seqs}")

#%%
# Run the experiment
print("Running experiment...")

runner = ExperimentRunner()

# For testing, you might want to run with no_cache=True to save GPU memory
# result = runner.run(config, no_cache=True)

# Uncomment the line above to actually run the experiment
# For now, let's create some mock data to demonstrate the visualizations

print("Creating mock data for visualization demo...")

# Generate realistic mock data
mock_results = []
models = TEST_MODELS
seq_lens = config.seq_lens
query_hops_range = [1, 2, 3, 4]

rng_mock = np.random.RandomState(123)

for model in models:
    for seq_len in seq_lens:
        for hop in query_hops_range:
            # Generate realistic accuracy patterns
            # Accuracy generally decreases with sequence length and query hops
            base_accuracy = 0.9 - (seq_len - 5) * 0.05 - (hop - 1) * 0.1
            base_accuracy = max(0.1, min(0.95, base_accuracy))
            
            # Add some model-specific variation
            if "0.6B" in model:
                base_accuracy -= 0.15
            
            # Generate individual samples
            n_samples = config.num_seqs
            for i in range(n_samples):
                # Add noise to individual samples
                accuracy = base_accuracy + rng_mock.normal(0, 0.1)
                is_correct = rng_mock.random() < max(0.05, min(0.95, accuracy))
                
                mock_results.append({
                    "experiment": "variable_binding_hops_test",
                    "model_id": model,
                    "seq_len": seq_len,
                    "seq_id": i,
                    "query_hops": hop,
                    "correct": is_correct,
                    "true_answer": rng_mock.randint(0, 10),
                    "predicted_answer": rng_mock.randint(0, 10),
                })

runner.results = mock_results
print(f"Generated {len(mock_results)} mock results")

#%%
# Basic analysis of query hops distribution
print("Analyzing query hops distribution...")

df = pd.DataFrame(runner.results)
print(f"Total samples: {len(df)}")
print(f"Query hops distribution:")
print(df["query_hops"].value_counts().sort_index())

print(f"\nOverall accuracy by query hops:")
hop_accuracy = df.groupby("query_hops")["correct"].mean()
for hop, acc in hop_accuracy.items():
    print(f"  {hop} hops: {acc:.1%}")

#%%
# Visualization 1: Heatmap showing accuracy by sequence length and query hops
print("Creating heatmap visualization...")

fig = runner.plot_heatmap("variable_binding_hops_test", figsize=(12, 8))
plt.show()

#%%
# Visualization 2: Small multiples for each model and query hop level
print("Creating small multiples visualization...")

fig = runner.plot_mall_multiples("variable_binding_hops_test", figsize=(15, 8))
plt.show()

#%%
# Visualization 3: Slope graph showing query hops effect
print("Creating slope graph visualization...")

fig = runner.plot_slope_graph("variable_binding_hops_test", figsize=(10, 6))
plt.show()

#%%
# Custom analysis: Query hops vs accuracy by model
print("Custom analysis: Model comparison by query hops...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left plot: Accuracy by query hops for each model
for model in df["model_id"].unique():
    model_data = df[df["model_id"] == model]
    hop_acc = model_data.groupby("query_hops")["correct"].mean()
    model_name = model.split('/')[-1]
    axes[0].plot(hop_acc.index, hop_acc.values, 'o-', label=model_name, linewidth=2, markersize=6)

axes[0].set_xlabel('Query Hops')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs Query Hops by Model')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right plot: Accuracy by sequence length for each model
for model in df["model_id"].unique():
    model_data = df[df["model_id"] == model]
    seq_acc = model_data.groupby("seq_len")["correct"].mean()
    model_name = model.split('/')[-1]
    axes[1].plot(seq_acc.index, seq_acc.values, 'o-', label=model_name, linewidth=2, markersize=6)

axes[1].set_xlabel('Sequence Length')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Sequence Length by Model')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

#%%
# Statistical analysis: Query hops effect size
print("Statistical analysis of query hops effect...")

# Calculate effect sizes
results_df = pd.DataFrame(runner.results)

print("Mean accuracy by query hops:")
hop_stats = results_df.groupby("query_hops")["correct"].agg(['mean', 'std', 'count'])
print(hop_stats)

print("\nEffect of increasing query hops (1 -> 4):")
hop1_acc = results_df[results_df["query_hops"] == 1]["correct"].mean()
hop4_acc = results_df[results_df["query_hops"] == 4]["correct"].mean()
print(f"1 hop: {hop1_acc:.1%}")
print(f"4 hops: {hop4_acc:.1%}")
print(f"Difference: {hop1_acc - hop4_acc:.1%}")

print("\nEffect of sequence length (5 -> 15):")
seq5_acc = results_df[results_df["seq_len"] == 5]["correct"].mean()
seq15_acc = results_df[results_df["seq_len"] == 15]["correct"].mean()
print(f"Length 5: {seq5_acc:.1%}")
print(f"Length 15: {seq15_acc:.1%}")
print(f"Difference: {seq5_acc - seq15_acc:.1%}")

#%%
# Real experiment runner (uncomment to run actual experiments)
"""
To run the actual experiment with real models, uncomment this cell:
"""
print("Running REAL experiment...")
MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    ]
runner_real = ExperimentRunner()
config_real = quick_experiment(
    name="variable_binding_hops",
    prompt_template=prompts.VARIABLE_BINDING,
    program_generator=generators.make_variable_binding_program,
    models=MODELS,
    num_seqs=100,
    seq_lens=list(range(5, 18, 2))
)
real_result = runner_real.run(config_real, no_cache=True)

print(f"Real experiment completed!")
print(f"Results saved to: {real_result['output_dir']}")
print(f"Overall accuracy: {real_result['summary']['overall_accuracy']:.1%}")

# Then run the visualizations on real data:
plots_dir = runner_real.output_dir / "plots"
fig = runner_real.plot_heatmap("variable_binding_hops_test")
plt.savefig(plots_dir / "heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

fig = runner_real.plot_mall_multiples("variable_binding_hops_test") 
plt.savefig(plots_dir / "small_multiples.png", dpi=300, bbox_inches='tight')
plt.show()

fig = runner_real.plot_slope_graph("variable_binding_hops_test")
plt.savefig(plots_dir / "slope_graph.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
# Iteration utilities for further analysis
print("Utility functions for further analysis...")

def analyze_difficulty_patterns(results, experiment_name):
    """Analyze which combinations of seq_len and query_hops are most difficult."""
    df = pd.DataFrame(results)
    df = df[df["experiment"] == experiment_name]
    
    difficulty = df.groupby(["seq_len", "query_hops"])["correct"].agg(['mean', 'count']).reset_index()
    difficulty = difficulty[difficulty[("correct", "count")] >= 5]  # Filter low-count combinations
    difficulty_sorted = difficulty.sort_values(("correct", "mean"))
    
    print("Most difficult combinations (lowest accuracy):")
    print(difficulty_sorted.head(10))
    
    return difficulty_sorted

def compare_models_by_hops(results, experiment_name):
    """Compare how different models handle different query hop levels."""
    df = pd.DataFrame(results)
    df = df[df["experiment"] == experiment_name]
    
    comparison = df.groupby(["model_id", "query_hops"])["correct"].mean().unstack()
    print("Model performance by query hops:")
    print(comparison)
    
    return comparison

# Run the analysis functions
print("\n" + "="*50)
difficulty_analysis = analyze_difficulty_patterns(runner.results, "variable_binding_hops_test")

print("\n" + "="*50)
model_comparison = compare_models_by_hops(runner.results, "variable_binding_hops_test")

#%%
print("🎉 Analysis complete!")
print("\nNext steps:")
print("1. Uncomment the real experiment cell to run with actual models")
print("2. Modify the model list to include larger/different models")
print("3. Adjust sequence lengths and sample sizes as needed")
print("4. Use the utility functions to dive deeper into specific patterns")
print("5. Create custom visualizations using the df = pd.DataFrame(runner.results) data") 