#%%
"""
Real Query Hops Analysis - Variable Binding Experiment
=====================================================
Analysis of completed experiment: variable_binding_hops_20250606_142010
"""

import sys
sys.path.append('src')

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from debug import ExperimentRunner

# Set up plotting style for Edward Tufte-inspired visualizations
plt.style.use('default')
sns.set_palette("husl")

print("🔬 Real Query Hops Analysis")
print("============================")
print("Experiment: variable_binding_hops_20250606_142010")

#%%
# Load the real experimental data
print("Loading experimental data...")

data_dir = Path("results/debug_experiments/variable_binding_hops_20250606_142010")
results_file = data_dir / "results.json"
summary_file = data_dir / "summary.json"

# Load results
with open(results_file, 'r') as f:
    real_results = json.load(f)

# Load summary
with open(summary_file, 'r') as f:
    summary = json.load(f)

print(f"✅ Loaded {len(real_results)} experimental results")
print(f"📊 Overall accuracy: {summary['overall_accuracy']:.1%}")
print(f"🎯 Models tested: {list(summary['by_model'].keys())}")
print(f"📏 Sequence lengths: {list(summary['by_seq_len'].keys())}")

#%%
# Set up the runner with real data
runner = ExperimentRunner()
runner.results = real_results
model_order = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B", 
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B"
]

# Create a mapping for sorting
model_order_map = {model: i for i, model in enumerate(model_order)}
runner.results.sort(key=lambda x: model_order_map.get(x["model_id"], 999))

# Convert to DataFrame for analysis
df = pd.DataFrame(real_results)

print("\n📈 Data Overview:")
print(f"Total samples: {len(df)}")
print(f"Experiments: {df['experiment'].unique()}")
print(f"Models: {df['model_id'].unique()}")
print(f"Sequence lengths: {sorted(df['seq_len'].unique())}")
print(f"Query hops range: {sorted(df['query_hops'].unique())}")

#%%
# Basic accuracy analysis
print("\n🎯 Basic Accuracy Analysis:")

print("\nAccuracy by model:")
for model, acc in summary['by_model'].items():
    model_name = model.split('/')[-1]
    print(f"  {model_name}: {acc:.1%}")

print("\nAccuracy by sequence length (14B model only):")
df_14b = df[df["model_id"] == "Qwen/Qwen3-14B"]
seq_accuracy = df_14b.groupby("seq_len")["correct"].mean()
for seq_len, acc in seq_accuracy.items():
    print(f"  Length {seq_len}: {acc:.1%}")

print("\nAccuracy by query hops (14B model only):")
hop_accuracy = df_14b.groupby("query_hops")["correct"].mean()
for hop, acc in hop_accuracy.items():
    print(f"  {hop} hops: {acc:.1%}")

#%%
# # Filter to only 14B model results
# print("\n🔧 Filtering to 14B model only...")
# original_count = len(runner.results)
# runner.results = [r for r in runner.results if r["model_id"] == "Qwen/Qwen3-14B"]
# print(f"Filtered from {original_count} to {len(runner.results)} results (14B model only)")

# # Update DataFrame as well
# df = pd.DataFrame(runner.results)
# print(f"DataFrame now contains {len(df)} samples from 14B model")

#%%
# Create Tufte-style heatmap visualization
print("\n📊 Creating heatmap visualization...")

fig = runner.plot_heatmap("variable_binding_hops", figsize=(14, 10))
if fig:
    # Save the plot
    plots_dir = data_dir / "analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.savefig(plots_dir / "query_hops_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved heatmap to: {plots_dir / 'query_hops_heatmap.png'}")
    plt.show()

#%%
# Create small multiples visualization
print("\n📊 Creating small multiples visualization...")

fig = runner.plot_small_multiples("variable_binding_hops", figsize=(18, 12))
if fig:
    plt.savefig(plots_dir / "query_hops_small_multiples.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved small multiples to: {plots_dir / 'query_hops_small_multiples.png'}")
    plt.show()

#%%
# Create slope graph visualization
print("\n📊 Creating slope graph visualization...")

fig = runner.plot_slope_graph("variable_binding_hops", figsize=(12, 8))
if fig:
    plt.savefig(plots_dir / "query_hops_slope_graph.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved slope graph to: {plots_dir / 'query_hops_slope_graph.png'}")
    plt.show()

#%%
# Detailed statistical analysis
print("\n📊 Detailed Statistical Analysis:")

# Effect of query hops
print("\n🔍 Query Hops Effect Analysis:")
hop_stats = df.groupby("query_hops")["correct"].agg(['count', 'sum', 'mean', 'std'])
hop_stats.columns = ['n_samples', 'n_correct', 'accuracy', 'std_dev']
print(hop_stats)

# Calculate effect size (1 hop vs 4 hops)
hop1_acc = df[df["query_hops"] == 1]["correct"].mean()
hop4_acc = df[df["query_hops"] == 4]["correct"].mean()
hop_effect = hop1_acc - hop4_acc
print(f"\n📉 Query Hops Effect (1 → 4 hops): {hop_effect:.1%} drop")
print(f"   1 hop accuracy: {hop1_acc:.1%}")
print(f"   4 hops accuracy: {hop4_acc:.1%}")

# Effect of sequence length
print("\n🔍 Sequence Length Effect Analysis:")
seq_stats = df.groupby("seq_len")["correct"].agg(['count', 'sum', 'mean', 'std'])
seq_stats.columns = ['n_samples', 'n_correct', 'accuracy', 'std_dev']
print(seq_stats)

# Calculate effect size (shortest vs longest)
seq_lengths = sorted(df["seq_len"].unique())
shortest_acc = df[df["seq_len"] == seq_lengths[0]]["correct"].mean()
longest_acc = df[df["seq_len"] == seq_lengths[-1]]["correct"].mean()
length_effect = shortest_acc - longest_acc
print(f"\n📉 Sequence Length Effect ({seq_lengths[0]} → {seq_lengths[-1]}): {length_effect:.1%} drop")
print(f"   Length {seq_lengths[0]} accuracy: {shortest_acc:.1%}")
print(f"   Length {seq_lengths[-1]} accuracy: {longest_acc:.1%}")

#%%
# Model comparison analysis
print("\n🤖 Model Comparison Analysis:")

# Create a detailed comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Variable Binding: Model Performance Analysis', fontsize=16, fontweight='bold')

# 1. Accuracy by query hops for each model
ax1 = axes[0, 0]
for model in sorted(df["model_id"].unique()):
    model_data = df[df["model_id"] == model]
    hop_acc = model_data.groupby("query_hops")["correct"].mean()
    model_name = model.split('/')[-1]
    ax1.plot(hop_acc.index, hop_acc.values, 'o-', label=model_name, linewidth=2, markersize=6)

ax1.set_xlabel('Query Hops')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Query Hops by Model')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 2. Accuracy by sequence length for each model
ax2 = axes[0, 1]
for model in sorted(df["model_id"].unique()):
    model_data = df[df["model_id"] == model]
    seq_acc = model_data.groupby("seq_len")["correct"].mean()
    model_name = model.split('/')[-1]
    ax2.plot(seq_acc.index, seq_acc.values, 'o-', label=model_name, linewidth=2, markersize=6)

ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Sequence Length by Model')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Heatmap of model performance by difficulty
ax3 = axes[1, 0]
# Create a difficulty score: higher seq_len and more hops = more difficult
df['difficulty'] = df['seq_len'] + df['query_hops'] * 2  # Weight hops more heavily
difficulty_acc = df.groupby(['model_id', 'difficulty'])['correct'].mean().unstack(fill_value=0)

# Clean up model names for readability
difficulty_acc.index = [name.split('/')[-1] for name in difficulty_acc.index]

sns.heatmap(difficulty_acc, annot=True, fmt='.2f', cmap='RdYlBu_r', 
            center=0.5, ax=ax3, cbar_kws={'label': 'Accuracy'})
ax3.set_title('Model Accuracy by Difficulty Level')
ax3.set_xlabel('Difficulty Score (seq_len + 2×hops)')
ax3.set_ylabel('Model')

# 4. Error analysis by query hops
ax4 = axes[1, 1]
error_rates = []
hop_levels = sorted(df['query_hops'].unique())

for hop in hop_levels:
    hop_data = df[df['query_hops'] == hop]
    error_rate = 1 - hop_data['correct'].mean()
    error_rates.append(error_rate)

bars = ax4.bar(hop_levels, error_rates, alpha=0.7, color='coral')
ax4.set_xlabel('Query Hops')
ax4.set_ylabel('Error Rate')
ax4.set_title('Error Rate by Query Hops')
ax4.grid(True, alpha=0.3, axis='y')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add value labels on bars
for bar, rate in zip(bars, error_rates):
    height = bar.get_height()
    ax4.annotate(f'{rate:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(plots_dir / "comprehensive_model_analysis.png", dpi=300, bbox_inches='tight')
print(f"💾 Saved comprehensive analysis to: {plots_dir / 'comprehensive_model_analysis.png'}")
plt.show()

#%%
# Interaction effects analysis
print("\n🔍 Interaction Effects Analysis:")

# Analyze how query hops and sequence length interact
interaction_data = df.groupby(['seq_len', 'query_hops'])['correct'].mean().unstack(fill_value=np.nan)

print("\nAccuracy matrix (rows=seq_len, cols=query_hops):")
print(interaction_data.round(3))

# Calculate correlations
print(f"\nCorrelation between seq_len and accuracy: {df[['seq_len', 'correct']].corr().iloc[0,1]:.3f}")
print(f"Correlation between query_hops and accuracy: {df[['query_hops', 'correct']].corr().iloc[0,1]:.3f}")

# Find the most/least difficult combinations
df['condition'] = df['seq_len'].astype(str) + '_len_' + df['query_hops'].astype(str) + '_hops'
condition_acc = df.groupby('condition')['correct'].mean().sort_values()

print(f"\n🎯 Most difficult conditions (lowest accuracy):")
for condition, acc in condition_acc.head(5).items():
    print(f"  {condition}: {acc:.1%}")

print(f"\n🎯 Easiest conditions (highest accuracy):")
for condition, acc in condition_acc.tail(5).items():
    print(f"  {condition}: {acc:.1%}")

#%%
# Generate final report
print("\n📋 Final Report Summary:")
print("=" * 50)

print(f"🧪 Experiment: Variable Binding with Query Hops Tracking")
print(f"📅 Date: 2025-06-06")
print(f"📊 Total samples: {len(df):,}")
print(f"🎯 Overall accuracy: {summary['overall_accuracy']:.1%}")

print(f"\n🔍 Key Findings:")
print(f"  • Query hops do not impact performance as much as we thought:")
print(f"    - 1 hop: {hop1_acc:.1%} accuracy")
print(f"    - 4 hops: {hop4_acc:.1%} accuracy")
print(f"    - Effect size: {hop_effect:.1%} drop")

print(f"  • Sequence length also impacts performance:")
print(f"    - Length {seq_lengths[0]}: {shortest_acc:.1%} accuracy")
print(f"    - Length {seq_lengths[-1]}: {longest_acc:.1%} accuracy")
print(f"    - Effect size: {length_effect:.1%} drop")

print(f"\n🤖 Model Performance Ranking:")
model_ranking = sorted(summary['by_model'].items(), key=lambda x: x[1], reverse=True)
for i, (model, acc) in enumerate(model_ranking, 1):
    model_name = model.split('/')[-1]
    print(f"  {i}. {model_name}: {acc:.1%}")

print(f"\n💾 Analysis outputs saved to: {plots_dir}")
print(f"   - query_hops_heatmap.png")
print(f"   - query_hops_small_multiples.png") 
print(f"   - query_hops_slope_graph.png")
print(f"   - comprehensive_model_analysis.png")

print(f"\n✅ Analysis complete!")

#%%