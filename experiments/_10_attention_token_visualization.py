#%%
"""
Interactive Attention Head Patching Visualization

This notebook provides interactive visualization and analysis of attention head 
patching experiment results. Use Jupyter's #%% cell structure to explore the data
and experiment with different visualization approaches.

Load results from: /share/u/lofty/code_llm/debug/results/attention_head_patching/20250717_142425/Qwen_Qwen3-14B
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to PYTHONPATH
import sys
project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(project_root))

from debug.causal_tracing import InterventionResult

#%%
# Configuration - Update this path to your experiment results
RESULTS_DIR = Path("/share/u/lofty/code_llm/debug/results/attention_head_patching/20250717_142425/Qwen_Qwen3-14B")
RESULTS_FILE = RESULTS_DIR / "intervention_results.json"
METADATA_FILE = RESULTS_DIR / "experiment_metadata.json"

print(f"Loading results from: {RESULTS_DIR}")
print(f"Results file exists: {RESULTS_FILE.exists()}")
print(f"Metadata file exists: {METADATA_FILE.exists()}")

#%%
# Load experiment results
def load_experiment_data():
    """Load and parse the experiment results."""
    
    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Load results
    with open(RESULTS_FILE, 'r') as f:
        raw_results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(raw_results)
    
    print(f"Experiment metadata:")
    print(f"  Model: {metadata['model_id']}")
    print(f"  RNG Seed: {metadata['rng_seed']}")
    print(f"  Program: {metadata['program']}")
    print(f"  Query Variable: {metadata['query_var']}")
    print(f"  Expected Answer: {metadata['expected_answer']}")
    print(f"  Intervention Targets: {metadata['intervention_targets']}")
    print(f"  Total Interventions: {len(df)}")
    
    return df, metadata

df, metadata = load_experiment_data()

#%%
# Basic data exploration
print("Dataset shape:", df.shape)
print("\\nColumns:", df.columns.tolist())
print("\\nBasic statistics:")
print(df[['layer_idx', 'head_idx', 'normalized_logit_difference', 'success_rate']].describe())

print("\\nLayers tested:", sorted(df['layer_idx'].unique()))
print("Heads tested:", sorted(df['head_idx'].unique()))

#%%
# Extract target information from target_name column
def parse_target_names(df):
    """Extract clean target names from the target_name column."""
    # Extract the target type from the full target name
    df['target_type'] = df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')
    
    # Create a mapping for cleaner names
    target_mapping = {
        'ref_depth_1_rhs': 'Ref Depth 1 (RHS)',
        'ref_depth_2_rhs': 'Ref Depth 2 (RHS)', 
        'ref_depth_3_rhs': 'Ref Depth 3 (RHS)',
        'prediction_token_pos': 'Prediction Token',
        'query_var': 'Query Variable'
    }
    
    df['target_clean'] = df['target_type'].map(target_mapping).fillna(df['target_type'])
    
    return df

df = parse_target_names(df)
print("\\nTarget types found:", df['target_clean'].unique())

#%%
# Function 1: Layer-wise heatmap (original approach but improved)
def plot_layer_heatmap(df, layer_idx, metric='normalized_logit_difference', figsize=(14, 8)):
    """Plot heatmap for a specific layer."""
    
    layer_data = df[df['layer_idx'] == layer_idx].copy()
    
    if layer_data.empty:
        print(f"No data for layer {layer_idx}")
        return None
        
    # Create pivot table
    pivot_data = layer_data.pivot_table(
        values=metric,
        index='head_idx',
        columns='target_clean',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate color scale
    vmax = max(abs(df[metric].min()), abs(df[metric].max()))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': metric.replace('_', ' ').title()},
        linewidths=0.5
    )
    
    ax.set_title(f'Layer {layer_idx}: Attention Head Causal Effects', fontsize=16, fontweight='bold')
    ax.set_xlabel('Intervention Target', fontsize=12)
    ax.set_ylabel('Attention Head', fontsize=12)
    
    plt.tight_layout()
    return fig

# Test with layer 0
for i in range(df['layer_idx'].max()):
    fig = plot_layer_heatmap(df, i)
    plt.show()

#%%
# Function 2: Summary heatmap across all layers
def plot_summary_heatmap(df, metric='normalized_logit_difference', figsize=(16, 12)):
    """Plot summary heatmap showing average effects across all layers."""
    
    # Average across layers for each head-target combination
    summary_data = df.groupby(['head_idx', 'target_clean'])[metric].mean().reset_index()
    
    pivot_summary = summary_data.pivot_table(
        values=metric,
        index='head_idx',
        columns='target_clean',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = max(abs(df[metric].min()), abs(df[metric].max()))
    
    sns.heatmap(
        pivot_summary,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': f'Average {metric.replace("_", " ").title()}'},
        linewidths=0.5
    )
    
    ax.set_title('Average Attention Head Effects Across All Layers', fontsize=16, fontweight='bold')
    ax.set_xlabel('Intervention Target', fontsize=12)
    ax.set_ylabel('Attention Head', fontsize=12)
    
    plt.tight_layout()
    return fig

fig = plot_summary_heatmap(df)
plt.show()

#%%
# Function 3: Target-specific analysis across layers
def plot_target_across_layers(df, target_name, metric='normalized_logit_difference', figsize=(16, 10)):
    """Plot how a specific intervention target affects different heads across layers."""
    
    target_data = df[df['target_clean'] == target_name].copy()
    
    if target_data.empty:
        print(f"No data for target: {target_name}")
        return None
    
    pivot_data = target_data.pivot_table(
        values=metric,
        index='layer_idx',
        columns='head_idx',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    vmax = max(abs(target_data[metric].min()), abs(target_data[metric].max()))
    
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=False,  # Too many numbers for layers x heads
        cbar_kws={'label': metric.replace('_', ' ').title()},
        linewidths=0.1
    )
    
    ax.set_title(f'{target_name}: Effects Across Layers and Heads', fontsize=16, fontweight='bold')
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    plt.tight_layout()
    return fig

# Test with each target
for target in df['target_clean'].unique():
    print(f"\\nPlotting: {target}")
    fig = plot_target_across_layers(df, target)
    if fig:
        plt.show()

#%%
# Function 4: Find and visualize top effects
def find_top_effects(df, metric='normalized_logit_difference', n=20):
    """Find the top N most significant effects."""
    
    # Sort by absolute value of the metric
    df_sorted = df.copy()
    df_sorted['abs_effect'] = df_sorted[metric].abs()
    top_effects = df_sorted.nlargest(n, 'abs_effect')
    
    print(f"Top {n} effects by |{metric}|:")
    for i, (_, row) in enumerate(top_effects.iterrows(), 1):
        print(f"{i:2d}. L{row['layer_idx']:2d}H{row['head_idx']:2d} {row['target_clean']:20s} {row[metric]:+.4f}")
    
    return top_effects

def plot_top_effects_bar(df, metric='normalized_logit_difference', n=15, figsize=(12, 8)):
    """Bar plot of top effects."""
    
    top_effects = find_top_effects(df, metric, n)
    
    # Create labels
    labels = [f"L{row['layer_idx']}H{row['head_idx']}\\n{row['target_clean']}" 
              for _, row in top_effects.iterrows()]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red' if x < 0 else 'blue' for x in top_effects[metric]]
    bars = ax.bar(range(len(top_effects)), top_effects[metric], color=colors, alpha=0.7)
    
    ax.set_xticks(range(len(top_effects)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Top {n} Attention Head Effects', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

top_effects = find_top_effects(df)
fig = plot_top_effects_bar(df)
plt.show()

#%%
# Function 5: Distribution analysis
def plot_effect_distributions(df, figsize=(15, 5)):
    """Plot distributions of different metrics."""
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Normalized logit difference distribution
    axes[0].hist(df['normalized_logit_difference'], bins=50, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Normalized Logit Difference')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Logit Differences')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Success rate distribution
    axes[1].hist(df['success_rate'], bins=50, alpha=0.7, color='green')
    axes[1].set_xlabel('Success Rate')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Success Rates')
    axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Random chance')
    axes[1].legend()
    
    # Effects by target type
    df.boxplot(column='normalized_logit_difference', by='target_clean', ax=axes[2])
    axes[2].set_xlabel('Target Type')
    axes[2].set_ylabel('Normalized Logit Difference')
    axes[2].set_title('Effects by Target Type')
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('')  # Remove automatic suptitle from boxplot
    plt.tight_layout()
    return fig

fig = plot_effect_distributions(df)
plt.show()

#%%
# Function 6: Layer progression analysis
def plot_layer_progression(df, metric='normalized_logit_difference', figsize=(12, 8)):
    """Show how effects change across layers."""
    
    # Calculate statistics by layer
    layer_stats = df.groupby('layer_idx')[metric].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean with error bars
    ax.errorbar(layer_stats['layer_idx'], layer_stats['mean'], 
                yerr=layer_stats['std'], fmt='o-', capsize=3, capthick=1,
                label='Mean ± Std', color='steelblue', linewidth=2)
    
    # Plot min/max envelope
    ax.fill_between(layer_stats['layer_idx'], layer_stats['min'], layer_stats['max'], 
                   alpha=0.2, color='steelblue', label='Min-Max Range')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Attention Head Effects Across Model Layers', fontsize=14, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

fig = plot_layer_progression(df)
plt.show()

#%%
# Function 7: Correlation analysis
def plot_correlation_analysis(df, figsize=(10, 8)):
    """Analyze correlations between different metrics and factors."""
    
    # Create correlation matrix
    corr_data = df[['layer_idx', 'head_idx', 'normalized_logit_difference', 'success_rate']].copy()
    corr_matrix = corr_data.corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax1,
                square=True, linewidths=0.5)
    ax1.set_title('Metric Correlations')
    
    # Scatter plot: logit difference vs success rate
    scatter = ax2.scatter(df['normalized_logit_difference'], df['success_rate'], 
                         alpha=0.5, c=df['layer_idx'], cmap='viridis')
    ax2.set_xlabel('Normalized Logit Difference')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Logit Difference vs Success Rate')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for layer info
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Layer Index')
    
    plt.tight_layout()
    return fig

fig = plot_correlation_analysis(df)
plt.show()

#%%
# Function 8: Interactive layer explorer
def explore_layers_interactively(df, start_layer=0, end_layer=5):
    """Create subplots showing multiple layers at once for comparison."""
    
    n_layers = min(end_layer - start_layer + 1, 6)  # Max 6 subplots
    layers_to_show = list(range(start_layer, start_layer + n_layers))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Calculate global color scale
    vmax = max(abs(df['normalized_logit_difference'].min()), 
               abs(df['normalized_logit_difference'].max()))
    
    for i, layer_idx in enumerate(layers_to_show):
        if i >= len(axes):
            break
            
        layer_data = df[df['layer_idx'] == layer_idx].copy()
        
        if layer_data.empty:
            axes[i].text(0.5, 0.5, f'No data\\nfor layer {layer_idx}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Layer {layer_idx}')
            continue
        
        # Create pivot table
        pivot_data = layer_data.pivot_table(
            values='normalized_logit_difference',
            index='head_idx',
            columns='target_clean',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=axes[i],
            annot=False,  # Too small for annotations
            cbar=False,   # Single colorbar for all
            linewidths=0.1
        )
        
        axes[i].set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    # Hide unused subplots
    for i in range(len(layers_to_show), len(axes)):
        axes[i].set_visible(False)
    
    # Add single colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', 
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6)
    cbar.set_label('Normalized Logit Difference', fontsize=12)
    
    plt.suptitle(f'Attention Head Effects: Layers {start_layer}-{start_layer + n_layers - 1}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Explore first few layers
fig = explore_layers_interactively(df, 0, 5)
plt.show()

#%%
# Function 9: Export capabilities
def save_key_visualizations(df, output_dir):
    """Save key visualizations to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Saving visualizations to: {output_path}")
    
    # 1. Summary heatmap
    fig = plot_summary_heatmap(df)
    fig.savefig(output_path / "summary_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Top effects
    fig = plot_top_effects_bar(df)
    fig.savefig(output_path / "top_effects.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Layer progression
    fig = plot_layer_progression(df)
    fig.savefig(output_path / "layer_progression.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Distributions
    fig = plot_effect_distributions(df)
    fig.savefig(output_path / "distributions.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Individual layers (first 10)
    for layer_idx in range(min(10, df['layer_idx'].max() + 1)):
        fig = plot_layer_heatmap(df, layer_idx)
        if fig:
            fig.savefig(output_path / f"layer_{layer_idx:02d}_heatmap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print("Key visualizations saved!")

# Uncomment to save visualizations
# save_key_visualizations(df, "/tmp/attention_visualizations")

#%%
# Summary and key insights
print("=== EXPERIMENT SUMMARY ===")
print(f"Model: {metadata['model_id']}")
print(f"Total interventions: {len(df)}")
print(f"Layers tested: {df['layer_idx'].nunique()} (0-{df['layer_idx'].max()})")
print(f"Heads tested: {df['head_idx'].nunique()} (0-{df['head_idx'].max()})")
print(f"Target types: {df['target_clean'].nunique()}")

print("\\n=== KEY STATISTICS ===")
print(f"Mean normalized logit difference: {df['normalized_logit_difference'].mean():.4f}")
print(f"Std normalized logit difference: {df['normalized_logit_difference'].std():.4f}")
print(f"Max |normalized logit difference|: {df['normalized_logit_difference'].abs().max():.4f}")
print(f"Mean success rate: {df['success_rate'].mean():.4f}")
print(f"Max success rate: {df['success_rate'].max():.4f}")

# Significant effects (|logit_diff| > 0.1)
significant = df[df['normalized_logit_difference'].abs() > 0.1]
print(f"\\nSignificant effects (|logit_diff| > 0.1): {len(significant)} ({100*len(significant)/len(df):.1f}%)")

print("\\n🎯 Use the functions above to explore specific aspects of the results!")
print("📊 Example: plot_layer_heatmap(df, layer_idx=15) for layer 15")
print("🔍 Example: plot_target_across_layers(df, 'Query Variable') for query variable effects")

#%%