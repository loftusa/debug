#%%
"""
Visualize MLP token patching results.

This script loads the data from the most recent MLP token patching experiment
and provides visualizations specifically designed for MLP layer analysis.
It creates heatmaps showing which MLP layers are most important for variable binding tasks.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from typing import List, Optional

# Add src to PYTHONPATH for custom module imports
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parents[1] / "src"
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from debug.causal_visualization import plot_causal_flow_heatmap
from debug.causal_tracing import InterventionResult

# --- Configuration ---------------------------------------------------------
# Find the latest experiment directory automatically
BASE_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mlp_token_patching"
if not BASE_RESULTS_DIR.exists():
    raise FileNotFoundError(f"Base results directory not found: {BASE_RESULTS_DIR}")

# Get the most recent timestamped experiment folder
try:
    latest_experiment_dir = sorted(
        [d for d in BASE_RESULTS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True
    )[0]
except IndexError:
    latest_experiment_dir = BASE_RESULTS_DIR

# Look for results in model subdirectory (typically Qwen_Qwen3-14B)
model_dirs = [d for d in latest_experiment_dir.iterdir() if d.is_dir()]
if model_dirs:
    results_dir = model_dirs[0]  # Take the first (and likely only) model directory
    RESULTS_FILE = results_dir / "intervention_results.json"
else:
    RESULTS_FILE = latest_experiment_dir / "intervention_results.json"

if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"Results file not found in: {latest_experiment_dir}")

print(f"Loading MLP results from: {RESULTS_FILE}")
# ---------------------------------------------------------------------------


def load_results_as_dataframe(path: Path) -> pd.DataFrame:
    """Load MLP token patching results into a pandas DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle both old list format and new dictionary format
    if isinstance(data, dict) and "intervention_results" in data:
        df = pd.DataFrame(data["intervention_results"])
    else:
        df = pd.DataFrame(data)
    
    # Extract token information for each intervention
    if 'token_labels' in df.columns and 'target_token_pos' in df.columns:
        df['actual_token'] = df.apply(
            lambda row: extract_actual_token(row['token_labels'], row['target_token_pos']),
            axis=1
        )
        df['target_description'] = df.apply(
            lambda row: create_target_description(row['target_name'], row.get('actual_token', '')),
            axis=1
        )

    df = df[df['target_description'] != 'Final Space']
    
    return df


def extract_actual_token(token_labels, target_pos):
    """Extract the actual token at the target position."""
    if not token_labels or target_pos is None:
        return ""
    
    if isinstance(token_labels, list) and 0 <= target_pos < len(token_labels):
        token = token_labels[target_pos]
        # Clean up common tokenizer artifacts
        if isinstance(token, str):
            token = token.replace('Ġ', ' ').replace('Ċ', '\\n').strip()
        return token
    return ""


def create_target_description(target_name, actual_token):
    """Create a human-readable description of the intervention target."""
    # Extract the target type from the target_name
    if not target_name:
        return "Unknown"
    
    # Pattern: "target_type_layer_N" -> extract "target_type"
    parts = target_name.split('_layer_')
    if len(parts) > 0:
        target_type = parts[0]
        # Make it more readable
        if target_type == "query_pos":
            return f"Query Position (#var:)"
        elif target_type == "root_pos":
            return f"Root Value (={actual_token})" if actual_token else "Root Value"
        elif target_type.startswith("ref_depth"):
            depth = target_type.split('_')[-2]  # Extract depth number
            return f"Ref Depth {depth} ({actual_token})" if actual_token else f"Ref Depth {depth}"
        else:
            return f"{target_type.replace('_', ' ').title()}"
    return target_name


#%%
# Load the results
df = load_results_as_dataframe(RESULTS_FILE)
print(f"Loaded {len(df)} MLP intervention results")
print(f"Layers tested: {sorted(df['layer_idx'].unique())}")
print(f"Unique intervention targets: {df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0].unique()}")


#%%
def plot_mlp_layer_heatmap(df: pd.DataFrame, metric: str = "normalized_logit_difference") -> plt.Figure:
    """
    Create a heatmap showing MLP effects across all layers and intervention targets.
    
    Args:
        df: DataFrame with MLP intervention results
        metric: Which metric to plot
    
    Returns:
        matplotlib Figure object
    """
    # Aggregate by layer and target type
    df['target_type'] = df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0]
    aggregated = df.groupby(['layer_idx', 'target_type'])[metric].mean().reset_index()
    
    # Pivot to create heatmap data
    heatmap_data = aggregated.pivot(index='layer_idx', columns='target_type', values=metric)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get color scale
    vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=False,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    ax.set_title(f'MLP Layer Effects: {metric.replace("_", " ").title()}')
    ax.set_xlabel('Intervention Target')
    ax.set_ylabel('Layer Index')
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


#%%
def plot_mlp_effects_by_target(df: pd.DataFrame, metric: str = "normalized_logit_difference") -> plt.Figure:
    """
    Create separate plots for each intervention target showing MLP effects across layers.
    
    Args:
        df: DataFrame with MLP intervention results
        metric: Which metric to plot
    
    Returns:
        matplotlib Figure object
    """
    # Get unique targets with descriptions
    if 'target_description' in df.columns:
        target_groups = df.groupby('target_description')
    else:
        # Fallback to extracting from target_name
        df['target_type'] = df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0]
        target_groups = df.groupby('target_type')
    
    n_targets = len(target_groups)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 8))
    if n_targets == 1:
        axes = [axes]
    
    # Get global y-axis limits for consistency
    ymin = df[metric].min()
    ymax = df[metric].max()
    y_range = ymax - ymin
    ylim = (ymin - 0.05 * y_range, ymax + 0.05 * y_range)
    
    for idx, (target_desc, target_df) in enumerate(target_groups):
        # Aggregate for this target across layers
        layer_effects = target_df.groupby('layer_idx')[metric].mean().reset_index()
        
        # Create line plot
        axes[idx].plot(layer_effects['layer_idx'], layer_effects[metric], 
                      marker='o', linewidth=2, markersize=6)
        
        # Add horizontal line at zero
        axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Styling
        axes[idx].set_title(f'{target_desc}')
        axes[idx].set_xlabel('Layer' if idx == n_targets // 2 else '')
        axes[idx].set_ylabel(metric.replace('_', ' ').title() if idx == 0 else '')
        axes[idx].set_ylim(ylim)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'MLP Effects by Intervention Target\n{metric.replace("_", " ").title()}', fontsize=14)
    plt.tight_layout()
    return fig


#%%
def plot_top_mlp_layers(df: pd.DataFrame, metric: str = "normalized_logit_difference", top_k: int = 10) -> plt.Figure:
    """
    Plot the top-k most effective MLP layers.
    
    Args:
        df: DataFrame with MLP intervention results
        metric: Which metric to rank by
        top_k: Number of top layers to show
    
    Returns:
        matplotlib Figure object
    """
    # Calculate mean effect for each layer across all positions
    layer_effects = df.groupby('layer_idx')[metric].mean().reset_index()
    
    # Sort by absolute effect size and take top k
    layer_effects['abs_effect'] = layer_effects[metric].abs()
    top_layers = layer_effects.nlargest(top_k, 'abs_effect')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(top_layers)), top_layers[metric], 
                  color=['red' if x < 0 else 'blue' for x in top_layers[metric]])
    
    ax.set_xticks(range(len(top_layers)))
    ax.set_xticklabels([f"L{x}" for x in top_layers['layer_idx']], rotation=45)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Top {top_k} MLP Layers by {metric.replace("_", " ").title()}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_layers[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    return fig


#%%
def plot_mlp_layer_progression(df: pd.DataFrame, metric: str = "normalized_logit_difference") -> plt.Figure:
    """
    Plot how MLP effects progress through layers for different target types.
    
    Args:
        df: DataFrame with MLP intervention results
        metric: Which metric to plot
    
    Returns:
        matplotlib Figure object
    """
    # Get target types
    if 'target_description' in df.columns:
        target_col = 'target_description'
    else:
        df['target_type'] = df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0]
        target_col = 'target_type'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each target type
    for target in df[target_col].unique():
        target_df = df[df[target_col] == target]
        layer_effects = target_df.groupby('layer_idx')[metric].mean().reset_index()
        
        ax.plot(layer_effects['layer_idx'], layer_effects[metric], 
               marker='o', linewidth=2, markersize=6, label=target)
    
    # Styling
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'MLP Layer Effects Progression\n{metric.replace("_", " ").title()}', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


#%%
def plot_mlp_intervention_summary(df: pd.DataFrame) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple panels.
    
    Args:
        df: DataFrame with MLP intervention results
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.3, wspace=0.3)
    
    # 1. Main heatmap (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Aggregate data for heatmap
    df['target_type'] = df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0]
    aggregated = df.groupby(['layer_idx', 'target_type'])['normalized_logit_difference'].mean().reset_index()
    heatmap_data = aggregated.pivot(index='layer_idx', columns='target_type', values='normalized_logit_difference')
    
    vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, 
                ax=ax1, cbar_kws={'label': 'Normalized Logit Difference'})
    ax1.set_title('MLP Layer Effects Across All Intervention Targets', fontsize=14)
    ax1.set_xlabel('Intervention Target')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Top layers bar chart (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    layer_effects = df.groupby('layer_idx')['normalized_logit_difference'].mean().reset_index()
    layer_effects['abs_effect'] = layer_effects['normalized_logit_difference'].abs()
    top_layers = layer_effects.nlargest(10, 'abs_effect')
    
    colors = ['red' if x < 0 else 'blue' for x in top_layers['normalized_logit_difference']]
    bars = ax2.bar(range(len(top_layers)), top_layers['normalized_logit_difference'], color=colors)
    ax2.set_xticks(range(len(top_layers)))
    ax2.set_xticklabels([f"L{x}" for x in top_layers['layer_idx']], rotation=45)
    ax2.set_ylabel('Mean Effect')
    ax2.set_title('Top 10 Most Effective MLP Layers')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Effect distribution (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.hist(df['normalized_logit_difference'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Normalized Logit Difference')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of MLP Effects')
    
    # 4. Layer progression plot (bottom, spanning both columns)
    ax4 = fig.add_subplot(gs[2, :])
    
    for target in df['target_type'].unique():
        target_df = df[df['target_type'] == target]
        layer_means = target_df.groupby('layer_idx')['normalized_logit_difference'].mean()
        ax4.plot(layer_means.index, layer_means.values, marker='o', label=target, linewidth=2)
    
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Mean Normalized Logit Difference')
    ax4.set_title('MLP Effect Progression Through Layers')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('MLP Token Patching Results Summary', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig


#%%
# Create visualizations
print("\n=== Creating MLP Visualizations ===")

# 1. Overall heatmap
fig1 = plot_mlp_layer_heatmap(df)
plt.show()

#%%
# 2. Effects by target
fig2 = plot_mlp_effects_by_target(df)
plt.show()

#%%
# 3. Top MLP layers
fig3 = plot_top_mlp_layers(df, top_k=15)
plt.show()

#%%
# 4. Layer progression
fig4 = plot_mlp_layer_progression(df)
plt.show()

#%%
# 5. Comprehensive summary
fig5 = plot_mlp_intervention_summary(df)
plt.show()

#%%
# Print summary statistics
print("\n=== MLP Intervention Summary Statistics ===")
print(f"Total interventions: {len(df)}")
print(f"Number of layers tested: {df['layer_idx'].nunique()}")
print(f"Number of intervention targets: {df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0].nunique()}")
print(f"\nMean normalized logit difference: {df['normalized_logit_difference'].mean():.4f}")
print(f"Std normalized logit difference: {df['normalized_logit_difference'].std():.4f}")
print(f"Max |normalized logit difference|: {df['normalized_logit_difference'].abs().max():.4f}")

# Find most effective interventions
print("\n=== Top 10 Most Effective MLP Interventions ===")
top_interventions = df.nlargest(10, 'normalized_logit_difference')[['layer_idx', 'target_name', 'normalized_logit_difference']]
for idx, row in top_interventions.iterrows():
    target_type = row['target_name'].split('_layer_')[0]
    print(f"Layer {row['layer_idx']}, {target_type}: {row['normalized_logit_difference']:.4f}")

# Compare effect magnitudes between target types
print("\n=== Mean Effects by Target Type ===")
target_effects = df.groupby(df['target_name'].str.extract(r'^([^_]+(?:_[^_]+)*?)_layer_')[0])['normalized_logit_difference'].agg(['mean', 'std', 'max'])
print(target_effects.round(4))

#%%
# Optional: Save key figures
save_figs = input("\nSave figures? (y/n): ").lower() == 'y'
if save_figs:
    output_dir = Path(RESULTS_FILE).parent / "additional_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    fig1.savefig(output_dir / "mlp_layer_heatmap.png", dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "mlp_effects_by_target.png", dpi=300, bbox_inches='tight')
    fig3.savefig(output_dir / "top_mlp_layers.png", dpi=300, bbox_inches='tight')
    fig4.savefig(output_dir / "mlp_layer_progression.png", dpi=300, bbox_inches='tight')
    fig5.savefig(output_dir / "mlp_intervention_summary.png", dpi=300, bbox_inches='tight')
    
    print(f"\nFigures saved to: {output_dir}")
# %%
