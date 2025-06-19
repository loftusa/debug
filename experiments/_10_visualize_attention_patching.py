#%%
"""
Visualize attention head patching results.

This script loads the data from the most recent attention head patching experiment
and provides visualizations specifically designed for attention head analysis.
It creates heatmaps showing which attention heads at which layers are most
important for variable binding tasks.
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
BASE_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "attention_head_patching"
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

# Look for results in model subdirectory (typically Qwen_Qwen3-0.6B)
model_dirs = [d for d in latest_experiment_dir.iterdir() if d.is_dir()]
if model_dirs:
    results_dir = model_dirs[0]  # Take the first (and likely only) model directory
    RESULTS_FILE = results_dir / "intervention_results.json"
else:
    RESULTS_FILE = latest_experiment_dir / "intervention_results.json"

if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"Results file not found in: {latest_experiment_dir}")

print(f"Loading attention head results from: {RESULTS_FILE}")
# ---------------------------------------------------------------------------


def load_results_as_dataframe(path: Path) -> pd.DataFrame:
    """Load attention head patching results into a pandas DataFrame."""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle both old list format and new dictionary format
    if isinstance(data, dict) and "intervention_results" in data:
        return pd.DataFrame(data["intervention_results"])
    else:
        return pd.DataFrame(data)


def plot_attention_head_heatmap(df: pd.DataFrame, metric: str = "normalized_logit_difference") -> plt.Figure:
    """
    Create a heatmap showing attention head effects across layers and heads.
    
    Args:
        df: DataFrame with attention head intervention results
        metric: Which metric to plot ("normalized_logit_difference", "success_rate", etc.)
    
    Returns:
        matplotlib Figure object
    """
    # Aggregate results across token positions to get layer × head effects
    aggregated = df.groupby(['layer_idx', 'head_idx'])[metric].mean().reset_index()
    
    # Create pivot table for heatmap
    heatmap_data = aggregated.pivot(index='layer_idx', columns='head_idx', values=metric)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a diverging colormap centered at 0
    vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    ax.set_title(f'Attention Head Effects: {metric.replace("_", " ").title()}')
    ax.set_xlabel('Attention Head Index')
    ax.set_ylabel('Layer Index')
    
    plt.tight_layout()
    return fig


def plot_top_attention_heads(df: pd.DataFrame, metric: str = "normalized_logit_difference", top_k: int = 10) -> plt.Figure:
    """
    Plot the top-k most effective attention heads.
    
    Args:
        df: DataFrame with attention head intervention results
        metric: Which metric to rank by
        top_k: Number of top heads to show
    
    Returns:
        matplotlib Figure object
    """
    # Calculate mean effect for each head across all positions
    head_effects = df.groupby(['layer_idx', 'head_idx'])[metric].mean().reset_index()
    
    # Sort by absolute effect size and take top k
    head_effects['abs_effect'] = head_effects[metric].abs()
    top_heads = head_effects.nlargest(top_k, 'abs_effect')
    
    # Create labels combining layer and head info
    top_heads['head_label'] = top_heads.apply(lambda x: f"L{x['layer_idx']}H{x['head_idx']}", axis=1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(top_heads)), top_heads[metric], 
                  color=['red' if x < 0 else 'blue' for x in top_heads[metric]])
    
    ax.set_xticks(range(len(top_heads)))
    ax.set_xticklabels(top_heads['head_label'], rotation=45)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Top {top_k} Attention Heads by {metric.replace("_", " ").title()}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_heads[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    return fig


def plot_position_effects(df: pd.DataFrame, layer_idx: int, head_idx: int) -> plt.Figure:
    """
    Plot intervention effects across token positions for a specific attention head.
    
    Args:
        df: DataFrame with attention head intervention results
        layer_idx: Which layer to analyze
        head_idx: Which head to analyze
    
    Returns:
        matplotlib Figure object
    """
    # Filter for the specific head
    head_data = df[(df['layer_idx'] == layer_idx) & (df['head_idx'] == head_idx)].copy()
    head_data = head_data.sort_values('target_token_pos')
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot normalized logit difference
    ax1.plot(head_data['target_token_pos'], head_data['normalized_logit_difference'], 
             'b-o', markersize=4, linewidth=2)
    ax1.set_ylabel('Normalized Logit Difference')
    ax1.set_title(f'Layer {layer_idx}, Head {head_idx}: Effect by Token Position')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot success rate
    ax2.plot(head_data['target_token_pos'], head_data['success_rate'], 
             'r-s', markersize=4, linewidth=2)
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Token Position')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random chance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add token labels if available
    if not head_data.empty and 'token_labels' in head_data.columns:
        try:
            token_labels = head_data['token_labels'].iloc[0]
            if token_labels and len(token_labels) > 0:
                # Show every few tokens to avoid crowding
                step = max(1, len(token_labels) // 10)
                for i in range(0, len(token_labels), step):
                    if i < len(token_labels):
                        ax2.text(i, -0.1, str(token_labels[i])[:8], 
                                rotation=45, ha='right', va='top', 
                                transform=ax2.get_xaxis_transform())
        except Exception:
            pass
    
    plt.tight_layout()
    return fig


def clean_token_labels(token_labels):
    """Clean up tokenizer artifacts for better visualization."""
    if not token_labels:
        return token_labels
    
    clean_labels = []
    for token in token_labels:
        if isinstance(token, str):
            clean_token = token.replace('\u0120', ' ')  # Ġ space prefix
            clean_token = clean_token.replace('\u010a', '\\n')  # newline
            clean_token = clean_token.replace('Ġ', ' ')  # Direct Ġ character
            clean_token = clean_token.replace('▁', ' ')  # SentencePiece space
            
            if clean_token.startswith('##'):
                clean_token = clean_token[2:]
                
            clean_labels.append(clean_token)
        else:
            clean_labels.append(str(token))
    
    return clean_labels


#%%
# Load the results into a pandas DataFrame
results_df = load_results_as_dataframe(RESULTS_FILE)
print("Attention head results loaded successfully. DataFrame preview:")
print(results_df.head())
print(f"\nDataset shape: {results_df.shape}")
print(f"Layers tested: {sorted(results_df['layer_idx'].unique())}")
print(f"Heads tested: {sorted(results_df['head_idx'].unique())}")
print(f"Token positions: {sorted(results_df['target_token_pos'].unique())}")

#%%
# --- Visualization Examples -----------------------------------------------

# 1. Overall attention head heatmap
print("\nGenerating attention head heatmap...")
fig1 = plot_attention_head_heatmap(results_df, metric="normalized_logit_difference")
plt.show()

#%%
# 2. Top performing attention heads
print("\nGenerating top attention heads plot...")
fig2 = plot_top_attention_heads(results_df, metric="normalized_logit_difference", top_k=15)
plt.show()

#%%
# 3. Success rate heatmap
print("\nGenerating success rate heatmap...")
fig3 = plot_attention_head_heatmap(results_df, metric="success_rate")
plt.show()

#%%
# 4. Position-specific analysis for the best head
# Find the head with the highest absolute effect
head_effects = results_df.groupby(['layer_idx', 'head_idx'])['normalized_logit_difference'].mean().abs()
best_head = head_effects.idxmax()
best_layer, best_head_idx = best_head

print(f"\nGenerating position analysis for best head: Layer {best_layer}, Head {best_head_idx}")
fig4 = plot_position_effects(results_df, best_layer, best_head_idx)
plt.show()

#%%
# 5. Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total interventions: {len(results_df)}")
print(f"Mean normalized logit difference: {results_df['normalized_logit_difference'].mean():.4f}")
print(f"Max normalized logit difference: {results_df['normalized_logit_difference'].max():.4f}")
print(f"Mean success rate: {results_df['success_rate'].mean():.4f}")
print(f"Max success rate: {results_df['success_rate'].max():.4f}")

# Find heads with significant effects (|logit_diff| > 0.1)
significant_effects = results_df[results_df['normalized_logit_difference'].abs() > 0.1]
print(f"Interventions with |logit_diff| > 0.1: {len(significant_effects)}/{len(results_df)} ({100*len(significant_effects)/len(results_df):.1f}%)")

# Find heads with high success rate (> 0.7)
high_success = results_df[results_df['success_rate'] > 0.7]
print(f"Interventions with success rate > 0.7: {len(high_success)}/{len(results_df)} ({100*len(high_success)/len(results_df):.1f}%)")

# %%