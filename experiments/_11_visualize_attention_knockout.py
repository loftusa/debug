"""
Attention Pattern Knockout Visualization

This script provides comprehensive visualization and analysis tools for the
attention pattern knockout experiment results. It helps determine whether
the model uses direct attention pointers vs step-by-step chain following.

The visualizations include:
- Direct vs Chain knockout effect comparisons
- Layer-wise breakdown of intervention effects
- Heatmaps showing which heads are most affected
- Statistical analysis of mechanism dominance

Usage:
    uv run experiments/_11_visualize_attention_knockout.py
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
if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parents[1] / "src" 
    sys.path.append(str(project_root))


def find_latest_experiment_results() -> Path:
    """Find the most recent attention knockout experiment results."""
    results_base = Path(__file__).resolve().parents[1] / "results" / "attention_knockout"
    
    if not results_base.exists():
        raise FileNotFoundError(f"No results directory found: {results_base}")
    
    # Get most recent timestamped directory
    experiment_dirs = [d for d in results_base.iterdir() if d.is_dir()]
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {results_base}")
    
    latest_dir = sorted(experiment_dirs, key=lambda d: d.name)[-1]
    
    results_file = latest_dir / "knockout_results.json"
    analysis_file = latest_dir / "knockout_analysis.json"
    
    if not results_file.exists() or not analysis_file.exists():
        raise FileNotFoundError(f"Results files not found in: {latest_dir}")
    
    print(f"Loading results from: {latest_dir}")
    return latest_dir


def load_experiment_data(results_dir: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load experimental results and analysis."""
    
    # Load raw results
    with open(results_dir / "knockout_results.json", 'r') as f:
        raw_results = json.load(f)
    
    # Load analysis
    with open(results_dir / "knockout_analysis.json", 'r') as f:
        analysis = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_results)
    
    print(f"Loaded {len(df)} experimental results")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Layers: {sorted(df['layer_idx'].unique())}")
    print(f"Heads per layer: {df[df['layer_idx'] == df['layer_idx'].iloc[0]]['head_idx'].nunique()}")
    
    return df, analysis


def plot_mechanism_comparison(df: pd.DataFrame, analysis: Dict[str, Any]) -> plt.Figure:
    """
    Create bar chart comparing direct vs chain knockout effects.
    
    Shows the overall strength of each mechanism across all layers and heads.
    """
    # Calculate mean effects for each mechanism
    direct_data = df[df['condition'] == 'direct_knockout']
    chain_data = df[df['condition'] == 'chain_knockout']
    
    mechanisms = ['Direct Pointer', 'Chain Following']
    mean_effects = [
        direct_data['knockout_effect'].abs().mean(),
        chain_data['knockout_effect'].abs().mean() if len(chain_data) > 0 else 0.0
    ]
    max_effects = [
        direct_data['knockout_effect'].abs().max(),
        chain_data['knockout_effect'].abs().max() if len(chain_data) > 0 else 0.0
    ]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean effects comparison
    colors = ['#2E86AB' if mechanisms[i] == analysis['dominant_mechanism'].replace('_', ' ').title() 
              else '#A23B72' for i in range(len(mechanisms))]
    
    bars1 = ax1.bar(mechanisms, mean_effects, color=colors, alpha=0.7)
    ax1.set_ylabel('Mean Knockout Effect')
    ax1.set_title('Average Intervention Impact')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mean_effects):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Max effects comparison
    bars2 = ax2.bar(mechanisms, max_effects, color=colors, alpha=0.7)
    ax2.set_ylabel('Maximum Knockout Effect')
    ax2.set_title('Peak Intervention Impact')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, max_effects):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.suptitle(f'Attention Mechanism Comparison\nDominant: {analysis["dominant_mechanism"].replace("_", " ").title()} '
                f'({analysis["dominance_ratio"]:.1f}x stronger)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_layer_breakdown(df: pd.DataFrame) -> plt.Figure:
    """
    Show knockout effects broken down by layer.
    
    This reveals at which layers the different mechanisms are most important.
    """
    # Aggregate by layer and condition
    layer_effects = df.groupby(['layer_idx', 'condition'])['knockout_effect'].agg(['mean', 'std']).reset_index()
    
    layers = sorted(df['layer_idx'].unique())
    direct_means = []
    direct_stds = []
    chain_means = []
    chain_stds = []
    
    for layer in layers:
        layer_data = layer_effects[layer_effects['layer_idx'] == layer]
        
        direct_row = layer_data[layer_data['condition'] == 'direct_knockout']
        if len(direct_row) > 0:
            direct_means.append(abs(direct_row['mean'].iloc[0]))
            direct_stds.append(direct_row['std'].iloc[0])
        else:
            direct_means.append(0.0)
            direct_stds.append(0.0)
        
        chain_row = layer_data[layer_data['condition'] == 'chain_knockout']
        if len(chain_row) > 0:
            chain_means.append(abs(chain_row['mean'].iloc[0]))
            chain_stds.append(chain_row['std'].iloc[0])
        else:
            chain_means.append(0.0)
            chain_stds.append(0.0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layer_labels = [f'Layer {l}' for l in layers]
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, direct_means, width, yerr=direct_stds, 
                   label='Direct Pointer Knockout', color='#2E86AB', alpha=0.7, capsize=3)
    bars2 = ax.bar(x + width/2, chain_means, width, yerr=chain_stds,
                   label='Chain Following Knockout', color='#A23B72', alpha=0.7, capsize=3)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean |Knockout Effect|')
    ax.set_title('Attention Knockout Effects by Layer\n(Error bars show standard deviation)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_head_heatmaps(df: pd.DataFrame) -> plt.Figure:
    """
    Create heatmaps showing knockout effects for each head in each layer.
    
    Separate heatmaps for direct and chain knockouts.
    """
    layers = sorted(df['layer_idx'].unique())
    n_layers = len(layers)
    
    # Get max number of heads
    max_heads = df['head_idx'].max() + 1
    
    # Create subplots
    fig, axes = plt.subplots(2, n_layers, figsize=(4*n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    # Create heatmap data
    conditions = ['direct_knockout', 'chain_knockout']
    condition_names = ['Direct Pointer Knockout', 'Chain Following Knockout']
    
    # Get global color scale
    all_effects = df[df['condition'].isin(conditions)]['knockout_effect'].abs()
    vmax = all_effects.quantile(0.95) if len(all_effects) > 0 else 1.0
    
    for cond_idx, (condition, cond_name) in enumerate(zip(conditions, condition_names)):
        for layer_idx, layer in enumerate(layers):
            # Get data for this layer and condition
            layer_cond_data = df[(df['layer_idx'] == layer) & (df['condition'] == condition)]
            
            if len(layer_cond_data) > 0:
                # Create matrix for heatmap
                effects_matrix = np.zeros((1, max_heads))
                for _, row in layer_cond_data.iterrows():
                    effects_matrix[0, row['head_idx']] = abs(row['knockout_effect'])
                
                # Create heatmap
                sns.heatmap(effects_matrix, annot=True, fmt='.3f', cmap='Reds',
                           vmin=0, vmax=vmax, ax=axes[cond_idx, layer_idx],
                           cbar=(layer_idx == n_layers - 1),  # Only show colorbar on last subplot
                           xticklabels=range(max_heads), yticklabels=['Effect'])
            else:
                # Empty heatmap
                axes[cond_idx, layer_idx].text(0.5, 0.5, 'No Data', 
                                               ha='center', va='center', 
                                               transform=axes[cond_idx, layer_idx].transAxes)
            
            # Set titles
            if cond_idx == 0:
                axes[cond_idx, layer_idx].set_title(f'Layer {layer}')
            axes[cond_idx, layer_idx].set_xlabel('Attention Head')
            
            if layer_idx == 0:
                axes[cond_idx, layer_idx].set_ylabel(cond_name)
    
    plt.suptitle('Knockout Effects by Layer and Head\n(Absolute values shown)', fontsize=16)
    plt.tight_layout()
    return fig


def plot_effect_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Show distributions of knockout effects for each condition.
    
    Helps understand the statistical significance of differences.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Filter out baseline condition for distribution plots
    knockout_data = df[df['condition'] != 'baseline']
    
    # 1. Histogram of effects by condition
    direct_effects = knockout_data[knockout_data['condition'] == 'direct_knockout']['knockout_effect']
    chain_effects = knockout_data[knockout_data['condition'] == 'chain_knockout']['knockout_effect']
    
    axes[0, 0].hist(direct_effects, alpha=0.7, label='Direct Knockout', bins=20, color='#2E86AB')
    if len(chain_effects) > 0:
        axes[0, 0].hist(chain_effects, alpha=0.7, label='Chain Knockout', bins=20, color='#A23B72')
    axes[0, 0].set_xlabel('Knockout Effect')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Knockout Effects')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot by condition
    knockout_data.boxplot(column='knockout_effect', by='condition', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Condition')
    axes[0, 1].set_ylabel('Knockout Effect')
    axes[0, 1].set_title('Effect Distribution by Condition')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45)
    
    # 3. Success rate comparison
    success_by_condition = knockout_data.groupby('condition')['success_rate'].mean()
    success_by_condition.plot(kind='bar', ax=axes[1, 0], color=['#2E86AB', '#A23B72'])
    axes[1, 0].set_xlabel('Condition')
    axes[1, 0].set_ylabel('Mean Success Rate')
    axes[1, 0].set_title('Success Rate by Condition')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot: effect vs success rate
    for condition, color in [('direct_knockout', '#2E86AB'), ('chain_knockout', '#A23B72')]:
        cond_data = knockout_data[knockout_data['condition'] == condition]
        if len(cond_data) > 0:
            axes[1, 1].scatter(cond_data['knockout_effect'], cond_data['success_rate'], 
                              alpha=0.6, label=condition.replace('_', ' ').title(), color=color)
    
    axes[1, 1].set_xlabel('Knockout Effect')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Effect Size vs Success Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Analysis of Knockout Effects', fontsize=16)
    plt.tight_layout()
    return fig


def generate_summary_report(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Generate a text summary of the experimental findings."""
    
    direct_data = df[df['condition'] == 'direct_knockout']
    chain_data = df[df['condition'] == 'chain_knockout']
    
    report = f"""
ATTENTION PATTERN KNOCKOUT EXPERIMENT SUMMARY
{'='*60}

EXPERIMENTAL SETUP:
- Model: Qwen3-14B  
- Layers tested: {', '.join(map(str, analysis['layers_tested']))}
- Heads per layer: {analysis['heads_per_layer']}
- Total interventions: {analysis['total_interventions']}

MECHANISM DOMINANCE:
🎯 DOMINANT MECHANISM: {analysis['dominant_mechanism'].replace('_', ' ').upper()}
📊 Dominance ratio: {analysis['dominance_ratio']:.2f}x stronger

DIRECT POINTER MECHANISM:
- Mean knockout effect: {analysis['direct_knockout_stats']['mean_effect']:.4f}
- Maximum knockout effect: {analysis['direct_knockout_stats']['max_effect']:.4f}
- Significant effects (>0.1): {analysis['direct_knockout_stats']['significant_effects']}
- Success rate change: {analysis['direct_knockout_stats']['success_rate_changes']:.3f}

CHAIN FOLLOWING MECHANISM:
- Mean knockout effect: {analysis['chain_knockout_stats']['mean_effect']:.4f}
- Maximum knockout effect: {analysis['chain_knockout_stats']['max_effect']:.4f}
- Significant effects (>0.1): {analysis['chain_knockout_stats']['significant_effects']}
- Success rate change: {analysis['chain_knockout_stats']['success_rate_changes']:.3f}

LAYER-BY-LAYER BREAKDOWN:"""
    
    for layer, stats in analysis['layer_breakdown'].items():
        report += f"""
Layer {layer}:
  - Direct pointer impact: {stats['direct_mean_effect']:.4f} (max: {stats['direct_max_effect']:.4f})
  - Chain following impact: {stats['chain_mean_effect']:.4f} (max: {stats['chain_max_effect']:.4f})"""
    
    # Interpretation
    report += f"""

INTERPRETATION:
"""
    
    if analysis['dominant_mechanism'] == 'direct_pointer':
        report += f"""
✅ The model primarily uses DIRECT ATTENTION POINTERS for variable binding.
   
   This means:
   - Query positions (#a:) directly attend to root values (1 in "l = 1")
   - Intermediate variables (c, l) are less critical for attention flow
   - The "information jump" observed in layers 27-28 supports direct pointer mechanism
   - Variable binding resolution bypasses step-by-step chain following
"""
    else:
        report += f"""
✅ The model primarily uses STEP-BY-STEP CHAIN FOLLOWING for variable binding.
   
   This means:
   - Query positions (#a:) attend through intermediate variables (a → c → l → 1)
   - Direct attention to root values is less important
   - Variable binding follows the explicit program structure
   - The model processes variable chains sequentially
"""
    
    report += f"""
STATISTICAL SIGNIFICANCE:
- Total knockout effects measured: {len(direct_data) + len(chain_data)}
- Effect size difference: {abs(analysis['direct_knockout_stats']['mean_effect'] - analysis['chain_knockout_stats']['mean_effect']):.4f}
- Dominance ratio indicates {analysis['dominance_ratio']:.1f}x stronger effect for dominant mechanism

EXPERIMENTAL VALIDATION:
{'✅ HYPOTHESIS CONFIRMED' if analysis['dominance_ratio'] > 1.5 else '⚠️  WEAK EVIDENCE'}: The experimental results {'strongly support' if analysis['dominance_ratio'] > 2.0 else 'provide evidence for'} the {analysis['dominant_mechanism'].replace('_', ' ')} mechanism.
"""
    
    return report


def main():
    """Main visualization and analysis function."""
    
    try:
        # Load experimental data
        results_dir = find_latest_experiment_results()
        df, analysis = load_experiment_data(results_dir)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # 1. Mechanism comparison
        fig1 = plot_mechanism_comparison(df, analysis)
        fig1.savefig(results_dir / "01_mechanism_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Layer breakdown
        fig2 = plot_layer_breakdown(df)
        fig2.savefig(results_dir / "02_layer_breakdown.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Head heatmaps
        fig3 = plot_head_heatmaps(df)
        fig3.savefig(results_dir / "03_head_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Statistical distributions
        fig4 = plot_effect_distributions(df)
        fig4.savefig(results_dir / "04_effect_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Generate summary report
        report = generate_summary_report(df, analysis)
        report_path = results_dir / "experiment_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Display summary
        print(report)
        
        # Save visualization info
        viz_info = {
            'visualizations_created': [
                '01_mechanism_comparison.png',
                '02_layer_breakdown.png', 
                '03_head_heatmaps.png',
                '04_effect_distributions.png'
            ],
            'summary_report': 'experiment_summary.txt',
            'dominant_mechanism': analysis['dominant_mechanism'],
            'dominance_ratio': analysis['dominance_ratio']
        }
        
        with open(results_dir / "visualization_info.json", 'w') as f:
            json.dump(viz_info, f, indent=2)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Visualizations saved in: {results_dir}")
        print(f"📝 Summary report: {report_path}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()