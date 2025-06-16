#%%
"""Visualization and analysis methods for causal tracing experiments.

Following Edward Tufte's principles: minimal chart junk, high data-ink ratio,
clear information graphics that focus on the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import sys, pathlib
sys.path.append(str(pathlib.Path(".").resolve() / "src"))

from debug.causal_tracing import InterventionResult
from debug.causal_experiment_runner import CausalExperimentResult

# from .causal_tracing import InterventionResult
# from .causal_experiment_runner import CausalExperimentResult


class CausalAnalyzer:
    """Analyzes and aggregates causal tracing experiment results."""
    
    def aggregate_by_target(self, results: List[InterventionResult]) -> Dict[str, Dict[str, Any]]:
        """Aggregate intervention results by target token type."""
        target_groups = defaultdict(list)
        
        # Group by target position (we'll infer target type from position patterns)
        for result in results:
            if result.target_token_pos is not None:
                # Create a key based on the characteristics we can infer
                key = f"pos_{result.target_token_pos}"
                target_groups[key].append(result)
        
        # Aggregate statistics for each target
        aggregated = {}
        for target_name, target_results in target_groups.items():
            success_rates = [r.success_rate for r in target_results if r.success_rate is not None]
            logit_diffs = [r.logit_difference for r in target_results if r.logit_difference is not None]
            
            if success_rates:
                best_idx = np.argmax(success_rates)
                aggregated[target_name] = {
                    "mean_success_rate": np.mean(success_rates),
                    "max_success_rate": np.max(success_rates),
                    "std_success_rate": np.std(success_rates),
                    "mean_logit_difference": np.mean(logit_diffs) if logit_diffs else 0.0,
                    "best_layer": target_results[best_idx].layer_idx,
                    "num_interventions": len(target_results)
                }
        
        return aggregated
    
    def find_critical_layers(self, results: List[InterventionResult], threshold: float = 0.5) -> List[int]:
        """Find layers where interventions are most effective."""
        layer_performance = defaultdict(list)
        
        for result in results:
            if result.success_rate is not None:
                layer_performance[result.layer_idx].append(result.success_rate)
        
        critical_layers = []
        for layer, rates in layer_performance.items():
            if np.mean(rates) >= threshold:
                critical_layers.append(layer)
        
        return sorted(critical_layers)


def apply_tufte_style(ax):
    """Apply Edward Tufte's minimal style to matplotlib axes."""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines lighter
    ax.spines['left'].set_color('0.7')
    ax.spines['bottom'].set_color('0.7')
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove tick marks
    ax.tick_params(length=0)
    
    return ax


def plot_layer_intervention_effects(results: List[InterventionResult], 
                                   save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot intervention effects across model layers.
    
    Shows how intervention success rate varies by layer, following Tufte's
    principle of showing the data clearly without unnecessary decoration.
    """
    # Organize data
    df_data = []
    for result in results:
        if result.success_rate is not None:
            df_data.append({
                'layer': result.layer_idx,
                'success_rate': result.success_rate,
                'logit_difference': result.logit_difference or 0.0,
                'target_pos': result.target_token_pos,
                'intervention_type': result.intervention_type
            })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Create figure with Tufte-style aesthetics
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot success rate by layer
    layer_means = df.groupby('layer')['success_rate'].agg(['mean', 'std']).reset_index()
    
    # Main line plot
    ax.plot(layer_means['layer'], layer_means['mean'], 'o-', 
            color='steelblue', linewidth=2, markersize=6, label='Success Rate')
    
    # Error bars (subtle)
    ax.errorbar(layer_means['layer'], layer_means['mean'], yerr=layer_means['std'],
                color='steelblue', alpha=0.3, capsize=3, capthick=1)
    
    # Formatting
    ax.set_xlabel('Model Layer', fontsize=12)
    ax.set_ylabel('Intervention Success Rate', fontsize=12)
    ax.set_title('Causal Intervention Effects Across Layers', fontsize=14, pad=20)
    
    # Set y-axis to show full range
    ax.set_ylim(0, 1)
    
    # Apply Tufte style
    apply_tufte_style(ax)
    
    # Add subtle annotation for peak performance
    if not layer_means.empty:
        best_layer = layer_means.loc[layer_means['mean'].idxmax()]
        ax.annotate(f'Peak: Layer {best_layer["layer"]:.0f}\n({best_layer["mean"]:.2%})',
                   xy=(best_layer["layer"], best_layer["mean"]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_success_rate_heatmap(results: List[InterventionResult],
                             save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap of success rates by layer and intervention target.
    
    Uses a minimal color scheme following Tufte's approach of using color
    purposefully to convey information.
    """
    # Organize data for heatmap
    df_data = []
    for result in results:
        if result.success_rate is not None:
            df_data.append({
                'layer': result.layer_idx,
                'target_pos': f'pos_{result.target_token_pos}',
                'success_rate': result.success_rate
            })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Pivot for heatmap
    pivot_df = df.pivot_table(values='success_rate', index='target_pos', 
                             columns='layer', aggfunc='mean')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use seaborn heatmap for nicer aesthetics (diverging palette centred at zero)
    sns.heatmap(
        pivot_df,
        cmap="RdBu_r",
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Success Rate"},
        ax=ax,
        square=False,
    )
    
    # Apply Tufte minimalist style
    apply_tufte_style(ax)
    
    # Invert y-axis so lower layers appear at bottom (optional aesthetic)
    ax.invert_yaxis()
    
    # Axis labels and title
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Target Position', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_referential_depth_analysis(program_results: List[Dict[str, Any]],
                                   save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Analyze intervention success by referential depth.
    
    Shows how variable chain depth affects causal tracing effectiveness,
    using clear visual hierarchy to highlight key insights.
    """
    # Extract referential depth and success rate data
    df_data = []
    for program in program_results:
        if 'metadata' in program and 'variable_chain' in program['metadata']:
            chain = program['metadata']['variable_chain']
            depth = getattr(chain, 'referential_depth', None)
            success_rate = program.get('best_success_rate', 0.0)
            
            if depth is not None:
                df_data.append({
                    'referential_depth': depth,
                    'success_rate': success_rate,
                    'seq_len': program.get('seq_len', 0)
                })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Create scatter plot with trend line
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(df['referential_depth'], df['success_rate'], 
                        alpha=0.6, s=60, color='steelblue', edgecolors='white', linewidth=1)
    
    # Add trend line if we have enough data
    if len(df) > 1:
        z = np.polyfit(df['referential_depth'], df['success_rate'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['referential_depth'].min(), df['referential_depth'].max(), 100)
        ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.7, linewidth=2,
               label=f'Trend (slope: {z[0]:.3f})')
    
    # Box plots for each depth level
    depths = sorted(df['referential_depth'].unique())
    for depth in depths:
        depth_data = df[df['referential_depth'] == depth]['success_rate']
        if len(depth_data) > 1:
            # Add subtle box plot
            bp = ax.boxplot(depth_data, positions=[depth], widths=0.2, 
                          patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.3)
    
    ax.set_xlabel('Referential Depth (Variable Chain Length)', fontsize=12)
    ax.set_ylabel('Best Intervention Success Rate', fontsize=12)
    ax.set_title('Causal Tracing Effectiveness vs Variable Chain Complexity', 
                fontsize=14, pad=20)
    
    # Set reasonable limits
    ax.set_ylim(0, 1)
    if depths:
        ax.set_xlim(min(depths) - 0.5, max(depths) + 0.5)
    
    # Apply Tufte style
    apply_tufte_style(ax)
    
    if len(df) > 1:
        ax.legend(frameon=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def create_intervention_summary_plot(experiment_result: CausalExperimentResult,
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a comprehensive summary plot of causal tracing experiment.
    
    Uses small multiples approach to show different aspects of the data
    in a coherent, easy-to-scan format following Tufte's principles.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplot layout: 2x3 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Layer-wise intervention effects
    ax1 = fig.add_subplot(gs[0, 0])
    if experiment_result.intervention_results:
        # Plot success rate by layer
        df_data = []
        for result in experiment_result.intervention_results:
            if result.success_rate is not None:
                df_data.append({
                    'layer': result.layer_idx,
                    'success_rate': result.success_rate
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            layer_means = df.groupby('layer')['success_rate'].mean()
            ax1.plot(layer_means.index, layer_means.values, 'o-', color='steelblue', linewidth=2)
            ax1.set_ylabel('Success Rate')
            ax1.set_title('A. Layer-wise Effects', fontsize=12, fontweight='bold')
            apply_tufte_style(ax1)
    
    # 2. Success rate distribution
    ax2 = fig.add_subplot(gs[0, 1])
    success_rates = [r.success_rate for r in experiment_result.intervention_results 
                    if r.success_rate is not None]
    if success_rates:
        ax2.hist(success_rates, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
        ax2.axvline(np.mean(success_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(success_rates):.2%}')
        ax2.set_xlabel('Success Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('B. Success Rate Distribution', fontsize=12, fontweight='bold')
        ax2.legend(frameon=False)
        apply_tufte_style(ax2)
    
    # 3. Logit difference by layer
    ax3 = fig.add_subplot(gs[1, 0])
    if experiment_result.intervention_results:
        df_data = []
        for result in experiment_result.intervention_results:
            if result.logit_difference is not None:
                df_data.append({
                    'layer': result.layer_idx,
                    'logit_difference': result.logit_difference
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            layer_means = df.groupby('layer')['logit_difference'].mean()
            ax3.plot(layer_means.index, layer_means.values, 's-', color='orange', linewidth=2)
            ax3.set_ylabel('Logit Difference')
            ax3.set_title('C. Logit Changes by Layer', fontsize=12, fontweight='bold')
            apply_tufte_style(ax3)
    
    # 4. Referential depth analysis
    ax4 = fig.add_subplot(gs[1, 1])
    depth_data = []
    for program in experiment_result.program_results:
        if 'metadata' in program and 'variable_chain' in program['metadata']:
            chain = program['metadata']['variable_chain']
            depth = getattr(chain, 'referential_depth', None)
            success_rate = program.get('best_success_rate', 0.0)
            if depth is not None:
                depth_data.append({'depth': depth, 'success_rate': success_rate})
    
    if depth_data:
        df = pd.DataFrame(depth_data)
        depth_means = df.groupby('depth')['success_rate'].agg(['mean', 'std'])
        ax4.errorbar(depth_means.index, depth_means['mean'], yerr=depth_means['std'],
                    fmt='o-', color='green', capsize=3)
        ax4.set_xlabel('Referential Depth')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('D. Depth vs Performance', fontsize=12, fontweight='bold')
        apply_tufte_style(ax4)
    
    # 5. Summary statistics (text)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary text
    summary_stats = experiment_result.summary_stats
    summary_text = f"""
    Experiment Summary: {experiment_result.config.name}
    Model: {experiment_result.config.model_name}
    
    Programs: {summary_stats.get('successful_programs', 0)}/{summary_stats.get('total_programs', 0)} successful
    Interventions: {summary_stats.get('total_interventions', 0)} total
    Mean Success Rate: {summary_stats.get('mean_success_rate', 0):.2%}
    Max Success Rate: {summary_stats.get('max_success_rate', 0):.2%}
    Mean Logit Difference: {summary_stats.get('mean_logit_difference', 0):.3f} ± {summary_stats.get('std_logit_difference', 0):.3f}
    """
    
    ax5.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3))
    
    # Main title
    fig.suptitle(f'Causal Tracing Analysis: {experiment_result.config.name}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_causal_flow_heatmap(intervention_results: List[InterventionResult],
                            token_labels: List[str],
                            information_movements: Optional[List[Dict[str, Any]]] = None,
                            save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a heatmap showing causal intervention effects across token positions and layers.
    
    This visualization replicates the style shown in the reference image, with:
    - Token positions on x-axis with variable labels
    - Model layers on y-axis
    - Logit differences shown as color intensity
    - Annotations for information movement across layers
    
    Args:
        intervention_results: List of intervention results to visualize
        token_labels: Labels for each token position (e.g., variable names)
        information_movements: Optional list of dicts describing information flow
        save_path: Optional path to save the figure
    
    Returns:
        Tuple of (figure, axes) objects
    """
    if not intervention_results:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, 'No intervention data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Organize data into a matrix format
    df_data = []
    for result in intervention_results:
        if result.logit_difference is not None and result.target_token_pos is not None:
            df_data.append({
                'layer': result.layer_idx,
                'token_pos': result.target_token_pos,
                'logit_difference': result.logit_difference
            })
    
    if not df_data:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    df = pd.DataFrame(df_data)
    
    # Create pivot table for heatmap
    pivot_df = df.pivot_table(values='logit_difference', index='layer', 
                             columns='token_pos', aggfunc='mean', fill_value=0)
    
    # Ensure we have all token positions, fill missing with 0
    all_positions = list(range(len(token_labels)))
    for pos in all_positions:
        if pos not in pivot_df.columns:
            pivot_df[pos] = 0.0
    
    # Sort columns by token position
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap with custom colormap (blue scale like the reference)
    im = ax.imshow(pivot_df.values, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_yticks(range(len(pivot_df.index)))
    
    # Set x-axis labels with token information
    x_labels = []
    for i, pos in enumerate(pivot_df.columns):
        if pos < len(token_labels):
            x_labels.append(token_labels[pos])
        else:
            x_labels.append(f'pos_{pos}')
    
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_yticklabels([f'{int(layer)}' for layer in pivot_df.index], fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Causal Intervention Effects: Logit Difference by Token Position and Layer', 
                fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Logit Difference', fontsize=12)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, len(pivot_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot_df.index), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add information movement annotations if provided
    if information_movements:
        for movement in information_movements:
            layer = movement.get('layer')
            description = movement.get('description', '')
            if layer is not None and layer in pivot_df.index:
                layer_idx = list(pivot_df.index).index(layer)
                # Add arrow annotation
                ax.annotate('', xy=(len(pivot_df.columns) - 0.5, layer_idx), 
                           xytext=(len(pivot_df.columns) + 1, layer_idx),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
                # Add text description
                ax.text(len(pivot_df.columns) + 1.5, layer_idx, description,
                       verticalalignment='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Invert y-axis to match the reference image (layer 1 at top)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_token_level_causal_trace(intervention_results: List[InterventionResult],
                                 program_text: str,
                                 variable_chain: Optional[List[str]] = None,
                                 save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a detailed token-level causal trace visualization.
    
    Shows the exact replication of the reference image with:
    - Full program text at bottom with token positions
    - Variable assignments highlighted in red
    - Information movement annotations
    - Layer-by-layer causal effects
    
    Args:
        intervention_results: Results from causal interventions
        program_text: The full program text to display
        variable_chain: Optional list describing the variable binding chain
        save_path: Optional path to save the figure
    
    Returns:
        Tuple of (figure, axes) objects
    """
    if not intervention_results:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.text(0.5, 0.5, 'No intervention data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Parse program text into tokens
    tokens = program_text.split()
    
    # Organize intervention data
    df_data = []
    for result in intervention_results:
        if result.logit_difference is not None and result.target_token_pos is not None:
            df_data.append({
                'layer': result.layer_idx,
                'token_pos': result.target_token_pos,
                'logit_difference': abs(result.logit_difference)  # Use absolute value for intensity
            })
    
    if not df_data:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    df = pd.DataFrame(df_data)
    
    # Get unique layers and token positions
    layers = sorted(df['layer'].unique())
    token_positions = sorted(df['token_pos'].unique())
    
    # Create matrix for heatmap
    matrix = np.zeros((len(layers), len(tokens)))
    
    for _, row in df.iterrows():
        layer_idx = layers.index(row['layer'])
        if row['token_pos'] < len(tokens):
            matrix[layer_idx, int(row['token_pos'])] = row['logit_difference']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Set up axes
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f'{i}' for i in range(len(tokens))], fontsize=8)
    ax.set_yticklabels([f'{layer}' for layer in layers])
    
    # Add token text at bottom
    token_text = ' '.join([f'{token}' for token in tokens])
    
    # Highlight variable assignments in red
    colored_tokens = []
    for i, token in enumerate(tokens):
        # Simple heuristic: highlight numbers that might be variable values
        if token.isdigit() or (len(token) == 1 and token.isalpha()):
            if any(op in tokens[max(0, i-2):i+3] for op in ['=']):
                colored_tokens.append(f'\\textcolor{{red}}{{{token}}}')
            else:
                colored_tokens.append(token)
        else:
            colored_tokens.append(token)
    
    # Add program text below the heatmap
    ax.text(0.5, -0.15, token_text, transform=ax.transAxes, 
           ha='center', va='top', fontsize=10, fontfamily='monospace')
    
    # Add labels
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Logit Difference', fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add information movement annotations
    # Find significant activations and annotate them
    significant_positions = []
    for layer_idx, layer in enumerate(layers):
        for token_idx in range(len(tokens)):
            if matrix[layer_idx, token_idx] > 0.5:  # Threshold for significance
                significant_positions.append({
                    'layer': layer,
                    'layer_idx': layer_idx,
                    'token_idx': token_idx,
                    'token': tokens[token_idx] if token_idx < len(tokens) else f'pos_{token_idx}'
                })
    
    # Group by approximate token positions and add movement annotations
    movement_groups = {}
    for pos in significant_positions:
        key = pos['token_idx'] // 5  # Group nearby positions
        if key not in movement_groups:
            movement_groups[key] = []
        movement_groups[key].append(pos)
    
    # Add annotations for information movement
    annotation_count = 1
    for group in movement_groups.values():
        if len(group) > 1:  # Only annotate if there are multiple layers involved
            # Find the range of layers
            min_layer_idx = min(p['layer_idx'] for p in group)
            max_layer_idx = max(p['layer_idx'] for p in group)
            center_token_idx = group[0]['token_idx']
            
            # Add arrow showing information flow
            ax.annotate('', xy=(center_token_idx, max_layer_idx + 0.3), 
                       xytext=(center_token_idx, min_layer_idx - 0.3),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            
            # Add information movement label
            movement_text = f"Information movement {annotation_count}"
            ax.text(center_token_idx + len(tokens) * 0.02, 
                   (min_layer_idx + max_layer_idx) / 2,
                   movement_text, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                   rotation=0)
            annotation_count += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def quick_visualization_demo():
    """Create example visualizations to verify the functions work."""
    print("Creating example visualizations...")
    
    # Create mock data for demonstration
    np.random.seed(42)
    
    # Mock intervention results
    results = []
    for layer in range(8):
        for target_pos in [5, 10, 15]:
            result = InterventionResult(
                intervention_type="residual_stream",
                layer_idx=layer,
                target_token_pos=target_pos,
                logit_difference=np.random.normal(0.2, 0.1),
                success_rate=np.random.beta(2, 3),  # Realistic success rate distribution
                original_top_token=1,
                intervened_top_token=2
            )
            results.append(result)
    
    # Create visualizations
    fig1, ax1 = plot_layer_intervention_effects(results)
    plt.title("Demo: Layer Intervention Effects")
    plt.show()
    
    fig2, ax2 = plot_success_rate_heatmap(results)
    plt.title("Demo: Success Rate Heatmap")
    plt.show()
    
    # Mock program results for depth analysis
    program_results = []
    for depth in [1, 2, 3, 4]:
        for i in range(10):
            program_results.append({
                "best_success_rate": np.random.beta(depth, 4-depth+1),  # Varies by depth
                "metadata": {
                    "variable_chain": type('', (), {
                        "referential_depth": depth
                    })()
                }
            })
    
    fig3, ax3 = plot_referential_depth_analysis(program_results)
    plt.title("Demo: Referential Depth Analysis")
    plt.show()
    
    print("Example visualizations created successfully!")


if __name__ == "__main__":
    quick_visualization_demo()

#%%