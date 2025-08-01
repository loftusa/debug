"""
MLP Token Patching Experiment

Tests causal effects of MLP layers on variable binding tasks.
Systematically patches each MLP layer at intervention target positions
to identify which MLP components are responsible for tracking variable bindings.

Based on methodology from "Tracing Knowledge in Language Models Back to the Training Data"
https://arxiv.org/abs/2505.20896

This complements the attention head patching and residual stream patching experiments
to provide a comprehensive view of how different model components contribute to 
variable binding resolution.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import torch
import gc
import numpy as np
from tqdm import tqdm

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.append(str(project_root))

from debug.counterfactual import CounterfactualGenerator
from debug.generators import make_variable_binding_program_with_metadata
from debug.causal_tracing import CausalTracer, InterventionResult
from debug.token_analyzer import TokenAnalyzer


def run_comprehensive_mlp_patching(
    tracer: CausalTracer,
    original_program: str,
    counterfactual_program: str,
    intervention_targets: Dict[str, int],
    max_layers: int = None,
) -> List[InterventionResult]:
    """
    Patch MLP layers at intervention target positions only.
    
    Args:
        tracer: A ready‐initialized CausalTracer.
        original_program: Clean program string.
        counterfactual_program: Counterfactual program string.
        intervention_targets: Dict with intervention target positions.
        max_layers: Optionally cap the number of layers to test.
    
    Returns:
        List of InterventionResult objects for all (target_pos, layer) combinations.
    """
    # Get model architecture info
    n_layers = tracer._n_layers if max_layers is None else min(tracer._n_layers, max_layers)
    
    # Extract valid intervention target positions
    target_positions = []
    for target_name, pos in intervention_targets.items():
        if pos is not None and isinstance(pos, int):
            target_positions.append((target_name, pos))
    
    print(f"Testing {len(target_positions)} target positions × {n_layers} layers = {len(target_positions) * n_layers} interventions")
    print(f"Target positions: {target_positions}")
    
    all_results: List[InterventionResult] = []
    program_id = 0
    
    for target_name, token_pos in tqdm(target_positions, desc="Target position"):
        for layer_idx in range(n_layers):
            intervention_name = f"{target_name}_layer_{layer_idx}"
            
            try:
                result = tracer.run_mlp_intervention(
                    original_program=original_program,
                    counterfactual_program=counterfactual_program,
                    target_token_pos=int(token_pos),
                    layer_idx=layer_idx,
                    program_id=program_id,
                    target_name=intervention_name
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error at {target_name} pos {token_pos}, layer {layer_idx}: {e}")
                continue
            
            # Memory cleanup every 50 interventions
            if len(all_results) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    return all_results


def plot_mlp_heatmaps_by_layer(
    results: List[InterventionResult], 
    output_dir: Path,
    metric: str = "normalized_logit_difference"
) -> List[Path]:
    """
    Create separate heatmap visualizations for each layer.
    
    Each image shows:
    - X-axis: Intervention target tokens 
    - Y-axis: Single row (MLP layer - no heads like attention)
    - Color: Causal effect metric
    
    Args:
        results: List of intervention results
        output_dir: Directory to save the figures
        metric: Metric to visualize
    
    Returns:
        List of paths to saved figure files
    """
    # Convert results to DataFrame
    df_data = []
    for result in results:
        if hasattr(result, metric) and getattr(result, metric) is not None:
            df_data.append({
                'layer_idx': result.layer_idx,
                'target_token_pos': result.target_token_pos,
                'target_name': result.target_name,
                metric: getattr(result, metric)
            })
    
    if not df_data:
        print("No data to plot")
        return []
    
    df = pd.DataFrame(df_data)
    
    # Get color scale limits across all data for consistency
    vmax = max(abs(df[metric].min()), abs(df[metric].max()))
    
    saved_paths = []
    
    # Create one heatmap per layer
    for layer_idx in sorted(df['layer_idx'].unique()):
        layer_data = df[df['layer_idx'] == layer_idx].copy()
        
        if layer_data.empty:
            continue
            
        # Create pivot table: single row (MLP) × intervention targets (columns)
        pivot_data = layer_data.pivot_table(
            values=metric,
            index=['layer_idx'],  # This will be a single row
            columns='target_name', 
            aggfunc='mean'
        )
        
        # Create figure for this layer with adaptive sizing
        n_targets = len(pivot_data.columns)
        
        # Calculate appropriate figure size (wider for targets, shorter since only 1 row)
        width = max(12, n_targets * 2.5 + 4)
        height = 6  # Fixed height since only one MLP layer per plot
        
        fig, ax = plt.subplots(figsize=(width, height))
        
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
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        # Labels and title
        ax.set_title(
            f'Layer {layer_idx}: MLP Causal Effects\\n'
            f'{metric.replace("_", " ").title()} by Intervention Target',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.set_xlabel('Intervention Target', fontsize=12)
        ax.set_ylabel('MLP Layer', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / f"layer_{layer_idx:02d}_mlp.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_paths.append(save_path)
        plt.close(fig)  # Close to free memory
        
        print(f"Layer {layer_idx} MLP heatmap saved to: {save_path}")
    
    return saved_paths


def create_comprehensive_mlp_overview(
    results: List[InterventionResult],
    output_dir: Path,
    metric: str = "normalized_logit_difference"
) -> Path:
    """
    Create a comprehensive overview heatmap showing all layers and targets.
    
    Args:
        results: List of intervention results
        output_dir: Directory to save the figure
        metric: Metric to visualize
    
    Returns:
        Path to saved overview figure
    """
    # Convert results to DataFrame
    df_data = []
    for result in results:
        if hasattr(result, metric) and getattr(result, metric) is not None:
            df_data.append({
                'layer_idx': result.layer_idx,
                'target_name': result.target_name,
                metric: getattr(result, metric)
            })
    
    if not df_data:
        print("No data for overview plot")
        return None
    
    df = pd.DataFrame(df_data)
    
    # Create pivot table: layers (rows) × intervention targets (columns)
    pivot_data = df.pivot_table(
        values=metric,
        index='layer_idx',
        columns='target_name', 
        aggfunc='mean'
    )
    
    # Create comprehensive overview figure
    n_targets = len(pivot_data.columns)
    n_layers = len(pivot_data.index)
    
    width = max(14, n_targets * 1.5 + 4)
    height = max(10, n_layers * 0.4 + 3)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Get color scale limits
    vmax = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        annot=False,  # Too many cells for annotation
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    
    # Labels and title
    ax.set_title(
        f'MLP Causal Effects Overview\\n'
        f'{metric.replace("_", " ").title()} Across All Layers and Intervention Targets',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_xlabel('Intervention Target', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / "mlp_comprehensive_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comprehensive MLP overview saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    # --- Configuration -----------------------------------------------------
    MODEL_ID = "Qwen/Qwen3-14B"
    SEQ_LEN = 17
    RNG_SEED = 12  # Use the robust seed from CLAUDE.md
    
    print(f"Starting comprehensive MLP token patching experiment...")
    print(f"Model: {MODEL_ID}")
    print(f"RNG Seed: {RNG_SEED}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "mlp_token_patching" / timestamp
    model_name_safe = MODEL_ID.replace("/", "_")
    output_dir = BASE_OUTPUT_DIR / model_name_safe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Program Generation ------------------------------------------------
    print("\\n=== Program Generation ===")
    
    # Load model and generate program
    causal_tracer = CausalTracer(MODEL_ID)
    
    rng = np.random.RandomState(RNG_SEED)
    program, answer, hops, metadata = make_variable_binding_program_with_metadata(
        seq_len=SEQ_LEN, rng=rng, tokenizer=causal_tracer.tokenizer
    )
    
    query_var = metadata["query_var"]
    intervention_targets = metadata["intervention_targets"]
    
    # Generate counterfactual
    counterfactual_generator = CounterfactualGenerator()
    counter_program = counterfactual_generator.create_counterfactual(program, query_var)
    
    print(f"Original program:\\n{program}")
    print(f"\\nCounterfactual program:\\n{counter_program}")
    print(f"\\nQuery variable: {query_var}")
    print(f"Expected answer: {answer}")
    print(f"Hops: {hops}")
    print(f"Intervention targets: {intervention_targets}")
    
    # Validate intervention targets exist
    assert intervention_targets, "No intervention targets found in metadata"
    
    # --- Run Experiment ---------------------------------------------------
    print("\\n=== Running Comprehensive MLP Token Patching ===")
    
    results = run_comprehensive_mlp_patching(
        tracer=causal_tracer,
        original_program=program,
        counterfactual_program=counter_program,
        intervention_targets=intervention_targets,
        max_layers=None  # Test all layers
    )
    
    print(f"\\nCompleted {len(results)} MLP interventions")
    
    # --- Save Results ------------------------------------------------------
    print("\\n=== Saving Results ===")
    
    # Serialize results to JSON
    serializable_results = []
    for r in results:
        result_dict = {
            "intervention_type": r.intervention_type,
            "layer_idx": r.layer_idx,
            "head_idx": r.head_idx,  # Will be None for MLP
            "target_token_pos": r.target_token_pos,
            "logit_difference": r.logit_difference,
            "normalized_logit_difference": r.normalized_logit_difference,
            "success_rate": r.success_rate,
            "original_top_token": r.original_top_token,
            "intervened_top_token": r.intervened_top_token,
            "program_id": r.program_id,
            "original_program": r.original_program,
            "counterfactual_program": r.counterfactual_program,
            "token_labels": r.token_labels,
            "target_name": r.target_name,
        }
        serializable_results.append(result_dict)
    
    # Save full results
    results_path = output_dir / "intervention_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Save experiment metadata
    metadata_path = output_dir / "experiment_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "rng_seed": RNG_SEED,
            "seq_len": SEQ_LEN,
            "program": program,
            "counterfactual_program": counter_program,
            "query_var": query_var,
            "expected_answer": answer,
            "hops": hops,
            "intervention_targets": intervention_targets,
            "num_interventions": len(results),
            "timestamp": timestamp
        }, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # --- Generate Layer-wise Visualizations --------------------------------
    print("\\n=== Generating Layer-wise MLP Visualizations ===")
    
    # Create separate heatmaps for each layer
    viz_paths = plot_mlp_heatmaps_by_layer(
        results=results,
        output_dir=output_dir,
        metric="normalized_logit_difference"
    )
    
    # Create comprehensive overview
    overview_path = create_comprehensive_mlp_overview(
        results=results,
        output_dir=output_dir,
        metric="normalized_logit_difference"
    )
    
    # Generate summary statistics
    print("\\n=== Summary Statistics ===")
    df = pd.DataFrame([{
        'layer_idx': r.layer_idx,
        'normalized_logit_difference': r.normalized_logit_difference,
        'success_rate': r.success_rate,
        'target_name': r.target_name
    } for r in results])
    
    print(f"Total interventions: {len(results)}")
    print(f"Mean normalized logit difference: {df['normalized_logit_difference'].mean():.4f}")
    print(f"Max |normalized logit difference|: {df['normalized_logit_difference'].abs().max():.4f}")
    print(f"Mean success rate: {df['success_rate'].mean():.4f}")
    print(f"Max success rate: {df['success_rate'].max():.4f}")
    
    # Find most effective MLP interventions
    top_effects = df.nlargest(5, 'normalized_logit_difference')
    print(f"\\nTop 5 positive MLP effects:")
    for _, row in top_effects.iterrows():
        print(f"  L{row['layer_idx']} ({row['target_name']}): {row['normalized_logit_difference']:.4f}")
    
    # Clean up GPU memory
    del causal_tracer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\\n🎉 MLP Token Patching Experiment completed successfully!")
    print(f"Results saved in: {output_dir}")
    print(f"Key outputs:")
    print(f"  - Results: {results_path}")
    print(f"  - Overview: {overview_path}")
    print(f"  - Layer visualizations: {len(viz_paths)} layer-wise heatmaps")
    for viz_path in viz_paths[:5]:  # Show first 5 paths
        print(f"    - {viz_path.name}")
    if len(viz_paths) > 5:
        print(f"    - ... and {len(viz_paths) - 5} more")
    print(f"  - Metadata: {metadata_path}")