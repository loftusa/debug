"""
Attention Pattern Knockout Experiment

This experiment tests whether Qwen3-14B uses direct attention "pointers" vs 
step-by-step chain following for variable binding resolution.

Based on observations from _10_attention_token_patching.py that information 
"jumps" from root values to final positions at layers 27-28, we test two hypotheses:

1. Direct Pointer: Query position (#a:) directly attends to root values (1 in "l = 1")
2. Chain Following: Query position attends through intermediate variables (a → c → l → 1)

The experiment selectively zeros out attention weights and measures which 
knockout has larger impact on final predictions.

Usage:
    uv run experiments/_11_attention_pattern_knockout.py
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.append(str(project_root))

from debug.attention_knockout import AttentionPatternKnockout, AttentionKnockoutResult
from debug.generators import make_variable_binding_program_with_metadata


def create_test_program(rng_seed: int = 12) -> tuple[str, str, Dict[str, Any]]:
    """
    Create the standard test program using the robust seed.
    
    Args:
        rng_seed: Random seed for program generation (default: 12)
        
    Returns:
        Tuple of (program, query_var, metadata)
    """
    rng = np.random.RandomState(rng_seed)
    
    # Use existing generator to create program with metadata
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    
    program, answer, hops, metadata = make_variable_binding_program_with_metadata(
        seq_len=17, rng=rng, tokenizer=tokenizer
    )
    
    query_var = metadata["query_var"]
    
    print(f"Generated program with {hops} hops:")
    print(f"Query variable: {query_var}")
    print(f"Expected answer: {answer}")
    print(f"Program:\n{program}")
    
    return program, query_var, metadata


def run_attention_knockout_experiment(
    model_id: str = "Qwen/Qwen3-14B",
    target_layers: List[int] = [27, 28, 29],
    rng_seed: int = 12,
    max_heads_per_layer: int = None
) -> List[AttentionKnockoutResult]:
    """
    Run the complete attention pattern knockout experiment.
    
    Args:
        model_id: Model to test
        target_layers: Layers to test (where information jump was observed)
        rng_seed: Random seed for program generation
        max_heads_per_layer: Optional limit on heads to test (for quick testing)
        
    Returns:
        List of all experimental results
    """
    print(f"Starting Attention Pattern Knockout Experiment")
    print(f"Model: {model_id}")
    print(f"Target layers: {target_layers}")
    print(f"RNG seed: {rng_seed}")
    
    # Create test program
    program, query_var, metadata = create_test_program(rng_seed)
    
    # Initialize knockout experiment framework
    knockout_tracer = AttentionPatternKnockout(model_id)
    
    # Limit heads for testing if specified
    num_heads = knockout_tracer.model.config.num_attention_heads
    if max_heads_per_layer:
        test_heads = min(num_heads, max_heads_per_layer)
        print(f"Testing {test_heads}/{num_heads} heads per layer")
    else:
        test_heads = num_heads
        print(f"Testing all {num_heads} heads per layer")
    
    # Run systematic experiment
    all_results = []
    total_interventions = len(target_layers) * test_heads * 3  # 3 conditions per head
    
    print(f"\nRunning {total_interventions} total interventions...")
    
    with tqdm(total=total_interventions, desc="Attention Knockout") as pbar:
        for layer_idx in target_layers:
            pbar.set_description(f"Layer {layer_idx}")
            
            for head_idx in range(test_heads):
                try:
                    # Run experiment for this layer/head combination
                    layer_head_results = knockout_tracer.run_attention_knockout_experiment(
                        program=program,
                        query_var=query_var,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        program_id=0
                    )
                    
                    all_results.extend(layer_head_results)
                    pbar.update(len(layer_head_results))
                    
                    # Memory cleanup every 16 heads
                    if head_idx % 16 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"\nError at L{layer_idx}H{head_idx}: {e}")
                    pbar.update(3)  # Skip 3 conditions
                    continue
    
    print(f"\nCompleted: {len(all_results)} results collected")
    
    # Clean up model to free GPU memory
    del knockout_tracer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_results


def analyze_knockout_results(results: List[AttentionKnockoutResult]) -> Dict[str, Any]:
    """
    Analyze results to determine direct vs chain mechanism dominance.
    
    Args:
        results: List of experimental results
        
    Returns:
        Dictionary with analysis results
    """
    # Convert to DataFrame for analysis
    df_data = []
    for result in results:
        df_data.append({
            'layer_idx': result.layer_idx,
            'head_idx': result.head_idx,
            'condition': result.condition,
            'knockout_effect': result.knockout_effect or 0.0,
            'normalized_logit_difference': result.normalized_logit_difference,
            'success_rate': result.success_rate,
            'query_pos': result.query_pos,
            'root_pos': result.root_pos,
            'intermediate_count': len(result.intermediate_positions)
        })
    
    df = pd.DataFrame(df_data)
    
    # Separate conditions
    baseline_df = df[df['condition'] == 'baseline']
    direct_df = df[df['condition'] == 'direct_knockout']
    chain_df = df[df['condition'] == 'chain_knockout']
    
    analysis = {
        'total_interventions': len(results),
        'layers_tested': sorted(df['layer_idx'].unique()),
        'heads_per_layer': df[df['layer_idx'] == df['layer_idx'].iloc[0]]['head_idx'].nunique(),
        
        # Direct knockout analysis
        'direct_knockout_stats': {
            'mean_effect': direct_df['knockout_effect'].mean(),
            'max_effect': direct_df['knockout_effect'].max(),
            'significant_effects': len(direct_df[direct_df['knockout_effect'].abs() > 0.1]),
            'success_rate_changes': direct_df['success_rate'].mean()
        },
        
        # Chain knockout analysis  
        'chain_knockout_stats': {
            'mean_effect': chain_df['knockout_effect'].mean() if len(chain_df) > 0 else 0.0,
            'max_effect': chain_df['knockout_effect'].max() if len(chain_df) > 0 else 0.0,
            'significant_effects': len(chain_df[chain_df['knockout_effect'].abs() > 0.1]) if len(chain_df) > 0 else 0,
            'success_rate_changes': chain_df['success_rate'].mean() if len(chain_df) > 0 else 0.0
        },
        
        # Layer-wise breakdown
        'layer_breakdown': {}
    }
    
    # Analyze by layer
    for layer in sorted(df['layer_idx'].unique()):
        layer_data = df[df['layer_idx'] == layer]
        layer_direct = layer_data[layer_data['condition'] == 'direct_knockout']
        layer_chain = layer_data[layer_data['condition'] == 'chain_knockout']
        
        analysis['layer_breakdown'][layer] = {
            'direct_mean_effect': layer_direct['knockout_effect'].mean(),
            'chain_mean_effect': layer_chain['knockout_effect'].mean() if len(layer_chain) > 0 else 0.0,
            'direct_max_effect': layer_direct['knockout_effect'].max(),
            'chain_max_effect': layer_chain['knockout_effect'].max() if len(layer_chain) > 0 else 0.0,
        }
    
    # Determine dominant mechanism
    direct_total_effect = abs(analysis['direct_knockout_stats']['mean_effect'])
    chain_total_effect = abs(analysis['chain_knockout_stats']['mean_effect'])
    
    if direct_total_effect > chain_total_effect:
        analysis['dominant_mechanism'] = 'direct_pointer'
        analysis['dominance_ratio'] = direct_total_effect / max(chain_total_effect, 1e-6)
    else:
        analysis['dominant_mechanism'] = 'chain_following'  
        analysis['dominance_ratio'] = chain_total_effect / max(direct_total_effect, 1e-6)
    
    return analysis


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values to handle numpy types in dictionary keys
        return {
            convert_to_serializable(key): convert_to_serializable(value) 
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_results(results: List[AttentionKnockoutResult], 
                analysis: Dict[str, Any],
                output_dir: Path) -> None:
    """Save experimental results and analysis."""
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        result_dict = {
            'layer_idx': convert_to_serializable(result.layer_idx),
            'head_idx': convert_to_serializable(result.head_idx),
            'condition': result.condition,
            'query_pos': convert_to_serializable(result.query_pos),
            'root_pos': convert_to_serializable(result.root_pos),
            'intermediate_positions': convert_to_serializable(result.intermediate_positions),
            'logit_difference': convert_to_serializable(result.logit_difference),
            'normalized_logit_difference': convert_to_serializable(result.normalized_logit_difference),
            'success_rate': convert_to_serializable(result.success_rate),
            'baseline_logit_diff': convert_to_serializable(result.baseline_logit_diff),
            'knockout_effect': convert_to_serializable(result.knockout_effect),
            'program_id': convert_to_serializable(result.program_id),
            'original_program': result.original_program,
            'token_labels': result.token_labels,
            'target_name': result.target_name
        }
        serializable_results.append(result_dict)
    
    # Convert analysis to serializable format
    serializable_analysis = convert_to_serializable(analysis)
    
    # Save detailed results
    results_path = output_dir / "knockout_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save analysis
    analysis_path = output_dir / "knockout_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print(f"  - Detailed results: {results_path}")
    print(f"  - Analysis: {analysis_path}")


def main():
    """Main experimental execution."""
    
    # Configuration
    MODEL_ID = "Qwen/Qwen3-14B"
    TARGET_LAYERS = [27, 28, 29]  # Layers where information jump was observed
    RNG_SEED = 12  # Robust seed from CLAUDE.md
    
    # For quick testing, limit heads (remove for full experiment)
    MAX_HEADS_PER_LAYER = None  # Set to small number like 4 for testing
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parents[1] / "results" / "attention_knockout" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run experiment
        results = run_attention_knockout_experiment(
            model_id=MODEL_ID,
            target_layers=TARGET_LAYERS,
            rng_seed=RNG_SEED,
            max_heads_per_layer=MAX_HEADS_PER_LAYER
        )
        
        # Analyze results
        analysis = analyze_knockout_results(results)
        
        # Save everything
        save_results(results, analysis, output_dir)
        
        # Print key findings
        print(f"\n{'='*60}")
        print("ATTENTION PATTERN KNOCKOUT EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Model: {MODEL_ID}")
        print(f"Layers tested: {analysis['layers_tested']}")
        print(f"Total interventions: {analysis['total_interventions']}")
        
        print(f"\nDOMINANT MECHANISM: {analysis['dominant_mechanism'].upper()}")
        print(f"Dominance ratio: {analysis['dominance_ratio']:.2f}x")
        
        print(f"\nDirect Pointer Knockout:")
        print(f"  Mean effect: {analysis['direct_knockout_stats']['mean_effect']:.4f}")
        print(f"  Max effect: {analysis['direct_knockout_stats']['max_effect']:.4f}")
        print(f"  Significant effects: {analysis['direct_knockout_stats']['significant_effects']}")
        
        print(f"\nChain Following Knockout:")
        print(f"  Mean effect: {analysis['chain_knockout_stats']['mean_effect']:.4f}")
        print(f"  Max effect: {analysis['chain_knockout_stats']['max_effect']:.4f}")
        print(f"  Significant effects: {analysis['chain_knockout_stats']['significant_effects']}")
        
        print(f"\nLayer Breakdown:")
        for layer, stats in analysis['layer_breakdown'].items():
            print(f"  Layer {layer}: Direct={stats['direct_mean_effect']:.4f}, Chain={stats['chain_mean_effect']:.4f}")
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()