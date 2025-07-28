#%%
"""
Pointer Tracking Experiment

Tests whether language models track variable bindings through abstract pointers 
versus concrete values by using cross-query interventions.

Based on the hypothesis that models maintain structural/positional pointers 
rather than just tracking specific variable names or values.

The experiment patches the hidden state from one query (#l:) into another (#a:) 
to test if pointer information transfers across different variable queries.
"""

import sys
sys.path.append('../src')

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import gc
from pathlib import Path
from datetime import datetime
import re

from debug.causal_tracing import CausalTracer, InterventionResult
from debug.token_analyzer import TokenAnalyzer

print("🧪 Pointer Tracking Experiment")
print("=" * 50)

#%%
# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Model and experimental parameters
MODEL_ID = "Qwen/Qwen3-14B"
DEVICE = "auto"

# Output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).resolve().parents[1] / "results" / "pointer_test" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results will be saved to: {output_dir}")

#%%
# =============================================================================
# PROGRAM DEFINITIONS
# =============================================================================

PROGRAM_1 = """a = 1 
b = 2
c = a
#c: """


PROGRAM_2 = """b = 3 
a = 4
c = a
#c: """

# Expected answers
EXPECTED_ANSWER_1 = "1"  # a -> c -> 1
EXPECTED_ANSWER_2 = "4"  # a -> c -> 4

print("Program 1 (query #c: ):")
print(PROGRAM_1)
print(f"Expected answer: {EXPECTED_ANSWER_1}")
print()

print("Program 2 (query #f: ):")  
print(PROGRAM_2)
print(f"Expected answer: {EXPECTED_ANSWER_2}")
print()

print("🔬 Experiment Design:")
print("We want to test if the model uses pointers or values to track the variable bindings.")
print("We will start patching at the 'c' token (since thats when the programs are equivalent")
print("And measure the tokens 1, 2, and 4")

# =============================================================================
# MODEL LOADING & TOKENIZATION
# =============================================================================

print("🤖 Loading model and tokenizer...")

# Initialize the causal tracer
# TODO: This loads the model - it may take a few minutes and uses significant GPU memory
tracer = CausalTracer(MODEL_ID, device=DEVICE)

print(f"✅ Model loaded: {MODEL_ID}")
print(f"Model architecture: {tracer.model.config.model_type}")
print(f"Number of layers: {tracer._n_layers}")
print(f"Number of attention heads: {tracer.model.config.num_attention_heads}")

# Initialize token analyzer for finding positions
token_analyzer = TokenAnalyzer(tokenizer=tracer.tokenizer)

# =============================================================================
# TOKEN POSITION ANALYSIS
# =============================================================================

print("🔍 Analyzing token positions...")

# TODO: Implement tokenization to find exact positions
# You need to:
# 1. Tokenize both programs
# 2. Find the position of the space token after "#a:" in Program 1
# 3. Find the position of the space token after "#l:" in Program 2
# 4. Verify both programs have the same length (they should!)

# Tokenize Program 1
tokens_1 = tracer.tokenizer.tokenize(PROGRAM_1)
token_ids_1 = tracer.tokenizer(PROGRAM_1, return_tensors="pt", add_special_tokens=False).input_ids[0]

# Tokenize Program 2  
tokens_2 = tracer.tokenizer.tokenize(PROGRAM_2)
token_ids_2 = tracer.tokenizer(PROGRAM_2, return_tensors="pt", add_special_tokens=False).input_ids[0]

print(f"Program 1 tokens: {len(tokens_1)}")
print(f"Program 2 tokens: {len(tokens_2)}")
print(f"Programs same length: {len(tokens_1) == len(tokens_2)}")

# Display tokens for inspection
print("\nProgram 1 tokens:")
for i, token in enumerate(tokens_1):
    print(f"  {i:2d}: '{token}'")

print("\nProgram 2 tokens:")  
for i, token in enumerate(tokens_2):
    print(f"  {i:2d}: '{token}'")

# TODO: Find the exact positions of the space tokens
# Look for the space token that comes after "#a:" and "#l:"
# This is where the intervention will be applied

# HINT: You're looking for tokens that contain a space character
# and come after the query tokens ("#a:" and "#l:")

# find the position of tokens containing "#a:" and "#l:"
regex = r"#c:|#w:"
space_pos_program_1 = re.search(regex, "".join(tokens_1)).end() + 1
space_pos_program_2 = re.search(regex, "".join(tokens_2)).end() + 1


# TODO: Implement the position finding logic here
# Example approach:
# 1. Find the position of tokens containing "#a:" and "#l:"
# 2. The space token should be right after these
# 3. Verify the positions make sense

print(f"\n📍 Key positions to find:")
print(f"  Space token after '#a:' in Program 1: {space_pos_program_1}")
print(f"  Space token after '#l:' in Program 2: {space_pos_program_2}")


#%%
# =============================================================================
# LAYER-WISE PATCHING EXPERIMENT
# =============================================================================

print("🧪 Setting up layer-wise patching experiment...")

# This is where the main experiment happens
# TODO: Implement the cross-query patching experiment

def run_pointer_test_at_layer(tracer, layer_idx: int) -> InterventionResult:
    """
    Run the pointer test by patching the space token from Program 2 into Program 1.
    
    TODO: You need to implement this function to:
    1. Use the CausalTracer to run residual stream intervention
    2. Patch from space_pos_program_2 in Program 2 to space_pos_program_1 in Program 1
    3. Return the intervention result
    
    Args:
        layer_idx: Which layer to perform the intervention at
        
    Returns:
        InterventionResult containing the effects of the intervention
    """
    result = tracer.run_residual_stream_intervention(
        original_program=PROGRAM_1,
        counterfactual_program=PROGRAM_2,
        target_token_pos=space_pos_program_1,
        layer_idx=layer_idx,
        store_logits=True,
        program_id=layer_idx,
        target_name="pointer_test"
    )
    
    return result

# Store results for all layers
all_results = []

print("Running interventions across all layers...")
print("This will test at which layers the pointer information affects the output.")

# TODO: Implement the full experiment loop
# You should:
# 1. Make sure space_pos_program_1 and space_pos_program_2 are found first
# 2. Loop through all layers (0 to tracer._n_layers - 1)
# 3. Run the intervention at each layer
# 4. Store the results

# Example loop structure:
# for layer_idx in range(tracer._n_layers):
#     print(f"Testing layer {layer_idx}/{tracer._n_layers - 1}")
#     result = run_pointer_test_at_layer(layer_idx)
#     if result:
#         all_results.append(result)

print("TODO: Implement the intervention loop above")

#%%
# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

print("📊 Analyzing results...")

# TODO: Implement results analysis
# You want to understand:
# 1. At which layers does the intervention have the strongest effect?
# 2. Does the effect change around layer 28 (where you saw the "jump" before)?
# 3. What does this tell us about pointer vs value tracking?

# Example analysis you might want to do:
# - Plot normalized_logit_difference by layer
# - Look for layers where intervention causes output to change
# - Compare with the layer 28 transition you observed before

if all_results:
    print(f"📈 Got results for {len(all_results)} layers")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{
        'layer_idx': r.layer_idx,
        'normalized_logit_difference': r.normalized_logit_difference,
        'success_rate': r.success_rate,
        'original_top_token': r.original_top_token,
        'intervened_top_token': r.intervened_top_token
    } for r in all_results])
    
    print("\nTop 5 strongest intervention effects:")
    top_effects = df.nlargest(5, 'normalized_logit_difference')
    for _, row in top_effects.iterrows():
        print(f"  Layer {row['layer_idx']:2d}: {row['normalized_logit_difference']:.4f}")
    
    print("\nLayers where intervention changed the output:")
    changed_output = df[df['success_rate'] > 0]
    if len(changed_output) > 0:
        for _, row in changed_output.iterrows():
            print(f"  Layer {row['layer_idx']:2d}: {row['original_top_token']} -> {row['intervened_top_token']}")
    else:
        print("  No layers changed the output (pointer tracking confirmed?)")
        
else:
    print("❌ No results to analyze - implement the intervention loop first!")

#%%
# =============================================================================
# VISUALIZATION
# =============================================================================

print("📈 Creating visualizations...")

# TODO: Create visualizations to understand the results
# Some useful plots might be:
# 1. Line plot of normalized_logit_difference across layers
# 2. Heatmap showing intervention effects
# 3. Comparison with your previous layer 28 results

if all_results:
    
    # Plot 1: Intervention effects across layers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Normalized logit difference by layer
    ax1.plot(df['layer_idx'], df['normalized_logit_difference'], 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Normalized Logit Difference') 
    ax1.set_title('Pointer Test: Intervention Effects by Layer')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add annotation for layer 28 if it exists
    if 28 in df['layer_idx'].values:
        layer_28_effect = df[df['layer_idx'] == 28]['normalized_logit_difference'].iloc[0]
        ax1.axvline(x=28, color='r', linestyle='--', alpha=0.7)
        ax1.annotate(f'Layer 28\n{layer_28_effect:.3f}', 
                    xy=(28, layer_28_effect), xytext=(28+2, layer_28_effect+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Success rate by layer
    ax2.plot(df['layer_idx'], df['success_rate'], 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Success Rate (Output Changed)')
    ax2.set_title('Pointer Test: Output Changes by Layer')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / "pointer_test_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_path}")
    
    plt.show()
    
else:
    print("❌ No results to visualize - implement the intervention loop first!")

print("\n🎯 Interpretation Guide:")
print("=" * 50)
print("If the model uses POINTER tracking:")
print("  - Interventions should have minimal effect (normalized_logit_difference ≈ 0)")
print("  - The model maintains abstract structural information")
print("  - Output should remain '6' even after patching")
print()
print("If the model uses VALUE tracking:")
print("  - Interventions might have stronger effects")  
print("  - The model stores concrete values at token positions")
print("  - Output might change based on what values are patched")
print()
print("Key layers to watch:")
print("  - Early layers (1-10): Where basic token representations are built")
print("  - Layer 28: Where you observed the 'jump' in previous experiments")
print("  - Late layers (25-28): Where final answer generation happens")

#%%
# =============================================================================
# CLEANUP
# =============================================================================

print("🧹 Cleaning up...")

# Clean up GPU memory 
del tracer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ Cleanup completed!")
print(f"📁 Results saved in: {output_dir}")

print("\n" + "=" * 60)
print("🎉 POINTER TEST NOTEBOOK READY!")
print("=" * 60)
print("Next steps:")
print("1. Run the cells above step by step")
print("2. Fill in the TODOs to implement token position finding")
print("3. Implement the intervention loop")
print("4. Analyze results to test the pointer hypothesis")
print("5. Compare with your previous layer 28 findings")
print()
print("Happy experimenting! 🔬")

#%%