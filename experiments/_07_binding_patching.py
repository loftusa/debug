"""
Causal Tracing for Variable Binding Analysis
============================================

Complete causal tracing experiment using nnsight interventions to understand
how language models track variable bindings in programming code.

Based on the methodology in CLAUDE.md:
- Generate original/counterfactual program pairs
- Run interventions on RHS tokens at different referential depths  
- Measure logit differences and success rates
- Analyze causal mechanisms across layers

Run cells with Ctrl+Enter (VS Code) or Shift+Enter (Jupyter)
"""

#%% Setup and Configuration
import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our complete causal tracing system
from debug.causal_experiment_runner import CausalExperimentRunner, quick_causal_experiment
from debug.causal_visualization import (
    plot_layer_intervention_effects,
    plot_success_rate_heatmap,
    plot_referential_depth_analysis,
    plot_causal_flow_heatmap,
    plot_token_level_causal_trace,
    create_intervention_summary_plot,
    CausalAnalyzer
)
from debug.generators import make_variable_binding_program_with_metadata
from debug.counterfactual import CounterfactualGenerator
from debug.causal_tracing import CausalTracer

print("=== Causal Tracing for Variable Binding ===")
print("🧠 Investigating how transformers track variable references")

#%% Experiment Configuration
print("\n=== Experiment Configuration ===")

# Configure causal tracing experiment
MODELS = [
    "Qwen/Qwen2.5-0.5B",  # Small model for testing
    # "Qwen/Qwen2.5-1.5B",  # Uncomment for larger scale
    # "Qwen/Qwen2.5-7B",
]

causal_config = quick_causal_experiment(
    name="variable_binding_causal_tracing",
    model_name=MODELS[0],
    num_programs=20,  # Increase for full experiment
    seq_lens=[3, 4, 5, 6],  # Focus on systematic cases
    intervention_types=["residual_stream"],  # Can add "attention_head"
    max_layers=12,  # Limit for computational efficiency
    filter_systematic=True,  # Focus on depth > 2 cases
    random_seed=42
)

print(f"📊 Experiment: {causal_config.name}")
print(f"🤖 Model: {causal_config.model_name}")
print(f"📝 Programs: {causal_config.num_programs} per sequence length")
print(f"📏 Sequence lengths: {causal_config.seq_lens}")
print(f"🎯 Intervention types: {causal_config.intervention_types}")
print(f"🔬 Max layers: {causal_config.max_layers}")
print(f"🧠 Filter systematic: {causal_config.filter_systematic}")

#%% Demo: Program Generation and Analysis
print("\n=== Demo: Program Generation ===")

# Generate example programs to understand the data
runner = CausalExperimentRunner(output_dir="../results/causal_experiments")

demo_programs = runner._generate_programs(
    seq_lens=[3, 4], 
    num_programs=3, 
    random_seed=42
)

print(f"Generated {len(demo_programs)} demo programs:")

for i, prog in enumerate(demo_programs[:3]):  # Show first 3
    print(f"\n--- Program {i+1} (seq_len={prog['seq_len']}) ---")
    print("Original:")
    print(prog['original_program'])
    
    print("Counterfactual:")
    print(prog['counterfactual_program'])
    
    # Analyze metadata
    metadata = prog['metadata']
    chain = metadata['variable_chain']
    targets = metadata.get('intervention_targets', {})
    
    print(f"Query: {metadata['query_var']} (depth: {chain.referential_depth})")
    print(f"Chain: {[(var, ref) for var, ref in chain.chain]}")
    print(f"Intervention targets: {len(targets)} positions")
    
    # Check if systematic
    is_systematic = runner._is_systematic_case(prog)
    print(f"Systematic case: {is_systematic}")

#%% Intervention Simulation (No Model Loading)
print("\n=== Intervention Simulation ===")
print("Simulating causal interventions without model loading...")

# Create realistic mock interventions based on research patterns
def simulate_realistic_interventions(programs, max_layers=6):
    """Simulate realistic intervention results based on research patterns."""
    results = []
    
    for prog in programs:
        if not runner._is_systematic_case(prog):
            continue
            
        metadata = prog['metadata']
        targets = runner._extract_intervention_targets(metadata)
        depth = metadata['variable_chain'].referential_depth
        
        for target_name, target_pos in targets.items():
            if target_pos is None:
                continue
                
            # Extract referential depth from target name
            ref_depth = 1  # Default
            if "ref_depth_" in target_name:
                try:
                    ref_depth = int(target_name.split("_")[2])
                except:
                    pass
            
            for layer in range(max_layers):
                # Simulate realistic patterns:
                # - Earlier layers less effective
                # - Shallower references easier to track
                # - Some randomness
                
                base_success = 0.3 + (layer / max_layers) * 0.4  # Layer effect
                depth_penalty = (ref_depth - 1) * 0.1  # Depth penalty
                base_success = max(0.1, base_success - depth_penalty)
                
                # Add realistic noise
                success_rate = np.random.beta(
                    base_success * 10, 
                    (1 - base_success) * 10
                )
                
                logit_diff = np.random.normal(
                    success_rate * 0.3,  # Higher success = higher logit diff
                    0.05
                )
                
                from debug.causal_tracing import InterventionResult
                result = InterventionResult(
                    intervention_type="residual_stream",
                    layer_idx=layer,
                    target_token_pos=target_pos,
                    logit_difference=logit_diff,
                    success_rate=success_rate,
                    original_top_token=50,
                    intervened_top_token=100
                )
                results.append(result)
    
    return results

# Generate simulation data
simulation_results = simulate_realistic_interventions(demo_programs, max_layers=8)
print(f"✅ Simulated {len(simulation_results)} interventions")

# Analyze simulation
analyzer = CausalAnalyzer()
aggregated = analyzer.aggregate_by_target(simulation_results)
critical_layers = analyzer.find_critical_layers(simulation_results, threshold=0.5)

print(f"\n📊 Simulation Analysis:")
print(f"Target groups: {len(aggregated)}")
print(f"Critical layers (>50% success): {critical_layers}")

for target, stats in aggregated.items():
    print(f"\n{target}:")
    print(f"  Mean success: {stats['mean_success_rate']:.2%}")
    print(f"  Max success: {stats['max_success_rate']:.2%}")
    print(f"  Best layer: {stats['best_layer']}")

#%% Visualization: Layer Effects
print("\n=== Visualization: Layer Effects ===")

# Plot layer-wise intervention effects
fig1, ax1 = plot_layer_intervention_effects(simulation_results)
plt.title("Variable Binding: Causal Intervention Effects by Layer")
plt.savefig("../results/layer_intervention_effects.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Layer effects plot saved to ../results/layer_intervention_effects.png")

#%% Visualization: Success Rate Heatmap  
print("\n=== Visualization: Success Rate Heatmap ===")

# Create heatmap of success rates
fig2, ax2 = plot_success_rate_heatmap(simulation_results)
plt.title("Variable Binding: Success Rate by Layer and Target")
plt.savefig("../results/success_rate_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Heatmap saved to ../results/success_rate_heatmap.png")

#%% Visualization: Referential Depth Analysis
print("\n=== Visualization: Referential Depth Analysis ===")

# Analyze by referential depth
depth_programs = []
for depth in [1, 2, 3, 4]:
    for i in range(15):
        # Simulate decreasing success with increasing depth
        success_rate = np.random.beta(5-depth+1, depth+1)
        depth_programs.append({
            "best_success_rate": success_rate,
            "seq_len": depth + 2,
            "metadata": {
                "variable_chain": type('', (), {
                    "referential_depth": depth
                })()
            }
        })

fig3, ax3 = plot_referential_depth_analysis(depth_programs)
plt.title("Variable Binding: Success vs Referential Depth")
plt.savefig("../results/referential_depth_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Depth analysis saved to ../results/referential_depth_analysis.png")

#%% Visualization: Causal Flow Heatmap
print("\n=== Visualization: Causal Flow Heatmap ===")

# Create token-level causal flow visualization
sample_program = demo_programs[0]['original_program']
token_labels = sample_program.split()

# Create information movement annotations
information_movements = [
    {"layer": 3, "description": "First information\nmovement\nMoving root value at layer 3"},
    {"layer": 5, "description": "Second information\nmovement\nMoving root value at layer 5"}, 
    {"layer": 7, "description": "Third information\nmovement\nMoving root value at layer 7"}
]

fig4, ax4 = plot_causal_flow_heatmap(simulation_results, token_labels, information_movements)
plt.title("Variable Binding: Causal Flow Across Tokens and Layers")
plt.savefig("../results/causal_flow_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Causal flow heatmap saved to ../results/causal_flow_heatmap.png")

#%% Visualization: Token-Level Causal Trace
print("\n=== Visualization: Token-Level Causal Trace ===")

# Create detailed token-level causal trace
sample_program_text = demo_programs[0]['original_program']
fig5, ax5 = plot_token_level_causal_trace(simulation_results, sample_program_text)
plt.title("Variable Binding: Detailed Token-Level Causal Trace")
plt.savefig("../results/token_level_causal_trace.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Token-level causal trace saved to ../results/token_level_causal_trace.png")

#%% Complete Experiment Summary
print("\n=== Complete Experiment Summary ===")

# Create comprehensive summary plot
from debug.causal_experiment_runner import CausalExperimentResult

summary_stats = {
    "total_programs": len(demo_programs),
    "successful_programs": len([p for p in demo_programs if runner._is_systematic_case(p)]),
    "total_interventions": len(simulation_results),
    "mean_success_rate": np.mean([r.success_rate for r in simulation_results]),
    "max_success_rate": np.max([r.success_rate for r in simulation_results]),
    "mean_logit_difference": np.mean([r.logit_difference for r in simulation_results]),
    "std_logit_difference": np.std([r.logit_difference for r in simulation_results])
}

experiment_result = CausalExperimentResult(
    config=causal_config,
    program_results=depth_programs,
    summary_stats=summary_stats,
    intervention_results=simulation_results,
    output_dir=Path("../results"),
    timestamp="simulation"
)

fig6 = create_intervention_summary_plot(experiment_result)
plt.savefig("../results/experiment_summary.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Summary plot saved to ../results/experiment_summary.png")

#%% Run Real Experiment (Uncomment to Execute)
print("\n=== Real Experiment Execution ===")

print("""
🚨 Ready to run real causal tracing experiment!

To execute with actual model loading, uncomment and run:

runner = CausalExperimentRunner(output_dir="../results/causal_experiments")
result = runner.run(causal_config, save=True)

This will:
1. Load the specified model with nnsight
2. Generate variable binding programs 
3. Create counterfactual pairs
4. Run systematic interventions across layers
5. Measure logit differences and success rates
6. Save comprehensive results and visualizations

⚠️  Warning: Requires GPU and ~15GB VRAM for 7B models
⏱️  Estimated time: 30-60 minutes depending on model size
""")

# Uncomment below to run actual experiment:
# print("🚀 Running real causal tracing experiment...")
# runner = CausalExperimentRunner(output_dir="../results/causal_experiments")
# result = runner.run(causal_config, save=True)
# 
# print(f"✅ Experiment completed!")
# print(f"📊 Results: {result.output_dir}")
# print(f"🎯 Success rate: {result.summary_stats['mean_success_rate']:.2%}")

#%% Analysis and Insights
print("\n=== Analysis and Insights ===")

print("""
🔬 Key Research Questions Addressed:

1. **Layer-wise Causal Flow**: Which transformer layers are most critical 
   for tracking variable bindings?

2. **Referential Depth Effects**: How does variable chain length affect
   the model's ability to maintain bindings?

3. **Token-specific Interventions**: Do interventions on root values have
   stronger effects than intermediate references?

4. **Systematic vs Heuristic**: Can we distinguish systematic variable 
   tracking from shallow line-based heuristics?

📊 Expected Findings:
- Early-to-middle layers most effective for variable tracking
- Success decreases with referential depth  
- Root value interventions most impactful
- Clear distinction between systematic and heuristic processing

🚀 Next Steps:
1. Run experiments on multiple model sizes
2. Compare different architectures (GPT vs LLama vs Qwen)
3. Analyze attention patterns during variable resolution
4. Investigate failure cases and model limitations
""")

print("\n🎉 Causal Tracing System: READY FOR DISCOVERY!")
print("💡 Transform our understanding of how language models process code")

#%%