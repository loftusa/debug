"""Orchestrates complete causal tracing experiments for variable binding analysis."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .generators import make_variable_binding_program_with_metadata
from .counterfactual import CounterfactualGenerator
from .causal_tracing import CausalTracer, InterventionResult
from .token_analyzer import VariableChain


@dataclass
class CausalExperimentConfig:
    """Configuration for causal tracing experiments."""
    name: str
    model_name: str
    num_programs: int = 10
    seq_lens: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    intervention_types: List[str] = field(default_factory=lambda: ["residual_stream"])
    max_layers: Optional[int] = None
    filter_systematic: bool = True  # Focus on systematic vs heuristic cases
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.intervention_types:
            raise ValueError("Must specify at least one intervention type")
        
        valid_types = ["residual_stream", "attention_head"]
        for itype in self.intervention_types:
            if itype not in valid_types:
                raise ValueError(f"Invalid intervention type: {itype}. Must be one of {valid_types}")


@dataclass 
class CausalExperimentResult:
    """Results from a complete causal tracing experiment."""
    config: CausalExperimentConfig
    program_results: List[Dict[str, Any]]  # Results for each program
    summary_stats: Dict[str, Any]
    intervention_results: List[InterventionResult]
    output_dir: Path
    timestamp: str


class CausalExperimentRunner:
    """Orchestrates systematic causal tracing experiments."""
    
    def __init__(self, output_dir: str = "results/causal_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self._causal_tracer = None  # Lazy loading
        self._counterfactual_generator = CounterfactualGenerator()
    
    def run(self, config: CausalExperimentConfig, save: bool = True) -> CausalExperimentResult:
        """
        Run a complete causal tracing experiment.
        
        Args:
            config: Experiment configuration
            save: Whether to save results to disk
            
        Returns:
            CausalExperimentResult with all experimental data
        """
        print(f"\n=== Running Causal Tracing Experiment: {config.name} ===")
        print(f"Model: {config.model_name}")
        print(f"Programs: {config.num_programs} per seq_len")
        print(f"Sequence lengths: {config.seq_lens}")
        print(f"Intervention types: {config.intervention_types}")
        
        # Validate configuration
        self._validate_config(config)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{config.name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)
        
        # Generate test programs
        print("\nGenerating test programs...")
        programs = self._generate_programs(
            seq_lens=config.seq_lens,
            num_programs=config.num_programs,
            random_seed=config.random_seed
        )
        
        # Filter for systematic cases if requested
        if config.filter_systematic:
            systematic_programs = [p for p in programs if self._is_systematic_case(p)]
            print(f"Filtered to {len(systematic_programs)}/{len(programs)} systematic cases")
            programs = systematic_programs
        
        # Initialize causal tracer
        print(f"\nInitializing causal tracer with {config.model_name}...")
        self._causal_tracer = CausalTracer(config.model_name)
        
        # Run interventions on each program
        print("\nRunning causal interventions...")
        all_intervention_results = []
        program_results = []
        
        for i, program_data in enumerate(programs):
            print(f"\nProgram {i+1}/{len(programs)} (seq_len={program_data['seq_len']})")
            
            try:
                # Run interventions for this program
                program_interventions = self._run_program_interventions(
                    program_data, config
                )
                
                all_intervention_results.extend(program_interventions)
                
                # Store program-level results
                program_result = {
                    "program_id": i,
                    "seq_len": program_data["seq_len"],
                    "original_program": program_data["original_program"],
                    "counterfactual_program": program_data["counterfactual_program"],
                    "num_interventions": len(program_interventions),
                    "best_success_rate": max([r.success_rate or 0.0 for r in program_interventions]) if program_interventions else 0.0,
                    "metadata": program_data["metadata"]
                }
                program_results.append(program_result)
                
            except Exception as e:
                print(f"  ERROR processing program {i}: {e}")
                program_results.append({
                    "program_id": i,
                    "seq_len": program_data["seq_len"],
                    "error": str(e)
                })
        
        # Aggregate results
        print("\nAggregating results...")
        summary_stats = self._create_summary_statistics(
            program_results, all_intervention_results
        )
        
        # Create experiment result
        result = CausalExperimentResult(
            config=config,
            program_results=program_results,
            summary_stats=summary_stats,
            intervention_results=all_intervention_results,
            output_dir=exp_dir,
            timestamp=timestamp
        )
        
        # Save results
        if save:
            self._save_experiment_results(result, exp_dir)
        
        # Store for analysis
        self.results.append(result)
        
        print(f"\nExperiment completed!")
        print(f"Results saved to: {exp_dir}")
        print(f"Overall success rate: {summary_stats.get('mean_success_rate', 0.0):.2%}")
        
        return result
    
    def _generate_programs(self, seq_lens: List[int], num_programs: int, random_seed: int) -> List[Dict[str, Any]]:
        """Generate test programs with counterfactuals."""
        programs = []
        base_rng = np.random.RandomState(random_seed)
        
        for seq_len in seq_lens:
            for i in range(num_programs):
                # Create unique seed for this program
                program_seed = base_rng.randint(0, 1000000)
                rng = np.random.RandomState(program_seed)
                
                # Generate original program with metadata
                program, answer, query_hops, metadata = make_variable_binding_program_with_metadata(
                    seq_len=seq_len, 
                    rng=rng, 
                    tokenizer=self._get_tokenizer()
                )
                
                # Generate counterfactual
                counterfactual = self._counterfactual_generator.create_counterfactual(
                    program, metadata["query_var"]
                )
                
                programs.append({
                    "seq_len": seq_len,
                    "program_id": f"{seq_len}_{i}",
                    "original_program": program,
                    "counterfactual_program": counterfactual,
                    "answer": answer,
                    "query_hops": query_hops,
                    "metadata": metadata,
                    "random_seed": program_seed
                })
        
        return programs
    
    def _is_systematic_case(self, program_data: Dict[str, Any]) -> bool:
        """Filter for systematic variable binding vs shallow heuristics."""
        metadata = program_data["metadata"]
        chain = metadata["variable_chain"]
        
        # Focus on cases with meaningful variable chains (depth > 2)
        # to avoid line-1/line-2 heuristic cases
        if chain.referential_depth <= 2:
            return False
        
        # Ensure we have a clear root value
        if chain.root_value is None or chain.is_circular:
            return False
        
        return True
    
    def _run_program_interventions(self, program_data: Dict[str, Any], config: CausalExperimentConfig) -> List[InterventionResult]:
        """Run all interventions for a single program."""
        results = []
        metadata = program_data["metadata"]
        
        # Extract intervention targets
        intervention_targets = self._extract_intervention_targets(metadata)
        
        for target_name, target_pos in intervention_targets.items():
            if target_pos is None:
                continue
                
            print(f"  Intervening on {target_name} at position {target_pos}")
            
            # Run interventions for each type
            for intervention_type in config.intervention_types:
                if intervention_type == "residual_stream":
                    layer_results = self._causal_tracer.run_systematic_intervention(
                        original_program=program_data["original_program"],
                        counterfactual_program=program_data["counterfactual_program"],
                        target_token_pos=target_pos,
                        max_layers=config.max_layers
                    )
                    results.extend(layer_results)
                
                elif intervention_type == "attention_head":
                    # For attention heads, we'd typically test multiple heads
                    # For now, just test a few representative layers and heads
                    test_layers = [2, 4, 6] if config.max_layers is None else list(range(min(3, config.max_layers)))
                    for layer_idx in test_layers:
                        for head_idx in [0, 2, 4]:  # Test a few heads
                            try:
                                result = self._causal_tracer.run_attention_head_intervention(
                                    original_program=program_data["original_program"],
                                    counterfactual_program=program_data["counterfactual_program"],
                                    target_token_pos=target_pos,
                                    layer_idx=layer_idx,
                                    head_idx=head_idx
                                )
                                results.append(result)
                            except Exception as e:
                                print(f"    Error with head {head_idx} at layer {layer_idx}: {e}")
        
        return results
    
    def _extract_intervention_targets(self, metadata: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """Extract intervention target positions from metadata."""
        intervention_targets = metadata.get("intervention_targets", {})
        
        # Focus on RHS tokens and query tokens
        relevant_targets = {}
        
        # Add RHS tokens based on referential depth
        chain = metadata["variable_chain"]
        for depth in range(1, min(chain.referential_depth + 1, 5)):  # Up to depth 4
            key = f"ref_depth_{depth}_rhs"
            if key in intervention_targets:
                relevant_targets[key] = intervention_targets[key]
        
        # Add query tokens
        for key in ["query_var", "query_colon"]:
            if key in intervention_targets:
                relevant_targets[key] = intervention_targets[key]
        
        return relevant_targets
    
    def _aggregate_results(self, results: List[InterventionResult], target_name: str) -> Dict[str, Any]:
        """Aggregate intervention results for analysis."""
        if not results:
            return {}
        
        success_rates = [r.success_rate for r in results if r.success_rate is not None]
        logit_diffs = [r.logit_difference for r in results if r.logit_difference is not None]
        
        summary = {
            "target_name": target_name,
            "num_interventions": len(results),
            "mean_success_rate": np.mean(success_rates) if success_rates else 0.0,
            "max_success_rate": np.max(success_rates) if success_rates else 0.0,
            "mean_logit_difference": np.mean(logit_diffs) if logit_diffs else 0.0,
            "std_logit_difference": np.std(logit_diffs) if logit_diffs else 0.0,
        }
        
        # Find best performing layer
        if success_rates:
            best_idx = np.argmax(success_rates)
            summary["best_layer"] = results[best_idx].layer_idx
            summary["best_success_rate"] = success_rates[best_idx]
        
        return summary
    
    def _create_summary_statistics(self, program_results: List[Dict[str, Any]], 
                                 intervention_results: List[InterventionResult]) -> Dict[str, Any]:
        """Create overall experiment summary statistics."""
        
        # Overall statistics
        total_programs = len(program_results)
        successful_programs = len([p for p in program_results if "error" not in p])
        
        # Intervention statistics
        all_success_rates = [r.success_rate for r in intervention_results if r.success_rate is not None]
        all_logit_diffs = [r.logit_difference for r in intervention_results if r.logit_difference is not None]
        
        summary = {
            "total_programs": total_programs,
            "successful_programs": successful_programs,
            "success_rate": successful_programs / total_programs if total_programs > 0 else 0.0,
            "total_interventions": len(intervention_results),
            "mean_success_rate": np.mean(all_success_rates) if all_success_rates else 0.0,
            "max_success_rate": np.max(all_success_rates) if all_success_rates else 0.0,
            "mean_logit_difference": np.mean(all_logit_diffs) if all_logit_diffs else 0.0,
            "std_logit_difference": np.std(all_logit_diffs) if all_logit_diffs else 0.0
        }
        
        # By sequence length
        seq_len_stats = {}
        for program in program_results:
            if "error" in program:
                continue
            seq_len = program["seq_len"]
            if seq_len not in seq_len_stats:
                seq_len_stats[seq_len] = []
            seq_len_stats[seq_len].append(program["best_success_rate"])
        
        summary["by_seq_len"] = {
            seq_len: {
                "count": len(rates),
                "mean_success_rate": np.mean(rates) if rates else 0.0
            }
            for seq_len, rates in seq_len_stats.items()
        }
        
        return summary
    
    def _validate_config(self, config: CausalExperimentConfig):
        """Validate experiment configuration."""
        if not config.intervention_types:
            raise ValueError("Must specify at least one intervention type")
        
        if config.num_programs <= 0:
            raise ValueError("num_programs must be positive")
        
        if not config.seq_lens:
            raise ValueError("Must specify at least one sequence length")
    
    def _get_tokenizer(self):
        """Get tokenizer for program generation."""
        if self._causal_tracer is not None:
            return self._causal_tracer.tokenizer
        else:
            # Import here to avoid circular dependency
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    def _save_experiment_results(self, result: CausalExperimentResult, output_dir: Path):
        """Save experiment results to disk."""
        
        # Save detailed results
        results_data = {
            "config": {
                "name": result.config.name,
                "model_name": result.config.model_name,
                "num_programs": result.config.num_programs,
                "seq_lens": result.config.seq_lens,
                "intervention_types": result.config.intervention_types,
                "max_layers": result.config.max_layers,
                "filter_systematic": result.config.filter_systematic,
                "random_seed": result.config.random_seed
            },
            "program_results": result.program_results,
            "summary_stats": result.summary_stats,
            "timestamp": result.timestamp
        }
        
        with open(output_dir / "experiment_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save intervention results separately (they're large)
        intervention_data = []
        for r in result.intervention_results:
            intervention_data.append({
                "intervention_type": r.intervention_type,
                "layer_idx": r.layer_idx,
                "head_idx": r.head_idx,
                "target_token_pos": r.target_token_pos,
                "logit_difference": r.logit_difference,
                "normalized_logit_difference": r.normalized_logit_difference,
                "success_rate": r.success_rate,
                "original_top_token": r.original_top_token,
                "intervened_top_token": r.intervened_top_token
            })
        
        with open(output_dir / "intervention_results.json", "w") as f:
            json.dump(intervention_data, f, indent=2)
        
        print(f"Detailed results saved to {output_dir}")
    
    def create_visualizations(self, result: CausalExperimentResult, save_plots: bool = True) -> Dict[str, Any]:
        """Create comprehensive visualizations for the experiment results."""
        from .causal_visualization import (
            plot_layer_intervention_effects,
            plot_success_rate_heatmap,
            plot_referential_depth_analysis,
            plot_causal_flow_heatmap,
            plot_token_level_causal_trace,
            create_intervention_summary_plot
        )
        
        plots = {}
        save_dir = result.output_dir if save_plots else None
        
        print(f"\nCreating visualizations for {result.config.name}...")
        
        # 1. Layer intervention effects
        if result.intervention_results:
            fig1, ax1 = plot_layer_intervention_effects(
                result.intervention_results,
                save_path=save_dir / "layer_effects.png" if save_dir else None
            )
            plots['layer_effects'] = (fig1, ax1)
        
        # 2. Success rate heatmap
        if result.intervention_results:
            fig2, ax2 = plot_success_rate_heatmap(
                result.intervention_results,
                save_path=save_dir / "success_heatmap.png" if save_dir else None
            )
            plots['success_heatmap'] = (fig2, ax2)
        
        # 3. Referential depth analysis
        if result.program_results:
            fig3, ax3 = plot_referential_depth_analysis(
                result.program_results,
                save_path=save_dir / "depth_analysis.png" if save_dir else None
            )
            plots['depth_analysis'] = (fig3, ax3)
        
        # 4. Create token-level causal traces for top programs
        for i, program in enumerate(result.program_results[:3]):  # Top 3 programs
            if 'error' in program:
                continue
                
            # Get interventions for this program
            program_interventions = [
                r for r in result.intervention_results 
                if hasattr(r, 'program_id') and r.program_id == program.get('program_id')
            ]
            
            if program_interventions:
                fig4, ax4 = plot_token_level_causal_trace(
                    program_interventions,
                    program['original_program'],
                    save_path=save_dir / f"causal_trace_program_{i+1}.png" if save_dir else None
                )
                plots[f'causal_trace_{i+1}'] = (fig4, ax4)
        
        # 5. Create causal flow heatmap if we have token information
        if result.intervention_results and result.program_results:
            # Extract token labels from first program
            first_program = next((p for p in result.program_results if 'error' not in p), None)
            if first_program:
                program_text = first_program['original_program']
                token_labels = program_text.split()
                
                # Create information movement annotations based on intervention results
                information_movements = self._extract_information_movements(result.intervention_results)
                
                fig5, ax5 = plot_causal_flow_heatmap(
                    result.intervention_results,
                    token_labels,
                    information_movements,
                    save_path=save_dir / "causal_flow_heatmap.png" if save_dir else None
                )
                plots['causal_flow_heatmap'] = (fig5, ax5)
        
        # 6. Comprehensive summary plot
        fig6 = create_intervention_summary_plot(
            result,
            save_path=save_dir / "experiment_summary.png" if save_dir else None
        )
        plots['summary'] = fig6
        
        print(f"✅ Created {len(plots)} visualizations")
        if save_plots:
            print(f"Plots saved to {result.output_dir}")
        
        return plots
    
    def _extract_information_movements(self, intervention_results: List[InterventionResult]) -> List[Dict[str, Any]]:
        """Extract information movement patterns from intervention results."""
        movements = []
        
        # Group by layer and find significant effects
        layer_effects = {}
        for result in intervention_results:
            if result.success_rate and result.success_rate > 0.5:  # Significant threshold
                layer = result.layer_idx
                if layer not in layer_effects:
                    layer_effects[layer] = []
                layer_effects[layer].append(result)
        
        # Create movement annotations for layers with significant effects
        movement_count = 1
        for layer, effects in sorted(layer_effects.items()):
            if len(effects) >= 2:  # Multiple significant effects in this layer
                movements.append({
                    "layer": layer,
                    "description": f"Information movement {movement_count}\nLayer {layer} interventions"
                })
                movement_count += 1
        
        return movements


def quick_causal_experiment(name: str, model_name: str, **kwargs) -> CausalExperimentConfig:
    """Quick setup for causal tracing experiments."""
    return CausalExperimentConfig(
        name=name,
        model_name=model_name,
        **kwargs
    )