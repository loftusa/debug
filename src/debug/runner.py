"""Experiment runner for debug experiments."""

import json
import gc
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer

from .core import ExperimentConfig


class ExperimentRunner:
    """Simple experiment runner."""
    
    def __init__(self, output_dir: str = "results/debug_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self._loaded_models = {}  # Cache for loaded models
    
    def run(self, config: ExperimentConfig, save=True, no_cache=False, batch_size=None, **inference_kwargs) -> Dict:
        """Run a single experiment configuration.
        
        Args:
            config: Experiment configuration
            save: Whether to save results to disk
            no_cache: If True, unload each model after use to save GPU memory
            batch_size: Number of prompts to process in each batch (None = all at once)
            **inference_kwargs: Additional arguments for model inference (e.g., temperature)
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{config.name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True)
        
        print(f"\n=== Running: {config.name} ===")
        print(f"Models: {config.models}")
        print(f"Seq lengths: {config.seq_lens}")
        print(f"Output: {exp_dir}")
        if no_cache:
            print("Memory mode: Models will be unloaded after each use")
        if batch_size:
            print(f"Batch size: {batch_size}")
        
        all_results = []
        
        # Default inference settings
        inference_settings = {
            "max_new_tokens": 10,
            "temperature": 0.0,
            "do_sample": False,
            "batch_size": batch_size or config.num_seqs,  # Use batch_size or all at once
            **inference_kwargs
        }
        
        for model_id in config.models:
            print(f"\n--- {model_id} ---")
            
            # Load model (use cache if available)
            llm = self._get_or_load_model(model_id)
            if llm is None:
                continue
            
            # Test each sequence length
            for seq_len in config.seq_lens:
                print(f"  Generating {config.num_seqs} test cases for seq_len {seq_len}...")
                
                # Generate all test cases for this seq_len
                test_cases = []
                for i in range(config.num_seqs):
                    generator_result = config.program_generator(seq_len, rng=np.random.RandomState(i))
                    if len(generator_result) == 2:
                        program, true_answer = generator_result
                        metadata = {}
                    elif len(generator_result) == 3:
                        program, true_answer, query_hops = generator_result
                        metadata = {
                            "query_hops": query_hops
                        }
                    else: 
                        raise ValueError(f"Invalid generator result: {generator_result}")

                    prompt = config.prompt_template.format(code=program)
                    test_cases.append({
                        "seq_id": i,
                        "program": program,
                        "prompt": prompt,
                        "true_answer": true_answer,
                        **metadata
                    })
                
                print(f"  Running batch inference...")
                
                # Extract prompts for batch processing
                prompts = [case["prompt"] for case in test_cases]
                
                try:
                    # Run batch inference
                    outputs = llm(prompts, **inference_settings)
                    
                    # Process results
                    correct = 0
                    for i, (case, output) in enumerate(zip(test_cases, outputs)):
                        try:
                            # Handle both flat list and nested list outputs from transformers pipeline
                            if isinstance(output, list):
                                output = output[0]  # Take first output if list of outputs
                            generated = output["generated_text"][len(case["prompt"]):]
                            predicted_answer = config.answer_parser(generated)
                            
                            # Check correctness
                            is_correct = predicted_answer == case["true_answer"]
                            if is_correct:
                                correct += 1
                            
                            # Store result
                            result = {
                            "experiment": config.name,
                            "model_id": model_id,
                            "seq_len": seq_len,
                            "seq_id": case["seq_id"],
                            "program": case["program"],
                            "true_answer": case["true_answer"],
                            "predicted_answer": predicted_answer,
                            "correct": is_correct,
                            "raw_output": generated.strip(),
                            **{k: v for k, v in case.items() if k not in ["seq_id", "program", "prompt", "true_answer"]}  # Include metadata
                            }
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"    ERROR processing seq {case['seq_id']}: {e}")
                            # Store error result
                            result = {
                                "experiment": config.name,
                                "model_id": model_id,
                                "seq_len": seq_len,
                                "seq_id": case["seq_id"],
                                "program": case["program"],
                                "true_answer": case["true_answer"],
                                "predicted_answer": None,
                                "correct": False,
                                "raw_output": f"ERROR: {e}",
                                **{k: v for k, v in case.items() if k not in ["seq_id", "program", "prompt", "true_answer"]}  # Include metadata
                            }
                            all_results.append(result)
                    
                    accuracy = correct / len(test_cases)
                    print(f"  seq_len {seq_len}: {accuracy:.1%} ({correct}/{len(test_cases)})")
                    
                except Exception as e:
                    print(f"    ERROR in batch inference for seq_len {seq_len}: {e}")
                    # Store error results for all test cases
                    for case in test_cases:
                        result = {
                            "experiment": config.name,
                            "model_id": model_id,
                            "seq_len": seq_len,
                            "seq_id": case["seq_id"],
                            "program": case["program"],
                            "true_answer": case["true_answer"],
                            "predicted_answer": None,
                            "correct": False,
                            "raw_output": f"BATCH_ERROR: {e}",
                            **{k: v for k, v in case.items() if k not in ["seq_id", "program", "prompt", "true_answer"]}  # Include metadata
                        }
                        all_results.append(result)
            
            # Unload model if no_cache is enabled
            if no_cache:
                self.unload_model(model_id)
        
        # Save results
        if save:
            with open(exp_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)
        
        # Create summary
        summary = self._summarize(all_results, config)
        if save:
            with open(exp_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        print(f"\nSaved to: {exp_dir}")
        
        # Store for plotting
        self.results.extend(all_results)
        
        return {
            "config": config,
            "results": all_results,
            "summary": summary,
            "output_dir": exp_dir
        }
    
    def _summarize(self, results: List[Dict], config: ExperimentConfig) -> Dict:
        """Create experiment summary."""
        if not results:
            return {"error": "No results"}
        
        df = pd.DataFrame(results)
        
        summary = {
            "experiment": config.name,
            "total_samples": len(results),
            "overall_accuracy": df["correct"].mean(),
            "by_model": df.groupby("model_id")["correct"].mean().to_dict(),
            "by_seq_len": df.groupby("seq_len")["correct"].mean().to_dict(),
        }
        
        return summary
    
    def plot(self, experiment_name: str = None, figsize=(10, 6)):
        """Plot experiment results."""
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results)
        if experiment_name:
            df = df[df["experiment"] == experiment_name]
        
        if len(df) == 0:
            print(f"No results found for: {experiment_name}")
            return
        
        # Plot accuracy by sequence length
        plt.figure(figsize=figsize)
        
        for model in df["model_id"].unique():
            model_data = df[df["model_id"] == model]
            seq_lens = []
            accuracies = []
            
            for seq_len in sorted(model_data["seq_len"].unique()):
                seq_data = model_data[model_data["seq_len"] == seq_len]
                accuracy = seq_data["correct"].mean()
                seq_lens.append(seq_len)
                accuracies.append(accuracy)
            
            # Clean up model name for legend
            model_name = model.split('/')[-1] if '/' in model else model
            plt.plot(seq_lens, accuracies, marker='o', label=model_name, linewidth=2)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Accuracy") 
        plt.title(f"Accuracy vs Sequence Length{' - ' + experiment_name if experiment_name else ''}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Auto-save the plot
        save_paths = []
        
        # Try to save to the most recent experiment directory for this experiment
        if experiment_name:
            # Find the most recent experiment directory
            exp_dirs = list(self.output_dir.glob(f"{experiment_name}_*"))
            if exp_dirs:
                latest_dir = max(exp_dirs, key=lambda p: p.name)
                plot_path = latest_dir / "accuracy_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                save_paths.append(str(plot_path))
        
        # Also save to a general plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_name = f"{experiment_name}_{timestamp}.png" if experiment_name else f"plot_{timestamp}.png"
        general_plot_path = plots_dir / plot_name
        plt.savefig(general_plot_path, dpi=300, bbox_inches='tight')
        save_paths.append(str(general_plot_path))
        
        plt.show()
        
        # Print save locations
        if save_paths:
            print(f"Plot saved to:")
            for path in save_paths:
                print(f"  {path}")
    
    def analyze_errors(self, experiment_name: str, n_examples: int = 3):
        """Show example errors from an experiment."""
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame(self.results)
        df = df[df["experiment"] == experiment_name]
        errors = df[~df["correct"]]
        
        if len(errors) == 0:
            print("No errors found!")
            return
        
        print(f"\nError Analysis: {experiment_name}")
        print(f"Total errors: {len(errors)}/{len(df)} ({len(errors)/len(df):.1%})")
        
        print(f"\nExample errors:")
        for i, (_, row) in enumerate(errors.head(n_examples).iterrows()):
            print(f"\n--- Error {i+1} ---")
            print(f"Program:\n{row['program']}")
            print(f"Expected: {row['true_answer']}")
            print(f"Got: {row['predicted_answer']}")
            print(f"Raw: '{row['raw_output']}'")
    
    def _get_or_load_model(self, model_id: str):
        """Get model from cache or load it if not cached."""
        if model_id in self._loaded_models:
            print(f"  Using cached model: {model_id}")
            return self._loaded_models[model_id]
        
        print(f"  Loading model: {model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            llm = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tok,
                trust_remote_code=True,
                device_map="cuda",
                torch_dtype=torch.bfloat16
            )
            self._loaded_models[model_id] = llm
            print(f"  Model loaded and cached: {model_id}")
            return llm
        except Exception as e:
            print(f"ERROR loading {model_id}: {e}")
            return None
    
    def preload_models(self, model_ids: List[str]):
        """Preload models for interactive use.
        
        Args:
            model_ids: List of model IDs to preload
        """
        print(f"Preloading {len(model_ids)} models...")
        for model_id in model_ids:
            self._get_or_load_model(model_id)
        print("All models preloaded!")
    
    def unload_model(self, model_id: str):
        """Unload a specific model from cache.
        
        Args:
            model_id: Model ID to unload
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"Unloaded model: {model_id}")
        else:
            print(f"Model not loaded: {model_id}")
    
    def unload_all_models(self):
        """Unload all cached models and free GPU memory."""
        if self._loaded_models:
            print(f"Unloading {len(self._loaded_models)} cached models...")
            self._loaded_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("All models unloaded!")
        else:
            print("No models to unload.")
    
    def list_loaded_models(self):
        """List currently loaded models."""
        if self._loaded_models:
            print("Loaded models:")
            for model_id in self._loaded_models.keys():
                print(f"  - {model_id}")
        else:
            print("No models currently loaded.")
    
    def clear(self):
        """Clear stored results."""
        self.results = []
    
    def plot_heatmap(self, experiment_name: str, figsize=(12, 8)):
        """Create a Tufte-style heatmap showing accuracy by sequence length and query hops."""
        df = pd.DataFrame(self.results)
        df = df[df["experiment"] == experiment_name]
        
        if "query_hops" not in df.columns:
            print("No query_hops data found!")
            return
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(["seq_len", "query_hops"])["correct"].agg(["mean", "count"]).reset_index()
        
        # The agg() creates MultiIndex columns: ('correct', 'mean') and ('correct', 'count')
        # Let's flatten them for easier access
        pivot_data.columns = ['seq_len', 'query_hops', 'accuracy_mean', 'sample_count']
        
        pivot_accuracy = pivot_data.pivot(index="query_hops", columns="seq_len", values="accuracy_mean")
        pivot_counts = pivot_data.pivot(index="query_hops", columns="seq_len", values="sample_count").fillna(0).astype(int)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main heatmap - accuracy
        sns.heatmap(pivot_accuracy, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                    center=0.5, ax=ax1, cbar_kws={'label': 'Accuracy'})
        ax1.set_title(f'{experiment_name}: Accuracy by Query Hops and Sequence Length', 
                      fontsize=14, pad=20)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Query Hops')
        
        # Sample counts heatmap
        sns.heatmap(pivot_counts, annot=True, fmt='d', cmap='Greys', 
                    ax=ax2, cbar_kws={'label': 'Sample Count'})
        ax2.set_title('Sample Counts', fontsize=12)
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Query Hops')
        
        plt.tight_layout()
        return fig

    def plot_mall_multiples(self, experiment_name: str, figsize=(15, 10)):
        """Create small multiples showing accuracy curves for each model and query hop level."""
        df = pd.DataFrame(self.results)
        df = df[df["experiment"] == experiment_name]
        
        if "query_hops" not in df.columns:
            print("No query_hops data found!")
            return
        
        models = df["model_id"].unique()
        hop_levels = sorted(df["query_hops"].unique())
        
        fig, axes = plt.subplots(len(hop_levels), len(models), 
                                figsize=figsize, sharey=True, sharex=True)
        
        if len(models) == 1:
            axes = axes.reshape(-1, 1)
        if len(hop_levels) == 1:
            axes = axes.reshape(1, -1)
        
        for i, hop_level in enumerate(hop_levels):
            for j, model in enumerate(models):
                ax = axes[i, j]
                
                # Filter data for this combination
                subset = df[(df["model_id"] == model) & (df["query_hops"] == hop_level)]
                
                if len(subset) > 0:
                    # Calculate accuracy by sequence length
                    accuracy_data = subset.groupby("seq_len")["correct"].agg(["mean", "count"])
                    
                    # Plot with confidence intervals based on sample size
                    ax.plot(accuracy_data.index, accuracy_data["mean"], 'o-', linewidth=1.5)
                    
                    # Add subtle confidence bands (assuming binomial)
                    se = np.sqrt(accuracy_data["mean"] * (1 - accuracy_data["mean"]) / accuracy_data["count"])
                    ax.fill_between(accuracy_data.index, 
                                   accuracy_data["mean"] - 1.96 * se,
                                   accuracy_data["mean"] + 1.96 * se,
                                   alpha=0.2)
                
                # Tufte-style minimal axes
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                
                # Labels only on edges
                if i == len(hop_levels) - 1:
                    ax.set_xlabel('Sequence Length')
                if j == 0:
                    ax.set_ylabel(f'{hop_level} hops\nAccuracy')
                if i == 0:
                    model_name = model.split('/')[-1] if '/' in model else model
                    ax.set_title(model_name, fontsize=10)
        
        plt.tight_layout()
        return fig

    def plot_slope_graph(self, experiment_name: str, figsize=(10, 8)):
        """Create a slope graph showing how accuracy changes with query hops for different sequence lengths."""
        df = pd.DataFrame(self.results)
        df = df[df["experiment"] == experiment_name]
        
        if "query_hops" not in df.columns:
            print("No query_hops data found!")
            return
        
        # Aggregate across models for clarity
        agg_data = df.groupby(["seq_len", "query_hops"])["correct"].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color map for sequence lengths
        seq_lens = sorted(agg_data["seq_len"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(seq_lens)))
        
        for i, seq_len in enumerate(seq_lens):
            subset = agg_data[agg_data["seq_len"] == seq_len]
            subset = subset.sort_values("query_hops")
            
            ax.plot(subset["query_hops"], subset["correct"], 
                   'o-', color=colors[i], label=f'Length {seq_len}',
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Query Hops')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{experiment_name}: Accuracy vs Query Hops by Sequence Length')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig 