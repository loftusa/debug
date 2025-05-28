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
from transformers import pipeline, AutoTokenizer

from .core import ExperimentConfig


class ExperimentRunner:
    """Simple experiment runner."""
    
    def __init__(self, output_dir: str = "results/debug_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self._loaded_models = {}  # Cache for loaded models
    
    def run(self, config: ExperimentConfig, save=True, no_cache=False, **inference_kwargs) -> Dict:
        """Run a single experiment configuration.
        
        Args:
            config: Experiment configuration
            save: Whether to save results to disk
            no_cache: If True, unload each model after use to save GPU memory
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
        
        master_rng = np.random.RandomState(12345)
        all_results = []
        
        # Default inference settings
        inference_settings = {
            "max_new_tokens": 10,
            "temperature": 0.0,
            "do_sample": False,
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
                correct = 0
                total = 0
                
                for i in range(config.num_seqs):
                    # Generate test case
                    program, true_answer = config.program_generator(seq_len, master_rng)
                    prompt = config.prompt_template.format(code=program)
                    
                    try:
                        # Get model prediction
                        outputs = llm(prompt, **inference_settings)
                        generated = outputs[0]["generated_text"][len(prompt):]
                        predicted_answer = config.answer_parser(generated)
                        
                        # Check correctness
                        is_correct = predicted_answer == true_answer
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        # Store result
                        result = {
                            "experiment": config.name,
                            "model_id": model_id,
                            "seq_len": seq_len,
                            "seq_id": i,
                            "program": program,
                            "true_answer": true_answer,
                            "predicted_answer": predicted_answer,
                            "correct": is_correct,
                            "raw_output": generated.strip()
                        }
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"    ERROR seq {i}: {e}")
                        total += 1
                
                accuracy = correct / total if total > 0 else 0
                print(f"  seq_len {seq_len}: {accuracy:.1%} ({correct}/{total})")
            
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
            
            plt.plot(seq_lens, accuracies, marker='o', label=model, linewidth=2)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Accuracy") 
        plt.title(f"Accuracy vs Sequence Length{' - ' + experiment_name if experiment_name else ''}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.show()
    
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
                device_map="auto",
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