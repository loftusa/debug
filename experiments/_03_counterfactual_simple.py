#!/usr/bin/env python3
"""
Simplified counterfactual pair collection using the debug framework.
Replaces the complex _03_collect_correct_pairs.py with clean, modular code.
"""

#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, generators, prompts, quick_experiment
from debug.core import parse_integer
import click
import json
from pathlib import Path
import numpy as np

def collect_counterfactual_pairs(model_name: str, num_pairs: int = 10, seq_len: int = 6):
    """Collect counterfactual pairs where model gets both programs correct."""
    
    # Create a simple experiment to test pairs
    config = quick_experiment(
        name="counterfactual_test",
        prompt_template=prompts.RANGE_TRACKING,
        program_generator=lambda sl, rng: generators.make_counterfactual_pair(sl, sl//2, rng),
        models=[model_name],
        num_seqs=1,  # We'll handle the iteration manually
        seq_lens=[seq_len]
    )
    
    runner = ExperimentRunner()
    correct_pairs = []
    rng = np.random.RandomState(42)
    
    print(f"Collecting counterfactual pairs with {model_name}...")
    
    attempts = 0
    while len(correct_pairs) < num_pairs and attempts < num_pairs * 10:  # Max attempts
        attempts += 1
        
        # Generate a counterfactual pair
        prog_a, prog_b, intermediates_a, intermediates_b = generators.make_counterfactual_pair(
            seq_len, seq_len // 2, rng
        )
        
        true_val_a = intermediates_a[-1]
        true_val_b = intermediates_b[-1]
        
        # Test both programs with the model
        try:
            # Create temporary configs for each program
            config_a = quick_experiment(
                name="temp_a",
                prompt_template=prompts.RANGE_TRACKING,
                program_generator=lambda sl, rng: (prog_a, true_val_a),
                models=[model_name],
                num_seqs=1,
                seq_lens=[seq_len]
            )
            
            config_b = quick_experiment(
                name="temp_b", 
                prompt_template=prompts.RANGE_TRACKING,
                program_generator=lambda sl, rng: (prog_b, true_val_b),
                models=[model_name],
                num_seqs=1,
                seq_lens=[seq_len]
            )
            
            # Run both
            result_a = runner.run(config_a)
            result_b = runner.run(config_b)
            
            # Check if both are correct
            correct_a = result_a['results'][0]['correct']
            correct_b = result_b['results'][0]['correct']
            
            if correct_a and correct_b:
                correct_pairs.append({
                    "program_a": prog_a,
                    "program_b": prog_b,
                    "true_val_a": true_val_a,
                    "true_val_b": true_val_b,
                    "intermediates_a": intermediates_a,
                    "intermediates_b": intermediates_b
                })
                print(f"Found pair {len(correct_pairs)}/{num_pairs} (attempt {attempts})")
            
        except Exception as e:
            print(f"Error testing pair {attempts}: {e}")
            continue
    
    print(f"Collected {len(correct_pairs)} correct pairs in {attempts} attempts")
    return correct_pairs

@click.command()
@click.option("--model", default="Qwen/Qwen3-1.7B", help="Model to use for collection")
@click.option("--num-pairs", default=10, help="Number of correct pairs to collect")
@click.option("--seq-len", default=6, help="Sequence length")
@click.option("--output-file", default="correct_counterfactual_pairs.json", help="Output file")
def main(model: str, num_pairs: int, seq_len: int, output_file: str):
    """Collect counterfactual pairs where the model gets both correct."""
    
    pairs = collect_counterfactual_pairs(model, num_pairs, seq_len)
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"\nSaved {len(pairs)} pairs to {output_path}")
    
    # Show examples
    if pairs:
        print("\nExample pair:")
        pair = pairs[0]
        print(f"Program A (result: {pair['true_val_a']}):")
        print(pair['program_a'])
        print(f"\nProgram B (result: {pair['true_val_b']}):")
        print(pair['program_b'])

if __name__ == "__main__":
    main()

#%% Interactive testing
if __name__ == "__main__":
    # Quick test to generate and inspect a counterfactual pair
    import numpy as np
    
    rng = np.random.RandomState(42)
    prog_a, prog_b, int_a, int_b = generators.make_counterfactual_pair(6, 3, rng)
    
    print("Program A:")
    print(prog_a)
    print(f"Intermediates: {int_a}")
    print(f"\nProgram B:")
    print(prog_b) 
    print(f"Intermediates: {int_b}")
    print(f"\nDivergence: A ends with {int_a[-1]}, B ends with {int_b[-1]}") 