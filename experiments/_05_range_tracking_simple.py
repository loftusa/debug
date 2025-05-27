#!/usr/bin/env python3
"""
Simplified range tracking experiment using the debug framework.
Replaces the complex _05_range_tracking.py with clean, modular code.
"""

#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, generators, prompts, quick_experiment
import click

@click.command()
@click.option("--program-type", default="lines", help="Program type: 'lines' or 'single'")
@click.option("--random-sum", is_flag=True, help="Use different values each iteration")
@click.option("--output-dir", default="results/range_tracking", help="Output directory")
@click.option("--models", default=None, help="Comma-separated model list")
@click.option("--num-seqs", default=100, help="Number of sequences per length")
def main(program_type: str, random_sum: bool, output_dir: str, models: str, num_seqs: int):
    """Run range tracking experiment with the debug framework."""
    
    # Choose generator based on program type
    if program_type == "lines":
        generator = generators.make_range_program_lines
    elif program_type == "single":
        generator = generators.make_range_program
    else:
        raise ValueError(f"Unknown program type: {program_type}")
    
    # Build experiment name
    name = f"range_tracking_{program_type}"
    if random_sum:
        name += "_random"
        generator = generators.make_variable_increments  # Use variable increments for random
    
    # Create configuration
    config = quick_experiment(
        name=name,
        prompt_template=prompts.RANGE_TRACKING,
        program_generator=generator,
        models=models.split(',') if models else ["Qwen/Qwen3-1.7B"],
        num_seqs=num_seqs,
        seq_lens=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    )
    
    # Run experiment
    runner = ExperimentRunner(output_dir)
    result = runner.run(config)
    
    # Create plots and analysis
    runner.plot(name)
    runner.analyze_errors(name, n_examples=5)
    
    print(f"\nExperiment completed! Results saved to: {result['output_dir']}")
    print(f"Overall accuracy: {result['summary']['overall_accuracy']:.1%}")

if __name__ == "__main__":
    main()

# %%

#%% Interactive testing
if __name__ == "__main__":
    # Quick test with range programs
    config = quick_experiment(
        name="range_test",
        prompt_template=prompts.RANGE_TRACKING,
        program_generator=generators.make_range_program,
        models=["Qwen/Qwen3-0.6B"],
        num_seqs=5,
        seq_lens=[2, 3, 4, 5]
    )
    
    runner = ExperimentRunner()
    result = runner.run(config)
    runner.plot("range_test") 