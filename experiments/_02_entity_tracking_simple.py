#!/usr/bin/env python3
"""
Simplified entity tracking experiment using the debug framework.
Replaces the complex _02_opensource_entity_tracking.py with clean, modular code.
"""

#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, generators, prompts
from debug.core import ExperimentConfig
import click
from pathlib import Path
from dataclasses import replace

# Use existing generators and prompt
ENTITY_TRACKING_CONFIG = ExperimentConfig(
    name="entity_tracking",
    prompt_template=prompts.RANGE_TRACKING,  # Use existing prompt
    program_generator=generators.make_sequence,  # Random +/- sequences  
    models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
    num_seqs=100,
    seq_lens=[2, 3, 4, 5, 6, 7, 8, 9, 10]
)

@click.command()
@click.option("--output-dir", default="results/entity_tracking", help="Output directory")
@click.option("--models", default=None, help="Comma-separated model list (overrides config)")
@click.option("--num-seqs", default=100, help="Number of sequences per length")
def main(output_dir: str, models: str, num_seqs: int):
    """Run entity tracking experiment with the debug framework."""
    
    # Create a new config with overrides (don't mutate the original)
    config_overrides = {}
    if models:
        config_overrides['models'] = [m.strip() for m in models.split(',')]
    config_overrides['num_seqs'] = num_seqs
    
    config = replace(ENTITY_TRACKING_CONFIG, **config_overrides)
    
    # Run experiment
    runner = ExperimentRunner(output_dir)
    result = runner.run(config)
    
    # Create plots
    runner.plot("entity_tracking")
    runner.analyze_errors("entity_tracking", n_examples=5)
    
    print(f"\nExperiment completed! Results saved to: {result['output_dir']}")
    print(f"Overall accuracy: {result['summary']['overall_accuracy']:.1%}")

if __name__ == "__main__":
    main()

#%% Interactive testing
if __name__ == "__main__":
    # Quick test with smaller scope
    test_config = ExperimentConfig(
        name="entity_test",
        prompt_template=prompts.RANGE_TRACKING,
        program_generator=generators.make_sequence,
        models=["Qwen/Qwen3-0.6B"],
        num_seqs=5,
        seq_lens=[2, 3, 4]
    )
    
    runner = ExperimentRunner()
    result = runner.run(test_config)
    runner.plot("entity_test") 