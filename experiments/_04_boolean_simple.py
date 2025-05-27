#!/usr/bin/env python3
"""
Simplified boolean logic experiment using the debug framework.
Replaces the complex _04_boolean_experiment.py with clean, modular code.
"""

#%%
import sys
sys.path.append('../src')

from debug import ExperimentRunner, prompts
from debug.core import ExperimentConfig, parse_boolean
import numpy as np
from typing import Tuple
import click

def make_boolean_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, bool]:
    """Generate boolean logic programs."""
    ops = ["and", "or", "not"]
    
    # Start with random boolean values
    x_val = rng.choice([True, False])
    y_val = rng.choice([True, False])
    
    lines = [f"x = {x_val}", f"y = {y_val}"]
    
    for i in range(seq_len):
        op = rng.choice(ops)
        if op == "not":
            lines.append("x = not x")
            x_val = not x_val
        else:
            lines.append(f"x = x {op} y")
            if op == "and":
                x_val = x_val and y_val
            else:  # or
                x_val = x_val or y_val
    
    program = "\n".join(lines)
    return program, x_val

# Boolean experiment configuration
BOOLEAN_CONFIG = ExperimentConfig(
    name="boolean_logic",
    prompt_template=prompts.BOOLEAN_LOGIC,
    program_generator=make_boolean_program,
    answer_parser=parse_boolean,
    models=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
    num_seqs=100,
    seq_lens=[2, 3, 4, 5, 6, 7, 8]
)

@click.command()
@click.option("--output-dir", default="results/boolean_logic", help="Output directory")
@click.option("--models", default=None, help="Comma-separated model list (overrides config)")
@click.option("--num-seqs", default=100, help="Number of sequences per length")
def main(output_dir: str, models: str, num_seqs: int):
    """Run boolean logic experiment with the debug framework."""
    
    # Override config if needed
    config = BOOLEAN_CONFIG
    if models:
        config.models = [m.strip() for m in models.split(',')]
    config.num_seqs = num_seqs
    
    # Run experiment
    runner = ExperimentRunner(output_dir)
    result = runner.run(config)
    
    # Create plots
    runner.plot("boolean_logic")
    runner.analyze_errors("boolean_logic", n_examples=5)
    
    print(f"\nExperiment completed! Results saved to: {result['output_dir']}")
    print(f"Overall accuracy: {result['summary']['overall_accuracy']:.1%}")

if __name__ == "__main__":
    main()

#%% Interactive testing
if __name__ == "__main__":
    # Quick test with smaller scope
    test_config = ExperimentConfig(
        name="boolean_test",
        prompt_template=prompts.BOOLEAN_LOGIC,
        program_generator=make_boolean_program,
        answer_parser=parse_boolean,
        models=["Qwen/Qwen3-0.6B"],
        num_seqs=5,
        seq_lens=[2, 3, 4]
    )
    
    runner = ExperimentRunner()
    result = runner.run(test_config)
    runner.plot("boolean_test") 