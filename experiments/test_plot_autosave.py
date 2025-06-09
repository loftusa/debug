#!/usr/bin/env python3

import sys
sys.path.append('../src')
import json
from debug import ExperimentRunner

def test_plot_autosave():
    """Test the modified plot method with auto-save functionality."""
    
    # Load existing results into the runner
    results_path = "results/debug_experiments/variable_binding_20250528_142250/results.json"
    
    print("Loading existing results...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create runner and load results
    runner = ExperimentRunner(output_dir="results/debug_experiments")
    runner.results = results
    
    print(f"Loaded {len(results)} results into runner")
    
    # Test the plot method with auto-save
    print("\nTesting plot method with auto-save...")
    runner.plot(experiment_name="variable_binding", figsize=(12, 8))
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_plot_autosave() 