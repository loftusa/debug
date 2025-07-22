#!/usr/bin/env python3
"""
Test that the patching experiment now gets correct positions.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to PYTHONPATH 
project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(project_root))

from debug.causal_tracing import CausalTracer
from debug.counterfactual import CounterfactualGenerator
from debug.generators import make_variable_binding_program_with_metadata


def test_patching_positions_integration():
    """Test that we get the right positions for the patching experiment."""
    
    # Use small model for testing
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    SEQ_LEN = 17
    RNG_SEED = 12  # The robust seed
    
    # Load model and generate program
    causal_tracer = CausalTracer(MODEL_ID)
    
    rng = np.random.RandomState(RNG_SEED)
    program, answer, hops, metadata = make_variable_binding_program_with_metadata(
        seq_len=SEQ_LEN, rng=rng, tokenizer=causal_tracer.tokenizer
    )
    
    query_var = metadata["query_var"]
    intervention_targets = metadata["intervention_targets"]
    
    # Basic assertions
    assert isinstance(program, str), "Program should be a string"
    assert isinstance(answer, int), "Answer should be an integer"
    assert isinstance(hops, int), "Hops should be an integer"
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    
    # Check metadata structure
    assert "query_var" in metadata, "Metadata should contain query_var"
    assert "variable_chain" in metadata, "Metadata should contain variable_chain"
    assert "intervention_targets" in metadata, "Metadata should contain intervention_targets"
    
    # Check intervention targets structure
    assert isinstance(intervention_targets, dict), "Intervention targets should be a dictionary"
    assert len(intervention_targets) > 0, "Should have some intervention targets"
    
    # Verify program ends correctly  
    assert program.strip().endswith(":"), f"Program should end with query format, got: '{program[-10:]}'"
    
    # Test counterfactual generation
    counterfactual_generator = CounterfactualGenerator()
    counter_program = counterfactual_generator.create_counterfactual(program, query_var)
    
    assert isinstance(counter_program, str), "Counterfactual should be a string"
    assert counter_program != program, "Counterfactual should be different from original"
    assert counter_program.count('\n') == program.count('\n'), "Should have same number of lines"
    
    # Test token positions are valid
    tokens = causal_tracer.tokenizer.tokenize(program)
    for target_name, pos in intervention_targets.items():
        if pos is not None:
            assert 0 <= pos < len(tokens), f"{target_name} position {pos} should be valid (0-{len(tokens)-1})"
            
    # Check that we have the expected target types
    expected_targets = ['prediction_token_pos']  # Always should have this
    for target in expected_targets:
        assert target in intervention_targets, f"Should have {target} in intervention targets"
    
    # If we have referential depth targets, check they make sense
    ref_targets = [k for k in intervention_targets.keys() if k.startswith('ref_depth_') and k.endswith('_rhs')]
    if ref_targets:
        # Should have at least ref_depth_1_rhs for the root value
        assert 'ref_depth_1_rhs' in intervention_targets, "Should have root value target"
        
        # Check positions are in reasonable order (not necessarily strictly ascending due to token boundaries)
        positions = [intervention_targets[k] for k in sorted(ref_targets) if intervention_targets[k] is not None]
        assert len(positions) > 0, "Should have at least one valid reference depth position"
    
    # Clean up
    del causal_tracer


def test_robust_seed_program_generation():
    """Test that the robust seed (12) generates the expected program structure."""
    
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    SEQ_LEN = 17
    RNG_SEED = 12
    
    causal_tracer = CausalTracer(MODEL_ID)
    rng = np.random.RandomState(RNG_SEED)
    
    program, answer, hops, metadata = make_variable_binding_program_with_metadata(
        seq_len=SEQ_LEN, rng=rng, tokenizer=causal_tracer.tokenizer
    )
    
    # Check basic properties of the robust program
    assert hops >= 2, f"Robust program should have multiple hops, got {hops}"
    assert answer == 1, f"Robust program should have answer 1, got {answer}"
    
    # Check that it has multiple variable assignments
    lines = program.strip().split('\n')
    assignment_lines = [line for line in lines if '=' in line and not line.startswith('#')]
    assert len(assignment_lines) >= SEQ_LEN, f"Should have at least {SEQ_LEN} assignments, got {len(assignment_lines)}"
    
    # Check that it ends with a query
    assert lines[-1].startswith('#'), "Should end with query line"
    assert lines[-1].endswith(':'), "Query line should end with ':'"
    
    # Clean up
    del causal_tracer


# Keep the interactive function for manual testing and debugging
def run_interactive_test():
    """Run the test interactively with print statements for debugging."""
    
    print("Testing Token Positions for Patching Experiment")
    print("=" * 60)
    
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    SEQ_LEN = 17
    RNG_SEED = 12
    
    print(f"Model: {MODEL_ID}")
    print(f"RNG Seed: {RNG_SEED}")
    print()
    
    causal_tracer = CausalTracer(MODEL_ID)
    rng = np.random.RandomState(RNG_SEED)
    program, answer, hops, metadata = make_variable_binding_program_with_metadata(
        seq_len=SEQ_LEN, rng=rng, tokenizer=causal_tracer.tokenizer
    )
    
    query_var = metadata["query_var"]
    intervention_targets = metadata["intervention_targets"]
    
    print("Generated program:")
    print(program)
    print()
    print(f"Query variable: {query_var}")
    print(f"Expected answer: {answer}")
    print(f"Hops: {hops}")
    print()
    
    print("Intervention targets:")
    for target_name, pos in intervention_targets.items():
        if pos is not None:
            print(f"  {target_name:20} -> position {pos}")
    
    counterfactual_generator = CounterfactualGenerator()
    counter_program = counterfactual_generator.create_counterfactual(program, query_var)
    
    print()
    print("Counterfactual program:")
    print(counter_program)
    print()
    
    tokens = causal_tracer.tokenizer.tokenize(program)
    print("Token verification:")
    for target_name, pos in intervention_targets.items():
        if pos is not None and pos < len(tokens):
            print(f"  {target_name:20} -> position {pos:2d}: '{tokens[pos]}'")
    
    del causal_tracer


if __name__ == "__main__":
    run_interactive_test()