#!/usr/bin/env python3
"""
Test the fixed TokenAnalyzer with the actual implementation.
"""

import sys
from pathlib import Path

# Add src to PYTHONPATH 
project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(project_root))

from debug.token_analyzer import TokenAnalyzer
from transformers import AutoTokenizer


def test_fixed_analyzer_referential_depth_mapping():
    """Test that the referential depth mapping fix is working correctly."""
    
    # The problematic program from the issue
    program = """l = 1
c = l
y = 5
p = 6
m = 8
q = p
f = m
a = c
j = 9
v = 0
x = f
o = q
r = a
w = 5
g = r
b = r
i = r
#a:"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Create analyzer
    analyzer = TokenAnalyzer(tokenizer)
    
    # Get intervention targets
    targets = analyzer.find_intervention_targets(program, "a")
    
    # Tokenize for verification
    tokens = tokenizer.tokenize(program)
    
    # Verify correct referential depth mapping
    assert "ref_depth_1_rhs" in targets, "ref_depth_1_rhs should be found"
    assert "ref_depth_2_rhs" in targets, "ref_depth_2_rhs should be found"
    assert "ref_depth_3_rhs" in targets, "ref_depth_3_rhs should be found"
    
    pos1 = targets["ref_depth_1_rhs"]
    pos2 = targets["ref_depth_2_rhs"] 
    pos3 = targets["ref_depth_3_rhs"]
    
    # Check tokens at positions
    token1 = tokens[pos1].replace('Ġ', '').strip()
    token2 = tokens[pos2].replace('Ġ', '').strip()
    token3 = tokens[pos3].replace('Ġ', '').strip()
    
    # Verify correct mapping: ref_depth_1_rhs = root value, etc.
    assert token1 == "1", f"ref_depth_1_rhs should map to root value '1', got '{token1}'"
    assert token2 == "l", f"ref_depth_2_rhs should map to first hop 'l', got '{token2}'"
    assert token3 == "c", f"ref_depth_3_rhs should map to second hop 'c', got '{token3}'"
    
    # Also verify positions are as expected
    assert pos1 == 3, f"ref_depth_1_rhs should be at position 3, got {pos1}"
    assert pos2 == 7, f"ref_depth_2_rhs should be at position 7, got {pos2}"
    assert pos3 == 34, f"ref_depth_3_rhs should be at position 34, got {pos3}"


def test_variable_chain_identification():
    """Test that variable chain identification works correctly."""
    
    program = "x = 1\ny = x\nz = y\n#z:"
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    analyzer = TokenAnalyzer(tokenizer)
    
    chain = analyzer.identify_variable_chain(program, "z")
    
    assert chain.query_var == "z"
    assert chain.root_value == "1"
    assert chain.referential_depth == 3
    assert not chain.is_circular
    
    # Check the chain structure: z -> y -> x -> 1
    expected_chain = [("z", "y"), ("y", "x"), ("x", "1")]
    assert chain.chain == expected_chain


# Keep the interactive function for manual testing
def run_interactive_test():
    """Run the test interactively with print statements."""
    program = """l = 1
c = l
y = 5
p = 6
m = 8
q = p
f = m
a = c
j = 9
v = 0
x = f
o = q
r = a
w = 5
g = r
b = r
i = r
#a:"""
    
    print("Testing Fixed TokenAnalyzer")
    print("=" * 50)
    print("Program:")
    print(program)
    print()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    analyzer = TokenAnalyzer(tokenizer)
    targets = analyzer.find_intervention_targets(program, "a")
    tokens = tokenizer.tokenize(program)
    
    print(f"Intervention targets: {targets}")
    print()
    print("Token verification:")
    for target_name, pos in targets.items():
        if pos is not None:
            print(f"{target_name:20} -> position {pos:2d}: '{tokens[pos]}'")
    
    print()
    print("Expected correct positions:")
    print("ref_depth_1_rhs      -> position  3: '1'   (root value in 'l = 1')")
    print("ref_depth_2_rhs      -> position  7: 'Ġl'  (first hop in 'c = l')")  
    print("ref_depth_3_rhs      -> position 34: 'Ġc'  (second hop in 'a = c')")


if __name__ == "__main__":
    run_interactive_test()