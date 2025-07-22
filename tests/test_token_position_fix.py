#!/usr/bin/env python3
"""
Test script to verify the corrected _find_token_position method.
"""

import re
from typing import Dict, List, Optional
from transformers import AutoTokenizer


class TokenAnalyzer:
    """Fixed version of TokenAnalyzer with corrected _find_token_position method."""
    
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer

    def _parse_assignments(self, program: str) -> Dict[str, str]:
        """Parse variable assignments from program text."""
        assignments = {}
        assignment_regex = re.compile(r"^\s*([a-zA-Z_]\w*)\s*=\s*(.*?)(\s*#.*)?$")
        
        for line in program.split('\n'):
            match = assignment_regex.match(line)
            if match:
                var, value, _ = match.groups()
                assignments[var] = value.strip()
        
        return assignments

    def _find_token_position_fixed(self, tokens: List[str], target_token: str, 
                                 lhs_var: str, assignments: Dict[str, str]) -> Optional[int]:
        """
        Find the position of target_token on the RHS of the assignment lhs_var = target_token.
        
        This is the corrected version that looks for the specific assignment context.
        """
        if lhs_var not in assignments:
            return None
        
        expected_rhs = assignments[lhs_var]
        if expected_rhs != target_token:
            return None
        
        # Now we need to find the assignment line "lhs_var = target_token" in the tokens
        # and return the position of target_token
        
        i = 0
        while i < len(tokens):
            # Look for lhs_var
            clean_token = tokens[i].replace('Ġ', '').strip()
            if clean_token == lhs_var:
                # Found lhs_var, now look for = and then target_token
                j = i + 1
                # Skip to find =
                while j < len(tokens) and '=' not in tokens[j].replace('Ġ', '').strip():
                    j += 1
                
                if j >= len(tokens):
                    i += 1
                    continue
                    
                # Found =, now look for target_token
                k = j + 1
                while k < len(tokens):
                    target_clean = tokens[k].replace('Ġ', '').strip()
                    if target_clean == target_token:
                        return k
                    # Stop looking if we hit a newline or start of next assignment
                    if '\n' in tokens[k] or (k + 1 < len(tokens) and 
                                           any(c.isalpha() for c in tokens[k + 1].replace('Ġ', '').strip()) and
                                           '=' in tokens[k + 2:k + 5] if k + 2 < len(tokens) else False):
                        break
                    k += 1
            i += 1
        
        return None

    def identify_variable_chain(self, program: str, query_var: str):
        """Simplified version for testing - trace variable chain."""
        assignments = self._parse_assignments(program)
        
        chain = []
        current_var = query_var
        visited = set()
        
        while current_var in assignments:
            if current_var in visited:
                break
            
            visited.add(current_var)
            refers_to = assignments[current_var]
            chain.append((current_var, refers_to))
            
            # Check if we've reached a literal value
            try:
                int(refers_to)
                break
            except ValueError:
                pass
                
            current_var = refers_to
        
        return chain


def test_token_position_fix():
    """Test the corrected _find_token_position method."""
    
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
    
    print("Program:")
    print(program)
    print()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Create analyzer
    analyzer = TokenAnalyzer(tokenizer)
    
    # Get variable chain for "a"
    chain = analyzer.identify_variable_chain(program, "a")
    print(f"Variable chain for 'a': {chain}")
    
    # Expected chain: a -> c -> l -> 1
    # So we want:
    # ref_depth_1_rhs: "1" in "l = 1" 
    # ref_depth_2_rhs: "l" in "c = l"
    # ref_depth_3_rhs: "c" in "a = c"
    
    # Tokenize program
    tokens = tokenizer.tokenize(program)
    print(f"\nTokens: {tokens}")
    print()
    
    # Get assignments
    assignments = analyzer._parse_assignments(program)
    print(f"Assignments: {assignments}")
    print()
    
    # Test the corrected method for each level
    for i, (var, refers_to) in enumerate(chain):
        depth = i + 1
        position = analyzer._find_token_position_fixed(tokens, refers_to, var, assignments)
        print(f"ref_depth_{depth}_rhs: Looking for '{refers_to}' in '{var} = {refers_to}' -> position {position}")
        if position is not None:
            print(f"  Token at position {position}: '{tokens[position]}'")
        print()

if __name__ == "__main__":
    test_token_position_fix()