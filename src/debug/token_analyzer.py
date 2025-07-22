"""Token analysis for causal tracing experiments."""

import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class VariableChain:
    """Represents a variable reference chain."""
    query_var: str
    chain: List[Tuple[str, str]]  # [(var, refers_to), ...]
    root_value: Optional[str]
    referential_depth: int
    is_circular: bool = False


@dataclass
class InterventionTargets:
    """Token positions for causal interventions."""
    ref_depth_1_rhs: Optional[int] = None  # Root value position
    ref_depth_2_rhs: Optional[int] = None  # Second level RHS
    ref_depth_3_rhs: Optional[int] = None  # Third level RHS
    ref_depth_4_rhs: Optional[int] = None  # Fourth level RHS
    query_var: Optional[int] = None        # Query variable position
    prediction_token_pos: Optional[int] = None # Position of the final token before generation


class TokenAnalyzer:
    """Analyzes program structure and identifies token positions for causal tracing."""
    
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer
        
    def identify_variable_chain(self, program: str, query_var: str) -> VariableChain:
        """
        Trace variable chain from query variable back to root value.
        
        Args:
            program: The program text
            query_var: Variable to trace (e.g., "c" from "#c:")
            
        Returns:
            VariableChain with full trace information
        """
        # Parse assignments from program
        assignments = self._parse_assignments(program)
        
        # Trace the chain
        chain = []
        current_var = query_var
        visited = set()
        
        while current_var in assignments:
            if current_var in visited:
                # Circular reference detected
                return VariableChain(
                    query_var=query_var,
                    chain=chain,
                    root_value=None,
                    referential_depth=len(chain),
                    is_circular=True
                )
            
            visited.add(current_var)
            refers_to = assignments[current_var]
            chain.append((current_var, refers_to))
            
            # Check if we've reached a literal value
            if self._is_literal(refers_to):
                return VariableChain(
                    query_var=query_var,
                    chain=chain,
                    root_value=refers_to,
                    referential_depth=len(chain),
                    is_circular=False
                )
            
            current_var = refers_to
        
        # If we exit the loop, we didn't find a literal
        return VariableChain(
            query_var=query_var,
            chain=chain,
            root_value=None,
            referential_depth=len(chain),
            is_circular=False
        )
    
    def find_intervention_targets(self, program: str, query_var: str) -> Dict[str, int]:
        """
        Find token positions for causal interventions.
        
        Args:
            program: The program text
            query_var: Variable to trace
            
        Returns:
            Dictionary mapping target names to token positions
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for token position identification")
        
        # Get variable chain
        chain = self.identify_variable_chain(program, query_var)
        
        # Tokenize the program
        tokens = self.tokenizer.tokenize(program)
        token_positions = {}
        
        # Find RHS token positions based on referential depth
        assignments = self._parse_assignments(program)
        
        for i, (var, refers_to) in enumerate(chain.chain):
            depth = len(chain.chain) - i  # Reverse depth: 1=root, 2=first hop, etc.
            target_key = f"ref_depth_{depth}_rhs"
            
            # Find position of the RHS token in the assignment
            position = self._find_token_position(tokens, refers_to, var, assignments)
            if position is not None:
                token_positions[target_key] = position
        
        # Find query variable and the final prompt token
        query_positions = self._find_query_positions(tokens, query_var)
        token_positions.update(query_positions)
        
        return token_positions
    
    def _parse_assignments(self, program: str) -> Dict[str, str]:
        """Parse variable assignments from program text."""
        assignments = {}
        # Regex to find 'var = val' lines, ignoring comments
        assignment_regex = re.compile(r"^\s*([a-zA-Z_]\w*)\s*=\s*(.*?)(\s*#.*)?$")
        
        for line in program.split('\n'):
            match = assignment_regex.match(line)
            if match:
                var, value, _ = match.groups()
                assignments[var] = value.strip()
        
        return assignments
    
    def _is_literal(self, value: str) -> bool:
        """Check if a value is a literal (number) rather than a variable."""
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def _find_token_position(self, tokens: List[str], target_token: str, 
                           lhs_var: str, assignments: Dict[str, str]) -> Optional[int]:
        """
        Find the position of target_token on the RHS of the assignment lhs_var = target_token.
        
        Args:
            tokens: Tokenized program
            target_token: The token to find (RHS value)
            lhs_var: The LHS variable in the assignment
            assignments: Dict mapping variables to their assigned values
            
        Returns:
            Token position of target_token in the specific assignment, or None if not found
        """
        if lhs_var not in assignments:
            return None
        
        expected_rhs = assignments[lhs_var]
        if expected_rhs != target_token:
            return None
        
        # Find the assignment line "lhs_var = target_token" in the tokens
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
                    
                # Found =, now look for target_token on the same line
                k = j + 1
                while k < len(tokens):
                    target_clean = tokens[k].replace('Ġ', '').strip()
                    if target_clean == target_token:
                        return k
                    # Stop looking if we hit a newline (end of assignment line)
                    if '\n' in tokens[k] or 'Ċ' in tokens[k]:
                        break
                    k += 1
            i += 1
        
        return None
    
    def _find_query_positions(self, tokens: List[str], query_var: str) -> Dict[str, int]:
        """Find positions of query variable and the final prompt token."""
        positions = {}
        
        # The prediction token is the last token of the sequence before generation starts.
        # Given the program format, this will be the last token in the list.
        # We add a check to ensure the prompt is not empty.
        if tokens:
            positions["prediction_token_pos"] = len(tokens) - 1

        # For completeness, we still find the query variable itself.
        # This is useful for other types of interventions.
        query_pattern = f"#{query_var}"
        
        # Search backwards from the end, as the query is at the end of the prompt.
        for i in range(len(tokens) - 1, -1, -1):
            clean_token = tokens[i].replace('Ġ', '').strip()
            if clean_token == query_pattern:
                positions["query_var"] = i
                break
        
        return positions