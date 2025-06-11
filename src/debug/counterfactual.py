"""Counterfactual program generation for causal tracing experiments."""

import random
from typing import Optional, List
from dataclasses import dataclass
from .token_analyzer import TokenAnalyzer, VariableChain


@dataclass
class CounterfactualResult:
    """Result of counterfactual program generation."""
    original_program: str
    counterfactual_program: str
    original_root_value: str
    counterfactual_root_value: str
    query_var: str
    chain_length: int
    variable_chain: VariableChain


class CounterfactualGenerator:
    """Generates counterfactual programs for causal tracing experiments."""
    
    def __init__(self, token_analyzer: Optional[TokenAnalyzer] = None):
        self.token_analyzer = token_analyzer or TokenAnalyzer()
    
    def create_counterfactual(self, 
                            original_program: str, 
                            query_var: str, 
                            new_root_value: Optional[str] = None) -> str:
        """
        Create a counterfactual version of the program.
        
        Args:
            original_program: The original program text
            query_var: Variable to trace (e.g., "c" from "#c:")
            new_root_value: New value for the root variable (auto-generated if None)
            
        Returns:
            Counterfactual program with modified root value
            
        Raises:
            ValueError: If circular reference detected or other issues
        """
        # Identify the variable chain
        chain = self.token_analyzer.identify_variable_chain(original_program, query_var)
        
        if chain.is_circular:
            raise ValueError(f"Cannot create counterfactual: circular reference detected in variable chain for '{query_var}'")
        
        if chain.root_value is None:
            raise ValueError(f"Cannot create counterfactual: no root value found for variable chain starting with '{query_var}'")
        
        # Generate new root value if not provided
        if new_root_value is None:
            new_root_value = self._generate_different_value(chain.root_value)
        
        # Create counterfactual by replacing root assignment
        return self._replace_root_assignment(original_program, chain, new_root_value)
    
    def create_counterfactual_with_metadata(self,
                                          original_program: str,
                                          query_var: str,
                                          new_root_value: Optional[str] = None) -> CounterfactualResult:
        """
        Create counterfactual program with full metadata.
        
        Args:
            original_program: The original program text
            query_var: Variable to trace
            new_root_value: New value for the root variable
            
        Returns:
            CounterfactualResult with all metadata
        """
        # Get variable chain info
        chain = self.token_analyzer.identify_variable_chain(original_program, query_var)
        
        if chain.is_circular:
            raise ValueError(f"Cannot create counterfactual: circular reference detected in variable chain for '{query_var}'")
        
        if chain.root_value is None:
            raise ValueError(f"Cannot create counterfactual: no root value found for variable chain starting with '{query_var}'")
        
        # Generate new root value if not provided
        if new_root_value is None:
            new_root_value = self._generate_different_value(chain.root_value)
        
        # Create counterfactual program
        counterfactual_program = self._replace_root_assignment(original_program, chain, new_root_value)
        
        return CounterfactualResult(
            original_program=original_program,
            counterfactual_program=counterfactual_program,
            original_root_value=chain.root_value,
            counterfactual_root_value=new_root_value,
            query_var=query_var,
            chain_length=chain.referential_depth,
            variable_chain=chain
        )
    
    def _generate_different_value(self, original_value: str) -> str:
        """Generate a different numerical value from the original."""
        try:
            original_int = int(original_value)
            # Generate a different value in a reasonable range
            candidates = [i for i in range(0, 10) if i != original_int]
            if not candidates:
                # Fallback if original is outside 0-9
                return str(original_int + 1)
            return str(random.choice(candidates))
        except ValueError:
            # If original isn't an integer, just use a default
            return "7"
    
    def _replace_root_assignment(self, 
                               program: str, 
                               chain: VariableChain, 
                               new_root_value: str) -> str:
        """Replace the root variable assignment with new value."""
        lines = program.split('\n')
        
        # Find the root variable (last item in chain)
        if not chain.chain:
            raise ValueError("Empty variable chain")
        
        root_var = chain.chain[-1][0]  # The variable that gets assigned the literal
        
        # Find and replace the root assignment
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(f"{root_var} ="):
                # Replace the assignment
                lines[i] = f"{root_var} = {new_root_value}"
                break
        else:
            raise ValueError(f"Could not find assignment for root variable '{root_var}'")
        
        return '\n'.join(lines)


def create_counterfactual_pair(original_program: str, query_var: str, 
                             new_root_value: Optional[str] = None) -> tuple[str, str]:
    """
    Convenience function to create an original/counterfactual pair.
    
    Args:
        original_program: The original program
        query_var: Variable to trace
        new_root_value: New root value (auto-generated if None)
        
    Returns:
        Tuple of (original_program, counterfactual_program)
    """
    generator = CounterfactualGenerator()
    counterfactual = generator.create_counterfactual(original_program, query_var, new_root_value)
    return original_program, counterfactual