"""Program generators for debug experiments."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from .token_analyzer import TokenAnalyzer, VariableChain
from transformers import AutoTokenizer

def make_variable_binding_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int, int]:
    """
    Generate variable binding programs following the paper specification.
    
    Creates seq_len assignment lines + 1 query line with referential depths 1-4.
    Uses cubic weighting for chain extension and rejection sampling for balance.
    
    Returns:
        program: The program string
        answer: The correct answer
        query_hops: Number of hops in the query chain
    """
    variables = 'abcdefghijklmnopqrstuvwxyz'
    constants = '0123456789'
    
    # Track variable definitions: var -> (reference, depth)
    defined_vars = {}
    assignments = []
    
    for _ in range(seq_len):
        # Choose LHS variable (prefer undefined)
        available = [v for v in variables if v not in defined_vars]
        lhs = rng.choice(available if available else list(variables))
        
        # Choose RHS: 30% constant, 70% variable (if any exist)
        if rng.random() < 0.3 or not defined_vars:
            rhs = rng.choice(list(constants))
            defined_vars[lhs] = (rhs, 1)
        else:
            # Cubic weighting for variable selection
            vars_list = list(defined_vars.keys())
            weights = np.array([defined_vars[v][1]**3 for v in vars_list], dtype=float)
            rhs = rng.choice(vars_list, p=weights/weights.sum())
            defined_vars[lhs] = (rhs, defined_vars[rhs][1] + 1)
        
        assignments.append(f"{lhs} = {rhs}")
    
    # Select query variable (depth 1-4, cubic weighted)
    valid_vars = [(v, d) for v, (_, d) in defined_vars.items() if 1 <= d <= 4]
    if not valid_vars:
        valid_vars = [(v, d) for v, (_, d) in defined_vars.items()]
    
    query_vars, depths = zip(*valid_vars)
    weights = np.array([d**3 for d in depths], dtype=float)
    query_var = rng.choice(query_vars, p=weights/weights.sum())
    query_hops = defined_vars[query_var][1]
    
    # Resolve final value with cycle detection
    def resolve(var, visited=None):
        if visited is None:
            visited = set()
        
        if var in visited:
            # Circular reference detected - return a default value
            return 0
        
        if var in constants:
            return int(var)
        
        if var not in defined_vars:
            # Variable not defined - return default value
            return 0
        
        visited.add(var)
        result = resolve(defined_vars[var][0], visited)
        visited.remove(var)
        return result
    
    program = "\n".join(assignments + [f"#{query_var}: "])
    return program, resolve(query_var), query_hops


def make_variable_binding_program_with_metadata(
    seq_len: int, 
    rng: np.random.RandomState,
    tokenizer: Optional[AutoTokenizer] = None
) -> Tuple[str, int, int, Dict[str, Any]]:
    """
    Enhanced variable binding generator that returns full metadata for causal tracing.
    
    Args:
        seq_len: Number of assignment lines to generate
        rng: Random state for reproducibility
        tokenizer: Optional tokenizer for token position analysis
        
    Returns:
        Tuple of (program, answer, query_hops, metadata_dict)
        
    The metadata dict contains:
        - variable_chain: VariableChain object with full chain analysis
        - query_var: The query variable name
        - assignments_dict: Dictionary of all variable assignments
        - intervention_targets: Token positions for causal interventions (if tokenizer provided)
        - referential_depth: Same as query_hops for convenience
    """
    # Generate the basic program using existing logic
    program, answer, query_hops = make_variable_binding_program(seq_len, rng)
    
    # Extract query variable from program
    lines = program.split('\n')
    query_line = [line for line in lines if line.startswith('#')][0]
    query_var = query_line.strip()[1:-1]  # Remove # and :
    
    # Create token analyzer and get variable chain
    analyzer = TokenAnalyzer(tokenizer)
    variable_chain = analyzer.identify_variable_chain(program, query_var)
    
    # Parse assignments for metadata
    assignments_dict = analyzer._parse_assignments(program)
    
    # Build metadata dictionary
    metadata = {
        "variable_chain": variable_chain,
        "query_var": query_var,
        "assignments_dict": assignments_dict,
        "referential_depth": query_hops,
        "seq_len": seq_len
    }
    
    # Add token positions if tokenizer is provided
    if tokenizer is not None:
        try:
            intervention_targets = analyzer.find_intervention_targets(program, query_var)
            metadata["intervention_targets"] = intervention_targets
        except Exception as e:
            # If tokenization fails, just skip token positions
            metadata["intervention_targets"] = {}
            metadata["tokenization_error"] = str(e)
    
    return program, answer, query_hops, metadata


def make_exception_program(
    seq_len: int, rng: np.random.RandomState
) -> Tuple[str, int]:
    """Generate programs that test exception handling over `seq_len` divisions.

    Each division may raise ZeroDivisionError; the except branch adds a fallback
    instead. Returns the program string and the correct final value of `result`.
    """
    numerators = [rng.randint(1, 21) for _ in range(seq_len)]
    denominators = []
    fallbacks = []
    for n in numerators:
        if rng.rand() < 0.3:  # 30 % chance to trigger an exception
            denominators.append(0)
            fallbacks.append(int(rng.randint(-5, 0)))
        else:
            factors = [d for d in range(1, 11) if n % d == 0]
            if factors:
                denominators.append(int(rng.choice(factors)))
            else:
                # Ensure we don't divide by zero accidentally
                denominators.append(int(rng.randint(1, 10)))
            fallbacks.append(int(rng.randint(-5, 0)))  # unused if no exception

    lines = ["result = 0"]
    total = 0
    for n, d, fb in zip(numerators, denominators, fallbacks):
        lines.append("try:")
        lines.append(f"    result += {n} // {d}")
        lines.append("except ZeroDivisionError:")
        lines.append(f"    result += {fb}")
        total += fb if d == 0 else n // d

    program = "\n".join(lines)
    return program, total



def make_range_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate simple range tracking programs.
    
    Creates: x = 0; for i in range(n): x += z
    """
    z = rng.randint(1, 10)
    program = f"""x = 0
for i in range({seq_len}):
    x += {z}"""
    return program, seq_len * z


def make_range_program_lines(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate explicit line-by-line programs.
    
    Creates: x = 0; x += z; x += z; ...
    """
    z = rng.randint(1, 10)
    lines = ["x = 0"]
    for _ in range(seq_len):
        lines.append(f"x += {z}")
    program = "\n".join(lines)
    return program, seq_len * z


def make_variable_increments(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate programs with different increments each step."""
    increments = [rng.randint(1, 5) for _ in range(seq_len)]
    
    lines = ["x = 0"]
    total = 0
    for i, inc in enumerate(increments):
        lines.append(f"x += {inc}  # step {i+1}")
        total += inc
    
    program = "\n".join(lines)
    return program, total


def make_arithmetic_sequence(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate arithmetic sequence tracking."""
    start = rng.randint(0, 5)
    step = rng.randint(1, 3)
    
    lines = [f"x = {start}"]
    current = start
    for i in range(seq_len):
        current += step
        lines.append(f"x += {step}  # step {i+1}")
    
    program = "\n".join(lines)
    return program, current


def make_counter_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate simple counter programs."""
    program = f"""count = 0
for i in range({seq_len}):
    count += 1"""
    return program, seq_len


def make_fibonacci_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """Generate Fibonacci-like sequence programs."""
    a, b = 1, 1
    lines = [f"a, b = {a}, {b}"]
    
    for i in range(seq_len):
        a, b = b, a + b
        lines.append(f"a, b = b, a + b  # step {i+1}")
    
    lines.append("result = a")
    program = "\n".join(lines)
    return program, a


# Refactored from existing sequence_generation.py
ARITHMETIC_GROUPS = {
    "add": ("+= v", lambda x, v: x + v),
    "sub": ("-= v", lambda x, v: x - v),
    "div": ("//= v", lambda x, v: x // v),
    "mod": ("%= v", lambda x, v: x % v),
    "abs": ("= abs(x - v)", lambda x, v: abs(x - v)),
}


def make_counterfactual_pair(seq_len: int, divergence_index: int, rng: np.random.RandomState = None) -> Tuple[str, str, List[int], List[int]]:
    """Generate counterfactual program pairs (refactored from existing code).
    
    Returns: (program_a, program_b, intermediates_a, intermediates_b)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    ops = list(ARITHMETIC_GROUPS.keys())
    
    # Initialize both programs
    program_a, program_b = ["x = 0"], ["x = 0"]
    intermediates_a, intermediates_b = [0], [0]
    x_val_a = x_val_b = 0

    for i in range(seq_len):
        if i < divergence_index:
            # Common prefix
            op = rng.choice(ops)
            v = rng.randint(1, 10)
            template, func = ARITHMETIC_GROUPS[op]
            expr = template.replace("v", str(v))
            
            program_a.append(f"x {expr}")
            program_b.append(f"x {expr}")
            
            x_val_a = func(x_val_a, v)
            x_val_b = x_val_a
            
            intermediates_a.append(x_val_a)
            intermediates_b.append(x_val_b)
            
        elif i == divergence_index:
            # Divergence point
            state = rng.get_state()
            
            # Branch A
            rng_a = np.random.RandomState()
            rng_a.set_state(state)
            op_a = rng_a.choice(ops)
            v_a = rng_a.randint(1, 10)
            template_a, func_a = ARITHMETIC_GROUPS[op_a]
            expr_a = template_a.replace("v", str(v_a))
            program_a.append(f"x {expr_a}")
            x_val_a = func_a(x_val_a, v_a)
            intermediates_a.append(x_val_a)
            
            # Branch B (ensure divergence)
            rng_b = np.random.RandomState()
            rng_b.set_state(state)
            op_b, v_b = op_a, v_a
            while op_b == op_a and v_b == v_a:
                op_b = rng_b.choice(ops)
                v_b = rng_b.randint(1, 10)
            
            template_b, func_b = ARITHMETIC_GROUPS[op_b]
            expr_b = template_b.replace("v", str(v_b))
            program_b.append(f"x {expr_b}")
            x_val_b = func_b(x_val_b, v_b)
            intermediates_b.append(x_val_b)
            
            rng = rng_a  # Continue with A's state
            
        else:
            # Mirrored suffix
            op = rng.choice(ops)
            v = rng.randint(1, 10)
            template, func = ARITHMETIC_GROUPS[op]
            expr = template.replace("v", str(v))
            
            program_a.append(f"x {expr}")
            x_val_a = func(x_val_a, v)
            intermediates_a.append(x_val_a)
            
            program_b.append(f"x {expr}")
            x_val_b = func(x_val_b, v)
            intermediates_b.append(x_val_b)

    prog_a = "\n".join(program_a)
    prog_b = "\n".join(program_b)
    
    return prog_a, prog_b, intermediates_a[1:], intermediates_b[1:]


def make_sequence(seq_len: int, num_range: Tuple[int, int] = (0, 10), rng: np.random.RandomState = None) -> Tuple[str, List[int]]:
    """Generate random addition/subtraction sequences (from legacy experiments).
    
    Creates: x = 0; x += n1; x -= n2; x += n3; ...
    Returns the program and list of intermediate x values.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    plus_minus = rng.randint(0, 2, seq_len)
    nums = rng.randint(*num_range, seq_len)

    expressions = ["x = 0"]
    intermediates = [0]  # Track all intermediate values
    x_val = 0
    for i in range(seq_len):
        op = '+' if plus_minus[i] else '-'
        expressions.append(f"x {op}= {nums[i]}")
        if plus_minus[i]:
            x_val += nums[i]
        else:
            x_val -= nums[i]
        intermediates.append(x_val)
    
    code = '\n'.join(expressions)
    return code, intermediates 


 