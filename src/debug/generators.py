"""Program generators for debug experiments."""

from typing import Tuple, List
import numpy as np

def make_variable_binding_program(seq_len: int, rng: np.random.RandomState) -> Tuple[str, int]:
    """
    Generate variable binding programs following the paper specification.
    
    Creates seq_len assignment lines + 1 query line with referential depths 1-4.
    Uses cubic weighting for chain extension and rejection sampling for balance.
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
    
    # Resolve final value
    def resolve(var):
        return int(var) if var in constants else resolve(defined_vars[var][0])
    
    program = "\n".join(assignments + [f"#{query_var}:"])
    return program, resolve(query_var)


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
            denominators.append(int(rng.choice(factors)) if factors else int(rng.randint(1, 11)))
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


