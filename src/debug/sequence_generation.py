#%%
import numpy as np
from typing import Tuple, List

GROUPS = {
    "add": ("+= v", lambda x, v: x + v),
    "sub": ("-= v", lambda x, v: x - v),
    # "mul": ("*= v", lambda x, v: x * v),
    "div": ("//= v", lambda x, v: x // v),
    "mod": ("%= v", lambda x, v: x % v),
    # "pow": ("**= v", lambda x, v: x**v),
    "abs": ("= abs(x - v)", lambda x, v: abs(x - v)),
}

OP_TO_GROUP = {
    "add": "additive",
    "sub": "additive",
    # "mul": "multiplicative",
    "div": "multiplicative",
    "mod": "modular",
    # "pow": "exponential",
    "abs": "absdiff",
} 


def make_counterfactual_pair(seq_len: int, divergence_index: int) -> Tuple[str, str, List[int], List[int]]:
    """
    Produce two programs of identical token length where they diverge at an early step `k` but both sequences are still solved correctly by the chosen model.
    Returns program_a, program_b, intermediates_a, intermediates_b. The intermediates lists contain the value of x *after* each operation line.
    """
    # Initialize RNG for reproducibility
    seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.RandomState(seed)
    ops = list(GROUPS.keys())

    # Prepare program lines and intermediates for both branches
    program_a, program_b = ["x = 0"], ["x = 0"]
    # Start intermediates with the initial value 0, corresponding to "x = 0"
    intermediates_a, intermediates_b = [0], [0] 
    # Track current x values for each branch
    x_val_a = 0
    x_val_b = 0

    for i in range(seq_len):
        if i < divergence_index:
            # Common prefix
            op = rng.choice(ops)
            v = rng.randint(1, 10)
            template, func = GROUPS[op]
            expr = template.replace("v", str(v))
            
            # Update programs
            program_a.append(f"x {expr}")
            program_b.append(f"x {expr}")
            
            # Calculate next value based on current x_val
            x_val_a = func(x_val_a, v) 
            x_val_b = x_val_a  # Keep values synchronized during prefix
            
            # Append the *result* of this step
            intermediates_a.append(x_val_a)
            intermediates_b.append(x_val_b)
        elif i == divergence_index:
            # Save RNG state to mirror the rest of the sequence generation
            state = rng.get_state()
            
            # --- Branch A divergence ---
            rng_a = np.random.RandomState()
            rng_a.set_state(state)
            op_a = rng_a.choice(ops)
            v_a = rng_a.randint(1, 10)
            template_a, func_a = GROUPS[op_a]
            expr_a = template_a.replace("v", str(v_a))
            program_a.append(f"x {expr_a}")
            # Update x_val_a based on its value *before* this step
            x_val_a = func_a(x_val_a, v_a) 
            intermediates_a.append(x_val_a) # Append result of divergence step A
            
            # --- Branch B divergence ---
            rng_b = np.random.RandomState()
            rng_b.set_state(state)
            # Ensure true divergence (op or v must differ)
            op_b, v_b = op_a, v_a 
            while op_b == op_a and v_b == v_a:
                op_b = rng_b.choice(ops)
                # If op changed, v can be anything
                if op_b != op_a:
                    v_b = rng_b.randint(1, 10)
                # If op is same, v must differ
                else: 
                    v_b = rng_b.randint(1, 10)
                    while v_b == v_a: # Ensure v differs if op is same
                        v_b = rng_b.randint(1, 10)
            
            template_b, func_b = GROUPS[op_b]
            expr_b = template_b.replace("v", str(v_b))
            program_b.append(f"x {expr_b}")
            # Update x_val_b based on its value *before* this step
            x_val_b = func_b(x_val_b, v_b) 
            intermediates_b.append(x_val_b) # Append result of divergence step B
            
            # Continue sequence generation using branch A's RNG state for mirroring
            rng = rng_a
        else:
            # Mirrored suffix
            op = rng.choice(ops)
            v = rng.randint(1, 10)
            template, func = GROUPS[op]
            expr = template.replace("v", str(v))
            
            # Update program A
            program_a.append(f"x {expr}")
            # Update x_val_a based on its value *before* this step
            x_val_a = func(x_val_a, v) 
            intermediates_a.append(x_val_a) # Append result of suffix step A
            
            # Update program B
            program_b.append(f"x {expr}")
            # Update x_val_b based on its value *before* this step
            x_val_b = func(x_val_b, v) 
            intermediates_b.append(x_val_b) # Append result of suffix step B

    prog_a = "\n".join(program_a)
    prog_b = "\n".join(program_b)
    
    # Return intermediates *after* each operation (excluding the initial x=0 state)
    return prog_a, prog_b, intermediates_a[1:], intermediates_b[1:]


# def make_sequence(n: int, num_range: Tuple[int, int] = (0, 10), window_distinct: int = 2, kind='groups') -> Tuple[str, List[int]]:
#     if kind == 'groups':
#         return make_sequence_groups(n, num_range, window_distinct)
#     elif kind == 'ops':
#         return make_sequence_ops(n, num_range, window_distinct)
    
#     else: 
#         raise NotImplementedError(f"Kind {kind} not implemented")


# def make_sequence_ops(
#     n: int, num_range: Tuple[int, int] = (0, 10), window_distinct: int = 2
# ) -> Tuple[str, List[int]]:
#     """
#     Simple sequence of abs, add, and sub.
#     """
#     min_v, max_v = max(1, num_range[0]), max(1, num_range[1])
#     ops = ['abs', 'add', 'sub']
#     expressions = ["x = 0"]
#     intermediate: List[int] = []
#     x_val = 0
#     for i in range(n):
#         op = np.random.choice(ops)
#         v = np.random.randint(min_v, max_v)
#         if op == 'abs':
#             x_val = abs(x_val - v)
#         elif op == 'add':
#             x_val += v
#         elif op == 'sub':
#             x_val -= v
#         template, _ = GROUPS[op]
#         expr = template.replace("v", str(v))
#         expressions.append(f"x {expr}")
#         intermediate.append(x_val)
    
#     return "\n".join(expressions), intermediate


# def make_sequence_groups(
#     n: int, num_range: Tuple[int, int] = (0, 10), window_distinct: int = 2
# ) -> Tuple[str, List[int]]:
#     """
#     - Ensures no two adjacent (or within window_distinct) ops are from the same group.
#     - Uses only primitives that can't be collapsed across lines.

#     TODO: 
#       - could just be + and *, numbers from -5 to 5, randomly drawn, maybe include *0
#     """
#     min_v, max_v = max(1, num_range[0]), max(1, num_range[1])

#     # Pre‐sample random v's and i's
#     vs = np.random.randint(min_v, max_v, size=n)
#     ops = list(GROUPS.keys())
#     sequence_groups: List[str] = []
#     expressions = ["x = 0"]
#     intermediate: List[int] = []
#     x_val = 0

#     for i in range(n):
#         # pick a group not in the last `window_distinct-1` picks
#         forbidden_groups = set(sequence_groups[-(window_distinct - 1) :])
        
#         # Define potential candidates based on all ops
#         current_ops = list(ops) # Start with all operations
#         if x_val < 0:
#             # Exclude sqrt if x is negative
#             if "sqrt" in current_ops:
#                 current_ops.remove("sqrt")
        
#         # Filter candidates based on forbidden groups
#         candidates = [op for op in current_ops if OP_TO_GROUP[op] not in forbidden_groups]
        
#         # If filtering removed all candidates (unlikely but possible), fallback to any op not in forbidden groups
#         if not candidates:
#             candidates = [op for op in ops if OP_TO_GROUP[op] not in forbidden_groups]
#             # If still no candidates (e.g., window_distinct > num_groups), fallback to any op
#             if not candidates:
#                 candidates = list(ops)

#         grp = np.random.choice(candidates)
#         group_name = OP_TO_GROUP[grp]  # e.g. "additive" or "multiplicative"
#         template, func = GROUPS[grp]
#         v = int(vs[i])
#         if grp == "pow":  # prevent numbers from getting absurdly large
#             v = 2
#         if grp == "mul":
#             v = int(v // np.sqrt(v)) if v > 1 else 1

#         # fill in template
#         expr = template.replace("v", str(v))
#         x_val = func(x_val, v)

#         expressions.append(f"x {expr}" if not expr.startswith("=") else f"x {expr}")
#         sequence_groups.append(group_name)
#         intermediate.append(x_val)

#     return "\n".join(expressions), intermediate

if __name__ == "__main__":
    prog_a, prog_b, intermediates_a, intermediates_b = make_counterfactual_pair(6, 2)
    print(prog_a, '\n', intermediates_a)
    print(prog_b, '\n', intermediates_b)
