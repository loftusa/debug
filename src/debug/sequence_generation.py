#%%
import numpy as np
from typing import Tuple, List

GROUPS = {
    "add": ("+= v", lambda x, v: x + v),
    "sub": ("-= v", lambda x, v: x - v),
    "mul": ("*= v", lambda x, v: x * v),
    "div": ("//= v", lambda x, v: x // v),
    "mod": ("%= v", lambda x, v: x % v),
    "pow": ("**= v", lambda x, v: x**v),
    "abs": ("= int(np.abs(x - v))", lambda x, v: int(np.abs(x - v))),
    "sqrt": ("= int(np.sqrt(x))", lambda x, v: int(np.sqrt(x))),
}

OP_TO_GROUP = {
    "add": "additive",
    "sub": "additive",
    "idx+": "additive",
    "idx-": "additive",
    "mul": "multiplicative",
    "div": "multiplicative",
    "mod": "modular",
    "pow": "exponential",
    "abs": "absdiff",
    "sqrt": "sqrt"
} 


def make_sequence(
    n: int, num_range: Tuple[int, int] = (0, 10), window_distinct: int = 2
) -> Tuple[str, List[int]]:
    """
    - Ensures no two adjacent (or within window_distinct) ops are from the same group.
    - Uses only primitives that can't be collapsed across lines.
    """
    min_v, max_v = max(1, num_range[0]), max(1, num_range[1])

    # Pre‐sample random v's and i's
    vs = np.random.randint(min_v, max_v, size=n)
    ops = list(GROUPS.keys())
    sequence_groups: List[str] = []
    expressions = ["x = 0"]
    intermediate: List[int] = []
    x_val = 0

    for i in range(n):
        # pick a group not in the last `window_distinct-1` picks
        forbidden_groups = set(sequence_groups[-(window_distinct - 1) :])
        
        # Define potential candidates based on all ops
        current_ops = list(ops) # Start with all operations
        if x_val < 0:
            # Exclude sqrt if x is negative
            if "sqrt" in current_ops:
                current_ops.remove("sqrt")
        
        # Filter candidates based on forbidden groups
        candidates = [op for op in current_ops if OP_TO_GROUP[op] not in forbidden_groups]
        
        # If filtering removed all candidates (unlikely but possible), fallback to any op not in forbidden groups
        if not candidates:
            candidates = [op for op in ops if OP_TO_GROUP[op] not in forbidden_groups]
            # If still no candidates (e.g., window_distinct > num_groups), fallback to any op
            if not candidates:
                candidates = list(ops)

        grp = np.random.choice(candidates)
        group_name = OP_TO_GROUP[grp]  # e.g. "additive" or "multiplicative"
        template, func = GROUPS[grp]
        v = int(vs[i])
        if grp == "pow":  # prevent numbers from getting absurdly large
            v = 2
        if grp == "mul":
            v = int(v // np.sqrt(v)) if v > 1 else 1

        # fill in template
        expr = template.replace("v", str(v))
        x_val = func(x_val, v)

        expressions.append(f"x {expr}" if not expr.startswith("=") else f"x {expr}")
        sequence_groups.append(group_name)
        intermediate.append(x_val)

    return "\n".join(expressions), intermediate

if __name__ == "__main__":
    expressions, intermediate = make_sequence(10)
    print(expressions, '\n')
    print(intermediate)
