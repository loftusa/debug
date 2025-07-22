# %%
import numpy as np
from debug.generators import make_variable_binding_program_with_metadata
from transformers import AutoTokenizer
RNG_SEED = 12
SEQ_LEN = 17

base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
rng = np.random.RandomState(RNG_SEED)
program, answer, hops, metadata = make_variable_binding_program_with_metadata(
    seq_len=SEQ_LEN, rng=rng, tokenizer=base_tokenizer
)

query_var = metadata["query_var"]

# Construct counterfactual by flipping root value
from debug.counterfactual import CounterfactualGenerator

counterfactual_generator = CounterfactualGenerator()
counter_program = counterfactual_generator.create_counterfactual(program, query_var)

print("Original program:\n", program)
print("\nCounterfactual program:\n", counter_program)