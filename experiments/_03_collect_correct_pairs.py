#%%

# What am I trying to do here?
# Generate a pair of sequences for which the model gets both correct (using rejection sampling)
# then, patch at every token and layer to check which ones increase probability of correct answer

#%%  # SETUP / HELPER FUNCTIONS
from typing import Optional
import re
from debug.generators import make_counterfactual_pair
from transformers import pipeline
import torch
from typing import List, Tuple

def _parse_int(text: str) -> Optional[int]:
    """Extract the integer following 'the final value of x is: ' (case-insensitive, commas ignored), or the first integer otherwise."""
    cleaned = text.replace(",", "")
    # Try to find integer after the explicit phrase
    phrase_match = re.search(r"is:\s*(-?\d+)", cleaned, flags=re.IGNORECASE)
    if phrase_match:
        return int(phrase_match.group(1))
    # Fallback: first integer anywhere
    match = re.search(r"-?\d+", cleaned)
    return int(match.group()) if match else None

# REJECTION SAMPLING
PROMPT_TEMPLATE = (
    "You are given a short Python program. "
    "Your task is to compute the final value of the variable x. "
    "Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.\n" 
    "```python\n{code}\n```\n"
    "The final value of x is: "
)

# rejection sample to get counterfactual pairs for which the model gets both 100% correct
correct_pairs: List[Tuple[str, str]] = []
TARGET_NUM_PAIRS = 10 # Define how many pairs we want
BATCH_SIZE = 8 # Define batch size for LLM calls


model_name = "google/gemma-3-1b-it" # Using a smaller, faster model for example
# Ensure do_sample=False for deterministic output if not using best_of_k logic here
# Added max_new_tokens for consistency and efficiency
llm = pipeline("text-generation", model=model_name, tokenizer=model_name, device_map="auto", torch_dtype=torch.bfloat16, do_sample=False, max_new_tokens=10)

while len(correct_pairs) < TARGET_NUM_PAIRS:
    prompts_batch: List[str] = []
    true_vals_batch: List[int] = []
    # Store original code pairs to add to correct_pairs if both are correct
    original_code_pairs_batch: List[Tuple[str, str]] = []

    print(f"Generating a new batch of {BATCH_SIZE} pairs...")
    for _ in range(BATCH_SIZE):
        # Ensure make_counterfactual_pair returns distinct intermediate_vals for code1 and code2
        code1, code2, intermediate_vals1, intermediate_vals2 = make_counterfactual_pair(seq_len=6, divergence_index=3)
        
        # Basic check for valid intermediate values
        if not intermediate_vals1 or not intermediate_vals2:
            print("Warning: make_counterfactual_pair returned empty intermediate_vals. Skipping this attempt.")
            continue

        true_val1 = intermediate_vals1[-1]
        true_val2 = intermediate_vals2[-1]
        
        prompts_batch.append(PROMPT_TEMPLATE.format(code=code1))
        prompts_batch.append(PROMPT_TEMPLATE.format(code=code2))
        
        true_vals_batch.append(true_val1)
        true_vals_batch.append(true_val2)
        
        original_code_pairs_batch.append((code1, code2))

    if not prompts_batch: # If all attempts in batch failed due to empty intermediate_vals
        continue

    print(f"Sending batch of {len(prompts_batch)} prompts to LLM...")
    # The pipeline handles batching automatically when given a list of prompts
    try:
        batch_results_raw = llm(prompts_batch)
    except Exception as e:
        print(f"Error during LLM batch inference: {e}. Skipping this batch.")
        continue
    
    # Ensure llm output is a list of lists of dicts, or a list of dicts
    # If it's a list of dicts (single output per prompt), wrap each in a list
    if batch_results_raw and isinstance(batch_results_raw[0], dict):
        batch_results = [[item] for item in batch_results_raw]
    else:
        batch_results = batch_results_raw


    if len(batch_results) != len(prompts_batch):
        print(f"Warning: Mismatch in number of results ({len(batch_results)}) vs prompts ({len(prompts_batch)}). Skipping batch.")
        continue
        
    print("Processing batch results...")
    # Process results in pairs
    for i in range(0, len(batch_results), 2):
        if len(correct_pairs) >= TARGET_NUM_PAIRS:
            break # Stop if we've collected enough pairs

        # Check if there are enough results left for a pair
        if i + 1 >= len(batch_results):
            print("Warning: Incomplete pair at the end of batch results. Skipping.")
            break

        result1_list = batch_results[i]
        result2_list = batch_results[i+1]

        # llm call returns a list of generated texts, typically one if num_return_sequences=1 (default)
        if not result1_list or not result2_list:
            print(f"Warning: Empty result for a prompt in pair {i//2 + 1} of batch. Skipping.")
            continue
            
        pred_text1 = result1_list[0]['generated_text']
        pred_text2 = result2_list[0]['generated_text']

        pred_val1 = _parse_int(pred_text1)
        pred_val2 = _parse_int(pred_text2)

        original_pair_idx = i // 2
        true_val1_original = true_vals_batch[i]
        true_val2_original = true_vals_batch[i+1]
        code_pair = original_code_pairs_batch[original_pair_idx]

        if pred_val1 == true_val1_original and pred_val2 == true_val2_original:
            correct_pairs.append(code_pair)
            print(f"Found pair {len(correct_pairs)}/{TARGET_NUM_PAIRS}: Correct! ({pred_val1}=={true_val1_original}, {pred_val2}=={true_val2_original})")
            # print(f"  Code 1: {code_pair[0].strip()}\\n  Code 2: {code_pair[1].strip()}")
        else:
            # Optional: print mismatches for debugging
            mismatch_info = []
            if pred_val1 != true_val1_original:
                mismatch_info.append(f"P1: {pred_val1} (pred) != {true_val1_original} (true)")
            if pred_val2 != true_val2_original:
                mismatch_info.append(f"P2: {pred_val2} (pred) != {true_val2_original} (true)")
            # print(f"Mismatch for pair {original_pair_idx + 1} in batch: {'; '.join(mismatch_info)}")
            pass # reduce verbosity

    print(f"Collected {len(correct_pairs)}/{TARGET_NUM_PAIRS} correct pairs so far.")

print(f"\nFinished. Collected {len(correct_pairs)} correct pairs.")
print("Correct pairs:")
for i, pair in enumerate(correct_pairs):
    print(f"Pair {i+1}:\\nCode 1: {pair[0]}\\nCode 2: {pair[1]}\\n")

# %% You can then save `correct_pairs` to a file, e.g., JSONL
import json
output_file = "correct_counterfactual_pairs.jsonl"
with open(output_file, 'w') as f:
    for code1, code2 in correct_pairs:
        # Storing more info might be useful, e.g., true values, divergence point if needed later
        f.write(json.dumps({"program_a": code1, "program_b": code2}) + "\n")
print(f"Saved {len(correct_pairs)} pairs to {output_file}")

#%%
from nnsight import LanguageModel

model = LanguageModel(model_name)





