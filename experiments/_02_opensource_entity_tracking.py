#%%
from ast import Import
import re
from pathlib import Path
import struct
from typing import List, Tuple, Optional, Dict

import click
from transformers import pipeline, AutoTokenizer
from debug.sequence_generation import make_counterfactual_pair
import csv
import json
import gc
import torch

print('imports loaded')


PROMPT_TEMPLATE = (
    "You are given a short Python program. "
    "Your task is to compute the final value of the variable x. "
    "Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.\n" 
    "```python\n{code}\n```\n"
    "The final value of x is: "
)


DEFAULT_MODELS: List[str] = [
    # "google/gemma-3-4b-it",
    # "google/gemma-3-1b-it",
    # "google/gemma-3-12b-it",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    # "openai-community/gpt2",
    # "deepseek-ai/deepseek-coder-1.3b-instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # "tiiuae/Falcon3-7B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    # "NTQAI/Nxcode-CQ-7B-orpo",
    # "Qwen/CodeQwen1.5-7B-Chat",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "google/gemma-2-9b-it",
    # "google/gemma-3-27b-it",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "Qwen/Qwen2.5-14B",
    # "allenai/OLMo-2-1124-7B-Instruct",
    # "allenai/OLMo-2-1124-13B-Instruct",
    # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
]


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



## DEBUG
# model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
# tok = AutoTokenizer.from_pretrained(model_id)
# llm = pipeline("text-generation", model=model_id, tokenizer=tok, temperature=0.8, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
#%%
import numpy as np
# --- Start of interactive test cell ---
# Configuration for the test
test_seq_len = 3
test_divergence_index = 2
best_of_k_test = 1 # Number of samples for _best_of_k

# Generate a single code sequence and its true value

GROUPS = {
    "add": ("+= v", lambda x, v: x + v),
    "sub": ("-= v", lambda x, v: x - v),
    # "mul": ("*= v", lambda x, v: x * v),
    # "div": ("//= v", lambda x, v: x // v),
    # "mod": ("%= v", lambda x, v: x % v),
    # "pow": ("**= v", lambda x, v: x**v),
    # "abs": ("= abs(x - v)", lambda x, v: abs(x - v)),
}

OP_TO_GROUP = {
    "add": "additive",
    "sub": "additive",
    # "mul": "multiplicative",
    # "div": "multiplicative",
    # "mod": "modular",
    # "pow": "exponential",
    # "abs": "absdiff",
}

def _best_of_k(outputs: List[Dict[str, str]], true_val: int) -> Optional[int]:
    """Return the integer prediction closest to *true_val* among *outputs*."""
    preds = [_parse_int(o["generated_text"]) for o in outputs]
    # Filter Nones
    preds = [p for p in preds if p is not None]
    if not preds:
        return None
    # If any exactly equals, prefer that
    for p in preds:
        if p == true_val:
            return p

    # Otherwise, return closest value to true
    return min(preds, key=lambda p: abs(p - true_val))

def make_counterfactual_pair(
    seq_len: int, divergence_index: int
) -> Tuple[str, str, List[int], List[int]]:
    """
    Produce two programs of identical token length where they diverge at an early step `k` but both sequences are still solved correctly by the chosen model.
    Returns program_a, program_b, intermediates_a, intermediates_b. The intermediates lists contain the value of x *after* each operation line.
    """
    # Initialize RNG for reproducibility
    seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.RandomState(seed)
    ops = list(GROUPS.keys())
    max_v = 5

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
            v = rng.randint(1, max_v)
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
            v_a = rng_a.randint(1, max_v)
            template_a, func_a = GROUPS[op_a]
            expr_a = template_a.replace("v", str(v_a))
            program_a.append(f"x {expr_a}")
            # Update x_val_a based on its value *before* this step
            x_val_a = func_a(x_val_a, v_a)
            intermediates_a.append(x_val_a)  # Append result of divergence step A

            # --- Branch B divergence ---
            rng_b = np.random.RandomState()
            rng_b.set_state(state)
            # Ensure true divergence (op or v must differ)
            op_b, v_b = op_a, v_a
            while op_b == op_a and v_b == v_a:
                op_b = rng_b.choice(ops)
                # If op changed, v can be anything
                if op_b != op_a:
                    v_b = rng_b.randint(1, max_v)
                # If op is same, v must differ
                else:
                    v_b = rng_b.randint(1, max_v)
                    while v_b == v_a:  # Ensure v differs if op is same
                        v_b = rng_b.randint(1, 10)

            template_b, func_b = GROUPS[op_b]
            expr_b = template_b.replace("v", str(v_b))
            program_b.append(f"x {expr_b}")
            # Update x_val_b based on its value *before* this step
            x_val_b = func_b(x_val_b, v_b)
            intermediates_b.append(x_val_b)  # Append result of divergence step B

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
            intermediates_a.append(x_val_a)  # Append result of suffix step A

            # Update program B
            program_b.append(f"x {expr}")
            # Update x_val_b based on its value *before* this step
            x_val_b = func(x_val_b, v)
            intermediates_b.append(x_val_b)  # Append result of suffix step B

    prog_a = "\n".join(program_a)
    prog_b = "\n".join(program_b)

    # Return intermediates *after* each operation (excluding the initial x=0 state)
    return prog_a, prog_b, intermediates_a[1:], intermediates_b[1:]

if __name__ == "__main__":
    num_test_iterations = 100
    correct_test_predictions = 0

    model_id = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = pipeline("text-generation", model=model_id, tokenizer=tok, temperature=0.8, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

    for iter_num in range(num_test_iterations):
        print(f"--- Iteration {iter_num + 1}/{num_test_iterations} ---")
        code, _, intermediate_vals, _ = make_counterfactual_pair(test_seq_len, divergence_index=test_divergence_index)
        if not intermediate_vals:
            print("Error: make_counterfactual_pair returned empty intermediate_vals. Skipping this iteration.")
            continue
        true_val = intermediate_vals[-1]

        # Create the prompt
        prompt = PROMPT_TEMPLATE.format(code=code)

        # Run the LLM
        # The pipeline expects a list of prompts, even for a single one.
        # The output will be a list containing one element: a list of generated sequences for that single prompt.
        pipeline_outputs_for_single_prompt = llm(
            [prompt],  # Pass prompt as a list
            num_return_sequences=best_of_k_test,
            max_new_tokens=10,
            do_sample=True, # Necessary if temperature > 0
            batch_size=1    # Processing a single prompt
        )
        
        # Extract the generated texts for the single prompt
        # pipeline_outputs_for_single_prompt is List[List[Dict[str, str]]]
        # We need the inner list for _best_of_k
        if not (pipeline_outputs_for_single_prompt and isinstance(pipeline_outputs_for_single_prompt, list) and len(pipeline_outputs_for_single_prompt) > 0):
            print("Error: LLM did not return expected output format. Skipping this iteration.")
            continue
            
        current_prompt_outputs = pipeline_outputs_for_single_prompt[0]
            
        # Get the best prediction
        pred_int = _best_of_k(current_prompt_outputs, true_val)

        is_correct = False
        if pred_int is not None:
            is_correct = (pred_int == true_val)
            if is_correct:
                correct_test_predictions += 1
        
        current_accuracy = (correct_test_predictions / (iter_num + 1)) * 100
        
        print(f"Correct: {is_correct}")
        print(f"True Value: {true_val}")
        print(f"Pred Value: {pred_int if pred_int is not None else 'N/A'}")
        print(f"Running Accuracy: {correct_test_predictions}/{iter_num + 1} ({current_accuracy:.2f}%)")
        print(f"Prompt (first 200 chars): {prompt[:200]}...") # Optional: uncomment to see prompts
        print(f"Raw Outputs: {current_prompt_outputs}") # Optional: uncomment for full raw output
        print(f"Generated Code:\\n{code}") # Optional: uncomment to see generated code
        print("\n")

    final_accuracy = (correct_test_predictions / num_test_iterations) * 100 if num_test_iterations > 0 else 0
    print(f"--- Final Test Accuracy ---")
    print(f"Total Correct: {correct_test_predictions}/{num_test_iterations}")
    print(f"Accuracy: {final_accuracy:.2f}%")

    with open("results/qwen2.5-coder-32b-instruct_test_results.json", "w") as f:
        json.dump({
            "model_id": model_id,
            "correct_test_predictions": correct_test_predictions,
            "num_test_iterations": num_test_iterations,
            "final_accuracy": final_accuracy
        }, f, indent=2)

# --- End of interactive test cell ---
#%%

@click.command()
@click.option("--num-seqs", "num_seqs", default=124, help="Number of sequences per model.")
@click.option("--seq_len", default=10, help="Maximum length (steps) of each sequence.")
@click.option("--best-of", "best_of", default=10, help="Number of parallel samples per prompt.")
@click.option("--kind", "kind", default='groups', help="Kind of sequence to generate.")
def main(num_seqs: int, seq_len: int, best_of: int, kind: str) -> None:  # noqa: D401
    """Evaluate multiple open‑source models on the variable‑tracking task."""
    results: List[Tuple[str, float]] = []
    results_dir = Path("results") / f"seq_len_{seq_len}{"_ops" if kind=='ops' else ""}"
    csv_path = results_dir / "results_summary.csv"
    if kind == 'ops':
        csv_path = results_dir / "results_summary_ops.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read existing results to skip already evaluated models
    processed_models = set()
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    processed_models.add(row[0])
        csv_mode = "a"
    else:
        csv_mode = "w"
    # Initialize summary CSV with header if creating new file
    with csv_path.open(csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["model_id", "seq_len", "accuracy"])
    for model_id in DEFAULT_MODELS:
        if model_id in processed_models:
            print(f"Skipping model {model_id} as it's already evaluated.")
            continue
        skip_model = False
        all_data = []
        print(f"\n--- Processing Model: {model_id}, Sequence Length: {seq_len}, Kind: {kind} ---")
        print(f"\nLoading model: {model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            llm = pipeline("text-generation", model=model_id, tokenizer=tok, temperature=0.8, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
            print(f"Model {model_id} loaded with torch_dtype: {llm.model.dtype}")
        except (RuntimeError, MemoryError) as e:
            print(f"Skipping model {model_id} due to load memory error: {e}")
            continue
        correct = 0

        prompts_batch: List[str] = []
        true_vals_batch: List[int] = []
        codes_batch: List[str] = []
        intermediates_batch: List[List[int]] = []

        for i_seq in range(num_seqs):
            # diverge halfway through the sequence
            code, _, intermediate_vals, _ = make_counterfactual_pair(seq_len, divergence_index=seq_len//2)
            print(f"  [Seq {i_seq+1}/{num_seqs}] Generated code: {code}")
            print(f"  [Seq {i_seq+1}/{num_seqs}] Intermediate values: {intermediate_vals}")
            if not intermediate_vals:
                print(f"  WARNING: make_counterfactual_pair returned empty intermediate_vals for seq {i_seq+1}, model {model_id}. Skipping this sample.")
                continue
            true_val = intermediate_vals[-1]
            print(f"  [Seq {i_seq+1}/{num_seqs}] True value: {true_val}")
            prompt = PROMPT_TEMPLATE.format(code=code)
            print(f"  [Seq {i_seq+1}/{num_seqs}] Prompt: {prompt[:200]}...") # Log first 200 chars of prompt

            prompts_batch.append(prompt)
            true_vals_batch.append(true_val)
            codes_batch.append(code)
            intermediates_batch.append(intermediate_vals)

        batched_pipeline_outputs: List[List[Dict[str, str]]] = []
        if prompts_batch:
            print(f"Generating {len(prompts_batch)} responses for model {model_id} with best_of={best_of} (batch_size=8)...")
            print(f"  LLM call params: num_return_sequences={best_of}, max_new_tokens=10, do_sample=True, temperature=0.8, batch_size=8")
            try:
                batched_pipeline_outputs = llm(
                    prompts_batch,
                    num_return_sequences=best_of, # Use the best_of parameter
                    max_new_tokens=10,           # Limit output length for efficiency
                    do_sample=True,              # Set do_sample=True if using temperature > 0
                    batch_size=8                 # Explicitly set batch_size to 8
                )
            except (RuntimeError, MemoryError) as e:
                print(f"Memory error during batched inference for {model_id}: {e}. Skipping model.")
                skip_model = True
        
        if not skip_model and batched_pipeline_outputs:
            if len(batched_pipeline_outputs) != len(prompts_batch):
                print(f"Warning: Number of outputs ({len(batched_pipeline_outputs)}) does not match number of prompts ({len(prompts_batch)}) for model {model_id}. Skipping result processing.")
                skip_model = True # Treat as an error for this model
            else:
                print(f"Processing {len(batched_pipeline_outputs)} batched responses for model {model_id}...")
                for i in range(len(batched_pipeline_outputs)):
                    current_prompt_outputs: List[Dict[str, str]] = batched_pipeline_outputs[i]
                    
                    true_val = true_vals_batch[i]
                    code = codes_batch[i]
                    intermediate_list = intermediates_batch[i]

                    pred_int = _best_of_k(current_prompt_outputs, true_val)
                    
                    all_data.append({
                        "code": code,
                        "intermediate": intermediate_list,
                        "true_val": true_val,
                        "pred_int": pred_int,
                        "outputs": current_prompt_outputs,
                        "correct": pred_int == true_val,
                    })
                    if pred_int == true_val:
                        correct += 1
        
        if skip_model:
            # If llm was loaded but an error occurred during its use, ensure cleanup.
            if 'llm' in locals() and llm is not None:
                print(f"Cleaning up model {model_id} due to skip.")
                del llm
                # Force garbage collection and cache clearing before next model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        acc = correct / num_seqs if num_seqs > 0 else 0.0
        results.append((model_id, acc))
        print(f"Model {model_id} accuracy: {acc:.2%} ({correct}/{num_seqs})")
        # Save detailed data for this model
        detailed_path = results_dir / f"{model_id.replace('/', '_')}_data.json"
        with detailed_path.open("w") as f:
            json.dump(all_data, f, indent=2)
        # Append this model's accuracy to the summary CSV
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_id, seq_len, f"{acc:.6f}"])
            print(f"Model {model_id} appended to results summary. Accuracy: {acc:.2%} ({correct}/{num_seqs})")
        
        # Clean up the current model to free memory
        if 'llm' in locals() and llm is not None:
            del llm
            llm = None # Ensure it's seen as gone
        # Free up memory between models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



# if __name__ == "__main__":
#     main()
