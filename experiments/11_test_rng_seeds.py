import gc
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel
import click

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1] / "src"
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from debug import prompts
from debug.generators import make_variable_binding_program_with_metadata
from debug.counterfactual import CounterfactualGenerator

# --- Configuration ---
MODEL_IDS = [
    # "Qwen/Qwen3-0.6B", 
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]
SEQ_LEN = 5
EXACT_HOPS = 2  # Exact number of hops in the variable binding chain
MAX_SEED_SEARCH = 30  # Maximum number of seeds to try
FIND_UNIVERSALLY_FAILED_PROGRAM = False  # Set to True to find a program all models fail on.


def run_inference_nnsight(model: LanguageModel, prompt: str, token_only=False) -> str:
    """
    Runs a single inference call using NNSight tracing.
    This does greedy decoding by taking argmax of the logits.
    """
    with model.trace(prompt):
        logits = model.lm_head.output.save()
    
    last_token_logits = logits[0, -1, :]
    predicted_token_id = last_token_logits.argmax().item()
    predicted_token = model.tokenizer.decode([predicted_token_id])
    if not token_only:
        predicted_token = prompt + predicted_token
    return predicted_token


def extract_answer(generated_text: str, prompt: str) -> str:
    """Extracts the answer from the generated text."""
    answer_part = generated_text[len(prompt) :]
    number_match = re.search(r'\b(\d+)\b(?=\n|$)', answer_part)
    if number_match:
        return number_match.group(1)
    
    return answer_part.strip()


def check_no_correct_answer_before_newline(generated_text: str, prompt: str, correct_answer: str) -> bool:
    """Checks that the correct answer doesn't appear before the first newline in generated text."""
    answer_part = generated_text[len(prompt) :]
    
    # Get text before first newline (or all text if no newline)
    first_line = answer_part.split('\n')[0]
    
    # Check if correct answer appears as a standalone number in first line
    pattern = r'\b' + re.escape(str(correct_answer)) + r'\b'
    return not bool(re.search(pattern, first_line))


def test_program_with_model(
    model: LanguageModel, program: str, true_answer: str, query_var: str, search_mode: str
) -> bool:
    """
    Tests a single program with a single model.
    Tests both original and counterfactual programs.

    Args:
        model: The loaded model to test with.
        program: The program string to test.
        true_answer: The expected answer for original program.
        query_var: The query variable (e.g., "d" from "#d:").
        search_mode: 'all_correct' or 'all_incorrect'.

    Returns:
        True if the condition is met for this model, False otherwise.
    """
    # Generate counterfactual program
    try:
        cf_generator = CounterfactualGenerator()
        cf_result = cf_generator.create_counterfactual_with_metadata(program, query_var)
        counterfactual_program = cf_result.counterfactual_program
        counterfactual_answer = cf_result.counterfactual_root_value
    except Exception as e:
        print(f"    -> Failed to generate counterfactual: {e}")
        return False

    # Test original program
    original_prompt = prompts.VARIABLE_BINDING.format(code=program)
    original_generated = run_inference_nnsight(model, original_prompt, token_only=True)
    original_correct = original_generated == str(true_answer)
    
    # For incorrect mode, also check that correct answer doesn't appear before first newline
    if search_mode == "all_incorrect":
        original_no_leak = check_no_correct_answer_before_newline(original_generated, original_prompt, true_answer)
        original_truly_incorrect = not original_correct and original_no_leak
    else:
        original_truly_incorrect = not original_correct
    
    print(
        f"    - Original: True answer: '{true_answer}', Model answer: '{original_generated}' -> {'Correct' if original_correct else 'Incorrect'}"
    )
    if search_mode == "all_incorrect" and not original_correct:
        print(f"      No answer leak before newline: {'✓' if original_no_leak else '✗'}")
    
    # Test counterfactual program
    cf_prompt = prompts.VARIABLE_BINDING.format(code=counterfactual_program)
    cf_generated = run_inference_nnsight(model, cf_prompt, token_only=True)
    cf_correct = cf_generated == str(counterfactual_answer)
    
    # For incorrect mode, also check that correct answer doesn't appear before first newline
    if search_mode == "all_incorrect":
        cf_no_leak = check_no_correct_answer_before_newline(cf_generated, cf_prompt, counterfactual_answer)
        cf_truly_incorrect = not cf_correct and cf_no_leak
    else:
        cf_truly_incorrect = not cf_correct
    
    print(
        f"    - Counterfactual: True answer: '{counterfactual_answer}', Model answer: '{cf_generated}' -> {'Correct' if cf_correct else 'Incorrect'}"
    )
    if search_mode == "all_incorrect" and not cf_correct:
        print(f"      No answer leak before newline: {'✓' if cf_no_leak else '✗'}")

    # Return whether this model meets the condition
    if search_mode == "all_correct":
        both_correct = original_correct and cf_correct
        return both_correct
    else:  # all_incorrect
        both_truly_incorrect = original_truly_incorrect and cf_truly_incorrect
        return both_truly_incorrect




def find_program(exact_hops, seq_len, max_seed_search, find_failed):
    """
    Searches for an RNG seed that produces a program which all specified models
    either solve correctly or fail to solve, based on configuration.
    Loads each model once and tests all seeds with it.
    """
    search_mode = "all_incorrect" if find_failed else "all_correct"

    if search_mode == "all_correct":
        print(
            f"Searching for a ROBUST program of sequence length {seq_len} with exactly {exact_hops} hops (all models must be correct)..."
        )
    else:
        print(
            f"Searching for a FAILED program of sequence length {seq_len} with exactly {exact_hops} hops (all models must be incorrect)..."
        )
    print(f"Models being tested: {', '.join(MODEL_IDS)}")

    # Use a base tokenizer for program generation (it's model-agnostic text)
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[0])
    
    # Pre-generate all programs to test
    print(f"Pre-generating programs with exactly {exact_hops} hops...")
    programs_to_test = []
    seed = 0
    attempts = 0
    max_attempts = max_seed_search * 10  # Try more seeds to find programs with desired hops
    
    while len(programs_to_test) < max_seed_search and attempts < max_attempts:
        rng = np.random.RandomState(seed)
        program, answer, query_hops, metadata = make_variable_binding_program_with_metadata(
            seq_len=seq_len, rng=rng, tokenizer=base_tokenizer
        )
        
        # Only include programs with the exact number of hops
        if query_hops == exact_hops:
            programs_to_test.append((seed, program, answer, metadata["query_var"], query_hops))
            print(f"  Found program with {query_hops} hops (seed {seed})")
        
        seed += 1
        attempts += 1
    
    if len(programs_to_test) < max_seed_search:
        print(f"Warning: Only found {len(programs_to_test)} programs with exactly {exact_hops} hops after {attempts} attempts")
    else:
        print(f"Generated {len(programs_to_test)} programs with exactly {exact_hops} hops")
    
    # Track results for each seed across all models
    seed_results = {}  # seed -> {model_id: bool}
    failed_seeds = set()  # Seeds that have already failed with at least one model
    
    # Test each model with all programs
    for model_id in MODEL_IDS:
        print(f"\n=== Loading model {model_id} ===")
        model = LanguageModel(model_id, device_map="auto")
        
        try:
            for seed, program, answer, query_var, query_hops in tqdm(
                programs_to_test, desc=f"Testing {model_id}", leave=False
            ):
                # Skip seeds that have already failed with a previous model
                if seed in failed_seeds:
                    continue
                
                if seed not in seed_results:
                    seed_results[seed] = {}
                
                print(f"--- Testing seed {seed} with {model_id} ---")
                result = test_program_with_model(model, program, str(answer), query_var, search_mode)
                seed_results[seed][model_id] = result
                
                print(f"    -> Result: {'✓' if result else '✗'}")
                
                # If this seed failed with this model, mark it as failed and skip for remaining models
                if not result:
                    failed_seeds.add(seed)
                    print(f"    -> Seed {seed} failed, will skip for remaining models")
        
        finally:
            # Clean up current model before loading next one
            print(f"Cleaning up {model_id}...")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Find seeds where all models meet the condition
    print("\n=== Analyzing results ===")
    for i, (seed, program, answer, query_var, query_hops) in enumerate(programs_to_test):
        if seed in seed_results:
            model_results = seed_results[seed]
            if len(model_results) == len(MODEL_IDS) and all(model_results.values()):
                # Found a seed that works for all models!
                
                print("\n" + "=" * 50)
                if search_mode == "all_correct":
                    print("🎉 Found a robust program (all models correct)!")
                else:
                    print("🎉 Found a universally failed program (all models incorrect)!")
                print(f"RNG Seed: {seed}")
                print(f"Expected Answer: {answer}")
                print(f"Number of Hops: {query_hops}")
                print("--- Program ---")
                print(program)
                print("=" * 50 + "\n")
                return seed, program, answer
    
    print(f"\nCould not find a suitable program after trying {max_seed_search} seeds.")
    return None, None, None


@click.command()
@click.option("--exact-hops", default=2, help="Exact number of hops in variable binding chain")
@click.option("--seq-len", default=17, help="Sequence length for programs")
@click.option("--max-seed-search", default=100, help="Maximum number of seeds to try")
@click.option("--find-failed", is_flag=True, help="Find universally failed programs instead of robust ones")
def main(exact_hops, seq_len, max_seed_search, find_failed):
    """Find robust seeds for causal tracing experiments."""
    find_program(exact_hops, seq_len, max_seed_search, find_failed)


if __name__ == "__main__":
    main() 