import gc
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer
from nnsight import LanguageModel

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1] / "src"
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from debug import prompts
from debug.generators import make_variable_binding_program_with_metadata

# --- Configuration ---
MODEL_IDS = [
    # "Qwen/Qwen3-0.6B", 
    # "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]
SEQ_LEN = 17
MAX_SEED_SEARCH = 1000  # Maximum number of seeds to try
FIND_UNIVERSALLY_FAILED_PROGRAM = False  # Set to True to find a program all models fail on.


def run_inference_nnsight(model: LanguageModel, prompt: str) -> str:
    """
    Runs a single inference call using NNSight tracing.
    This does greedy decoding by taking argmax of the logits.
    """
    with model.trace(prompt):
        logits = model.lm_head.output.save()
    
    last_token_logits = logits[0, -1, :]
    predicted_token_id = last_token_logits.argmax().item()
    predicted_token = model.tokenizer.decode([predicted_token_id])
    return prompt + predicted_token


def extract_answer(generated_text: str, prompt: str) -> str:
    """Extracts the answer from the generated text."""
    answer_part = generated_text[len(prompt) :]
    number_match = re.search(r'\b(\d+)\b(?=\n|$)', answer_part)
    if number_match:
        return number_match.group(1)
    
    return answer_part.strip()


def check_program_with_condition(
    program: str, true_answer: str, search_mode: str, model_dict: dict
) -> bool:
    """
    Checks if a program satisfies the search criteria across all models.

    Args:
        program: The program string to test.
        true_answer: The expected answer.
        search_mode: 'all_correct' or 'all_incorrect'.
        model_dict: Dictionary containing loaded models.

    Returns:
        True if the condition is met, False otherwise.
    """
    prompt = prompts.VARIABLE_BINDING.format(code=program)

    for model_id in MODEL_IDS:
        print(f"  Testing model: {model_id}...")
        model = model_dict[model_id]

        generated_text = run_inference_nnsight(model, prompt)
        extracted = extract_answer(generated_text, prompt)

        correct = extracted == str(true_answer)
        print(
            f"    - True answer: '{true_answer}', Model answer: '{extracted}' -> {'Correct' if correct else 'Incorrect'}"
        )
        print(f"generated_text: {generated_text}")

        # Early exit conditions based on search mode
        if search_mode == "all_correct" and not correct:
            print(
                "    -> Condition not met (expected correct). Failing this seed early."
            )
            return False
        if search_mode == "all_incorrect" and correct:
            print(
                "    -> Condition not met (expected incorrect). Failing this seed early."
            )
            return False

    # If we completed the loop, the early exit was never triggered, so the condition was met.
    return True


def load_all_models():
    """Load all models once using NNSight and return a dictionary."""
    print("Loading all models...")
    model_dict = {}
    
    for model_id in MODEL_IDS:
        print(f"Loading {model_id}...")
        # NNSight LanguageModel wraps the HuggingFace model and provides tracing capabilities
        model = LanguageModel(model_id, device_map="auto")
        model_dict[model_id] = model
    
    print("All models loaded successfully!")
    return model_dict


def find_program():
    """
    Searches for an RNG seed that produces a program which all specified models
    either solve correctly or fail to solve, based on configuration.
    """
    search_mode = "all_incorrect" if FIND_UNIVERSALLY_FAILED_PROGRAM else "all_correct"

    if search_mode == "all_correct":
        print(
            f"Searching for a ROBUST program of sequence length {SEQ_LEN} (all models must be correct)..."
        )
    else:
        print(
            f"Searching for a FAILED program of sequence length {SEQ_LEN} (all models must be incorrect)..."
        )
    print(f"Models being tested: {', '.join(MODEL_IDS)}")

    # Load all models once
    model_dict = load_all_models()

    try:
        # Use a base tokenizer for program generation (it's model-agnostic text)
        base_tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[0])

        for seed in trange(
            MAX_SEED_SEARCH, desc=f"Searching for {search_mode.replace('_', ' ')} seed"
        ):
            # This print statement is useful for seeing progress in logs
            # even if tqdm is redirected.
            print(f"--- Trying RNG seed: {seed} ---")
            rng = np.random.RandomState(seed)

            program, answer, _, _ = make_variable_binding_program_with_metadata(
                seq_len=SEQ_LEN, rng=rng, tokenizer=base_tokenizer
            )

            if check_program_with_condition(program, str(answer), search_mode, model_dict):
                print("\n" + "=" * 50)
                if search_mode == "all_correct":
                    print("🎉 Found a robust program (all models correct)!")
                else:
                    print("🎉 Found a universally failed program (all models incorrect)!")
                print(f"RNG Seed: {seed}")
                print(f"Expected Answer: {answer}")
                print("--- Program ---")
                print(program)
                print("=" * 50 + "\n")
                return seed, program, answer

        print(f"\nCould not find a suitable program after trying {MAX_SEED_SEARCH} seeds.")
        return None, None, None
    
    finally:
        # Clean up all models at the end
        print("Cleaning up models...")
        for model_id in model_dict:
            del model_dict[model_id]
        del model_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    find_program() 