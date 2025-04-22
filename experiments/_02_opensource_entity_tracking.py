import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import click
from transformers import pipeline, AutoTokenizer
from debug.sequence_generation import make_sequence
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
    # Otherwise, return first parsed value
    return preds[0]


@click.command()
@click.option("--model-id", required=True, help="Hugging Face model ID to evaluate.")
@click.option("--gpu-id", required=True, type=int, help="GPU device ID to run the model on.")
@click.option("--num-seqs", "num_seqs", default=124, help="Number of sequences per model.")
@click.option("--seq_len", default=10, help="Maximum length (steps) of each sequence.")
@click.option("--best-of", "best_of", default=10, help="Number of parallel samples per prompt.")
def main(model_id: str, gpu_id: int, num_seqs: int, seq_len: int, best_of: int) -> None:  # noqa: D401
    """Evaluate a single open‑source model on the variable‑tracking task on a specific GPU."""
    results_dir = Path("results") / f"seq_len_{seq_len}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results_summary.csv"
    detailed_path = results_dir / f"{model_id.replace('/', '_')}_data.json"

    # Check if this specific model/seq_len has already been run by checking detailed file
    if detailed_path.exists():
        print(f"Skipping model {model_id} for seq_len {seq_len} as detailed results file already exists: {detailed_path}")
        return

    # Initialize summary CSV with header if it doesn't exist
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_id", "seq_len", "accuracy"])

    # --- Run evaluation for the single specified model ---
    all_data = []
    print(f"\nLoading model: {model_id} onto GPU: {gpu_id}")
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, running on CPU.")

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        llm = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tok,
            temperature=0.8,
            trust_remote_code=True,
            device=device, # Assign to specific GPU
            torch_dtype=torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability(gpu_id)[0] >= 7 else torch.float32 # Use float16 if Ampere+
        )
    except Exception as e: # Catch broader exceptions during loading
        print(f"ERROR: Failed to load model {model_id} on GPU {gpu_id}: {e}")
        # Optionally write a failure entry? Or just let it skip.
        return # Exit if model loading fails

    correct = 0
    skip_model = False
    for i in range(num_seqs):
        code, intermediate = make_sequence(seq_len)
        true_val = intermediate[-1]
        prompt = PROMPT_TEMPLATE.format(code=code)

        try:
            req_outputs = llm(
                prompt,
                num_return_sequences=best_of,
                max_new_tokens=10,
                do_sample=True,
                pad_token_id=tok.eos_token_id # Set pad_token_id
            )
        except (RuntimeError, MemoryError) as e:
            print(f"ERROR: OOM or Runtime error during inference for {model_id} on GPU {gpu_id}: {e}. Skipping remaining sequences for this model.")
            skip_model = True
            break # Stop processing sequences for this model
        except Exception as e:
            print(f"ERROR: Unexpected error during inference for {model_id} on GPU {gpu_id}: {e}. Skipping remaining sequences for this model.")
            skip_model = True
            break # Stop processing sequences for this model


        # Parse the best prediction from the multiple outputs
        pred_int = _best_of_k(req_outputs, true_val)
        all_data.append({
            "code": code,
            "intermediate": intermediate,
            "true_val": true_val,
            "pred_int": pred_int,
            "outputs": [o['generated_text'] for o in req_outputs], # Store only text
            "correct": pred_int == true_val,
        })
        if pred_int == true_val:
            correct += 1

        # Minimal logging per sequence
        print(f"Model: {model_id}, Seq: {i+1}/{num_seqs}, Correct: {pred_int == true_val} (Pred: {pred_int}, True: {true_val})")

    # --- Finalize results for this model ---
    if not all_data: # Handle case where inference failed on first sequence
        print(f"WARNING: No sequences were successfully processed for model {model_id} on GPU {gpu_id}.")

    else:
        # Calculate accuracy based on processed sequences
        processed_count = len(all_data)
        acc = correct / processed_count if processed_count > 0 else 0.0

        print(f"Finished model {model_id} on GPU {gpu_id}. Accuracy: {acc:.2%} ({correct}/{processed_count})")

        # Save detailed data
        with detailed_path.open("w") as f:
            json.dump(all_data, f, indent=2)
        print(f"Detailed results saved to {detailed_path}")

        # Append accuracy to the summary CSV
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_id, seq_len, f"{acc:.6f}"])
        print(f"Appended summary to {csv_path}")

    # --- Cleanup ---
    print(f"Cleaning up resources for model {model_id} on GPU {gpu_id}")
    del llm
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
