import re
from pathlib import Path
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
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-12b-it",
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
    "google/gemma-2-9b-it",
    "google/gemma-3-27b-it",
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
@click.option("--num-seqs", "num_seqs", default=124, help="Number of sequences per model.")
@click.option("--seq_len", default=10, help="Maximum length (steps) of each sequence.")
@click.option("--best-of", "best_of", default=10, help="Number of parallel samples per prompt.")
@click.option("--kind", "kind", default='groups', help="Kind of sequence to generate.")
def main(num_seqs: int, seq_len: int, best_of: int, kind: str) -> None:  # noqa: D401
    """Evaluate multiple open‑source models on the variable‑tracking task."""
    results: List[Tuple[str, float]] = []
    results_dir = Path("results") / f"seq_len_{seq_len}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results_summary.csv"
    if kind == 'ops':
        csv_path = results_dir / "results_summary_ops.csv"
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
        print(f"\nLoading model: {model_id}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            llm = pipeline("text-generation", model=model_id, tokenizer=tok, temperature=0.8, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        except (RuntimeError, MemoryError) as e:
            print(f"Skipping model {model_id} due to load memory error: {e}")
            continue
        correct = 0
        for _ in range(num_seqs):
            code, intermediate = make_sequence(seq_len, kind=kind)
            true_val = intermediate[-1]
            prompt = PROMPT_TEMPLATE.format(code=code)

            # Call the pipeline, requesting multiple sequences for best-of-k
            try:
                req_outputs = llm(
                    prompt,
                    num_return_sequences=best_of, # Use the best_of parameter
                    max_new_tokens=10,           # Limit output length for efficiency
                    do_sample=True,             
                                             # Set do_sample=True if using temperature > 0
                )
            except (RuntimeError, MemoryError) as e:
                print(f"Memory error during inference for {model_id}: {e}. Skipping model.")
                skip_model = True
                break

            print(req_outputs, '\n\n') # Keep for debugging if needed
            
            # Parse the best prediction from the multiple outputs
            pred_int = _best_of_k(req_outputs, true_val)
            all_data.append({
                "code": code,
                "intermediate": intermediate,
                "true_val": true_val,
                "pred_int": pred_int,
                "outputs": req_outputs,
                "correct": pred_int == true_val,
            })
            if pred_int == true_val:
                correct += 1
        # Skip finalizing this model if it ran out of memory
        if skip_model:
            continue
        acc = correct / num_seqs
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
        del llm
        # Free up memory between models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
