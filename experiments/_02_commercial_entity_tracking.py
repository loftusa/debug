import re
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import click
import requests
import csv
import json
import time
from debug.sequence_generation import make_sequence

print('imports loaded')

PROMPT_TEMPLATE = (
    "You are given a short Python program. "
    "Your task is to compute the final value of the variable x. "
    "Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.\n" 
    "```python\n{code}\n```\n"
    "The final value of x is: "
)

# Models to evaluate via OpenRouter
COMMERCIAL_MODELS: List[str] = [
    "openai/gpt-4o",
    # "anthropic/claude-3.7-sonnet"
    # "google/gemini-1.5-pro-latest",
    # "deepseek/deepseek-chat-v3.1"
]

# Set up OpenRouter client
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

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


def openrouter_completion(
    model_id: str, 
    prompt: str, 
    temperature: float = 0.7, 
    max_tokens: int = 20,
    n: int = 1
) -> List[Dict[str, Any]]:
    """Make a request to OpenRouter API for chat completion."""
    # Filter the primary model out of the fallback list
    fallback_models = [m for m in COMMERCIAL_MODELS if m != model_id]
    
    # Models known to cause 400 errors when 'models' is present with n > 1
    problematic_models = [
        "anthropic/claude-3-5-sonnet", 
        "google/gemini-1.5-pro-latest", 
        "deepseek/deepseek-chat-v3.1"
    ]
    
    payload = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # Conditionally add the 'models' fallback parameter
    if model_id not in problematic_models:
        payload["models"] = fallback_models
    
    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=OPENROUTER_HEADERS,
            json=payload
        )
        response.raise_for_status()
        response_data = response.json()
        # Get actual model used (might be fallback)
        model_used = response_data.get("model", model_id)
        
        # Format response to match transformers pipeline format
        formatted_outputs = []
        for choice in response_data.get("choices", []):
            formatted_outputs.append({
                "generated_text": choice.get("message", {}).get("content", ""),
                "model_used": model_used
            })
        return formatted_outputs
    except requests.exceptions.RequestException as e:
        print(f"Error making request to OpenRouter: {e}")
        # Return empty list on error
        return []


def _best_of_k(outputs: List[Dict[str, str]], true_val: int) -> Tuple[Optional[int], Optional[str]]:
    """Return the integer prediction closest to *true_val* among *outputs* and the model used."""
    if not outputs:
        return None, None
        
    preds = []
    for o in outputs:
        parsed = _parse_int(o.get("generated_text", ""))
        if parsed is not None:
            preds.append((parsed, o.get("model_used")))
            
    if not preds:
        return None, outputs[0].get("model_used") if outputs else None
        
    # If any exactly equals, prefer that
    for p, model in preds:
        if p == true_val:
            return p, model
            
    # Otherwise, return first parsed value
    return preds[0][0], preds[0][1]


@click.command()
@click.option("--num-seqs", "num_seqs", default=50, help="Number of sequences per model.")
@click.option("--seq_len", default=10, help="Maximum length (steps) of each sequence.")
@click.option("--best-of", "best_of", default=4, help="Number of parallel samples per prompt.")
def main(num_seqs: int, seq_len: int, best_of: int) -> None:  # noqa: D401
    """Evaluate commercial models via OpenRouter on the variable‑tracking task."""
    results: List[Tuple[str, float]] = []
    results_dir = Path("results") / f"seq_len_{seq_len}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results_summary.csv"
    
    # Read existing results to skip already evaluated models
    processed_models = set()
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
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
            
    for model_id in COMMERCIAL_MODELS:
        if model_id in processed_models:
            print(f"Skipping model {model_id} as it's already evaluated.")
            continue
            
        all_data = []
        print(f"\nEvaluating model: {model_id}")
        
        correct = 0
        for seq_idx in range(num_seqs):
            print(f"  Sequence {seq_idx+1}/{num_seqs}")
            code, intermediate = make_sequence(seq_len)
            true_val = intermediate[-1]
            prompt = PROMPT_TEMPLATE.format(code=code)

            # Make API request with retries
            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    req_outputs = openrouter_completion(
                        model_id=model_id,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=20,
                        n=best_of
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error, retrying in {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
                        req_outputs = []
            
            # Print first output for debugging
            if req_outputs:
                print(f"Sample output: {req_outputs[0]['generated_text']}")
            
            # Parse the best prediction from the multiple outputs
            pred_int, model_used = _best_of_k(req_outputs, true_val)
            all_data.append({
                "code": code,
                "intermediate": intermediate,
                "true_val": true_val,
                "pred_int": pred_int,
                "outputs": req_outputs,
                "correct": pred_int == true_val,
                "model_used": model_used
            })
            
            if pred_int == true_val:
                correct += 1
                
            # Add small delay to avoid rate limits
            time.sleep(0.5)
            
        acc = correct / num_seqs if num_seqs > 0 else 0
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


if __name__ == "__main__":
    main() 