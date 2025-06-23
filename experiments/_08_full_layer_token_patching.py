from datetime import datetime
import json
from pathlib import Path
from typing import List
import torch
import gc
import numpy as np
from tqdm import tqdm
import re

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.append(str(project_root))

from debug import prompts
from debug.generators import make_variable_binding_program_with_metadata
from debug.causal_tracing import CausalTracer, InterventionResult


def run_full_token_layer_patching(
    tracer: CausalTracer,
    original_program: str,
    counterfactual_program: str,
    max_layers: int | None = None,
) -> List[InterventionResult]:
    """Patch *every* token position at *every* layer.

    Args:
        tracer: A ready‐initialised CausalTracer.
        original_program: Clean program string.
        counterfactual_program: Counterfactual program string (root value flipped).
        max_layers: Optionally cap the number of layers to test.

    Returns:
        List of InterventionResult objects covering all (token, layer) pairs.
    """
    # Tokenise once to know valid positions (use underlying HF tokenizer)
    enc = tracer.tokenizer
    token_ids = enc(
        original_program, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    seq_len = token_ids.shape[0]

    all_results: List[InterventionResult] = []

    # We are testing a single program, so we can assign a consistent ID
    program_id = 0

    for token_pos in tqdm(range(seq_len), desc="Token position"):
        # The target name helps identify the intervention later
        target_name = f"pos_{token_pos}"
        
        results_for_pos = tracer.run_systematic_intervention(
            original_program=original_program,
            counterfactual_program=counterfactual_program,
            target_token_pos=int(token_pos),
            max_layers=max_layers,
            store_logits=False,  # Do not accumulate large tensors in memory
            program_id=program_id,
            target_name=target_name
        )
        all_results.extend(results_for_pos)

    return all_results


def run_inference_nnsight(model, prompt: str) -> str:
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
    answer_part = generated_text[len(prompt):]
    number_match = re.search(r'\b(\d+)\b(?=\n|$)', answer_part)
    if number_match:
        return number_match.group(1)
    return answer_part.strip()


if __name__ == "__main__":
    # --- Configuration -----------------------------------------------------
    MODEL_IDS = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B", 
    # "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",  
    ]
    SEQ_LEN = 5 
    RNG_SEED = 4038
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "full_token_layer_patching_1_hop_seq_5" / timestamp
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Program Generation (once for all models) --------------------------
    from transformers import AutoTokenizer

    # Use the tokenizer from the first model for program generation.
    # The program text itself is model-agnostic.
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[0])
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

    # --- Run Experiment for each model -------------------------------------
    for model_id in MODEL_IDS:

        print(f"\n\n=== Running experiment for model: {model_id} ===")
        
        # Create a model-specific output directory
        model_name_safe = model_id.replace("/", "_")
        model_output_dir = BASE_OUTPUT_DIR / model_name_safe
        model_output_dir.mkdir(parents=True, exist_ok=True)


        # Load the model
        tracer = CausalTracer(model_id)
        
        # Test if the model gets the original program correct
        print("\nTesting base accuracy...")
        
        prompt = prompts.VARIABLE_BINDING.format(code=program)
        generated_text = run_inference_nnsight(tracer.model, prompt)
        model_answer = extract_answer(generated_text, prompt)
        
        if model_answer != str(answer):
            print(f"\n⚠️  WARNING: {model_id} gives INCORRECT answer on base program!")
            print(f"   Expected: {answer}, Got: {model_answer}")
            print(f"   This may affect intervention results.\n")
        else:
            print(f"✓ Model correctly answers: {answer}")

        print("\nRunning full token × layer patching …")
        results = run_full_token_layer_patching(tracer, program, counter_program)

        # Save the full intervention data for richer analysis later
        serialisable = [
            {
                "intervention_type": r.intervention_type,
                "layer_idx": r.layer_idx,
                "head_idx": r.head_idx,
                "target_token_pos": r.target_token_pos,
                "logit_difference": r.logit_difference,
                "normalized_logit_difference": r.normalized_logit_difference,
                "success_rate": r.success_rate,
                "original_top_token": r.original_top_token,
                "intervened_top_token": r.intervened_top_token,
                "program_id": r.program_id,
                "original_program": r.original_program,
                "counterfactual_program": r.counterfactual_program,
                "token_labels": r.token_labels,
                "target_name": r.target_name,
            }
            for r in results
        ]

        out_path = model_output_dir / "intervention_results.json"
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        print(f"Saved {len(results)} intervention results for {model_id} ➜ {out_path}\n") 
        del tracer
        del results
        del serialisable
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()