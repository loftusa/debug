#%%
"""
Attention Head Patching Experiment

Tests causal effects of individual attention heads on variable binding tasks.
Systematically patches each attention head at each token position and layer
to identify which heads are responsible for tracking variable bindings.

Based on methodology from "Tracing Knowledge in Language Models Back to the Training Data"
https://arxiv.org/abs/2505.20896
"""

from datetime import datetime
import json
from pathlib import Path
from typing import List
import torch
import gc
import numpy as np
from tqdm import tqdm

# Add src to PYTHONPATH when running as a script
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.append(str(project_root))

from debug.counterfactual import CounterfactualGenerator
from debug.generators import make_variable_binding_program_with_metadata
from debug.causal_tracing import CausalTracer, InterventionResult
#%%

# Quick test of attention head intervention
tracer = CausalTracer("Qwen/Qwen3-0.6B")

# program generation 
rng = np.random.RandomState(42)
program, answer, hops, metadata = make_variable_binding_program_with_metadata(
    seq_len=17, rng=rng, tokenizer=tracer.tokenizer
)

counterfactual_generator = CounterfactualGenerator()
counter_program = counterfactual_generator.create_counterfactual(program, metadata["query_var"])

target_token_pos = 2
layer_idx = 3
head_idx = 0
n_heads = tracer.model.config.num_attention_heads

print(f"Model has {n_heads} attention heads")
print(f"Targeting layer {layer_idx}, head {head_idx}, token position {target_token_pos}")


result = tracer.run_attention_head_intervention(
    original_program=program,
    counterfactual_program=counter_program,
    target_token_pos=target_token_pos,
    layer_idx=layer_idx,
    head_idx=head_idx,
)
#%%
print(result)

#%%
# softmax the logits
# original_probs = torch.softmax(original_logits, dim=-1)
# patched_probs = torch.softmax(patched_logits, dim=-1)

# # get KL divergence
# kl_div = torch.nn.functional.kl_div(
#     torch.log(patched_probs),
#     torch.log(original_probs), 
#     reduction='batchmean',
#     log_target=True
# )

# print(f"KL divergence: {kl_div}")



#%%
# def run_full_attention_head_patching(
#     tracer: CausalTracer,
#     original_program: str,
#     counterfactual_program: str,
#     max_layers: int | None = None,
#     max_heads: int | None = None,
# ) -> List[InterventionResult]:
#     """Patch attention heads at every token position and layer.

#     Args:
#         tracer: A ready‐initialised CausalTracer.
#         original_program: Clean program string.
#         counterfactual_program: Counterfactual program string (root value flipped).
#         max_layers: Optionally cap the number of layers to test.
#         max_heads: Optionally cap the number of heads to test per layer.

#     Returns:
#         List of InterventionResult objects covering all (token, layer, head) combinations.
#     """
#     # Tokenise once to know valid positions (use underlying HF tokenizer)
#     enc = tracer.tokenizer
#     token_ids = enc(
#         original_program, return_tensors="pt", add_special_tokens=False
#     ).input_ids[0]
#     seq_len = token_ids.shape[0]
    
#     # Get model architecture info
#     n_layers = tracer._n_layers if max_layers is None else min(tracer._n_layers, max_layers)
#     n_heads = tracer.model.config.num_attention_heads
#     test_heads = n_heads if max_heads is None else min(n_heads, max_heads)
    
#     print(f"Testing {seq_len} tokens × {n_layers} layers × {test_heads} heads = {seq_len * n_layers * test_heads} interventions")

#     all_results: List[InterventionResult] = []

#     # We are testing a single program, so we can assign a consistent ID
#     program_id = 0

#     for token_pos in tqdm(range(seq_len), desc="Token position"):
#         for layer_idx in range(n_layers):
#             for head_idx in range(test_heads):
#                 # The target name helps identify the intervention later
#                 target_name = f"pos_{token_pos}_layer_{layer_idx}_head_{head_idx}"
                
#                 try:
#                     result = tracer.run_attention_head_intervention(
#                         original_program=original_program,
#                         counterfactual_program=counterfactual_program,
#                         target_token_pos=int(token_pos),
#                         layer_idx=layer_idx,
#                         head_idx=head_idx,
#                         program_id=program_id,
#                         target_name=target_name
#                     )
#                     all_results.append(result)
#                 except Exception as e:
#                     print(f"Error at token {token_pos}, layer {layer_idx}, head {head_idx}: {e}")
#                     continue
                
#                 # Memory cleanup every few interventions
#                 if len(all_results) % 50 == 0:
#                     gc.collect()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()

#     return all_results


# if __name__ == "__main__":
#     # --- Configuration -----------------------------------------------------
#     # Test only the smallest model for attention head patching
#     MODEL_IDS = [
#         "Qwen/Qwen3-0.6B",
#     ]
#     SEQ_LEN = 17
#     RNG_SEED = 42
#     # Limit testing to first few layers and heads for efficiency
#     MAX_LAYERS = 4  # Test only first 4 layers
#     MAX_HEADS = 4   # Test only first 4 heads per layer
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     BASE_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "attention_head_patching" / timestamp
#     BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     # --- Program Generation (once for all models) --------------------------
#     from transformers import AutoTokenizer

#     # Use the tokenizer from the first model for program generation.
#     # The program text itself is model-agnostic.
#     base_tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[0])
#     rng = np.random.RandomState(RNG_SEED)
#     program, answer, hops, metadata = make_variable_binding_program_with_metadata(
#         seq_len=SEQ_LEN, rng=rng, tokenizer=base_tokenizer
#     )

#     query_var = metadata["query_var"]

#     # Construct counterfactual by flipping root value
#     from debug.counterfactual import CounterfactualGenerator

#     counterfactual_generator = CounterfactualGenerator()
#     counter_program = counterfactual_generator.create_counterfactual(program, query_var)

#     print("Original program:\n", program)
#     print("\nCounterfactual program:\n", counter_program)

#     # --- Run Experiment for each model -------------------------------------
#     for model_id in MODEL_IDS:

#         print(f"\n\n=== Running experiment for model: {model_id} ===")
        
#         # Create a model-specific output directory
#         model_name_safe = model_id.replace("/", "_")
#         model_output_dir = BASE_OUTPUT_DIR / model_name_safe
#         model_output_dir.mkdir(parents=True, exist_ok=True)


#         # Load the model
#         tracer = CausalTracer(model_id)

#         print("\nRunning full token × layer × attention head patching …")
#         results = run_full_attention_head_patching(
#             tracer, program, counter_program, max_layers=MAX_LAYERS, max_heads=MAX_HEADS
#         )

#         # Save the full intervention data for richer analysis later
#         serialisable = [
#             {
#                 "intervention_type": r.intervention_type,
#                 "layer_idx": r.layer_idx,
#                 "head_idx": r.head_idx,
#                 "target_token_pos": r.target_token_pos,
#                 "logit_difference": r.logit_difference,
#                 "normalized_logit_difference": r.normalized_logit_difference,
#                 "success_rate": r.success_rate,
#                 "original_top_token": r.original_top_token,
#                 "intervened_top_token": r.intervened_top_token,
#                 "program_id": r.program_id,
#                 "original_program": r.original_program,
#                 "counterfactual_program": r.counterfactual_program,
#                 "token_labels": r.token_labels,
#                 "target_name": r.target_name,
#             }
#             for r in results
#         ]

#         out_path = model_output_dir / "intervention_results.json"
#         with open(out_path, "w") as f:
#             json.dump(serialisable, f, indent=2)

#         print(f"Saved {len(results)} attention head intervention results for {model_id} ➜ {out_path}\n") 
#         del tracer
#         del results
#         del serialisable
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()