#%%
"""Causal tracing implementation using nnsight for intervention experiments."""

from torch import Tensor
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from nnsight import LanguageModel
import numpy as np
from jaxtyping import Float
import torch
import einops  # For reshaping attention head tensors
import gc

@dataclass
class InterventionResult:
    """Result of a causal intervention experiment."""
    intervention_type: str  # "residual_stream" or "attention_head"
    layer_idx: int
    head_idx: Optional[int] = None
    target_token_pos: int = None
    logit_difference: float = None
    normalized_logit_difference: float = None
    success_rate: float = None
    original_logits: Optional[Float[Tensor, "batch seq vocab"]] = None
    intervened_logits: Optional[Float[Tensor, "batch seq vocab"]] = None
    original_top_token: Optional[int] = None
    intervened_top_token: Optional[int] = None
    # Program information for visualization
    program_id: Optional[int] = None  # Links back to which program this intervention came from
    original_program: Optional[str] = None  # Full program text
    counterfactual_program: Optional[str] = None  # Counterfactual program text
    token_labels: Optional[List[str]] = None  # Token labels for visualization
    target_name: Optional[str] = None  # Name of the intervention target (e.g., "ref_depth_2_rhs")


class CausalTracer:
    """Performs causal tracing experiments using nnsight interventions."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize CausalTracer with a language model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for model ("auto", "cuda", "cpu")
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.device = device

        # Validate model is from Qwen series
        if "qwen" not in self.model_name.lower():
            raise ValueError(f"Model {self.model_name} is not a Qwen series model. "
                           f"This method requires a Qwen model for proper layer access.")
        
        # Load model with nnsight
        self.model = LanguageModel(model_name, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._n_layers = self.model.config.num_hidden_layers
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded: {self.model.config.num_hidden_layers} layers")
    
    def _analyze_intervention_results(self,
                                    original_logits: Float[Tensor, "batch seq vocab"],
                                    intervened_logits: Float[Tensor, "batch seq vocab"],
                                    intervention_type: str,
                                    layer_idx: int,
                                    target_token_pos: int,
                                    head_idx: Optional[int] = None,
                                    store_logits: bool = False,
                                    program_id: Optional[int] = None,
                                    original_program: Optional[str] = None,
                                    counterfactual_program: Optional[str] = None,
                                    token_labels: Optional[List[str]] = None,
                                    target_name: Optional[str] = None) -> InterventionResult:
        """Analyze the results of an intervention experiment."""
        
        # Get top tokens
        original_top = original_logits[:, -1, :].argmax(dim=-1).item()
        intervened_top = intervened_logits[:, -1, :].argmax(dim=-1).item()
        
        # Get final position logits for calculation
        orig_final = original_logits[:, -1, :]  # [batch, vocab]
        interv_final = intervened_logits[:, -1, :]  # [batch, vocab]
        
        # Calculate raw logit difference for the intervened top token
        # This represents the change in pre-softmax activation
        raw_logit_diff = (interv_final[:, intervened_top] - orig_final[:, intervened_top]).mean().item()
        
        # Calculate normalized logit difference
        # Normalize by maximum possible difference across the vocabulary
        max_logit_range = orig_final.max() - orig_final.min()
        if max_logit_range > 0:
            normalized_logit_diff = raw_logit_diff / max_logit_range.item()
            # Clamp to [-1, 1] range
            normalized_logit_diff = max(-1.0, min(1.0, normalized_logit_diff))
        else:
            normalized_logit_diff = 0.0
        
        return InterventionResult(
            intervention_type=intervention_type,
            layer_idx=layer_idx,
            head_idx=head_idx,
            target_token_pos=target_token_pos,
            logit_difference=raw_logit_diff,
            normalized_logit_difference=normalized_logit_diff,
            success_rate=1.0 if intervened_top != original_top else 0.0,
            original_logits=original_logits.cpu() if store_logits else None,
            intervened_logits=intervened_logits.cpu() if store_logits else None,
            original_top_token=original_top,
            intervened_top_token=intervened_top,
            program_id=program_id,
            original_program=original_program,
            counterfactual_program=counterfactual_program,
            token_labels=token_labels,
            target_name=target_name
        )  
    def run_residual_stream_intervention(self,
                                       original_program: str,
                                       counterfactual_program: str,
                                       target_token_pos: int,
                                       layer_idx: int,
                                       store_logits: bool = False,
                                       program_id: Optional[int] = None,
                                       target_name: Optional[str] = None) -> InterventionResult:
        """
        Run causal intervention on residual stream activations.
        This implementation uses separate trace contexts for each step to improve
        stability with the nnsight library, avoiding complex graph issues.
        
        Args:
            original_program: The original program text
            counterfactual_program: The counterfactual program text
            target_token_pos: Token position to intervene on
            layer_idx: Layer index for intervention
            
        Returns:
            InterventionResult with intervention effects
        """
        
        # 1. Get clean logits on the original program
        with self.model.trace(original_program):
            clean_logits = self.model.lm_head.output.save()
        
        # 2. Get the corrupting activation from the counterfactual program
        with self.model.trace(counterfactual_program):
            corrupt_activations = self.model.model.layers[layer_idx].output[0][:, [target_token_pos], :].save()
        
        # 3. Run the original program again, but patch in the corrupt activation
        with self.model.trace(original_program):
            self.model.model.layers[layer_idx].output[0][:, [target_token_pos], :] = corrupt_activations
            patched_logits = self.model.lm_head.output.save()
        
        # Generate token labels for visualization
        token_labels = None
        if original_program:
            try:
                token_labels = self.tokenizer.tokenize(original_program)
            except Exception:
                # Fallback to simple splitting if tokenization fails
                token_labels = original_program.split()
        
        # Calculate intervention effects
        result = self._analyze_intervention_results(
            original_logits=clean_logits,
            intervened_logits=patched_logits,
            intervention_type="residual_stream",
            layer_idx=layer_idx,
            target_token_pos=target_token_pos,
            store_logits=store_logits,
            program_id=program_id,
            original_program=original_program,
            counterfactual_program=counterfactual_program,
            token_labels=token_labels,
            target_name=target_name
        )
        
        return result
    
    def run_attention_head_intervention(self,
                                      original_program: str,
                                      counterfactual_program: str,
                                      target_token_pos: int,
                                      layer_idx: int,
                                      head_idx: int,
                                      program_id: Optional[int] = None,
                                      target_name: Optional[str] = None) -> InterventionResult:
        """
        Run causal intervention on attention head outputs.
        
        Args:
            original_program: The original program text
            counterfactual_program: The counterfactual program text
            target_token_pos: Token position to intervene on
            layer_idx: Layer index for intervention
            head_idx: Attention head index
            
        Returns:
            InterventionResult with intervention effects
        """
        # Proper head-level activation patching.
        # NOTE: Qwen-style models follow Llama architecture: each layer has `self_attn` with projection `o_proj`.
        # The tensor flowing **into** o_proj has shape [batch, seq, n_heads * head_dim]. We reshape to access a single head.

        n_heads = self.model.config.num_attention_heads

        with self.model.trace() as tracer:
            # 1) Clean run – record original logits
            with tracer.invoke(original_program):
                original_logits = self.model.lm_head.output.save()
        
            # 2) Corrupted run – capture hidden state of the chosen head
            with tracer.invoke(counterfactual_program):
                z_corrupt = self.model.model.layers[layer_idx].self_attn.o_proj.input  # [b, s, n_heads*h_dim]
                z_corrupt = einops.rearrange(z_corrupt, 'b s (nh dh) -> b s nh dh', nh=n_heads)
                head_activation = z_corrupt[:, target_token_pos, head_idx, :].save()

            # 3) Patched run – replace the single head activation in the clean prompt
            with tracer.invoke(original_program):
                z_clean = self.model.model.layers[layer_idx].self_attn.o_proj.input
                z_clean = einops.rearrange(z_clean, 'b s (nh dh) -> b s nh dh', nh=n_heads)
                z_clean[:, target_token_pos, head_idx, :] = head_activation
                z_clean = einops.rearrange(z_clean, 'b s nh dh -> b s (nh dh)', nh=n_heads)
                # Write back the patched tensor using index 0
                self.model.model.layers[layer_idx].self_attn.o_proj.input[0] = z_clean

                patched_logits = self.model.lm_head.output.save()
        
        # Generate token labels for visualization
        token_labels = None
        if original_program:
            try:
                token_labels = self.tokenizer.tokenize(original_program)
            except Exception:
                # Fallback to simple splitting if tokenization fails
                token_labels = original_program.split()
        
        # Calculate intervention effects
        result = self._analyze_intervention_results(
            original_logits=original_logits,
            intervened_logits=patched_logits,
            intervention_type="attention_head",
            layer_idx=layer_idx,
            head_idx=head_idx,
            target_token_pos=target_token_pos,
            store_logits=False,
            program_id=program_id,
            original_program=original_program,
            counterfactual_program=counterfactual_program,
            token_labels=token_labels,
            target_name=target_name
        )
        
        return result
    
    def run_systematic_intervention(self,
                                  original_program: str,
                                  counterfactual_program: str,
                                  target_token_pos: int,
                                  max_layers: Optional[int] = None,
                                  store_logits: bool = False,
                                  program_id: Optional[int] = None,
                                  target_name: Optional[str] = None) -> List[InterventionResult]:
        """
        Run a systematic intervention across all specified layers for a given
        token position by calling `run_residual_stream_intervention` in a loop.
        """
        num_layers = self._n_layers if max_layers is None else min(self._n_layers, max_layers)
        results: List[InterventionResult] = []

        for layer_idx in range(num_layers):
            print(f"  Testing layer {layer_idx}/{num_layers-1}")
            result = self.run_residual_stream_intervention(
                    original_program=original_program,
                    counterfactual_program=counterfactual_program,
                    target_token_pos=target_token_pos,
                    layer_idx=layer_idx,
                    store_logits=store_logits,
                    program_id=program_id,
                    target_name=target_name
                )
            results.append(result)

            # Aggressively clean up memory after each layer's intervention
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results
    
    def calculate_success_rate(self,
                             intervened_logits: torch.Tensor,
                             original_token: int,
                             counterfactual_token: int) -> float:
        """
        Calculate success rate: percentage where counterfactual token becomes top prediction.
        
        Args:
            intervened_logits: Logits after intervention [batch, seq, vocab]
            original_token: Original answer token ID
            counterfactual_token: Counterfactual answer token ID
            
        Returns:
            Success rate as float between 0.0 and 1.0
        """
        # Get top prediction at final position
        final_logits = intervened_logits[:, -1, :]  # [batch, vocab]
        top_tokens = final_logits.argmax(dim=-1)  # [batch]
        
        # Calculate success rate
        success_count = (top_tokens == counterfactual_token).sum().item()
        total_count = top_tokens.size(0)
        
        return success_count / total_count if total_count > 0 else 0.0
    
    
if __name__ == "__main__":
    print("=== Causal Tracing Debug Test ===")
    
    # Create a simple test case
    original_program = """a = 5
b = a
c = b
#c:"""
    
    counterfactual_program = """a = 8
b = a
c = b
#c:"""
    
    print("Original program:")
    print(original_program)
    print("\nCounterfactual program:")
    print(counterfactual_program)
    
    from nnsight import LanguageModel
    model = LanguageModel("Qwen/Qwen3-0.6B", device_map="auto")

    layer = 1
    with model.trace() as tracer:
        activations = model.layers[layer].output.save()

#%%