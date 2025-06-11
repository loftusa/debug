#%%
"""Causal tracing implementation using nnsight for intervention experiments."""

import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from nnsight import LanguageModel
import numpy as np


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
    original_logits: Optional[torch.Tensor] = None
    intervened_logits: Optional[torch.Tensor] = None
    original_top_token: Optional[int] = None
    intervened_top_token: Optional[int] = None


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
        
        # Load model with nnsight
        self.model = LanguageModel(model_name, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded: {self.model.config.num_hidden_layers} layers")
    
    def run_residual_stream_intervention(self,
                                       original_program: str,
                                       counterfactual_program: str,
                                       target_token_pos: int,
                                       layer_idx: int) -> InterventionResult:
        """
        Run causal intervention on residual stream activations.
        
        Args:
            original_program: The original program text
            counterfactual_program: The counterfactual program text
            target_token_pos: Token position to intervene on
            layer_idx: Layer index for intervention
            
        Returns:
            InterventionResult with intervention effects
        """
        # Tokenize both programs
        original_tokens = self.tokenizer(original_program, return_tensors="pt")
        counterfactual_tokens = self.tokenizer(counterfactual_program, return_tensors="pt")
        
        # Get model predictions without intervention
        with self.model.trace() as tracer:
            with tracer.invoke(original_program) as clean:
                clean_logits = self.model.lm_head.output
            with tracer.invoke(counterfactual_program) as counterfactual:
                counterfactual_logits = self.model.lm_head.output
        
        # Run intervention: patch counterfactual activation into original
        with self.model.trace(counterfactual_program) as tracer:
            # Save counterfactual activation at target position
            counterfactual_activation = self.model.transformer.h[layer_idx].output[0][:, target_token_pos, :].save()
        
        with self.model.trace(original_tokens) as tracer:
            # Patch counterfactual activation into original run
            self.model.transformer.h[layer_idx].output[0][:, target_token_pos, :] = counterfactual_activation
            intervened_logits = self.model.lm_head.output.save()
        
        # Calculate intervention effects
        result = self._analyze_intervention_results(
            original_logits=original_logits.value,
            intervened_logits=intervened_logits.value,
            intervention_type="residual_stream",
            layer_idx=layer_idx,
            target_token_pos=target_token_pos
        )
        
        return result
    
    def run_attention_head_intervention(self,
                                      original_program: str,
                                      counterfactual_program: str,
                                      target_token_pos: int,
                                      layer_idx: int,
                                      head_idx: int) -> InterventionResult:
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
        # Tokenize both programs
        original_tokens = self.tokenizer(original_program, return_tensors="pt")
        counterfactual_tokens = self.tokenizer(counterfactual_program, return_tensors="pt")
        
        # Get model predictions without intervention
        with self.model.trace(original_tokens) as tracer:
            original_logits = self.model.lm_head.output.save()
        
        # Run intervention on attention head output
        with self.model.trace(counterfactual_tokens) as tracer:
            # Access attention head output - this may need model-specific adjustment
            counterfactual_head_output = self.model.transformer.h[layer_idx].attn.output[0][:, target_token_pos, :].save()
        
        with self.model.trace(original_tokens) as tracer:
            # Patch counterfactual attention head output
            self.model.transformer.h[layer_idx].attn.output[0][:, target_token_pos, :] = counterfactual_head_output
            intervened_logits = self.model.lm_head.output.save()
        
        # Calculate intervention effects
        result = self._analyze_intervention_results(
            original_logits=original_logits.value,
            intervened_logits=intervened_logits.value,
            intervention_type="attention_head",
            layer_idx=layer_idx,
            head_idx=head_idx,
            target_token_pos=target_token_pos
        )
        
        return result
    
    def run_systematic_intervention(self,
                                  original_program: str,
                                  counterfactual_program: str,
                                  target_token_pos: int,
                                  max_layers: Optional[int] = None) -> List[InterventionResult]:
        """
        Run systematic intervention across all model layers.
        
        Args:
            original_program: The original program text
            counterfactual_program: The counterfactual program text
            target_token_pos: Token position to intervene on
            max_layers: Maximum number of layers to test (None = all layers)
            
        Returns:
            List of InterventionResults for each layer
        """
        num_layers = self.model.config.num_hidden_layers
        if max_layers is not None:
            num_layers = min(num_layers, max_layers)
        
        results = []
        
        for layer_idx in range(num_layers):
            print(f"  Testing layer {layer_idx}/{num_layers-1}")
            
            try:
                result = self.run_residual_stream_intervention(
                    original_program=original_program,
                    counterfactual_program=counterfactual_program,
                    target_token_pos=target_token_pos,
                    layer_idx=layer_idx
                )
                results.append(result)
                
            except Exception as e:
                print(f"    Error at layer {layer_idx}: {e}")
                # Create empty result for failed layer
                result = InterventionResult(
                    intervention_type="residual_stream",
                    layer_idx=layer_idx,
                    target_token_pos=target_token_pos,
                    logit_difference=0.0,
                    normalized_logit_difference=0.0,
                    success_rate=0.0
                )
                results.append(result)
        
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
    
    def calculate_normalized_logit_difference(self,
                                            original_logits: torch.Tensor,
                                            intervened_logits: torch.Tensor,
                                            original_token: int,
                                            counterfactual_token: int) -> float:
        """
        Calculate normalized logit difference for intervention effect.
        
        Args:
            original_logits: Original model logits [batch, seq, vocab]
            intervened_logits: Post-intervention logits [batch, seq, vocab]
            original_token: Original answer token ID
            counterfactual_token: Counterfactual answer token ID
            
        Returns:
            Normalized logit difference between -1.0 and 1.0
        """
        # Get final position logits
        orig_final = original_logits[:, -1, :]  # [batch, vocab]
        interv_final = intervened_logits[:, -1, :]  # [batch, vocab]
        
        # Calculate raw logit difference
        orig_diff = (orig_final[:, counterfactual_token] - orig_final[:, original_token]).mean()
        interv_diff = (interv_final[:, counterfactual_token] - interv_final[:, original_token]).mean()
        
        raw_difference = interv_diff - orig_diff
        
        # Normalize by maximum possible difference (approximate)
        max_logit_range = orig_final.max() - orig_final.min()
        if max_logit_range > 0:
            normalized = raw_difference / max_logit_range
            # Clamp to [-1, 1] range
            normalized = torch.clamp(normalized, -1.0, 1.0)
            return normalized.item()
        else:
            return 0.0
    
    def _analyze_intervention_results(self,
                                    original_logits: torch.Tensor,
                                    intervened_logits: torch.Tensor,
                                    intervention_type: str,
                                    layer_idx: int,
                                    target_token_pos: int,
                                    head_idx: Optional[int] = None) -> InterventionResult:
        """Analyze the results of an intervention experiment."""
        
        # Get top tokens
        original_top = original_logits[:, -1, :].argmax(dim=-1).item()
        intervened_top = intervened_logits[:, -1, :].argmax(dim=-1).item()
        
        # For now, we'll calculate a basic logit difference
        # In a real experiment, we'd need the actual target tokens
        logit_diff = (intervened_logits[:, -1, intervened_top] - 
                     original_logits[:, -1, original_top]).mean().item()
        
        return InterventionResult(
            intervention_type=intervention_type,
            layer_idx=layer_idx,
            head_idx=head_idx,
            target_token_pos=target_token_pos,
            logit_difference=logit_diff,
            normalized_logit_difference=logit_diff,  # Simplified for now
            success_rate=1.0 if intervened_top != original_top else 0.0,
            original_logits=original_logits,
            intervened_logits=intervened_logits,
            original_top_token=original_top,
            intervened_top_token=intervened_top
        )
    
from nnsight import LanguageModel
model = LanguageModel("Qwen/Qwen3-0.6B", device_map="auto")
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

print(model)
print(dir(model))
print(f"Number of layers: {model.model.config.num_hidden_layers}")
"""
residual stream patching pseudocode:

def residual_stream_patch(model, original_program, counterfactual_program, target_tokens):
    # Run forward pass on both programs
    original_activations = model.forward_with_cache(original_program)
    counterfactual_activations = model.forward_with_cache(counterfactual_program)
    
    # For each layer and target token position
    for layer in range(model.num_layers):
        for token_pos in target_tokens:
            # Replace residual stream at (layer, token_pos) with counterfactual value
            patched_activations = original_activations.copy()
            patched_activations[layer][token_pos] = counterfactual_activations[layer][token_pos]
            
            # Continue forward pass from this point
            logits = model.forward_from_layer(patched_activations, start_layer=layer)
            
            # Measure success: does model now predict new_root_value?
            success = (logits.argmax() == new_root_value)

attention head patching pseudocode:

def attention_head_patch(model, original_program, counterfactual_program, target_tokens):
    original_activations = model.forward_with_cache(original_program)
    counterfactual_activations = model.forward_with_cache(counterfactual_program)
    
    for layer in range(model.num_layers):
        for head in range(model.num_heads):
            for token_pos in target_tokens:
                # Replace specific head's contribution to residual stream
                patched_activations = original_activations.copy()
                head_output = counterfactual_activations[layer].attention_heads[head][token_pos]
                patched_activations[layer][token_pos] += head_output - original_activations[layer].attention_heads[head][token_pos]
                
                logits = model.forward_from_layer(patched_activations, start_layer=layer+1)
                success = (logits.argmax() == new_root_value)
"""

#%%