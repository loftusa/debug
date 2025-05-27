"""Core experiment framework for debug experiments."""

import re
from typing import List, Optional, Callable, Any, Tuple
import numpy as np


def parse_integer(text: str) -> Optional[int]:
    """Parse integer from model response."""
    cleaned = text.replace(",", "")
    # Try common patterns
    for pattern in [r"is:\s*(-?\d+)", r"answer:\s*(-?\d+)", r"result:\s*(-?\d+)", r"=\s*(-?\d+)"]:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    # Fallback: first integer
    match = re.search(r"-?\d+", cleaned)
    return int(match.group()) if match else None


def parse_boolean(text: str) -> Optional[bool]:
    """Parse boolean from model response."""
    text_lower = text.lower()
    if "true" in text_lower:
        return True
    elif "false" in text_lower:
        return False
    # Try parsing as 0/1
    integer_result = parse_integer(text)
    if integer_result == 0:
        return False
    elif integer_result == 1:
        return True
    return None


class ExperimentConfig:
    """Simple experiment configuration.
    
    Args:
        name: Experiment name
        prompt_template: Template with {code} placeholder
        program_generator: Function(seq_len, rng) -> (program, expected_answer)
        answer_parser: Function to parse model responses
        models: List of model IDs to test
        num_seqs: Number of test sequences per configuration
        seq_lens: List of sequence lengths to test
    """
    
    def __init__(self, 
                 name: str,
                 prompt_template: str,
                 program_generator: Callable[[int, np.random.RandomState], Tuple[str, Any]],
                 answer_parser: Callable[[str], Optional[Any]] = parse_integer,
                 models: List[str] = None,
                 num_seqs: int = 10,
                 seq_lens: List[int] = None):
        
        self.name = name
        self.prompt_template = prompt_template
        self.program_generator = program_generator
        self.answer_parser = answer_parser
        self.models = models or ["Qwen/Qwen3-1.7B"]
        self.num_seqs = num_seqs
        self.seq_lens = seq_lens or [2, 3, 4, 5, 6]
    
    def __repr__(self) -> str:
        return f"ExperimentConfig('{self.name}', {len(self.models)} models, seq_lens={self.seq_lens})" 