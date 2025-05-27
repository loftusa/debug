"""Debug experiment framework - simple and modular experiment tools."""

# Core components
from .core import ExperimentConfig, parse_integer, parse_boolean
from .runner import ExperimentRunner

# Import generators and prompts as modules
from . import generators
from . import prompts

# Quick setup helper
def quick_experiment(name, prompt_template, program_generator, **kwargs):
    """Create a quick experiment configuration."""
    return ExperimentConfig(
        name=name,
        prompt_template=prompt_template, 
        program_generator=program_generator,
        **kwargs
    )

# Version
__version__ = "0.1.0"

# Main exports
__all__ = [
    "ExperimentConfig",
    "ExperimentRunner", 
    "parse_integer",
    "parse_boolean",
    "generators",
    "prompts",
    "quick_experiment"
]
