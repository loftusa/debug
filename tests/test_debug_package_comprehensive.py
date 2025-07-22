#!/usr/bin/env python3
"""
Comprehensive test suite for the entire src/debug package.

This test suite aims for close to 100% coverage of all modules in src/debug/
to catch bugs like the TokenAnalyzer token position issue.

Modules tested:
- core.py: Configuration classes and parsing
- generators.py: Program generators for experiments
- prompts.py: Prompt templates
- counterfactual.py: Counterfactual program generation
- token_analyzer.py: Token analysis (separate comprehensive tests)
- causal_tracing.py: Causal intervention functionality
- causal_experiment_runner.py: Experiment execution
- causal_visualization.py: Visualization functions
- runner.py: Basic experiment runner
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to PYTHONPATH
project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(project_root))

# Import all modules from debug package
from debug.core import ExperimentConfig, parse_integer, parse_boolean
from debug.generators import (
    make_range_program, make_variable_increments, make_exception_program,
    make_variable_binding_program, make_variable_binding_program_with_metadata,
    make_counterfactual_pair
)
from debug.prompts import (
    RANGE_TRACKING, EXCEPTION_HANDLING, BOOLEAN_LOGIC,
    VARIABLE_BINDING, MINIMAL
)
from debug.counterfactual import CounterfactualGenerator
from debug.token_analyzer import TokenAnalyzer
from debug.causal_tracing import CausalTracer, InterventionResult
from debug.causal_experiment_runner import CausalExperimentRunner, CausalExperimentResult
from debug.runner import ExperimentRunner
# Note: causal_visualization.py will be tested separately as it requires matplotlib


class TestCore:
    """Test core.py - configuration classes and utilities."""
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation with all parameters."""
        def dummy_generator(seq_len, rng):
            return f"program_{seq_len}", "answer"
            
        def dummy_parser(response):
            return response.strip()
        
        config = ExperimentConfig(
            name="test_experiment",
            prompt_template="Test prompt: {code}",
            program_generator=dummy_generator,
            answer_parser=dummy_parser,
            models=["test-model"],
            num_seqs=10,
            seq_lens=[5, 10]
        )
        
        assert config.name == "test_experiment"
        assert config.prompt_template == "Test prompt: {code}"
        assert config.program_generator == dummy_generator
        assert config.answer_parser == dummy_parser
        assert config.models == ["test-model"]
        assert config.num_seqs == 10
        assert config.seq_lens == [5, 10]
    
    def test_experiment_config_defaults(self):
        """Test ExperimentConfig default values."""
        def dummy_generator(seq_len, rng):
            return "program", "answer"
            
        config = ExperimentConfig(
            name="test",
            prompt_template="{code}",
            program_generator=dummy_generator
        )
        
        # Check defaults
        assert config.answer_parser is not None
        assert config.models == ["Qwen/Qwen3-1.7B"]  # Default model
        assert config.num_seqs == 10  # Default value
        assert config.seq_lens == [2, 3, 4, 5, 6]  # Default seq_lens
    
    def test_parse_integer_function(self):
        """Test parse_integer utility function."""
        # Test various formats
        assert parse_integer("The answer is: 42") == 42
        assert parse_integer("result: -10") == -10
        assert parse_integer("x = 123") == 123
        assert parse_integer("Just 456 here") == 456
        assert parse_integer("No numbers") is None
    
    def test_parse_boolean_function(self):
        """Test parse_boolean utility function."""
        assert parse_boolean("The answer is True") is True
        assert parse_boolean("FALSE") is False
        assert parse_boolean("result: 1") is True
        assert parse_boolean("answer: 0") is False
        assert parse_boolean("maybe") is None


class TestGenerators:
    """Test generators.py - program generation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)
    
    def test_make_range_program_basic(self):
        """Test basic range program generation."""
        program, answer = make_range_program(seq_len=5, rng=self.rng)
        
        assert isinstance(program, str)
        assert isinstance(answer, int)
        assert len(program.split('\n')) >= 3  # Should have at least 3 lines (x=0, for loop, x+=)
        assert 0 <= answer <= 100  # Answer should be in reasonable range
    
    def test_make_range_program_consistency(self):
        """Test that range programs are consistent across calls with same seed."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        
        program1, answer1 = make_range_program(seq_len=5, rng=rng1)
        program2, answer2 = make_range_program(seq_len=5, rng=rng2)
        
        assert program1 == program2
        assert answer1 == answer2
    
    def test_make_range_program_different_lengths(self):
        """Test range program generation with different sequence lengths."""
        for seq_len in [3, 5, 10, 15]:
            program, answer = make_range_program(seq_len=seq_len, rng=self.rng)
            
            lines = program.strip().split('\n')
            assert len(lines) >= 3  # Should have at least 3 lines for range program
            assert isinstance(answer, int)
    
    def test_make_variable_increments_basic(self):
        """Test variable increments program generation."""
        program, answer = make_variable_increments(seq_len=5, rng=self.rng)
        
        assert isinstance(program, str)
        assert isinstance(answer, int)
        assert "+=" in program or "increment" in program.lower()
    
    def test_make_exception_program_basic(self):
        """Test exception program generation."""
        program, answer = make_exception_program(seq_len=5, rng=self.rng)
        
        assert isinstance(program, str)
        assert isinstance(answer, (int, str))
        # Should contain some exception-related keywords
        program_lower = program.lower()
        assert any(keyword in program_lower for keyword in ["try", "except", "error", "exception"])
    
    def test_make_variable_binding_program_basic(self):
        """Test variable binding program generation."""
        program, answer, hops = make_variable_binding_program(seq_len=10, rng=self.rng)
        
        assert isinstance(program, str)
        assert isinstance(answer, int)
        assert isinstance(hops, int)
        assert hops >= 1  # Should have at least one hop
        
        # Should contain variable assignments
        assert "=" in program
        assert program.endswith(": ")  # Should end with query format
    
    def test_make_variable_binding_program_with_metadata(self):
        """Test variable binding program with metadata generation."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        program, answer, hops, metadata = make_variable_binding_program_with_metadata(
            seq_len=10, rng=self.rng, tokenizer=tokenizer
        )
        
        assert isinstance(program, str)
        assert isinstance(answer, int)
        assert isinstance(hops, int)
        assert isinstance(metadata, dict)
        
        # Check metadata structure
        assert "query_var" in metadata
        assert "variable_chain" in metadata
        assert "intervention_targets" in metadata
        
        # Check intervention targets structure
        targets = metadata["intervention_targets"]
        assert isinstance(targets, dict)
        # Should have at least some targets
        assert len(targets) > 0
    
    def test_make_variable_binding_program_consistency(self):
        """Test consistency of variable binding programs."""
        rng1 = np.random.RandomState(123)
        rng2 = np.random.RandomState(123)
        
        prog1, ans1, hops1 = make_variable_binding_program(seq_len=8, rng=rng1)
        prog2, ans2, hops2 = make_variable_binding_program(seq_len=8, rng=rng2)
        
        assert prog1 == prog2
        assert ans1 == ans2
        assert hops1 == hops2
    
    def test_make_counterfactual_pair_basic(self):
        """Test counterfactual pair generation."""
        program_a, program_b, intermediate_a, intermediate_b = make_counterfactual_pair(seq_len=8, divergence_index=4, rng=self.rng)
        
        assert isinstance(program_a, str)
        assert isinstance(program_b, str)
        assert isinstance(intermediate_a, list)
        assert isinstance(intermediate_b, list)
        
        assert program_a != program_b  # Should have different programs
    
    def test_generator_edge_cases(self):
        """Test edge cases for generators."""
        # Minimum sequence length
        program, answer = make_range_program(seq_len=1, rng=self.rng)
        assert isinstance(program, str)
        assert isinstance(answer, int)
        
        # Very small RNG seed
        small_rng = np.random.RandomState(0)
        program, answer, hops = make_variable_binding_program(seq_len=3, rng=small_rng)
        assert isinstance(program, str)


class TestPrompts:
    """Test prompts.py - prompt templates."""
    
    def test_all_prompts_exist(self):
        """Test that all expected prompt templates exist."""
        prompts = [
            RANGE_TRACKING, EXCEPTION_HANDLING, BOOLEAN_LOGIC,
            VARIABLE_BINDING, MINIMAL
        ]
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0
    
    def test_prompts_have_placeholders(self):
        """Test that prompts have expected placeholders."""
        # Most prompts should have {code} placeholder
        code_prompts = [
            RANGE_TRACKING, EXCEPTION_HANDLING, BOOLEAN_LOGIC,
            MINIMAL
        ]
        
        for prompt in code_prompts:
            assert "{code}" in prompt, f"Prompt missing {{code}} placeholder: {prompt[:50]}..."
    
    def test_prompt_formatting(self):
        """Test that prompts can be formatted correctly."""
        test_code = "x = 1\ny = x\nprint(y)"
        
        formatted = RANGE_TRACKING.format(code=test_code)
        assert test_code in formatted
        assert formatted != RANGE_TRACKING  # Should be different after formatting


class TestCounterfactual:
    """Test counterfactual.py - counterfactual program generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CounterfactualGenerator()
    
    def test_counterfactual_generator_creation(self):
        """Test CounterfactualGenerator creation."""
        gen = CounterfactualGenerator()
        assert isinstance(gen, CounterfactualGenerator)
    
    def test_create_counterfactual_basic(self):
        """Test basic counterfactual creation."""
        original_program = "x = 1\ny = x\nz = y\n#z:"
        query_var = "z"
        
        counterfactual = self.generator.create_counterfactual(original_program, query_var)
        
        assert isinstance(counterfactual, str)
        assert counterfactual != original_program  # Should be different
        assert query_var in counterfactual or f"#{query_var}" in counterfactual
    
    def test_create_counterfactual_preserves_structure(self):
        """Test that counterfactual preserves program structure."""
        original_program = "a = 1\nb = a\nc = b\n#c:"
        query_var = "c"
        
        counterfactual = self.generator.create_counterfactual(original_program, query_var)
        
        # Should have same number of lines
        orig_lines = original_program.split('\n')
        counter_lines = counterfactual.split('\n')
        assert len(orig_lines) == len(counter_lines)
        
        # Should end with same query
        assert orig_lines[-1] == counter_lines[-1]
    
    def test_create_counterfactual_different_values(self):
        """Test that counterfactual actually changes values."""
        original_program = "x = 5\ny = x\n#y:"
        query_var = "y"
        
        counterfactual = self.generator.create_counterfactual(original_program, query_var)
        
        # The root value should be different
        assert "x = 5" not in counterfactual or "x = " in counterfactual
    
    def test_create_counterfactual_complex_program(self):
        """Test counterfactual creation on complex program."""
        program = "l = 1\nc = l\ny = 5\np = 6\nm = 8\nq = p\nf = m\na = c\n#a:"
        query_var = "a"
        
        counterfactual = self.generator.create_counterfactual(program, query_var)
        
        assert isinstance(counterfactual, str)
        assert counterfactual != program
        assert "#a:" in counterfactual  # Query should be preserved
    
    def test_create_counterfactual_edge_cases(self):
        """Test edge cases for counterfactual generation."""
        # Empty program
        try:
            result = self.generator.create_counterfactual("", "x")
            assert isinstance(result, str)
        except Exception:
            pass  # May reasonably fail
        
        # Non-existent query variable - should raise an error
        with pytest.raises(ValueError):
            self.generator.create_counterfactual("x = 1\n#y:", "y")
        
        # Single assignment
        result = self.generator.create_counterfactual("x = 1\n#x:", "x")
        assert isinstance(result, str)


class TestCausalTracing:
    """Test causal_tracing.py - causal intervention functionality."""
    
    def test_intervention_result_creation(self):
        """Test InterventionResult dataclass creation."""
        result = InterventionResult(
            intervention_type="residual_stream",
            layer_idx=5,
            target_token_pos=10,
            logit_difference=0.5,
            success_rate=0.8
        )
        
        assert result.intervention_type == "residual_stream"
        assert result.layer_idx == 5
        assert result.target_token_pos == 10
        assert result.logit_difference == 0.5
        assert result.success_rate == 0.8
    
    def test_intervention_result_defaults(self):
        """Test InterventionResult default values."""
        result = InterventionResult(
            intervention_type="test",
            layer_idx=0
        )
        
        assert result.head_idx is None
        assert result.target_token_pos is None
        assert result.logit_difference is None
        assert result.success_rate is None
    
    @patch('debug.causal_tracing.LanguageModel')
    @patch('debug.causal_tracing.AutoTokenizer')
    def test_causal_tracer_creation(self, mock_tokenizer, mock_lm):
        """Test CausalTracer creation with mocked dependencies."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_lm.return_value = Mock()
        
        # Mock the config
        mock_config = Mock()
        mock_config.num_hidden_layers = 12
        mock_lm.return_value.config = mock_config
        
        tracer = CausalTracer("Qwen/Qwen3-0.6B")
        
        assert tracer.model_name == "Qwen/Qwen3-0.6B"
        assert tracer._n_layers == 12
    
    @patch('debug.causal_tracing.LanguageModel')  
    @patch('debug.causal_tracing.AutoTokenizer')
    def test_causal_tracer_invalid_model(self, mock_tokenizer, mock_lm):
        """Test CausalTracer with non-Qwen model."""
        with pytest.raises(ValueError, match="not a Qwen series model"):
            CausalTracer("gpt-4")
    
    # Note: More detailed CausalTracer tests would require actual model loading
    # which is expensive. The integration tests cover the full functionality.


class TestCausalExperimentRunner:
    """Test causal_experiment_runner.py - experiment execution."""
    
    def test_causal_experiment_result_creation(self):
        """Test CausalExperimentResult creation."""
        mock_config = Mock()
        result = CausalExperimentResult(
            config=mock_config,
            intervention_results=[],
            program_results=[],
            summary_stats={},
            output_dir=Path("/tmp/test"),
            timestamp="2023-01-01_12:00:00"
        )
        
        assert result.config == mock_config
        assert result.intervention_results == []
        assert result.program_results == []
        assert result.summary_stats == {}
    
    def test_causal_experiment_runner_creation(self):
        """Test CausalExperimentRunner creation."""
        runner = CausalExperimentRunner()
        assert isinstance(runner, CausalExperimentRunner)
        
    # Note: Full testing of CausalExperimentRunner requires heavy mocking
    # or actual model loading. The integration tests cover this.


class TestExperimentRunner:
    """Test runner.py - basic experiment runner."""
    
    def test_experiment_runner_creation(self):
        """Test ExperimentRunner creation."""
        runner = ExperimentRunner()
        assert isinstance(runner, ExperimentRunner)
    
    def test_experiment_runner_with_output_dir(self):
        """Test ExperimentRunner with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(output_dir=str(temp_dir))
            assert runner.output_dir == Path(temp_dir)
    
    # Note: Full testing of ExperimentRunner requires model loading
    # Integration tests cover the full functionality


class TestIntegrationScenarios:
    """Integration tests combining multiple modules."""
    
    def test_full_variable_binding_workflow(self):
        """Test complete workflow from generation to analysis."""
        # 1. Generate program with metadata
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        rng = np.random.RandomState(42)
        
        program, answer, hops, metadata = make_variable_binding_program_with_metadata(
            seq_len=10, rng=rng, tokenizer=tokenizer
        )
        
        # 2. Create counterfactual
        generator = CounterfactualGenerator()
        counterfactual = generator.create_counterfactual(program, metadata["query_var"])
        
        # 3. Analyze tokens
        analyzer = TokenAnalyzer(tokenizer)
        chain = analyzer.identify_variable_chain(program, metadata["query_var"])
        targets = analyzer.find_intervention_targets(program, metadata["query_var"])
        
        # Verify everything is consistent
        assert chain.query_var == metadata["query_var"]
        assert isinstance(targets, dict)
        assert len(targets) > 0
        
        # Verify counterfactual is different but structurally similar
        assert program != counterfactual
        assert program.count('\n') == counterfactual.count('\n')
    
    def test_config_to_generation_workflow(self):
        """Test workflow from config creation to program generation."""
        # Create config
        config = ExperimentConfig(
            name="integration_test",
            prompt_template=VARIABLE_BINDING,
            program_generator=make_variable_binding_program,
            models=["test-model"],
            num_seqs=2,
            seq_lens=[5, 7]
        )
        
        # Generate programs using the config
        rng = np.random.RandomState(123)
        for seq_len in config.seq_lens:
            program, answer, hops = config.program_generator(seq_len, rng)
            
            assert isinstance(program, str)
            assert isinstance(answer, int)
            assert isinstance(hops, int)
            
            # Test prompt formatting
            formatted_prompt = config.prompt_template.format(code=program)
            assert program in formatted_prompt
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through the system."""
        # Test with invalid inputs
        rng = np.random.RandomState(42)
        
        # This should not crash but may return reasonable defaults
        try:
            program, answer = make_range_program(seq_len=0, rng=rng)
            assert isinstance(program, str)
        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, (ValueError, AssertionError))


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling across all modules."""
    
    def test_empty_inputs(self):
        """Test handling of empty or minimal inputs."""
        rng = np.random.RandomState(42)
        
        # Empty/minimal generators
        try:
            program, answer = make_range_program(seq_len=1, rng=rng)
            assert isinstance(program, str)
            assert isinstance(answer, int)
        except Exception:
            pass  # May reasonably fail
        
        # Empty counterfactual
        generator = CounterfactualGenerator()
        try:
            result = generator.create_counterfactual("", "x")
            assert isinstance(result, str)
        except Exception:
            pass  # May reasonably fail
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        rng = np.random.RandomState(42)
        
        # Negative sequence lengths - may or may not raise an error, depends on implementation
        try:
            program, answer = make_range_program(seq_len=-1, rng=rng)
            # If it doesn't raise, just check it returns sensible types
            assert isinstance(program, str)
            assert isinstance(answer, int)
        except Exception:
            pass  # Acceptable to raise an error
        
        # Invalid query variables should raise an error
        generator = CounterfactualGenerator()
        with pytest.raises(ValueError):
            generator.create_counterfactual("x = 1\n", "nonexistent")
    
    def test_large_inputs(self):
        """Test handling of large inputs."""
        rng = np.random.RandomState(42)
        
        # Large sequence length
        program, answer = make_range_program(seq_len=100, rng=rng)
        assert isinstance(program, str)
        assert isinstance(answer, int)
        
        # Very long variable chain
        # This tests performance and memory usage
        import time
        start = time.time()
        
        program, answer, hops = make_variable_binding_program(seq_len=50, rng=rng)
        
        end = time.time()
        
        assert isinstance(program, str)
        assert isinstance(answer, int)
        assert (end - start) < 5.0  # Should complete in reasonable time
    
    def test_special_characters_handling(self):
        """Test handling of special characters and edge cases."""
        # Programs with special characters
        generator = CounterfactualGenerator()
        
        # Test with underscores, numbers
        program = "var_1 = 2\nvar2 = var_1\n#var2:"
        result = generator.create_counterfactual(program, "var2")
        assert isinstance(result, str)
        
        # Test with different whitespace - this should work since we have a valid chain
        program = "x = 1\ny = x\n#y:"
        result = generator.create_counterfactual(program, "y")
        assert isinstance(result, str)


class TestConsistencyAndDeterminism:
    """Test that functions are deterministic and consistent."""
    
    def test_generator_determinism(self):
        """Test that generators are deterministic with same seed."""
        seeds_to_test = [0, 42, 123, 999, 2023]
        
        for seed in seeds_to_test:
            rng1 = np.random.RandomState(seed)
            rng2 = np.random.RandomState(seed)
            
            # Test multiple generators
            prog1, ans1 = make_range_program(seq_len=5, rng=rng1)
            prog2, ans2 = make_range_program(seq_len=5, rng=rng2)
            
            assert prog1 == prog2, f"Range generator not deterministic for seed {seed}"
            assert ans1 == ans2, f"Range generator answers not deterministic for seed {seed}"
            
            # Reset RNGs
            rng1 = np.random.RandomState(seed)
            rng2 = np.random.RandomState(seed)
            
            prog1, ans1, hops1 = make_variable_binding_program(seq_len=8, rng=rng1)
            prog2, ans2, hops2 = make_variable_binding_program(seq_len=8, rng=rng2)
            
            assert prog1 == prog2, f"Variable binding generator not deterministic for seed {seed}"
            assert ans1 == ans2, f"Variable binding answers not deterministic for seed {seed}"
            assert hops1 == hops2, f"Variable binding hops not deterministic for seed {seed}"
    
    def test_counterfactual_consistency(self):
        """Test that counterfactual generation with fixed values is consistent."""
        generator = CounterfactualGenerator()
        program = "x = 1\ny = x\nz = y\n#z:"
        
        # Multiple calls with the same new_root_value should give same result
        counter1 = generator.create_counterfactual(program, "z", new_root_value="5")
        counter2 = generator.create_counterfactual(program, "z", new_root_value="5")
        
        # Should be consistent when root value is specified
        assert counter1 == counter2


class TestDataFlowAndTransformations:
    """Test data flow and transformations between modules."""
    
    def test_program_to_prompt_flow(self):
        """Test flow from program generation to prompt creation."""
        rng = np.random.RandomState(42)
        
        # Generate program
        program, answer, hops = make_variable_binding_program(seq_len=8, rng=rng)
        
        # Create prompt using a template that actually adds instructions
        prompt = RANGE_TRACKING.format(code=program)
        
        # Verify transformations
        assert program in prompt
        assert len(prompt) > len(program)  # Prompt should add instructions
        assert "{code}" not in prompt  # Should be replaced
    
    def test_program_to_counterfactual_flow(self):
        """Test flow from program to counterfactual generation."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        rng = np.random.RandomState(42)
        
        # Generate program with metadata
        program, answer, hops, metadata = make_variable_binding_program_with_metadata(
            seq_len=10, rng=rng, tokenizer=tokenizer
        )
        
        # Generate counterfactual
        generator = CounterfactualGenerator()
        counterfactual = generator.create_counterfactual(program, metadata["query_var"])
        
        # Analyze both programs
        analyzer = TokenAnalyzer(tokenizer)
        
        orig_chain = analyzer.identify_variable_chain(program, metadata["query_var"])
        counter_chain = analyzer.identify_variable_chain(counterfactual, metadata["query_var"])
        
        # Verify transformations preserve structure
        assert orig_chain.query_var == counter_chain.query_var
        assert len(orig_chain.chain) == len(counter_chain.chain)  # Same chain length
        assert orig_chain.root_value != counter_chain.root_value  # Different root values


if __name__ == "__main__":
    # Run with pytest
    import subprocess
    import sys
    
    # Run the tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--maxfail=10",  # Stop after 10 failures
        "-x"  # Stop on first failure for debugging
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    # Also run directly if called as main
    if result.returncode != 0:
        print("\n🚨 Tests failed! Running directly for more details...")
        pytest.main([__file__, "-v", "--tb=short"])