#!/usr/bin/env python3
"""
Comprehensive test suite for TokenAnalyzer to ensure correct token position identification.

This test suite aims for close to 100% coverage of TokenAnalyzer functionality,
with particular focus on edge cases that could cause incorrect intervention targeting.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

# Add src to PYTHONPATH
project_root = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(project_root))

from debug.token_analyzer import TokenAnalyzer, VariableChain, InterventionTargets
from transformers import AutoTokenizer


class TestTokenAnalyzer:
    """Comprehensive tests for TokenAnalyzer class."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a test tokenizer."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    @pytest.fixture
    def analyzer(self, tokenizer):
        """Create TokenAnalyzer with tokenizer."""
        return TokenAnalyzer(tokenizer)
    
    @pytest.fixture
    def analyzer_no_tokenizer(self):
        """Create TokenAnalyzer without tokenizer."""
        return TokenAnalyzer(None)

    # ========================================================================
    # Test Variable Chain Identification
    # ========================================================================
    
    def test_simple_chain_identification(self, analyzer):
        """Test basic variable chain identification."""
        program = "x = 1\ny = x\nz = y\n#z:"
        chain = analyzer.identify_variable_chain(program, "z")
        
        assert chain.query_var == "z"
        assert chain.chain == [("z", "y"), ("y", "x"), ("x", "1")]
        assert chain.root_value == "1"
        assert chain.referential_depth == 3
        assert not chain.is_circular
    
    def test_direct_literal_assignment(self, analyzer):
        """Test variable directly assigned to literal."""
        program = "x = 42\n#x:"
        chain = analyzer.identify_variable_chain(program, "x")
        
        assert chain.query_var == "x"
        assert chain.chain == [("x", "42")]
        assert chain.root_value == "42"
        assert chain.referential_depth == 1
        assert not chain.is_circular
    
    def test_circular_reference_detection(self, analyzer):
        """Test detection of circular variable references."""
        program = "x = y\ny = z\nz = x\n#x:"
        chain = analyzer.identify_variable_chain(program, "x")
        
        assert chain.query_var == "x"
        assert chain.is_circular
        assert chain.root_value is None
    
    def test_undefined_variable_chain(self, analyzer):
        """Test handling of undefined variables in chain."""
        program = "x = undefined_var\n#x:"
        chain = analyzer.identify_variable_chain(program, "x")
        
        assert chain.query_var == "x"
        assert chain.chain == [("x", "undefined_var")]
        assert chain.root_value is None
        assert chain.referential_depth == 1
    
    def test_nonexistent_query_variable(self, analyzer):
        """Test querying a variable that doesn't exist."""
        program = "x = 1\n#y:"
        chain = analyzer.identify_variable_chain(program, "y")
        
        assert chain.query_var == "y"
        assert chain.chain == []
        assert chain.root_value is None
        assert chain.referential_depth == 0
    
    def test_complex_chain_with_distractors(self, analyzer):
        """Test the exact case from the bug report."""
        program = "l = 1\nc = l\ny = 5\np = 6\nm = 8\nq = p\nf = m\na = c\nj = 9\nv = 0\nx = f\no = q\nr = a\nw = 5\ng = r\nb = r\ni = r\n#a:"
        chain = analyzer.identify_variable_chain(program, "a")
        
        assert chain.query_var == "a"
        assert chain.chain == [("a", "c"), ("c", "l"), ("l", "1")]
        assert chain.root_value == "1"
        assert chain.referential_depth == 3
        assert not chain.is_circular

    # ========================================================================
    # Test Assignment Parsing
    # ========================================================================
    
    def test_parse_basic_assignments(self, analyzer):
        """Test parsing of basic assignments."""
        program = "x = 1\ny = 2\nz = x"
        assignments = analyzer._parse_assignments(program)
        
        expected = {"x": "1", "y": "2", "z": "x"}
        assert assignments == expected
    
    def test_parse_assignments_with_comments(self, analyzer):
        """Test parsing assignments with inline comments."""
        program = "x = 1  # comment\ny = 2 # another comment\nz = x"
        assignments = analyzer._parse_assignments(program)
        
        expected = {"x": "1", "y": "2", "z": "x"}
        assert assignments == expected
    
    def test_parse_assignments_with_whitespace(self, analyzer):
        """Test parsing assignments with various whitespace."""
        program = "  x   =   1  \n\ty=2\n z = 3 "
        assignments = analyzer._parse_assignments(program)
        
        expected = {"x": "1", "y": "2", "z": "3"}
        assert assignments == expected
    
    def test_parse_assignments_ignore_comments(self, analyzer):
        """Test that comment lines are ignored."""
        program = "x = 1\n# this is a comment\ny = 2\n#z: query"
        assignments = analyzer._parse_assignments(program)
        
        expected = {"x": "1", "y": "2"}
        assert assignments == expected
    
    def test_parse_assignments_ignore_invalid_lines(self, analyzer):
        """Test that invalid assignment lines are ignored."""
        program = "x = 1\ninvalid line\ny = 2\nx == 3\ny += 4"
        assignments = analyzer._parse_assignments(program)
        
        expected = {"x": "1", "y": "2"}
        assert assignments == expected

    # ========================================================================
    # Test Literal Detection
    # ========================================================================
    
    def test_is_literal_positive_integers(self, analyzer):
        """Test literal detection for positive integers."""
        assert analyzer._is_literal("0")
        assert analyzer._is_literal("1")
        assert analyzer._is_literal("42")
        assert analyzer._is_literal("999")
    
    def test_is_literal_negative_integers(self, analyzer):
        """Test literal detection for negative integers."""
        assert analyzer._is_literal("-1")
        assert analyzer._is_literal("-42")
    
    def test_is_literal_variables(self, analyzer):
        """Test that variables are not detected as literals."""
        assert not analyzer._is_literal("x")
        assert not analyzer._is_literal("var")
        assert not analyzer._is_literal("variable_name")
    
    def test_is_literal_expressions(self, analyzer):
        """Test that expressions are not detected as literals."""
        assert not analyzer._is_literal("x + 1")
        assert not analyzer._is_literal("1.5")  # Float not supported
        assert not analyzer._is_literal("'string'")

    # ========================================================================
    # Test Token Position Finding (Core Bug Fix Tests)
    # ========================================================================
    
    def test_find_token_position_simple_case(self, analyzer):
        """Test finding token position in simple assignment."""
        program = "x = 1\n"
        tokens = analyzer.tokenizer.tokenize(program)
        assignments = analyzer._parse_assignments(program)
        
        position = analyzer._find_token_position(tokens, "1", "x", assignments)
        
        assert position is not None
        # The position should correspond to the "1" token
        assert tokens[position].replace('Ġ', '').strip() == "1"
    
    def test_find_token_position_bug_case(self, analyzer):
        """Test the specific bug case: finding 'c' in 'a = c' vs 'c = l'."""
        program = "l = 1\nc = l\na = c\n"
        tokens = analyzer.tokenizer.tokenize(program)
        assignments = analyzer._parse_assignments(program)
        
        # This should find the 'c' in 'a = c', NOT the 'c' in 'c = l'
        position = analyzer._find_token_position(tokens, "c", "a", assignments)
        
        assert position is not None
        
        # Verify it's the correct 'c' by checking context
        # The 'c' should come after 'a' and '='
        found_token = tokens[position].replace('Ġ', '').strip()
        assert found_token == "c"
        
        # Additional verification: look backwards for 'a' and '='
        found_a = False
        found_equals = False
        for i in range(max(0, position-5), position):
            clean_token = tokens[i].replace('Ġ', '').strip()
            if clean_token == "a":
                found_a = True
            elif clean_token == "=":
                found_equals = True
        
        assert found_a and found_equals, f"Token at position {position} is not in 'a = c' context"
    
    def test_find_token_position_multiple_occurrences(self, analyzer):
        """Test finding correct token when target appears multiple times."""
        program = "x = y\ny = x\nz = x\n"
        tokens = analyzer.tokenizer.tokenize(program)
        assignments = analyzer._parse_assignments(program)
        
        # Find 'x' in 'z = x' (should be the last occurrence)
        position = analyzer._find_token_position(tokens, "x", "z", assignments)
        
        assert position is not None
        assert tokens[position].replace('Ġ', '').strip() == "x"
        
        # Verify it's in the context of 'z = x'
        found_z = False
        found_equals = False
        for i in range(max(0, position-5), position):
            clean_token = tokens[i].replace('Ġ', '').strip()
            if clean_token == "z":
                found_z = True
            elif clean_token == "=":
                found_equals = True
        
        assert found_z and found_equals
    
    def test_find_token_position_nonexistent(self, analyzer):
        """Test finding token that doesn't exist in assignment."""
        program = "x = 1\ny = 2\n"
        tokens = analyzer.tokenizer.tokenize(program)
        assignments = analyzer._parse_assignments(program)
        
        # Try to find 'z' in 'x = z' (but assignment is 'x = 1')
        position = analyzer._find_token_position(tokens, "z", "x", assignments)
        
        assert position is None
    
    def test_find_token_position_invalid_assignment(self, analyzer):
        """Test finding token for non-existent assignment."""
        program = "x = 1\ny = 2\n"
        tokens = analyzer.tokenizer.tokenize(program)
        assignments = analyzer._parse_assignments(program)
        
        # Try to find token for assignment that doesn't exist
        position = analyzer._find_token_position(tokens, "1", "z", assignments)
        
        assert position is None

    # ========================================================================
    # Test Query Position Finding
    # ========================================================================
    
    def test_find_query_positions_basic(self, analyzer):
        """Test finding query variable and prediction token positions."""
        program = "x = 1\n#x: "
        tokens = analyzer.tokenizer.tokenize(program)
        
        positions = analyzer._find_query_positions(tokens, "x")
        
        assert "prediction_token_pos" in positions
        assert positions["prediction_token_pos"] == len(tokens) - 1
        
        assert "query_var" in positions
        # The query_var should be the token containing the variable name
        query_pos = positions["query_var"]
        assert "#x" in tokens[query_pos] or "x" in tokens[query_pos]
    
    def test_find_query_positions_empty_tokens(self, analyzer):
        """Test handling of empty token list."""
        positions = analyzer._find_query_positions([], "x")
        
        # Should handle empty tokens gracefully
        assert isinstance(positions, dict)

    # ========================================================================
    # Test Full Integration (find_intervention_targets)
    # ========================================================================
    
    def test_find_intervention_targets_no_tokenizer(self, analyzer_no_tokenizer):
        """Test that method requires tokenizer."""
        program = "x = 1\n#x:"
        
        with pytest.raises(ValueError, match="Tokenizer required"):
            analyzer_no_tokenizer.find_intervention_targets(program, "x")
    
    def test_find_intervention_targets_simple_case(self, analyzer):
        """Test full intervention target finding for simple case."""
        program = "x = 1\ny = x\n#y:"
        
        targets = analyzer.find_intervention_targets(program, "y")
        
        assert "ref_depth_1_rhs" in targets  # Should find "x" in "y = x"
        assert "ref_depth_2_rhs" in targets  # Should find "1" in "x = 1"
        assert "query_var" in targets
        assert "prediction_token_pos" in targets
        
        # Verify the positions correspond to correct tokens
        tokens = analyzer.tokenizer.tokenize(program)
        
        # ref_depth_1_rhs should be "1" in "x = 1" (root value)
        pos1 = targets["ref_depth_1_rhs"]
        assert tokens[pos1].replace('Ġ', '').strip() == "1"
        
        # ref_depth_2_rhs should be "x" in "y = x" (first hop from root)
        pos2 = targets["ref_depth_2_rhs"]
        assert tokens[pos2].replace('Ġ', '').strip() == "x"
    
    def test_find_intervention_targets_bug_case(self, analyzer):
        """Test the specific bug case from the original issue."""
        program = "l = 1\nc = l\ny = 5\np = 6\nm = 8\nq = p\nf = m\na = c\nj = 9\nv = 0\nx = f\no = q\nr = a\nw = 5\ng = r\nb = r\ni = r\n#a: "
        
        targets = analyzer.find_intervention_targets(program, "a")
        tokens = analyzer.tokenizer.tokenize(program)
        
        assert "ref_depth_1_rhs" in targets  # Should find "1" in "l = 1" (root value)
        assert "ref_depth_2_rhs" in targets  # Should find "l" in "c = l" (first hop from root)
        assert "ref_depth_3_rhs" in targets  # Should find "c" in "a = c" (second hop from root)
        
        # Verify correct tokens are found
        pos1 = targets["ref_depth_1_rhs"]  # "1" in "l = 1" (root value)
        pos2 = targets["ref_depth_2_rhs"]  # "l" in "c = l" (first hop from root)
        pos3 = targets["ref_depth_3_rhs"]  # "c" in "a = c" (second hop from root)
        
        assert tokens[pos1].replace('Ġ', '').strip() == "1"
        assert tokens[pos2].replace('Ġ', '').strip() == "l" 
        assert tokens[pos3].replace('Ġ', '').strip() == "c"
        
        # Verify these are NOT the wrong positions
        # pos1 should NOT be position 5 (which was the bug)
        assert pos1 != 5, f"ref_depth_1_rhs should not be position 5 (bug case)"
    
    def test_find_intervention_targets_circular_chain(self, analyzer):
        """Test handling of circular variable chains."""
        program = "x = y\ny = z\nz = x\n#x:"
        
        targets = analyzer.find_intervention_targets(program, "x")
        
        # Should still return positions for existing assignments
        assert isinstance(targets, dict)
        # But may have fewer ref_depth entries due to circularity
    
    def test_find_intervention_targets_single_assignment(self, analyzer):
        """Test case with single direct assignment."""
        program = "x = 42\n#x:"
        
        targets = analyzer.find_intervention_targets(program, "x")
        
        assert "ref_depth_1_rhs" in targets
        assert "query_var" in targets
        assert "prediction_token_pos" in targets
        
        tokens = analyzer.tokenizer.tokenize(program)
        pos = targets["ref_depth_1_rhs"]
        assert tokens[pos].replace('Ġ', '').strip() == "42"

    # ========================================================================
    # Test Edge Cases and Error Handling
    # ========================================================================
    
    def test_empty_program(self, analyzer):
        """Test handling of empty program."""
        program = ""
        chain = analyzer.identify_variable_chain(program, "x")
        
        assert chain.query_var == "x"
        assert chain.chain == []
        assert chain.root_value is None
    
    def test_program_with_only_comments(self, analyzer):
        """Test handling of program with only comments."""
        program = "# This is a comment\n# Another comment\n#x:"
        chain = analyzer.identify_variable_chain(program, "x")
        
        assert chain.query_var == "x"
        assert chain.chain == []
    
    def test_program_with_special_characters(self, analyzer):
        """Test handling of programs with special characters."""
        program = "x_var = 1\ny2 = x_var\n#y2:"
        chain = analyzer.identify_variable_chain(program, "y2")
        
        assert chain.query_var == "y2"
        assert chain.chain == [("y2", "x_var"), ("x_var", "1")]
        assert chain.root_value == "1"
    
    def test_very_long_chain(self, analyzer):
        """Test handling of very long variable chains."""
        # Create a long chain: a -> b -> c -> ... -> z -> 1
        assignments = []
        prev_var = "1"
        for i in range(25, 0, -1):  # z to a
            var = chr(ord('a') + i - 1)  # Convert to letter
            assignments.append(f"{var} = {prev_var}")
            prev_var = var
        
        program = "\n".join(assignments) + "\n#z:"
        chain = analyzer.identify_variable_chain(program, "z")
        
        assert chain.query_var == "z"
        assert chain.root_value == "1"
        assert chain.referential_depth == 25
        assert not chain.is_circular
    
    def test_tokenizer_edge_cases(self, analyzer):
        """Test edge cases specific to tokenizer behavior."""
        # Test with program that might have unusual tokenization
        program = "  x  =  1  \n  #x:  "
        
        targets = analyzer.find_intervention_targets(program, "x")
        
        # Should still work despite extra whitespace
        assert "ref_depth_1_rhs" in targets
        assert "query_var" in targets


class TestVariableChain:
    """Test the VariableChain dataclass."""
    
    def test_variable_chain_creation(self):
        """Test creating VariableChain instances."""
        chain = VariableChain(
            query_var="x",
            chain=[("x", "y"), ("y", "1")],
            root_value="1",
            referential_depth=2,
            is_circular=False
        )
        
        assert chain.query_var == "x"
        assert chain.chain == [("x", "y"), ("y", "1")]
        assert chain.root_value == "1"
        assert chain.referential_depth == 2
        assert not chain.is_circular
    
    def test_variable_chain_defaults(self):
        """Test VariableChain default values."""
        chain = VariableChain(
            query_var="x",
            chain=[],
            root_value=None,
            referential_depth=0
        )
        
        assert not chain.is_circular  # Default should be False


class TestInterventionTargets:
    """Test the InterventionTargets dataclass."""
    
    def test_intervention_targets_creation(self):
        """Test creating InterventionTargets instances."""
        targets = InterventionTargets(
            ref_depth_1_rhs=10,
            ref_depth_2_rhs=20,
            query_var=30,
            prediction_token_pos=40
        )
        
        assert targets.ref_depth_1_rhs == 10
        assert targets.ref_depth_2_rhs == 20
        assert targets.query_var == 30
        assert targets.prediction_token_pos == 40
    
    def test_intervention_targets_defaults(self):
        """Test InterventionTargets default values."""
        targets = InterventionTargets()
        
        assert targets.ref_depth_1_rhs is None
        assert targets.ref_depth_2_rhs is None
        assert targets.ref_depth_3_rhs is None
        assert targets.ref_depth_4_rhs is None
        assert targets.query_var is None
        assert targets.prediction_token_pos is None


class TestIntegrationWithRealExperiment:
    """Integration tests using real experiment data."""
    
    @pytest.fixture
    def real_experiment_data(self):
        """Real program from the bug report."""
        return {
            "program": "l = 1\nc = l\ny = 5\np = 6\nm = 8\nq = p\nf = m\na = c\nj = 9\nv = 0\nx = f\no = q\nr = a\nw = 5\ng = r\nb = r\ni = r\n#a: ",
            "query_var": "a",
            "expected_chain": [("a", "c"), ("c", "l"), ("l", "1")],
            "expected_root": "1"
        }
    
    def test_real_experiment_variable_chain(self, analyzer, real_experiment_data):
        """Test variable chain identification on real experiment data."""
        chain = analyzer.identify_variable_chain(
            real_experiment_data["program"], 
            real_experiment_data["query_var"]
        )
        
        assert chain.query_var == real_experiment_data["query_var"]
        assert chain.chain == real_experiment_data["expected_chain"]
        assert chain.root_value == real_experiment_data["expected_root"]
        assert chain.referential_depth == 3
        assert not chain.is_circular
    
    def test_real_experiment_intervention_targets(self, analyzer, real_experiment_data):
        """Test intervention target finding on real experiment data."""
        targets = analyzer.find_intervention_targets(
            real_experiment_data["program"],
            real_experiment_data["query_var"]
        )
        
        # Verify all expected targets are found
        assert "ref_depth_1_rhs" in targets
        assert "ref_depth_2_rhs" in targets  
        assert "ref_depth_3_rhs" in targets
        assert "query_var" in targets
        assert "prediction_token_pos" in targets
        
        # Verify tokens are correct
        tokens = analyzer.tokenizer.tokenize(real_experiment_data["program"])
        
        # Check that the positions correspond to the right tokens  
        pos1 = targets["ref_depth_1_rhs"]  # Should be "1" in "l = 1" (root value)
        pos2 = targets["ref_depth_2_rhs"]  # Should be "l" in "c = l" (first hop from root)
        pos3 = targets["ref_depth_3_rhs"]  # Should be "c" in "a = c" (second hop from root)
        
        assert tokens[pos1].replace('Ġ', '').strip() == "1"
        assert tokens[pos2].replace('Ġ', '').strip() == "l"
        assert tokens[pos3].replace('Ġ', '').strip() == "c"
        
        # Verify these are NOT the buggy positions from the original experiment
        buggy_targets = {"ref_depth_1_rhs": 5, "ref_depth_2_rhs": 7, "ref_depth_3_rhs": 3}
        
        # The corrected positions should be different from the buggy ones
        # (except for ref_depth_2_rhs which might be correct)
        assert targets["ref_depth_1_rhs"] != buggy_targets["ref_depth_1_rhs"]
        assert targets["ref_depth_3_rhs"] != buggy_targets["ref_depth_3_rhs"]


# ========================================================================
# Performance and Stress Tests
# ========================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_program_performance(self, analyzer):
        """Test performance on large programs."""
        # Generate a large program with many assignments
        lines = []
        for i in range(1000):
            lines.append(f"var_{i} = {i}")
        
        lines.append("final = var_999")
        lines.append("#final:")
        program = "\n".join(lines)
        
        # This should complete in reasonable time
        import time
        start = time.time()
        
        targets = analyzer.find_intervention_targets(program, "final")
        
        end = time.time()
        
        # Should complete in under 1 second
        assert (end - start) < 1.0
        assert "ref_depth_1_rhs" in targets
        assert "ref_depth_2_rhs" in targets
    
    def test_deep_chain_performance(self, analyzer):
        """Test performance on very deep variable chains."""
        # Create a deep chain: x0 -> x1 -> x2 -> ... -> x99 -> 42
        lines = ["x99 = 42"]
        for i in range(98, -1, -1):
            lines.append(f"x{i} = x{i+1}")
        
        lines.append("#x0:")
        program = "\n".join(lines)
        
        import time
        start = time.time()
        
        chain = analyzer.identify_variable_chain(program, "x0")
        
        end = time.time()
        
        # Should complete in reasonable time and find the full chain
        assert (end - start) < 1.0
        assert chain.referential_depth == 100
        assert chain.root_value == "42"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])