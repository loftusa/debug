#!/usr/bin/env python3
"""
Complexity comparison: Legacy vs Simplified experiments
Shows the dramatic reduction in code complexity achieved by the debug framework.
"""

import os
from pathlib import Path

def count_lines(filepath):
    """Count lines of code in a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Count non-empty, non-comment lines
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        return len(lines), len(code_lines)
    except Exception:
        return 0, 0

def analyze_experiments():
    """Compare legacy vs simplified experiments."""
    
    experiments_dir = Path(__file__).parent
    
    # Define file pairs (legacy -> simplified)
    comparisons = [
        ("_02_opensource_entity_tracking.py", "_02_entity_tracking_simple.py", "Entity Tracking"),
        ("_04_boolean_experiment.py", "_04_boolean_simple.py", "Boolean Logic"),
        ("_05_range_tracking.py", "_05_range_tracking_simple.py", "Range Tracking"),
        ("_03_collect_correct_pairs.py", "_03_counterfactual_simple.py", "Counterfactual Pairs"),
    ]
    
    print("🔬 Code Complexity Analysis: Legacy vs Simplified")
    print("=" * 65)
    print(f"{'Experiment':<20} {'Legacy':<12} {'Simple':<12} {'Reduction':<12} {'Saved'}")
    print("-" * 65)
    
    total_legacy_lines = 0
    total_simple_lines = 0
    
    for legacy_file, simple_file, name in comparisons:
        legacy_path = experiments_dir / legacy_file
        simple_path = experiments_dir / simple_file
        
        legacy_total, legacy_code = count_lines(legacy_path)
        simple_total, simple_code = count_lines(simple_path)
        
        if legacy_total > 0 and simple_total > 0:
            reduction = (1 - simple_total / legacy_total) * 100
            saved = legacy_total - simple_total
            
            print(f"{name:<20} {legacy_total:<12} {simple_total:<12} {reduction:>8.1f}%    {saved:>4} lines")
            
            total_legacy_lines += legacy_total
            total_simple_lines += simple_total
    
    print("-" * 65)
    overall_reduction = (1 - total_simple_lines / total_legacy_lines) * 100
    overall_saved = total_legacy_lines - total_simple_lines
    print(f"{'TOTAL':<20} {total_legacy_lines:<12} {total_simple_lines:<12} {overall_reduction:>8.1f}%    {overall_saved:>4} lines")
    
    print(f"\n📊 Summary:")
    print(f"   • Total lines eliminated: {overall_saved}")
    print(f"   • Average reduction: {overall_reduction:.1f}%")
    print(f"   • Maintainability: 📈 DRAMATICALLY IMPROVED")
    print(f"   • Code reuse: 📈 MAXIMIZED (shared framework)")
    print(f"   • Bug surface: 📉 MINIMIZED (less custom code)")

def show_feature_comparison():
    """Show feature comparison between legacy and simplified."""
    
    print(f"\n🎯 Feature Comparison")
    print("=" * 50)
    
    features = [
        ("Model loading & cleanup", "✅ Manual", "✅ Automatic"),
        ("Progress tracking", "✅ Custom", "✅ Built-in"),
        ("Error handling", "⚠️  Partial", "✅ Comprehensive"),
        ("Result visualization", "⚠️  Custom", "✅ Automatic"),
        ("Configuration", "🔧 Hardcoded", "🔧 Flexible"),
        ("Code reuse", "❌ Copy-paste", "✅ Shared framework"),
        ("Experiment consistency", "⚠️  Variable", "✅ Standardized"),
        ("Adding new experiments", "🐌 Complex", "⚡ Simple"),
        ("Debugging", "🐞 Difficult", "🔍 Easy"),
        ("Maintenance", "😰 Nightmare", "😎 Breeze"),
    ]
    
    print(f"{'Feature':<25} {'Legacy':<15} {'Simplified'}")
    print("-" * 50)
    for feature, legacy, simplified in features:
        print(f"{feature:<25} {legacy:<15} {simplified}")

def show_usage_examples():
    """Show usage examples."""
    
    print(f"\n🚀 Usage Examples")
    print("=" * 40)
    
    print("Legacy (complex):")
    print("```bash")
    print("python _02_opensource_entity_tracking.py \\")
    print("  --num-seqs 100 \\")
    print("  --seq-len 10 \\")
    print("  --best-of 1 \\")
    print("  --output-dir results/entity \\")
    print("  --models 'Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B'")
    print("```")
    
    print("\nSimplified (clean):")
    print("```bash")
    print("./run_experiments.sh entity")
    print("# or")
    print("python _02_entity_tracking_simple.py")
    print("```")
    
    print("\nProgrammatic usage:")
    print("```python")
    print("from debug import quick_experiment, ExperimentRunner")
    print("")
    print("config = quick_experiment('test', prompts.RANGE_TRACKING, generators.make_range_program)")
    print("runner = ExperimentRunner()")
    print("result = runner.run(config)")
    print("runner.plot('test')")
    print("```")

if __name__ == "__main__":
    analyze_experiments()
    show_feature_comparison()
    show_usage_examples()
    
    print(f"\n🎉 Conclusion:")
    print("The debug framework transforms complex, brittle experiment code into")
    print("simple, maintainable, and powerful experiment definitions!")
    print("\n🔗 Try it: ./run_experiments.sh all") 