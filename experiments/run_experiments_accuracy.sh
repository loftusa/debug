#!/bin/bash

# Unified experiment runner for the simplified debug framework experiments
# Replaces all the complex individual bash scripts

set -e

# Default values
EXPERIMENT="range"
OUTPUT_DIR="results/simplified_experiments"
MODELS="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B"
NUM_SEQS=50

# Help function
show_help() {
    echo "Usage: $0 [EXPERIMENT] [OPTIONS]"
    echo ""
    echo "EXPERIMENTS:"
    echo "  range           Run range tracking experiment"
    echo "  entity          Run entity tracking experiment"  
    echo "  boolean         Run boolean logic experiment"
    echo "  counterfactual  Collect counterfactual pairs"
    echo "  all             Run all experiments"
    echo ""
    echo "OPTIONS:"
    echo "  -o, --output DIR     Output directory (default: results/simplified_experiments)"
    echo "  -m, --models LIST    Comma-separated model list (default: Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B)"
    echo "  -n, --num-seqs N     Number of sequences per length (default: 50)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 range                                    # Run range tracking"
    echo "  $0 boolean -m Qwen/Qwen3-0.6B -n 100       # Run boolean with specific model"
    echo "  $0 all -o my_results                       # Run all experiments"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -n|--num-seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        range|entity|boolean|counterfactual|all)
            EXPERIMENT="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "=== Debug Framework Experiments ==="
echo "Experiment: $EXPERIMENT"
echo "Output: $OUTPUT_DIR"
echo "Models: $MODELS"
echo "Sequences: $NUM_SEQS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to experiments directory
cd "$(dirname "$0")"

# Run experiments
case $EXPERIMENT in
    range)
        echo "Running range tracking experiments..."
        python _05_range_tracking_simple.py --output-dir "$OUTPUT_DIR/range_tracking" --models "$MODELS" --num-seqs "$NUM_SEQS"
        python _05_range_tracking_simple.py --program-type single --output-dir "$OUTPUT_DIR/range_tracking" --models "$MODELS" --num-seqs "$NUM_SEQS"
        ;;
    entity)
        echo "Running entity tracking experiment..."
        python _02_entity_tracking_simple.py --output-dir "$OUTPUT_DIR/entity_tracking" --models "$MODELS" --num-seqs "$NUM_SEQS"
        ;;
    boolean)
        echo "Running boolean logic experiment..."
        python _04_boolean_simple.py --output-dir "$OUTPUT_DIR/boolean_logic" --models "$MODELS" --num-seqs "$NUM_SEQS"
        ;;
    counterfactual)
        echo "Collecting counterfactual pairs..."
        python _03_counterfactual_simple.py --model "$(echo $MODELS | cut -d',' -f1)" --num-pairs 20 --output-file "$OUTPUT_DIR/counterfactual_pairs.json"
        ;;
    all)
        echo "Running all experiments..."
        python _05_range_tracking_simple.py --output-dir "$OUTPUT_DIR/range_tracking" --models "$MODELS" --num-seqs "$NUM_SEQS"
        python _02_entity_tracking_simple.py --output-dir "$OUTPUT_DIR/entity_tracking" --models "$MODELS" --num-seqs "$NUM_SEQS"
        python _04_boolean_simple.py --output-dir "$OUTPUT_DIR/boolean_logic" --models "$MODELS" --num-seqs "$NUM_SEQS"
        python _03_counterfactual_simple.py --model "$(echo $MODELS | cut -d',' -f1)" --num-pairs 20 --output-file "$OUTPUT_DIR/counterfactual_pairs.json"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        show_help
        exit 1
        ;;
esac

echo ""
echo "=== Experiments completed! ==="
echo "Results saved to: $OUTPUT_DIR" 