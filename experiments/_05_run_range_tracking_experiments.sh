#!/usr/bin/env bash
set -e

# Help function
show_help() {
    echo "Usage: $0 [PROGRAM_TYPE] [RANDOM_SUM]"
    echo ""
    echo "Arguments:"
    echo "  PROGRAM_TYPE    Type of program to generate: 'lines' or 'single' (default: lines)"
    echo "  RANDOM_SUM      Whether to use random sums: 'true' or 'false' (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Use defaults: lines, false"
    echo "  $0 lines true         # Use lines with random sums"
    echo "  $0 single false       # Use single (for loop) format with fixed sums"
    echo ""
    exit 0
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

# Parse command line arguments with defaults
PROGRAM_TYPE="${1:-lines}"     # Default to "lines" if not provided
RANDOM_SUM="${2:-false}"       # Default to "false" if not provided

# Validate PROGRAM_TYPE
if [ "$PROGRAM_TYPE" != "lines" ] && [ "$PROGRAM_TYPE" != "single" ]; then
    echo "Error: PROGRAM_TYPE must be 'lines' or 'single', got: $PROGRAM_TYPE"
    echo "Use -h or --help for usage information."
    exit 1
fi

# Validate RANDOM_SUM
if [ "$RANDOM_SUM" != "true" ] && [ "$RANDOM_SUM" != "false" ]; then
    echo "Error: RANDOM_SUM must be 'true' or 'false', got: $RANDOM_SUM"
    echo "Use -h or --help for usage information."
    exit 1
fi

echo "Running range tracking experiment with:"
echo "  PROGRAM_TYPE: $PROGRAM_TYPE"
echo "  RANDOM_SUM: $RANDOM_SUM"
echo ""

# Ensure we run from the script's directory or a known base if more appropriate
cd "$(dirname "$0")"

# Create a timestamped directory for results
QWEN_VERSION="3"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_BASE_DIR="results/range_tracking_${PROGRAM_TYPE}_${RANDOM_SUM}_QWEN${QWEN_VERSION}_${TIMESTAMP}"
mkdir -p "${RESULTS_BASE_DIR}"
echo "Results will be saved in: ${RESULTS_BASE_DIR}"

# Define the models to test (matching DEFAULT_MODELS_ALL from _05_range_tracking.py)
MODELS_TO_TEST=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-32B"
)
# Convert array to comma-separated string for the Python script's --models argument
MODEL_IDS_STR=$(IFS=,; echo "${MODELS_TO_TEST[*]}")

# Build random-sum argument conditionally
RANDOM_SUM_ARG=""
if [ "$RANDOM_SUM" = "true" ]; then
    RANDOM_SUM_ARG="--random-sum"
fi

# Loop seq_len from 2 to 16
for seq_len in {2..16}; do
# for seq_len in {2..3}; do # For quicker testing
  echo "=== Running Range Tracking Experiment: seq_len=${seq_len} ==="
  if ! uv run accelerate launch --num_processes 1 _05_range_tracking.py \
    --seq-len "${seq_len}" \
    --num-seqs 100 \
    --best-of 1 \
    --output-dir "${RESULTS_BASE_DIR}" \
    --models "${MODEL_IDS_STR}" \
    --program-type "${PROGRAM_TYPE}" \
    ${RANDOM_SUM_ARG}; then 
    echo "=== Error running _05_range_tracking.py for seq_len=${seq_len}, continuing ==="
  fi
done

# Finally, build the visualization
echo "=== Generating range tracking accuracy plot ==="
uv run _05_range_tracking_visualizations.py --input-dir "${RESULTS_BASE_DIR}" --program-type "${PROGRAM_TYPE}"

echo "Range tracking experiment run complete. Results and plot are in ${RESULTS_BASE_DIR}" 