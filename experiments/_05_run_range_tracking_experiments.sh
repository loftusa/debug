#!/usr/bin/env bash
set -e

# Ensure we run from the script's directory or a known base if more appropriate
cd "$(dirname "$0")"

# Create a timestamped directory for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROGRAM_TYPE="lines"
RANDOM_SUM=true
RESULTS_BASE_DIR="results/range_tracking_${PROGRAM_TYPE}_${RANDOM_SUM}_${TIMESTAMP}"
mkdir -p "${RESULTS_BASE_DIR}"
echo "Results will be saved in: ${RESULTS_BASE_DIR}"

# Define the models to test (matching DEFAULT_MODELS_ALL from _05_range_tracking.py)
MODELS_TO_TEST=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-32B"
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