#!/usr/bin/env bash
set -e

# Ensure we run from the script's directory or a known base if more appropriate
cd "$(dirname "$0")"

# Create a timestamped directory for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_BASE_DIR="results/integer_${TIMESTAMP}" # Changed from boolean
mkdir -p "${RESULTS_BASE_DIR}"
echo "Results will be saved in: ${RESULTS_BASE_DIR}"

# Define the models to test (mirroring DEFAULT_MODELS_ALL from the Python script)
MODELS_TO_TEST=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)
# Convert array to comma-separated string for the Python script's --models argument
MODEL_IDS_STR=$(IFS=,; echo "${MODELS_TO_TEST[*]}")

# Loop seq_len from 2 to 16
for seq_len in {2..16}; do
# for seq_len in {2..3}; do # For quicker testing
  echo "=== Running Integer Arithmetic Experiment: seq_len=${seq_len} ===" # Changed description
  # Updated script name and added new arguments, removed --kind
  if ! uv run accelerate launch --num_processes 1 _02_opensource_entity_tracking.py \
    --seq-len "${seq_len}" \
    --num-seqs 100 \
    --best-of 1 \
    --output-dir "${RESULTS_BASE_DIR}" \
    --models "${MODEL_IDS_STR}"; then 
    echo "=== Error running _02_opensource_entity_tracking.py for seq_len=${seq_len}, continuing ==="
  fi
done

# Finally, build the visualization
# Assuming _02_tracking_visualizations.py can also accept --input-dir or will be adapted
echo "=== Generating integer arithmetic accuracy plot ===" # Changed description
uv run _02_tracking_visualizations.py --input-dir "${RESULTS_BASE_DIR}" # Added --input-dir

echo "Integer arithmetic experiment run complete. Results and plot are in ${RESULTS_BASE_DIR}"