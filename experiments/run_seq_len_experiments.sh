#!/usr/bin/env bash
set -e

# Ensure we run from this directory
cd "$(dirname "$0")"

# Loop seq_len from 2 to 16
for seq_len in {2..16}; do
  echo "=== Running seq_len=${seq_len} ==="
  uv run accelerate launch --num_processes 1 experiments/_02_opensource_entity_tracking.py --seq_len "${seq_len}"
done

# Finally, build the visualization
echo "=== Generating accuracy plot ==="
uv run _02_tracking_visualizations.py