#!/usr/bin/env bash
set -e

# Ensure we run from this directory
cd "$(dirname "$0")"

# Check for OpenRouter API key
export OPENROUTER_API_KEY="nan"

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable not set"
    echo "Please set it with: export OPENROUTER_API_KEY=your_api_key"
    exit 1
fi

# Loop seq_len from 2 to 16
for seq_len in {2..16}; do
  echo "=== Running commercial models with seq_len=${seq_len} ==="
  uv run _02_commercial_entity_tracking.py --seq_len "${seq_len}" --num-seqs 50 --best-of 4
  
  # Add a delay between runs to avoid API rate limits
  echo "Waiting 10 seconds before next run..."
  sleep 10
done

# Finally, build the visualization
echo "=== Generating accuracy plot ==="
uv run _02_tracking_visualizations.py 