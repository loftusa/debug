#!/bin/bash

# Run visualizations on all 0626 experiments for the 6 configurations
set -e

cd experiments

MODELS=("Qwen_Qwen3-4B" "Qwen_Qwen3-8B" "Qwen_Qwen3-14B")
FIGURE_PATHS=()

# Configuration directories mapping
declare -A CONFIGS
CONFIGS["1_5"]="../results/full_token_layer_patching_1_hop_seq_5/20250626_150524"
CONFIGS["2_5"]="../results/full_token_layer_patching_2_hop_seq_5/20250626_153646"
CONFIGS["3_5"]="../results/full_token_layer_patching_3_hop_seq_5/20250626_160756"
CONFIGS["1_17"]="../results/full_token_layer_patching_1_hop_seq_17/20250626_163848"
CONFIGS["2_17"]="../results/full_token_layer_patching_2_hop_seq_17/20250626_182140"
CONFIGS["3_17"]="../results/full_token_layer_patching_3_hop_seq_17/20250626_200619"

echo "Running visualizations for all 0626 experiments..."
echo "=================================================="

for CONFIG_KEY in "${!CONFIGS[@]}"; do
    RESULTS_DIR="${CONFIGS[$CONFIG_KEY]}"
    echo ""
    echo "Processing configuration: $CONFIG_KEY"
    echo "Results directory: $RESULTS_DIR"
    echo "--------------------------------------------"
    
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "  ❌ Directory not found: $RESULTS_DIR"
        continue
    fi
    
    for MODEL in "${MODELS[@]}"; do
        echo "  Generating visualization for $MODEL..."
        
        # Run the visualization script
        uv run _09_visualize_patching_auto.py --model "$MODEL" --base-results-dir "$(dirname "$RESULTS_DIR")" > "viz_${CONFIG_KEY}_${MODEL}.log" 2>&1
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            # Extract the figure path from the output
            FIGURE_PATH=$(grep "Visualization complete:" "viz_${CONFIG_KEY}_${MODEL}.log" | awk '{print $3}')
            
            if [ -n "$FIGURE_PATH" ]; then
                FIGURE_PATHS+=("$FIGURE_PATH")
                echo "    ✅ Generated: $FIGURE_PATH"
            else
                echo "    ⚠️  Could not extract figure path"
                echo "    Log output:"
                cat "viz_${CONFIG_KEY}_${MODEL}.log"
            fi
        else
            echo "    ❌ Visualization failed"
            echo "    Last 10 lines of log:"
            tail -10 "viz_${CONFIG_KEY}_${MODEL}.log"
        fi
        
        # Clean up log file
        rm "viz_${CONFIG_KEY}_${MODEL}.log"
    done
    
    echo "  ✅ Completed configuration: $CONFIG_KEY"
done

echo ""
echo "=================================================="
echo "Batch visualization complete!"
echo "=================================================="
echo "Generated figures:"
for path in "${FIGURE_PATHS[@]}"; do
    echo "  $path"
done
echo ""
echo "Total figures generated: ${#FIGURE_PATHS[@]}"