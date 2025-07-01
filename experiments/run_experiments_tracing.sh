#!/bin/bash

# Full pipeline script to find robust seeds and run causal tracing experiments
# 1. Find random seeds that work for 4B, 8B, and 14B models
# 2. Run full layer token patching for each model
# 3. Generate visualizations for each model

set -e  # Exit on any error

echo "=========================================="
echo "Starting Full Causal Tracing Pipeline"
echo "=========================================="

# Change to experiments directory if we're not already there
if [ "$(basename "$PWD")" != "experiments" ]; then
    cd experiments
fi

# Configuration: Hardcoded seeds (update these from find_all_seeds.sh results)
HOPS=(1 2 3)
SEQ_LENS=(5 17)
declare -A LATEST_DIRS

# TODO: Update these seeds by running find_all_seeds.sh and copying results here
# Format: HARDCODED_SEEDS[hops_seqlen]=seed
declare -A HARDCODED_SEEDS

# Sequence length 5 seeds
HARDCODED_SEEDS["1_5"]=11     # 1 hop, seq len 5
HARDCODED_SEEDS["2_5"]=3      # 2 hop, seq len 5  
HARDCODED_SEEDS["3_5"]=2      # 3 hop, seq len 5

# Sequence length 17 seeds (from CLAUDE.md)
HARDCODED_SEEDS["1_17"]=5     # 1 hop, seq len 17
HARDCODED_SEEDS["2_17"]=14    # 2 hop, seq len 17
HARDCODED_SEEDS["3_17"]=12    # 3 hop, seq len 17

echo "Using hardcoded seeds:"
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  Sequence length $SEQ_LEN:"
    for HOP_COUNT in "${HOPS[@]}"; do
        SEED_KEY="${HOP_COUNT}_${SEQ_LEN}"
        echo "    $HOP_COUNT hop(s): seed ${HARDCODED_SEEDS[$SEED_KEY]}"
    done
done
echo ""

MODELS=("Qwen_Qwen3-4B" "Qwen_Qwen3-8B" "Qwen_Qwen3-14B")
ALL_FIGURE_PATHS=()

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for HOP_COUNT in "${HOPS[@]}"; do
        SEED_KEY="${HOP_COUNT}_${SEQ_LEN}"
        SEED=${HARDCODED_SEEDS[$SEED_KEY]}
        
        if [ -z "$SEED" ]; then
            echo "ERROR: No seed defined for $HOP_COUNT hop(s), seq len $SEQ_LEN! Update HARDCODED_SEEDS array."
            exit 1
        fi
        
        # Step 2: Run full layer token patching with the found seed
        echo ""
        echo "Step 2.${HOP_COUNT}_${SEQ_LEN}: Running full layer token patching for $HOP_COUNT hop(s), seq len $SEQ_LEN..."
        echo "------------------------------------------------------------"
        
        echo "Running patching experiment with seed $SEED for models: 4B, 8B, 14B"
        python _08_full_layer_token_patching.py --seed $SEED --seq-len $SEQ_LEN --hops $HOP_COUNT
        
        # Get the timestamp from the most recent results directory
        RESULTS_BASE="../results/full_token_layer_patching_${HOP_COUNT}_hop_seq_${SEQ_LEN}"
        LATEST_DIR=$(ls -td $RESULTS_BASE/*/ | head -n 1)
        LATEST_DIRS["${HOP_COUNT}_${SEQ_LEN}"]="$LATEST_DIR"
        
        echo "Results saved to: $LATEST_DIR"
        
        # Step 3: Generate visualizations immediately for this configuration
        echo ""
        echo "Step 3.${HOP_COUNT}_${SEQ_LEN}: Generating visualizations for $HOP_COUNT hop(s), seq len $SEQ_LEN..."
        echo "------------------------------------------------------------"
        
        for MODEL in "${MODELS[@]}"; do
            echo "  Generating visualization for $MODEL ($HOP_COUNT hop(s), seq len $SEQ_LEN)..."
            
            # Run the visualization script with click parameters
            python _09_visualize_patching.py --model "$MODEL" --base-results-dir "$(dirname "$LATEST_DIR")" > "viz_${SEED_KEY}_${MODEL}.log" 2>&1
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                # Extract the figure path from the output
                FIGURE_PATH=$(grep "Visualization complete:" "viz_${SEED_KEY}_${MODEL}.log" | awk '{print $3}')
                
                if [ -n "$FIGURE_PATH" ]; then
                    ALL_FIGURE_PATHS+=("$FIGURE_PATH")
                    echo "    ✅ Generated figure: $FIGURE_PATH"
                else
                    echo "    WARNING: Could not find figure path for $MODEL ($HOP_COUNT hop(s), seq len $SEQ_LEN)"
                    echo "    Log output:"
                    cat "viz_${SEED_KEY}_${MODEL}.log"
                fi
            else
                echo "    ERROR: Visualization failed for $MODEL ($HOP_COUNT hop(s), seq len $SEQ_LEN)"
                echo "    Log output:"
                cat "viz_${SEED_KEY}_${MODEL}.log"
            fi
            
            # Clean up log file
            rm "viz_${SEED_KEY}_${MODEL}.log"
        done
        
        echo "✅ Completed configuration: $HOP_COUNT hop(s), seq len $SEQ_LEN"
    done
done

# Step 4: Summary
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results by configuration:"
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  Sequence length $SEQ_LEN:"
    for HOP_COUNT in "${HOPS[@]}"; do
        COMBO_KEY="${HOP_COUNT}_${SEQ_LEN}"
        echo "    $HOP_COUNT hop(s): seed ${HARDCODED_SEEDS[$COMBO_KEY]} -> ${LATEST_DIRS[$COMBO_KEY]}"
    done
done
echo ""
echo "Generated figures:"
for path in "${ALL_FIGURE_PATHS[@]}"; do
    echo "  $path"
done
echo ""
echo "To view results for each configuration:"
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  Sequence length $SEQ_LEN:"
    for HOP_COUNT in "${HOPS[@]}"; do
        COMBO_KEY="${HOP_COUNT}_${SEQ_LEN}"
        echo "    $HOP_COUNT hop(s): cd ${LATEST_DIRS[$COMBO_KEY]}"
    done
done