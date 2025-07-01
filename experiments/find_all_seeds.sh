#!/bin/bash

# Script to find robust seeds for all combinations of hops (1,2,3) and sequence lengths (5,17)
# Saves only the results to a single text file

set -e

RESULTS_FILE="seed_results.txt"
HOPS=(1 2 3)
SEQ_LENS=(5 17)

echo "=========================================" > $RESULTS_FILE
echo "ROBUST SEED SEARCH RESULTS" >> $RESULTS_FILE
echo "Started at: $(date)" >> $RESULTS_FILE
echo "=========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Change to experiments directory if not already there
if [ "$(basename "$PWD")" != "experiments" ]; then
    cd experiments
fi

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for HOP_COUNT in "${HOPS[@]}"; do
        echo ""
        echo "========================================="
        echo "Finding seeds for $HOP_COUNT hop(s), sequence length $SEQ_LEN"
        echo "========================================="
        echo "Started at: $(date)"
        
        # Run the seed finding script
        echo "Running: uv run 11_test_rng_seeds.py --exact-hops $HOP_COUNT --seq-len $SEQ_LEN"
        
        # Run and capture output, saving results immediately
        TEMP_OUTPUT="temp_output_${HOP_COUNT}hop_${SEQ_LEN}len.txt"
        uv run 11_test_rng_seeds.py --exact-hops $HOP_COUNT --seq-len $SEQ_LEN > $TEMP_OUTPUT 2>&1
        EXIT_CODE=$?
        
        # Immediately append results to the main file
        echo "Configuration: $HOP_COUNT hop(s), sequence length $SEQ_LEN" >> $RESULTS_FILE
        echo "----------------------------------------" >> $RESULTS_FILE
        
        if [ $EXIT_CODE -eq 0 ]; then
            # Extract and save the successful result
            grep -A 10 "🎉 Found a robust program" $TEMP_OUTPUT | head -n 20 >> $RESULTS_FILE
            echo "✅ SUCCESS: Found seed for $HOP_COUNT hop(s), sequence length $SEQ_LEN"
            echo "✅ Completed at: $(date)" >> $RESULTS_FILE
        else
            echo "❌ FAILED: $HOP_COUNT hop(s), sequence length $SEQ_LEN" >> $RESULTS_FILE
            echo "Error details:" >> $RESULTS_FILE
            tail -5 $TEMP_OUTPUT >> $RESULTS_FILE
            echo "ERROR: Seed finding failed for $HOP_COUNT hop(s), seq len $SEQ_LEN"
            echo "❌ Failed at: $(date)" >> $RESULTS_FILE
        fi
        
        echo "" >> $RESULTS_FILE
        echo "" >> $RESULTS_FILE
        
        # Clean up temp file
        rm $TEMP_OUTPUT
        
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=========================================" >> $RESULTS_FILE
echo "SEARCH COMPLETED" >> $RESULTS_FILE
echo "Finished at: $(date)" >> $RESULTS_FILE
echo "=========================================" >> $RESULTS_FILE

echo ""
echo "========================================="
echo "ALL SEED SEARCHES COMPLETE!"
echo "========================================="
echo "Results saved to: $(pwd)/$RESULTS_FILE"
echo ""
echo "Summary of results:"
cat $RESULTS_FILE | grep -E "(🎉|FAILED|RNG Seed)"