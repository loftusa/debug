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
