#!/usr/bin/env bash
set -e

# Ensure we run from this directory
cd "$(dirname "$0")"

# --- Configuration ---
# Default values if not passed as environment variables
: ${NUM_SEQS:=124}
: ${BEST_OF:=10}

# List of models to evaluate (adjust as needed)
DEFAULT_MODELS=(
    "openai-community/gpt2"
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "tiiuae/Falcon3-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    "NTQAI/Nxcode-CQ-7B-orpo"
    "Qwen/CodeQwen1.5-7B-Chat"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-14B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/Qwen2.5-Coder-14B-Instruct"
    "Qwen/Qwen2.5-14B"
    "allenai/OLMo-2-1124-7B-Instruct"
    "allenai/OLMo-2-1124-13B-Instruct"
    # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
)

PYTHON_SCRIPT="_02_opensource_entity_tracking.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# --- GPU Detection ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi command not found. Cannot detect GPUs."
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader)
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "WARNING: No GPUs detected or invalid output from nvidia-smi. Running sequentially on CPU (GPU ID 0)."
    NUM_GPUS=1 # Treat as 1 "GPU" for sequential CPU execution
    # Fallback to CPU: The python script should handle device='cpu' if gpu_id=0 and cuda not available
fi
echo "Detected $NUM_GPUS GPUs. Using NUM_SEQS=$NUM_SEQS, BEST_OF=$BEST_OF"


# Trap SIGINT (Ctrl+C) to clean up background processes for the current seq_len
trap 'echo "Caught SIGINT, killing background jobs for current seq_len..."; pkill -P $$; exit 1' SIGINT


# Loop seq_len from 2 to 16
for seq_len in {2..16}; do
  echo "========================================"
  echo "=== Running seq_len=${seq_len} ==="
  echo "========================================"

  # --- Parallel Execution Logic for this seq_len ---
  declare -a gpu_pids
  declare -A pid_to_gpu
  declare -A pid_to_model

  # Initialize GPU PIDs array
  for (( i=0; i<NUM_GPUS; i++ )); do
      gpu_pids[$i]=0
  done

  model_idx=0
  total_models=${#DEFAULT_MODELS[@]}
  launched_count=0
  completed_count=0
  failed_count=0

  while [[ $completed_count -lt $total_models ]]; do
      found_gpu=false
      # Try to launch on a free GPU
      for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
          if [[ ${gpu_pids[$gpu_id]} -eq 0 ]] && [[ $model_idx -lt $total_models ]]; then
              model_id=${DEFAULT_MODELS[$model_idx]}
              log_file="${LOG_DIR}/run_seq${seq_len}_${model_id//\//_}_gpu${gpu_id}.log"
              echo "[SeqLen $seq_len] Launching model $model_id (${model_idx + 1}/$total_models) on GPU $gpu_id -> ${log_file}"

              # Run the python script in the background using uv run
              # Assign GPU 0 even if no CUDA detected, python script handles CPU fallback
              uv run python "$PYTHON_SCRIPT" \\
                  --model-id "$model_id" \\
                  --gpu-id "$gpu_id" \\
                  --seq_len "$seq_len" \\
                  --num-seqs "$NUM_SEQS" \\
                  --best-of "$BEST_OF" &> "$log_file" &

              pid=$!
              gpu_pids[$gpu_id]=$pid
              pid_to_gpu[$pid]=$gpu_id
              pid_to_model[$pid]=$model_id
              ((model_idx++))
              ((launched_count++))
              found_gpu=true
              # Optional: Short sleep to stagger launches slightly if needed
              # sleep 0.5
          fi
      done

      # If no GPU was free or all models are launched, wait for a job to finish
      if ! $found_gpu || [[ $model_idx -ge $total_models ]]; then
          if [[ $launched_count -gt $completed_count ]]; then
               # wait -n waits for the next background job to finish
              wait -n -p finished_pid # Get the PID of the finished process
              exit_status=$?
              if [[ -n "$finished_pid" ]] && [[ -v pid_to_gpu[$finished_pid] ]]; then
                  gpu_id=${pid_to_gpu[$finished_pid]}
                  model_id=${pid_to_model[$finished_pid]}
                  if [[ $exit_status -eq 0 ]]; then
                      echo "[SeqLen $seq_len] Model $model_id on GPU $gpu_id finished successfully."
                  else
                       echo "ERROR: [SeqLen $seq_len] Model $model_id on GPU $gpu_id failed with exit code $exit_status. Check log: ${LOG_DIR}/run_seq${seq_len}_${model_id//\//_}_gpu${gpu_id}.log"
                       ((failed_count++))
                  fi
                  # Mark GPU as free and remove PID mapping
                  gpu_pids[$gpu_id]=0
                  unset pid_to_gpu[$finished_pid]
                  unset pid_to_model[$finished_pid]
                  ((completed_count++))
              else
                  # wait -n might return spuriously or for a non-managed process
                  echo "[SeqLen $seq_len] Wait returned for unknown PID $finished_pid or PID not tracked. Status: $exit_status. Continuing..."
                  # Add a small sleep to prevent potential busy loops in edge cases
                  sleep 1
              fi
          else
              # Break the inner loop if all models for this seq_len are completed or if nothing could be launched
              if [[ $completed_count -ge $total_models ]] || [[ $launched_count -eq 0 && $model_idx -lt $total_models ]]; then
                   if [[ $launched_count -eq 0 && $model_idx -lt $total_models ]]; then
                       echo "ERROR: Could not launch any models for seq_len $seq_len. Check GPU availability or logs."
                   fi
                  break
              fi
              # Add a small sleep if waiting and nothing finished immediately
              sleep 1
          fi
      fi
  done # End of while loop for models within a seq_len

  echo "=== Finished seq_len=${seq_len}. Completed: $completed_count/$total_models models. Failed: $failed_count ==="
  # Wait for any remaining stragglers explicitly (shouldn't be needed with wait -n loop)
  wait
  echo "----------------------------------------"
  sleep 2 # Small pause between sequence lengths

done # End of seq_len loop

# Untrap SIGINT
trap - SIGINT

# Finally, build the visualization
echo "=== Generating accuracy plot ==="
uv run _02_tracking_visualizations.py

echo "=== All experiments finished ==="
exit 0