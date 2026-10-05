#!/bin/bash
#SBATCH --job-name=lr-sweep-4gpu
#SBATCH --account=mp309_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j-controller.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1

export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
export WANDB_DIR="$PSCRATCH/cs336-assignment1/wandb"
export WANDB_CACHE_DIR="$PSCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$PSCRATCH/.cache/wandb-data"

mkdir -p "$UV_CACHE_DIR"
mkdir -p "$WANDB_DIR"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_DATA_DIR"
mkdir -p "$PSCRATCH/cs336-assignment1/checkpoints"

: "${SWEEP_ID:?Submit with SWEEP_ID set}"

pids=()

for agent_number in 1 2 3 4; do
    echo "Starting W&B agent ${agent_number}"

    srun --exact \
        --ntasks=1 \
        --cpus-per-task=32 \
        --gpus-per-task=1 \
        --gpu-bind=single:1 \
        --output="slurm-lr-sweep-${SLURM_JOB_ID}-agent-${agent_number}.out" \
        --error="slurm-lr-sweep-${SLURM_JOB_ID}-agent-${agent_number}.err" \
        .venv/bin/wandb agent --count 1 "$SWEEP_ID" &

    pids+=("$!")
done

status=0

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

exit "$status"