#!/bin/bash
#SBATCH --job-name=train-lm
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --licenses=scratch
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1

export WANDB_DIR="$SCRATCH/cs336-assignment1/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb-data"

mkdir -p \
    "$SCRATCH/cs336-assignment1/checkpoints" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_DATA_DIR"

for path in \
    .venv/bin/python \
    configs.json \
    TinyStories/training_ids.bin \
    TinyStories/valid_ids.bin; do
    if [[ ! -e "$path" ]]; then
        echo "Required path not found: $path" >&2
        exit 1
    fi
done

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

srun --cpu-bind=cores --gpu-bind=single:1 \
    .venv/bin/python -c \
    'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name())'

srun --cpu-bind=cores --gpu-bind=single:1 \
    .venv/bin/python -u -m cs336_basics.train_lm \
    --config_path configs.json

echo "Finished: $(date)"