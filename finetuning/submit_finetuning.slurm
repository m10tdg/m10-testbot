#!/bin/bash

#SBATCH --job-name=qwen14b-lora

#SBATCH --account=project_465003167

#SBATCH --partition=small-g

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --gpus-per-node=1

#SBATCH --mem=60G

#SBATCH --time=03:00:00

#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.out

#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.err


set -euo pipefail


# ============================================================================
# ENVIRONMENT
# ============================================================================

export SLURM_CPU_BIND=none

# One allocated GCD.
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Helps PyTorch memory allocation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================================
# LUMI MODULES
# ============================================================================

module purge

module load LUMI/25.03

module load cray-python

module load rocm


# ============================================================================
# PYTHON ENVIRONMENT
# ============================================================================

VENV_PATH="/project/project_465003167/m10-testbot/finetuning/qwen-finetune-env"

source "${VENV_PATH}/bin/activate"


# ============================================================================
# WORKING DIRECTORY
# ============================================================================

cd /project/project_465003167/m10-testbot/finetuning

mkdir -p logs
mkdir -p training_results


# ============================================================================
# JOB INFORMATION
# ============================================================================

echo "============================================================"
echo "QWEN 2.5 14B LoRA FINE-TUNING"
echo "============================================================"

echo "Hostname: $(hostname)"

echo "Date: $(date)"

echo "Working directory: $(pwd)"

echo "Python: $(which python)"

python --version


# ============================================================================
# GPU INFORMATION
# ============================================================================

echo

echo "============================================================"
echo "ROCm / GPU CHECK"
echo "============================================================"

rocm-smi || true


python - <<'PY'
import torch

print("PyTorch:", torch.__version__)

print("ROCm/HIP:", torch.version.hip)

print(
    "CUDA API available:",
    torch.cuda.is_available()
)

print(
    "GPU count:",
    torch.cuda.device_count()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        torch.cuda.get_device_properties(
            0
        ).total_memory / 1024**3,
        "GiB"
    )

    if hasattr(
        torch.cuda,
        "is_bf16_supported"
    ):

        print(
            "BF16 supported:",
            torch.cuda.is_bf16_supported()
        )
PY


# ============================================================================
# START
# ============================================================================

echo

echo "============================================================"
echo "STARTING FINE-TUNING"
echo "============================================================"

python -u finetune_qwen.py


# ============================================================================
# FINISHED
# ============================================================================

echo

echo "============================================================"
echo "FINE-TUNING JOB FINISHED"
echo "============================================================"

echo "Date: $(date)"