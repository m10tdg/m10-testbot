#!/bin/bash

#SBATCH --job-name=deepseek-coder-v2-lite-lora

#SBATCH --account=project_465003167

#SBATCH --partition=small-g

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --gpus-per-node=1

#SBATCH --mem=120G

#SBATCH --time=09:00:00

#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-deepseek-%j.out

#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-deepseek-%j.err


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

# DeepSeek-V2 uses custom CUDA/HIP kernels loaded via trust_remote_code.
# Silence the Transformers warning that flags remote code execution.
export TRUST_REMOTE_CODE=1

# Prevents HuggingFace Datasets from spawning too many processes.
export DATASETS_NUM_PROC=4


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

# Reuse the same venv as your Qwen job (peft, transformers, datasets are
# already installed).  If you created a separate one, update this path.
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
echo "DEEPSEEK-CODER-V2-LITE-INSTRUCT LoRA FINE-TUNING"
echo "============================================================"

echo "Hostname:          $(hostname)"
echo "Date:              $(date)"
echo "Working directory: $(pwd)"
echo "Python:            $(which python)"

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
print("CUDA API available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "GiB",
    )
    if hasattr(torch.cuda, "is_bf16_supported"):
        print("BF16 supported:", torch.cuda.is_bf16_supported())
PY


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo
echo "============================================================"
echo "DEPENDENCY VERSIONS"
echo "============================================================"

python - <<'PY'
import importlib

deps = [
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "tensorboard",
]

for dep in deps:
    try:
        mod = importlib.import_module(dep)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {dep}: {version}")
    except ImportError:
        print(f"  {dep}: NOT INSTALLED")
PY


# ============================================================================
# START
# ============================================================================

echo
echo "============================================================"
echo "STARTING FINE-TUNING"
echo "============================================================"

python -u finetune_deepseek.py


# ============================================================================
# FINISHED
# ============================================================================

echo
echo "============================================================"
echo "FINE-TUNING JOB FINISHED"
echo "============================================================"

echo "Date: $(date)"