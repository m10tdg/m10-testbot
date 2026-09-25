#!/bin/bash

#SBATCH --job-name=llama8b-lora
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-%j.err

set -euo pipefail

# ============================================================================
# LUMI / ROCm ENVIRONMENT
# ============================================================================

export SLURM_CPU_BIND=none

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=false

# MI250X / gfx90a
export TORCH_ROCM_AMDGPU_TARGETS="gfx90a"

export DATASETS_NUM_PROC=4

export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library

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
mkdir -p llama-finetuned
mkdir -p llama-finetuned-final

# ============================================================================
# MODEL
# ============================================================================
#
# IMPORTANT:
# Meta's official Llama 3.3 model is 70B, not 8B.
#
# Set this to the exact Llama 3.3 8B-compatible Hugging Face repository
# you have selected.
#
# Example for the official 8B alternative:
#   export LLAMA_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
#
# Or:
#   export LLAMA_MODEL_NAME="your-org/your-llama-3.3-8b-model"
#
if [[ -z "${LLAMA_MODEL_NAME:-}" ]]; then
    echo "WARNING: LLAMA_MODEL_NAME is not set."
    echo "The Python script will reject its placeholder model ID."
    echo ""
    echo "Set it before sbatch, for example:"
    echo '  export LLAMA_MODEL_NAME="your-org/your-model"'
    echo ""
fi

# ============================================================================
# JOB INFORMATION
# ============================================================================

echo "============================================================"
echo "LLAMA LoRA FINE-TUNING"
echo "============================================================"

echo "Hostname:          $(hostname)"
echo "Date:              $(date)"
echo "Working directory: $(pwd)"
echo "Python:            $(which python)"

python --version

if [[ -n "${LLAMA_MODEL_NAME:-}" ]]; then
    echo "Model:             ${LLAMA_MODEL_NAME}"
else
    echo "Model:             NOT SET"
fi

# ============================================================================
# GPU INFORMATION
# ============================================================================

echo ""
echo "============================================================"
echo "ROCm / GPU CHECK"
echo "============================================================"

rocm-smi || true

python -c "
import torch

print('PyTorch:', torch.__version__)
print('ROCm/HIP:', torch.version.hip)
print('CUDA API available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

    mem = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print('GPU memory:', round(mem, 2), 'GiB')

    if hasattr(torch.cuda, 'is_bf16_supported'):
        print(
            'BF16 supported:',
            torch.cuda.is_bf16_supported()
        )
"

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo ""
echo "============================================================"
echo "DEPENDENCY VERSIONS"
echo "============================================================"

python -c "
import importlib

for dep in [
    'transformers',
    'peft',
    'datasets',
    'accelerate',
    'tensorboard',
]:
    try:
        mod = importlib.import_module(dep)
        print(
            '  ' + dep + ': ' +
            getattr(mod, '__version__', 'unknown')
        )
    except ImportError:
        print('  ' + dep + ': NOT INSTALLED')
"

# ============================================================================
# HUGGING FACE ACCESS CHECK
# ============================================================================

echo ""
echo "============================================================"
echo "HUGGING FACE CHECK"
echo "============================================================"

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN: available"
else
    echo "HF_TOKEN: not set"
    echo "If the selected model is gated, configure HF_TOKEN before sbatch."
fi

# ============================================================================
# START
# ============================================================================

echo ""
echo "============================================================"
echo "STARTING LLAMA FINE-TUNING"
echo "============================================================"

python -u finetune_llama.py

echo ""
echo "============================================================"
echo "FINE-TUNING JOB FINISHED"
echo "============================================================"

echo "Date: $(date)"
