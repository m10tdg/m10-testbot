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

# Helps PyTorch memory allocation on MI250X.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Silence tokenizer fork warnings.
# We use dataloader_num_workers=0 but this suppresses any stray warnings.
export TOKENIZERS_PARALLELISM=false

# MI250X (gfx90a) target -- lets hipBLAS/rocBLAS pick optimised kernels
# without spending time on auto-detection.
export TORCH_ROCM_AMDGPU_TARGETS="gfx90a"

# Silence the Transformers "remote code" warning.
export TRUST_REMOTE_CODE=1

# HuggingFace Datasets worker count (for map() at tokenisation).
export DATASETS_NUM_PROC=4

# Avoids a rocBLAS path lookup warning on some LUMI configurations.
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
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print('GPU memory:', round(mem, 2), 'GiB')
    if hasattr(torch.cuda, 'is_bf16_supported'):
        print('BF16 supported:', torch.cuda.is_bf16_supported())
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
for dep in ['transformers', 'peft', 'datasets', 'accelerate', 'tensorboard']:
    try:
        mod = importlib.import_module(dep)
        print('  ' + dep + ': ' + getattr(mod, '__version__', 'unknown'))
    except ImportError:
        print('  ' + dep + ': NOT INSTALLED')
"

# ============================================================================
# PATCH PRE-FLIGHT CHECK
# Verify get_imports symbol exists before wasting a queued GPU slot.
# ============================================================================

echo ""
echo "============================================================"
echo "PATCH PRE-FLIGHT CHECK"
echo "============================================================"

python -c "
import transformers.dynamic_module_utils as dmu
assert hasattr(dmu, 'get_imports'), \
    'transformers.dynamic_module_utils.get_imports not found -- patch will fail'
print('  get_imports symbol present: OK')
"

# ============================================================================
# START
# ============================================================================

echo ""
echo "============================================================"
echo "STARTING FINE-TUNING"
echo "============================================================"

python -u finetune_deepseek.py

# ============================================================================
# FINISHED
# ============================================================================

echo ""
echo "============================================================"
echo "FINE-TUNING JOB FINISHED"
echo "============================================================"
echo "Date: $(date)"