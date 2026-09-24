#!/bin/bash

#SBATCH --job-name=llama33-70b-lora
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-%j.err

# NOTE: Llama-3.3-70B in BF16 (~140 GB) does not fit on one MI250X GCD
# (64 GiB), so this job requests a FULL LUMI-G node (8 GCDs) and shards the
# model across all of them from a single Python process.
# If small-g rejects the request, switch to:  #SBATCH --partition=standard-g
# (and drop --mem; standard-g allocates whole nodes; max walltime is 2 days).

set -euo pipefail

# ============================================================================
# ENVIRONMENT
# ============================================================================

export SLURM_CPU_BIND=none

# Do NOT set HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES to a single GPU here:
# the model needs all 8 GCDs. Slurm already exposes the allocated ones.

# Helps PyTorch memory allocation on MI250X.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7

# MI250X (gfx90a) target.
export TORCH_ROCM_AMDGPU_TARGETS="gfx90a"

export DATASETS_NUM_PROC=4

export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library

# ----------------------------------------------------------------------------
# Hugging Face cache
# Llama-3.3 is a GATED model and LUMI compute nodes normally have no internet.
# Download it ONCE from a login node into this cache (see README/notes), then
# run fully offline here.
# ----------------------------------------------------------------------------
export HF_HOME=/project/project_465003167/m10-testbot/finetuning/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ============================================================================
# LUMI MODULES
# ============================================================================

module purge
module load LUMI/25.03
module load cray-python
module load rocm

# ============================================================================
# PYTHON ENVIRONMENT
# Needs transformers >= 4.43 (Llama 3.x rope scaling), peft, accelerate.
# The DeepSeek env pinned to transformers 4.41.2 will NOT work -- use a
# separate venv.
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
echo "LLAMA-3.3-70B-INSTRUCT LoRA FINE-TUNING"
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
n = torch.cuda.device_count()
print('GPU count:', n)
total = 0
for i in range(n):
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    total += mem
    print('  GPU', i, torch.cuda.get_device_name(i), round(mem, 1), 'GiB')
print('Total GPU memory:', round(total, 1), 'GiB')
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
# PRE-FLIGHT CHECKS (fail fast before burning GPU time)
#  1. transformers is new enough for Llama 3.x
#  2. gated model is present in the local HF cache
# ============================================================================

echo ""
echo "============================================================"
echo "PRE-FLIGHT CHECKS"
echo "============================================================"

python -c "
import transformers
from packaging import version
v = version.parse(transformers.__version__)
assert v >= version.parse('4.43.0'), (
    'transformers ' + transformers.__version__ + ' is too old for Llama 3.x '
    '(need >= 4.43.0)')
print('  transformers version OK:', transformers.__version__)

from huggingface_hub import snapshot_download
path = snapshot_download('meta-llama/Llama-3.3-70B-Instruct', local_files_only=True)
print('  Llama-3.3-70B-Instruct cached at:', path)
"

# ============================================================================
# START
# ============================================================================

echo ""
echo "============================================================"
echo "STARTING FINE-TUNING"
echo "============================================================"

python -u finetune_llama.py

# ============================================================================
# FINISHED
# ============================================================================

echo ""
echo "============================================================"
echo "FINE-TUNING JOB FINISHED"
echo "============================================================"
echo "Date: $(date)"
