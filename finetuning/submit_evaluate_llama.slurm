#!/bin/bash

#SBATCH --job-name=llama-eval
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-eval-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-llama-eval-%j.err

set -euo pipefail

# ============================================================================
# ENVIRONMENT
# ============================================================================

export SLURM_CPU_BIND=none
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_ROCM_AMDGPU_TARGETS="gfx90a"
export TRUST_REMOTE_CODE=1
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library

# Must match the LLAMA_MODEL_NAME used during finetune_llama.py.
# Baked in here so the job is reproducible without relying on the
# submitting shell's exported environment.
export LLAMA_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

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
# JOB INFO
# ============================================================================

echo "============================================================"
echo "LLAMA 3.1 8B-INSTRUCT (LoRA)  EVALUATION"
echo "============================================================"
echo "Hostname:          $(hostname)"
echo "Date:              $(date)"
echo "Working directory: $(pwd)"
echo "LLAMA_MODEL_NAME:  ${LLAMA_MODEL_NAME}"
python --version

rocm-smi || true

python -c "
import torch
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print('GPU memory:', round(mem, 2), 'GiB')
"

# ============================================================================
# RUN EVALUATION
# ============================================================================

echo ""
echo "============================================================"
echo "STARTING EVALUATION"
echo "============================================================"

python -u evaluate_llama.py

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "============================================================"
echo "EVALUATION FINISHED"
echo "============================================================"
echo "Date: $(date)"
