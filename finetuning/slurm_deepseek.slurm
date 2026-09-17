#!/bin/bash
#SBATCH --job-name=deepseek-lora
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.err

set -euo pipefail

export SLURM_CPU_BIND=none
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load LUMI/25.03
module load cray-python
module load rocm

VENV_PATH="/project/project_465003167/m10-testbot/finetuning/qwen-finetune-env"
source "${VENV_PATH}/bin/activate"

cd /project/project_465003167/m10-testbot/finetuning

python -u finetune_deepseek.py