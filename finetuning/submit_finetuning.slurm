#!/bin/bash
#SBATCH --job-name=qwen-finetune
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/finetuning/logs/slurm-%j.err

export SLURM_CPU_BIND="none"

# Use the current supported software stack on LUMI
module purge
module load LUMI/25.03
module load cray-python
module load rocm

# Set environment path explicitly before activating
VENV_PATH="/project/project_465003167/m10-testbot/finetuning/qwen-finetune-env"
export PATH="$VENV_PATH/bin:$PATH"
source "$VENV_PATH/bin/activate"

cd /project/project_465003167/m10-testbot/finetuning
mkdir -p logs

# Execute script via virtual environment python directly
srun --cpu-bind=none python finetune_qwen.py