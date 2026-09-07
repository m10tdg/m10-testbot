#!/bin/bash
#SBATCH --job-name=qwen-finetune
#SBATCH --account=project_465003167
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/project/project_465003167/m10-testbot/logs/slurm-%j.out
#SBATCH --error=/project/project_465003167/m10-testbot/logs/slurm-%j.err

# Load modules
module load LUMI/25.03
module load cray-python
module load rocm/6.3.4

# Activate virtual environment
source /project/project_465003167/m10-testbot/finetuning/qwen-finetune-env/bin/activate

# Change to working directory
cd /project/project_465003167/m10-testbot/finetuning

# Run fine-tuning script
python finetune_qwen.py