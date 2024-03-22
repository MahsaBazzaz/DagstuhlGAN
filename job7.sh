#!/bin/bash
# SLURM directives
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=11:59:00
#SBATCH --job-name=myjob
#SBATCH --output=./log/output_%j.txt
#SBATCH --error=./log/error_%j.txt

module load anaconda3/2022.05
module load python/3.8.1
pip install -r requirements.txt

command5="python pytorch/cmain.py --experiment out_CDCGAN50 --niter"
$command5 > logs/CDCGAN.txt
