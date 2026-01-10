#!/bin/bash
#SBATCH --job-name=cyclegan
#SBATCH --partition=rtx2080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cyclegan_%j.out
#SBATCH --error=logs/cyclegan_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
source ~/venv/bin/activate
python train/train_cyclegan.py

