#!/bin/bash
#SBATCH --job-name=code_regression_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --account=bgdj-delta-gpu

# Change to the correct directory
cd $HOME/spurious-regression/regress-lm-model

# Activate virtual environment (adjust path if different)
source venv/bin/activate

pip install transformers==4.53.2
pip list | grep transformers

# Run evaluation
echo "Starting Code Metrics dataset evaluation with RegressLM..."
echo ""

python -u code_metrics_regression.py

echo ""
echo "Evaluation complete!"