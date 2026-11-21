#!/bin/bash
#SBATCH --job-name=qwen-validation
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-task=a40:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen-%j.log

set -euo pipefail
cd /scratch1/xianhanh/movie-rec-transfer/movie-rec-discovery
module load python/3.12.8
source .venv/bin/activate
export HF_HOME=/scratch1/xianhanh/hf-cache
export PYTHONUNBUFFERED=1

result_dir="outputs/qwen-validation-${SLURM_JOB_ID}"
mkdir -p outputs
cp configs/embedding-Qwen3-0.6B-discovery.json \
   "outputs/qwen-config-${SLURM_JOB_ID}.json"

date
nvidia-smi

srun python -m src.retrieval.run_embedding \
  --data data/processed/ml25m-global-v1 \
  --config "outputs/qwen-config-${SLURM_JOB_ID}.json" \
  --split validation \
  --cache cache/qwen-cuda \
  --output "$result_dir"

srun python -m src.retrieval.diagnostics \
  --data data/processed/ml25m-global-v1 \
  --predictions "$result_dir/predictions.jsonl" \
  --output "$result_dir/validation-metrics.json"

echo "FULL VALIDATION COMPLETED: $result_dir"
date
