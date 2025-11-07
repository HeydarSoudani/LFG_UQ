#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=4:00:00
#SBATCH --mem=80GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
dataset="factscore_bio"
retriever_name="bm25"
claim_evaluation_method="RAFE"
run="run_1"


accelerate launch --multi_gpu $HOME/LONGFORM_RAG_UE/modules/generation/run_module.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --retriever_name "$retriever_name" \
    --claim_evaluation_method "$claim_evaluation_method" \
    --run "$run"

