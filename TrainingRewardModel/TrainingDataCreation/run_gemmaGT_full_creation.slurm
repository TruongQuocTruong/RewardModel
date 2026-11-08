#!/bin/bash
#SBATCH --job-name=gemmaGT_full_creation
#SBATCH --output=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation/logs/gemmaGT_full_creation_%j.out
#SBATCH --error=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation/logs/gemmaGT_full_creation_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1g.18gb:1

set -euo pipefail
cd /datastore/truongtq/TrainingRewardModel/TrainingRewardModel/TrainingDataCreation

echo "SLURM JOB ID: $SLURM_JOB_ID  NODE: $(hostname)  START: $(date)"
python gemmaGT_data_creation.py \
    --input-dir /datastore/truongtq/TrainingRewardModel/GemmaGenGroundtruth/editreward_full_GemmaGT \
    --image-root /datastore/shared/datasets/EditReward-Data/extracted_data \
    --skip-image-check \
    --output-file editreward_overall_finegrained_gemmaGT_full.jsonl
wc -l editreward_overall_finegrained_gemmaGT_full.jsonl
echo "END: $(date)"
