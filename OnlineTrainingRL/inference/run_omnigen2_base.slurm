#!/bin/bash
#SBATCH --job-name=omnigen2_base_gedit
#SBATCH --output=logs/omnigen2_base_%j.out
#SBATCH --error=logs/omnigen2_base_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --qos=mig2-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail
INFERENCE_DIR="${SLURM_SUBMIT_DIR:?Submit from inference}"
test -f "$INFERENCE_DIR/run_inference.sh" || { echo 'ERROR: cd to inference before sbatch' >&2; exit 1; }
exec bash "$INFERENCE_DIR/run_inference.sh" base "$@"
