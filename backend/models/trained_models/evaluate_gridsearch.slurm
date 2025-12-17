#!/bin/bash
#SBATCH --job-name=gnn_final_eval
#SBATCH --partition=amc
#SBATCH --qos=amc
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/eval_%A_%a.out
#SBATCH --error=slurm_logs/eval_%A_%a.err
#SBATCH --array=0-$(($(wc -l < trained_models/model_combinations.txt) - 1))

module purge
source /projects/ropo8156/miniconda3/etc/profile.d/conda.sh
conda activate powernn
COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" trained_models/model_combinations.txt)
CLASSIFIER_MODEL=$(echo $COMBO | cut -d',' -f1)
ESTIMATOR_MODEL=$(echo $COMBO | cut -d',' -f2)
RESULTS_FILE="results_gridsearch.csv"
python evaluate.py --classifier "trained_models/$CLASSIFIER_MODEL" --estimator "trained_models/$ESTIMATOR_MODEL" --results_file "$RESULTS_FILE"
