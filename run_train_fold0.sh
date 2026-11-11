#!/bin/bash -l
#
#$ -N vi-conch-f0
#$ -P luadfrp
#$ -j y
#$ -l h_rt=4:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_type=L40S

set -uo pipefail

PROJ=$HOME/vi-experiment
cd "$PROJ" || { echo "[ERROR] Cannot cd to $PROJ"; exit 1; }

export TORCH_HOME="$PROJ/.cache/torch"
export HF_HOME="$PROJ/.cache/hf"
mkdir -p "$TORCH_HOME" "$HF_HOME"

module load python3/3.12.4
# Use Tsing's env (has torch_geometric and dependencies)
source /restricted/projectnb/luadfrp/tsing/vi-luad-classifier/env/bin/activate
export PYTHONPATH="$HOME/vi-experiment/lib:${PYTHONPATH:-}"

echo "[INFO] Working directory: $(pwd)"
nvidia-smi || echo "[WARN] nvidia-smi unavailable"

CONFIG_ID="conch_graphperc_test"
mkdir -p checkpoints/$CONFIG_ID predictions/$CONFIG_ID experiments/shards

echo "[INFO] Test run: fold 0, 1 seed, CONCH 768-dim features"

python -u src/train_eval.py \
    --splits_dir       splits_filtered \
    --checkpoint_dir   checkpoints/$CONFIG_ID \
    --predictions_dir  predictions/$CONFIG_ID \
    --results_csv      experiments/shards/${CONFIG_ID}_fold0.csv \
    --config_id        $CONFIG_ID \
    --n_folds          5 \
    --folds            0 \
    --emb_dim          256 \
    --num_gnn_layers   3 \
    --gmt_k            128 \
    --gmt_heads        4 \
    --gmt_encoder_blocks 2 \
    --drop_ratio       0.25 \
    --jk               sum \
    --epochs           50 \
    --batch_size       4 \
    --lr               2e-4 \
    --weight_decay     1e-4 \
    --patience         10 \
    --label_smoothing  0.05 \
    --early_stop_metric patient_logloss \
    --n_seeds          1 \
    --base_seed        42 \
    --clip_eps         0.02 \
    --patient_agg      max \
    --input_dim        768 \
    --num_workers      8

echo "[INFO] Finished with exit code $?"
