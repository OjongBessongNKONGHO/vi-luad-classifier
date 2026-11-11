#!/bin/bash -l
#
#$ -N vi-uni2-f0
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
CONFIG_ID="uni_v2_graphperc_test"

cd "$PROJ"
mkdir -p checkpoints/$CONFIG_ID predictions/$CONFIG_ID experiments/shards

module load python3/3.10.12 cuda/12.2
source /restricted/projectnb/luadfrp/tsing/vi-luad-classifier/env/bin/activate
export PYTHONPATH="$HOME/vi-experiment/lib:${PYTHONPATH:-}"

echo "[INFO] Test run: fold 0, 1 seed, UNI v2 1536-dim features"

python -u src/train_eval.py \
    --splits_dir       splits_uni_v2 \
    --checkpoint_dir   checkpoints/$CONFIG_ID \
    --predictions_dir  predictions/$CONFIG_ID \
    --results_csv      experiments/shards/${CONFIG_ID}_fold0.csv \
    --config_id        $CONFIG_ID \
    --n_seeds          1 \
    --base_seed        42 \
    --input_dim        1536 \
    --num_workers      8
