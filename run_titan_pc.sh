#!/bin/bash -l
#$ -N titan_pc
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=4:00:00
#$ -j y
#$ -m ea

source /restricted/projectnb/luadfrp/tsing/vi-luad-classifier/env/bin/activate

cd /usr2/collab/onkongho/vi-experiment

python train_titan.py \
    --splits_dir splits_titan_pc \
    --output_csv experiments/shards/titan_positive_control_sp.csv \
    --n_folds 5 \
    --n_seeds 5 \
    --epochs 50 \
    --patience 10 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --dropout 0.25 \
    --batch_size 32

echo "TITAN positive control done."
