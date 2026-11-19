#!/bin/bash -l
#$ -N abmil_pc
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=24:00:00
#$ -j y
#$ -m ea

module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd /usr2/collab/onkongho/vi-experiment

python src/train_eval.py \
    --model abmil \
    --splits_dir splits_positive_control \
    --config_id abmil_positive_control_sp \
    --input_dim 768 \
    --results_csv experiments/shards/positive_control_sp_abmil.csv \
    --checkpoint_dir checkpoints/abmil_pc_sp \
    --predictions_dir predictions/abmil_pc_sp \
    --notes "ABMIL positive control, sp.label (SPTUMOR vs NONSPTUMOR), CONCH features"
