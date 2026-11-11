#!/bin/bash -l

#$ -N batch9_univ2
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch9_univ2_$JOB_ID.log

TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch9_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch9_7x"

module load python3/3.10.12
module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident
cd "$TRIDENT_DIR"

echo "============================================="
echo "Batch 9: UNI v2 patch feature extraction"
echo "Date: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "============================================="

python run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 \
    --patch_encoder uni_v2 \
    --skip_errors \
    --wsi_ext .svs

echo "============================================="
echo "Completed at $(date)"
echo "Exit code: $?"
echo "============================================="
