#!/bin/bash -l
#$ -N batch7_rerun
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch7_rerun_$JOB_ID.log

echo "============================================="
echo "Batch 7 RERUN: Full TRIDENT Pipeline"
echo "Date: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "============================================="

TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch7_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch7_7x"

module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd "$TRIDENT_DIR"

# Step 1: Full pipeline with CONCH v1.5 (seg + coords + features)
echo "Step 1: Full pipeline (seg + coords + CONCH v1.5) at $(date)"
python run_batch_of_slides.py \
    --task all \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 \
    --patch_encoder conch_v15 \
    --skip_errors \
    --wsi_ext .svs

# Step 2: UNI v2 patch features
echo "Step 2: UNI v2 patch features at $(date)"
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

# Step 3: TITAN slide-level features
echo "Step 3: TITAN slide-level features at $(date)"
python run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 \
    --slide_encoder titan \
    --skip_errors \
    --wsi_ext .svs

echo "============================================="
echo "Batch 7 RERUN COMPLETE at $(date)"
echo "Exit code: $?"
echo "============================================="
