#!/bin/bash -l
#$ -N batch8_full
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch8_full_$JOB_ID.log

echo "============================================="
echo "Batch 8: Extract + Full TRIDENT Pipeline"
echo "Date: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "============================================="

TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch8_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch8_7x"
ZIP_FILE="/restricted/projectnb/luadfrp/NLST/NLST_pathhe_batch8.zip"

module load python3/3.10.12
module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

echo "Step 1: Extracting batch 8..."
unzip -j "$ZIP_FILE" -d "$WSI_DIR/"
SVS_COUNT=$(ls "$WSI_DIR"/*.svs 2>/dev/null | wc -l)
echo "Extraction done: $SVS_COUNT SVS files"

if [ "$SVS_COUNT" -eq 0 ]; then
    echo "ERROR: No SVS files found after extraction. Aborting."
    exit 1
fi

cd "$TRIDENT_DIR"

echo "Step 2: Full pipeline (seg + coords + CONCH v1.5) at $(date)"
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

echo "Step 3: UNI v2 patch features at $(date)"
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

echo "Step 4: TITAN slide-level features at $(date)"
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

echo "Batch 8 COMPLETE at $(date)"
