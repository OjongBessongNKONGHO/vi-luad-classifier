#!/bin/bash -l
#$ -N batch3_v2
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=48:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch3_v2_$JOB_ID.log

TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch3_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch3_7x"

module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd "$TRIDENT_DIR"

echo "============================================"
echo "TRIDENT Batch 3 v2 - Standardized Parameters"
echo "============================================"
echo "Start time: $(date)"
echo "WSI directory: $WSI_DIR"
echo "Output directory: $JOB_DIR"
echo "SVS count: $(ls $WSI_DIR/*.svs 2>/dev/null | wc -l)"
echo "============================================"

mkdir -p "$JOB_DIR"

echo "Step 1: Full pipeline (seg + coords + CONCH v1.5) at $(date)"
python run_batch_of_slides.py \
    --task all \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --segmenter grandqc \
    --seg_conf_thresh 0.5 \
    --remove_artifacts \
    --artifact_mpp mpp15 \
    --min_tissue_proportion 0.25 \
    --skip_errors \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 \
    --patch_encoder conch_v15 \
    --wsi_ext .svs

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

echo "============================================"
echo "All steps completed at $(date)"
echo "SVS files preserved in $WSI_DIR"
echo "============================================"
