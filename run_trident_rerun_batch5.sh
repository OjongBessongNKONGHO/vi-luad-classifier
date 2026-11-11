#!/bin/bash -l

# =============================================================================
# Batch 5: TRIDENT Full Pipeline - Artifact Removal Enabled
# Standardized v2 parameters | GrandQC segmenter | 7x artifact model
# =============================================================================

#$ -N trident_b5
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=48:00:00
#$ -j y
#$ -m ea
#$ -o /usr2/collab/onkongho/vi-experiment/logs/trident_b5_$JOB_ID.log

TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch5_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output/nlst_batch5_7x"

module load cuda/12.2
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd "$TRIDENT_DIR"

echo "============================================="
echo "Batch 5: Full Pipeline - Artifact Removal Enabled"
echo "Date: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "============================================="
echo "Parameters:"
echo "  segmenter:              grandqc"
echo "  seg_conf_thresh:        0.5"
echo "  remove_artifacts:       True"
echo "  artifact_mpp:           mpp15 (7x)"
echo "  min_tissue_proportion:  0.25"
echo "  skip_errors:            True"
echo "  mag:                    20"
echo "  patch_size:             256"
echo "============================================="

mkdir -p "$JOB_DIR"
mkdir -p /usr2/collab/onkongho/vi-experiment/logs

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

echo "============================================="
echo "Batch 5 COMPLETE at $(date)"
echo "Exit code: $?"
echo "============================================="

echo "SVS files preserved in $WSI_DIR"
