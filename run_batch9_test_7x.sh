#!/bin/bash -l

# =============================================================================
# Batch 9 Test Re-run: standardized params, 7x artifact model
# New output dir for comparison with old trident_output
# =============================================================================

#$ -N batch9_7x
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch9_7x_$JOB_ID.log

# Paths
TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_ZIP="/restricted/projectnb/luadfrp/NLST/NLST_pathhe_batch9.zip"
WSI_DIR="/restricted/projectnb/luadfrp/batch9_svs"
NEW_JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch9_7x"

# Load modules
module load python3/3.10.12
module load cuda/12.2

# Activate conda
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd "$TRIDENT_DIR"

echo "============================================="
echo "Batch 9 Test Re-run (7x artifact model)"
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

# Unzip SVS files if not already done
if [ ! -d "$WSI_DIR" ] || [ -z "$(ls -A "$WSI_DIR"/*.svs 2>/dev/null)" ]; then
    echo "Unzipping batch 9 SVS files..."
    mkdir -p "$WSI_DIR"
    unzip -o -j "$WSI_ZIP" -d "$WSI_DIR"
    echo "Unzipped $(ls "$WSI_DIR"/*.svs 2>/dev/null | wc -l) SVS files"
fi

# Create output directory
mkdir -p "$NEW_JOB_DIR"
mkdir -p /usr2/collab/onkongho/vi-experiment/logs

# Run TRIDENT with standardized parameters + 7x artifact model
python run_batch_of_slides.py \
    --task all \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$NEW_JOB_DIR" \
    --segmenter grandqc \
    --seg_conf_thresh 0.5 \
    --remove_artifacts \
    --artifact_mpp mpp15 \
    --min_tissue_proportion 0.25 \
    --skip_errors \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 --patch_encoder conch_v15 \
    --wsi_ext .svs

EXIT_CODE=$?

echo "============================================="
echo "Completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================="

# Do NOT delete SVS files (per Dr. Beane's request)
echo "SVS files preserved in $WSI_DIR"
