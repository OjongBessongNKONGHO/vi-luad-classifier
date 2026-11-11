#!/bin/bash -l

# =============================================================================
# Batch 9: UNI v2 patch features + TITAN slide features
# Feature-only run on existing job_dir (seg + coords already complete)
# Requested by Dr. Beane for hackathon comparison
# =============================================================================

#$ -N batch9_univ2_titan
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /usr2/collab/onkongho/vi-experiment/logs/batch9_univ2_titan_$JOB_ID.log

# Paths
TRIDENT_DIR="/usr2/collab/onkongho/trident"
WSI_DIR="/restricted/projectnb/luadfrp/batch9_svs"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output_v2/nlst_batch9_7x"

# Load modules
module load python3/3.10.12
module load cuda/12.2

# Activate conda
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident

cd "$TRIDENT_DIR"

echo "============================================="
echo "Batch 9: UNI v2 + TITAN feature extraction"
echo "Date: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "============================================="
echo "Parameters:"
echo "  task:           feat (feature extraction only)"
echo "  patch_encoder:  uni_v2"
echo "  slide_encoder:  titan"
echo "  max_workers:    3"
echo "  mag:            20"
echo "  patch_size:     256"
echo "  job_dir:        $JOB_DIR (reusing existing seg+coords)"
echo "============================================="

# Run feature extraction only
python run_batch_of_slides.py \
    --task feat \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --mag 20 \
    --patch_size 256 \
    --max_workers 3 \
    --patch_encoder uni_v2 \
    --slide_encoder titan \
    --skip_errors \
    --wsi_ext .svs

EXIT_CODE=$?

echo "============================================="
echo "Completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================="
