#!/bin/bash -l
#$ -N grandqc_b1
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=24:00:00
#$ -m ea
#$ -j y

conda activate grandqc

GRANDQC_DIR="/usr2/collab/onkongho/grandqc/01_WSI_inference_OPENSLIDE_QC"
SVS_DIR="/restricted/projectnb/luadfrp/nlst_svs/batch1_svs"
BATCH_OUTPUT="/restricted/projectnb/luadfrp/trident_output/nlst_batch1_7x/grandqc_masks"

cd "$GRANDQC_DIR"

mkdir -p "$BATCH_OUTPUT"

SLIDE_COUNT=$(ls "$SVS_DIR"/*.svs 2>/dev/null | wc -l)

echo "========================================="
echo "BATCH 1 - $SLIDE_COUNT slides"
echo "Input:  $SVS_DIR"
echo "Output: $BATCH_OUTPUT"
echo "Started: $(date)"
echo "========================================="

# Tissue detection (~0.4s per slide)
echo "Running tissue detection..."
python wsi_tis_detect.py \
    --slide_folder "$SVS_DIR" \
    --output_dir "$BATCH_OUTPUT"

# Artifact detection with 7x model (~35s per slide)
echo "Running artifact detection..."
python main.py \
    --slide_folder "$SVS_DIR" \
    --output_dir "$BATCH_OUTPUT" \
    --create_geojson "Y" \
    --mpp_model 1.5

echo "Batch 1 finished: $(date)"
