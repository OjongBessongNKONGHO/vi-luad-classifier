#!/bin/bash -l
#$ -N grandqc_b2
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=24:00:00
#$ -m ea
#$ -j y

conda activate grandqc

GRANDQC_DIR="/usr2/collab/onkongho/grandqc/01_WSI_inference_OPENSLIDE_QC"
SVS_DIR="/restricted/projectnb/luadfrp/nlst_svs/batch2_svs"
BATCH_OUTPUT="/restricted/projectnb/luadfrp/trident_output/nlst_batch2_7x/grandqc_masks"

cd "$GRANDQC_DIR"

echo "Batch 2 - 136 slides"
echo "Started: $(date)"

echo "=== Step 1: Tissue Detection ==="
python wsi_tis_detect.py \
    --slide_folder "$SVS_DIR" \
    --output_dir "$BATCH_OUTPUT"

echo "=== Step 2: QC Inference ==="
python main.py \
    --slide_folder "$SVS_DIR" \
    --output_dir "$BATCH_OUTPUT" \
    --create_geojson "Y" \
    --mpp_model 1.5

echo "Batch 2 finished: $(date)"
