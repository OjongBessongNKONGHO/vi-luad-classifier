#!/bin/bash -l
#$ -N grandqc_b9
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=24:00:00
#$ -m ea
#$ -j y

conda activate grandqc

GRANDQC_DIR="/usr2/collab/onkongho/grandqc/01_WSI_inference_OPENSLIDE_QC"
SVS_DIR="/restricted/projectnb/luadfrp/batch9_svs"
BATCH_OUTPUT="/restricted/projectnb/luadfrp/trident_output_v2_no_artifact_removal/grandqc_masks_batch9"

cd "$GRANDQC_DIR"

echo "BATCH 9 RESUME - 62 slides remaining"
echo "Started: $(date)"

python main.py \
    --slide_folder "$SVS_DIR" \
    --output_dir "$BATCH_OUTPUT" \
    --create_geojson "Y" \
    --mpp_model 1.5

echo "Batch 9 finished: $(date)"
