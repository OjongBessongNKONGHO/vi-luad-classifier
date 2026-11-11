#!/bin/bash -l
#$ -N grandqc_art
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=12:00:00
#$ -m ea
#$ -j y

conda activate grandqc

GRANDQC_DIR="/usr2/collab/onkongho/grandqc/01_WSI_inference_OPENSLIDE_QC"
OUTPUT_BASE="/restricted/projectnb/luadfrp/trident_output_v2"

cd "$GRANDQC_DIR"

for BATCH in 3 4 5 6 7 8 9; do
    SVS_DIR="/restricted/projectnb/luadfrp/batch${BATCH}_svs"

    if [ ! -d "$SVS_DIR" ]; then
        echo "SKIP: $SVS_DIR does not exist"
        continue
    fi

    BATCH_OUTPUT="${OUTPUT_BASE}/grandqc_masks_batch${BATCH}"

    echo "========================================="
    echo "BATCH $BATCH - Artifact detection only"
    echo "Input:  $SVS_DIR"
    echo "Output: $BATCH_OUTPUT"
    echo "Started: $(date)"
    echo "========================================="

    python main.py \
        --slide_folder "$SVS_DIR" \
        --output_dir "$BATCH_OUTPUT" \
        --create_geojson "Y" \
        --mpp_model 1.5

    echo "Batch $BATCH finished: $(date)"
    echo ""
done

echo "All batches complete: $(date)"
