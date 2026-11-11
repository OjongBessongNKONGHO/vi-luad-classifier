#!/bin/bash -l
#$ -N grandqc_b3_v2
#$ -P luadfrp
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=24:00:00
#$ -m ea
#$ -j y

GRANDQC_DIR="/usr2/collab/onkongho/grandqc/01_WSI_inference_OPENSLIDE_QC"
SVS_DIR="/restricted/projectnb/luadfrp/batch3_svs"
BATCH_OUTPUT="/restricted/projectnb/luadfrp/trident_output_v2/grandqc_masks_batch3"

cd "$GRANDQC_DIR"
mkdir -p "$BATCH_OUTPUT"

# Tissue detection (trident env)
conda activate /rprojectnb/luadfrp/onkongho/.conda/envs/trident
echo "Tissue detection for batch 3 - $(date)"
python wsi_tis_detect.py --slide_folder "$SVS_DIR" --output_dir "$BATCH_OUTPUT"

# Artifact detection (grandqc env)
conda activate grandqc
echo "Artifact detection for batch 3 - $(date)"
python main.py --slide_folder "$SVS_DIR" --output_dir "$BATCH_OUTPUT" --create_geojson "Y" --mpp_model 1.5

echo "Batch 3 GrandQC complete: $(date)"
