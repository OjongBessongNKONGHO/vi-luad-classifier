#!/bin/bash -l
#
#$ -N trident-inova
#$ -P luadfrp
#$ -j y
#$ -l h_rt=48:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_type=L40S

set -euo pipefail

SRC_DIR="/restricted/projectnb/lungpath/whole_slide_HE_images/INOVA"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output/lungpath_INOVA"
SCRATCH_DIR="/scratch/${USER}/inova_$$"
CHUNK_SIZE=10

conda activate trident

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$JOB_DIR" "$SCRATCH_DIR"

python3 -c "
import os
svs = sorted([f for f in os.listdir('$SRC_DIR') if f.endswith('.svs')])
cs = $CHUNK_SIZE
for i in range((len(svs)+cs-1)//cs):
    chunk = svs[i*cs:(i+1)*cs]
    with open(f'$SCRATCH_DIR/chunk_{i:02d}.txt', 'w') as f:
        f.write('\n'.join(chunk))
print(f'Total slides: {len(svs)}, chunks: {(len(svs)+cs-1)//cs}')
"

for CHUNK_FILE in "$SCRATCH_DIR"/chunk_*.txt; do
    CHUNK_NUM=$(basename "$CHUNK_FILE" .txt)
    WSI_DIR="$SCRATCH_DIR/wsis"
    mkdir -p "$WSI_DIR"

    echo ""
    echo "=========================================="
    echo "[${CHUNK_NUM}] Copying slides to scratch..."
    echo "=========================================="

    while IFS= read -r slide; do
        cp "$SRC_DIR/$slide" "$WSI_DIR/"
    done < "$CHUNK_FILE"

    echo "[${CHUNK_NUM}] TRIDENT: CONCH v1.5 (512px) + TITAN..."
    python ~/trident/run_batch_of_slides.py \
        --task all \
        --wsi_dir "$WSI_DIR" \
        --job_dir "$JOB_DIR" \
        --patch_encoder conch_v15 \
        --patch_size 512 \
        --slide_encoder titan \
        --mag 20 \
        --skip_errors

    echo "[${CHUNK_NUM}] TRIDENT: UNI v2 (256px)..."
    python ~/trident/run_batch_of_slides.py \
        --task all \
        --wsi_dir "$WSI_DIR" \
        --job_dir "$JOB_DIR" \
        --patch_encoder uni_v2 \
        --patch_size 256 \
        --mag 20 \
        --skip_errors

    echo "[${CHUNK_NUM}] Cleaning SVS files..."
    rm -f "$WSI_DIR"/*.svs
    rmdir "$WSI_DIR" 2>/dev/null || true

    echo "[${CHUNK_NUM}] Done."
done

rm -rf "$SCRATCH_DIR"
echo "[INFO] All INOVA slides processed."
