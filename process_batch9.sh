#!/bin/bash -l
#
#$ -N trident-b8
#$ -P luadfrp
#$ -j y
#$ -l h_rt=48:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_type=L40S

set -euo pipefail

BATCH_ZIP="/restricted/projectnb/luadfrp/NLST/NLST_pathhe_batch9.zip"
JOB_DIR="/restricted/projectnb/luadfrp/trident_output/nlst_batch9"
SCRATCH_DIR="/scratch/${USER}/batch9_$$"
CHUNK_SIZE=10

conda activate trident

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$JOB_DIR" "$SCRATCH_DIR"

# Generate chunk lists with Python
python3 -c "
import zipfile, os
z = zipfile.ZipFile('$BATCH_ZIP')
svs = sorted([f for f in z.namelist() if f.endswith('.svs')])
cs = $CHUNK_SIZE
for i in range((len(svs)+cs-1)//cs):
    chunk = svs[i*cs:(i+1)*cs]
    with open(f'$SCRATCH_DIR/chunk_{i:02d}.txt', 'w') as f:
        f.write('\n'.join(chunk))
z.close()
print(f'Generated {(len(svs)+cs-1)//cs} chunk files')
"

for CHUNK_FILE in "$SCRATCH_DIR"/chunk_*.txt; do
    CHUNK_NUM=$(basename "$CHUNK_FILE" .txt)
    WSI_DIR="$SCRATCH_DIR/wsis"
    mkdir -p "$WSI_DIR"

    echo ""
    echo "=========================================="
    echo "[${CHUNK_NUM}] Extracting slides to scratch..."
    echo "=========================================="

    while IFS= read -r entry; do
        unzip -o -j "$BATCH_ZIP" "$entry" -d "$WSI_DIR"
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
echo "[INFO] All 191 slides processed."
