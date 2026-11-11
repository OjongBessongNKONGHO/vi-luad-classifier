# VI-LUAD Classifier

Graph Perceiver model for classifying visceral pleural invasion in lung adenocarcinoma whole-slide images. Downstream of the [WSI Processing Pipeline](https://github.com/OjongBessongNKONGHO/wsi-processing-pipeline), which extracts patch-level feature embeddings from gigapixel slides.

Built during a Data Engineering internship at Boston University School of Medicine, Section of Computational Biomedicine, supervised by Dr. Jennifer Beane.

## Model

Graph Perceiver MIL: GIN-based graph neural network with Graph Multiset Transformer (GMT) pooling. Takes CONCH v1.5 patch embeddings (768-dim), builds 8-connectivity spatial graphs from patch coordinates, and classifies at the slide level.

- Patient-stratified 5-fold cross-validation
- 5 seeds per fold (seed ensembling)
- Early stopping on patient-level log loss
- Temperature calibration on validation set
- Probability clipping

## Experiments

**Positive control** (in progress): binary classification of TUMOR (506 slides) vs NONTUMOR (203 slides), 709 slides total. Expected AUC near 1.0 to validate the pipeline end to end.

**Visceral invasion classification** (next): VITUMOR (214 slides) vs NONVITUMOR (292 slides). The core research question.

## Data

Features are extracted from 1,225 NLST whole-slide images across 9 batches using standardized parameters (GrandQC segmenter, artifact removal at mpp15, 256px patches at 20x). Three encoders: CONCH v1.5 (768-dim patch features), UNI v2 (1,536-dim patch features), TITAN (768-dim slide features).

## Structure

| Path | Description |
|------|-------------|
| `src/train_eval.py` | Training loop, 5-fold CV, metrics, logging |
| `src/dataset.py` | SlideGraphDataset: reads .h5, builds spatial graphs |
| `src/model.py` | GraphPerceiverMIL architecture |
| `src/aggr.py` | GMT pooling |
| `ajpa2024/` | Reference implementation (Tsing et al.) |
| `splits_positive_control/` | TUMOR vs NONTUMOR splits |
| `splits_filtered/` | VI classification splits |
| `experiments/shards/` | Results CSVs |

## Usage

```bash
# Activate environment
source /restricted/projectnb/luadfrp/tsing/vi-luad-classifier/env/bin/activate

# Run positive control (SGE)
qsub run_positive_control.sh

# Run single fold test
qsub run_train_fold0.sh
```
