"""
Inference script for the Graph Perceiver MIL ensemble.

Loads one checkpoint per fold (5 total), runs each model on the provided
feature .pt files, and writes per-patient predictions to stdout as JSON.

Usage (the organizer runs this):
    python src/predict.py \
        --features_dir /path/to/processed \
        --checkpoints_dir checkpoints \
        --slide_list slides.txt \
        --output_json predictions/test_patients.json

The model must accept a feature tensor and return logits (B, 2) as its first
output element. Patient score = max P(VITUMOR) across slides, averaged over
the 5-fold ensemble.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from scipy.special import softmax
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import SlideGraphDataset
from model import GraphPerceiverMIL


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = GraphPerceiverMIL(
        input_dim=args.get("input_dim", 1536),
        emb_dim=args.get("emb_dim", 256),
        num_gnn_layers=args.get("num_gnn_layers", 3),
        gmt_k=args.get("gmt_k", 128),
        gmt_heads=args.get("gmt_heads", 4),
        gmt_encoder_blocks=args.get("gmt_encoder_blocks", 2),
        drop_ratio=0.0,
        num_classes=2,
        jk=args.get("jk", "sum"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("temperature", 1.0)


@torch.no_grad()
def predict_slides(model, loader, temperature, device):
    """Returns dict: slide_path → P(VITUMOR) after temperature scaling."""
    results = {}
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)                       # (B, 2)
        probs = softmax(logits.cpu().numpy() / temperature, axis=1)
        for path, prob in zip(batch.slide_path, probs):
            results[path] = float(np.clip(prob[1], 1e-7, 1 - 1e-7))
    return results


def patient_aggregate_ensemble(ensemble_slide_probs, slide_records):
    """Average each slide's P(VITUMOR) across ensemble models, then take patient max."""
    # ensemble_slide_probs: list of dicts {slide_path: prob_vi}
    n_models = len(ensemble_slide_probs)
    pid_probs = {}
    pid_label = {}

    for rec in slide_records:
        path = rec["slide_path"]
        avg_prob = np.mean([m.get(path, 0.5) for m in ensemble_slide_probs])
        pid = rec["pid"]
        pid_probs.setdefault(pid, []).append(avg_prob)
        if "label" in rec:
            pid_label[pid] = rec["label"]

    patients = []
    for pid, probs in pid_probs.items():
        score = float(np.max(probs))
        patients.append({
            "pid": pid,
            "patient_score": score,
            "pred_label": int(score >= 0.5),
            "n_slides": len(probs),
            **({"true_label": pid_label[pid]} if pid in pid_label else {}),
        })
    return patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir",    type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--slide_list",      type=str, required=True,
                        help="Text file: one slide filename per line (no label needed)")
    parser.add_argument("--output_json",     type=str, default="predictions/test_patients.json")
    parser.add_argument("--n_folds",         type=int, default=5)
    parser.add_argument("--device",          type=int, default=0)
    args = parser.parse_args()

    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() else torch.device("cpu"))

    # Build slide records (no labels for test set)
    with open(args.slide_list) as f:
        filenames = [line.strip() for line in f if line.strip()]

    slide_records = [
        {
            "slide_path": os.path.join(args.features_dir, fn),
            "pid": fn.replace(".pt", "").split("_")[0],  # adjust if needed
            "label": -1,
        }
        for fn in filenames
    ]

    transform = T.ToSparseTensor(remove_edge_index=False)
    dataset = SlideGraphDataset(slide_records, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Run every checkpoint in the ensemble. train_eval.py writes one per
    # (fold, seed) as fold_{i}_seed_{s}.pth, so a 5-fold x 5-seed run yields 25;
    # the older fold_{i}.pth layout is still picked up if present.
    ckpt_paths = sorted(glob.glob(
        os.path.join(args.checkpoints_dir, "fold_*_seed_*.pth")))
    if not ckpt_paths:
        ckpt_paths = sorted(glob.glob(
            os.path.join(args.checkpoints_dir, "fold_*.pth")))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints found in {args.checkpoints_dir}")

    print(f"Ensembling {len(ckpt_paths)} checkpoints")
    ensemble_preds = []
    for ckpt_path in ckpt_paths:
        model, temperature = load_model(ckpt_path, device)
        ensemble_preds.append(
            predict_slides(model, loader, temperature, device))
        print(f"  {os.path.basename(ckpt_path)} done (T={temperature:.3f})")

    patients = patient_aggregate_ensemble(ensemble_preds, slide_records)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(patients, f, indent=2)
    print(f"Predictions saved to {args.output_json}")


if __name__ == "__main__":
    main()
