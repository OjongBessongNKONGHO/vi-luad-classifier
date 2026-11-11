"""
Training and evaluation for GraphPerceiverMIL on VI-LUAD.

Reports BOTH slide-level and patient-level metrics at every stage. The baseline
study found patient AUC (0.745) materially overstates slide AUC (0.692), because
patient score is max() over a patient's slides and one strong slide carries the
rest. Closing that gap is a goal of this architecture, so slide-level numbers are
tracked as a primary metric here, not a post-hoc check.

Comparison protocol vs. team15_breakingbad (ACMIL):
  - identical patient folds (see create_splits.py)
  - identical seed-ensembling rigor (--n_seeds, default 5)
  - identical probability clipping (--clip_eps, default 0.02)
Differences are the architecture: Graph Perceiver (spatial graph + GIN + GMT)
replaces ACMIL's gated-attention branches and sinusoidal positional encoding.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime

# ToSparseTensor falls back to torch.sparse_csr (torch_sparse is not installed);
# the beta-state warning fires once per batch and buries the training log.
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
warnings.filterwarnings("ignore", message=".*Converting a tensor with requires_grad.*")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from dataset import SlideGraphDataset
from model import GraphPerceiverMIL


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_score):
    """Log loss / AUC / accuracy for binary scores. AUC is nan if single-class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    probs_2d = np.stack([1.0 - y_score, y_score], axis=1)
    ll = log_loss(y_true, probs_2d, labels=[0, 1])
    auc = (roc_auc_score(y_true, y_score)
           if len(np.unique(y_true)) == 2 else float("nan"))
    acc = float(np.mean((y_score >= 0.5).astype(int) == y_true))
    return {"log_loss": float(ll), "auc": float(auc), "acc": acc}


def slide_metrics(slide_records):
    """Per-slide discrimination — every slide scored against its own label."""
    return _metrics([r["true_label"] for r in slide_records],
                    [r["prob_vi"] for r in slide_records])


def patient_aggregate(slide_records, rule="max"):
    """Aggregate slide scores to patient level.

    max: a patient is positive if any slide looks positive — vascular invasion
    need only appear in one slide. Ablation 2 of the baseline study found
    max/mean/softmax all within noise; max was marginally best.
    """
    pid_map = {}
    for r in slide_records:
        pid = r["pid"]
        pid_map.setdefault(pid, {"pid": pid, "true_label": r["true_label"],
                                 "probs": []})
        pid_map[pid]["probs"].append(r["prob_vi"])

    patients = []
    for pid, d in sorted(pid_map.items()):
        probs = np.array(d["probs"])
        if rule == "max":
            score = float(probs.max())
        elif rule == "mean":
            score = float(probs.mean())
        elif rule == "softmax":
            w = softmax(probs)
            score = float((w * probs).sum())
        else:
            raise ValueError(f"Unknown patient_agg rule: {rule}")
        patients.append({
            "pid": pid,
            "true_label": d["true_label"],
            "patient_score": score,
            "pred_label": int(score >= 0.5),
            "n_slides": len(probs),
        })
    return patients


def patient_metrics(patient_records):
    return _metrics([r["true_label"] for r in patient_records],
                    [r["patient_score"] for r in patient_records])


def calibrate_temperature(logits, labels, lr=0.01, steps=200):
    """Learn a temperature scalar on the val set (minimizes NLL)."""
    T_param = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.LBFGS([T_param], lr=lr, max_iter=steps)

    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels), dtype=torch.long)

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(logits_t / T_param, labels_t)
        loss.backward()
        return loss

    optim.step(closure)
    temp = float(T_param.item())
    # LBFGS can wander to a non-positive or absurd temperature on tiny val sets.
    if not np.isfinite(temp) or temp <= 0:
        print(f"  [WARN] temperature calibration returned {temp}; falling back to 1.0")
        return 1.0
    return temp


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels, all_pids, all_paths = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.extend(batch.y.view(-1).tolist())
        all_pids.extend(batch.pid)
        all_paths.extend(batch.slide_path)
    return (torch.cat(all_logits, dim=0).numpy(),
            all_labels, all_pids, all_paths)


def build_model(args, device):
    return GraphPerceiverMIL(
        input_dim=args.input_dim,
        emb_dim=args.emb_dim,
        num_gnn_layers=args.num_gnn_layers,
        gmt_k=args.gmt_k,
        gmt_heads=args.gmt_heads,
        gmt_encoder_blocks=args.gmt_encoder_blocks,
        drop_ratio=args.drop_ratio,
        num_classes=2,
        jk=args.jk,
    ).to(device)


def train_one_seed(fold_idx, seed, loaders, pos_weight, args, device):
    """Train a single model. Returns (test_probs, test_meta, temperature)."""
    train_loader, val_loader, test_loader = loaders

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device),
        label_smoothing=args.label_smoothing,
    )

    print(f"\n  [seed {seed}]  GraphPerceiverMIL: {n_params:,} trainable params")

    ckpt_path = os.path.join(args.checkpoint_dir,
                             f"fold_{fold_idx}_seed_{seed}.pth")
    best_score = -np.inf
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()

        val_logits, val_labels, val_pids, _ = eval_epoch(model, val_loader, device)
        val_probs = softmax(val_logits, axis=1)[:, 1]
        val_slides = [{"pid": p, "true_label": l, "prob_vi": float(pr)}
                      for p, l, pr in zip(val_pids, val_labels, val_probs)]
        val_pat = patient_metrics(patient_aggregate(val_slides, args.patient_agg))

        if epoch % 5 == 0 or epoch == 1:
            val_sl = slide_metrics(val_slides)
            print(f"    Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_pat_ll={val_pat['log_loss']:.4f} "
                  f"val_pat_auc={val_pat['auc']:.4f}  "
                  f"val_slide_auc={val_sl['auc']:.4f}")

        # Checkpoint selection. Keep this on val patient log loss.
        #
        # Tested directly: run 7069088 (log loss) vs 7071619 (patient AUC), both
        # 5 folds x 5 seeds on identical splits. Log loss won by a wide margin —
        # test patient log loss 0.614 vs 0.691, six times the ~0.013 noise floor,
        # and it took AUC with it (0.736 vs 0.712).
        #
        # The tempting argument for AUC selection — "temperature scaling is
        # monotonic, so calibrate afterwards and select on discrimination" —
        # does not survive contact with this dataset, for two reasons:
        #   1. A single scalar T fit on ~30 val patients corrects average
        #      confidence, not the miscalibration of an overfit model. Median T
        #      rose 1.67 -> 1.87 under AUC selection and test log loss still blew up.
        #   2. Val AUC over ~30 patients is a noisier signal than val log loss —
        #      it moves in coarse jumps as individual patients reorder. The epochs
        #      it selected were scattered 1..29, i.e. close to random.
        # Early stopping on log loss is doing real regularization work here.
        if args.early_stop_metric == "patient_logloss":
            score = -val_pat["log_loss"]
        elif args.early_stop_metric == "patient_auc":
            score = val_pat["auc"]
        elif args.early_stop_metric == "slide_auc":
            score = slide_metrics(val_slides)["auc"]
        else:
            raise ValueError(f"Unknown early_stop_metric: {args.early_stop_metric}")

        # A single-class val batch yields nan AUC — never let that win.
        if np.isnan(score):
            score = -np.inf

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_count = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "fold": fold_idx, "seed": seed, "epoch": epoch}, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, "
                      f"{args.early_stop_metric}={abs(best_score):.4f})")
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"    Restored epoch {ckpt.get('epoch', '?')} "
          f"(selected on val {args.early_stop_metric})")

    val_logits, val_labels, _, _ = eval_epoch(model, val_loader, device)
    temperature = calibrate_temperature(val_logits, val_labels)
    print(f"    Calibration temperature: {temperature:.4f}")

    test_logits, test_labels, test_pids, test_paths = eval_epoch(
        model, test_loader, device)
    test_probs = softmax(test_logits / temperature, axis=1)[:, 1]

    ckpt["temperature"] = temperature
    torch.save(ckpt, ckpt_path)

    meta = {"labels": test_labels, "pids": test_pids, "paths": test_paths}
    return test_probs, meta, temperature


def train_fold(fold_idx, split, args, device):
    n_train, n_val, n_test = (len(split["train"]), len(split["val"]),
                              len(split["test"]))
    print(f"\n{'=' * 72}")
    print(f"FOLD {fold_idx}  —  {n_train} train / {n_val} val / {n_test} test slides")
    print(f"{'=' * 72}")

    transform = T.ToSparseTensor(remove_edge_index=False)
    train_ds = SlideGraphDataset(split["train"], transform=transform)
    val_ds = SlideGraphDataset(split["val"], transform=transform)
    test_ds = SlideGraphDataset(split["test"], transform=transform)

    loaders = (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers),
        DataLoader(val_ds, batch_size=1, shuffle=False,
                   num_workers=args.num_workers),
        DataLoader(test_ds, batch_size=1, shuffle=False,
                   num_workers=args.num_workers),
    )

    n_neg = sum(r["label"] == 0 for r in split["train"])
    n_pos = sum(r["label"] == 1 for r in split["train"])
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  class balance: {n_neg} neg / {n_pos} pos  →  pos_weight={pos_weight:.3f}")

    # Train the seed ensemble. Slide probabilities are averaged across seeds
    # BEFORE patient aggregation, so one overconfident model cannot dominate a
    # patient's max().
    seed_probs, meta, temps = [], None, []
    for s in range(args.n_seeds):
        seed = args.base_seed + s
        probs, meta, temperature = train_one_seed(
            fold_idx, seed, loaders, pos_weight, args, device)
        seed_probs.append(probs)
        temps.append(temperature)

    ens_probs = np.mean(np.stack(seed_probs, axis=0), axis=0)
    ens_probs = np.clip(ens_probs, args.clip_eps, 1.0 - args.clip_eps)

    slide_records = [
        {
            "filename": os.path.basename(path),
            "slide_path": path,
            "pid": pid,
            "true_label": int(label),
            "prob_nonvi": float(1.0 - p),
            "prob_vi": float(p),
            "pred_label": int(p >= 0.5),
        }
        for path, pid, label, p in zip(meta["paths"], meta["pids"],
                                       meta["labels"], ens_probs)
    ]
    patient_records = patient_aggregate(slide_records, args.patient_agg)

    s_m = slide_metrics(slide_records)
    p_m = patient_metrics(patient_records)

    print(f"\n  [Fold {fold_idx} test — {args.n_seeds}-seed ensemble]")
    print(f"    {'':<14}{'Log Loss':>10}{'AUC':>10}{'Accuracy':>11}")
    print(f"    {'per-slide':<14}{s_m['log_loss']:>10.4f}{s_m['auc']:>10.4f}"
          f"{s_m['acc']:>11.4f}")
    print(f"    {'per-patient':<14}{p_m['log_loss']:>10.4f}{p_m['auc']:>10.4f}"
          f"{p_m['acc']:>11.4f}")
    print(f"    gap (patient AUC - slide AUC): {p_m['auc'] - s_m['auc']:+.4f}")

    return slide_records, patient_records, s_m, p_m, temps


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

RESULTS_FIELDS = [
    "timestamp", "config_id", "fold", "slide_auc", "slide_log_loss",
    "patient_auc", "patient_log_loss", "patient_acc", "n_seeds", "emb_dim",
    "num_gnn_layers", "gmt_k", "gmt_heads", "gmt_encoder_blocks",
    "patient_agg", "include_nontumor", "lr", "early_stop_metric", "notes",
]


def log_results(csv_path, config_id, fold, s_m, p_m, args, notes=""):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "config_id": config_id,
            "fold": fold,
            "slide_auc": s_m["auc"],
            "slide_log_loss": s_m["log_loss"],
            "patient_auc": p_m["auc"],
            "patient_log_loss": p_m["log_loss"],
            "patient_acc": p_m["acc"],
            "n_seeds": args.n_seeds,
            "emb_dim": args.emb_dim,
            "num_gnn_layers": args.num_gnn_layers,
            "gmt_k": args.gmt_k,
            "gmt_heads": args.gmt_heads,
            "gmt_encoder_blocks": args.gmt_encoder_blocks,
            "patient_agg": args.patient_agg,
            "include_nontumor": args.include_nontumor,
            "lr": args.lr,
            "early_stop_metric": args.early_stop_metric,
            "notes": notes,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Graph Perceiver MIL for VI-LUAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--splits_dir", type=str, default="splits")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--predictions_dir", type=str, default="predictions")
    p.add_argument("--results_csv", type=str, default="experiments/results.csv")
    p.add_argument("--config_id", type=str, default="graph_perceiver",
                   help="Label for this configuration in results.csv")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--folds", type=str, default=None,
                   help="Comma-separated fold subset, e.g. '0,1'. Default: all.")

    # Model
    p.add_argument("--input_dim", type=int, default=1536)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--num_gnn_layers", type=int, default=3)
    p.add_argument("--gmt_k", type=int, default=128)
    p.add_argument("--gmt_heads", type=int, default=4)
    p.add_argument("--gmt_encoder_blocks", type=int, default=2)
    p.add_argument("--drop_ratio", type=float, default=0.25)
    p.add_argument("--jk", type=str, default="sum", choices=["last", "sum"])

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--early_stop_metric", type=str, default="patient_logloss",
                   choices=["patient_logloss", "patient_auc", "slide_auc"],
                   help="Val metric for checkpoint selection. Keep patient_logloss "
                        "— it was tested against patient_auc and won decisively.")
    p.add_argument("--num_workers", type=int, default=4)

    # Ensembling / calibration — match baseline rigor for a fair comparison
    p.add_argument("--n_seeds", type=int, default=5,
                   help="Models per fold. The baseline study's noise floor is "
                        "~0.013 log loss; 1-seed results are not comparable to "
                        "the 5-seed ACMIL baseline.")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--clip_eps", type=float, default=0.02,
                   help="Clip probabilities to bound worst-case log loss.")
    p.add_argument("--patient_agg", type=str, default="max",
                   choices=["max", "mean", "softmax"])

    p.add_argument("--include_nontumor", action="store_true",
                   help="Recorded in results.csv only; NONTUMOR inclusion is "
                        "decided in create_splits.py.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    device = (torch.device(f"cuda:{args.device}")
              if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("  [WARN] no GPU visible — training on CPU will be very slow")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)

    folds = (list(range(args.n_folds)) if args.folds is None
             else [int(x) for x in args.folds.split(",")])

    all_s, all_p = [], []
    for fold_idx in folds:
        with open(os.path.join(args.splits_dir, f"fold_{fold_idx}.json")) as f:
            split = json.load(f)

        slide_recs, patient_recs, s_m, p_m, temps = train_fold(
            fold_idx, split, args, device)

        with open(os.path.join(args.predictions_dir,
                               f"fold_{fold_idx}.json"), "w") as f:
            json.dump(slide_recs, f, indent=2)
        with open(os.path.join(args.predictions_dir,
                               f"fold_{fold_idx}_patients.json"), "w") as f:
            json.dump(patient_recs, f, indent=2)

        log_results(args.results_csv, args.config_id, fold_idx, s_m, p_m,
                    args, notes=args.notes)

        all_s.append(s_m)
        all_p.append(p_m)

    # Cross-validation summary
    print("\n" + "=" * 72)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 72)

    for name, ms, score_key in (("Per-slide", all_s, "slide"),
                                ("Per-patient", all_p, "patient")):
        print(f"\n  [{name}]")
        print(f"  {'Fold':>6}  {'Log Loss':>10}  {'AUC':>8}  {'Accuracy':>10}")
        print(f"  {'-' * 6}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
        for i, m in zip(folds, ms):
            print(f"  {i:>6}  {m['log_loss']:>10.4f}  {m['auc']:>8.4f}  "
                  f"{m['acc']:>10.4f}")
        print(f"  {'-' * 6}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
        lls = [m["log_loss"] for m in ms]
        aucs = [m["auc"] for m in ms]
        accs = [m["acc"] for m in ms]
        print(f"  {'mean':>6}  {np.mean(lls):>10.4f}  {np.nanmean(aucs):>8.4f}  "
              f"{np.mean(accs):>10.4f}")
        print(f"  {'std':>6}  {np.std(lls):>10.4f}  {np.nanstd(aucs):>8.4f}  "
              f"{np.std(accs):>10.4f}")

    print(f"\n  Slide→patient AUC gap: "
          f"{np.nanmean([m['auc'] for m in all_p]) - np.nanmean([m['auc'] for m in all_s]):+.4f}"
          f"   (ACMIL baseline: +0.0530)")
    print(f"\n  Results appended to {args.results_csv} "
          f"under config_id='{args.config_id}'")
    print()


if __name__ == "__main__":
    main()
