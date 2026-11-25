#!/usr/bin/env python3
"""
TITAN positive control training.
5-fold CV x 5-seed ensembling, temperature calibration, patient max aggregation.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model_titan import TITANClassifier
from dataset_titan import TITANDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for features, labels, _ in loader:
        features, labels = features.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        n += features.size(0)
    return total_loss / n

def evaluate(model, loader, device, temperature=1.0):
    model.eval()
    all_logits, all_labels, all_patients = [], [], []
    with torch.no_grad():
        for features, labels, pids in loader:
            features = features.to(device)
            logits, _ = model(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_patients.extend(pids)

    all_logits = torch.cat(all_logits, 0).squeeze(1)
    all_labels = torch.cat(all_labels, 0)
    scaled_logits = all_logits / temperature
    slide_probs = torch.sigmoid(scaled_logits).numpy()
    slide_labels = all_labels.numpy()
    eps = 1e-7

    slide_auc = roc_auc_score(slide_labels, slide_probs)
    slide_probs_c = np.clip(slide_probs, eps, 1 - eps)
    slide_logloss = log_loss(slide_labels, slide_probs_c)

    patient_probs_d, patient_labels_d = defaultdict(list), {}
    for prob, lbl, pid in zip(slide_probs, slide_labels, all_patients):
        patient_probs_d[pid].append(prob)
        patient_labels_d[pid] = lbl

    p_probs, p_labels = [], []
    for pid in sorted(patient_probs_d.keys()):
        p_probs.append(max(patient_probs_d[pid]))
        p_labels.append(patient_labels_d[pid])
    p_probs, p_labels = np.array(p_probs), np.array(p_labels)

    patient_auc = roc_auc_score(p_labels, p_probs)
    p_probs_c = np.clip(p_probs, eps, 1 - eps)
    patient_logloss = log_loss(p_labels, p_probs_c)
    patient_acc = accuracy_score(p_labels, (p_probs >= 0.5).astype(int))

    return {"slide_auc": slide_auc, "slide_logloss": slide_logloss,
            "patient_auc": patient_auc, "patient_logloss": patient_logloss,
            "patient_acc": patient_acc, "slide_probs": slide_probs,
            "slide_labels": slide_labels, "patient_probs": p_probs,
            "patient_labels": p_labels, "all_patients": all_patients}

def learn_temperature(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for features, labels, _ in loader:
            features = features.to(device)
            logits, _ = model(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0).unsqueeze(1)
    temperature = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    def eval_fn():
        opt.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    opt.step(eval_fn)
    return temperature.item()

def collate_fn(batch):
    features = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pids = [b[2] for b in batch]
    return features, labels, pids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="/usr2/collab/onkongho/vi-experiment/splits_titan_pc")
    parser.add_argument("--output_csv", default="/usr2/collab/onkongho/vi-experiment/experiments/shards/titan_positive_control_sp.csv")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    all_results = []

    for fold in range(args.n_folds):
        split_json = os.path.join(args.splits_dir, f"fold_{fold}.json")
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
        fold_seed_probs = []

        for seed_idx in range(args.n_seeds):
            seed = args.base_seed + seed_idx
            set_seed(seed)
            print(f"\n--- Seed {seed} ---")

            train_ds = TITANDataset(split_json, "train")
            val_ds = TITANDataset(split_json, "val")
            test_ds = TITANDataset(split_json, "test")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            model = TITANClassifier(input_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.BCEWithLogitsLoss()

            best_val_ploss = float("inf")
            best_epoch, best_state, epochs_no_improve = 0, None, 0

            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_m = evaluate(model, val_loader, device)
                if val_m["patient_logloss"] < best_val_ploss:
                    best_val_ploss = val_m["patient_logloss"]
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epoch % 10 == 0 or epochs_no_improve == 0:
                    print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_ploss={val_m['patient_logloss']:.4f} val_pauc={val_m['patient_auc']:.4f}{' *' if epochs_no_improve == 0 else ''}")
                if epochs_no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                    break

            model.load_state_dict(best_state)
            temperature = learn_temperature(model, val_loader, device)
            print(f"  Temperature: {temperature:.4f}")
            test_m = evaluate(model, test_loader, device, temperature)
            print(f"  Test slide AUC:   {test_m['slide_auc']:.4f}")
            print(f"  Test patient AUC: {test_m['patient_auc']:.4f}")
            fold_seed_probs.append(test_m)

        slide_probs_ens = np.mean([s["slide_probs"] for s in fold_seed_probs], axis=0)
        patient_probs_ens = np.mean([s["patient_probs"] for s in fold_seed_probs], axis=0)
        sl, pl = fold_seed_probs[0]["slide_labels"], fold_seed_probs[0]["patient_labels"]
        eps = 1e-7

        slide_auc_ens = roc_auc_score(sl, slide_probs_ens)
        patient_auc_ens = roc_auc_score(pl, patient_probs_ens)
        patient_logloss_ens = log_loss(pl, np.clip(patient_probs_ens, eps, 1-eps))
        patient_acc_ens = accuracy_score(pl, (patient_probs_ens >= 0.5).astype(int))

        print(f"\nFold {fold} ENSEMBLE ({args.n_seeds} seeds):")
        print(f"  Slide AUC:       {slide_auc_ens:.4f}")
        print(f"  Patient AUC:     {patient_auc_ens:.4f}")
        print(f"  Patient log loss: {patient_logloss_ens:.4f}")
        print(f"  Patient acc:     {patient_acc_ens:.4f}")
        all_results.append({"fold": fold, "slide_auc": slide_auc_ens,
                           "patient_auc": patient_auc_ens, "patient_logloss": patient_logloss_ens,
                           "patient_acc": patient_acc_ens})

    results_df = pd.DataFrame(all_results)
    print(f"\n{'='*60}\nFINAL RESULTS (TITAN positive control, sp.label)\n{'='*60}")
    print(f"Mean slide AUC:   {results_df['slide_auc'].mean():.4f} +/- {results_df['slide_auc'].std():.4f}")
    print(f"Mean patient AUC: {results_df['patient_auc'].mean():.4f} +/- {results_df['patient_auc'].std():.4f}")
    print(f"Mean patient log loss: {results_df['patient_logloss'].mean():.4f} +/- {results_df['patient_logloss'].std():.4f}")
    print(f"Mean patient acc: {results_df['patient_acc'].mean():.4f} +/- {results_df['patient_acc'].std():.4f}")
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
