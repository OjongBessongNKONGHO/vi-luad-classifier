#!/usr/bin/env python3
"""
Create patient-stratified 5-fold splits for TITAN positive control.
"""
import os, json, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold

LABEL_FILE = "/restricted/projectnb/luadfrp/VI_LUAD_Project_Medaihack/hackathon_label.txt"
FEATURE_BASE = "/restricted/projectnb/luadfrp/trident_output"
BATCH_DIRS = [f"nlst_batch{i}_7x" for i in range(3, 10)]
OUTPUT_DIR = "/usr2/collab/onkongho/vi-experiment/splits_titan_pc"
N_FOLDS = 5
SEED = 42

def find_titan_h5(slide_id):
    for bd in BATCH_DIRS:
        p = os.path.join(FEATURE_BASE, bd, "20x_256px_0px_overlap",
                         "slide_features_titan", f"{slide_id}.h5")
        if os.path.exists(p):
            return p
    return None

def main():
    df = pd.read_csv(LABEL_FILE, sep="\t")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    df["slide_id"] = df["filename"].str.replace(".svs", "", regex=False)
    df = df[df["sp.label"].isin(["SPTUMOR", "NONSPTUMOR"])].copy()
    print(f"After filtering to SPTUMOR/NONSPTUMOR: {len(df)} slides")
    df["label"] = (df["sp.label"] == "SPTUMOR").astype(int)

    records, missing_ids = [], []
    for _, row in df.iterrows():
        h5 = find_titan_h5(row["slide_id"])
        if h5:
            records.append({"slide_id": row["slide_id"], "patient_id": str(row["pid"]),
                           "label": row["label"], "feature_path": h5})
        else:
            missing_ids.append(row["slide_id"])
    print(f"Found TITAN features: {len(records)} slides")
    print(f"Missing TITAN features: {len(missing_ids)} slides (OOM during extraction)")

    slide_df = pd.DataFrame(records)
    patient_labels = slide_df.groupby("patient_id")["label"].max().reset_index()
    patients_both = slide_df.groupby("patient_id")["label"].nunique()
    patients_both = patients_both[patients_both > 1]
    if len(patients_both) > 0:
        print(f"WARNING: {len(patients_both)} patients have slides in BOTH classes!")
    else:
        print("Clean split: no patient has slides in both classes")

    n_pos = (patient_labels["label"] == 1).sum()
    n_neg = (patient_labels["label"] == 0).sum()
    print(f"Patient-level: {n_pos} SPTUMOR, {n_neg} NONSPTUMOR")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    patients = patient_labels["patient_id"].values
    labels = patient_labels["label"].values

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patients, labels)):
        test_patients = set(patients[test_idx])
        tv_patients = patients[train_val_idx]
        tv_labels = labels[train_val_idx]
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold_idx)
        inner_train_idx, inner_val_idx = next(inner_skf.split(tv_patients, tv_labels))
        train_patients = set(tv_patients[inner_train_idx])
        val_patients = set(tv_patients[inner_val_idx])

        fold_data = {"train": [], "val": [], "test": []}
        for _, row in slide_df.iterrows():
            entry = {"slide_id": row["slide_id"], "patient_id": row["patient_id"],
                     "label": int(row["label"]), "feature_path": row["feature_path"]}
            if row["patient_id"] in train_patients:
                fold_data["train"].append(entry)
            elif row["patient_id"] in val_patients:
                fold_data["val"].append(entry)
            elif row["patient_id"] in test_patients:
                fold_data["test"].append(entry)

        for split_name in ["train", "val", "test"]:
            sl = [e["label"] for e in fold_data[split_name]]
            print(f"Fold {fold_idx} {split_name}: {len(sl)} slides ({sum(sl)} pos, {len(sl)-sum(sl)} neg)")

        fold_path = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}.json")
        with open(fold_path, "w") as f:
            json.dump(fold_data, f, indent=2)

    print(f"\nDone. {N_FOLDS} folds saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
