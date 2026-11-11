import csv
import json
import os
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold

LABEL_FILE = "/restricted/projectnb/luadfrp/VI_LUAD_Project_Medaihack/hackathon_label.txt"
FEATURE_DIRS = [
    f"/restricted/projectnb/luadfrp/trident_output/nlst_batch{i}_7x/20x_256px_0px_overlap/features_conch_v15"
    for i in range(1, 10)
]
OUTPUT_DIR = os.path.expanduser("~/vi-experiment/splits_positive_control")
N_FOLDS = 5
SEED = 42
VAL_FRACTION = 0.15

def find_feature_path(slide_name):
    for d in FEATURE_DIRS:
        path = os.path.join(d, f"{slide_name}.h5")
        if os.path.exists(path):
            return path
    return None

def main():
    slides = []
    with open(LABEL_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            row = [field.strip().strip('"') for field in row]
            pid, filename, vi_label = row[1], row[2].replace(".svs", ""), row[3]
            if vi_label == "NONTUMOR":
                label = 0
            elif vi_label in ("VITUMOR", "NONVITUMOR"):
                label = 1
            else:
                continue
            feature_path = find_feature_path(filename)
            if feature_path is None:
                print(f"[WARN] No feature file for {filename}")
                continue
            slides.append({"slide_path": feature_path, "label": label, "pid": pid})

    print(f"Loaded {len(slides)} slides")
    print(f"  TUMOR (1): {sum(s['label']==1 for s in slides)}")
    print(f"  NONTUMOR (0): {sum(s['label']==0 for s in slides)}")

    patient_slides = defaultdict(list)
    for s in slides:
        patient_slides[s["pid"]].append(s)

    patients = sorted(patient_slides.keys())
    patient_labels = np.array([1 if any(s["label"]==1 for s in patient_slides[p]) else 0 for p in patients])
    print(f"\n{len(patients)} patients ({(patient_labels==1).sum()} TUMOR / {(patient_labels==0).sum()} NONTUMOR)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patients, patient_labels)):
        tv_patients = [patients[i] for i in trainval_idx]
        tv_labels = patient_labels[trainval_idx]
        rng = np.random.RandomState(SEED + fold_idx)
        indices = np.arange(len(tv_patients))
        rng.shuffle(indices)
        n_val = max(1, int(len(tv_patients) * VAL_FRACTION))
        pos_idx = [i for i in indices if tv_labels[i]==1]
        neg_idx = [i for i in indices if tv_labels[i]==0]
        n_val_pos = max(1, int(n_val * len(pos_idx) / len(indices)))
        n_val_neg = max(1, n_val - n_val_pos)
        val_indices = pos_idx[:n_val_pos] + neg_idx[:n_val_neg]
        train_indices = pos_idx[n_val_pos:] + neg_idx[n_val_neg:]
        train_pids = set(tv_patients[i] for i in train_indices)
        val_pids = set(tv_patients[i] for i in val_indices)
        test_pids = set(patients[i] for i in test_idx)
        assert not (train_pids & val_pids) and not (train_pids & test_pids) and not (val_pids & test_pids)

        def collect(pid_set):
            return [{"slide_path": s["slide_path"], "label": s["label"], "pid": s["pid"]}
                    for pid in sorted(pid_set) for s in patient_slides[pid]]

        split = {"train": collect(train_pids), "val": collect(val_pids), "test": collect(test_pids)}
        for part in ("train", "val", "test"):
            n0 = sum(r["label"]==0 for r in split[part])
            n1 = sum(r["label"]==1 for r in split[part])
            n_pat = len(set(r["pid"] for r in split[part]))
            print(f"  Fold {fold_idx} {part:>5}: {len(split[part]):>4} slides ({n0} NONTUMOR / {n1} TUMOR), {n_pat} patients")

        with open(os.path.join(OUTPUT_DIR, f"fold_{fold_idx}.json"), "w") as f:
            json.dump(split, f, indent=2)

    print(f"\nSplits written to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
