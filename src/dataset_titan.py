"""
Dataset loader for TITAN slide-level features.
TITAN h5 files: features (768,) float32 -- single vector per slide.
Fixed-size tensors, standard batching works.
"""
import json, h5py, torch
from torch.utils.data import Dataset

class TITANDataset(Dataset):
    def __init__(self, split_json, split_name="train"):
        with open(split_json, "r") as f:
            data = json.load(f)
        self.entries = data[split_name]
        print(f"  {split_name}: {len(self.entries)} slides "
              f"({sum(e['label'] for e in self.entries)} pos, "
              f"{sum(1 - e['label'] for e in self.entries)} neg)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        with h5py.File(entry["feature_path"], "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
        label = torch.tensor(entry["label"], dtype=torch.float32)
        return features, label, entry["patient_id"]
