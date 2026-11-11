"""
SlideGraphDataset: loads pre-extracted CONCH v1.5 patch embeddings from .h5,
builds a spatial 8-connectivity graph, returns a PyG Data object.

Adapted from Tsing's dataset.py for .h5 files (CONCH 768-dim features).
Each .h5 file has:
    features  : (N, 768) float32
    coords    : (N, 2)   int64
"""

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


def build_adjacency(coords: torch.Tensor) -> torch.Tensor:
    """Return edge_index (2, E) for 8-connectivity spatial graph."""
    coords_np = coords.numpy()
    index_of = {(int(c), int(r)): i for i, (c, r) in enumerate(coords_np)}

    src, dst = [], []
    for i, (c, r) in enumerate(coords_np):
        c, r = int(c), int(r)
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                j = index_of.get((c + dc, r + dr))
                if j is not None:
                    src.append(i)
                    dst.append(j)

    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


class SlideGraphDataset(Dataset):
    """PyG dataset for WSI patch-feature graphs from .h5 files."""

    def __init__(self, slide_records, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.records = slide_records

    def len(self):
        return len(self.records)

    def get(self, idx):
        rec = self.records[idx]

        with h5py.File(rec["slide_path"], "r") as f:
            features = torch.from_numpy(f["features"][:]).float()   # (N, 768)
            coords = torch.from_numpy(f["coords"][:]).long()        # (N, 2)

        edge_index = build_adjacency(coords)

        data = Data(
            x=features,
            edge_index=edge_index,
            node_coords=coords,
            y=torch.tensor([rec["label"]], dtype=torch.long),
            pid=rec["pid"],
            slide_path=rec["slide_path"],
        )
        return data
