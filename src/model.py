"""
GraphPerceiverMIL: Graph Perceiver Network for vascular invasion classification.

Architecture:
  1. Node encoder: linear projection (1536 → emb_dim)
  2. GNN message passing (GIN, 3 layers with JK-sum)
  3. Graph Multiset Transformer (GMT) pooling with Fourier positional encoding
     — reduces N patch nodes to a single slide embedding
  4. Classification head: Linear → 2 logits

Imports GNN_node and GraphMultisetAggregation from the ajpa2024 submodule.
"""

import importlib.util
import os
import sys

# GNN_node comes from the ajpa2024 submodule, but the submodule is loaded by
# explicit file path rather than by putting it on sys.path. It ships its own
# aggr.py, and a sys.path entry would shadow the local src/aggr.py — which is
# the whole point of this module (Fourier PE removed; the GNN's topology already
# encodes spatial structure, and dropping PE keeps the PMA cross-attention
# square at 256 instead of 256->282).
_AJPA_CONV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ajpa2024", "conv.py"))
if not os.path.exists(_AJPA_CONV):
    raise ImportError(
        f"ajpa2024 submodule not initialized (missing {_AJPA_CONV}). "
        "Run: git submodule update --init --recursive")
_spec = importlib.util.spec_from_file_location("ajpa2024_conv", _AJPA_CONV)
_ajpa_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ajpa_conv)
GNN_node = _ajpa_conv.GNN_node

from aggr import GraphMultisetAggregation          # noqa: E402  (src/aggr.py)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class GraphPerceiverMIL(nn.Module):
    """
    Graph Perceiver MIL for WSI-level binary classification.

    Args:
        input_dim:    Patch feature dimension (1536 for UNI2-h).
        emb_dim:      Hidden embedding dimension for GNN and GMT.
        num_gnn_layers: Number of GIN message-passing layers.
        gmt_k:        Number of latent pooling nodes in GMT.
        gmt_heads:    Number of attention heads in GMT.
        gmt_encoder_blocks: Number of SAB encoder blocks in GMT.
        drop_ratio:   Dropout probability.
        num_classes:  Output classes (2 for binary).
        jk:           Jumping knowledge type ("last" | "sum").
    """

    def __init__(
        self,
        input_dim: int = 1536,
        emb_dim: int = 256,
        num_gnn_layers: int = 3,
        gmt_k: int = 128,
        gmt_heads: int = 4,
        gmt_encoder_blocks: int = 2,
        drop_ratio: float = 0.25,
        num_classes: int = 2,
        jk: str = "sum",
    ):
        super().__init__()

        self.gnn = GNN_node(
            num_layer=num_gnn_layers,
            input_dim=input_dim,
            emb_dim=emb_dim,
            drop_ratio=drop_ratio,
            JK=jk,
            gnn_type="gin",
        )

        self.pool = GraphMultisetAggregation(
            dim=emb_dim,
            k=gmt_k,
            num_pma_blocks=1,
            num_encoder_blocks=gmt_encoder_blocks,
            heads=gmt_heads,
            dropout=drop_ratio,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim // 2, num_classes),
        )

    def forward(self, data: Data):
        """
        Args:
            data: PyG Batch / Data with fields x, edge_index (adj_t after ToSparseTensor),
                  node_coords, batch.
        Returns:
            logits: (B, num_classes)
        """
        node_emb = self.gnn(data)                              # (N_total, emb_dim)
        slide_emb = self.pool(
            x=node_emb,
            batch=data.batch,
        )                                                       # (B, emb_dim)
        return self.classifier(slide_emb)                      # (B, num_classes)
