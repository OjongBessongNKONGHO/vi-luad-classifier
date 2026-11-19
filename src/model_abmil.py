"""
ABMIL: Attention-Based Multiple Instance Learning for WSI classification.

Gated attention (Ilse et al., ICML 2018) learns which patches in a slide
are most informative, then aggregates them into a slide-level representation.

Architecture:
  1. Feature projection: Linear(input_dim, hidden_dim) + GELU + Dropout
  2. Gated attention: softmax(w^T (tanh(V*h) * sigmoid(U*h)))
  3. Weighted aggregation: z = sum(a_k * h_k)
  4. Classifier: LayerNorm + Linear + GELU + Dropout + Linear -> 2 logits
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class ABMIL(nn.Module):
    """
    Attention-Based MIL for WSI-level binary classification.

    Args:
        input_dim:   Patch feature dimension (768 for CONCH, 1536 for UNI v2).
        hidden_dim:  Hidden embedding dimension.
        dropout:     Dropout probability.
        num_classes: Output classes (2 for binary).
    """

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.25, num_classes=2):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim // 2, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data: Data):
        """
        Args:
            data: PyG Batch with x (N_total, input_dim) and batch (N_total,).
                  Edge structure is ignored.
        Returns:
            logits: (B, num_classes)
        """
        h = self.projection(data.x)                            # (N_total, hidden_dim)

        # Gated attention scores
        V = self.attention_V(h)                                # (N_total, hidden_dim//2)
        U = self.attention_U(h)                                # (N_total, hidden_dim//2)
        attn_logits = self.attention_w(V * U).squeeze(-1)      # (N_total,)

        # Softmax within each slide (not across the whole batch)
        attn_weights = pyg_softmax(attn_logits, data.batch)    # (N_total,)

        # Weighted sum of projected features per slide
        weighted = h * attn_weights.unsqueeze(-1)              # (N_total, hidden_dim)
        slide_emb = global_add_pool(weighted, data.batch)      # (B, hidden_dim)

        return self.classifier(slide_emb)                      # (B, num_classes)
