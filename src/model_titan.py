"""
TITAN Classifier -- Simple MLP on slide-level embeddings.
TITAN produces a single 768-dim vector per slide.
No attention pooling needed.
"""
import torch
import torch.nn as nn

class TITANClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch_size, 768) slide-level embedding
        Returns: logits (batch_size, 1), None (no attention weights)
        """
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits, None
