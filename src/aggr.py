"""
GraphMultisetAggregation without positional encoding.

Adapted from ajpa2024/aggr.py (Gindra et al., Am. J. Pathol. 2024).
Fourier/sincos PE removed: the GNN's message passing already encodes spatial
structure through graph topology, making absolute position redundant for VI
prediction. Removing PE also eliminates the 256→282 dimension mismatch in PMA,
keeping query and context dims identical.
"""

import math

import torch
from einops import repeat, rearrange
from torch import Tensor, einsum
from torch_geometric.utils import to_dense_batch


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def exists(val):
    return val is not None


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if context_dim is not None else None
        self.fn = fn
        torch.nn.init.constant_(self.norm.weight, 1)
        torch.nn.init.constant_(self.norm.bias, 0)
        if context_dim is not None:
            torch.nn.init.constant_(self.norm_context.weight, 1)
            torch.nn.init.constant_(self.norm_context.bias, 0)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            normed_K = self.norm_context(kwargs['K'])
            kwargs.update(K=normed_K)
        return self.fn(x, **kwargs)

    def get_attention_map(self):
        return self.fn.get_attention_map()

    def get_attn_gradients(self):
        return self.fn.get_attn_gradients()


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim, bias=False),
            torch.nn.Dropout(dropout),
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class MAB(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=1, dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        self.latent_dim = query_dim
        self.heads = heads
        self.dim_head = self.latent_dim // self.heads
        self.scale_factor = self.dim_head ** -0.5
        self.T = 1.0
        self.attend = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc_q = torch.nn.Linear(self.query_dim, self.latent_dim, bias=False)
        self.fc_kv = torch.nn.Linear(self.context_dim, self.latent_dim * 2, bias=False)
        trunc_normal_(self.fc_q.weight, std=.02)
        trunc_normal_(self.fc_kv.weight, std=.02)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, Q: Tensor, K: Tensor, mask: Tensor = None, register_hook: bool = False) -> Tensor:
        Q = self.fc_q(Q)
        K, V = self.fc_kv(K).chunk(2, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (Q, K, V))
        scaled_dot_prod = einsum('b h c d, b h n d -> b h c n', Q, K) * self.scale_factor
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(scaled_dot_prod.dtype).max
            mask = repeat(mask, 'b n -> b h () n', h=self.heads)
            scaled_dot_prod.masked_fill_(~mask, max_neg_value)
        A = self.attend(scaled_dot_prod / self.T)
        A = self.dropout(A)
        self.save_attention_map(A)
        if register_hook:
            A.register_hook(self.save_attn_gradients)
        O = einsum('b h c n, b h n d -> b h c d', A, V)
        O = rearrange(O, 'b h c d -> b c (h d)')
        return O


class SetAttentionBlock(torch.nn.Module):
    def __init__(self, query_dim, heads=1, dropout=0.0):
        super().__init__()
        self.mab = PreNorm(query_dim, MAB(query_dim, heads=heads, dropout=dropout))
        self.ff = PreNorm(query_dim, FeedForward(query_dim, query_dim, dropout=dropout))

    def forward(self, x: Tensor, register_hook=False) -> Tensor:
        kwargs = {'K': x, 'register_hook': register_hook}
        x = self.mab(x, **kwargs) + x
        x = self.ff(x) + x
        return x


class PoolingByMultiheadAttentionBlock(torch.nn.Module):
    def __init__(self, query_dim, num_seed_points=1, heads=1, dropout=0.0):
        super().__init__()
        self.query_dim = query_dim
        self.num_seed_points = num_seed_points
        self.seed = torch.nn.Parameter(torch.randn(num_seed_points, query_dim))
        # context_dim == query_dim: no PE concat, so PMA cross-attention is square
        self.mab = PreNorm(query_dim, MAB(query_dim, query_dim, heads, dropout),
                           context_dim=query_dim)
        self.ff = PreNorm(query_dim, FeedForward(query_dim, query_dim, dropout))

    def forward(self, x: Tensor, mask: Tensor, seeds=None, register_hook=False) -> Tensor:
        b = x.size(0)
        if seeds is None:
            seeds = repeat(self.seed, 'c d -> b c d', b=b)
        kwargs = {'K': x, 'mask': mask, 'register_hook': register_hook}
        x_pool = self.mab(seeds, **kwargs) + seeds
        x_pool = self.ff(x_pool) + x_pool
        return x_pool


class GraphMultisetAggregation(torch.nn.Module):
    """GMT pooling without positional encoding.

    Aggregates N patch embeddings into a single slide embedding via:
      1. PMA: k latent queries cross-attend to all N patches  O(k·N)
      2. SABs: latents self-attend                            O(k²)
      3. CLS token extraction

    Args:
        dim: Node embedding dimension (query and context are the same dim).
        k: Number of latent pooling nodes.
        num_pma_blocks: Number of PMA stages (default 1).
        num_encoder_blocks: Number of SAB layers after PMA.
        heads: Attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, dim, k, num_pma_blocks=1, num_encoder_blocks=2,
                 heads=4, dropout=0.0):
        super().__init__()
        self.k = k
        self.num_pma_blocks = num_pma_blocks

        self.layers = torch.nn.ModuleList()
        for _ in range(num_pma_blocks):
            pma = PoolingByMultiheadAttentionBlock(dim, k, heads=1, dropout=dropout)
            encoders = torch.nn.ModuleList(
                [SetAttentionBlock(dim, heads, dropout=dropout)
                 for _ in range(num_encoder_blocks)]
            )
            self.layers.append(torch.nn.ModuleList([pma, encoders]))

        self.cls_token = torch.nn.Parameter(torch.randn(1, dim))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, batch=None, node_coords=None,
                register_hook=False) -> Tensor:
        x_g, mask = to_dense_batch(x, batch)   # (B, N, dim)

        x_p = None
        for idx, (pma, encoders) in enumerate(self.layers):
            x_p = pma(x=x_g, mask=mask, seeds=x_p, register_hook=register_hook)

            if idx == len(self.layers) - 1:
                b = x_p.size(0)
                cls_tokens = repeat(self.cls_token, '1 d -> b 1 d', b=b)
                x_p = torch.cat((cls_tokens, x_p), dim=1)
                x_p = self.dropout(x_p)

            for layer in encoders:
                x_p = layer(x_p, register_hook=register_hook)

        return x_p[:, 0]   # CLS token → (B, dim)
