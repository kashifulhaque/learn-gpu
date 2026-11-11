"""Inputs, float64 references, and eager baselines for the capstone lab (chapter 41).

The tests and the reader's starter both import this file. Nothing here uses
CUDA, so it also runs on the CPU. The eager baselines are what the starter
begins with, and the tests time them as the leaderboard's reference.
"""

from __future__ import annotations

# Luma weights (ITU-R BT.601) and the 3 by 3 blur [1 2 1] x [1 2 1] / 16.
GRAY_WEIGHTS = (0.299, 0.587, 0.114)
BLUR_1D = (1.0, 2.0, 1.0)

# The leaderboard shapes: (H, W), (rows, vocabulary, k), and (M, d_model, d_hidden).
GRAY_SHAPE = (4096, 4096)
TOPK_SHAPE = (4096, 32000, 8)
MLP_SHAPE = (4096, 1024, 4096)


def make_image(h, w, seed=0, device="cpu"):
    """A float32 RGB image of shape (h, w, 3) with values in [0, 1)."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.rand(h, w, 3, generator=gen).to(device)


def make_logits(rows, n, seed=0, ties=False, device="cpu"):
    """Float32 logits. With ties=True the values are rounded to one decimal, so a row has many equal values."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=gen)
    if ties:
        x = (x * 4).round() / 4
    return x.to(device)


def make_mlp(m, d, h, seed=0, device="cpu"):
    """bf16 inputs for y = gelu(x w1 + b1) w2 + b2, scaled so activations stay near 1."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, d, generator=gen)
    w1 = torch.randn(d, h, generator=gen) / d**0.5
    b1 = torch.randn(h, generator=gen) * 0.5
    w2 = torch.randn(h, d, generator=gen) / h**0.5
    b2 = torch.randn(d, generator=gen) * 0.5
    return tuple(t.to(torch.bfloat16).to(device) for t in (x, w1, b1, w2, b2))


def grayscale_blur_ref(x):
    """Float64 reference: luma, then the 3 by 3 blur with zero padding outside the image."""
    import torch

    g = x.double() @ torch.tensor(GRAY_WEIGHTS, dtype=torch.float64, device=x.device)
    h, w = g.shape
    pad = torch.zeros(h + 2, w + 2, dtype=torch.float64, device=x.device)
    pad[1:-1, 1:-1] = g
    out = torch.zeros(h, w, dtype=torch.float64, device=x.device)
    for dr in range(3):
        for dc in range(3):
            out += BLUR_1D[dr] * BLUR_1D[dc] / 16.0 * pad[dr : dr + h, dc : dc + w]
    return out


def topk_ref(x, k):
    """Reference: a stable descending sort, so a tie keeps the lower index first (chapter 30)."""
    import torch

    values, indices = torch.sort(x, dim=-1, descending=True, stable=True)
    return values[:, :k], indices[:, :k]


def mlp_ref(x, w1, b1, w2, b2):
    """Float64 reference with exact (erf) GELU and no rounding of the hidden layer."""
    import torch

    h = x.double() @ w1.double() + b1.double()
    h = 0.5 * h * (1.0 + torch.special.erf(h / 2**0.5))
    return h @ w2.double() + b2.double()


# Eager baselines: correct, unfused, and what the starter begins with.


def eager_grayscale_blur(x):
    import torch
    import torch.nn.functional as F

    g = x @ torch.tensor(GRAY_WEIGHTS, dtype=x.dtype, device=x.device)
    k1 = torch.tensor(BLUR_1D, dtype=x.dtype, device=x.device)
    kernel = (k1[:, None] * k1[None, :] / 16.0)[None, None]
    return F.conv2d(g[None, None], kernel, padding=1)[0, 0]


def eager_topk(x, k):
    values, indices = topk_ref(x, k)
    return values, indices


def eager_mlp(x, w1, b1, w2, b2):
    import torch
    import torch.nn.functional as F

    h = F.gelu(torch.addmm(b1, x, w1))
    return torch.addmm(b2, h, w2)
