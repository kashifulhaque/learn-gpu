"""Lab 37: chunked scans for linear attention. The worked solution.

The CPU half builds the pieces in torch float64: the associative operator of a
first-order recurrence, a sequential and a parallel scan over it, a sequential
linear-attention reference with a scalar decay, the chunked algorithm, a
log-space decay matrix, and a cost model. The GPU half is a Triton kernel that
runs one chunked forward pass per head. Triton exists only on the A100 worker,
so the import is guarded and the CPU checks never touch it.

Conventions. A decay is passed as its logarithm, `logg`, with `g = exp(logg)`
and `logg <= 0`. The state is S with shape (dk, dv):

    S_t = g_t * S_{t-1} + outer(k_t, v_t)        o_t = q_t @ S_t
"""

from __future__ import annotations

import math

import torch

try:  # Triton is installed on the A100 worker only.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the CPU path
    triton = None
    tl = None


# --------------------------------------------------------------------------
# 1. The operator and the scans
# --------------------------------------------------------------------------


def _expand(a, like):
    """Reshape `a` so that it broadcasts against `like` along its trailing dims."""
    a = torch.as_tensor(a)
    return a.reshape(a.shape + (1,) * (like.dim() - a.dim()))


def combine(x, y):
    """Compose two steps of h = a * h + b. `x` runs first, then `y`.

    Applying (a1, b1) and then (a2, b2) maps h to a2 * (a1 * h + b1) + b2,
    which is the pair (a1 * a2, a2 * b1 + b2).
    """
    a1, b1 = x
    a2, b2 = y
    return a1 * a2, _expand(a2, b1) * b1 + b2


def scan_sequential(a, b, h0=None):
    """h_t = a_t * h_{t-1} + b_t along dim 0, one step at a time."""
    h = torch.zeros_like(b[0]) if h0 is None else h0
    out = []
    for t in range(b.shape[0]):
        h = _expand(a[t], b[t]) * h + b[t]
        out.append(h)
    return torch.stack(out)


def scan_parallel(a, b):
    """The same recurrence from h_{-1} = 0, by a Hillis-Steele scan over `combine`.

    Step d = 1, 2, 4, ... replaces element i >= d by combine(element i - d,
    element i). After ceil(log2 T) steps, element i holds the composition of
    steps 0 through i. Its b component is h_i.
    """
    A, B = a.clone(), b.clone()
    t = A.shape[0]
    d = 1
    while d < t:
        na, nb = combine((A[:-d], B[:-d]), (A[d:], B[d:]))
        A = torch.cat([A[:d], na])
        B = torch.cat([B[:d], nb])
        d *= 2
    return B


# --------------------------------------------------------------------------
# 2. Linear attention with a scalar decay
# --------------------------------------------------------------------------


def linear_attention_sequential(q, k, v, logg):
    """The reference: one state update per token. Returns (o, S_final)."""
    s = torch.zeros(k.shape[1], v.shape[1], dtype=q.dtype)
    out = []
    for t in range(q.shape[0]):
        s = torch.exp(logg[t]) * s + torch.outer(k[t], v[t])
        out.append(q[t] @ s)
    return torch.stack(out), s


def decay_matrix(logg):
    """D[i, j] = exp(sum of logg over j+1..i) for j <= i, and 0 above the diagonal.

    Mask the exponent before exp. exp(b_i - b_j) above the diagonal is large
    and positive when the decays are small, and inf * 0 is NaN.
    """
    b = torch.cumsum(logg, 0)
    diff = b[:, None] - b[None, :]
    keep = torch.tril(torch.ones_like(diff, dtype=torch.bool))
    return torch.exp(torch.where(keep, diff, torch.full_like(diff, -math.inf)))


# --------------------------------------------------------------------------
# 3. The chunked algorithm
# --------------------------------------------------------------------------


def _bounds(t, chunk):
    return [(s, min(s + chunk, t)) for s in range(0, t, chunk)]


def chunk_states(k, v, logg, chunk):
    """Each chunk as one step of the recurrence S = A * S + U.

    A_c = exp(sum of logg in the chunk), a scalar. U_c is the state the chunk
    would build from zero: sum over j of exp(b_last - b_j) * outer(k_j, v_j),
    where b is the cumulative log decay inside the chunk.
    Returns (A, U) with shapes (N,) and (N, dk, dv).
    """
    a_list, u_list = [], []
    for s, e in _bounds(k.shape[0], chunk):
        b = torch.cumsum(logg[s:e], 0)
        w = torch.exp(b[-1] - b)
        a_list.append(torch.exp(b[-1]))
        u_list.append((k[s:e] * w[:, None]).T @ v[s:e])
    return torch.stack(a_list), torch.stack(u_list)


def entering_states(a, u):
    """The state that enters each chunk, and the final state.

    Scan the chunk steps with `scan_parallel`. Entering state c is the scan's
    result after chunk c - 1, and the first chunk enters with zeros.
    """
    after = scan_parallel(a, u)
    entering = torch.cat([torch.zeros_like(after[:1]), after[:-1]])
    return entering, after[-1]


def chunked_linear_attention(q, k, v, logg, chunk):
    """The chunked algorithm. Returns (o, S_final), equal to the sequential reference."""
    a, u = chunk_states(k, v, logg, chunk)
    s_in, s_final = entering_states(a, u)
    out = torch.zeros(q.shape[0], v.shape[1], dtype=q.dtype)
    for c, (s, e) in enumerate(_bounds(q.shape[0], chunk)):
        b = torch.cumsum(logg[s:e], 0)
        intra = ((q[s:e] @ k[s:e].T) * decay_matrix(logg[s:e])) @ v[s:e]
        inter = torch.exp(b)[:, None] * (q[s:e] @ s_in[c])
        out[s:e] = intra + inter
    return out, s_final


# --------------------------------------------------------------------------
# 4. The cost model
# --------------------------------------------------------------------------


def chunk_flops(t, chunk, dk, dv):
    """FLOPs of the chunked forward pass, counting every full C by C block.

    Per chunk: Q K^T (2 C^2 dk), the masked scores times V (2 C^2 dv),
    q @ S (2 C dk dv), and the state update K^T V (2 C dk dv).
    """
    n = math.ceil(t / chunk)
    return n * (2 * chunk * chunk * (dk + dv) + 4 * chunk * dk * dv)


def chunk_bytes(t, chunk, dk, dv, io_bytes=2, state_bytes=4):
    """HBM bytes when the chunk states go through memory (the three-phase form).

    q and k (dk each), v and the output o (dv each) cross HBM once per token,
    in `io_bytes`. Every chunk's state (dk by dv, `state_bytes`) is written by
    one phase and read back by the next.
    """
    n = math.ceil(t / chunk)
    return t * (2 * dk + 2 * dv) * io_bytes + 2 * n * dk * dv * state_bytes


def chunk_time(t, chunk, dk, dv, peak_flops, bandwidth):
    """Roofline time in seconds: the larger of the FLOP time and the byte time."""
    return max(chunk_flops(t, chunk, dk, dv) / peak_flops, chunk_bytes(t, chunk, dk, dv) / bandwidth)


def best_chunk(t, dk, dv, peak_flops, bandwidth, candidates=(16, 32, 64, 128, 256, 512, 1024)):
    """The candidate chunk size with the smallest modelled time. Ties go to the smaller."""
    return min(candidates, key=lambda c: (chunk_time(t, c, dk, dv, peak_flops, bandwidth), c))


def softmax_flops(t, dk, dv):
    """FLOPs of causal softmax attention, counting the lower triangle only."""
    return t * t * (dk + dv)


def crossover_length(chunk, dk, dv):
    """The sequence length above which the chunked FLOPs are below softmax attention's."""
    return 2 * chunk + 4 * dk * dv / (dk + dv)


# --------------------------------------------------------------------------
# 5. The Triton kernel (A100 only)
# --------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _chunk_fwd_kernel(
        q_ptr, k_ptr, v_ptr, lg_ptr, o_ptr,
        T,
        DK: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
    ):
        """One program per head. Walks the chunks in order, carrying S in registers."""
        head = tl.program_id(0).to(tl.int64)
        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, DV)
        causal = offs_c[:, None] >= offs_c[None, :]
        s = tl.zeros((DK, DV), dtype=tl.float32)
        for start in range(0, T, C):
            pos = start + offs_c
            live = pos < T
            q = tl.load(q_ptr + (head * T + pos[:, None]) * DK + offs_k[None, :], mask=live[:, None], other=0.0)
            k = tl.load(k_ptr + (head * T + pos[:, None]) * DK + offs_k[None, :], mask=live[:, None], other=0.0)
            v = tl.load(v_ptr + (head * T + pos[:, None]) * DV + offs_v[None, :], mask=live[:, None], other=0.0)
            lg = tl.load(lg_ptr + head * T + pos, mask=live, other=0.0)
            b = tl.cumsum(lg, axis=0)
            b_last = tl.sum(lg, axis=0)
            # Intra-chunk: the masked decay matrix, built from differences of b.
            diff = tl.where(causal, b[:, None] - b[None, :], -float("inf"))
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * tl.exp(diff)
            o = tl.dot(scores, v, input_precision="ieee")
            # Inter-chunk: the entering state, decayed to each position.
            o += tl.exp(b)[:, None] * tl.dot(q, s, input_precision="ieee")
            tl.store(o_ptr + (head * T + pos[:, None]) * DV + offs_v[None, :], o, mask=live[:, None])
            # State handoff: decay S across the chunk and add the chunk's update.
            kd = k * tl.exp(b_last - b)[:, None]
            s = s * tl.exp(b_last) + tl.dot(tl.trans(kd), v, input_precision="ieee")


def chunked_forward_triton(q, k, v, logg, chunk):
    """q, k: (H, T, DK); v: (H, T, DV); logg: (H, T); all float32 on the GPU.

    DK, DV, and `chunk` are powers of two of at least 16. Returns o: (H, T, DV).
    """
    h, t, dk = q.shape
    dv = v.shape[2]
    q, k, v, logg = (x.contiguous() for x in (q, k, v, logg))
    o = torch.empty(h, t, dv, device=q.device, dtype=torch.float32)
    _chunk_fwd_kernel[(h,)](q, k, v, logg, o, t, DK=dk, DV=dv, C=chunk)
    return o
