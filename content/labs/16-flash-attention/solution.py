"""FlashAttention-2 forward and backward in Triton: the worked solution.

Layout: q is (B, H, N, D); k and v are (B, H_KV, N, D) with H a multiple of
H_KV (grouped-query attention); D is a power of two, 16 to 128. Self-attention
only: the query and key lengths are both N.

Four kernels:

* `_attn_fwd`: one program per (query tile, batch and head). Keeps the running
  maximum m, running sum l, and output accumulator in registers, and writes O
  and the logsumexp LSE.
* `_attn_bwd_preprocess`: delta = rowsum(dO * O), one value per query row.
* `_attn_bwd_dkdv`: one program per (key tile, batch and KV head). Loops over
  query tiles and over the query heads that share this KV head, and writes dK
  and dV once, with no atomics.
* `_attn_bwd_dq`: one program per (query tile, batch and head). Loops over key
  tiles and writes dQ once.

The two backward kernels both recompute P from Q, K, and LSE. That costs two
extra matmuls compared with the single-kernel form that accumulates dQ with
atomic adds, and buys a deterministic result.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------


@triton.jit
def _attn_fwd(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    H, N, GROUP, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    hk = h // GROUP

    q_base = Q + b * stride_qb + h * stride_qh
    o_base = O + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + hk * stride_kh
    v_base = V + b * stride_kb + hk * stride_kh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=m_mask[:, None], other=0.0)

    # Work in base 2: exp(x) = exp2(x * log2(e)). qk_scale folds 1/sqrt(d) and log2(e).
    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal block skipping: this query tile never looks past its last row.
    hi = N
    if CAUSAL:
        hi = tl.minimum(N, (pid_m + 1) * BLOCK_M)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None], mask=n_mask[None, :], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, k) * qk_scale
        keep = n_mask[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    # LSE in natural-log units, so it matches torch.logsumexp of the scaled scores.
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    tl.store(o_base + offs_m[:, None] * stride_qm + offs_d[None, :], acc.to(O.dtype.element_ty), mask=m_mask[:, None])
    tl.store(LSE + pid_bh.to(tl.int64) * N + offs_m, lse, mask=m_mask)


# --------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------


@triton.jit
def _attn_bwd_preprocess(
    O, DO, DELTA,
    stride_qb, stride_qh, stride_qm,
    H, N,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N
    ptrs = b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    o = tl.load(O + ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(DELTA + pid_bh.to(tl.int64) * N + offs_m, delta, mask=m_mask)


@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DO, DK, DV, LSE, DELTA,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    H_KV, N, GROUP, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bhk = tl.program_id(1)
    b = (pid_bhk // H_KV).to(tl.int64)
    hk = (pid_bhk % H_KV).to(tl.int64)
    H = H_KV * GROUP

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n < N

    kv_off = b * stride_kb + hk * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    k = tl.load(K + kv_off, mask=n_mask[:, None], other=0.0)
    v = tl.load(V + kv_off, mask=n_mask[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Causal: query rows before this key tile's first key see none of it.
    lo = 0
    if CAUSAL:
        lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M

    for g in range(GROUP):
        h = hk * GROUP + g
        q_base = Q + b * stride_qb + h * stride_qh
        do_base = DO + b * stride_qb + h * stride_qh
        row_base = (b * H + h) * N
        for start_m in range(lo, N, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < N
            q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            do = tl.load(do_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=m_mask[:, None], other=0.0)
            lse = tl.load(LSE + row_base + offs_m, mask=m_mask, other=0.0)
            delta = tl.load(DELTA + row_base + offs_m, mask=m_mask, other=0.0)

            # Work with the transposed tiles S^T, P^T, dP^T, dS^T of shape (BLOCK_N, BLOCK_M).
            st = tl.dot(k, tl.trans(q)) * qk_scale
            pt = tl.exp2(st - lse[None, :] * 1.4426950408889634)
            keep = m_mask[None, :] & n_mask[:, None]
            if CAUSAL:
                keep = keep & (offs_m[None, :] >= offs_n[:, None])
            pt = tl.where(keep, pt, 0.0)

            dv += tl.dot(pt.to(do.dtype), do)
            dpt = tl.dot(v, tl.trans(do))
            dst = pt * (dpt - delta[None, :])
            dk += tl.dot(dst.to(q.dtype), q)

    dk = dk * sm_scale
    tl.store(DK + kv_off, dk.to(DK.dtype.element_ty), mask=n_mask[:, None])
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=n_mask[:, None])


@triton.jit
def _attn_bwd_dq(
    Q, K, V, DO, DQ, LSE, DELTA,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    H, N, GROUP, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    hk = h // GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N
    q_off = b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_base = K + b * stride_kb + hk * stride_kh
    v_base = V + b * stride_kb + hk * stride_kh

    q = tl.load(Q + q_off, mask=m_mask[:, None], other=0.0)
    do = tl.load(DO + q_off, mask=m_mask[:, None], other=0.0)
    row = pid_bh.to(tl.int64) * N
    lse = tl.load(LSE + row + offs_m, mask=m_mask, other=0.0)
    delta = tl.load(DELTA + row + offs_m, mask=m_mask, other=0.0)

    qk_scale = sm_scale * 1.4426950408889634
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    hi = N
    if CAUSAL:
        hi = tl.minimum(N, (pid_m + 1) * BLOCK_M)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        kt = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None], mask=n_mask[None, :], other=0.0)
        vt = tl.load(v_base + offs_n[None, :] * stride_kn + offs_d[:, None], mask=n_mask[None, :], other=0.0)

        s = tl.dot(q, kt) * qk_scale
        p = tl.exp2(s - lse[:, None] * 1.4426950408889634)
        keep = n_mask[None, :] & m_mask[:, None]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(keep, p, 0.0)

        dp = tl.dot(do, vt)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(kt.dtype), tl.trans(kt))

    dq = dq * sm_scale
    tl.store(DQ + q_off, dq.to(DQ.dtype.element_ty), mask=m_mask[:, None])


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------


def _fwd_configs(d):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages), best first. Later entries use less memory."""
    if d <= 64:
        return [(128, 64, 4, 3), (64, 64, 4, 2), (32, 32, 4, 2)]
    return [(128, 64, 8, 2), (64, 64, 8, 2), (32, 32, 4, 2)]


def _bwd_configs(d):
    warps = 4 if d <= 64 else 8
    return [(64, 64, warps, 2), (32, 32, 4, 2)]


def _try(configs, launch):
    """Run `launch(*config)` with each config until one fits in shared memory."""
    last = None
    for config in configs:
        try:
            return launch(*config)
        except OutOfResources as err:
            last = err
    raise last


def _prepare(q, k, v):
    assert q.dim() == 4 and k.dim() == 4 and k.shape == v.shape, "expect q (B,H,N,D), k and v (B,H_KV,N,D)"
    B, H, N, D = q.shape
    assert k.shape[0] == B and k.shape[2] == N and k.shape[3] == D, "self-attention: q and k share B, N, D"
    assert H % k.shape[1] == 0, "H must be a multiple of H_KV"
    assert D in (16, 32, 64, 128), "head dimension must be a power of two from 16 to 128"
    assert q.dtype in (torch.bfloat16, torch.float16) and q.dtype == k.dtype == v.dtype
    return q.contiguous(), k.contiguous(), v.contiguous()


def flash_forward(q, k, v, causal=False, sm_scale=None):
    """Return (O, LSE). LSE is float32 with shape (B, H, N): logsumexp of the scaled scores."""
    q, k, v = _prepare(q, k, v)
    B, H, N, D = q.shape
    H_KV = k.shape[1]
    sm_scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    o = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    def launch(bm, bn, warps, stages):
        grid = (triton.cdiv(N, bm), B * H)
        _attn_fwd[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            H, N, H // H_KV, sm_scale,
            BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, CAUSAL=causal,
            num_warps=warps, num_stages=stages,
        )

    _try(_fwd_configs(D), launch)
    return o, lse


def flash_backward(q, k, v, o, lse, do, causal=False, sm_scale=None):
    """Return (dQ, dK, dV) given the forward's O and LSE and the upstream gradient dO."""
    q, k, v = _prepare(q, k, v)
    o, do = o.contiguous(), do.contiguous()
    B, H, N, D = q.shape
    H_KV = k.shape[1]
    sm_scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    delta = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

    pre_block = 64
    _attn_bwd_preprocess[(triton.cdiv(N, pre_block), B * H)](
        o, do, delta, q.stride(0), q.stride(1), q.stride(2), H, N,
        BLOCK_M=pre_block, HEAD_DIM=D,
    )

    def launch_dkdv(bm, bn, warps, stages):
        _attn_bwd_dkdv[(triton.cdiv(N, bn), B * H_KV)](
            q, k, v, do, dk, dv, lse, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            H_KV, N, H // H_KV, sm_scale,
            BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, CAUSAL=causal,
            num_warps=warps, num_stages=stages,
        )

    def launch_dq(bm, bn, warps, stages):
        _attn_bwd_dq[(triton.cdiv(N, bm), B * H)](
            q, k, v, do, dq, lse, delta,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            H, N, H // H_KV, sm_scale,
            BLOCK_M=bm, BLOCK_N=bn, HEAD_DIM=D, CAUSAL=causal,
            num_warps=warps, num_stages=stages,
        )

    _try(_bwd_configs(D), launch_dkdv)
    _try(_bwd_configs(D), launch_dq)
    return dq, dk, dv


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        o, lse = flash_forward(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_backward(q, k, v, o, lse, do, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None


def flash_attention(q, k, v, causal=False, sm_scale=None):
    """Differentiable attention: `flash_forward` in the forward, `flash_backward` in the backward."""
    return FlashAttention.apply(q, k, v, causal, sm_scale)
