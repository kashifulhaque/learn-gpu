"""Lab 22: Streams and overlap. Reference solution."""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Provided: the kernel and the stream cache
# ---------------------------------------------------------------------------

@triton.jit
def _fma_chain_kernel(x_ptr, y_ptr, n, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    for _ in range(iters):
        x = x * 0.999 + 0.001
    tl.store(y_ptr + offs, x, mask=mask)


def launch_kernel(x: torch.Tensor, y: torch.Tensor, iters: int) -> None:
    """Queue y = f(x) on the current stream, where f applies x -> 0.999 x + 0.001 `iters` times."""
    n = x.numel()
    if n == 0:
        return
    _fma_chain_kernel[(triton.cdiv(n, 1024),)](x, y, n, iters, BLOCK=1024)


_STREAMS: list = []


def get_streams(k: int) -> list:
    """Return k cached side streams. Creating a stream on every call would cost time."""
    while len(_STREAMS) < k:
        _STREAMS.append(torch.cuda.Stream())
    return _STREAMS[:k]


# ---------------------------------------------------------------------------
# The solution
# ---------------------------------------------------------------------------


def make_pinned(n: int) -> torch.Tensor:
    return torch.empty(n, dtype=torch.float32, pin_memory=True)


def chunk_bounds(n: int, n_chunks: int) -> list:
    if n <= 0 or n_chunks <= 0:
        return []
    size = -(-n // n_chunks)  # ceiling division
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def pipeline_streams(h_in, h_out, d_in, d_out, iters: int, n_chunks: int, n_streams: int) -> None:
    streams = get_streams(n_streams)
    cur = torch.cuda.current_stream()
    for s in streams:
        s.wait_stream(cur)  # d_in and d_out were produced on the current stream
    for i, (a, b) in enumerate(chunk_bounds(h_in.numel(), n_chunks)):
        s = streams[i % n_streams]
        with torch.cuda.stream(s):
            d_in[a:b].copy_(h_in[a:b], non_blocking=True)
            launch_kernel(d_in[a:b], d_out[a:b], iters)
            h_out[a:b].copy_(d_out[a:b], non_blocking=True)
    for s in streams:
        cur.wait_stream(s)  # GPU-side join: the host does not block


def pipeline_staged(h_in, h_out, d_in, d_out, iters: int, n_chunks: int) -> None:
    s_in, s_kernel, s_out = get_streams(3)
    cur = torch.cuda.current_stream()
    for s in (s_in, s_kernel, s_out):
        s.wait_stream(cur)
    for a, b in chunk_bounds(h_in.numel(), n_chunks):
        copied = torch.cuda.Event()
        computed = torch.cuda.Event()
        with torch.cuda.stream(s_in):
            d_in[a:b].copy_(h_in[a:b], non_blocking=True)
            copied.record(s_in)
        s_kernel.wait_event(copied)
        with torch.cuda.stream(s_kernel):
            launch_kernel(d_in[a:b], d_out[a:b], iters)
            computed.record(s_kernel)
        s_out.wait_event(computed)
        with torch.cuda.stream(s_out):
            h_out[a:b].copy_(d_out[a:b], non_blocking=True)
    for s in (s_in, s_kernel, s_out):
        cur.wait_stream(s)


def pipeline_time_model(h2d_ms: float, kernel_ms: float, d2h_ms: float, n_chunks: int) -> float:
    stages = (h2d_ms / n_chunks, kernel_ms / n_chunks, d2h_ms / n_chunks)
    return sum(stages) + (n_chunks - 1) * max(stages)


def overlap_bound_ms(h2d_ms: float, kernel_ms: float, d2h_ms: float) -> float:
    return max(h2d_ms, kernel_ms, d2h_ms)


def overlap_efficiency(bound_ms: float, measured_ms: float) -> float:
    return bound_ms / measured_ms
