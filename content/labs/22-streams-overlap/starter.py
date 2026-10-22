"""Lab 22: Streams and overlap.

A job has three stages: copy a large host tensor to the GPU, run a kernel on
it, and copy the result back. Run as one piece, the stages take the sum of
their times. Cut into chunks and spread over CUDA streams, the three engines
(a copy engine in each direction, and the SMs) work on different chunks at
once, and the time approaches the longest stage.

You write the pieces of that pipeline. The tests check the output against a
float64 reference, check that your code uses pinned memory and orders its
streams correctly, and report how close the overlap gets to the
max(copy, compute) bound.

The kernel and a stream cache are provided. Everything under "Your code" is
yours.
"""

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
    """Queue y = f(x) on the CURRENT stream, where f applies x -> 0.999 x + 0.001 `iters` times.

    The tensors may be slices of a larger tensor. The call returns as soon as the
    kernel is queued.
    """
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
# Your code
# ---------------------------------------------------------------------------


def make_pinned(n: int) -> torch.Tensor:
    """Return an uninitialized pinned (page-locked) float32 CPU tensor of n elements.

    A DMA engine can read pinned pages directly, so a copy from this tensor can
    be asynchronous and overlap with a kernel. A copy from an ordinary
    (pageable) tensor goes through a staging buffer, blocks the host, and can't.
    Hint: torch.empty(..., pin_memory=True).
    """
    # TODO: allocate and return the pinned tensor.
    raise NotImplementedError


def chunk_bounds(n: int, n_chunks: int) -> list:
    """Split range(n) into at most n_chunks contiguous (start, stop) pairs.

    Rules the tests check:
    - the pairs start at 0, end at n, and touch with no gap or overlap;
    - no pair is empty, so n = 1 gives [(0, 1)] and n = 9 with 8 chunks gives
      fewer than 8 pairs;
    - at most n_chunks pairs.
    Hint: a chunk size of ceil(n / n_chunks), and range(0, n, size).
    """
    # TODO: return the list of (start, stop) pairs.
    raise NotImplementedError


def pipeline_streams(h_in, h_out, d_in, d_out, iters: int, n_chunks: int, n_streams: int) -> None:
    """Stream-per-chunk pipeline.

    h_in and h_out are pinned CPU tensors, d_in and d_out are CUDA tensors, all
    float32 with the same number of elements. d_in and d_out were allocated and
    filled on the current stream. Chunk i goes in stream (i % n_streams) and
    does, in that stream, in order:
        d_in[a:b]  <- h_in[a:b]          (copy_ with non_blocking=True)
        launch_kernel(d_in[a:b], d_out[a:b], iters)
        h_out[a:b] <- d_out[a:b]         (copy_ with non_blocking=True)

    Rules the tests check:
    1. Before queueing anything on a side stream, make it wait for the current
       stream: s.wait_stream(torch.cuda.current_stream()).
    2. Queue each chunk with `with torch.cuda.stream(s):`.
    3. When everything is queued, make the CURRENT stream wait for every side
       stream: cur.wait_stream(s). This is a GPU-side wait, so the tests' end
       event lands after all the work.
    4. Do not synchronize the host: no .item(), .cpu(), print of a tensor, or
       torch.cuda.synchronize(). The function must return long before the GPU
       is done.
    Use get_streams(n_streams) for the streams, and chunk_bounds for the chunks.
    """
    # TODO: fork, queue every chunk on its stream, join.
    raise NotImplementedError


def pipeline_staged(h_in, h_out, d_in, d_out, iters: int, n_chunks: int) -> None:
    """Stream-per-stage pipeline, with events carrying each chunk's dependencies.

    Use three streams from get_streams(3): one for copies in, one for kernels,
    one for copies out. For each chunk:
        1. In the copy-in stream, copy h_in[a:b] to d_in[a:b], then record an
           Event (torch.cuda.Event(), no timing needed) on that stream.
        2. Make the kernel stream wait for that event (stream.wait_event(ev)),
           then queue the kernel and record a second event.
        3. Make the copy-out stream wait for the second event, then queue the
           copy of d_out[a:b] to h_out[a:b].
    Fork from the current stream first and join back into it at the end, as in
    pipeline_streams. A missing wait leaves NaNs in the output: the tests fill
    both outputs with NaN before the call.
    """
    # TODO: three streams, two events per chunk, fork and join.
    raise NotImplementedError


def pipeline_time_model(h2d_ms: float, kernel_ms: float, d2h_ms: float, n_chunks: int) -> float:
    """Time of the pipeline for n_chunks equal chunks, in ms.

    The stage totals are h2d_ms, kernel_ms, and d2h_ms. One chunk's stage times
    are the totals divided by n_chunks: s1, s2, s3. The first chunk crosses all
    three stages alone. Each later chunk then leaves the pipeline one longest
    stage after the one before:
        T = s1 + s2 + s3 + (n_chunks - 1) * max(s1, s2, s3)
    Check: three 8.52 ms stages with 8 chunks give 10.65 ms.
    """
    # TODO: the formula above.
    raise NotImplementedError


def overlap_bound_ms(h2d_ms: float, kernel_ms: float, d2h_ms: float) -> float:
    """The fastest any pipeline can run: the longest stage, in ms."""
    # TODO: one line.
    raise NotImplementedError


def overlap_efficiency(bound_ms: float, measured_ms: float) -> float:
    """bound / measured: 1.0 means the pipeline runs at the longest stage's time."""
    # TODO: one division.
    raise NotImplementedError
