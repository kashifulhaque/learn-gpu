"""Lab 29: GEMM scheduling.

Do the CPU pieces first, in order. They need no GPU, and each check names the
function it grades. The Triton kernels at the end run only on a CUDA GPU.

1. wave_stats
2. split_k_ranges, split_k_matmul
3. reduce_partials
4. stream_k_plan, fixup_tiles
5. persistent_tile_order, simulate_tile_counter, static_makespan
6. apply_epilogue, epilogue_bytes
7. schedule_cost_us, choose_schedule
8. splitk_matmul, persistent_matmul, fused_matmul (GPU)
"""

from __future__ import annotations

import math

try:  # Triton is not installed on a CPU-only machine.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False

# Constants of the cost model. The first three are the A100 80GB PCIe figures in
# docs/FACTS.md. MIN_GAIN is a modelling choice, not a hardware number.
PEAK_TFLOPS = 312.0  # dense bf16 tensor cores
BW_GBS = 1275.0  # measured copy ceiling, learn-inference chapter 10
LAUNCH_US = 5.0  # host launch cost of the second (reduction) kernel
MIN_GAIN = 0.05  # a schedule must beat data-parallel by 5% to be chosen

EPILOGUES = ("bias", "bias_relu", "bias_gelu", "bias_silu", "bias_residual", "scale_residual")


# ---------------------------------------------------------------------------
# 1. Waves
# ---------------------------------------------------------------------------


def wave_stats(
    m: int, n: int, tile_m: int, tile_n: int, sm_count: int = 108, ctas_per_sm: int = 1
) -> tuple[int, int, float]:
    """Return (tiles, waves, efficiency) for one CTA per output tile.

    Each dimension rounds up to whole tiles on its own. Efficiency is
    tiles / (waves * ctas_per_sm * sm_count). For 2304 x 1544 on 256 x 128
    tiles, the answer is (117, 2, 0.5417).
    """
    raise NotImplementedError("count tiles, then waves, then efficiency")


# ---------------------------------------------------------------------------
# 2. Split-K
# ---------------------------------------------------------------------------


def split_k_ranges(k_iters: int, splits: int) -> list[tuple[int, int]]:
    """Cut `k_iters` K-iterations into `splits` half-open ranges.

    Sizes differ by at most one, and the first `k_iters % splits` ranges are the
    longer ones. When splits > k_iters the trailing ranges are empty.
    split_k_ranges(16, 5) is [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)].
    """
    raise NotImplementedError("deal the iterations like cards")


def split_k_matmul(a, b, splits: int, block_k: int):
    """A @ B as `splits` fp32 partial products, reduced in fp32 in split order.

    `a` is (M, K) and `b` is (K, N), on the CPU, in any floating dtype. K is cut
    in units of `block_k`, so the last K-iteration can be shorter than block_k.
    Return a float32 (M, N) tensor.
    """
    raise NotImplementedError("one partial per split, kept in float32")


def reduce_partials(partials, order) -> float:
    """Sum scalar partials in fp32, one at a time, in the given order.

    This is what the atomic variant does, with the order chosen by the hardware.
    For partials [1e8, 1, -1e8, 1], the order (0, 1, 2, 3) gives 1.0, the order
    (0, 2, 1, 3) gives 2.0, and the order (1, 3, 0, 2) gives 0.0.
    """
    raise NotImplementedError("accumulate in a float32 tensor, not a Python float")


# ---------------------------------------------------------------------------
# 3. Stream-K
# ---------------------------------------------------------------------------


def stream_k_plan(tiles: int, iters_per_tile: int, ctas: int) -> list[list[tuple[int, int, int]]]:
    """Deal the tiles * iters_per_tile MAC-iterations to `ctas` CTAs.

    Iteration i belongs to tile i // iters_per_tile. CTA c gets one contiguous
    range of the global iteration order. Sizes differ by at most one, and the
    first `total % ctas` CTAs take the extra one. Return, for each CTA, its
    segments as (tile, k_begin, k_end), where k is local to the tile and a
    segment never crosses a tile boundary.
    """
    raise NotImplementedError("a start offset, a length, then cut at tile boundaries")


def fixup_tiles(plan) -> dict[int, list[int]]:
    """Map each tile that two or more CTAs touch to its sorted list of CTAs."""
    raise NotImplementedError("group the segments by tile")


# ---------------------------------------------------------------------------
# 4. Persistent kernels
# ---------------------------------------------------------------------------


def persistent_tile_order(
    cta: int, ctas: int, tiles_m: int, tiles_n: int, group_m: int
) -> list[tuple[int, int]]:
    """The (pid_m, pid_n) tiles one persistent CTA computes, in order.

    CTA `cta` takes linear tile ids cta, cta + ctas, cta + 2 * ctas, ... below
    tiles_m * tiles_n. Each id maps to a tile with grouped ordering: vertical
    strips of `group_m` rows, the row index changing fastest inside a strip,
    and a shorter last strip when tiles_m is not a multiple of group_m.
    Take the position inside the strip first, tile_id % (group_m * tiles_n),
    and use that position for both the row and the column.
    """
    raise NotImplementedError("stride over tile ids, then map each id")


def simulate_tile_counter(costs, ctas: int) -> tuple[float, list[list[int]]]:
    """Dynamic scheduling with an atomic tile counter.

    All CTAs start at time 0. Whenever a CTA is free it takes the next tile id
    and works for costs[id]. When several CTAs are free at once, the lowest CTA
    index goes first. Return (makespan, tile ids per CTA, in the order taken).
    simulate_tile_counter([5, 1, 1, 1, 1, 1], 2) is (5.0, [[0], [1, 2, 3, 4, 5]]).
    """
    raise NotImplementedError("track when each CTA is next free")


def static_makespan(costs, ctas: int) -> float:
    """Static round-robin: tile i goes to CTA i % ctas. Return the longest CTA time."""
    raise NotImplementedError("sum the costs per CTA")


# ---------------------------------------------------------------------------
# 5. Epilogues
# ---------------------------------------------------------------------------


def apply_epilogue(acc, kind: str, bias=None, residual=None, alpha: float = 1.0, beta: float = 1.0):
    """Apply one epilogue to an fp32 accumulator tile. All inputs are fp32.

    bias:           acc + bias
    bias_relu:      relu(acc + bias)
    bias_gelu:      gelu_tanh(acc + bias), the tanh approximation
    bias_silu:      silu(acc + bias)
    bias_residual:  acc + bias + residual
    scale_residual: alpha * acc + beta * residual
    """
    raise NotImplementedError("one branch per kind")


def epilogue_bytes(m: int, n: int, kind: str, fused: bool, elem_bytes: int = 2) -> int:
    """Bytes of C-sized traffic that the output stage moves, in HBM.

    Fused: the GEMM writes C once, and reads the epilogue's operands, which are
    a bias row of n elements (every kind except scale_residual) and, for the two
    residual kinds, an m x n residual.
    Unfused: the GEMM writes C, then every stage is its own kernel that reads
    its m x n input, reads its operand, and writes m x n. Stages per kind:
    bias 1, and 2 for each of the other five (bias then activation, bias then
    residual, scale then residual).
    """
    raise NotImplementedError("count the C write, the stage reads and writes, and the operands")


# ---------------------------------------------------------------------------
# 6. The schedule chooser
# ---------------------------------------------------------------------------


def schedule_cost_us(
    m: int,
    n: int,
    k: int,
    schedule: str,
    splits: int = 1,
    tile: tuple[int, int, int] = (128, 128, 64),
    sm_count: int = 108,
) -> float:
    """Predicted microseconds for one bf16 GEMM, one CTA per SM.

    time = max(compute, memory) + extra, where
      compute = critical-path K-iterations * t_iter, and t_iter is one
                tile_m x tile_n x tile_k step at the per-SM peak, 312e12 / sm_count;
      memory  = compulsory bytes 2 * (mk + kn + mn) at 1275 GB/s, scaled by
                sm_count / active CTAs, because a grid that fills fewer SMs
                pulls proportionally less bandwidth;
      extra   = split-K: a 5 us launch plus writing and reading `splits` fp32
                m x n partials at 1275 GB/s (none for splits = 1);
                stream-K: writing and reading one fp32 tile for each tile that
                needs a fix-up.
    Schedules and their critical path:
      "data_parallel": ceil(tiles / sm_count) * iters_per_tile
      "split_k":       ceil(tiles * splits / sm_count) * ceil(iters_per_tile / splits)
      "stream_k":      ceil(total / g), with total = tiles * iters_per_tile and
                       g = min(sm_count, total) CTAs; use stream_k_plan and fixup_tiles.
    Anchors: (64, 4096, 4096, "data_parallel", tile (64, 128, 64)) is 91.60 us,
    and the same shape with "stream_k" is 28.78 us.
    """
    raise NotImplementedError("compute, memory, extra")


def choose_schedule(
    m: int,
    n: int,
    k: int,
    tile: tuple[int, int, int] = (128, 128, 64),
    sm_count: int = 108,
    allow_stream_k: bool = True,
) -> tuple[str, int]:
    """Pick the cheapest schedule, as (name, splits).

    Candidates: data_parallel; split_k with 2 <= splits <= min(16, k-iterations);
    and stream_k when allowed. A candidate other than data_parallel wins only if
    its cost is at most (1 - MIN_GAIN) times data_parallel's. Among those, the
    cheapest wins, and the earlier candidate in the list above wins a tie.
    """
    raise NotImplementedError("score every candidate with schedule_cost_us")


# ---------------------------------------------------------------------------
# 7. Triton kernels (GPU)
# ---------------------------------------------------------------------------
# Write these after the CPU checks pass. The helpers below are given. Each wrapper
# takes fp16 or bf16 CUDA tensors and returns a tensor of the same dtype.
#
# Split-K. Grid (tiles, splits). Program (t, s) runs the K-iterations that
# split_k_ranges gives split s and stores its fp32 partial into a workspace of
# shape (splits, M, N). A second kernel adds the splits in a fixed order and casts.
#
# Persistent. Grid (min(items, num_sms),), with items = tiles * splits. Program p
# loops `for work in range(p, items, NUM_SMS)`. Work item w is tile w // splits and
# split w % splits. Map the tile with the grouped arithmetic from
# persistent_tile_order. With splits = 1, store. With splits > 1, tl.atomic_add the
# fp32 partial into a zeroed fp32 buffer, then cast.
#
# Fused. The lab 14 kernel with an EPILOGUE constexpr, applied to the fp32
# accumulator before the cast. 0.5 * x * (1 + tanh(u)) equals x * sigmoid(2u),
# so GELU needs only tl.sigmoid.


def _block_m(m: int) -> int:
    """Smallest tile height of 16, 32, 64, or 128 that covers m."""
    for size in (16, 32, 64, 128):
        if m <= size:
            return size
    return 128


def _check_inputs(a, b):
    import torch

    assert a.is_cuda and b.is_cuda and a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert a.shape[1] == b.shape[0]
    return a.shape[0], b.shape[1], a.shape[1]


def splitk_matmul(a, b, splits: int):
    """C = A @ B with split-K: fp32 partials in a workspace, then a reduction kernel."""
    raise NotImplementedError("write the partial kernel and the reduction kernel")


def persistent_matmul(a, b, splits: int = 1):
    """C = A @ B with one program per SM looping over (tile, split) work items."""
    raise NotImplementedError("write the persistent kernel")


def fused_matmul(a, b, kind: str, bias=None, residual=None, alpha: float = 1.0, beta: float = 1.0):
    """C = epilogue(A @ B), with the epilogue applied to the fp32 accumulator."""
    raise NotImplementedError("write the fused kernel")
