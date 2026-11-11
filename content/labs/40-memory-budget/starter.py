"""Lab 40: a training-memory budget, a caching-allocator simulator, a snapshot
parser, and a checkpoint chooser.

Everything except the last function runs on a CPU. `fixture_snapshot.py`, next
to this file, holds a synthesized snapshot and an allocation trace. It has the
shape of `torch.cuda.memory._snapshot()` output, but nobody captured it. Do
not edit it.

Work in this order:

1. state_bytes: bytes for weights, gradients, and Adam states.
2. activation_layer_bytes and mlp_stack_saved_bytes: bytes one layer saves.
3. peak_activation_bytes: the peak with some layers checkpointed.
4. round_size and segment_size: how the allocator sizes a block and a segment.
5. CachingAllocator and run_trace: allocated against reserved.
6. reserved_and_allocated, largest_free_block, fragmentation_ratio, classify_oom.
7. peak_allocation: replay the trace to the peak.
8. choose_checkpoints: the cheapest layers to checkpoint within a budget.
9. measure_peaks: measure max_memory_allocated on a GPU.
"""

from __future__ import annotations

MiB = 1 << 20

# PyTorch's native caching allocator, c10/cuda/CUDACachingAllocator.cpp.
MIN_BLOCK = 512  # every size rounds up to a multiple of this
SMALL_SIZE = 1 * MiB  # a request up to this size uses the small pool
SMALL_BUFFER = 2 * MiB  # the segment for a small request
LARGE_BUFFER = 20 * MiB  # the segment for a request of 1 MiB to 10 MiB
MIN_LARGE_ALLOC = 10 * MiB  # from here up, the segment is the request rounded up
ROUND_LARGE = 2 * MiB  # ... to a multiple of this


# ---------------------------------------------------------------------------
# Part 1: the byte budget
# ---------------------------------------------------------------------------


def state_bytes(n_params: int, mode: str) -> dict:
    """Bytes held for `n_params` parameters while training with Adam.

    `mode` is "fp32" (fp32 weights and gradients) or "mixed" (bf16 weights and
    gradients plus an fp32 master copy). Return a dict with the keys
    "weights", "grads", "master", "adam_m", "adam_v", and "total". A mode with
    no master copy has "master" equal to 0.
    """
    raise NotImplementedError


def activation_layer_bytes(s: int, b: int, h: int, a: int, flash: bool = False) -> int:
    """Bytes one GPT-style layer saves for backward: 34*s*b*h + 5*a*s^2*b.

    With `flash` true, the score-matrix term is gone. Chapter 19 derives it.
    """
    raise NotImplementedError


def peak_activation_bytes(full, boundary, ckpt) -> int:
    """Peak activation bytes with the layers in `ckpt` checkpointed.

    `full[i]` is what layer i saves when it isn't checkpointed, and
    `boundary[i]` is the input it keeps when it is. A checkpointed layer is
    rebuilt during the backward pass, which holds `full[i]` again at that
    moment, on top of every layer from 0 to i still stored. Layers above i
    have already released theirs. The peak is the larger of the stored total
    after the forward pass and the biggest rebuild.
    """
    raise NotImplementedError


def mlp_stack_saved_bytes(layers: int, tokens: int, hidden: int, elem_bytes: int) -> int:
    """Bytes autograd saves for `layers` of Linear(h, h, bias=False) plus GELU.

    The input is `tokens` by `hidden`. Weights don't count: they exist anyway.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the caching allocator
# ---------------------------------------------------------------------------


class OutOfMemory(Exception):
    """Raised when a new segment doesn't fit in the device's capacity."""


def round_size(nbytes: int) -> int:
    """Round up to a multiple of MIN_BLOCK, and never below MIN_BLOCK."""
    raise NotImplementedError


def segment_size(rounded: int) -> int:
    """Bytes of the `cudaMalloc` that serves a request of `rounded` bytes.

    SMALL_BUFFER up to SMALL_SIZE, LARGE_BUFFER below MIN_LARGE_ALLOC, and the
    size rounded up to a multiple of ROUND_LARGE otherwise.
    """
    raise NotImplementedError


class CachingAllocator:
    """A model of PyTorch's native caching allocator for one stream.

    - Two pools: small (rounded size at most SMALL_SIZE) and large.
    - `malloc` takes the smallest free block that fits, lowest address on a
      tie. With none, it creates a segment of `segment_size`. If that would
      pass `capacity`, it first releases every segment that is one free
      block, and raises OutOfMemory if it still doesn't fit.
    - A block splits when the remainder is at least MIN_BLOCK (small pool) or
      more than SMALL_SIZE (large pool).
    - `free` marks the block free and merges it with free neighbors in the
      same segment. A segment stays reserved.
    - `allocated` is the sum of live block sizes. `reserved` is the sum of
      segment sizes.
    """

    def __init__(self, capacity=None):
        self.capacity = capacity
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        # Add whatever else you need: segments, a handle table, an address counter.

    def malloc(self, nbytes: int) -> int:
        """Return an integer handle for the new block."""
        raise NotImplementedError

    def free(self, handle: int) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict:
        """The state in the shape of `torch.cuda.memory._snapshot()`.

        {"segments": [...], "device_traces": [[]]}. A segment has "address",
        "total_size", "segment_type" ("small" or "large"), "allocated_size",
        "active_size", and "blocks". A block has "address", "size",
        "requested_size", "state" ("active_allocated" or "inactive"), and
        "frames" (an empty list here).
        """
        raise NotImplementedError


def run_trace(events, capacity=None) -> dict:
    """Run ("alloc", name, nbytes) and ("free", name) events.

    Return {"allocated": [...], "reserved": [...], "peak_allocated": int,
    "peak_reserved": int, "oom_at": index or None, "allocator": the object}.
    The two lists hold the value after each event that ran. On OutOfMemory,
    stop, and put the index of the failing event in "oom_at".
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3: read a snapshot
# ---------------------------------------------------------------------------


def reserved_and_allocated(snapshot) -> tuple:
    """(reserved, allocated) in bytes. Allocated counts active blocks, of both kinds."""
    raise NotImplementedError


def largest_free_block(snapshot) -> int:
    """Size in bytes of the largest "inactive" block, or 0."""
    raise NotImplementedError


def fragmentation_ratio(snapshot) -> float:
    """1 - largest free block / total free bytes, or 0.0 when nothing is free."""
    raise NotImplementedError


def peak_allocation(snapshot) -> dict:
    """Replay `snapshot["device_traces"][0]` to the highest running total.

    Add `size` on "alloc" and subtract it on "free_requested". Ignore every
    other action. Return {"peak_bytes", "event", "addr", "size", "frame"}: the
    first event that reaches the maximum, its index in the list, its address
    and size, and its innermost frame as "file:line name". Use None for the
    frame when the entry has none.
    """
    raise NotImplementedError


def classify_oom(snapshot):
    """Why the last "oom" entry failed, or None when there is none.

    "capacity" when the free bytes in cached blocks are fewer than the request,
    "fragmentation" when they would cover it but no single block does, and
    "other" otherwise.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4: choose which layers to checkpoint
# ---------------------------------------------------------------------------


def choose_checkpoints(full, boundary, cost, budget):
    """The cheapest set of layers to checkpoint so the peak fits `budget`.

    `cost[i]` is the price of rebuilding layer i. Return (layers, total_cost),
    where `layers` is a tuple of indices, or None when no choice fits. The peak
    is the one `peak_activation_bytes` computes.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5: measure on the GPU
# ---------------------------------------------------------------------------


def measure_peaks(layers: int, tokens: int, hidden: int) -> tuple:
    """Return (plain_bytes, checkpointed_bytes) from torch.cuda.max_memory_allocated.

    Build `layers` of Sequential(Linear(hidden, hidden, bias=False), GELU) in
    fp32 on the GPU and an input of `tokens` by `hidden`. Run forward and
    backward of `y.pow(2).mean()`, once plainly and once with
    `torch.utils.checkpoint.checkpoint(layer, y, use_reentrant=False)` around
    every layer. Each number is the peak above `torch.cuda.memory_allocated()`
    taken just before the forward pass, after `reset_peak_memory_stats()`.
    Clear gradients with `zero_grad(set_to_none=True)` and warm each mode up
    once first: the first backward allocates a cuBLAS workspace.
    """
    raise NotImplementedError
