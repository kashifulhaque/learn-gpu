"""Lab 38: Tile languages.

Every tile language answers the same four questions: what is the tile, which
thread holds which element (the layout), how does the tile arrive (the copy),
and how do the copy and the math overlap (the pipeline). You model each one
here, in plain Python, with no GPU.

`fixture_tiles.py`, next to this file, is a tiny tile API: `arange`,
`load_tile`, `store_tile`, `mma`, and the `BlockSpec` pair. Do not edit it.
`load_tile` and `store_tile` raise `OutOfBounds` when an index that the mask
does not turn off lies outside the array. That is the bug a real kernel hides.

Work in this order:

1. `grid_shape`: how many blocks cover an array.
2. `block_slices` and `gemm_block_indices`: a Pallas index map returns block
   indices, not element offsets.
3. `pallas_call`: a small emulator of `pl.pallas_call`.
4. `tile_gemm`: a masked tile loop, one program per output tile.
5. `online_softmax`: a running (max, sum), then one normalizing write.
6. `blocked_owner`: which (warp, lane, register) holds an element.
7. `pipeline_schedule` and `max_resident`: S-stage software pipelining.
8. `softmax_triton`: the same softmax on the GPU. Optional. It is skipped
   when there is no CUDA device or no Triton.
"""

from __future__ import annotations

import itertools  # noqa: F401
import math  # noqa: F401

from fixture_tiles import BlockSpec, arange, load_tile, mma, store_tile  # noqa: F401

try:  # Triton is installed on the A100 worker image, not on a laptop.
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def grid_shape(array_shape, block_shape):
    """Blocks per dimension: ceil(array / block). Return a tuple."""
    raise NotImplementedError("grid_shape")


def block_slices(block_index, block_shape, array_shape):
    """The tuple of slices that one *block index* names, clipped at the array edge.

    Block 2 of width 8 starts at element 16. The last block may be shorter.
    """
    raise NotImplementedError("block_slices")


def gemm_block_indices(i, j, k):
    """Block index of each GEMM operand at grid point (i, j, k).

    Return {"a": ..., "b": ..., "c": ...}. A is (M, K), B is (K, N), C is (M, N).
    """
    raise NotImplementedError("gemm_block_indices")


def pallas_call(kernel, grid, in_specs, out_specs, out_shape):
    """Return a function of the input arrays that runs `kernel` over `grid`.

    At each grid point, `kernel(*in_blocks, out_block)` runs. An input block that
    hangs over the array edge is padded with zeros to the full block shape. The
    kernel writes `out_block` in place. Copy only the in-bounds part of it into a
    float32 output of shape `out_shape`. Use `block_slices` and each spec's
    `index_map` and `block_shape`.
    """
    raise NotImplementedError("pallas_call")


def tile_gemm(a, b, bm=16, bn=16, bk=16):
    """C = A @ B with one (bm, bn) output tile per program and a loop over K tiles.

    Use `arange`, `load_tile`, `mma`, and `store_tile`. Mask every load and the
    store, so a tile that hangs over the edge touches nothing outside the array.
    Return a float32 (M, N) tensor.
    """
    raise NotImplementedError("tile_gemm")


def online_softmax(x, block_n=16):
    """Row softmax with one program per row.

    Pass 1 walks the row in tiles of `block_n` and keeps a running maximum m and
    a running sum l. Pass 2 walks it again and stores exp(x - m) / l. Load the
    masked tail with other=-inf. Return a tensor shaped like x.
    """
    raise NotImplementedError("online_softmax")


def blocked_owner(row, col, shape, size_per_thread, threads_per_warp, warps_per_cta, order):
    """(warp, lane, register) that holds element (row, col) of a 2-D tile.

    The arguments are a Gluon `BlockedLayout`. Each is a pair, one entry per
    dimension (0 is rows, 1 is columns). `order` lists the dimensions from the
    fastest-varying to the slowest. This lab's convention:

    - Along each dimension, the coordinate splits into digits, from the lowest:
      the position inside a thread's run (size_per_thread), the thread coordinate
      (threads_per_warp), the warp coordinate (warps_per_cta), and the repeat.
    - lane is the thread coordinates linearized with order[0] fastest. warp is the
      warp coordinates linearized the same way.
    - register is the position inside the run, linearized with order[0] fastest,
      plus size_per_thread[0] * size_per_thread[1] times the repeat, linearized
      the same way over the repeat counts shape[d] // (spt * tpw * wpc).

    The layout covers the shape exactly: each coverage divides the shape.
    """
    raise NotImplementedError("blocked_owner")


def pipeline_schedule(num_tiles, stages):
    """The schedule of an S-stage software pipeline.

    Return a list of (load, mma) pairs, one per step. Each is a tile index or None.
    Step s issues the load of tile s, and runs the MMA of tile s - (stages - 1).
    Raise ValueError when stages is below 2.
    """
    raise NotImplementedError("pipeline_schedule")


def max_resident(schedule):
    """The most tiles in shared memory at one step.

    A tile is resident from the step that issues its load through the step that
    runs its MMA, inclusive.
    """
    raise NotImplementedError("max_resident")


def softmax_triton(x):
    """Row softmax of a CUDA float32 matrix with one Triton program per row.

    Optional, and only graded on a GPU. Define a `@triton.jit` kernel at module
    level (inside `if triton is not None:`). Load with a mask and other=-inf,
    subtract tl.max, exponentiate, divide by tl.sum, and store with the mask.
    Use BLOCK = triton.next_power_of_2(n_cols).
    """
    raise NotImplementedError("softmax_triton")
