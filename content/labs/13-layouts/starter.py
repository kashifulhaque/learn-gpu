"""Lab 13: CuTe layout algebra in Python.

A layout is a pair (shape, stride) of congruent nested tuples, and it is a
function from a coordinate to an integer offset. Implement the pieces below,
in this order; each builds on the one before it. The tests check your code
against the worked examples in the CuTe documentation (cute/02_layout_algebra.md),
so the arithmetic in the chapter is your specification.

1. `_crd2idx`, `Layout.size`, `Layout.cosize`, `Layout.__call__`
2. `coalesce`
3. `_compose_mode` and `composition`
4. `complement`
5. `logical_divide`, `logical_product`, `zipped_divide`
6. `swizzle` and `bank_conflict_degree`
7. `gather_with_layout`, which reads a tensor through a layout

Strides are nonnegative integers. The tuple helpers and `from_str` are given.
Run the tests on any machine: the core is pure Python, and the final gather
test uses the GPU when there is one.
"""

from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# Tuple helpers (given; not part of the exercise in the starter)
# ---------------------------------------------------------------------------


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


def product(x) -> int:
    """The product of every integer in a nested tuple, or `x` itself for an int."""
    if is_tuple(x):
        out = 1
        for e in x:
            out *= product(e)
        return out
    return x


def flatten(x) -> tuple:
    """Every integer of a nested tuple, left to right, as a flat tuple."""
    if is_tuple(x):
        out = ()
        for e in x:
            out += flatten(e)
        return out
    return (x,)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _same_profile(a, b) -> bool:
    """True when `a` and `b` have the same nesting (they are congruent)."""
    if is_tuple(a) != is_tuple(b):
        return False
    if is_tuple(a):
        return len(a) == len(b) and all(_same_profile(x, y) for x, y in zip(a, b))
    return True


def _fmt(x) -> str:
    if is_tuple(x):
        return "(" + ",".join(_fmt(e) for e in x) + ")"
    return str(x)


def _crd2idx(crd, shape, stride) -> int:
    """Map a coordinate to an offset: the sum over modes of coordinate * stride.

    - `shape` and `stride` are congruent nested tuples (or ints).
    - A tuple coordinate matches the nesting of `shape`: recurse mode by mode and add.
    - An int coordinate into a tuple shape is read colexicographically (first
      mode fastest): peel off `crd % product(mode_shape)` for each mode but the
      last, divide `crd` by that product, and give the last mode what is left.
    - An int shape with an int coordinate is `crd * stride`.
    """
    raise NotImplementedError


def _pack(shapes: list, strides: list):
    """Build (shape, stride) from parallel lists: a single mode stays a bare int."""
    if not shapes:
        return 1, 0
    if len(shapes) == 1:
        return shapes[0], strides[0]
    return tuple(shapes), tuple(strides)


class Layout:
    """A CuTe layout: shape and stride, congruent nested tuples (or ints)."""

    def __init__(self, shape, stride=None):
        if stride is None:
            # Column-major (compact, first mode fastest) strides for the shape.
            stride = self._colmajor_stride(shape)
        if not _same_profile(shape, stride):
            raise ValueError(f"shape {shape} and stride {stride} are not congruent")
        self.shape = shape
        self.stride = stride

    @staticmethod
    def _colmajor_stride(shape, start: int = 1):
        if is_tuple(shape):
            out = []
            for s in shape:
                out.append(Layout._colmajor_stride(s, start))
                start *= product(s)
            return tuple(out)
        return start

    @classmethod
    def from_str(cls, text: str) -> "Layout":
        """Parse the CuTe notation, for example "(4,(2,3)):(2,(1,8))"."""
        shape_text, stride_text = text.split(":", 1)
        return cls(ast.literal_eval(shape_text), ast.literal_eval(stride_text))

    # -- basic queries -------------------------------------------------------

    def __repr__(self) -> str:
        return f"{_fmt(self.shape)}:{_fmt(self.stride)}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Layout) and self.shape == other.shape and self.stride == other.stride

    def __hash__(self) -> int:
        return hash((self.shape, self.stride))

    def __len__(self) -> int:
        """The rank: the number of top-level modes (1 for an int shape)."""
        return len(self.shape) if is_tuple(self.shape) else 1

    def __getitem__(self, i: int) -> "Layout":
        """The i-th top-level mode as its own layout."""
        if not is_tuple(self.shape):
            if i != 0:
                raise IndexError(i)
            return self
        return Layout(self.shape[i], self.stride[i])

    def modes(self) -> list["Layout"]:
        return [self[i] for i in range(len(self))]

    def size(self) -> int:
        """The number of coordinates: the product of the shape."""
        raise NotImplementedError

    def cosize(self) -> int:
        """One past the largest offset the layout produces, which is the length
        of the smallest buffer that holds every element. For nonnegative strides
        the largest offset is at the last coordinate."""
        raise NotImplementedError

    def __call__(self, crd) -> int:
        """Map a coordinate (an int, a tuple, or a nested tuple) to an offset."""
        raise NotImplementedError

    def offsets(self) -> list[int]:
        """Offsets for every integer coordinate 0 .. size-1, in colexicographic order."""
        return [self(i) for i in range(self.size())]

    # -- operations ----------------------------------------------------------

    def coalesce(self, profile=None) -> "Layout":
        return coalesce(self, profile)


def make_layout(*layouts: Layout) -> Layout:
    """Concatenate layouts as the modes of a new layout."""
    return Layout(tuple(l.shape for l in layouts), tuple(l.stride for l in layouts))


# ---------------------------------------------------------------------------
# Layout algebra
# ---------------------------------------------------------------------------


def coalesce(layout: Layout, profile=None) -> Layout:
    """Flatten and merge modes without changing the layout function.

    Rules for neighbours s0:d0 and s1:d1: drop a size-1 mode, and merge the two
    into (s0*s1):d0 when d1 == s0*d0. With a `profile` tuple (one entry per
    mode, None to coalesce the whole mode), each top-level mode is coalesced on
    its own so the rank is kept.
    """
    raise NotImplementedError


def _compose_mode(lhs_modes: list[tuple[int, int]], shape: int, stride: int) -> Layout:
    """Compose a flat lhs with the single-mode layout `shape:stride`."""
    raise NotImplementedError


def composition(lhs: Layout, rhs) -> Layout:
    """R = lhs o rhs, so that R(c) = lhs(rhs(c)) for every c in rhs's domain.

    `rhs` is a Layout, or a tiler: a tuple whose entries pair up with the modes
    of `lhs` (each entry a Layout, an int meaning n:1, or a nested tuple).
    """
    raise NotImplementedError


def complement(layout: Layout, cotarget: int | None = None) -> Layout:
    """The layout that steps over what `layout` skips, up to `cotarget` elements.

    The result has positive, increasing strides and a codomain disjoint from
    the input's. The input's modes, sorted by stride, must nest: each stride
    must be a multiple of the extent covered so far.
    """
    raise NotImplementedError


def logical_divide(layout: Layout, tiler) -> Layout:
    """A o (B, complement(B, size(A))): mode 0 is the tile, mode 1 indexes tiles.

    `tiler` is a Layout, or a tuple for a by-mode divide of a multi-mode layout.
    """
    raise NotImplementedError


def logical_product(layout: Layout, tiler: Layout) -> Layout:
    """(A, complement(A, size(A) * cosize(B)) o B): mode 0 is A, mode 1 repeats it."""
    raise NotImplementedError


def zipped_divide(layout: Layout, tiler) -> Layout:
    """logical_divide by mode, regrouped as ((tile modes), (rest modes))."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Swizzle and bank conflicts
# ---------------------------------------------------------------------------


def swizzle(offset: int, b: int, m: int, s: int) -> int:
    """Swizzle<B, M, S> applied to an integer offset.

    Take the B-bit field that starts at bit M + max(0, S), shift it by S toward
    the low end (right for S > 0, left for S < 0), and XOR it into the B-bit
    field that starts at bit M - min(0, S).
    """
    raise NotImplementedError


class Swizzle:
    """The functor Swizzle<B, M, S>, callable on an offset."""

    def __init__(self, b: int, m: int, s: int):
        if abs(s) < b:
            raise ValueError("|S| must be at least B so the two bit fields don't overlap")
        self.b, self.m, self.s = b, m, s

    def __call__(self, offset: int) -> int:
        return swizzle(offset, self.b, self.m, self.s)

    def __repr__(self) -> str:
        return f"Swizzle<{self.b},{self.m},{self.s}>"


def bank_conflict_degree(word_offsets, banks: int = 32) -> int:
    """The number of shared-memory transactions one warp access needs.

    `word_offsets` are 4-byte-word offsets, one per active thread. Bank = offset
    mod `banks`. Threads that read the same word broadcast and don't conflict, so
    the answer is the largest number of *distinct* words in any one bank.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Driving a gather with a layout on the GPU
# ---------------------------------------------------------------------------


def layout_index_tensor(layout: Layout, device=None):
    """The layout's offsets for coordinates 0 .. size-1, as an int64 tensor."""
    import torch

    return torch.tensor(layout.offsets(), dtype=torch.int64, device=device)


def gather_with_layout(x, layout: Layout):
    """Read the flat tensor `x` through `layout`: out[i] = x[layout(i)]."""
    raise NotImplementedError

