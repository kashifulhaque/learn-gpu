"""Lab 13: CuTe layout algebra in Python (worked solution).

A layout is a pair (shape, stride) of congruent nested tuples. It is a function
from a coordinate to an integer offset. This module implements the CuTe
operations from the CUTLASS documentation (cute/02_layout_algebra.md) on
nonnegative strides: the layout function, coalesce, composition, complement,
logical divide and product, zipped divide, and the Swizzle functor.
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
    """Coordinate to offset. An int coordinate into a tuple shape is read
    colexicographically (first mode fastest)."""
    if is_tuple(shape):
        if is_tuple(crd):
            if len(crd) != len(shape):
                raise ValueError(f"coordinate {crd} does not match shape {shape}")
            return sum(_crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        total = 0
        last = len(shape) - 1
        for i, (s, d) in enumerate(zip(shape, stride)):
            if i == last:
                total += _crd2idx(crd, s, d)
            else:
                n = product(s)
                total += _crd2idx(crd % n, s, d)
                crd //= n
        return total
    if is_tuple(crd):
        raise ValueError(f"coordinate {crd} is too deep for shape {shape}")
    return crd * stride


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
        return product(self.shape)

    def cosize(self) -> int:
        """One past the largest offset: the size of the smallest buffer that holds the codomain."""
        return self(self.size() - 1) + 1 if self.size() > 0 else 0

    def __call__(self, crd) -> int:
        """Map a coordinate (an int, a tuple, or a nested tuple) to an offset."""
        return _crd2idx(crd, self.shape, self.stride)

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
    if profile is not None:
        if not is_tuple(profile):
            profile = (profile,) * len(layout)
        parts = []
        for mode, p in zip(layout.modes(), profile):
            parts.append(coalesce(mode, p) if is_tuple(p) else coalesce(mode))
        return make_layout(*parts)
    shapes: list[int] = []
    strides: list[int] = []
    for s, d in zip(flatten(layout.shape), flatten(layout.stride)):
        if s == 1:
            continue
        if shapes and strides[-1] * shapes[-1] == d:
            shapes[-1] *= s
        else:
            shapes.append(s)
            strides.append(d)
    return Layout(*_pack(shapes, strides))


def _compose_mode(lhs_modes: list[tuple[int, int]], shape: int, stride: int) -> Layout:
    """Compose a flat lhs with the single-mode layout `shape:stride`."""
    if stride == 0 or shape == 1:
        return Layout(shape, 0)
    out_shape: list[int] = []
    out_stride: list[int] = []
    rest_shape, rest_stride = shape, stride
    for i, (s, d) in enumerate(lhs_modes):
        if rest_shape == 1:
            break
        if i == len(lhs_modes) - 1:
            # The last mode of the lhs extends without bound.
            out_shape.append(rest_shape)
            out_stride.append(rest_stride * d)
            rest_shape = 1
            break
        if rest_stride >= s:
            if rest_stride % s != 0:
                raise ValueError(f"stride {stride} is not divisible by lhs mode of size {s}")
            rest_stride //= s
            continue
        if s % rest_stride != 0:
            raise ValueError(f"lhs mode of size {s} is not divisible by stride {rest_stride}")
        take = min(s // rest_stride, rest_shape)
        if rest_shape % take != 0:
            raise ValueError(f"shape {rest_shape} is not divisible by {take}")
        if take > 1:
            out_shape.append(take)
            out_stride.append(rest_stride * d)
        rest_shape //= take
        rest_stride = 1
    return Layout(*_pack(out_shape, out_stride))


def composition(lhs: Layout, rhs) -> Layout:
    """R = lhs o rhs, so that R(c) = lhs(rhs(c)) for every c in rhs's domain.

    `rhs` is a Layout, or a tiler: a tuple whose entries pair up with the modes
    of `lhs` (each entry a Layout, an int meaning n:1, or a nested tuple).
    """
    if is_tuple(rhs):
        parts = []
        for i, t in enumerate(rhs):
            parts.append(composition(lhs[i], t))
        return make_layout(*parts)
    if isinstance(rhs, int):
        rhs = Layout(rhs, 1)
    if is_tuple(rhs.shape):
        return make_layout(*[composition(lhs, m) for m in rhs.modes()])
    flat = list(zip(flatten(lhs.shape), flatten(lhs.stride)))
    return _compose_mode(flat, rhs.shape, rhs.stride)


def complement(layout: Layout, cotarget: int | None = None) -> Layout:
    """The layout that steps over what `layout` skips, up to `cotarget` elements.

    The result has positive, increasing strides and a codomain disjoint from
    the input's. The input's modes, sorted by stride, must nest: each stride
    must be a multiple of the extent covered so far.
    """
    if cotarget is None:
        cotarget = layout.cosize()
    modes = [(d, s) for s, d in zip(flatten(layout.shape), flatten(layout.stride)) if s != 1 and d != 0]
    modes.sort()
    out_shape: list[int] = []
    out_stride: list[int] = []
    current = 1
    for d, s in modes:
        if d % current != 0:
            raise ValueError(f"{layout} has no complement: stride {d} is not a multiple of {current}")
        gap = d // current
        if gap > 1:
            out_shape.append(gap)
            out_stride.append(current)
        current = s * d
    tail = ceil_div(cotarget, current)
    if tail > 1:
        out_shape.append(tail)
        out_stride.append(current)
    return Layout(*_pack(out_shape, out_stride))


def logical_divide(layout: Layout, tiler) -> Layout:
    """A o (B, complement(B, size(A))): mode 0 is the tile, mode 1 indexes tiles.

    `tiler` is a Layout, or a tuple for a by-mode divide of a multi-mode layout.
    """
    if is_tuple(tiler):
        return make_layout(*[logical_divide(layout[i], t) for i, t in enumerate(tiler)])
    if isinstance(tiler, int):
        tiler = Layout(tiler, 1)
    return composition(layout, make_layout(tiler, complement(tiler, layout.size())))


def logical_product(layout: Layout, tiler: Layout) -> Layout:
    """(A, complement(A, size(A) * cosize(B)) o B): mode 0 is A, mode 1 repeats it."""
    return make_layout(layout, composition(complement(layout, layout.size() * tiler.cosize()), tiler))


def zipped_divide(layout: Layout, tiler) -> Layout:
    """logical_divide by mode, regrouped as ((tile modes), (rest modes))."""
    divided = logical_divide(layout, tiler)
    if not is_tuple(tiler):
        return divided
    tiles = make_layout(*[m[0] for m in divided.modes()])
    rests = make_layout(*[m[1] for m in divided.modes()])
    return make_layout(tiles, rests)


# ---------------------------------------------------------------------------
# Swizzle and bank conflicts
# ---------------------------------------------------------------------------


def swizzle(offset: int, b: int, m: int, s: int) -> int:
    """Swizzle<B, M, S> applied to an integer offset.

    Take the B-bit field that starts at bit M + max(0, S), shift it by S toward
    the low end (right for S > 0, left for S < 0), and XOR it into the B-bit
    field that starts at bit M - min(0, S).
    """
    mask = (1 << b) - 1
    yyy = mask << (m + max(0, s))
    shifted = (offset & yyy) >> s if s >= 0 else (offset & yyy) << -s
    return offset ^ shifted


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
    per_bank: dict[int, set[int]] = {}
    for w in word_offsets:
        per_bank.setdefault(w % banks, set()).add(w)
    return max(len(words) for words in per_bank.values()) if per_bank else 0


# ---------------------------------------------------------------------------
# Driving a gather with a layout on the GPU
# ---------------------------------------------------------------------------


def layout_index_tensor(layout: Layout, device=None):
    """The layout's offsets for coordinates 0 .. size-1, as an int64 tensor."""
    import torch

    return torch.tensor(layout.offsets(), dtype=torch.int64, device=device)


def gather_with_layout(x, layout: Layout):
    """Read the flat tensor `x` through `layout`: out[i] = x[layout(i)]."""
    return x.reshape(-1)[layout_index_tensor(layout, x.device)]
