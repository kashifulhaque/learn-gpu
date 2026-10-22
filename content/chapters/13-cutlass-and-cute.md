---
title: CUTLASS and CuTe layouts
slug: 13-cutlass-and-cute
part: "Part 3 — Matrix multiplication"
summary: How CuTe turns the index arithmetic of every kernel so far into one algebra of layouts, how swizzles remove bank conflicts, and how CUTLASS builds a tensor-core GEMM from that algebra.
minutes: 120
gpu: true
objectives:
  - "Read a CuTe layout written as shape:stride, and compute the offset and the size and cosize it defines."
  - "Evaluate integer, tuple, and hierarchical coordinates, including the colexicographic order that an integer coordinate uses."
  - "Apply coalesce, composition, complement, logical divide, and logical product by hand, and say what each is for."
  - "Compute a Swizzle<B,M,S> offset, and show why it removes the bank conflicts of a column read and of an ldmatrix load."
  - "Describe a TiledCopy and a TiledMMA as thread-value layouts, and read the three steps of a CuTe GEMM."
  - "Name the layers of CUTLASS 3 and 4, and say when to choose CUTLASS, Triton, or hand-written CUDA."
lab: 13-layouts
---

# CUTLASS and CuTe layouts

> [!TLDR]
> - A CuTe layout is a pair of nested tuples, shape and stride, that defines a function from a coordinate to a memory offset. `(4,3):(1,4)` is a column-major 4 by 3 matrix, and `(4,3):(3,1)` is row-major.
> - A small algebra works on layouts: coalesce simplifies one, composition chains two, complement finds the gaps, and logical divide and product turn them into tilings. The index arithmetic in every kernel so far is an instance of it.
> - A swizzle, `Swizzle<B,M,S>`, XORs a few address bits into others. It permutes a shared-memory tile so that a column read of a 32 by 32 float tile goes from a 32-way bank conflict to none, and an `ldmatrix` phase from 8-way to none.
> - A TiledCopy and a TiledMMA are layouts that map a thread and a value index to a position in a tile. A CuTe GEMM is three steps: tile the matrices, partition the tiles among threads, and call `copy` and `gemm`.
> - CUTLASS 3 and 4 stack that algebra into atom, tiled MMA, collective, kernel, and device layers. The CuTe DSL exposes the same layouts in Python.

Chapters 10 to 12 built a GEMM from index arithmetic. Every kernel had lines like `row * lda + col`, a shared-memory pad of one word, an `ldmatrix` address computed from the lane number, and a loop that walked tiles. Each worked, and each had to be rewritten for the next tile shape, the next data type, and the next architecture.

CUTLASS is the NVIDIA library that stopped rewriting them. Its core, CuTe, treats the mapping from a logical coordinate to a physical offset as a first-class object that you can compose, divide, and print. When you read CUTLASS source, a tensor-core GEMM for Ampere and one for Hopper differ mostly in which layouts and which atoms they name, not in how the arithmetic is written.

This chapter teaches that algebra from first principles, and the lab has you implement it in Python, where you can check every result against the CuTe documentation. Then it shows how the pieces become a GEMM, and when to reach for CUTLASS instead of Triton or hand-written CUDA.

## Before you start

You need the following from earlier chapters:

- **Row-major and column-major.** Element `(i, j)` of an $M \times N$ row-major matrix is at offset $iN + j$, and of a column-major matrix at $i + jM$. The distance between neighbours in one direction is the *stride*. ([Chapter 10](/c/10-gemm-from-scratch) uses this for every operand.)
- **Shared-memory banks.** Shared memory has 32 banks of 4 bytes, and the bank of a word is its address in words modulo 32. A warp access takes as many transactions as the most words that share one bank. ([Chapter 8](/c/08-shared-memory-and-transpose).)
- **`mma.sync` and `ldmatrix`.** One warp computes a 16 by 8 by 16 tile product from registers, and `ldmatrix` fills those registers from shared memory. ([Chapter 11](/c/11-tensor-cores).)
- **`cp.async`.** A thread copies 16 bytes from global to shared memory without staging through registers. ([Chapter 12](/c/12-pipelining-and-async-copies).)

## Why a layout algebra

This section answers one question: what problem does CuTe solve that a helper function for `row * lda + col` doesn't?

Take the ingredients of the tensor-core GEMM from chapter 11 and list the index arithmetic each one needed:

| Step | The arithmetic you wrote | What changes when the tile shape changes |
|---|---|---|
| Global tile load | block row $\times$ `lda` plus thread row $\times$ `lda` plus a vector offset | The thread-to-element map |
| Shared-memory store | An XOR or a pad so the later reads miss the same banks | The swizzle bits |
| `ldmatrix` addresses | Lane number split into a matrix number and a row number | The tile width |
| `mma.sync` operands | The PTX fragment table, for example row $= \lfloor \text{lane}/4 \rfloor$ | The instruction shape |
| Epilogue store | The accumulator fragment layout, again | The instruction shape |

Each row is a function from *where I am logically* to *where the byte is physically*. You wrote the functions by hand, in different notations, and each carried its own bugs. A layout algebra replaces them with one notation and a few operations that are correct by construction. The payoff is that code like "partition this tile among 128 threads" becomes a library call that works for any tile.

> [!KEY] One object, many jobs
> In CuTe, the same type describes a matrix in global memory, a swizzled tile in shared memory, the ownership of elements by threads, and the fragment of one `mma` instruction. A kernel is then a sequence of operations on layouts, and a new architecture mostly means new layouts.

## A layout is a function

A layout is a *shape* and a *stride*, written `shape:stride`. This section defines the function they make and the two numbers, size and cosize, that describe it.

Let the shape be a tuple $S = (s_0, s_1, \ldots, s_{n-1})$ and the stride a tuple $D = (d_0, d_1, \ldots, d_{n-1})$ of the same length. A *coordinate* is a tuple $(c_0, \ldots, c_{n-1})$ with $0 \le c_i < s_i$. Multiply each coordinate by its stride and add:

$$
\boxed{L(c_0, \ldots, c_{n-1}) = \sum_i \hla{c_i}\,\hlb{d_i}}
$$

Here $\hla{c_i}$ is the position along mode $i$ and $\hlb{d_i}$ is how far, in elements, one step along that mode moves you in memory. A layout does nothing else: it multiplies and adds.

Two numbers describe it. The *size* is the number of coordinates, $\prod_i s_i$. The *cosize* is one past the largest offset the layout produces, which is the length of the smallest buffer that holds every element. For nonnegative strides:

$$
\text{size}(L) = \prod_i s_i, \qquad \text{cosize}(L) = 1 + \sum_i (s_i - 1)\,d_i
$$

The following table evaluates two layouts of a 4 by 3 matrix. Each cell is the offset of the element at that row and column:

| `(4,3):(1,4)`, column-major | col 0 | col 1 | col 2 | `(4,3):(3,1)`, row-major | col 0 | col 1 | col 2 |
|---|---|---|---|---|---|---|---|
| row 0 | 0 | 4 | 8 | row 0 | 0 | 1 | 2 |
| row 1 | 1 | 5 | 9 | row 1 | 3 | 4 | 5 |
| row 2 | 2 | 6 | 10 | row 2 | 6 | 7 | 8 |
| row 3 | 3 | 7 | 11 | row 3 | 9 | 10 | 11 |

Both have size 12 and cosize 12, because both are compact. A padded layout wastes the difference. Take `(4,3):(2,16)`: it has size 12 and cosize $1 + 3 \cdot 2 + 2 \cdot 16 = 39$, so a buffer for it holds 39 words and uses 12 of them.

> [!EXAMPLE] A padded tile
> Chapter 8 padded a 32 by 32 float tile to 33 words per row to avoid bank conflicts. As a row-major layout that is `(32,32):(33,1)`, with size 1024 and cosize $1 + 31 \cdot 33 + 31 = 1055$. The buffer you allocate is 32 rows of 33 words, 1056 words, which is 3.1% more than the 1024 the data needs.

The lab's `Layout` class stores a shape and a stride, and its `__call__`, `size`, and `cosize` are the first things you implement. Every table in this chapter was produced by running the lab's solution, so you can reproduce each one.

## Hierarchical layouts and coordinates

A shape's entries can themselves be tuples. This section shows why that is the feature that lets one layout describe a tiled matrix, and how an integer coordinate is read.

Take the layout `((2,2),3):((1,6),2)`. Its first mode has shape `(2,2)` and stride `(1,6)`, and its second mode is `3:2`. As a matrix it has 4 rows and 3 columns, but a row index is a pair, $(i_0, i_1)$, with row $i = i_0 + 2 i_1$. The offset of the element at row $(i_0, i_1)$ and column $j$ is:

$$
L\big((i_0, i_1), j\big) = 1 \cdot i_0 + 6 \cdot i_1 + 2 \cdot j
$$

Evaluating all twelve elements gives the following grid:

| | col 0 | col 1 | col 2 |
|---|---|---|---|
| row 0 | 0 | 2 | 4 |
| row 1 | 1 | 3 | 5 |
| row 2 | 6 | 8 | 10 |
| row 3 | 7 | 9 | 11 |

Rows 0 and 1 form a 2 by 3 column-major block at offsets 0 to 5, and rows 2 and 3 form a second block at offsets 6 to 11. The nesting records that the matrix is two blocks stacked, and the stride 6 is the distance between the blocks. A flat shape can't say that.

The figure that follows evaluates layouts on a 4 by 6 grid. Choose a preset, click a cell to see its hierarchical coordinate, and set the two strides yourself with **Custom** to see when offsets collide.

```viz
13-layout-grid
```

### One layout, three kinds of coordinate

CuTe accepts three kinds of coordinate for the same layout, and they name the same element:

- A **tuple** that matches the shape's nesting, such as `((1,0),2)` for the layout above.
- A **tuple with an integer where the shape has a sub-tuple**, such as `(2, 2)`: row 2, column 2.
- A single **integer**, which indexes the whole layout.

An integer coordinate $c$ into a shape $(s_0, s_1, \ldots)$ is split *colexicographically*, with the first mode varying fastest:

$$
c_0 = c \bmod s_0, \qquad c_1 = \left\lfloor \frac{c}{s_0} \right\rfloor \bmod s_1, \qquad \ldots
$$

For the 4 by 3 matrix above, integer $c = 5$ is row $5 \bmod 4 = 1$ and column $\lfloor 5/4 \rfloor = 1$, which is offset 3. That is the order of a column-major matrix: the index runs down the first column, then the second.

> [!NOTE] Why colexicographic
> The order puts the first mode at stride 1 when the layout is column-major, so an integer index into a column-major matrix is the offset itself. The lab's tests check this: for `(2,(3,2)):(1,(2,6))`, the integers 0 to 11 map to the offsets 0 to 11 unchanged.

Because every layout accepts an integer coordinate, you can treat any layout as a one-dimensional sequence of offsets: `offsets = [L(i) for i in range(size(L))]`. Most of the algebra below is a statement about that sequence.

## Coalesce

Coalesce answers one question: is there a simpler layout that gives the same sequence of offsets? It flattens nesting, drops size-1 modes, and merges neighbours that continue each other.

Take two adjacent modes $s_0{:}d_0$ and $s_1{:}d_1$. CuTe applies these rules from left to right:

1. A mode of size 1 contributes nothing, so drop it.
2. If $d_1 = s_0 d_0$, the second mode starts exactly where the first ends, so merge them into $(s_0 s_1){:}d_0$.
3. Otherwise keep both.

The CuTe documentation's example is `(2,(1,6)):(1,(6,2))`. Flattened, it is `(2,1,6):(1,6,2)`. Drop the size-1 mode to get `(2,6):(1,2)`. The stride of the second mode, 2, equals the first mode's $2 \times 1$, so they merge into `12:1`. The layout was a contiguous run of 12 elements in disguise.

Coalesce also takes a *profile* that coalesces each top-level mode separately and keeps the rank. With the profile `(1,1)`, the same layout becomes `(2,6):(1,2)`. A layout with a gap resists: `(8,4):(1,10)` has a second stride of 10, not 8, so it stays as it is.

> [!TIP] Coalesce tells you the vector width
> A thread that owns a layout whose coalesced form is `8:1` holds eight consecutive elements. For 16-bit data, that is 16 bytes, so it can use one 128-bit load or `cp.async`. A coalesced form like `(2,4):(1,10)` can't. CuTe uses exactly this test to pick the widest copy instruction.

## Composition

Composition answers: if one layout picks elements out of another, where do they land? This is the operation that turns "give each thread its elements" into code.

For layouts $A$ and $B$, the composition $R = A \circ B$ is the layout with $R(c) = A(B(c))$. Think of $B$ as choosing *which* elements, as indices into a sequence, and $A$ as saying *where* each index lives. The result is a layout whose domain is $B$'s and whose codomain is $A$'s.

Take $A$ to be `20:2`, every second element of a buffer, and $B$ to be `(5,4):(4,1)`, a row-major 5 by 4 index grid:

$$
\texttt{20:2} \circ \texttt{(5,4):(4,1)} = \texttt{(5,4):(8,2)}
$$

At coordinate $(i, j)$, $B$ gives the index $4i + j$ and $A$ turns that index into the offset $2(4i + j) = 8i + 2j$, which is the layout `(5,4):(8,2)`.

When $A$ has several modes the result needs more care. The next example is the CuTe documentation's: $A$ is `(6,2):(8,2)` and $B$ is `4:3`. Read $A$ as a sequence of 12 indices, with the index $k = a_0 + 6 a_1$ at offset $8 a_0 + 2 a_1$. The layout $B$ takes every third index: $k = 0, 3, 6, 9$. These are $(a_0, a_1) = (0,0), (3,0), (0,1), (3,1)$, so the offsets are $0, 24, 2, 26$:

$$
\texttt{(6,2):(8,2)} \circ \texttt{4:3} = \texttt{(2,2):(24,2)}
$$

The result steps by 24 first, then by 2. The stride 3 divides the first mode's size 6, so $B$ takes two steps of 3 inside mode 0, at offset stride $3 \cdot 8 = 24$, and then continues into mode 1 at stride 2.

### The divisibility condition

Composition isn't defined for every pair. Walk through $A$'s modes from the first, with a remaining stride $r$ from $B$. For each mode of size $s$, one of the following must hold:

- $r$ is a multiple of $s$: skip the whole mode and divide $r$ by $s$.
- $s$ is a multiple of $r$: take up to $s / r$ elements of $B$ from this mode, at stride $r$ times the mode's stride, then set $r$ to 1.

If neither holds, the result isn't a layout. Compose `(6,2):(8,2)` with `4:4`: the stride 4 isn't a multiple of 6, and 6 isn't a multiple of 4, so the composition is undefined, and the lab's implementation raises an error. The Colfax note on the algebra ([Shah, 2024](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)) states the conditions precisely.

### Composition by mode

When $B$ is a tuple, CuTe pairs its entries with $A$'s modes and composes each pair. A tuple of layouts used this way is a *tiler*. A tuple of plain integers is a tiler of layouts $n{:}1$. For $A$ = `(12,(4,8)):(59,(13,1))`, the tiler `(3,8)` takes the first three of mode 0 and eight of mode 1:

$$
\texttt{(12,(4,8)):(59,(13,1))} \circ \texttt{(3,8)} = \texttt{(3,(4,2)):(59,(13,1))}
$$

Mode 0 became `3:59`, and the eight elements of mode 1 are the whole first sub-mode of 4 plus 2 of the second. The tiler `<3:4, 8:2>`, with strides, gives `(3,(2,4)):(236,(26,1))` instead.

This is how CuTe selects a sub-block of a matrix: compose the matrix layout with a tiler of the block's shape.

## Complement

Complement answers: which offsets does this layout skip? It is the second half of tiling.

Given a layout $A$ and a target size $M$, the complement $A^{*}$ is the layout with positive, increasing strides that steps over everything $A$ doesn't reach, up to $M$. Its codomain is disjoint from $A$'s, and together they cover $M$ elements. The documentation's examples, all with $M = 24$, are:

| Layout $A$ | $A^{*}$ for $M = 24$ | Reading |
|---|---|---|
| `4:1` | `6:4` | $A$ covers 0 to 3, so the complement steps by 4, six times. |
| `6:4` | `4:1` | $A$ covers every 4th offset from 0 to 20, so the complement is the four offsets 0 to 3 that shift it. |
| `(4,6):(1,4)` | `1:0` | $A$ already covers all 24. |
| `4:2` | `(2,3):(1,8)` | $A$ covers 0, 2, 4, 6. The complement is `2:1` for the odd offset, then `3:8` for the next three blocks. |
| `(2,4):(1,6)` | `3:2` | $A$ covers the pair $\{0, 1\}$ at every 6th offset, so the complement steps by 2, three times, to fill each block of 6. |
| `(2,2):(1,6)` | `(3,2):(2,12)` | $A$ covers 0, 1, 6, 7. The complement steps by 2 three times, then by 12. |

Check the fourth row by adding. The offsets of `4:2` are $\{0, 2, 4, 6\}$ and of `(2,3):(1,8)` are $\{0, 1, 8, 9, 16, 17\}$. Every sum of one from each is distinct, and together they are exactly $0, \ldots, 23$.

The algorithm sorts $A$'s modes by stride. It keeps a running extent, starting at 1. For a mode with stride $d$ and size $s$, the gap before it is $d$ divided by the extent, and it emits a complement mode of that size at stride equal to the extent. It then sets the extent to $s \cdot d$. A final mode of $\lceil M / \text{extent} \rceil$ closes the range. That requires each stride to be a multiple of the extent so far, and a layout that fails it has no complement.

## Logical divide and logical product

Divide and product answer the question that a GEMM asks of every matrix: how do I split this into tiles, and how do I tile this to cover a larger space?

The *logical divide* of $A$ by a tile $B$ composes $A$ with the pair of $B$ and its complement:

$$
A \oslash B = A \circ (B,\ B^{*})
$$

The tile $B$ picks a set of elements, and its complement $B^{*}$ enumerates the tiles. The result has two modes: mode 0 indexes inside a tile, and mode 1 indexes which tile.

### A one-dimensional divide

The simplest case is a contiguous buffer. Divide `16:1` by `4:1`, which is 4 contiguous elements:

$$
\texttt{16:1} \oslash \texttt{4:1} = \texttt{(4,4):(1,4)}
$$

Mode 0 is `4:1`, four adjacent elements, and mode 1 is `4:4`, four tile origins that are 4 apart. Now divide by `4:4`, which is 4 elements at stride 4:

$$
\texttt{16:1} \oslash \texttt{4:4} = \texttt{(4,4):(4,1)}
$$

This time the tile is strided: element 0 of tile 0 is offset 0, element 1 is offset 4. Tile 1 starts at offset 1. The tiles interleave. The same operation gives blocked and cyclic partitions, and the choice is the tile layout.

The documentation's example with a non-trivial $A$ is $A$ = `(4,2,3):(2,1,8)` divided by `4:2`. The complement of `4:2` for size 24 is `(2,3):(1,8)`, so $(B, B^{*})$ = `(4,(2,3)):(2,(1,8))`, and the divide is:

$$
\texttt{(4,2,3):(2,1,8)} \oslash \texttt{4:2} = \texttt{((2,2),(2,3)):((4,1),(2,8))}
$$

### Tiling a matrix

For matrices, divide each mode by its own tile with a tiler tuple. Take an 8 by 6 column-major matrix, `(8,6):(1,8)`, and a 4 by 3 tile. The *zipped* divide regroups the result as `((tile modes), (rest modes))`:

$$
\text{zipped\_divide}\big(\texttt{(8,6):(1,8)},\ (4{:}1,\ 3{:}1)\big) = \texttt{((4,3),(2,2)):((1,8),(4,24))}
$$

Mode 0, `(4,3):(1,8)`, is one 4 by 3 tile. Mode 1, `(2,2):(4,24)`, is the 2 by 2 grid of tiles. The tile at grid position $(1,1)$ starts at offset $4 \cdot 1 + 24 \cdot 1 = 28$. In CUTLASS the call `local_tile(mA, tiler, coord)` does this division and then fixes the rest mode to a block's coordinate, which leaves exactly the tile that one thread block owns. A *tiled* divide leaves the rest modes unzipped: `((4,3),2,2)`.

> [!EXAMPLE] The tile of a GEMM operand
> Take a 128 by 64 row-major slice of A, `(128,64):(64,1)`, and divide it by an 8 by 16 tile. The zipped divide is `((8,16),(16,4)):((64,1),(512,16))`. The size is $8 \cdot 16 \cdot 16 \cdot 4 = 8192 = 128 \cdot 64$. The tile below the first starts at offset 512, which is 8 rows of 64, and the tile to its right starts at offset 16. The tile layout `(8,16):(64,1)` keeps the row stride of 64, so a tile is not contiguous, and CuTe records that for free.

### The logical product

The *logical product* is the opposite direction: it repeats a layout $A$ according to a layout $B$ of positions:

$$
A \otimes B = \big(A,\ A^{*} \circ B\big)
$$

Mode 0 is $A$ itself and mode 1 places copies of it. Take the documentation's example, $A$ = `(2,2):(4,1)` and $B$ = `6:1`. The complement of $A$ for $\text{size}(A) \cdot \text{cosize}(B) = 4 \cdot 6 = 24$ elements is `(2,3):(2,8)`, and the product is:

$$
\texttt{(2,2):(4,1)} \otimes \texttt{6:1} = \texttt{((2,2),(2,3)):((4,1),(2,8))}
$$

It is the same layout as the divide of `(4,2,3):(2,1,8)` by `4:2`, because divide and product describe one structure from two ends. A blocked product of a 2 by 2 column-major block with a 3 by 2 grid is `(2,2):(1,2)` $\otimes$ `(3,2):(1,3)` $=$ `((2,2),(3,2)):((1,2),(4,12))`. Each block is contiguous, and block $(p, q)$ starts at $4p + 12q$.

> [!KEY] Divide to read, product to build
> Use logical divide when you have a layout and want to split it into tiles, as when a block takes its slice of a matrix. Use logical product when you have a tile and want to repeat it, as when a thread's 16-byte copy is replicated across a warp.

## Swizzle

A swizzle isn't an operation on shape and stride. It is a function on the offset that the layout produces, and it answers one question: how do you keep both a row read and a column read of a shared-memory tile free of bank conflicts?

Chapter 8 gave one answer, padding. Padding by one word breaks 16-byte alignment: a row of 33 words starts at byte 132, so you lose 128-bit loads, `cp.async`, and `ldmatrix`. Padding by a whole 16-byte chunk keeps the alignment, but for a row of 64 halves it adds 8 halves, which is 12.5% more shared memory. A swizzle keeps the rows aligned and compact, and rearranges *within* each row.

### The functor

`Swizzle<B,M,S>` acts on an integer offset. Its three parameters are:

- $B$, the number of bits in the mask.
- $M$, the number of low bits that stay in place. The $2^M$ elements that share them move together as a unit.
- $S$, the shift distance.

Take the $B$-bit field that starts at bit $M + \max(0, S)$, shift it by $S$ toward the low end, and XOR it into the $B$-bit field that starts at bit $M - \min(0, S)$. For $S > 0$ the source field is above the target field. The requirement $|S| \ge B$ keeps the two fields from overlapping.

The Lei Mao post on CuTe swizzles gives a check. Apply `Swizzle<5,0,6>` to offset 65, binary `1000001`. The source field is bits 6 to 10, and only bit 6 is set, so the field holds 1. Shift it down by 6 to bit 0 and XOR: $65 \oplus 1 = 64$. Because XOR with a function of *other* bits is its own inverse, a swizzle is a bijection on each aligned block of $2^{M+B+|S|}$ offsets, and applying it twice returns the original offset.

### A 32 by 32 float tile

For floats, the tile has offset $32r + c$ at row $r$ and column $c$. The bank is the offset modulo 32, which is the column. A warp reading column 5 of the tile touches offsets $5, 37, 69, \ldots$, all in bank 5: a 32-way conflict, the 32-transaction case from chapter 8.

Apply `Swizzle<5,0,5>`. With $M = 0$ and $S = 5$, the source field is bits 5 to 9, which is the row $r$, and the target is bits 0 to 4, which is the column. The new column is $c \oplus r$:

$$
\text{bank}(r) = (c \oplus r) \bmod 32
$$

Column 5 now lands in banks $5, 4, 7, 6, 1, 0, \ldots$ for rows $0, 1, 2, 3, 4, 5, \ldots$. The 32 values $5 \oplus r$ are all different, so the column read is conflict-free. A row read, where $r$ is fixed, changes every column by the same XOR, so it stays a permutation of the 32 banks and is conflict-free too. The lab's tests assert all of this: 32-way before, 1-way after, for both orders of access.

The figure that follows lets you try other parameters. Change $B$, $M$, and $S$ and watch the bars. The swizzle `<5,0,5>` is the one that fixes this tile, and a swizzle with $B$ larger than $S$ breaks the bijection, which the readout reports.

```viz
13-swizzle-banks
```

### The tile that `ldmatrix` reads

Tensor-core tiles use 16-bit elements, and the unit that matters is a 16-byte chunk, the width of one `ldmatrix` row (chapter 11). Take a tile of 8 rows of 64 halves. A row is 128 bytes, so every row starts at the same bank. An `ldmatrix` phase reads one 16-byte chunk from each of 8 rows. Without a swizzle, the 8 addresses are 128 bytes apart, so all 8 chunks cover the same 4 banks:

$$
8 \text{ rows} \times 4 \text{ words} = 32 \text{ words in } 4 \text{ banks} \;\Rightarrow\; 8\text{-way conflict}
$$

Use `Swizzle<3,3,3>` on element offsets. The mask has $B = 3$ bits, the low $M = 3$ bits (8 halves, one chunk) stay put, and $S = 3$ moves the row's low 3 bits (offset bits 6 to 8) onto the chunk index (offset bits 3 to 5). The chunk index becomes $\text{chunk} \oplus (r \bmod 8)$. Printing the physical chunk of each logical chunk gives:

| Row | chunk 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| 1 | 1 | 0 | 3 | 2 | 5 | 4 | 7 | 6 |
| 2 | 2 | 3 | 0 | 1 | 6 | 7 | 4 | 5 |
| 3 | 3 | 2 | 1 | 0 | 7 | 6 | 5 | 4 |
| 4 | 4 | 5 | 6 | 7 | 0 | 1 | 2 | 3 |
| 5 | 5 | 4 | 7 | 6 | 1 | 0 | 3 | 2 |
| 6 | 6 | 7 | 4 | 5 | 2 | 3 | 0 | 1 |
| 7 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

Reading logical chunk 0 from the 8 rows now touches physical chunks $0, 1, \ldots, 7$, which cover all 32 banks once: conflict-free. The `cp.async` that writes a row of 8 chunks still writes 8 different chunks, so it stays conflict-free. The lab's `bank_conflict_degree` returns 8 for the plain tile and 1 for the swizzled one on this access.

> [!NOTE] Bytes, elements, and Hopper
> The same swizzle in byte offsets is `Swizzle<3,4,3>`: the 16-byte chunk is $M = 4$ bits. For 2-byte elements the two forms agree, which the lab's tests confirm for 8192 offsets. On Hopper, the TMA engine can write a tile into shared memory already in a 128-byte swizzle pattern, and `wgmma` reads shared-memory operands in such layouts. Both use these swizzles through the same algebra. Chapter 12 introduces TMA and `wgmma`, and the lab covers only the Ampere case.

### A swizzle composes with a layout

A swizzle takes an offset and returns an offset, so it chains after any layout: `composition(Swizzle<3,3,3>{}, Layout<Shape<_8,_64>, Stride<_64,_1>>{})` is a row-major 8 by 64 tile followed by the XOR. CuTe calls the result a *composed layout*. It is still a function from a coordinate to an offset, so every other operation that accepts a layout accepts it. That is the point: you can ask for the swizzled tile and then partition it among threads, and the partition respects the swizzle.

## Thread-value layouts

A layout can also say who owns what. This section shows how CuTe describes a copy or a matrix multiply as a *thread-value* layout, which is the representation that TiledCopy and TiledMMA use.

A thread-value (TV) layout maps a pair (thread index, value index) to a coordinate in a tile. The thread mode says which of the threads is asking, and the value mode says which of that thread's elements it is asking for.

### A copy

Take 32 threads copying an 8 by 32 half-precision tile, row-major, `(8,32):(32,1)`. Give each thread 8 consecutive halves, which is 16 bytes. Four threads cover one 32-half row, and eight such groups cover the tile. The TV layout is:

$$
\texttt{((4,8),8):((8,32),1)}
$$

The thread index $t = t_0 + 4 t_1$ splits into $t_0 \in [0,4)$ with stride 8 (which 8-element chunk of the row) and $t_1 \in [0,8)$ with stride 32 (which row). The value mode `8:1` is eight consecutive halves. Evaluating all $32 \cdot 8$ pairs gives each of the 256 tile offsets exactly once, so the layout partitions the tile, and its value mode coalesces to `8:1`, so the hardware can use a 128-bit access. This is the TV layout of a `TiledCopy` with a 128-bit copy atom.

### An `mma` fragment

The same notation describes a tensor-core instruction. The accumulator of `mma.sync.m16n8k16` is a 16 by 8 tile that each of the 32 lanes holds 4 elements of. CuTe's mma traits for the Ampere 16 by 8 by 16 half-precision instruction give the accumulator TV layout as `((4,8),(2,2)):((32,1),(16,8))`, with offsets into a column-major 16 by 8 tile (offset $= m + 16 n$).

Decode it. Lane $\ell = t_0 + 4 t_1$ and value $v = v_0 + 2 v_1$. The offset is $32 t_0 + t_1 + 16 v_0 + 8 v_1$, so:

$$
m = t_1 + 8 v_1 = \left\lfloor \frac{\ell}{4} \right\rfloor + 8 \left\lfloor \frac{v}{2} \right\rfloor, \qquad n = 2 t_0 + v_0 = 2 (\ell \bmod 4) + (v \bmod 2)
$$

This is the accumulator fragment table of the PTX ISA: lane $\ell$ holds rows $\lfloor \ell/4 \rfloor$ and $\lfloor \ell/4 \rfloor + 8$, and columns $2(\ell \bmod 4)$ and $2(\ell \bmod 4)+1$. You can verify it in a few lines of Python with your `Layout`: evaluate all 128 pairs and compare with the formulas. The layout is the table.

The A operand uses `((4,8),(2,2,2)):((32,1),(16,8,128))` over a column-major 16 by 16 tile. Its 256 pairs cover each of the 256 offsets once, and lane 1 holds the elements at (row, column) $(0,2), (0,3), (8,2), (8,3), (0,10), (0,11), (8,10), (8,11)$, which follows the A-fragment table in the PTX ISA for this instruction.

### Atoms, tiles, and partitions

A *MMA atom* is one hardware instruction and its three TV layouts (A, B, C). A *copy atom* is one copy instruction and its source and destination TV layouts. A *tiled* version repeats the atom across more threads or more values with a layout of its own:

- `make_tiled_mma(atom, Layout<Shape<_2,_2,_1>>{})` arranges 2 by 2 by 1 copies of the 16 by 8 by 16 atom over M, N, and K. That is 4 warps, 128 threads, covering a 32 by 16 by 16 tile.
- `make_tiled_copy(copy_atom, thread_layout, value_layout)` repeats a copy atom over a thread layout, with each thread owning a value layout.

*Partitioning* is where the algebra pays off. Given a tiled MMA and a thread index, `get_slice(tid).partition_C(tile)` composes the tile's layout with the TV layout and returns the elements that this thread owns, with the same call for any tile size. The extra modes that appear when the tile is larger than the MMA's own tile are the *rest* modes, which are what the K loop and the repeated issue of instructions iterate over.

## A GEMM in CuTe

This section puts the pieces together. It shows the shape of a CuTe GEMM kernel for one thread block, so that you can read the real CUTLASS examples.

The excerpt that follows is abbreviated from the style of CUTLASS's Ampere examples and the Colfax tutorials. It is not compiled by this course, and it omits the multistage pipeline of chapter 12 and the epilogue. Read it for its structure:

```cpp
using namespace cute;

// One block computes a 128 x 128 tile of C and walks K in steps of 64.
auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<64>{});

// Global tensors: shape and stride are all a tensor needs.
Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(N, K), make_stride(K, Int<1>{})));
Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, N), make_stride(N, Int<1>{})));

// Step 1: tile. local_tile is a zipped divide plus a slice by block coordinate.
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (128, 64, k_tiles)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (128, 64, k_tiles)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (128, 128)

// Shared-memory tiles: a swizzled 8 x 64 atom, repeated to fill 128 x 64.
auto atom = composition(Swizzle<3,3,3>{}, Layout<Shape<_8,_64>, Stride<_64,_1>>{});
auto sA_layout = tile_to_shape(atom, make_shape(Int<128>{}, Int<64>{}));
__shared__ half_t smem_a[cosize_v<decltype(sA_layout)>];
Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);
// sB is built the same way.

// Step 2: partition among threads.
TiledCopy copy_a = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},   // thread layout: 16 rows x 8 chunks
    Layout<Shape<_1, _8>>{});                    // value layout: 8 halves = 16 bytes
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);        // this thread's source elements
Tensor tAsA = thr_copy_a.partition_D(sA);        // this thread's destination elements

TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_2, _2, _1>>{});
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);
Tensor tCrA = thr_mma.make_fragment_A(tCsA);     // registers for A fragments
Tensor tCrC = thr_mma.partition_fragment_C(gC);  // accumulators
clear(tCrC);

// Step 3: copy a K tile, then multiply it.
for (int k = 0; k < size<2>(gA); ++k) {
  copy(copy_a, tAgA(_, _, _, k), tAsA);
  cp_async_fence(); cp_async_wait<0>(); __syncthreads();
  // (copy sB likewise; then ldmatrix from smem into tCrA and tCrB)
  gemm(mma, tCrA, tCrB, tCrC);
  __syncthreads();
}
// Epilogue: write tCrC to gC through the C partition.
```

Read it in three beats. Tiling picks the block's slice of each matrix. Partitioning gives each thread its elements of that slice, using the TV layouts of the copy and the MMA, and the shared-memory layout carries the swizzle so that those partitions read and write without bank conflicts. The loop then issues `copy` and `gemm`, which expand to `cp.async`, `ldmatrix`, and `mma.sync`, as in the kernels you wrote by hand.

The arithmetic of this tile follows from the layouts:

| Quantity | Value | How |
|---|---|---|
| Shared memory for A | 16 KiB | $128 \times 64 \times 2$ bytes |
| Shared memory for A and B | 32 KiB | Two operands, one stage |
| Threads | 128 | 2 by 2 warps |
| Accumulators per thread | 128 fp32 registers | $128 \cdot 128 / 128$ threads, of the 255 a thread can use |
| `mma.sync` per warp per K step | 128 | Each warp owns 4096 of the 16384 accumulators, a 64 by 64 piece: $(64/16) \cdot (64/8) \cdot (64/16) = 4 \cdot 8 \cdot 4$ |

The last row follows from the layouts: the warp's share of C is 4 instruction tiles of 16 rows by 8 tiles of 8 columns, and the 64-deep K step takes four 16-deep instructions. At 4096 FLOP per `mma.sync` (chapter 11), a block does $4 \cdot 128 \cdot 4096 = 2{,}097{,}152$ FLOP per K step, which equals $2 \cdot 128 \cdot 128 \cdot 64$.

> [!WARNING] Static and dynamic integers
> In CuTe C++, `_8` and `Int<8>{}` are compile-time constants, and plain `int` is dynamic. A layout with static strides costs no registers and lets the compiler unroll and vectorize. `make_stride(K, Int<1>{})` is a dynamic leading dimension with a static unit stride, which is what you want for a row-major matrix. Making everything dynamic compiles, but it can cost you the vectorized loads.

## The layers of CUTLASS

CuTe is the foundation, and CUTLASS is a stack built on it. This section names the layers so that you can find your way through the repository.

NVIDIA's CUTLASS 3 design post ([Kerr and others, 2023](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)) describes five layers. From the bottom:

| Layer | What it is | Chooses |
|---|---|---|
| Atom | One hardware instruction and its layouts: an MMA atom or a copy atom. | The instruction: `mma.sync`, `wgmma`, `cp.async`, TMA |
| Tiled MMA and tiled copy | An atom repeated over threads and values. | The thread arrangement and the tile |
| Collective | A mainloop that moves tiles and multiplies them, plus an epilogue. | The pipeline: stages, producer and consumer roles |
| Kernel | A collective mainloop and epilogue wrapped in a grid, with a tile scheduler. | The work distribution across blocks |
| Device | The host-side adapter that launches the kernel. | The launch and workspace |

In code the top-level pieces are `cutlass::gemm::collective::CollectiveBuilder`, which assembles a mainloop from an architecture, a tile shape, and element types, `cutlass::epilogue::collective::CollectiveBuilder` for the epilogue, `cutlass::gemm::kernel::GemmUniversal` for the kernel, and `cutlass::gemm::device::GemmUniversalAdapter` for the device layer. A user who only needs a standard GEMM with a fused epilogue picks types at the top. A user who needs something new reaches down to a collective, or to the atom layer.

The design is *orthogonal*: the layers can be swapped independently. A new epilogue, such as a bias plus an activation, doesn't touch the mainloop. A new architecture swaps the atoms and the pipeline and leaves the kernel layer's scheduling. That is why one repository covers Volta through Blackwell, and why the Hopper and Blackwell chapters of this course show the same ideas with different atoms.

## CuTe in Python

CUTLASS 4 added a Python front end, the CuTe DSL, so that you can write the same layouts and kernels without C++ templates. This section says what it is and where the lab's Python fits.

CUTLASS 4.8.1 is the release the course targets ([the CUTLASS repository](https://github.com/NVIDIA/cutlass)). The [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) describes it as a Python framework that mirrors CuTe's layouts, tensors, and atoms, and that JIT-compiles Python kernels into device code at runtime. It lists Ampere, Hopper, and Blackwell support, and the [quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) lists the install command, `pip install nvidia-cutlass-dsl`, for Linux and Python 3.10 to 3.14.

The layout functions have the names you now know. The excerpt below follows the API reference for the 4.x series. The course's authors had no GPU when writing it, so check the signatures against the reference for your version before you rely on it:

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def layouts():
    a = cute.make_layout((6, 2), stride=(8, 2))
    b = cute.make_layout((4, 3), stride=(3, 1))
    print(cute.composition(a, b))              # the layout from the composition section
    print(cute.complement(cute.make_layout(4, stride=2), 24))
    tiled = cute.zipped_divide(cute.make_layout((8, 6), stride=(1, 8)), (4, 3))
    print(tiled)

layouts()  # traces the function, and prints the static layouts while tracing
```

The function runs at trace time, so the `print` shows layouts, and the reference's signatures take a layout or a shape for the target of a complement, and a layout, shape, or tiler for composition and divide. The printed form can group modes differently from the form in this chapter, because a layout with dynamic values prints as `((5,1),...)`, but the function is the same. The lab's `Layout` is a pure-Python model of these operations, and that is the reason to write it: when the DSL prints `((2,2),3):((24,2),8)`, you can derive it.

PyTorch 2.14's Inductor uses the DSL as a backend, called NVGEMM, which generates CUTLASS kernels with fused epilogues and autotunes them against Triton and ATen ([chapter 17](/c/17-fusion-and-compilers) covers this). The practical effect is that a layout you can read is also the layout of a kernel that `torch.compile` might pick for you.

## CUTLASS, Triton, or hand-written CUDA

Choosing between the three is a question about what you control and what it costs. This section gives a rule for each.

| Tool | Choose it when | What you pay |
|---|---|---|
| cuBLAS or `torch.matmul` | The operation is a standard GEMM. | Nothing; it is the baseline every other choice has to beat. |
| Triton ([chapter 14](/c/14-triton)) | You want a fused or custom kernel in a day, in Python, and the compiler's tile-level decisions are acceptable. | You can't name the instruction, the shared-memory layout, or the warp roles. |
| CUTLASS and CuTe, in C++ or the DSL | You need the tensor-core path on a given architecture, a custom epilogue, an unusual data type, or control of the pipeline and warp specialization. | Template-heavy C++ and compile times, or a younger Python toolchain, and a steeper learning curve. |
| Hand-written CUDA | The kernel isn't a tile algorithm, or you need an instruction or memory-access pattern that no framework exposes. | You write and verify every index, swizzle, and barrier yourself. |

Two heuristics follow from the chapter. First, if your problem is "a GEMM with something fused onto it", start with Triton or an existing CUTLASS epilogue. Second, if you find yourself writing a swizzle or a fragment table by hand, you are rebuilding CuTe, so use it instead. CUTLASS pays off most when the target is the newest tensor-core hardware, where the instruction's layouts are complicated enough that deriving them by hand costs more than learning the algebra.

## What goes wrong

- **Reading shape and stride as a matrix and its transpose.** `(4,3):(1,4)` and `(4,3):(3,1)` have the same shape and different strides. Row-major or column-major is a property of the stride, not the shape. When a result looks transposed, print the layout.
- **Composing without checking divisibility.** `(6,2):(8,2)` composed with `4:4` isn't defined, and a quick implementation that guesses can return a wrong layout. Your lab code raises on it. Test composition against the identity $R(i) = A(B(i))$ on every $i$, which the lab's property tests do.
- **Forgetting that coalesce keeps the function, not the shape.** A coalesced layout has the same sequence of offsets but a different rank. Don't coalesce a layout whose modes you intend to index separately, or use the profile form.
- **Expecting a complement for a layout whose strides don't nest.** `(2,2):(1,3)` has no complement for any target, because the stride 3 isn't a multiple of the 2 elements the first mode covers, and the lab's implementation raises. Separately, a broadcast layout such as `(4,6):(1,0)` has size 24 and cosize 4, so its size says nothing about how much memory it touches.
- **Using a swizzle with $|S| < B$.** The two bit fields overlap, and the result isn't a permutation, so two logical elements can share one physical word. Check that the swizzle is a bijection on its block, as the lab's test does.
- **Swizzling the data but not the address of the vector access.** A swizzle permutes 16-byte chunks only when $M$ covers the vector width. For 16-bit data and 16-byte vectors, $M$ must be at least 3. With $M = 0$ you swizzle individual halves, which breaks the vector alignment.
- **Believing the printed layout is faster.** A layout is a map. Performance comes from the access pattern it produces: whether adjacent threads touch adjacent bytes, and whether a warp's shared-memory addresses fall in different banks. The lab's `bank_conflict_degree` measures the second.

> [!RECAP]
> - A layout is `shape:stride`, a function that sums coordinate times stride. Its size is the product of the shape, and its cosize is the largest offset plus 1.
> - An integer coordinate is read colexicographically, so every layout is also a sequence of offsets, and nested shapes record tiling.
> - Coalesce simplifies without changing the function. Composition chains layouts, with a divisibility condition. Complement finds the offsets a layout skips.
> - Logical divide splits a layout into tiles and logical product repeats one. A zipped divide gives `((tile), (rest))`, which is what `local_tile` slices.
> - `Swizzle<B,M,S>` XORs bit field $[M+S, M+S+B)$ into $[M, M+B)$. It makes a 32 by 32 float column read conflict-free and an `ldmatrix` phase conflict-free, while keeping 16-byte alignment.
> - TiledCopy and TiledMMA are thread-value layouts. A CuTe GEMM tiles, partitions, then calls `copy` and `gemm`, and CUTLASS stacks atoms, collectives, kernels, and devices around that.

## Check your understanding

> [!QUESTION] What are the size and cosize of `(4,3):(2,16)`, and what fraction of the buffer holds data?
> The size is $4 \cdot 3 = 12$. The cosize is $1 + (4-1) \cdot 2 + (3-1) \cdot 16 = 39$. A buffer of 39 words holds 12 elements, so $12 / 39 = 30.8\%$ of it is data.

> [!QUESTION] Evaluate `((2,2),3):((1,6),2)` at the integer coordinate 7.
> Split 7 colexicographically over the two top-level modes. The first mode has size 4, so the row is $7 \bmod 4 = 3$ and the column is $\lfloor 7/4 \rfloor = 1$. Row 3 in the mode `(2,2):(1,6)` is $(i_0, i_1) = (1, 1)$, which is offset $1 + 6 = 7$. Column 1 adds $2 \cdot 1$. The offset is 9. The grid in the chapter shows 9 at row 3, column 1.

> [!QUESTION] Coalesce `(4,(1,2,3)):(1,(0,4,8))`. What stays, and why?
> Flatten to `(4,1,2,3):(1,0,4,8)`. Drop the size-1 mode, leaving `(4,2,3):(1,4,8)`. The second mode's stride 4 equals $4 \cdot 1$, so it merges into `8:1`. The third mode's stride 8 equals $8 \cdot 1$, so it merges too, giving `24:1`. The stride of 0 belonged to the size-1 mode, so it never mattered.

> [!QUESTION] Compute `20:2` composed with `(5,4):(4,1)`, and say what each mode of the result means.
> The result is `(5,4):(8,2)`. $B(i, j) = 4i + j$ is the index, and $A$ doubles it, so the offset is $8i + 2j$. Mode 0 steps over rows of 4 indices, which is 8 offsets, and mode 1 steps over single indices, which is 2 offsets.

> [!QUESTION] Why is the complement of `4:2` for $M = 24$ equal to `(2,3):(1,8)`?
> `4:2` covers $\{0, 2, 4, 6\}$. The first gap is the odd offset: the stride is 1, and the extent so far is 1, so the first complement mode is `2:1`. After the layout's last offset the extent is $4 \cdot 2 = 8$, so the closing mode is $\lceil 24/8 \rceil = 3$ at stride 8. Every sum of one offset from each layout is distinct, and together they cover 0 to 23.

> [!QUESTION] Why does `Swizzle<5,0,5>` fix a column read of a 32 by 32 float tile without hurting a row read?
> The offset is $32 r + c$, so bits 5 to 9 are the row and bits 0 to 4 are the column. The swizzle replaces the column with $c \oplus r$. For a fixed column, the 32 values $c \oplus r$ for $r = 0..31$ are all different, so the 32 banks differ. For a fixed row, XOR with the constant $r$ permutes the 32 columns, so the 32 words still fall in 32 distinct banks.

> [!QUESTION] An 8 by 64 half tile is read 16 bytes per row by `ldmatrix`. How many ways is the bank conflict with and without `Swizzle<3,3,3>`?
> Without the swizzle, each row is 128 bytes, so the 8 chunks start at the same bank and occupy the same 4 banks: 8-way. With it, chunk $c$ of row $r$ moves to $c \oplus (r \bmod 8)$, so reading the same logical chunk from 8 rows touches 8 different chunks, which together cover all 32 banks: no conflict.

> [!QUESTION] The A100 supports `mma.sync` and `cp.async` but not TMA or `wgmma`. What changes in a CuTe GEMM for Hopper, and what stays?
> The atoms change: a TMA copy atom and a `wgmma` MMA atom replace `cp.async` and `mma.sync`, along with their TV layouts and the swizzled shared-memory layouts they require. The algebra stays: you still tile with a divide, partition with TV layouts, and call `copy` and `gemm`. The collective's pipeline changes to producer and consumer warps, which is a different layer.

## Lab

> [!TRY] Implement the layout algebra in Python
> Open `13-layouts`. You implement a `Layout` class with `__call__`, `size`, and `cosize`, then `coalesce`, `composition`, `complement`, `logical_divide`, `logical_product`, `zipped_divide`, `swizzle`, and `bank_conflict_degree`. The tests check your code against the worked examples from the CuTe documentation, including every example in this chapter, run property tests on random layouts, and count the bank conflicts of a 32 by 32 float tile with and without `Swizzle<5,0,5>`. The final tests use a layout to drive an index gather in torch: you gather the transpose of a matrix, the tiles of a 12 by 20 matrix, and a swizzled tile, and compare each with a PyTorch reference. On the A100 the test also times a gather and reports `gather_gbs`. The core is pure Python, so it runs on any machine. Passing means every documentation case matches, the swizzled column read is 1-way, and the gather is exact.

The lab reports four metrics:

- `doc_examples_passed`: how many of the documentation's worked cases your code reproduces.
- `conflicts_unswizzled` and `conflicts_swizzled`: the bank-conflict degree of a column read before and after `Swizzle<5,0,5>`, which the chapter predicts as 32 and 1.
- `gather_gbs`: the bandwidth of an index gather on the GPU. The test counts 16 bytes per element: an 8-byte index read, a 4-byte data read, and a 4-byte write. The count is the useful bytes, and a gather through a transposed index reads the data in a scattered order, so each 4-byte read pulls in a whole 32-byte sector. The prediction is a figure well below the copy ceiling of 1275 GB/s that learn-inference measured on this card, and the lab measures how far below.

## Further reading

- [CuTe layout algebra](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/cute/02_layout_algebra.html), NVIDIA. The reference for every definition and worked example the lab uses.
- [CuTe quickstart](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/cute/00_quickstart.html), NVIDIA. Shapes, strides, layouts, and tensors in C++.
- [A note on the algebra of CuTe layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/), Jay Shah, Colfax Research. The precise conditions under which composition, complement, and divide are defined.
- [CuTe layout representation and algebra](https://arxiv.org/abs/2603.02298), Cris Cecka. The formal definitions, including inverses.
- [Categorical foundations for CuTe layouts](https://arxiv.org/abs/2601.05972), Carlisle, Shah, Stern, and VanKoughnett. The category-theoretic view, for readers who want the full rigor.
- [CuTe swizzle](https://leimao.github.io/blog/CuTe-Swizzle/), Lei Mao. How `Swizzle<B,M,S>` acts on offsets, with worked bit-level examples.
- [Fast matrix transpose in CUDA with CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/), Colfax Research. Swizzled shared-memory layouts as the alternative to padding, as in chapter 8.
- [CUTLASS 3.x: orthogonal, reusable, and composable abstractions for GEMM kernel design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/), NVIDIA. The atom, tiled, collective, kernel, and device layers.
- [The CUTLASS repository](https://github.com/NVIDIA/cutlass) and the [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html). Source, examples, and the Python front end.
- The companion course's [fusion in Triton](https://qwen.ifkash.dev/c/13-fusion-in-triton) and [FlashAttention](https://qwen.ifkash.dev/c/14-flash-attention) chapters, for the Triton side of the CUTLASS-or-Triton choice.
