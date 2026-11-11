---
title: Convolution and stencils
slug: 31-convolution-and-stencils
part: "Part 2 — CUDA patterns"
summary: Tile a convolution with halo cells and a constant-memory filter, coarsen a 3D stencil along z, then lower convolution to a GEMM with im2col and without it, using an index map.
minutes: 150
gpu: true
objectives:
  - "Compute the output size of a convolution from stride, padding, and dilation, and the arithmetic intensity of a direct convolution, and say when it is memory bound."
  - "Put a small filter in constant memory, and say what makes that cheaper than global or shared memory."
  - "Count the halo overhead of the input-tile and output-tile strategies as a function of tile and filter size, and make a halo load read zeros at the image border."
  - "Write a 3D 7-point stencil that keeps the z neighbors in registers and coarsens along z, and predict its time from the copy ceiling."
  - "Lower a convolution with im2col, size its memory blow-up, and say why PyTorch convolution is cross-correlation and needs no filter flip."
  - "Map GEMM coordinates (m, k) to input coordinates (n, c, h, w) without building the im2col matrix, including padding zeros, for NCHW and NHWC filters."
  - "Fuse convolution, ReLU, and maxpool, and count the bytes the fusion removes."
lab: 31-conv-stencil
---

# Convolution and stencils

> [!TLDR]
> - A direct convolution does $2k^2$ FLOP per output element and, with perfect reuse, moves 8 bytes. A single-channel $3 \times 3$ filter has an intensity of 2.25 FLOP per byte, well under the fp32 ridge of 10.1, so it is a memory kernel. Reuse is the whole game.
> - Put the filter in constant memory: every thread of a warp reads the same tap at once. Put the input tile in shared memory with a halo. A $16 \times 16$ output tile with a $3 \times 3$ filter loads 1.27 input elements per output, and a $5 \times 5$ filter loads 1.56.
> - A 3D 7-point stencil keeps its x-y plane in shared memory and its z neighbors in registers, and one thread walks a column of cells. That cuts the global reads per output from 7 to a predicted 1.6.
> - im2col turns a convolution into a GEMM and writes the input $k^2$ times: 9 times for a $3 \times 3$ filter at stride 1. Implicit GEMM keeps the GEMM and drops the matrix, by computing the input coordinate of each $(m, k)$ on the fly.
> - Fusing ReLU and a $2 \times 2$ maxpool into the convolution removes 109 MB of intermediate traffic on a ResNet-sized layer. NHWC makes the reduction index contiguous, which is why tensor-core convolutions prefer it.

Your course so far has been about transformers, and a transformer is mostly matrix multiplications and pointwise operations. Convolutions are the other half of what a GPU runs. A vision model's patch embedding is a strided convolution. Whisper and wav2vec-style speech encoders start with convolutions over audio. Mamba-style layers use a short causal 1D convolution. Outside ML, a stencil, which is a convolution with fixed coefficients applied over a grid, is the core of every finite-difference simulation.

The kernels are worth learning for a second reason: they are the cleanest examples of three ideas you already met. [Chapter 8](/c/08-shared-memory-and-transpose) stages a tile in shared memory. [Chapter 10](/c/10-gemm-from-scratch) turns reuse into arithmetic intensity. Here the tile needs a border, the *halo*, that its neighbors also load, and the question is how much that border costs. The last third of the chapter ties convolution to the GEMM of chapters 10 and 14.

This chapter is about kernels. Which convolutional network to build, how to count a layer's weights and receptive field, and how the backward pass works belong to the companion ml course. This chapter links there and does not repeat it.

## Before you start

**Shared-memory tiles and barriers.** A block stages a tile in shared memory, calls `__syncthreads()`, and reads it back at the speed of an SM, not of HBM. [Chapter 8](/c/08-shared-memory-and-transpose) builds this for a transpose.

**The roofline.** A kernel is memory bound when its FLOP per byte is below the ridge of FLOP/s over bytes/s. On the A100 80GB PCIe the fp32 ridge is $19.5 \times 10^{12} / 1.935 \times 10^{12} = 10.1$ FLOP per byte. Against the measured 1275 GB/s copy ceiling it is 15.3. ([Chapter 3](/c/03-memory-and-bandwidth) and [chapter 4](/c/04-roofline-and-latency).)

**A GEMM tile.** The loop over $K$ in a GEMM loads an $A$ tile and a $B$ tile and accumulates into registers. [Chapter 10](/c/10-gemm-from-scratch) builds it, and [chapter 14](/c/14-triton) writes the same loop in Triton, where this chapter's last kernel uses it.

**The convolution itself.** The ml course covers [convolution and the receptive field](https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field) and [im2col and the backward pass](https://ml.ifkash.dev/lessons/039-im2col-and-convolution-backward) in NumPy, and [residual networks](https://ml.ifkash.dev/lessons/040-residual-networks) for the architecture. You don't need them first, but you do need the definition, which the next section restates.

## Direct convolution and its arithmetic intensity

This section answers one question: how much arithmetic does a convolution do per byte, and is that enough to leave the memory side of the roofline?

Picture a small window, the filter, sliding over an image. At each position the window multiplies its weights by the pixels under it and adds the products into one output pixel. Every output pixel needs $k^2$ input pixels, and neighboring output pixels share most of them.

The deep learning convention, which PyTorch follows, does not flip the filter. The operation is *cross-correlation*, and the PyTorch documentation says so. Textbook convolution flips the filter first. For an $N \times C \times H \times W$ input $x$ and a $K \times C \times R \times S$ filter $w$, with stride $\sigma$, padding $\pi$, and dilation $\delta$, the output at $(n, k, p, q)$ is

$$
y[n,k,p,q] = \sum_{c=0}^{C-1} \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} \hla{w[k,c,r,s]}\; \hlb{x[n,\,c,\;p\sigma - \pi + r\delta,\;q\sigma - \pi + s\delta]}
$$

where an $x$ coordinate outside the image reads 0. The $\hla{w}$ factor is the filter tap, and the $\hlb{x}$ factor is the input pixel that the tap lands on. There is no minus sign on $r$ or $s$, which is the whole difference from textbook convolution.

The output height follows from where the last window fits. With a dilated filter spanning $\delta(R-1)+1$ input rows, PyTorch's formula is

$$
\boxed{P = \left\lfloor \frac{H + 2\pi - \delta(R-1) - 1}{\sigma} \right\rfloor + 1}
$$

and $Q$ is the same expression for the width. A 224-pixel input with a 7-wide filter, stride 2, and padding 3 gives $\lfloor (224 + 6 - 6 - 1)/2 \rfloor + 1 = 112$. The lab's first function, `conv_out_size`, is this formula.

> [!EXAMPLE] A 3 by 3 filter on a 5 by 5 image
> Take $H = W = 5$, $R = S = 3$, stride 1. With padding 0, $P = Q = 5 - 3 + 1 = 3$, so the output is $3 \times 3$ and each output reads 9 pixels. With padding 1, $P = Q = 5$, the output matches the input, and the corner output reads only 4 real pixels because 5 of its 9 taps land in the padding.

Count the work. One output element does $RS$ multiplies and $RS$ adds per input channel, so a single channel does $2k^2$ FLOP for a $k \times k$ filter. If the kernel reads each input element once and writes each output once, in fp32, it moves $4 + 4 = 8$ bytes per output. That is an upper bound on reuse, and the arithmetic intensity is

$$
I_{\text{ideal}} = \frac{2k^2}{8} = \frac{k^2}{4}\ \text{FLOP per byte.}
$$

A kernel with no reuse at all, where every thread reads its own $k^2$ inputs from global memory, moves $4k^2 + 4$ bytes, and its intensity is about 0.45 to 0.49 FLOP per byte for every filter from $3 \times 3$ to $9 \times 9$.

| Filter | FLOP per output | Ideal bytes | Ideal intensity | No-reuse intensity |
|---|---|---|---|---|
| $3 \times 3$ | 18 | 8 | 2.25 | 0.45 |
| $5 \times 5$ | 50 | 8 | 6.25 | 0.48 |
| $7 \times 7$ | 98 | 8 | 12.25 | 0.49 |
| $9 \times 9$ | 162 | 8 | 20.25 | 0.49 |

Compare these with the fp32 ridge, which is 10.1 against the rated bandwidth and 15.3 against the copy ceiling. A single-channel filter up to $5 \times 5$ is memory bound even with perfect reuse, so the best that tiling can do is move the kernel to the copy ceiling. A $7 \times 7$ filter crosses the rated ridge of 10.1 but not the 15.3 that the copy ceiling implies, and a $9 \times 9$ filter crosses both. Without reuse, every one of these kernels sits about 20 times below the ridge, and that gap is what tiling closes.

A real layer has channels. The ResNet-style layer in `scripts/figures/31-convolution-and-stencils.py` has $N = 32$, $C = K = 64$, a $56 \times 56$ image, a $3 \times 3$ filter, and padding 1, in fp32. It does $2 \cdot 32 \cdot 56^2 \cdot 64 \cdot 64 \cdot 9 = 7.40 \times 10^{9}$ FLOP and, with ideal reuse, moves $4 \times (32 \cdot 64 \cdot 56^2 + 64 \cdot 64 \cdot 9 + 32 \cdot 64 \cdot 56^2) = 51.5$ MB. The intensity is 143.6 FLOP per byte, far above the ridge. Channels are what make convolution layers compute bound: each loaded input pixel feeds $K$ filters, and each filter reads $C$ channels. The compute floor on the fp32 CUDA cores is $7.40 \times 10^9 / 19.5 \times 10^{12} = 0.379$ ms. This is a prediction from the ratings. The lab measures only the single-channel kernels.

> [!KEY] Single-channel convolution is a memory problem, and a multi-channel layer is a compute problem
> The first half of this chapter makes a $k \times k$ filter reach the memory ceiling, which is what a depthwise convolution, an audio front end, and a stencil need. The second half turns the many-channel case into a GEMM, which is what the tensor cores need.

## The filter belongs in constant memory

This section answers one question: where should the $k^2$ filter weights live?

Picture a warp of 32 threads computing 32 neighboring outputs. At filter tap $(a, b)$, all 32 threads multiply by the same weight $w[a][b]$ and by 32 different pixels. The weight is a broadcast, and the pixels are a coalesced row.

CUDA has a memory space for this pattern. *Constant memory* is a 64 KB region, read-only in device code, that the hardware caches. The CUDA C++ Best Practices Guide says a read can be "as fast as a register access" when all threads in a warp read the same location, and that accesses to different addresses by threads in a warp are serialized. A filter tap is the first case exactly.

You declare the region at file scope, and the host fills it:

```cuda
constexpr int MAXK = 15;
__constant__ float c_filter[MAXK * MAXK];   // 225 floats, 900 bytes

// host side, on the current stream:
cudaMemcpyToSymbolAsync(c_filter, w_device_ptr, K * K * sizeof(float), 0,
                        cudaMemcpyDeviceToDevice, stream);
```

A $15 \times 15$ filter is 900 bytes, about 1.4% of the 64 KB, so the whole region is almost never the limit. A convolution layer with many channels breaks that: $K \times C \times R \times S$ weights for a layer with 64 filters over 64 channels is $64 \cdot 64 \cdot 9 \cdot 4 = 147{,}456$ bytes, which doesn't fit. That is one reason the many-channel case uses GEMM tiles that stage the weights in shared memory instead.

Constant memory also has a trap. If the lanes of a warp read different constant addresses, the reads serialize, so a kernel that indexes the filter by `threadIdx.x` loses the benefit. In the tiled kernel below, the loop variables $a$ and $b$ are the same in every thread, and only the data index depends on the thread.

## Tiling with halo cells

This section answers one question: how much extra input does a tile of outputs need, and who loads it?

Picture a $16 \times 16$ block of output pixels. Each one needs a $3 \times 3$ window, so the block as a whole needs the $16 \times 16$ inputs under it plus a one-pixel border on every side: an $18 \times 18$ patch. The border is the *halo*, and its cells belong to the neighboring tiles' interiors too, so every halo cell is loaded by more than one block.

For a tile of $T$ outputs per side, a filter of $k$ per side, and $d$ dimensions, the patch has $(T + k - 1)^d$ inputs and the tile produces $T^d$ outputs. The loads per output, the *halo overhead*, are

$$
\boxed{\text{loads per output} = \left(\frac{T + k - 1}{T}\right)^{d}}
$$

For $T = 16$, $k = 3$, $d = 2$ the patch is $18^2 = 324$ elements for $256$ outputs, which is $324 / 256 = 1.266$. For $k = 5$ it is $20^2 / 16^2 = 1.5625$. For $T = 8$ and $k = 7$ it is $(14/8)^2 = 3.06$, because the halo of a small tile dominates. In 3D, a $T = 8$ tile with a $3 \times 3 \times 3$ filter loads $(10/8)^3 = 1.95$ per output.

```viz
31-halo-overhead
```

There are two ways to organize the threads around that patch, and PMPP describes both.

**The output-tile strategy.** The block has $T \times T$ threads, one per output. The patch is bigger than the block, so each thread loads several inputs in a loop, and then every thread computes. Threads never idle during the compute phase. The load loop is a little more complex, and it loads with the coalescing pattern of your choice.

**The input-tile strategy.** The block has $B \times B$ threads, one per input in the patch, and each thread loads exactly one element. Only the interior $(B - k + 1) \times (B - k + 1)$ threads then compute, because only they have all their neighbors. The load is simple, and the compute phase wastes the rest. The fraction of threads that write is

$$
\text{active fraction} = \left(\frac{B - k + 1}{B}\right)^{d}.
$$

A $16 \times 16$ block with a $3 \times 3$ filter has $14^2 / 16^2 = 76.6\%$ of its threads active. With a $7 \times 7$ filter that falls to $10^2/16^2 = 39\%$, and a block of 8 can't hold a $9 \times 9$ filter at all: the fraction is 0. The output-tile strategy has no such cliff. The lab's `halo_overhead` and `active_fraction` are these two formulas.

> [!NOTE] Is the shared-memory tile needed?
> Neighboring blocks read the same halo cells, so the L1 and L2 caches may serve them without a tile. Reading the input directly through the cache is a valid baseline for small filters. The lab times the tiled kernel against `F.conv2d` and records both without grading them.

The following kernel is the output-tile strategy for one channel, with the filter in constant memory. It is the lab's solution, and the lab's starter leaves the body for you.

```cuda
constexpr int TILE = 16;
constexpr int MAXK = 15;
constexpr int IN_DIM = TILE + MAXK - 1;
__constant__ float c_filter[MAXK * MAXK];

__global__ void conv2d_tiled_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int H, int W, int K) {
    __shared__ float tile[IN_DIM][IN_DIM];
    const int r = K / 2;
    const int in_dim = TILE + K - 1;
    const float* img = in + (size_t)blockIdx.z * H * W;
    float* dst = out + (size_t)blockIdx.z * H * W;
    const int row0 = blockIdx.y * TILE - r;
    const int col0 = blockIdx.x * TILE - r;

    for (int i = threadIdx.y; i < in_dim; i += TILE)
        for (int j = threadIdx.x; j < in_dim; j += TILE) {
            const int gr = row0 + i, gc = col0 + j;
            tile[i][j] = (gr >= 0 && gr < H && gc >= 0 && gc < W) ? img[(size_t)gr * W + gc] : 0.0f;
        }
    __syncthreads();

    const int orow = blockIdx.y * TILE + threadIdx.y;
    const int ocol = blockIdx.x * TILE + threadIdx.x;
    if (orow < H && ocol < W) {
        float acc = 0.0f;
        for (int a = 0; a < K; ++a)
            for (int b = 0; b < K; ++b)
                acc += c_filter[a * K + b] * tile[threadIdx.y + a][threadIdx.x + b];
        dst[(size_t)orow * W + ocol] = acc;
    }
}
```

Read three things in that kernel. The halo load strides by `TILE` in both directions, so one thread loads up to four elements for a $3 \times 3$ filter, and the first loop is the part the input-tile strategy doesn't need. The `__syncthreads()` is outside every `if`, which [chapter 8](/c/08-shared-memory-and-transpose) and [chapter 27](/c/27-debugging-gpu-kernels) explain: a barrier inside a divergent branch can hang. And the loop over `a` and `b` indexes `c_filter` with values that every thread shares, which keeps the constant reads broadcasts.

### The border is part of the halo

The ternary in the load is the zero padding. At the image border, the halo hangs off the image, and a halo cell outside $[0, H) \times [0, W)$ must read 0.

The failure mode is quiet in a high-level language. In Python and in torch, `x[-1]` doesn't raise an error. It wraps to the last row, so a Python model of the halo load that forgets the lower bound puts the bottom row of the image above the top row. The lab's `halo_tile` function returns the tile at any corner with zeros outside, and its check uses a tile at $(-1, -1)$ so a wrapped index fails. In CUDA the equivalent mistake is worse: a negative index reads whatever memory precedes the buffer. Both are the same bug, a halo load that ignores padding.

The tile's origin is a common off-by-one: for a filter of odd side $k$ and "same" padding, the tile starts $\lfloor k/2 \rfloor$ before the block's first output, which is `blockIdx.y * TILE - r` above.

## A 3D stencil: registers and coarsening

This section answers one question: how does a 3D stencil reuse its z neighbors without a third shared-memory dimension?

A *stencil* is a convolution with fixed, usually symmetric coefficients, applied at every cell of a grid. The 7-point stencil in three dimensions updates a cell from itself and its six face neighbors:

$$
\text{out}[z,y,x] = \hla{c_0}\, \text{in}[z,y,x] + \hlb{c_1}\left(\text{in}[z \pm 1,y,x] + \text{in}[z,y \pm 1,x] + \text{in}[z,y,x \pm 1]\right)
$$

where $\text{in}[z \pm 1, y, x]$ stands for the sum of the two neighbors. In this chapter's convention the boundary cells, on the faces of the grid, are copied unchanged. That is $1 + 5 + 1 + 1 = 8$ FLOP per interior cell: one multiply by $\hla{c_0}$, five adds for six neighbors, one multiply by $\hlb{c_1}$, and one add.

Count the bytes. Ideal reuse reads each cell once and writes it once, in fp32: 8 bytes for 8 FLOP, an intensity of 1.0. That is far below the ridge, so the kernel is memory bound and the copy sets its speed limit, as it did for the transpose in chapter 8. A $256^3$ grid has $16{,}777{,}216$ cells and moves $134{,}217{,}728$ bytes read plus written. At the 1275 GB/s copy ceiling the floor is

$$
\frac{134{,}217{,}728\ \text{B}}{1.275 \times 10^{12}\ \text{B/s}} = 0.1053\ \text{ms}.
$$

This is a prediction from the measured copy ceiling, and the lab measures the stencil. A kernel with one thread per cell and every neighbor read from global memory issues 7 reads per output. The caches absorb some of that, and how much is something to measure.

A shared-memory tile in 3D has a halo on six faces. A full 3D tile has a cost: a $T^3$ tile of fp32 needs $(T+2)^3 \times 4$ bytes of shared memory, which is 23,328 bytes for $T = 16$. A 7-point stencil never reads the corners of that box, so the box wastes space. The loads that the stencil needs are the tile plus six faces of $T^2$ each, which is $1 + 6/T$ per output: 1.75 at $T = 8$, 1.375 at $T = 16$, and 1.1875 at $T = 32$. The box formula $((T+2)/T)^3$ gives 1.95, 1.42, and 1.20.

*Register tiling along z* avoids the 3D tile. The block holds one x-y plane in shared memory. Each thread keeps its three z values in registers, called `prev`, `curr`, and `next`. To advance one plane, the thread loads the new `next`, stores `curr` into the shared plane, computes, and shifts the registers: `prev = curr`, `curr = next`. Each input element is loaded once per plane and used for three different outputs, as the center of one, the `next` of the one below, and the `prev` of the one above.

*Thread coarsening* is the other half. Instead of one thread per cell, one thread owns a column of `ZC` cells along z and walks it. The registers only work if one thread visits consecutive z planes, so coarsening is what makes the register tile possible. It also cuts the grid by a factor of `ZC`, and the block count must stay far above the SM count. With a $256^3$ grid, a $32 \times 8$ block, and `ZC` = 16, the grid is $9 \times 43 \times 16 = 6192$ blocks for 108 SMs.

```cuda
constexpr int TX = 32, TY = 8, ZC = 16;

__global__ void stencil7_coarse(const float* __restrict__ in, float* __restrict__ out,
                                int X, int Y, int Z, float c0, float c1) {
    __shared__ float plane[TY][TX];
    const int tx = threadIdx.x, ty = threadIdx.y;
    // The block advances by TX-2 by TY-2 cells. The outer ring only loads.
    const int gx = blockIdx.x * (TX - 2) + tx - 1;
    const int gy = blockIdx.y * (TY - 2) + ty - 1;
    const bool inside = gx >= 0 && gx < X && gy >= 0 && gy < Y;
    const bool owner = inside && tx >= 1 && tx < TX - 1 && ty >= 1 && ty < TY - 1;
    const int z0 = blockIdx.z * ZC, z1 = min(z0 + ZC, Z);
    const size_t S = (size_t)X * Y;
    const size_t xy = inside ? (size_t)gy * X + gx : 0;

    float prev = 0.f, curr = 0.f, next = 0.f;
    if (inside) {
        if (z0 > 0) prev = in[(size_t)(z0 - 1) * S + xy];
        curr = in[(size_t)z0 * S + xy];
    }
    for (int z = z0; z < z1; ++z) {
        next = (inside && z + 1 < Z) ? in[(size_t)(z + 1) * S + xy] : 0.f;
        plane[ty][tx] = curr;
        __syncthreads();
        if (owner) {
            const bool edge = gx == 0 || gx == X - 1 || gy == 0 || gy == Y - 1 || z == 0 || z == Z - 1;
            out[(size_t)z * S + xy] = edge ? curr
                : c0 * curr + c1 * (prev + next + plane[ty][tx - 1] + plane[ty][tx + 1]
                                    + plane[ty - 1][tx] + plane[ty + 1][tx]);
        }
        __syncthreads();
        prev = curr; curr = next;
    }
}
```

This kernel uses the input-tile strategy in x and y: a $32 \times 8$ block loads 256 elements per plane, and the inner $30 \times 6 = 180$ threads write. The active fraction is $180/256 = 70.3\%$. Each plane therefore costs $256/180 = 1.42$ global reads per output. A block of `ZC` = 16 outputs per column loads 18 planes, the 16 it computes plus the plane below and the plane above, so the factor is $18/16 = 1.125$. The global reads per output come to $1.42 \times 1.125 = 1.6$, against 7 for the naive kernel. These are arithmetic on the kernel's shape, and they count requests the block issues, not the bytes that reach DRAM. The halo reads that overlap neighbors are the ones the L2 is likely to serve.

Two details carry correctness. The barrier appears twice, once after the plane is stored and once before it is overwritten, and both sit outside the `if (owner)`. The boundary cells are copied, and the copy happens inside the same branch as the update, so a cell on the grid face never reads a neighbor outside the grid.

> [!WARNING] A stencil built from torch.roll isn't a stencil with a boundary
> `x.roll(1, 0) + x.roll(-1, 0) + ...` is the shortest way to write the 7-point update in torch, and it wraps the grid around: the cell at $z = 0$ gets the cell at $z = Z - 1$ as a neighbor. The reference in the lab slices the interior and leaves the boundary unchanged. The check compares against a plain triple loop, so the wrapped version fails on the boundary cells.

## From convolution to a matrix: im2col

This section answers one question: what does it cost to turn a convolution into a matrix multiply?

Picture the input as a stack of windows. If you copy every window into a column of a new matrix, one column per output position, then the convolution is that matrix multiplied by the filters, flattened into rows. The GEMM of chapters 10, 11, and 14 does the rest.

The matrix is the *im2col* matrix, called `unfold` in PyTorch. For one image it has $CRS$ rows and $PQ$ columns. Row $c \cdot RS + r \cdot S + s$ holds filter tap $(r, s)$ of channel $c$, and column $p \cdot Q + q$ is output position $(p, q)$. The filter matrix has $K$ rows and $CRS$ columns, and the product has $K$ rows and $PQ$ columns, which is the output.

> [!EXAMPLE] A 4 by 4 image and a 2 by 2 filter
> Number the pixels 1 to 16 in row order, and take one channel and stride 1. There are $3 \times 3 = 9$ output positions, and each column holds a window of 4 pixels. The first column is $(1, 2, 5, 6)$ and the second is $(2, 3, 6, 7)$. The matrix has 36 entries for 16 pixels, a ratio of 2.25. Pixel 6 appears in four columns.

The ratio is the cost. Each output position copies $CRS$ values, so the matrix has $N \cdot C \cdot R S \cdot P Q$ entries against $N \cdot C \cdot H W$ pixels:

$$
\boxed{\frac{\text{im2col elements}}{\text{input elements}} = \frac{RS\,PQ}{HW}}
$$

At stride 1 with "same" padding, $PQ = HW$ and the ratio is $RS$: 9 for a $3 \times 3$ filter, 25 for $5 \times 5$, and 49 for $7 \times 7$. A $7 \times 7$ filter at stride 2 and padding 3 on a 224 input gives $PQ = 112^2$, and the ratio is $49 \cdot 112^2 / 224^2 = 12.25$. Stride is the only thing that shrinks it, by about $\sigma^2$.

```viz
31-im2col-blowup
```

For the layer from earlier ($N = 32$, $C = 64$, $56 \times 56$, $3 \times 3$), the input is 25,690,112 bytes and the im2col matrix is $9 \times$ that, 231,211,008 bytes. A kernel writes the matrix to HBM, and the GEMM reads it back: 462,422,016 bytes of traffic. At the 1275 GB/s copy ceiling that traffic alone takes 0.363 ms. The whole layer's compute floor on the fp32 CUDA cores is 0.379 ms, so the lowering costs as much time as the arithmetic. A GEMM on tensor cores in tf32 or bf16 has a much lower compute floor, and then the lowering dominates. These are predictions from the ratings and the copy ceiling.

There is a second cost: memory. The matrix for one layer of one batch is 231 MB, and a deep network holds one per layer during training if the backward pass keeps it. A network that keeps one such matrix per layer for the backward pass multiplies that figure by its depth.

The lab's `im2col` builds the matrix from slices of the padded input, one per filter tap, and checks it against `F.unfold` entry by entry. Two conventions matter, and both must match PyTorch.

- **Row order.** `F.unfold` orders rows channel first: $c \cdot RS + r \cdot S + s$. An im2col that orders taps first and channels last is a different matrix. It is the right matrix for an NHWC filter, which is why this ordering returns below.
- **No flip.** The GEMM is `w.reshape(K, C*R*S) @ cols`, with the filter in its stored order. If you flip the filter first, as textbook convolution does, the result differs from `F.conv2d`. The lab's check uses a filter that has a single nonzero tap in the top-left corner, and a flipped filter reads the wrong pixel.

> [!KEY] PyTorch convolution is cross-correlation
> `F.conv2d` doesn't flip the filter. A kernel written from a textbook definition of convolution, with $x[p - r]$ in the sum, matches only on symmetric filters, so a test with a random filter catches it and a test with a Gaussian blur doesn't.

## Implicit GEMM

This section answers one question: how do you keep the GEMM and drop the matrix?

Picture a GEMM kernel from chapter 10. It loads an $A$ tile, a block of $BM$ rows by $BK$ columns, from memory into shared memory. If $A$ is the im2col matrix, the load reads entries of a matrix that doesn't exist. But every entry of that matrix is some pixel of the input, and its position follows from the row and column. So the load can compute that position and read the pixel directly. That is *implicit GEMM*: the GEMM is the same, and the $A$ tile is gathered from the input tensor on the way into shared memory.

cuDNN's performance guide describes the main tensor-core method this way. The matrices are never created in memory, and arithmetic intensity is computed from the original tensor sizes. The guide's forward-pass GEMM dimensions are $M = N P Q$, $N = K$, and $K = C R S$, where the convolution's $N$ and $K$ are not the GEMM's. This chapter writes the GEMM dimensions as $M$, $K_{\text{out}}$, and $K_g$ for the reduction, to keep the two apart:

| GEMM dimension | Size | Meaning |
|---|---|---|
| $M$ | $N P Q$ | One row per output position of every image |
| $K_{\text{out}}$ | $K$ | One column per output channel |
| $K_g$ | $C R S$ | The reduction over input channels and filter taps |

The index map is a pair of divisions. Row $m$ splits into the image, output row, and output column, and reduction index $k$ splits into the channel and filter tap:

$$
n = \lfloor m / PQ \rfloor,\quad p = \lfloor (m \bmod PQ)/Q \rfloor,\quad q = m \bmod Q
$$

$$
c = \lfloor k / RS \rfloor,\quad r = \lfloor (k \bmod RS)/S \rfloor,\quad s = k \bmod S
$$

The input coordinate then follows from the convolution definition: $h = p\sigma - \pi + r\delta$ and $w = q\sigma - \pi + s\delta$. If $h$ or $w$ falls outside the image, the entry is a padding zero, and the load must produce 0 and not read memory.

> [!EXAMPLE] One entry, with real numbers
> Take $N = 1$, $C = 2$, a $5 \times 5$ image, a $3 \times 3$ filter, stride 1, padding 1. Then $P = Q = 5$, $M = 25$, and $K_g = 18$. Entry $(m, k) = (7, 11)$: $m = 7$ gives $n = 0$, $p = 1$, $q = 2$. $k = 11$ gives $c = 1$, and $11 \bmod 9 = 2$ gives $r = 0$, $s = 2$. The input row is $h = 1 - 1 + 0 = 0$ and the column is $w = 2 - 1 + 2 = 3$, so the entry is $x[0, 1, 0, 3]$. Entry $(0, 0)$ has $p = q = 0$ and $r = s = 0$, so $h = w = -1$, which is a padding zero.

The lab's `implicit_gemm_coords` implements the map and returns `None` for a padding entry. Its check walks every $(m, k)$ of seven odd shapes and compares the value it finds, or 0, against the corresponding entry of `F.unfold`. The padding is the part that goes wrong, for the same reason as in the halo load: a negative $h$ indexes from the other end, unless you mask it.

The next function in the lab, `implicit_gemm_tile`, builds a $BM \times BK$ tile of the matrix straight from the input, with clamped coordinates and a mask. `conv2d_implicit` loops over tiles and accumulates `a @ w[:, k0:k1].T`. No full im2col matrix exists, and a tile size of 7 by 5, which divides neither $M$ nor $K_g$ in most test shapes, checks that the edges are masked.

Here is the same idea as a Triton kernel, which is the lab's optional third part. The gather is in the `tl.load` of `a`, and the dot is unchanged from chapter 14:

```python
# Inside the K loop of a Triton matmul: decode, mask, gather, then tl.dot as in chapter 14.
c = offs_k // (R * S); rs = offs_k % (R * S); r = rs // S; s = rs % S
h = p[:, None] * stride - pad + r[None, :]
wd = q[:, None] * stride - pad + s[None, :]
mask = (offs_m[:, None] < M) & (offs_k[None, :] < Kg) & (h >= 0) & (h < H) & (wd >= 0) & (wd < W)
a = tl.load(x_ptr + ((n[:, None] * C + c[None, :]) * H + h) * W + wd, mask=mask, other=0.0)
```

The lab's `solution.py` holds the full kernel. What implicit GEMM buys is bytes: it reads the input about once per tile, not $RS$ times from a lowered copy, and it writes nothing but the output. What it costs is integer arithmetic and predicates in the load path. Every element of the $A$ tile costs a division by $PQ$, by $Q$, by $RS$, and by $S$, plus four comparisons. Production kernels hoist that work: for a fixed $m$, $(n, p, q)$ doesn't change across the $K_g$ loop, and for a fixed $k$ the filter offset doesn't change across $m$ tiles. CUTLASS's implicit-GEMM iterators compute the decomposition once, precompute pointer deltas for the next $s$, the next $r$, and the next channel block, and advance by table lookup. That is an optimization of the index map, and the map itself is what you write here.

A note on the flip. CUTLASS's implicit-GEMM documentation writes its reference loop with the filter index as $R - r - 1$, which is the flipped, textbook form, and CUTLASS offers both conventions. The map in this chapter uses $r\delta$ with no flip, which matches PyTorch.

## NCHW and NHWC

This section answers one question: in which order should the four tensor axes sit in memory?

Picture the order in which a warp's 32 lanes read. In the direct kernels of this chapter the lanes span $w$, adjacent pixels in a row, and in NCHW the pixels of one channel's row are contiguous, so the reads coalesce. In NHWC the channels at one pixel are contiguous, and adjacent pixels are $C$ elements apart.

For a GEMM the answer changes. The reduction index decides it. With the NCHW filter $(K, C, R, S)$, $k = (c \cdot R + r) \cdot S + s$, so consecutive $k$ walk across $s$ and then $r$: they land on neighboring pixels of one channel, which in an NCHW input are contiguous only for $S$ elements before the stride jumps to the next row. With the NHWC filter $(K, R, S, C)$, $k = (r \cdot S + s) \cdot C + c$, so consecutive $k$ walk across channels at one pixel, which are contiguous in an NHWC input for $C$ elements. A tensor-core tile with $BK$ of 16 or 32 reads 16 or 32 contiguous values, and vector loads work.

The lab's `implicit_gemm_coords` takes a `layout` argument for this reason. The same $(m, k)$ names a different input element in the two layouts, and the check compares both against `F.unfold`, with the rows reordered for NHWC. In the worked example above, $(m, k) = (7, 11)$ is $x[0, 1, 0, 3]$ in NCHW and $x[0, 1, 1, 3]$ in NHWC, because the $k = 11$ splits as $(r = 1, s = 2, c = 1)$ in tap-major order.

The cuDNN performance guide recommends NHWC for tensor-core convolutions. NCHW still runs on tensor cores, but cuDNN then adds transposes, which cost more on larger tensors and smaller filters. The same guide asks for channel counts that are multiples of 8 for fp16 and of 4 for TF32 so that the loads align, and it says older cuDNN versions fall back to a non-tensor-core kernel without that alignment. In PyTorch, `memory_format=torch.channels_last` stores NCHW-indexed tensors in NHWC order. The first convolution layer of an image model has $C = 3$, which fits neither multiple, and the guide suggests padding it to 4.

## Fusing convolution, ReLU, and maxpool

This section answers one question: how many bytes does a fused conv-ReLU-pool kernel avoid?

Picture the unfused pipeline as three kernels. The convolution writes its output tensor to HBM. The ReLU kernel reads it and writes another. The maxpool kernel reads that and writes a tensor a quarter the size. Each hand-off goes through HBM, and each intermediate is large.

Take the layer from earlier, with $N = 32$, $K = 64$ output channels, a $56 \times 56$ output, in fp32, and a $2 \times 2$ pool. The convolution output is $4 \cdot 32 \cdot 64 \cdot 56^2 = 25{,}690{,}112$ bytes, and the pooled tensor is a quarter of that, $6{,}422{,}528$ bytes. The unfused extra traffic is

$$
\underbrace{25.69}_{\text{conv write}} + \underbrace{2 \times 25.69}_{\text{ReLU read, write}} + \underbrace{25.69}_{\text{pool read}} + \underbrace{6.42}_{\text{pool write}} = 109.18\ \text{MB}.
$$

A fused kernel writes only the pooled tensor: 6.42 MB. At the 1275 GB/s copy ceiling the unfused path costs 0.0856 ms of extra traffic and the fused path costs 0.0050 ms. These are predictions from the copy ceiling, and the conv's own input and weight reads are the same in both cases.

Fusion is legal here because of two properties. A bias is the same across a pool window, since it depends only on the output channel. ReLU is monotonic, so $\max_i \text{ReLU}(z_i) = \text{ReLU}(\max_i z_i)$. Together they let you take the maximum of the raw convolution outputs over the window and then add the bias and apply ReLU once:

$$
\boxed{\text{maxpool}(\text{ReLU}(y + b)) = \text{ReLU}\left(\max_{\text{window}} y + b\right)}
$$

On the GPU, the fused kernel assigns one thread block an output tile that is a multiple of the pool size, so that every pool window lies in one block's tile. Each thread, or each GEMM fragment, computes the 4 conv outputs of a window and keeps the maximum in a register. The conv output tile never leaves the chip.

The lab's `conv_relu_maxpool` does this on the CPU. For each offset $(d_i, d_j)$ inside the pool window it takes the matching strided columns of the im2col matrix, runs one GEMM, and keeps a running max. It then adds the bias and applies ReLU once. The comparison is `F.max_pool2d(F.relu(F.conv2d(...)), pool)`. One detail catches implementations: when the conv output has an odd side, `max_pool2d` drops the partial window at the edge, and a fused kernel that keeps it returns a different shape.

How the framework gets this fusion is the topic of [chapter 17](/c/17-fusion-and-compilers). A compiler fuses the pointwise ReLU into its own kernels, but the convolution is a library call, so fusing it with its epilogue means a library or a template that exposes one, as the bias-plus-GELU epilogue of [chapter 24](/c/24-cute-dsl-gemm) does for a GEMM. Implicit GEMM shows where the fusion goes: the conv is a GEMM, and the pool window lives in the epilogue, which sees the accumulator before it is written.

## What goes wrong

- **A flipped filter.** A convolution written from the textbook definition matches `F.conv2d` only for a symmetric filter. The test with a one-tap filter catches `flipped_filter`. Remember that PyTorch computes cross-correlation.
- **A halo load that ignores the border.** The tile at the corner reads memory outside the image. In Python a negative index wraps to the far edge, and in CUDA it reads whatever precedes the buffer. The lab calls this `negative_index_wraps`. Mask both bounds.
- **The wrong row order in im2col.** Channel first matches `F.unfold` and an NCHW filter. Tap first is a valid layout for an NHWC filter and a wrong one otherwise. The lab calls it `tap_major_rows`.
- **A stencil that wraps.** `torch.roll` takes the opposite face as a neighbor. The lab calls this `roll_wraps_boundary`.
- **A reference in the wrong precision.** On the GPU, `torch.backends.cudnn.allow_tf32` defaults to `True`, so an fp32 `F.conv2d` on an A100 uses TF32 tensor cores and carries a TF32 error. Compare your fp32 kernel against a float64 reference, as the lab does, and not against the TF32 result.
- **A barrier inside a guard.** The halo load and the plane store have to be followed by `__syncthreads()` in code every thread reaches. An early `return` for out-of-range threads before the barrier is the classic hang, and [chapter 27](/c/27-debugging-gpu-kernels) shows how `compute-sanitizer` finds it.
- **Constant memory used per thread.** Indexing `c_filter[threadIdx.x]` makes each lane read a different address, and the reads serialize.
- **A fused pool with a ceiling.** Keeping the partial pool window at an odd edge returns a larger shape than `max_pool2d`, which drops it. The lab calls this `ceil_pool`.

> [!RECAP]
> - A $k \times k$ single-channel convolution has an ideal intensity of $k^2/4$ FLOP per byte, so filters up to $5 \times 5$ are memory bound against the fp32 ridge of 10.1. Many channels raise the intensity, to 143.6 for a ResNet-sized layer.
> - The filter goes in constant memory because a warp reads one tap at once. The input goes in shared memory with a halo, which costs $((T+k-1)/T)^d$ loads per output in the output-tile strategy and wastes $1 - ((B-k+1)/B)^d$ of the threads in the input-tile strategy. A halo cell outside the image reads 0.
> - A 3D 7-point stencil keeps the plane in shared memory and the z neighbors in registers, and one thread walks a column of `ZC` cells. The predicted global reads per output fall from 7 to 1.6, and the floor for a $256^3$ grid is 0.1053 ms at the copy ceiling.
> - im2col writes the input $RS \cdot PQ/HW$ times, 9 for a $3 \times 3$ filter at stride 1. Implicit GEMM computes $(n, c, h, w)$ from $(m, k)$ with divisions, and masks padding to 0.
> - NHWC makes the reduction index contiguous. PyTorch convolution is cross-correlation, so don't flip the filter. A fused conv, ReLU, and maxpool writes 6.42 MB where the unfused path moves 109.18 MB of intermediates on the example layer.

## Check your understanding

> [!QUESTION] What is the output size for a 224-pixel input, a $7 \times 7$ filter, stride 2, padding 3, and dilation 1? What changes at dilation 2?
> $\lfloor (224 + 6 - 6 - 1)/2 \rfloor + 1 = 112$. With dilation 2 the filter spans $2 \cdot 6 + 1 = 13$ pixels, so the numerator is $224 + 6 - 12 - 1 = 217$, and $\lfloor 217/2 \rfloor + 1 = 109$.

> [!QUESTION] A $16 \times 16$ output tile and a $5 \times 5$ filter. How many input elements does the block load, and how many per output? What fraction of threads write in the input-tile strategy with a $16 \times 16$ block?
> The patch is $(16 + 4)^2 = 400$ elements for 256 outputs, which is 1.5625 per output. With a $16 \times 16$ block of input-tile threads, $(16 - 4)^2 = 144$ of 256 write, which is 56.25%.

> [!QUESTION] Why does `x[row0 + i, col0 + j]` fail the border check in a tile loader, even though Python doesn't raise an error?
> A negative index wraps. At $(\text{row0}, \text{col0}) = (-1, -1)$ the corner cell reads the last row and last column of the image, which are real pixels, not zeros. A tile loader must test both bounds on each axis.

> [!QUESTION] Why does the coarsened stencil need two `__syncthreads()` per plane?
> The first makes the stores to the shared plane visible before any thread reads a neighbor from it. The second keeps a fast thread from overwriting `plane[ty][tx]` with the next plane's value while a slow thread is still reading the current one.

> [!QUESTION] A $5 \times 5$ filter, stride 2, padding 2, on a $64 \times 64$ input. What is the im2col blow-up?
> $P = Q = \lfloor (64 + 4 - 4 - 1)/2 \rfloor + 1 = 32$. The ratio is $RS \cdot PQ / HW = 25 \cdot 1024 / 4096 = 6.25$.

> [!QUESTION] Take an NCHW input with $C = 2$, a $5 \times 5$ image, a $3 \times 3$ filter, and padding 1. Which input element does GEMM entry $(m, k) = (0, 4)$ read?
> $m = 0$ gives $n = p = q = 0$. $k = 4$ gives $c = 0$ and $r = 1$, $s = 1$. The input row is $h = 0 - 1 + 1 = 0$ and the column is $w = 0$, so the entry is $x[0, 0, 0, 0]$, the center tap of the corner window. That tap is inside the image. The tap $(0, 0)$ of the same window, $k = 0$, is the padding zero.

> [!QUESTION] Why can you apply ReLU after a $2 \times 2$ max instead of before it?
> ReLU is monotonic, so the maximum of the ReLU values is the ReLU of the maximum. The bias is the same for every element of a window, so adding it after the max is the same as adding it before.

## Lab

> [!TRY] Lower a convolution three ways, then tile it on the GPU
> Open `31-conv-stencil`. Part 1 runs on any machine. The first function, `conv_out_size`, passes the first check. Then write `halo_overhead` and `active_fraction`, `halo_tile` with zeros at the border, `im2col` against `F.unfold`, `conv2d_im2col` against `F.conv2d` with no flip, the implicit-GEMM map and its tiled convolution, `stencil7_reference` against a plain loop, and the fused `conv_relu_maxpool`. Every check compares against a torch operation or a loop, on odd shapes with padding, stride, and dilation.
>
> Part 2 is CUDA. Write `conv2d_tiled_kernel`, with the filter in constant memory and a halo load that pads with zeros, and the two stencil kernels. The launchers are given. The tests compare the convolution against a float64 `F.conv2d` for filter sides 1 to 15, compare both stencil kernels against the reference, and require the coarsened stencil to reach 30% of the 1275 GB/s copy ceiling. Part 3, a Triton implicit-GEMM conv, is optional and records an error and never fails the lab.
>
> On a machine without a GPU, Part 1's eight checks run and Part 2 skips with a message. The first compile takes 30 to 60 seconds.

The lab reports these metrics:

- `conv_ms` and `torch_conv_ms`: the time of your tiled conv2d and of `F.conv2d` on eight $2048 \times 2048$ images with a $5 \times 5$ filter. They are recorded, not graded.
- `stencil_naive_gbs` and `stencil_coarse_gbs`: bytes read plus written divided by time for a $256^3$ grid, with the L2 flushed. `stencil_fraction_of_ceiling` is the coarsened rate over 1275 GB/s.
- `triton_max_err`: the optional kernel's error against float64. It uses TF32 by default, so expect an error near $10^{-2}$.

## Further reading

- Wen-mei Hwu, David Kirk, and Izzat El Hajj, *Programming Massively Parallel Processors*, 4th edition, Morgan Kaufmann, 2022. Chapter 7 is convolution, with constant memory and the input-tile and output-tile strategies. Chapter 8 is stencils, with register tiling and thread coarsening. Chapter 16 is the deep learning chapter, with convolution as a GEMM. The chapter numbers follow the research plan for this course and were not re-checked against the book here.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), NVIDIA. The constant memory section: 64 KB, cached, as fast as a register when a warp reads one address, serialized when the addresses differ.
- [Convolutional layers user's guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html), NVIDIA Deep Learning Performance. Implicit GEMM, the forward-pass GEMM dimensions, NHWC against NCHW, and the channel alignment for tensor cores.
- [Implicit GEMM convolution](https://github.com/NVIDIA/cutlass/blob/v3.9.2/media/docs/cpp/implicit_gemm_convolution.md), NVIDIA CUTLASS 3.9.2. The mapping of $M$, $N$, $K$ to the convolution, the activation tile iterators, and the precomputed pointer deltas.
- [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html), PyTorch. The output-size formula with dilation, and the statement that the operator is cross-correlation.
- Sharan Chetlur and others, [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/abs/1410.0759), 2014. The library paper behind the convolution algorithms that PyTorch calls.
- [Stanford CS149, Parallel Computing, Fall 2024](https://gfxcourses.stanford.edu/cs149/fall24), the lecture titled "Efficiently Evaluating DNNs (Software Solutions)". The schedule page lists it as covering how convolutions map to matrix multiplication and layer fusion. The page doesn't number its lectures, so this chapter cites the title only.
- The ml course, [Convolution and the receptive field](https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field) and [im2col and convolution backward](https://ml.ifkash.dev/lessons/039-im2col-and-convolution-backward), for the layer arithmetic and the backward pass that this chapter leaves out.
- [Chapter 8](/c/08-shared-memory-and-transpose), for the tile and the barrier, [chapter 10](/c/10-gemm-from-scratch), for the GEMM the lowering feeds, and [chapter 14](/c/14-triton), for the Triton `tl.dot` loop that the optional kernel uses.
