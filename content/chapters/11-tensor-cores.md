---
title: Tensor cores
slug: 11-tensor-cores
part: "Part 3 — Matrix multiplication"
summary: What a tensor core computes per instruction, how WMMA and mma.sync with ldmatrix feed it, and why the A100's 312 TFLOP/s needs a carefully fed warp tile.
minutes: 120
gpu: true
objectives:
  - "Say what one tensor-core instruction computes, and why its inputs are 16-bit and its accumulator is fp32."
  - "Derive the 16x gap between the A100's CUDA-core fp32 peak (19.5 TFLOP/s) and its bf16 tensor peak (312 TFLOP/s) from clocks and per-SM rates."
  - "Write a GEMM with the WMMA API: fragments, load_matrix_sync, mma_sync, store_matrix_sync."
  - "State which lane holds which element of an mma.sync.m16n8k16 fragment, and derive it from the lane number."
  - "Use ldmatrix (x1, x2, x4, .trans) to load fragments from shared memory, and explain why it exists."
  - "Compute the shared-memory traffic per tensor-core instruction for a warp tile, and tell when shared memory, not the tensor core, is the bottleneck."
  - "Count the registers a warp tile's accumulators and fragments need."
  - "Say how Hopper's wgmma and Blackwell's tcgen05 change the model."
lab: 11-wmma-gemm
---

# Tensor cores

> [!TLDR]
> - A tensor core computes a whole small matrix product, $D = A \cdot B + C$, in
>   one instruction. On the A100 the inputs are 16-bit (bf16 or fp16) and the
>   accumulator is fp32.
> - That is why the A100 is rated at 312 TFLOP/s on tensor cores and 19.5 on the
>   CUDA cores: a 16x gap, from the per-SM rate alone.
> - WMMA is the portable C++ view: opaque fragments, `load_matrix_sync`,
>   `mma_sync`. `mma.sync.m16n8k16` in PTX is the real instruction, and it
>   fixes exactly which lane holds which element.
> - `ldmatrix` loads those fragments from shared memory, one instruction for
>   four 8x8 tiles, and `.trans` fixes the layout of the B operand.
> - The tensor core is fast enough that shared memory, registers, and latency
>   become the limits. A 32 x 32 warp tile saturates shared memory before the
>   tensor cores; a 64 x 32 tile doesn't.

Almost every FLOP in a transformer is a matrix multiplication, and on a modern
NVIDIA GPU almost every matrix multiplication runs on tensor cores. When you
call `torch.matmul` on a bf16 tensor on an A100, the work lands on units that
the CUDA cores, the ones chapter [10](/c/10-gemm-from-scratch) tuned, never see.
Skip them and you leave 16x of the card's arithmetic on the table.

This chapter shows what those units compute and how you feed them. You first
meet the portable WMMA API, which hides the data layout. Then you drop to the
PTX instruction `mma.sync` and to `ldmatrix`, where the layout is explicit, so
you can see exactly which thread holds which number. The chapter ends at the
boundary of what a single, unpipelined warp can do. [Chapter 12](/c/12-pipelining-and-async-copies)
closes the gap to peak.

## Before you start

**Warp.** Thirty-two threads that execute one instruction together. A tensor
core instruction is *warp-wide*: all 32 lanes cooperate, and each lane holds
a slice of the operands. See [the execution model](/c/02-the-execution-model).

**Register tile.** The SGEMM in [chapter 10](/c/10-gemm-from-scratch) keeps a
small block of C in each thread's registers and updates it from shared-memory
tiles. The tensor-core GEMM keeps the same structure: block tile, warp tile,
K loop. Only the innermost update changes.

**bfloat16 and fp16.** Both are 16-bit. fp16 has 10 mantissa bits and a
maximum of 65504. bf16 has 7 mantissa bits and the exponent range of fp32, with
a machine epsilon of $2^{-7} = 7.8 \times 10^{-3}$. See [floating point on the GPU](/c/09-floating-point-on-the-gpu).

**Bank conflicts.** Shared memory has 32 banks of 4 bytes, so one wavefront
serves 128 bytes, and two threads that hit the same bank at different addresses
serialize. See [shared memory and the transpose](/c/08-shared-memory-and-transpose).

## What one tensor core computes

A tensor core is a fixed-function unit that multiplies two small matrices and adds a third, in one step. The section answers what the unit does per instruction, and why the precision looks the way it does.

The operation is a matrix multiply-accumulate on a small tile:

$$
\boxed{\hla{D} = \hlb{A} \cdot \hlc{B} + C}
\qquad \hlb{A} \in \mathbb{R}^{m \times k},\ \hlc{B} \in \mathbb{R}^{k \times n},\ C, \hla{D} \in \mathbb{R}^{m \times n}
$$

The tile sizes $m$, $n$, and $k$ are fixed by the instruction. One execution
does $m \cdot n \cdot k$ fused multiply-adds (FMAs), which is $2mnk$ FLOP. A big
GEMM is a loop that chains thousands of these, with $D$ of one step feeding $C$
of the next.

On the A100 (third-generation tensor cores), the instruction you use for
bf16 and fp16 is `mma.sync.m16n8k16`. It multiplies a 16 x 16 tile of A by a
16 x 8 tile of B and adds a 16 x 8 tile of C:

$$
16 \times 8 \times 16 = 2048\ \text{FMA} = 4096\ \text{FLOP per instruction.}
$$

For comparison, the SGEMM micro-kernel in chapter 10 does one FMA per thread per
instruction, so 32 FMA per warp instruction. One `mma.sync` does 64 times that.

### Mixed precision

The inputs are 16-bit and the accumulator is 32-bit. That choice is the whole
design, and it has a numerical reason and a hardware reason.

The numerical reason: a product of two bf16 numbers has an 8-bit by 8-bit
significand, so 16 significant bits. That fits exactly in fp32's 24-bit
significand, so **every individual product is exact**. All the rounding happens
in the additions. If you accumulated in bf16, with its 7 mantissa bits, a long
sum would stall, as [chapter 9](/c/09-floating-point-on-the-gpu) showed: once
the running sum is 256 times the next term, the term is rounded away. In fp32
the same stall comes about 65,536 times later.

The hardware reason: a 16-bit multiplier is far smaller than a 32-bit one, so
you fit many more of them in the same silicon. That is where the speedup comes
from.

> [!NOTE] The order of accumulation is unspecified
> The PTX ISA says only that accumulation is at least single precision, and
> doesn't fix the order or rounding of the adds inside one instruction. The
> paper [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/abs/2206.02874)
> studies this behavior. The practical consequence is in the lab: your tensor-core
> result won't match an fp32 reference bit for bit, so the tests compare with a
> small relative tolerance.

### What the A100 supports

The A100 (sm_80) tensor cores accept the following input types. The first row is
this chapter's subject.

| Input type | Accumulator | `mma.sync` shape (sm_80) | A100 dense peak |
|---|---|---|---|
| bf16 or fp16 | fp32 (fp16 also allows fp16) | m16n8k16 | 312 TFLOP/s |
| tf32 | fp32 | m16n8k8 | 156 TFLOP/s |
| int8 | int32 | m16n8k32 | 624 TOPS |
| fp64 | fp64 | m8n8k4 | 19.5 TFLOP/s |

There is no fp8 on the A100: fp8 tensor math starts at compute capability 8.9
for `mma.sync` and 9.0 for `wgmma`. Every figure here is from NVIDIA's
[A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
and PTX ISA, and is collected in the course's fact sheet.

## Where the 16x comes from

The A100's 312 TFLOP/s for bf16 and its 19.5 for fp32 are not marketing: both
follow from the per-SM rates and the clock. This section rebuilds both numbers,
because the gap between them is why this chapter exists.

Each SM has four tensor cores, one per sub-core, and the GA100 whitepaper says
each one executes 256 fp16 FMAs per clock. The SM also has 64 fp32 CUDA cores
that do one FMA per clock. Multiply by 2 FLOP per FMA, by 108 SMs, and by the 1410 MHz boost clock:

| Unit | FMA per clock per SM | FLOP per clock per SM | Arithmetic | Peak |
|---|---|---|---|---|
| fp32 CUDA cores | 64 | 128 | $108 \times 128 \times 1.41 \times 10^{9}$ | 19.5 TFLOP/s |
| bf16 or fp16 tensor cores | $4 \times 256 = 1024$ | 2048 | $108 \times 2048 \times 1.41 \times 10^{9}$ | 311.9 TFLOP/s |

The ratio is $2048 / 128 = 16$. Each SM does 16 times more bf16 tensor-core math per clock than fp32 CUDA-core math, and the clock is the same.

> [!KEY] The tensor core raises the ridge point 16-fold
> The CUDA cores' ridge point against 1935 GB/s is $19.5 / 1.935 = 10.1$ FLOP
> per byte. The bf16 tensor cores' ridge point is $312 / 1.935 = 161.2$. A
> kernel that was compute-bound on the CUDA cores can become memory-bound the
> moment it moves to tensor cores, unless you reuse every byte 16 times more.

That is a statement about reuse, and it sets the design of everything below.
Take a $4096^3$ bf16 GEMM that writes an fp32 result, as the lab does. It does
$2 \times 4096^3 = 137.4$ GFLOP. It must read A and B, each $4096^2 \times 2$ B
$= 33.6$ MB, and write C, $4096^2 \times 4$ B $= 67.1$ MB, for a minimum of
134.2 MB. Its intensity is

$$
\frac{137.4 \times 10^{9}\ \text{FLOP}}{134.2 \times 10^{6}\ \text{B}} = 1024\ \text{FLOP/B},
$$

which is 6.4 times the ridge point of 161. So the kernel is compute-bound *if*
you reach that minimum traffic, and the time floors are:

| Executed on | Floor for $4096^3$ | Arithmetic |
|---|---|---|
| bf16 tensor cores, 312 TFLOP/s | 0.44 ms | $137.4 \times 10^{9} / 312 \times 10^{12}$ |
| fp32 CUDA cores, 19.5 TFLOP/s | 7.05 ms | $137.4 \times 10^{9} / 19.5 \times 10^{12}$ |
| Moving 134.2 MB at the 1275 GB/s copy ceiling | 0.105 ms | $134.2 \times 10^{6} / 1.275 \times 10^{12}$ |

Memory is not the bottleneck at the minimum traffic: 0.105 ms is a quarter of the
tensor floor. But you only get minimum traffic by tiling, and your kernel's
tiles must supply 312 TFLOP/s of operands without stalling. The rest of the
chapter is about that.

> [!NOTE] Measured peak of the instruction
> Luo and others measured a loop of `mma.sync.m16n8k16` on an A100 PCIe (40 GB):
> 310.6 TFLOP/s with fp16 accumulate and 303.4 with fp32 accumulate, which is 97%
> of the 312 rating ([Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499)).
> They also measured a latency of 24.6 to 26.3 cycles per instruction. The
> rating is nearly reachable by the instruction alone. Real GEMM kernels reach
> less, and the Triton matmul tutorial quotes 220 to 245 TFLOP/s on an A100.

## The WMMA API

WMMA, the warp matrix multiply-accumulate API in `<mma.h>`, is the highest-level way to use a tensor core from CUDA C++. The section answers what the API looks like and what it hides.

The API has four pieces. A **fragment** is an opaque object, spread across the
registers of a warp, that holds a tile. `load_matrix_sync` fills a fragment from
memory. `mma_sync` multiplies two fragments into an accumulator fragment.
`store_matrix_sync` writes an accumulator to memory. All four are *warp-wide*:
every lane of the warp must call them together, with the same arguments.

```cpp
#include <mma.h>
using namespace nvcuda;

// One warp computes a 16x16 tile of C = A * B, with K a multiple of 16.
// A is row-major M x K bf16, B is row-major K x N bf16, C is row-major fp32.
__global__ void wmma_tile(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          float* C, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + k, K);           // leading dimension K
        wmma::load_matrix_sync(b, B + k * N, N);       // leading dimension N
        wmma::mma_sync(acc, a, b, acc);                // acc = a * b + acc
    }
    wmma::store_matrix_sync(C, acc, N, wmma::mem_row_major);
}
```

This kernel is a complete, correct 16 x 16 x K product for one warp. Launch one
warp per 16 x 16 tile of C and you have a GEMM.

### Shapes and rules on sm_80

The tile shape is a template parameter, and only a few combinations exist.

| Input | Accumulator | Shapes (m x n x k) | Min arch |
|---|---|---|---|
| fp16 | fp16 or fp32 | 16x16x16, 32x8x16, 8x32x16 | sm_70 |
| bf16 | fp32 | 16x16x16, 32x8x16, 8x32x16 | sm_80 |
| tf32 | fp32 | 16x16x8 | sm_80 |
| fp64 | fp64 | 8x8x4 | sm_80 |

Two alignment rules cause most first-time failures. The pointer you pass to
`load_matrix_sync` and `store_matrix_sync` must be 256-bit (32-byte) aligned,
and the leading dimension must be a multiple of 8 elements for 16-bit types and
4 for fp32. A tf32 input also needs an explicit conversion, `__float_to_tf32`.

### What WMMA hides, and what it costs

WMMA does not tell you which lane holds which element. The fragment's layout is
"unspecified" by design, so the same source can compile to different PTX on
different architectures. You can't index into a fragment by (row, column). You
can read its elements with `frag.x[i]` and `frag.num_elements`, but the mapping
from `i` to a matrix position is not documented and can change.

That has three costs, and each is the reason to leave WMMA when you want
performance.

1. **No epilogue fusion on registers.** To apply a bias, an activation, or a
   scale to the output, you need to know which output element each register
   holds. With WMMA you must store the accumulator to shared memory first, and then re-read it.
2. **No control over how fragments load.** `load_matrix_sync` from shared memory
   compiles to loads whose pattern you can't tune, and the shared-memory layout
   must satisfy its 32-byte alignment, which forces padding.
3. **No shape choice.** You can't use `m16n8k16`, the instruction the hardware
   executes natively. A 16 x 16 x 16 WMMA op typically compiles to two `m16n8k16`
   instructions, one for each 8-column half of B.

> [!TIP] Start with WMMA
> Write the WMMA version first, even if you plan to end at PTX. It is a
> correct baseline in about forty lines, and it is the reference you compare
> your `mma.sync` kernel against. The lab does exactly this.

## What `mma.sync` holds, lane by lane

The PTX instruction `mma.sync.aligned.m16n8k16` is what the tensor core runs, and unlike WMMA it fixes which of the 32 lanes holds which element. The section answers where every number lives.

The full instruction for bf16 inputs and an fp32 accumulator is:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {d0, d1, d2, d3},          // D: 4 fp32 registers per lane
    {a0, a1, a2, a3},          // A: 4 32-bit registers per lane (8 bf16)
    {b0, b1},                  // B: 2 32-bit registers per lane (4 bf16)
    {c0, c1, c2, c3};          // C: 4 fp32 registers per lane
```

The name reads left to right: `.sync.aligned` means all 32 lanes execute it
together and converge first. `m16n8k16` is the tile shape. `.row.col` says A is
row-major and B is column-major, the only layouts allowed for 16-bit types.
The remaining four types are D, A, B, and C, in that order. The tile has
$16 \times 16 = 256$ elements of A. Over 32 lanes that is 8 elements, or four
32-bit registers, per lane. B has $16 \times 8 = 128$ elements, so 4 per lane,
two registers. C and D have 128 each, and in fp32 that is 4 registers per lane.

### The layout rule

Two quantities, computed from the lane number, name everything. The *group ID*
$g$ and the *thread ID in group* $t$ are:

$$
\hla{g} = \left\lfloor \frac{\text{lane}}{4} \right\rfloor, \qquad \hlb{t} = \text{lane} \bmod 4.
$$

Lanes 0 to 3 are group 0, lanes 4 to 7 are group 1, and so on up to group 7.
Within a group, $t$ runs 0 to 3. Every element position is a function of $g$,
$t$, and the register element index $i$.

**Operand A (16 x 16, row-major).** Register element $a_i$, for $i = 0$ to 7,
holds the element at

$$
\text{row} =
\begin{cases} \hla{g} & i \in \{0, 1, 4, 5\} \\ \hla{g} + 8 & i \in \{2, 3, 6, 7\} \end{cases}
\qquad
\text{col} = 2\hlb{t} + (i \bmod 2) + \begin{cases} 0 & i < 4 \\ 8 & i \ge 4. \end{cases}
$$

Read that in words. Each 32-bit register `a0..a3` holds two adjacent columns of
one row: `a0` (the pair $a_0, a_1$) is row $g$, columns $2t$ and $2t+1$;
`a1` (the pair $a_2, a_3$) is row $g + 8$, same columns; `a2` is row $g$,
columns $8 + 2t$ and $9 + 2t$; and `a3` is row $g + 8$, those columns. The tile
is four 8 x 8 blocks, and the registers walk them in the order top-left,
bottom-left, top-right, bottom-right.

**Operand B (16 x 8, column-major).** Element $b_i$, for $i = 0$ to 3, holds

$$
\text{row}\ (k) = 2\hlb{t} + (i \bmod 2) + \begin{cases} 0 & i < 2 \\ 8 & i \ge 2 \end{cases}
\qquad
\text{col}\ (n) = \hla{g}.
$$

**Operands C and D (16 x 8, fp32).** Element $c_i$, for $i = 0$ to 3, holds

$$
\text{row} = \begin{cases} \hla{g} & i < 2 \\ \hla{g} + 8 & i \ge 2 \end{cases}
\qquad
\text{col} = 2\hlb{t} + (i \bmod 2).
$$

The following figure lets you pick a lane and see its elements in each operand.
Start with lane 5 on operand A: its group is $g = 1$, $t = 1$, so it holds
columns 2 and 3 of rows 1 and 9, and columns 10 and 11 of the same two rows.

```viz
11-fragment-map
```

### The 8 x 8 building block

Every fragment is made of 8 x 8 blocks, and one block has a single, memorable
shape. Eight rows by eight columns is 64 elements across 32 lanes, so each lane
holds two adjacent elements of one row, packed in one 32-bit register. Lane
numbers fill the block like this:

```text
        col  0-1   2-3   4-5   6-7
row 0:     L0    L1    L2    L3
row 1:     L4    L5    L6    L7
row 2:     L8    L9   L10   L11
  ...
row 7:    L28   L29   L30   L31
```

Row $g$ belongs to lanes $4g$ to $4g + 3$, and lane $4g + t$ holds columns $2t$
and $2t + 1$. A fragment of A is four such blocks. A fragment of B is two,
stacked along K. A C fragment is two, stacked along M, in fp32, where each lane
holds two 32-bit values per block instead of one packed pair.

> [!INTUITION] Why this layout
> The layout makes each lane's loads and stores *contiguous in K for A and B,
> and in N for C*. A lane's two elements of a register are adjacent in memory in the
> natural layouts, so one 32-bit load or store moves both. It also means that a
> group of four lanes together covers 8 contiguous elements of a row, which is
> 16 bytes.

### How the instruction is used

Chaining instructions in a loop is a matter of the accumulator staying in place.
The `d` registers of one call are the `c` registers of the next, so the compiler
reuses the same four registers, and the kernel never moves the accumulator
through shared memory. The inline-PTX wrapper you write in the lab looks like this:

```cpp
// D = A * B + D for one 16x8x16 tile, bf16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
```

The `"+f"` constraints say the four floats are both read and written, which is
why `%0` to `%3` appear as both D and C. The `"r"` registers hold packed bf16
pairs; the instruction never sees them as numbers you can index.

## `ldmatrix`: loading fragments from shared memory

You now know where each element must end up in registers. The section answers how to get a 16 x 16 tile from shared memory into those registers quickly.

The direct way is for each lane to compute its own addresses from the formulas
above and issue 32-bit loads. For A that is four loads per lane, one for each of
`a0` to `a3`. For B stored as `[k][n]` it is worse: lane $(g, t)$ needs
elements $(2t, g)$ and $(2t + 1, g)$, which are in different rows, so each lane
needs two 16-bit loads per register plus an instruction to pack them. Both the
address arithmetic and the instruction count are overhead that competes with the
tensor core for issue slots.

`ldmatrix`, introduced on Turing (sm_75), does the whole job in one instruction.
It loads one, two, or four 8 x 8 matrices of 16-bit elements from shared memory
and delivers each in exactly the fragment layout above.

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0, r1, r2, r3}, [addr];
```

### How the addresses work

The unusual part is who supplies the addresses. Each 8 x 8 matrix has 8 rows,
each row is 8 x 16 bits = 16 bytes, and the *row addresses* come from
the lanes:

| Variant | Registers per lane | Matrices | Lanes that supply addresses |
|---|---|---|---|
| `.x1` | 1 | 1 | 0 to 7 (matrix 0) |
| `.x2` | 2 | 2 | 0 to 7 (matrix 0), 8 to 15 (matrix 1) |
| `.x4` | 4 | 4 | 0 to 7, 8 to 15, 16 to 23, 24 to 31 (matrices 0 to 3) |

Lane $\ell$ of an `.x4` call supplies the address of row $\ell \bmod 8$ of
matrix $\lfloor \ell / 8 \rfloor$. Each row must be 16-byte aligned, and the
rows of a matrix need not be contiguous: that is the point, because a tile in
shared memory has a row stride bigger than 16 bytes. The hardware gathers the
eight 16-byte rows and then redistributes the 64 elements so that lane $4g + t$
holds row $g$, columns $2t$ and $2t + 1$, in the destination register. The
addresses come from lanes, but the *data* goes to lanes by the 8 x 8 rule.

### A fragment of A, in one instruction

To load a 16 x 16 tile of A at row $m_0$, column $k_0$, order the four matrices
as the A fragment wants them: top-left, bottom-left, top-right, bottom-right.
Lane $\ell$ computes:

$$
\text{row} = m_0 + (\ell \bmod 8) + 8\,\big(\lfloor \ell / 8 \rfloor \bmod 2\big),
\qquad
\text{col} = k_0 + 8\,\lfloor \ell / 16 \rfloor.
$$

Matrix 0 (lanes 0 to 7) is rows 0 to 7 and columns 0 to 7. Matrix 1 (lanes 8 to 15)
is rows 8 to 15, columns 0 to 7. Matrix 2 (lanes 16 to 23) is rows 0 to 7,
columns 8 to 15, and matrix 3 (lanes 24 to 31) is rows 8 to 15, columns 8 to 15.
The four result registers are then exactly `a0`, `a1`, `a2`, `a3`.

In CUDA, the shared-memory address needs converting to a 32-bit shared-space
address, and the instruction is inline PTX:

```cpp
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
```

### `.trans` and the B operand

The `mma.sync` instruction wants B in "column-major" form, meaning lane $(g, t)$
holds $B[2t, g]$ and $B[2t + 1, g]$: two consecutive **k** values at one **n**.
In a PyTorch GEMM `A @ B`, B is stored row-major as `[k][n]`, with $n$
contiguous. Then an ordinary `ldmatrix` of an 8 x 8 block would give lane
$(g, t)$ elements $B[g, 2t]$ and $B[g, 2t + 1]$, which is the transpose of what
the instruction needs.

The `.trans` qualifier fixes it. It transposes each 8 x 8 matrix on the way
into the registers, so lane $(g, t)$ receives rows $2t$ and $2t + 1$ of column
$g$ of the block as stored. Load $B$ from its natural `[k][n]` shared-memory
tile with `.trans`, and you need no transposed copy of B anywhere.

For a 16 x 16 piece of B (16 values of k, 16 of n), the four matrices are
$(k_{0\text{-}7}, n_{0\text{-}7})$, $(k_{8\text{-}15}, n_{0\text{-}7})$,
$(k_{0\text{-}7}, n_{8\text{-}15})$, and $(k_{8\text{-}15}, n_{8\text{-}15})$.
Their address formula is the same as for A, with row as k and column as n. The
first two result registers are `b0, b1` for the n8 tile at $n_0$, and the last
two are `b0, b1` for the tile at $n_0 + 8$. One `ldmatrix.x4.trans` therefore
feeds two `mma.sync` instructions.

> [!KEY] ldmatrix replaces the layout arithmetic with an address rule
> You never compute "which lane holds which element" in a kernel. You choose the
> matrix each lane addresses, `ldmatrix` does the redistribution, and the
> fragment lands in the layout `mma.sync` expects. The only formula you write is the
> row and column each lane points at.

## Feeding the tensor cores from shared memory

A tensor core is only fast if its operands arrive. The section answers a sharper question than "is shared memory fast enough": how much shared-memory traffic does one `mma.sync` cost, and at what warp tile does it exceed the budget?

### The conflict-free layout

Each 8 x 8 matrix `ldmatrix` reads is eight 16-byte rows. Shared memory serves
128 bytes per wavefront, so those eight rows are served in one wavefront if
and only if their 16-byte segments cover 32 distinct banks. Whether they do
depends on the tile's row stride, because row $r$ starts at byte
$r \times \text{stride}$ and bank $= (\text{byte} / 4) \bmod 32$.

```viz
11-ldmatrix-banks
```

The table gives the result for the strides this chapter uses, with a bf16 tile:

| Row stride (elements) | Stride (bytes) | Row $r$ starts at bank | Rows per bank (worst) |
|---|---|---|---|
| 32 | 64 | $16r \bmod 32$: 0, 16, 0, 16, ... | 4 |
| 64 | 128 | $0$ for every row | 8 |
| 40 | 80 | $20r \bmod 32$: 0, 20, 8, 28, 16, 4, 24, 12 | 1 |
| 136 | 272 | $(68r) \bmod 32 = 4r$: 0, 4, 8, ... 28 | 1 |

A stride of 64 elements, which is what an unpadded 64-wide tile has, puts all
eight rows on the same four banks: an 8-way conflict, so one `ldmatrix` takes
eight wavefronts instead of one. The lab's A tile is 32 wide, and pads each row
by 8 elements to a stride of 40 (80 bytes). Its B tile is 128 wide and
pads to 136 (272 bytes). Both strides keep every row 16-byte aligned and the
eight rows on eight distinct 16-byte bank groups. Padding costs shared memory:
$128 \times 40 \times 2 = 10{,}240$ B for the A tile instead of 8192 B. A
*swizzle*, which permutes the column positions instead of leaving gaps,
removes the conflicts at zero extra storage, and [chapter 13](/c/13-cutlass-and-cute)
builds it.

### The shared-memory budget per warp tile

Now the question of how much traffic. A warp that owns a $W_M \times W_N$ tile
of C, and steps through K 16 at a time, executes
$(W_M / 16)(W_N / 8)$ `mma.sync` instructions per K step. It loads $W_M \times 16$
elements of A and $16 \times W_N$ of B, each 2 bytes, with `ldmatrix`. Compare
the data volume with the tensor core's math time.

Take two warp tiles and work them through. Assume the SM's four tensor cores are
all busy (one warp on each), each doing 256 FMA per clock, and shared memory
delivers 128 bytes per clock per SM.

> [!EXAMPLE] A 32 x 32 warp tile against a 64 x 32 warp tile
> **32 x 32.** Per K step, a warp does $(32/16)(32/8) = 8$ instructions of 2048
> FMA each, 16,384 FMA. A tensor core needs $16384 / 256 = 64$ clocks for them.
> The warp loads $32 \times 16 \times 2 = 1024$ B of A and 1024 B of B, so 2048 B.
> Four warps need 8192 B, which takes $8192 / 128 = 64$ clocks. Shared memory
> takes exactly as long as the tensor cores: 100% busy, with no slack for a
> single conflict.
>
> **64 x 32.** Per K step: $(64/16)(32/8) = 16$ instructions, 32,768 FMA, so 128
> clocks of math. The loads are $64 \times 16 \times 2 = 2048$ B of A plus 1024 B
> of B, so 3072 B per warp and 12,288 B for four warps, which is 96 clocks. Shared
> memory is 75% busy while the tensor cores are 100% busy.

The general rule: the shared-memory clocks over the math clocks falls as the warp
tile grows, because math grows with the area $W_M W_N$ and loads grow with the
perimeter $W_M + W_N$. At 64 x 64 the ratio is 50%. The lab's warp tile is 64 x 32.

The example assumes conflict-free access and 128 B per clock. It says a 32 x 32
warp tile *cannot* reach peak; it doesn't say a 64 x 32 tile will, because that
depends on latency hiding, below.

## Registers

The tensor core reads and writes registers, so a warp tile costs registers per
thread. The section answers how many, and what that does to occupancy.

For the lab's 64 x 32 warp tile, per thread:

| Item | Count | Registers |
|---|---|---|
| Accumulators: $64 \times 32 = 2048$ fp32 values over 32 lanes | 64 floats | 64 |
| A fragments for one K step: 4 tiles, each 4 registers | 16 | 16 |
| B fragments for one K step: 4 n8 tiles, each 2 registers | 8 | 8 |
| Addresses, loop counters, global-to-shared staging | | tens more |

The accumulators alone are 64 registers, a quarter of the 255-register
per-thread limit. A register file has 65,536 registers per SM, so at 256 threads
per block and 128 registers per thread, a block uses 32,768, and two blocks fit
per SM (16 warps, 25% of the 64-warp limit). That 128 is an illustration; the
actual count depends on the compiler, and you can read it from `cuobjdump
--dump-resource-usage` or the `-Xptxas -v` flag.

Two consequences follow. First, a bigger warp tile improves the shared-memory
ratio but costs accumulator registers: a 128 x 64 warp tile would need 256
accumulator registers per thread, past the limit, so the compiler would spill to
local memory. The warp tile is a trade between smem bandwidth and registers.
Second, low occupancy is normal here. A tensor-core GEMM runs happily at 8 to 16
warps per SM, because each warp has a lot of independent work, 16 `mma.sync` per
K step.

> [!WARNING] Spills are silent
> A kernel that exceeds the register limit doesn't fail: it spills accumulators to
> local memory, which lives in the same DRAM-backed hierarchy, and slows down by an
> amount you can't predict from the source. Compile with `-Xptxas -v` and look
> for "spill stores".

## Why you cannot reach peak without pipelining

The lab's kernels are correct but unpipelined: each iteration loads a tile from
global memory, waits, computes, waits again. This section answers how far that
can go.

Count the time per K iteration for one block, on one SM. The block tile is
128 x 128 x 32, which is $128 \times 128 \times 32 = 524{,}288$ FMA. At the SM's
1024 FMA per clock, that is 512 clocks of tensor-core math. The tile loads
$(128 + 128) \times 32 \times 2 = 16{,}384$ B from global memory. Global
memory latency is on the order of hundreds of cycles; Luo and others measured
466 cycles on an A100 40 GB PCIe. If the kernel issues the loads, waits for
them, then computes, each iteration costs the load latency plus the math time
instead of the larger of the two.

With the loads taking roughly 466 clocks to arrive and 512 clocks of math, a
fully serialized iteration takes about $466 + 512 = 978$ clocks against the 512
the math alone needs, 52% of peak at best from this term alone. Two resident
blocks per SM overlap one block's load wait with the other's math, which helps
but doesn't remove the dependence. This is arithmetic from the measured latency,
not a measurement of the lab's kernel: the lab reports what you get.

The fix is to **issue the loads for iteration $i + 1$ while the tensor cores run
iteration $i$**. This is software pipelining, and the Ampere hardware supports it
with `cp.async`, which copies global to shared memory without going through
registers and without blocking the warp. [Chapter 12](/c/12-pipelining-and-async-copies)
builds the multistage version. The tensor-core inner loop you write here doesn't
change: only how its operands arrive.

> [!KEY] The tensor core is not the hard part
> By the numbers, one `mma.sync` takes 8 clocks per tensor core
> ($2048 / 256$), and its latency is about 25. You need at least $25 / 8 \approx 3$
> independent instructions in flight per tensor core to keep it busy. The warp
> tile gives you 16 independent ones per K step. The hard part is the operands:
> bytes from HBM to shared memory to registers, on time.

## Hopper and Blackwell

Everything above is the Ampere model, and later GPUs change it in one direction: they make the tensor core bigger and take the operand traffic out of the registers and the warp. The section answers what changes and which chapter picks it up.

| | A100 (sm_80) | H100 (sm_90) | B200 (sm_100) |
|---|---|---|---|
| Instruction | `mma.sync` m16n8k16 | `wgmma.mma_async` m64nNk16 | `tcgen05.mma` |
| Issued by | one warp (32 lanes) | a warpgroup (128 threads) | one thread |
| A operand from | registers (via `ldmatrix`) | registers or shared memory | shared memory or tensor memory |
| B operand from | registers (via `ldmatrix`) | shared memory | shared memory |
| Accumulator in | registers | registers | tensor memory (TMEM) |
| BF16 dense peak | 312 TFLOP/s | 989.4 TFLOP/s (SXM) | 2250 TFLOP/s (HGX) |

The peaks are from the fact sheet: H100 SXM is rated at 989.4 TFLOP/s dense and
the HGX B200 at 2250. Per SM and per clock the bf16 tensor rate doubles each
generation: 2048 FLOP on the A100, 4096 on the H100, 8192 on the B200.

**Hopper.** `wgmma` is issued by four warps together, computes a $64 \times N \times 16$ tile
with $N$ up to 256, is *asynchronous*, and reads B (and optionally A) directly
from shared memory through a descriptor. There is no `ldmatrix` for B and no
fragment registers for it. You issue `wgmma.fence`, then `wgmma.mma_async`, then
`wgmma.commit_group`, and later `wgmma.wait_group`; touching the accumulator
before the wait is undefined. The Tensor Memory Accelerator (TMA) feeds the shared
memory. On an H800 PCIe, Luo and others measured 729 TFLOP/s for FP16 `wgmma`, 96%
of that part's 756.5 rating.

**Blackwell.** `tcgen05.mma` is issued by a single thread. The accumulator
lives in a new 256 KiB-per-SM memory, tensor memory: 128 rows by 512 columns of
32-bit cells. Registers no longer hold accumulators, so the register pressure of the
previous section moves to tensor memory, and a thread reads the result with
`tcgen05.ld`. [Chapter 12](/c/12-pipelining-and-async-copies) covers the
asynchronous machinery of both, and [chapter 13](/c/13-cutlass-and-cute) the
layouts. You can't run either on the lab A100, so the lab exercises the A100's
version of the same idea.

## How Triton and cuBLAS get here

You rarely write `mma.sync` by hand. `tl.dot` in Triton lowers to `mma.sync` on
sm_80 and to `wgmma` on sm_90 ([chapter 14](/c/14-triton)), and cuBLAS and
CUTLASS use the same instructions. Knowing the instruction level still matters
when you read profiler output and SASS that names `HMMA` and `LDSM` (the SASS
names of `mma.sync` and `ldmatrix`), when a tile shape wastes the tensor core,
and when you fuse an epilogue into the accumulator registers, which needs the
lane-to-element map from earlier.

## The lab kernels in detail

The lab builds one block tile, used by both kernels, and changes only the inner
update. This section walks through the shared design so the TODOs make sense.

**Tile geometry.** A block computes a 128 x 128 tile of C with 256 threads, eight
warps arranged 2 down M by 4 across N. Each warp owns a 64 x 32 piece. The K loop
steps 32 at a time, so each block iteration is two 16-wide steps.

**Loading.** Each thread copies two 16-byte chunks of the A tile and two of the B
tile with `uint4` loads into shared memory: A as `[128][40]` and B as `[32][136]`
bf16, padded as in the bank table. The total is $10{,}240 + 8{,}704 = 18{,}944$ B
per block. The Python wrapper zero-pads M and N to multiples of 128 and K to a
multiple of 32, so the kernels have no bounds checks. Zeros add nothing to the
products, and the wrapper slices the padded rows and columns off the result.

**Padding costs work.** With 128 x 128 tiles, $M = 129$ computes 256 rows. That
is tile quantization, and it's why the performance test uses $4096^3$.

**The WMMA update.** Per 16-wide K step each warp loads four `matrix_a`
fragments (one per 16 rows of its 64) and two `matrix_b` fragments (one per 16 of
its 32 columns) and calls `mma_sync` eight times. The accumulators are a
`4 x 2` array of fragments.

**The `mma.sync` update.** Per 16-wide K step each warp issues four
`ldmatrix.x4` for A, two `ldmatrix.x4.trans` for B, and 16 `mma.sync`, for a
`4 x 4` array of m16n8 accumulators. B fragment $j$ is registers $2(j \bmod 2)$ and
$2(j \bmod 2) + 1$ of load $\lfloor j / 2 \rfloor$.

**The epilogue.** Lane $4g + t$ writes $c_0, c_1$ to row $g$, columns $2t$ and
$2t + 1$, as one `float2`, and $c_2, c_3$ to row $g + 8$.

Predict before you run. The tensor floor for the $4096^3$ product is 0.44 ms.
Your lab reports TFLOP/s and the percentage of `torch.matmul` on the same
shape. Expect both kernels to land well below the vendor kernel, because neither
pipelines its loads. The test fails only if a kernel doesn't beat 19.5 TFLOP/s,
the fp32 CUDA-core peak: below that the tensor cores aren't doing the work.

## What goes wrong

- **A fragment pointer that isn't 32-byte aligned.** `load_matrix_sync` needs
  256-bit alignment. An odd leading dimension or a tile offset that isn't a multiple of
  16 elements breaks it, sometimes silently with wrong numbers.
- **Mixing up B's layout.** If you `ldmatrix` B without `.trans` from a `[k][n]`
  tile, every lane holds the transpose of what the instruction expects, and the
  result is a plausible-looking wrong matrix. Test with the identity matrix, as
  the lab does: if C is not A, the layout is wrong.
- **Wrong lane-to-address mapping in `ldmatrix`.** The row address comes from lane
  $\ell \bmod 8$ of matrix $\lfloor \ell / 8 \rfloor$. Swapping the second and third
  matrices swaps the registers `a1` and `a2`, which exchanges the two 8 x 8
  blocks of A in the K and M directions.
- **Divergence around a warp-wide instruction.** `mma.sync`, `ldmatrix`, and
  every WMMA call need all 32 lanes. An `if (threadIdx.x < ...)` around one gives undefined results.
- **Unpadded shared tiles.** A 64-element-wide bf16 tile makes every `ldmatrix`
  an 8-way bank conflict. The result is correct and slow.
- **Comparing with a reference in bf16.** Compare against an fp32 product of the
  same bf16 inputs, not against bf16 output rounded to 7 mantissa bits, or the
  tolerance has to be wide enough to hide real bugs.
- **Treating 312 as the target.** Luo and others got 97% from the instruction in
  isolation on an A100 PCIe 40 GB. The Triton tutorial reaches 71 to 79% on a full GEMM.
  Judge your kernel against `torch.matmul` on your own card.

> [!RECAP]
> - A tensor core computes $D = A \cdot B + C$ on a small tile per instruction:
>   $16 \times 8 \times 16$ for `mma.sync` in bf16 or fp16, with fp32
>   accumulation and exact products.
> - The A100's 312 versus 19.5 TFLOP/s is 2048 versus 128 FLOP per clock per
>   SM, a 16x gap, and it raises the ridge point to 161 FLOP per byte.
> - WMMA hides the layout behind fragments. `mma.sync` fixes it: lane $4g + t$
>   holds row $g$ (and $g + 8$), columns $2t$ and $2t + 1$ of each 8 x 8 block.
> - `ldmatrix` loads one to four 8 x 8 blocks, with lane $\ell$ supplying the
>   address of row $\ell \bmod 8$ of matrix $\lfloor \ell / 8 \rfloor$, and
>   `.trans` handles a `[k][n]` B operand.
> - A 32 x 32 warp tile needs 100% of shared-memory bandwidth to keep the tensor
>   cores busy, a 64 x 32 tile needs 75%, and a 64-wide tile without padding
>   hits 8-way conflicts.
> - Reaching peak needs pipelined loads ([chapter 12](/c/12-pipelining-and-async-copies));
>   Hopper and Blackwell move operands and accumulators out of registers.

## Check your understanding

> [!QUESTION] How many FLOP does one `mma.sync.m16n8k16` do, and how many clocks does one tensor core need for it at full rate?
> $16 \times 8 \times 16 = 2048$ FMA, which is 4096 FLOP. A tensor core does 256
> FMA per clock, so it needs $2048 / 256 = 8$ clocks.

> [!QUESTION] Why is the A100's tensor-core peak 16 times its fp32 peak, and what does that do to the ridge point?
> Per SM and clock, the tensor cores do 2048 FLOP and the CUDA cores 128, a
> ratio of 16 at the same clock. The ridge point against 1935 GB/s rises from
> $19.5 / 1.935 = 10.1$ to $312 / 1.935 = 161.2$ FLOP per byte, so a kernel needs
> 16 times more reuse per byte to stay compute-bound.

> [!QUESTION] In an `mma.sync.m16n8k16` A fragment, which row and columns does lane 13 hold in register `a0`?
> Lane 13 has $g = \lfloor 13 / 4 \rfloor = 3$ and $t = 13 \bmod 4 = 1$. Register
> `a0` holds the pair $a_0, a_1$: row $g = 3$, columns $2t = 2$ and $3$. Register
> `a1` holds row 11, columns 2 and 3.

> [!QUESTION] Why does loading B with `ldmatrix` from a `[k][n]` shared-memory tile need `.trans`?
> The instruction wants lane $(g, t)$ to hold $B[2t, g]$ and $B[2t + 1, g]$: two
> consecutive k values at one n. Plain `ldmatrix` hands lane $(g, t)$ two
> consecutive elements of a stored row, which in a `[k][n]` tile are two
> consecutive n values at one k. That is the transpose, and `.trans` undoes it.

> [!QUESTION] A bf16 tile in shared memory has a row stride of 64 elements. What happens to an `ldmatrix`, and what are two fixes?
> Each row is 128 bytes apart, so all eight 16-byte rows start at the same bank
> and hit the same four banks: an 8-way conflict, eight wavefronts instead of one.
> Pad each row by 8 elements, giving a stride of 72 (144 bytes), or swizzle the
> column index so that each row's 16-byte chunks land on different banks.

> [!QUESTION] A warp tile is 32 x 64. Is shared memory or the tensor core the bottleneck, assuming four busy warps per SM and conflict-free loads?
> Per K step: $(32/16)(64/8) = 16$ instructions, 32,768 FMA, 128 clocks of math.
> Loads: $32 \times 16 \times 2 = 1024$ B of A and $16 \times 64 \times 2 = 2048$
> B of B, so 3072 B per warp, 12,288 B for four warps, 96 clocks at 128 B per
> clock. Shared memory is at 75% of the math time, so the tensor core is the
> limit, the same as for a 64 x 32 tile.

> [!QUESTION] A 128 x 64 warp tile in fp32 accumulators needs how many registers per thread, and why is that a problem?
> $128 \times 64 = 8192$ values over 32 lanes is 256 per thread, more than the
> 255-register limit before counting any fragment or address register. The
> compiler spills accumulators to local memory, which slows the kernel.

> [!QUESTION] Why does the lab's identity-matrix test catch bugs that random inputs can hide?
> With A a ramp and B the identity, C must equal A exactly, element for element,
> so any swapped or transposed fragment moves a value to the wrong place and the
> test fails. With random data and a loose relative tolerance, a small
> permutation within a tile can still be hard to see.

## Lab

> [!TRY] WMMA, then mma.sync with ldmatrix
> Open the lab `11-wmma-gemm`. In `starter.py`, fill in the WMMA kernel's
> accumulate loop and store, then the three inline-PTX helpers (`ldmatrix_x4`,
> `ldmatrix_x4_trans`, `mma_16816`), the lane-to-address arithmetic, the
> `mma.sync` main loop, and the epilogue. The tests check both `wmma_gemm` and
> `mma_gemm` against an fp32 reference on eight shapes, including 1 x 1 x 1,
> 17 x 33 x 19, and 257 x 129 x 65, and on an identity product. They then time
> a $4096^3$ bf16 GEMM and report `wmma_tflops`, `mma_tflops`, `torch_tflops`, and
> each kernel's percentage of `torch.matmul`. You pass when both kernels are
> correct and each exceeds 19.5 TFLOP/s, the fp32 CUDA-core peak. The first run
> compiles with `nvcc` and takes 30 to 60 seconds.

## Further reading

- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/):
  the WMMA API and the fragment model.
- [CUDA Programming Guide, warp matrix functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html):
  the supported shapes, types, and alignment rules.
- [PTX ISA: warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html):
  `mma`, `ldmatrix`, and the exact fragment layouts per type.
- [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/abs/2206.02874):
  latency, throughput, and the numeric behavior inside the unit.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499):
  the A100 `mma.sync` and H800 `wgmma` measurements this chapter quotes.
- [How to Write a Fast Matrix Multiplication from Scratch with Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html):
  a readable sequence from WMMA to a cuBLAS-class kernel on an A100.
- [NVIDIA A100 architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf):
  256 fp16 FMA per tensor core per clock, and the TF32, BF16, and FP64 modes.
