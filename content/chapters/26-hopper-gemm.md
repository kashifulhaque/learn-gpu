---
title: Hopper GEMM
slug: 26-hopper-gemm
part: "Part 3 — Matrix multiplication"
summary: Encode a TMA box, pick the swizzle that matches it, and issue one wgmma from a shared-memory descriptor. The kernels run only on an H100.
minutes: 95
gpu: true
objectives:
  - "Compute the byte offset of one element of a TMA box, including the box origin, and fill out-of-bounds elements with zero."
  - "Apply the 128-byte swizzle as an XOR on 16-byte chunks, and choose the span that equals the box's inner width."
  - "Count the bytes an mbarrier expects, and pack a wgmma descriptor with the 128-byte mode stored as layout type 1."
  - "Say which thread issues the bulk tensor copy and which threads issue wgmma, and predict the H100 SXM roofline floor without treating it as a measurement."
lab: 26-hopper-gemm
---

# Hopper GEMM

> [!TLDR]
> - A TMA box is a rectangle on the tensor. The byte address adds the box origin before it scales by the element size and the row stride.
> - The 128-byte swizzle XORs the row into the 16-byte chunk index. Row 1, chunk 1 lands at byte 128, not at the unswizzled byte 144.
> - An mbarrier phase completes when the arrival count and the transaction-byte count are both zero. A 128 by 64 bf16 tile plus a 256 by 64 bf16 tile is 49,152 bytes.
> - `wgmma.mma_async` reads B from a shared-memory descriptor and keeps the accumulator in registers. The 128-byte mode is layout type 1 in that descriptor, which is a different number from the driver enum.
> - On an H100 SXM5 the roofline floor of a 4096 cubed bf16 GEMM is 0.1389 ms. The lab measures the kernel. This chapter states the prediction.

The file is numbered 26 so chapters 00 through 25 keep their names. The part is matrix multiplication, with chapters 10 to 14 and chapter 24. You need [chapter 12](/c/12-pipelining-and-async-copies) and [chapter 24](/c/24-cute-dsl-gemm) before the pieces below are functions you have already named.

Chapter 12 names TMA, the mbarrier, and `wgmma`, and it builds the Ampere ring that overlaps copies with math. This chapter is where you build the box address, the swizzle, the byte count, and the descriptor that a Hopper issue uses. It does not rebuild that ring.

## Before you start

You need three tools, and you need them by name:

- **A layout is an address.** The offset is coordinate times stride. Dimension 0 of a TMA tensor is the contiguous mode. ([Chapter 13](/c/13-cutlass-and-cute) and [chapter 24](/c/24-cute-dsl-gemm).)
- **A swizzle is an XOR.** Chapter 13's `Swizzle<3,4,3>` XORs bits of the row into the 16-byte chunk field. The table in this chapter is that function, written out for eight rows.
- **The ring already exists.** Chapter 12 owns the prologue, the stage count, and the phase bit that a pipeline reuses. Here you compute one byte count and issue one copy.

## A tile is a box on the tensor

This section answers one question: where, in bytes from the base pointer, does one element of the box sit?

Picture a matrix and a rectangle drawn on it. The rectangle does not have to start at the origin of the matrix. Its top-left corner is a column and a row, and every element inside the rectangle is that corner plus a local coordinate. TMA calls the rectangle a box. One thread names the box. The copy engine writes the elements.

`cuTensorMapEncodeTiled` stores the base pointer, the size of each dimension, and the stride of every dimension except the contiguous one. For a matrix, that leftover stride is the row stride, in bytes. The copy then names the box origin as coordinates, and the address ==adds the origin before it scales==.

Take a bf16 matrix whose row stride is 20 bytes, ten columns times 2. The box origin is column 4, row 2. The local element at column 0, row 0 sits at byte $(4+0)\times 2 + (2+0)\times 20 = 48$. The local element at column 3, row 1 sits at byte $(4+3)\times 2 + (2+1)\times 20 = 74$.

A stride of 20 is legal arithmetic and an illegal tensor map. The driver requires every `globalStrides` entry to be a multiple of 16 bytes, so a bf16 matrix that a kernel actually copies has a column count that is a multiple of 8. The 48 and the 74 are the address rule. The GPU tensors in the lab use a legal stride.

$$
\boxed{\mathrm{byte} = (\mathrm{origin\_col} + \mathrm{local\_col})\,\hla{b} + (\mathrm{origin\_row} + \mathrm{local\_row})\,\hlb{s}}
$$

$\hla{b}$ is the element size in bytes. $\hlb{s}$ is the row stride in bytes, which is `globalStrides[0]`. Dimension 0 is the contiguous column, so it contributes no entry to `globalStrides`. A rank-2 map stores one stride.

```python
def box_byte_offset(origin_col, origin_row, local_col, local_row, elem_bytes, stride_bytes):
    col = origin_col + local_col
    row = origin_row + local_row
    return col * elem_bytes + row * stride_bytes
```

A coordinate outside the matrix is still part of the box. With `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` the engine writes a zero there, and the byte still counts toward the barrier. On a 3 by 5 matrix, the box at column 3, row 1, with 3 columns and 2 rows, is `[[9, 10, 0], [14, 15, 0]]`. The 0 is the column that fell off the right edge. A load that treats the local index as the global index returns `[[1, 2, 3], [6, 7, 8]]` instead. The lab calls that mistake `forgot_box_origin`.

> [!KEY] The origin is part of the global coordinate
> Local $(0, 0)$ of the box above is the element 9, not the element 1. Forgetting the origin reads a different tile and still looks like a rectangle.

The rest of the encode is a short list of constraints, from the driver API. The tensor rank is at most 5. Each `boxDim` entry is at most 256. The base pointer is 16-byte aligned, and the tensor map itself is 128 bytes with a 64-byte-aligned encode pointer. When the interleave mode is none, `elementStrides[0]` is ignored and the inner box dimension, in bytes, is a multiple of 16. Tensor maps exist on compute capability 9.0 and later.

## The swizzle is an XOR on 16-byte chunks

This section answers one question: which byte of shared memory holds logical chunk $c$ of row $r$?

Picture eight rows, and eight chunks along each row. A chunk is 16 bytes. A row is $8 \times 16 = 128$ bytes, and eight rows are a 1024-byte atom. If the copy wrote the chunks in logical order, row $r$ chunk $c$ would sit at byte $128r + 16c$. The 128-byte swizzle does not. It permutes the chunks inside the row so that the same chunk index on consecutive rows lands in different places.

The permutation is an XOR. The physical chunk is the logical chunk XOR the row, and the row is taken mod 8. Row 1, chunk 1 has physical chunk $1 \oplus 1 = 0$, so the byte is $128 \times 1 + 16 \times 0 = 128$. The unswizzled address of that same chunk is 144.

$$
\boxed{p = c \oplus (r \bmod 8), \qquad \mathrm{byte} = 128r + 16p}
$$

```python
def swizzle_128b_byte(row, chunk):
    physical = chunk ^ (row & 7)
    return row * 128 + physical * 16
```

```viz
26-swizzle-map
```

The eight rows, computed the same way, are the PTX 128-byte table for 16-byte atomicity. Each entry is the physical chunk for logical chunks 0 through 7:

| Row | Physical chunks |
|---|---|
| 0 | 0, 1, 2, 3, 4, 5, 6, 7 |
| 1 | 1, 0, 3, 2, 5, 4, 7, 6 |
| 2 | 2, 3, 0, 1, 6, 7, 4, 5 |
| 3 | 3, 2, 1, 0, 7, 6, 5, 4 |
| 4 | 4, 5, 6, 7, 0, 1, 2, 3 |
| 5 | 5, 4, 7, 6, 1, 0, 3, 2 |
| 6 | 6, 7, 4, 5, 2, 3, 0, 1 |
| 7 | 7, 6, 5, 4, 3, 2, 1, 0 |

The XOR hits bits 4 to 6 of the byte offset, because a chunk is 16 bytes and $16 = 2^4$. XORing the row into bits 0 to 2 leaves the chunk index alone and scrambles the byte inside the chunk. Row 1, chunk 1 then lands at $144 \oplus 1 = 145$. The lab calls that mistake `xor_row_into_low_byte`.

The 64-byte mode uses $c \oplus (r \bmod 4)$. The 32-byte mode uses $c \oplus (r \bmod 2)$. You pick a mode by matching the span to the box, not by guessing.

The inner width of the box, in bytes, is the element size times the inner dimension. When that product is 128, 64, or 32, the matching mode is `"128B"`, `"64B"`, or `"32B"`. Any other multiple of 16 is `"none"`. A width that is not a multiple of 16 is `"illegal"`. A bf16 box of 48 columns is 96 bytes, so the mode is none. A bf16 box of 80 columns is 160 bytes, also none: 160 is a multiple of 16, and it is wider than every swizzle span. The driver allows a span at least as wide as the inner box. Matching the box means the span ==equals the inner width==. A wider span is a different layout.

> [!WARNING] The driver enum and the descriptor field use different numbers
> `CU_TENSOR_MAP_SWIZZLE_128B` is the fourth driver enumerator, value 3, after none, 32B, and 64B. The wgmma descriptor stores the 128-byte mode as layout type 1. Writing 3 into bits 62 and 63 describes a 32-byte swizzle to `wgmma`.

## The barrier completes when the bytes land

This section answers one question: when is the copy finished?

A thread-arrival count is the wrong completion test. TMA is one thread's instruction and many bytes of traffic. The mbarrier tracks both. The phase completes when the pending arrival count is zero and the pending transaction-byte count is zero.

The producer thread calls `mbarrier.arrive.expect_tx` with the byte count, which arrives once and adds that many bytes to the count. It then issues `cp.async.bulk.tensor`. The copy engine decrements the byte count as it writes shared memory, including the zeros it writes past the edge of the tensor. The other threads wait on the phase bit.

Chapter 12's two Hopper tiles are a 128 by 64 bf16 tile and a 256 by 64 bf16 tile. The first is $128 \times 64 \times 2 = 16{,}384$ bytes. The second is $256 \times 64 \times 2 = 32{,}768$ bytes. The sum is 49,152. That is the `expect_tx` argument for those two boxes. Chapter 12 reuses the phase bit inside its ring. This chapter only computes the count.

```ptx
mbarrier.init.shared::cta.b64 [bar], 1;
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], 49152;
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [dst], [map, {col, row}], [bar];
mbarrier.try_wait.parity.shared::cta.b64 pred, [bar], 0;
```

The addresses are shared-memory addresses. The coordinates list dimension 0 first, the contiguous column, then the row. A count below the real box size lets the phase flip before every byte is visible. A count above it never reaches zero, so the wait does not complete.

> [!KEY] The byte count is the whole box
> Out-of-bounds zeros are still bytes in shared memory. `expect_tx` counts them. It does not count only the elements that were inside the tensor.

## wgmma reads a 64-bit descriptor

This section answers one question: what does one `wgmma.mma_async` read, and from where?

A warpgroup is four contiguous warps, 128 threads, and the first warp's rank is a multiple of 4. $M$ is always 64. For bf16, $K$ is 16, and $N$ runs from 8 to 256 in steps of 8. This chapter's instruction is `m64n8k16`: one warpgroup, $N = 8$.

$B$ comes from a shared-memory descriptor. $A$ comes from registers or from a second descriptor. The accumulator stays in registers. The instruction is asynchronous. The warpgroup fences, issues `wgmma.mma_async`, commits the group, and waits. Reading the accumulator before the matching `wgmma.wait_group` is undefined.

The descriptor is 64 bits. The start address, the leading-byte offset, and the stride-byte offset each drop their low 4 bits and keep 14 bits: $(\mathrm{value} \mathbin{\&} \mathtt{0x3FFFF}) \gg 4$. The start field occupies bits 0 to 13, the leading offset bits 16 to 29, and the stride bits 32 to 45. A 3-bit base offset sits at bits 49 to 51 and is 0 when the pattern starts on its alignment boundary. The swizzle field occupies bits 62 and 63.

That field is not the driver enum. Layout type 0 is no swizzle, 1 is the 128-byte mode, 2 is the 64-byte mode, and 3 is the 32-byte mode. The lab calls the mix-up `driver_enum_as_descriptor`.

A descriptor with start 0, leading offset 128, stride 1024, and layout type 1 packs as `0x4000004000080000`. The 1 in bit 62 is the 128-byte mode. The 8 at bits 16 to 29 is $128 / 16$. The `0x40` at bits 32 to 45 is $1024 / 16$.

```python
def wgmma_descriptor(start, leading, stride, layout_type):
    def enc(value):
        return (value & 0x3FFFF) >> 4
    return enc(start) | (enc(leading) << 16) | (enc(stride) << 32) | (layout_type << 62)
```

The kernel in the lab does not use that 128-byte descriptor. It fills shared memory with ordinary stores, in the K-major layout with no swizzle, and then points the descriptor at that fill. For bf16 the atom's leading offset is 128 bytes and its stride is 256 bytes, both from the eight-element groups of the core matrix: $T = 8$ because $8 \times 2 = 16$ bytes, and the next group of eight rows is 256 bytes away. Layout type 0 matches that fill. A 128-byte TMA box is a different byte arrangement. Using one descriptor for the other reads the wrong elements.

Each thread of `m64n8k16` with an fp32 accumulator holds four registers. Number them $v = 0, 1, 2, 3$, and split $v = v_0 + 2v_1$ with $v_0$ the faster index. Lane $\ell$ and warp $w$ inside the warpgroup own

$$
\mathrm{row} = \lfloor \ell / 4 \rfloor + 16w + 8v_1, \qquad \mathrm{col} = 2(\ell \bmod 4) + v_0.
$$

That map is a bijection onto the 64 by 8 tile. It is the column-major reading of the CUTLASS `CLayout_64x8` strides: thread mode `(4,8,4):(128,1,16)` and value mode `(2,2):(64,8)`. One instruction performs $2 \times 64 \times 8 \times 16 = 16{,}384$ FLOP. Chapter 12 and `docs/FACTS.md` record the larger `m64n256k16` atom and its 128-cycle latency. This chapter uses the 8-wide atom so the register map stays small enough to check.

```ptx
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {d0, d1, d2, d3}, descA, descB, p, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

The predicate `p` is false when the scale on the old accumulator is zero, so the instruction ignores whatever was in `d0` through `d3`. The two scale immediates are 1. The two transpose immediates are 0, which is K-major for both operands. Shared memory that was filled by ordinary stores needs `fence.proxy.async.shared::cta` before `wgmma.fence`, so the async proxy sees those stores.

> [!EXAMPLE] A product you can check by hand
> Set $A[\mathrm{row}, 0] = \mathrm{row}$, $A[\mathrm{row}, 1] = 1$, and the rest of $A$ to 0. Set $B[0, \mathrm{col}] = 1$, $B[1, \mathrm{col}] = \mathrm{col}$, and the rest of $B$ to 0. The fp32 result at $(\mathrm{row}, \mathrm{col})$ is $\mathrm{row} + \mathrm{col}$. Every value is exact in bf16 on the way in and exact in fp32 on the way out. A wrong fragment map fails this pattern.

## A producer copies and a consumer multiplies

This section answers one question: which threads issue which instruction?

One elected thread is the producer. It initializes the mbarrier, arrives with the byte count, and issues the bulk tensor copy. The warpgroup is the consumer. It waits on the phase, fences, and issues `wgmma`. The descriptor's layout type is the swizzle the copy used. A producer that swizzles at 128 bytes and a consumer that describes an unswizzled layout disagree about every row after the first.

The lab keeps the two sides as two kernels, both compiled for `sm_90a`. The copy kernel encodes the tensor map, issues one `cp.async.bulk.tensor.2d.shared::cta`, waits, and unswizzles with the XOR above. The math kernel stages both operands in the K-major layout and issues one `m64n8k16`. A fused GEMM is those two issue sequences on one barrier. Building that fusion also means building the shared-memory layout the descriptor names. Chapter 12 owns the ring that repeats the pair across K. This chapter stops once each instruction is real.

`load_cuda` already takes an `arch` argument. The Hopper compile passes `arch="sm_90a"`. A caller that omits `arch` stays on `sm_80`. The lab's `meta.json` sets `"gpu": "H100-80GB"`. A deployment with no `RUNPOD_ENDPOINT_H100` reports `needs H100, not configured on this deployment` and does not send the job to the A100 endpoint. The CPU checks still run, because the lab is marked device-agnostic.

> [!KEY] Two kernels, one issue order
> Thread 0 copies. The warpgroup multiplies. The descriptor names the layout the copy wrote. The ring that overlaps several of these pairs is chapter 12.

## The roofline is a floor the lab measures

This section answers one question: what time do you predict, and what does the lab refuse to grade?

Count $2MNK$ FLOP and $2(MK + NK + MN)$ bytes. The H100 SXM5 ratings in `docs/FACTS.md` are 989.4 TFLOP/s dense bf16 and 3350 GB/s. The ridge is $989.4 \times 10^{12} / 3350 \times 10^{9} = 295.3$ FLOP per byte.

A 4096 cubed bf16 GEMM is $2 \times 4096^3 = 137{,}438{,}953{,}472$ FLOP and $100{,}663{,}296$ bytes. The intensity is $4096/3 = 1365.3$ FLOP per byte, on the compute side of the ridge. The floor is

$$
137{,}438{,}953{,}472 / (989.4 \times 10^{12}) = 0.1389\,\mathrm{ms}.
$$

A 128 by 128 by 64 bf16 GEMM is $2{,}097{,}152$ FLOP and $65{,}536$ bytes. The intensity is 32.0 FLOP per byte, on the memory side. The floor is $65{,}536 / (3350 \times 10^{9}) = 1.956 \times 10^{-5}$ ms.

The lab's `predicted_ms` returns both. When an H100 is present, the run records the TFLOP/s of one `m64n8k16` launch and prints the 0.1389 ms floor beside it. There is no pass/fail threshold on that rate. This chapter has no measured TFLOP/s to quote, because the kernel was not run on an H100 here.

## What goes wrong

- **The local index used as the global index.** On the 3 by 5 matrix, local $(0, 0)$ of the box at column 3, row 1 is 9. `forgot_box_origin` returns 1.
- **The XOR in the wrong bits.** Row 1, chunk 1 is byte 128. `xor_row_into_low_byte` returns 145, because $144 \oplus 1$ flips bits 0 to 2 and leaves the chunk index at 1.
- **The driver enum stored in the descriptor.** Layout type 3 is the 32-byte mode. `driver_enum_as_descriptor` writes 3 because `CU_TENSOR_MAP_SWIZZLE_128B` is the fourth enumerator. The 128-byte mode in the descriptor is 1, and the packed word is `0x4000004000080000`.
- **An `expect_tx` that is not the box size.** A short count completes the phase early. A long count never completes it. Zeros past the edge are part of the count.
- **A swizzle span that does not equal the inner width.** A 96-byte box is none, not 128B. A 128-byte mode on a 64-byte box is a different layout, even when the driver accepts the wider span.
- **A read of the accumulator before `wgmma.wait_group`.** The instruction is asynchronous. The fence, the commit, and the wait are the completion path. Skipping the wait reads registers the math has not finished.

> [!RECAP]
> - The byte offset adds the box origin, then scales by the element size and the row stride. Out-of-bounds elements are zeros, and they still count.
> - The 128-byte swizzle is $p = c \oplus (r \bmod 8)$ on 16-byte chunks. Row 1, chunk 1 is byte 128. The span you choose equals the box's inner width.
> - The mbarrier phase flips when arrivals and transaction bytes are both zero. The two tiles from chapter 12 are 49,152 bytes.
> - The wgmma descriptor stores the 128-byte mode as layout type 1. The driver enum stores it as 3. $B$ is a shared-memory descriptor, and the accumulator is registers.
> - One thread issues TMA. The warpgroup issues `wgmma`. The 4096 cubed bf16 floor on an H100 SXM5 is 0.1389 ms, and the lab measures rather than grades it.

## Check your understanding

> [!QUESTION] Origin column 4, row 2, local column 3, row 1, bf16, stride 20. What is the byte offset?
> 74. The column is $4+3 = 7$, times 2 bytes, which is 14. The row is $2+1 = 3$, times 20 bytes, which is 60. The sum is 74. Local $(0, 0)$ of the same box is 48.

> [!QUESTION] Why is a 20-byte stride fine in that example and illegal in `cuTensorMapEncodeTiled`?
> The example is the address rule. The driver requires each `globalStrides` entry to be a multiple of 16 bytes. A bf16 matrix needs a column count that is a multiple of 8 before the encode succeeds.

> [!QUESTION] Row 1, logical chunk 1. What is the swizzled byte, what is the unswizzled byte, and what does an XOR into bits 0 to 2 return?
> The physical chunk is $1 \oplus 1 = 0$, so the byte is 128. The unswizzled byte is $128 + 16 = 144$. XORing the row into bits 0 to 2 returns $144 \oplus 1 = 145$.

> [!QUESTION] A bf16 box is 48 columns wide. Which swizzle mode matches it, and which modes are legal for a 6-byte inner width?
> 48 columns times 2 bytes is 96. That equals neither 32, 64, nor 128, and 96 is a multiple of 16, so the mode is none. A 6-byte width is not a multiple of 16, so it is illegal.

> [!QUESTION] What is `expect_tx` for a 128 by 64 bf16 tile and a 256 by 64 bf16 tile?
> $16{,}384 + 32{,}768 = 49{,}152$ bytes. The count is the whole of both boxes. Zeros the engine writes past the tensor are included.

> [!QUESTION] Start 0, leading offset 128, stride 1024, 128-byte swizzle. What is the descriptor, and why is layout type 3 the wrong encoding?
> The packed word is `0x4000004000080000`. Layout type 1, in bits 62 and 63, is the 128-byte mode. Layout type 3 is the 32-byte mode. The driver enumerator 3 means 128B only inside `cuTensorMapEncodeTiled`.

> [!QUESTION] Is 0.1389 ms a measurement of the lab kernel?
> It is the roofline floor of a 4096 cubed bf16 GEMM on the H100 SXM5 ratings, 989.4 TFLOP/s and a compute-side intensity of 1365.3 FLOP per byte. The lab records a measured rate when an H100 runs the kernel, and it does not fail the run for missing the floor.

## Lab

> [!TRY] Build the box, the swizzle, and one wgmma
> Open `26-hopper-gemm`. `box_byte_offset` is the one function the first check needs. Then implement `box_load`, `swizzle_128b_byte`, `choose_swizzle`, `expect_tx_bytes`, `wgmma_descriptor`, and `predicted_ms` on the CPU. The swizzle check compares all 64 chunks with the PTX table. The descriptor check compares `0x4000004000080000` and a second word with layout type 0. The last two functions, `tma_copy` and `wgmma_64x8x16`, are the kernels. Compile them with `arch="sm_90a"`. `tma_copy` returns the bf16 box, including zeros past the edge, for swizzle `"none"` or `"128B"`. `wgmma_64x8x16` takes a bf16 `(64, 16)` and a bf16 `(16, 8)` and returns fp32 `(64, 8)`. On a machine without an H100 the kernels print a skip message and return. They are not compiled for `sm_80`, and they are not sent to the A100 endpoint.

The lab reports one metric:

- `tflops`: when an H100 runs the kernel, the useful FLOP of one `m64n8k16`, which is 16,384, divided by the measured seconds. The 0.1389 ms figure is the floor of a different shape, the 4096 cube, and the check prints it beside the measurement. The metric is a record, not a threshold. A machine without an H100 records `skipped (no H100)`.

## Further reading

- [Tensor map object management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html), the CUDA driver API. `cuTensorMapEncodeTiled`: stride alignment, the swizzle enum, the 256 limit on each box dimension, and zero fill outside the tensor.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), NVIDIA. Section 5.5.7 is the swizzle table this chapter prints. The mbarrier section states that the phase completes when the arrival count and the transaction-byte count are both zero, and it gives `cp.async.bulk.tensor` with `complete_tx::bytes`.
- [CUTLASS `mma_sm90_desc.hpp`](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/arch/mma_sm90_desc.hpp), NVIDIA, CUTLASS 3.9.2. The 64-bit descriptor fields, and layout type 0, 1, 2, 3 for none, 128B, 64B, and 32B.
- [CuTe DSL primitives](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html), NVIDIA. The same descriptor bit layout on the Python side, including the rule that the start address advances in multiples of 16 bytes.
- [CUTLASS tutorial: wgmma on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/), Colfax Research. Warpgroup operand sources, the K-major core-matrix layout, the leading and stride byte offsets, and the fence, commit, and wait sequence.
- [GPU MODE, lecture 36: CUTLASS and FlashAttention-3](https://github.com/gpu-mode/lectures/tree/main/lecture_036), Jay Shah. The lecture index titles this session CUTLASS and FlashAttention-3. The slides cover WGMMA, TMA, and warp specialization.
- [GPU MODE, lecture 23: Tensor Cores](https://github.com/gpu-mode/lectures#lecture-23-tensor-cores), Vijay Thakkar and Pradeep Ramani. The lecture index titles this session Tensor Cores. It is the series' tensor-core lecture, not the TMA session.
- [Chapter 12](/c/12-pipelining-and-async-copies), for the ring this chapter does not rebuild, and [chapter 24](/c/24-cute-dsl-gemm), for the Ampere atom the same layout algebra targets on `sm_80`.
- [Chapter 35](/c/35-blackwell-tcgen05), for the Blackwell version of this chapter: tensor memory, the `tcgen05` descriptors, and CTA pairs, with a Blackwell descriptor that differs from this one in its version field and a 3-bit layout type.