---
title: A CuTe DSL GEMM
slug: 24-cute-dsl-gemm
part: "Part 3 — Matrix multiplication"
summary: Compile the layout algebra into an Ampere GEMM in the CuTe DSL, with a TiledCopy, a TiledMMA, and a bias-plus-GELU epilogue fused into the kernel.
minutes: 100
gpu: true
objectives:
  - "Build a tensor from a pointer and a layout, and list the offsets that layout produces."
  - "Partition a tile with a thread-value layout, and check the partition against the layout algebra from chapter 13."
  - "Construct a TiledMMA for mma.sync.m16n8k16 and read one lane's accumulator cells."
  - "Call cute.copy and cute.gemm, and fuse a bias plus the erf GELU into the epilogue so the raw product stays off the global store."
  - "Predict the roofline floor of an fp16 GEMM on the A100 80GB PCIe, and leave the measured TFLOP/s to the lab."
lab: 24-cute-dsl-gemm
---

# A CuTe DSL GEMM

> [!TLDR]
> - A CuTe tensor is a pointer plus a layout. The layout is the address map you already evaluate by hand.
> - A TiledCopy and a TiledMMA are thread-value layouts. `cute.copy` and `cute.gemm` issue the copy atom and the MMA atom those layouts describe.
> - On Ampere the atom is `mma.sync.m16n8k16`, with fp16 operands and an fp32 accumulator. Four warps cover a 128 by 128 tile when you arrange them 2 by 2.
> - Fuse bias and GELU in the accumulator. For a 128 by 128 fp16 tile the two extra kernels move 131,584 bytes, and the fused epilogue adds a 512-byte bias read.
> - The roofline floor for a 4096 cubed fp16 GEMM on the A100 80GB PCIe is 0.4405 ms. The lab measures the kernel. This chapter states the prediction.

[Chapter 13](/c/13-cutlass-and-cute) gives you the layout algebra and stops on the page. The C++ sketch there is how a CUTLASS mainloop is shaped. This chapter compiles the same three steps in Python: build a tensor from a layout, partition it with a TiledCopy and a TiledMMA, then call `cute.copy` and `cute.gemm`. The epilogue adds a bias and applies GELU before the store.

The file is numbered 24 so chapters 00 through 23 keep their names. The part is matrix multiplication, with chapters 10 to 14. You need chapter 13 and its lab, `13-layouts`, before the partitions below will look like functions you have already implemented.

The worker image carries the package this chapter compiles against. `docs/FACTS.md` records `nvidia-cutlass-dsl[cu13]==4.8.1` and the CUTLASS 3.9.2 headers at `/opt/cutlass/include`. The lab calls the Python package. The [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) describes that package as the Python programming model aligned with CuTe layouts, tensors, copy atoms, MMA atoms, and tiled operations, JIT-compiled through the CUTLASS Python compiler.

## Before you start

You need four ideas from earlier chapters, and you need them as tools, because this chapter uses each one by name:

- **A layout is `shape:stride`.** The offset is the sum of coordinate times stride. An integer coordinate into a nested shape is colexicographic, first mode fastest. ([Chapter 13](/c/13-cutlass-and-cute).)
- **`mma.sync.m16n8k16`.** One warp computes a 16 by 8 by 16 product from registers, fp16 in and fp32 accumulated. Lane $\ell$ holds rows $\lfloor \ell/4 \rfloor$ and $\lfloor \ell/4 \rfloor + 8$. ([Chapter 11](/c/11-tensor-cores).)
- **`cp.async`.** A thread copies 16 bytes from global memory to shared memory. [Chapter 12](/c/12-pipelining-and-async-copies) overlaps several of those copies with the math. This chapter issues one stage, so the DSL calls stay visible.
- **A Triton tile program.** [Chapter 14](/c/14-triton) owns GEMM tiling in Triton, including the grouped launch order. Here the atom is yours to name. That is the whole contrast.

## A tensor is a pointer plus a layout

This section answers one question: what do you pass to a CuTe DSL kernel instead of a raw pointer and a leading dimension?

Picture a 4 by 3 matrix and a pointer to its first element. Two readers can share that pointer and disagree about every other address. One walks down a column before stepping to the next, so element $(2, 1)$ sits 6 words past the base. The other walks along a row, so the same element sits 7 words past the base. The pointer did not decide. The layout did.

CuTe calls the pair a *tensor*: an iterator, usually a pointer, and a layout. The [overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) lists the tensor as the basic CuTe data object. You already know the layout half. Chapter 13's `(4,3):(1,4)` is column-major, and `(4,3):(3,1)` is row-major. The DSL writes the same pair.

$$
\boxed{L(c_0, \ldots, c_{n-1}) = \sum_i \hla{c_i}\,\hlb{d_i}}
$$

$\hla{c_i}$ is the coordinate along mode $i$, and $\hlb{d_i}$ is the stride of that mode, in elements. A row-major $M \times K$ matrix has shape $(M, K)$ and stride $(K, 1)$: the $K$ mode is contiguous, which is what a 128-bit load wants.

> [!EXAMPLE] Two strides, one pointer
> `(4,3):(1,4)` at integer coordinates 0 through 11 produces offsets 0 through 11. `(4,3):(3,1)` produces $0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11$. The hierarchical layout `((2,2),3):((1,6),2)` produces $0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11$. Those three lists are the first thing the lab checks, and they are the same lists chapter 13 prints.

The following code builds the row-major tensor the GEMM reads as $A$. `m` and `k` are dynamic sizes. The unit stride on $K$ stays static, which is what lets the copy stay 16 bytes wide:

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def matrix_a(ptr, m, k):
    layout = cute.make_layout((m, k), stride=(k, 1))
    return cute.make_tensor(ptr, layout)
```

> [!KEY] The layout is the whole address map
> Once the tensor exists, tiling and partitioning are layout operations. You stop writing `row * lda + col` at the call site.

$B$ has a second constraint. The Ampere `mma.sync` atom in the [warp-level MMA guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wmma_programming.html) fixes the operand layout: $A$ is row-major along $K$, and $B$ is column-major along $K$. A PyTorch `(K, N)` row-major matrix is contiguous in $N$. The lab transposes it to a contiguous `(N, K)` tensor, stride `(K, 1)`, and passes that pointer. The transpose is a host copy. The timer in the lab starts after it.

## A TiledCopy is a thread-value layout

This section answers one question: which thread owns which elements of a tile?

Picture 32 threads loading an 8 by 32 tile of fp16 values, row-major, `(8,32):(32,1)`. Each thread takes 8 consecutive halves, which is 16 bytes, one 128-bit access. Four threads cover a row. Eight such groups cover the tile. Chapter 13 writes that ownership as a thread-value layout:

$$
\texttt{((4,8),8):((8,32),1)}
$$

The first mode is the thread. An integer thread index $t$ splits colexicographically into $t_0 = t \bmod 4$ and $t_1 = \lfloor t/4 \rfloor$. The second mode is the value, eight consecutive elements. The offset of thread $t$ and value $v$ is:

$$
\boxed{8\, t_0 + 32\, t_1 + v}
$$

Thread 1 has $t_0 = 1$ and $t_1 = 0$, so value 0 lands at offset 8. A grid-stride map, element $t + v \cdot 32$, lands at offset 1 for the same pair. ==The layout and the grid-stride map disagree at the first interesting thread.== The lab's partition check fails the grid-stride map there, and the comment in `tests.py` names that wrong function `striped_thread_partition`.

| Thread | Value | Layout offset | Grid-stride offset |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 8 | 1 |
| 4 | 0 | 32 | 4 |
| 1 | 3 | 11 | 97 |

Both maps are bijections of the 256 elements. Only the layout map is eight contiguous halves per thread, which is the access the copy atom can issue as one 128-bit instruction.

`make_tiled_copy_tv` builds that object from three pieces: a copy atom, a thread layout, and a value layout. For the global-to-shared load of a 128 by 64 fp16 tile the lab uses a 128-bit `cp.async` atom, a thread layout `(16, 8):(8, 1)` (16 rows of threads, 8 threads along the row), and a value layout `(1, 8)`:

```python
atom = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(),
    cutlass.Float16,
    num_bits_per_copy=128,
)
thread = cute.make_layout((16, 8), stride=(8, 1))
value = cute.make_layout((1, 8))
tiled_copy = cute.make_tiled_copy_tv(atom, thread, value)
```

`get_slice(tid)` restricts the tiled copy to one thread. `partition_S` is that thread's source elements, and `partition_D` is its destination elements. `cute.copy` then issues the atom. The predicate argument, when you pass one, drops the elements that fall past the matrix. The lab pads instead, so every thread's copy is full and the predicate stays out of the listing.

> [!INTUITION] Partition, then call
> You can print the partition on the CPU, with the same algebra as lab 13, before you ever launch the kernel. If thread 1 owns offset 8 on the CPU, it owns offset 8 in the generated copy.

The shared-memory side of the same tile uses the swizzle from chapter 13. `Swizzle<3,3,3>` on an 8 by 64 half tile, composed with `(8,64):(64,1)` and repeated up to 128 by 64, is the layout the lab allocates. The swizzle is what keeps the later `ldmatrix` from stacking eight rows on the same banks. Chapter 13 already counts that conflict. This chapter reuses the layout and spends its attention on the copy call.

## A TiledMMA repeats one instruction

This section answers one question: how does one `mma.sync` become a 128 by 128 tile?

One Ampere warp instruction covers 16 rows, 8 columns, and 16 along $K$. The accumulator thread-value layout chapter 13 quotes is `((4,8),(2,2)):((32,1),(16,8))`, over a column-major 16 by 8 tile whose offset is $m + 16 n$. Decoding the layout gives the fragment table:

$$
m = \left\lfloor \frac{\ell}{4} \right\rfloor + 8 \left\lfloor \frac{v}{2} \right\rfloor, \qquad
n = 2\,(\ell \bmod 4) + (v \bmod 2)
$$

Lane 1, value 0 is $(m, n) = (0, 2)$, offset $0 + 16 \cdot 2 = 32$. Its four values are $(0, 2)$, $(0, 3)$, $(8, 2)$, and $(8, 3)$, at offsets 32, 48, 40, and 56. `scripts/figures/24-cute-dsl-gemm.py` prints that grid. A guess that hands lane $\ell$ the cell $(\ell \bmod 16,\; \lfloor \ell/16 \rfloor)$ and ignores $v$ puts lane 1 at $(1, 0)$. The lab names that wrong function `row_major_fragment`.

```viz
24-fragment-owners
```

The figure's initial lane is 1, so the printed page shows those four cells. Drag the lane control and the four highlights move together: two columns, and the same two columns eight rows down. That is the whole fragment table.

A *TiledMMA* repeats the atom across warps and across values. The lab builds it from `MmaF16BF16Op`, which the warp-level guide defines for fp16 or bf16 operands, an fp32 accumulator, and instruction shape `(16, 8, 16)`, on sm_80 and later:

```python
op = cute.nvgpu.warp.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32, (16, 8, 16)
)
tiled_mma = cute.make_tiled_mma(
    op, (2, 2, 1), permutation_mnk=(32, 32, 16)
)
```

`(2, 2, 1)` places four warps, two along $M$ and two along $N$. `permutation_mnk=(32, 32, 16)` is the tile one warp walks: 32 along $M$, 32 along $N$, and one instruction along $K$. The CTA tile is 128 by 128 by 64, so each partition has a rest mode. The $K$ loop is that rest mode. `thr_mma.make_fragment_C(...)` allocates the fp32 accumulators, and `fill(0.0)` clears them. `cute.gemm(tiled_mma, acc, a_frag, b_frag, acc)` reads the accumulator and writes it back. There is no separate "accumulate or overwrite" flag. The guide states that the call is the synchronous `mma.sync` instruction, and that $A$ and $B$ already sit in registers.

The register load from shared memory is another tiled copy, this time an `ldmatrix` atom, `LdMatrix8x8x16bOp`, retiled onto the MMA fragments with `make_tiled_copy_A` and `make_tiled_copy_B`. Chapter 11 wrote the lane-to-address table for that instruction. The DSL copy atom carries the same table.

| Piece | Value | Where it comes from |
|---|---|---|
| Instruction tile | 16 by 8 by 16 | `MmaF16BF16Op` shape, chapter 11 |
| Warps | 4, arranged 2 by 2 | Atom layout `(2, 2, 1)`, 128 threads |
| One stage of $A$ and $B$ | 32 KiB | $2 \times 128 \times 64 \times 2$ bytes |
| Epilogue tile of $C$ | 32 KiB | $128 \times 128 \times 2$ bytes, fp16 |
| CTA shared memory | 64 KiB | The two lines above, one stage |

64 KiB fits in the 163 KiB a block may request. Chapter 12's multistage pipeline multiplies the $A$ and $B$ term by the stage count. One stage is the listing you can read in one sitting.

## Copy, then gemm, then the epilogue

This section answers one question: what does the inner loop actually call, and where does the activation go?

The loop for one $K$ tile has three beats. `cute.copy` moves the global tile into shared memory through the `cp.async` TiledCopy. A wait and a block barrier make that shared memory visible. `cute.copy` again, through the `ldmatrix` TiledCopy, fills the register fragments. `cute.gemm` consumes one $K$-block of those fragments. With $K$ stepped by 64 and an instruction $K$ of 16, the inner loop runs four times per tile.

```python
for k_tile in cutlass.range(k_tiles):
    cute.copy(tiled_copy_ab, tAgA[None, None, None, k_tile], tAsA)
    cute.copy(tiled_copy_ab, tBgB[None, None, None, k_tile], tBsB)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    for k_block in cutlass.range(k_blocks, unroll_full=True):
        cute.copy(s2r_a, tCsA[None, None, k_block], tCrA[None, None, k_block])
        cute.copy(s2r_b, tCsB[None, None, k_block], tCrB[None, None, k_block])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
    cute.arch.sync_threads()
```

The `None` slices keep every mode except the one named index. `k_tile` selects the $K$ tile. `k_block` selects the 16-wide slice inside it. After the last tile the accumulator holds the fp32 product, distributed across the fragment table above.

### Bias, then GELU, in the fragment

A GEMM that writes $C$ and returns, followed by a bias kernel and a GELU kernel, is three launches and two extra trips through $C$. The fused epilogue adds the bias and applies GELU while the product still sits in registers, and the global store is the activated value.

The activation is the erf form, the one `torch.nn.functional.gelu` computes when `approximate` is left at its default:

$$
\mathrm{GELU}(x) = \tfrac{1}{2}\, x \left(1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

> [!EXAMPLE] One element, two orders
> Take the product element $1$ and a bias of $0.5$. The sum is $1.5$, and $\mathrm{GELU}(1.5) = 1.399789$. Applying GELU first and adding the bias after yields $\mathrm{GELU}(1) + 0.5 = 1.341345$. The lab names that second function `gelu_then_bias`. The kernel has to match the first one, because that is what the PyTorch reference computes.

The bias is a length-$N$ fp32 vector. As a tensor of shape $(M, N)$ with stride $(0, 1)$ it broadcasts across rows, so the same TiledCopy that stores $C$ can load the bias. The arithmetic is elementwise on the register fragment the copy owns:

```python
acc = fragment.load().to(cutlass.Float32) + bias_fragment.load()
out = acc * 0.5 * (1.0 + cute.math.erf(acc * 0.7071067811865476))
fragment.store(out.to(cutlass.Float16))
```

$1/\sqrt{2} = 0.7071067811865476$. `cute.math.erf` is the DSL math op, applied to the whole fragment value, so the loop over elements is the op's own, and you keep the fragment's layout.

The byte count for one 128 by 128 fp16 tile, with an fp32 bias, is small enough to do by hand. $C$ is $128 \times 128 \times 2 = 32{,}768$ bytes. The bias is $128 \times 4 = 512$ bytes.

| Schedule | Traffic beyond the GEMM's own store of $C$ |
|---|---|
| Bias kernel: read $C$, read bias, write $C$ | $32{,}768 + 512 + 32{,}768 = 66{,}048$ bytes |
| GELU kernel: read $C$, write $C$ | $65{,}536$ bytes |
| Unfused total | $131{,}584$ bytes |
| Fused epilogue, extra read | $512$ bytes |

The fused schedule moves $131{,}584 / 512 = 257$ times less extra traffic. The GEMM still stores $C$ once. Fusion removes the two kernels that would have read that store back.

> [!WARNING] The product has to stay off the global store
> A kernel that writes the raw GEMM and then, in the same launch, reads it back to apply GELU has fused the launches and left the bytes. The accumulator fragment is the place the activation belongs.

### The floor, and what the lab measures

Count $2MNK$ FLOP and $2(MK + NK + MN)$ bytes for fp16 $A$, $B$, and $C$. On the A100 80GB PCIe the dense fp16 tensor-core peak is 312 TFLOP/s and the rated bandwidth is 1935 GB/s, both from `docs/FACTS.md`. The ridge is $312 \times 10^{12} / 1935 \times 10^{9} = 161.2$ FLOP per byte.

> [!EXAMPLE] A 4096 cubed GEMM
> The product is $2 \times 4096^3 = 137{,}438{,}953{,}472$ FLOP. The three matrices are $100{,}663{,}296$ bytes. The intensity is $1365.3$ FLOP per byte, on the compute side of the 161.2 ridge. The floor is $137{,}438{,}953{,}472 / 312 \times 10^{12} = 0.4405$ ms. A 64 cubed GEMM is on the memory side: $24{,}576$ bytes over 1935 GB/s. The lab's `predicted_ms` returns both, and the A100 run reports the measured TFLOP/s beside the floor. This chapter has no measured TFLOP/s to quote.

Chapter 12's hand-written pipeline is the thing to compare against, and the comparison is about the text you no longer write. That kernel names the thread's row, the `ldmatrix` address, the `cp.async` group, and the wait count. The DSL kernel names a TiledCopy, a TiledMMA, and the three calls above. The wait is still yours: one stage hides none of the copy latency chapter 12 measured. Putting the stages back is a change to the loop around `cute.copy`, with the same atoms. The lab leaves that change out so a correct epilogue is the thing the test can see.

Triton, in chapter 14, writes this as a tile program and a `tl.dot`. You choose a block size and a warp count, and the compiler chooses the instruction. Use that chapter when the instruction is allowed to move. Use this chapter when you name `mma.sync.m16n8k16` and keep the epilogue on the accumulator fragment. Chapter 14 keeps the Triton side. This chapter keeps the atom.

## What goes wrong

- **A grid-stride partition.** `thread + value * num_threads` is a bijection, and it is the wrong one. Thread 1 value 0 of the 8 by 32 copy layout is offset 8. The striped map returns 1, and the vectorized copy breaks because the eight values are no longer contiguous.
- **GELU, then the bias.** $\mathrm{GELU}(1) + 0.5 = 1.341345$. $\mathrm{GELU}(1.5) = 1.399789$. The reference adds first. A kernel that matches itself and misses PyTorch has this order.
- **$B$ left in row-major $(K, N)$.** The atom wants $B$ contiguous in $K$. A host transpose to contiguous `(N, K)` is the fix. Forgetting it compiles, and the product is the product against the transposed operand.
- **A dynamic stride on the contiguous mode.** Chapter 13's warning still holds. A layout whose every stride is a runtime integer compiles. The 128-bit copy wants the contiguous stride to be the static integer 1.
- **Quoting the floor as a result.** 0.4405 ms is $2 \times 4096^3 / 312 \times 10^{12}$, on the rated peak. The lab prints the measured TFLOP/s. Until that print exists, the floor is a prediction.

> [!RECAP]
> - A tensor is a pointer plus a layout. The layout you evaluate in lab 13 is the layout `make_tensor` stores.
> - A TiledCopy's thread-value layout says which thread owns which elements. `cute.copy` issues the atom. For the 8 by 32 half tile, thread 1 value 0 is offset 8.
> - A TiledMMA repeats `mma.sync.m16n8k16`. Lane 1 holds accumulator cells $(0, 2)$, $(0, 3)$, $(8, 2)$, and $(8, 3)$. `cute.gemm` accumulates in fp32.
> - The epilogue adds the bias and applies the erf GELU on the fragment. A 128 by 128 tile drops from 131,584 extra bytes to a 512-byte bias read.
> - The 4096 cubed fp16 floor on the A100 80GB PCIe is 0.4405 ms. The lab measures the kernel. One stage is a readable kernel, and chapter 12 is how you overlap the copy.

## Check your understanding

> [!QUESTION] What offsets does `(4,3):(3,1)` produce for integer coordinates 0 through 11?
> $0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11$. Coordinate $c$ is row $c \bmod 4$ and column $\lfloor c/4 \rfloor$, and the stride is $(3, 1)$, so the offset is $3 \times \text{row} + \text{column}$.

> [!QUESTION] Thread 1, value 0 of `((4,8),8):((8,32),1)`. What is the offset, and what does a grid-stride map return?
> $t_0 = 1$, $t_1 = 0$, so the offset is $8 \cdot 1 + 32 \cdot 0 + 0 = 8$. The grid-stride element $1 + 0 \cdot 32$ is 1. The layout is the one the 128-bit copy can issue.

> [!QUESTION] Which four cells of the 16 by 8 accumulator does lane 1 hold?
> Values 0 through 3 are $(m, n) = (0, 2)$, $(0, 3)$, $(8, 2)$, $(8, 3)$. The offsets in the column-major tile are 32, 48, 40, and 56.

> [!QUESTION] Why is the atom layout `(2, 2, 1)` with `permutation_mnk=(32, 32, 16)` a 128 by 128 tile rather than a 32 by 32 tile?
> `(2, 2, 1)` places four warps. The permutation is the tile one warp iterates, 32 by 32. The CTA tile passed to `local_tile` is 128 by 128, and the partition's rest mode covers the four steps from 32 to 128 along each side. The $K$ permutation of 16 is one instruction, and a $K$ tile of 64 is four of those.

> [!QUESTION] GELU of 1.5, or GELU of 1 plus 0.5. Which one is the fused epilogue, and what are the two numbers?
> The fused epilogue adds first. $\mathrm{GELU}(1.5) = 1.399789$. $\mathrm{GELU}(1) + 0.5 = 1.341345$. The second number is `gelu_then_bias`.

> [!QUESTION] A 128 by 128 fp16 tile. How many extra bytes does the unfused bias-plus-GELU pair move, and how many does the fused epilogue add?
> The bias kernel moves $66{,}048$ bytes and the GELU kernel moves $65{,}536$, together $131{,}584$. The fused epilogue adds the 512-byte fp32 bias read. The ratio is 257.

> [!QUESTION] Is 0.4405 ms a measurement of this kernel?
> It is the roofline floor: $2 \times 4096^3$ FLOP divided by the A100 80GB PCIe's rated 312 TFLOP/s. The intensity, 1365.3 FLOP per byte, is on the compute side of the 161.2 ridge. The lab measures the real TFLOP/s. A one-stage kernel has no obligation to land on the floor, and the lab records the number without a pass/fail threshold.

## Lab

> [!TRY] Compile the algebra
> Open `24-cute-dsl-gemm`. `tensor_offsets` is the one function the first check needs: the colexicographic offsets of a layout, the same rule as lab 13. Then implement `partition_tiled_copy`, `accumulator_mn`, `bias_gelu`, `epilogue_bytes`, and `predicted_ms` on the CPU. The partition check compares your thread-value layout with an independent copy of the lab 13 algebra. The last function, `cute_gemm_bias_gelu`, is the kernel: fp16 `(M, K)` times `(K, N)`, plus an fp32 bias, returning fp16 $\mathrm{GELU}(AB + \text{bias})$. Build a TiledMMA and a TiledCopy, call `cute.copy` and `cute.gemm`, and apply the epilogue before the global store. Shapes that miss the 128 by 128 by 64 tile get zero padding. The kernel checks run on the A100 and skip, with a message, on a machine without a CUDA GPU. Passing on the CPU means the six layout and arithmetic checks match. Passing on the A100 means the kernel matches `torch.nn.functional.gelu` of the fp32 product within $2 \times 10^{-2}$ absolute error, and the run records `tflops`.

The lab reports one metric:

- `tflops`: useful FLOP of the timed shape, $2MNK$, divided by the measured seconds, for a 2048 cubed GEMM. The floor for that shape is $2 \times 2048^3 / 312 \times 10^{12} = 0.0551$ ms. The metric is a record, not a threshold. A one-stage kernel and a heavily tuned one can both be correct.

## Further reading

- [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html), NVIDIA. The programming model this chapter compiles: tensors, layouts, atoms, and tiled operations.
- [Warp-level MMA instructions](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wmma_programming.html), NVIDIA CUTLASS 4.5.2. `MmaF16BF16Op`, `make_tiled_mma`, `cute.copy`, and `cute.gemm` for `mma.sync` on Ampere.
- [The CUTLASS repository](https://github.com/NVIDIA/cutlass), NVIDIA. The Ampere CuTe DSL examples under `examples/python/CuTeDSL` are the long form of the kernel in the lab.
- [GPU MODE, lecture 15: CUTLASS](https://github.com/gpu-mode/lectures#lecture-15-cutlass), Eric Auld. The lecture list's CUTLASS session.
- [GPU MODE, lecture 57: CuTE](https://github.com/gpu-mode/lectures#lecture-57-cute), Cris Cecka. The layout algebra, from the list of lecture titles.
- [GPU MODE, lecture 86: Introduction to CuTeDSL (for NVIDIA competition)](https://github.com/gpu-mode/lectures/tree/main/lecture_086), Vicki Wang. The DSL session, with the slide PDF in that folder.
- [GPU MODE, lecture 103: Fundamentals of CuTe Layout Algebra and Category-theoretic Interpretation](https://github.com/gpu-mode/lectures#lecture-103-fundamentals-of-cute-layout-algebra-and-category-theoretic-interpretation), Jack Carlisle and Jay Shah. The algebra chapter 13 implements, as a lecture.
- [GPU MODE, lecture 114: PyCuTe](https://github.com/gpu-mode/lectures#lecture-114-pycute), Cris Cecka. PyCuTe is Cecka's Python library at [NVlabs/CuTe](https://github.com/NVlabs/CuTe). It is the algebra in Python. The kernel in this chapter uses NVIDIA's CuTe DSL, which is a different package.
- [Chapter 13](/c/13-cutlass-and-cute), for the layout algebra, and [chapter 12](/c/12-pipelining-and-async-copies), for the multistage loop this kernel leaves as one stage. [Chapter 14](/c/14-triton) is the Triton GEMM, if the atom is allowed to be the compiler's choice.
