---
title: Quantized kernels
slug: 18-quantized-kernels
part: "Part 4 — Kernels for ML"
summary: Why fewer bits make decode faster, where the scales live in a GEMM, how int4 weights are packed and unpacked at speed, and how a fused dequantize-GEMM in Triton turns a 3.8x byte saving into a 3.8x time saving. INT8, FP8, MX, and NVFP4 tensor cores in prose.
minutes: 150
gpu: true
objectives:
  - "Derive the byte count and the roofline time of a decode GEMV in fp16 and in group-quantized int4, and say why the speedup tracks the byte ratio."
  - "Compute the batch size at which a W4A16 kernel stops being memory bound, for the A100's measured copy ceiling."
  - "Write the affine dequantization map with its scale and zero point, and say which granularity of scale can leave the inner sum over K."
  - "Pack two int4 codes per byte, unpack them in a kernel, and explain the magic-number fp16 trick and the lop3 instruction behind it."
  - "Structure a fused W4A16 dequantize-GEMM in Triton, with the factored scale and the even and odd nibble split."
  - "Explain what INT8 tensor cores compute on the A100, why the dequantize happens in the epilogue, and why a group scale breaks it."
  - "Describe FP8 on Hopper, block scaling in MXFP8 and MXFP4, and NVFP4 on Blackwell, and say why the A100 has none of them."
lab: 18-w4a16
---

# Quantized kernels

> [!TLDR]
> - Decode reads every weight once and does two FLOPs on it, so its time is the weight bytes over the bandwidth. Storing weights in 4 bits plus a share of a scale and a zero point cuts those bytes by 3.82x for an 8192 x 8192 matrix, and the roofline predicts the same factor in time.
> - A quantized weight is a small integer plus a scale shared by a group. The scale can leave the sum over K only when it doesn't depend on K, which is why per-group scales suit weight-only kernels and force a dequantize epilogue in INT8 GEMMs.
> - A fused W4A16 kernel reads packed bytes, converts nibbles to fp16 in registers with a handful of instructions, and feeds the fp16 tensor cores. The unpack must fit in a budget of about 4 to 8 instructions per weight, which is why kernels use bit tricks such as the lop3 magic number.
> - On the A100, the 4-bit kernel turns compute bound at about 64 rows, against 245 for fp16, so the gain holds for decode and small batches and fades beyond. Prefill needs lower-precision arithmetic, and fewer bytes alone don't help.
> - INT8 tensor cores run at 624 TOPS on the A100. FP8 arrives with Hopper, and block-scaled MXFP8, MXFP4, and NVFP4 with Blackwell. The A100 has no FP8 or FP4 tensor cores.

A decode step generates one token per sequence. It multiplies a tiny activation matrix by every weight matrix in the model, and the weights are read from HBM each time. The arithmetic per byte is so low that the tensor cores wait. A kernel that reads fewer bytes finishes sooner.

That makes quantization the most direct speedup for serving. The companion course derives the numerics in its [quantization chapter](https://qwen.ifkash.dev/c/18-quantization): the error bound, per-tensor against per-group scaling, and why int4 needs smarter rounding than the nearest level. This chapter is about the kernel side. How do you store 4-bit weights so a GPU can read them quickly, and how do you turn them back into numbers fast enough that the unpacking never becomes the bottleneck?

By the end you can predict how fast a W4A16 kernel should run, write one in Triton, and say where the approach stops working. You also have a map of the low-precision tensor cores on Hopper and Blackwell, which the A100 lab can't run but the same ideas drive.

## Before you start

**Decode and prefill, from [the roofline chapter](/c/04-roofline-and-latency).** A kernel's *arithmetic intensity* is its FLOPs per byte moved. The A100 80GB PCIe has a ridge point of 161.2 FLOP per byte against the rated 1935 GB/s, and 244.7 against the 1275 GB/s that the companion course measured for a device-to-device copy. Below the ridge point a kernel waits on memory, and above it a kernel waits on arithmetic. This chapter uses 244.7, because a kernel is judged against what the card delivers.

**Tensor cores, from [the tensor cores chapter](/c/11-tensor-cores).** A tensor core multiplies small tiles. On the A100, the fp16 instruction `mma.sync.m16n8k16` computes a 16 x 8 output tile from a 16 x 16 tile of A and a 16 x 8 tile of B in one warp-wide operation. Operands come from registers, and the accumulator is fp32. Triton's `tl.dot` compiles to these instructions.

**Affine quantization, from the companion course.** A real weight $w$ is stored as an integer code $q$, a *scale* $s$ that says what one integer step is worth, and, for an asymmetric format, a *zero point* $z$ that says which code stands for 0.0. Dequantizing is $\hat{w} = s\,(q - z)$.

**Triton, from [the Triton chapter](/c/14-triton).** A Triton program computes one tile of the output. You write the tile's pointer arithmetic with `tl.arange`, load with masks, accumulate with `tl.dot`, and store.

## Why bytes set the speed

A decode GEMV is limited by how fast the weights stream from HBM, so its time is the weight bytes divided by the bandwidth. This section works one matrix through, and everything later builds on its numbers.

Take a weight matrix with $N = 8192$ output features and $K = 8192$ input features, which is 67,108,864 weights. In fp16 it takes $2 \times 67{,}108{,}864 = 134{,}217{,}728$ bytes. Against the 1275 GB/s copy ceiling, a pass over it takes $134.2 \times 10^{6} / 1.275 \times 10^{12} = 105.3\ \mu s$. The matrix does $2 \times 67.1 \times 10^6 = 134$ MFLOP per row of the activation, which at 312 TFLOP/s is 0.43 $\mu s$, so a single row finishes its arithmetic hundreds of times faster than the bytes arrive.

Now store the same matrix in the format the lab uses: 4-bit codes, an fp16 scale and an 8-bit zero point for each group of 128 weights along $K$. The byte count follows:

| Part | Count | Bytes |
|---|---|---|
| Packed codes, two per byte | $67{,}108{,}864 / 2$ | 33,554,432 |
| fp16 scales, one per group | $8192 \times 64 = 524{,}288$ | 1,048,576 |
| uint8 zero points, one per group | 524,288 | 524,288 |
| Total | | 35,127,296 |

The ratio against fp16 is $134{,}217{,}728 / 35{,}127{,}296 = 3.82$. Per weight, the format costs $4 + (16 + 8)/128 = 4.1875$ bits. At 1275 GB/s the pass takes $35.13 \times 10^6 / 1.275 \times 10^{12} = 27.6\ \mu s$.

> [!KEY] The bytes ratio is the speedup you can ask for
> A memory-bound kernel's time is its bytes over the bandwidth, so a format that moves 3.82x fewer bytes can run at most 3.82x faster. The scales and zero points are the whole gap from a clean 4x. Your lab reports this ratio as `bytes_ratio` and measures how close the kernel gets.

A symmetric format drops the zero point and costs $4 + 16/128 = 4.125$ bits, a ratio of 3.88. That matches the Marlin repository's statement that the ceiling is 3.87x, with 0.125 bits of group scale per weight (the README reports its peak speedup on an NVIDIA A10; the paper's abstract doesn't name a GPU). The extra byte for the zero point in the lab layout is a choice, and the next sections explain why.

The same arithmetic scales to a whole model. The companion course's 27B-parameter model reads 53.8 GB of bfloat16 weights per decode step, a floor of 42.2 ms at 1275 GB/s. In int4 with group size 128 and an fp16 scale, the weights are 13.9 GB and the floor is 10.9 ms. Quantizing doesn't change the model's FLOPs. It changes how long the memory system takes to deliver them.

## Weight-only against weight-and-activation

A quantized GEMM can store weights in low precision and still multiply in fp16, or it can reduce both operands and use a low-precision tensor core. The two answer different bottlenecks, so name them before writing any kernel.

- **Weight-only, such as W4A16.** Weights are int4, activations are fp16. The kernel reads packed weights, dequantizes them to fp16 in registers, and does an fp16 tensor-core multiply. The arithmetic is the same as the fp16 kernel, so it can only win where memory is the bottleneck: decode and small batches.
- **Weight-and-activation, such as W8A8.** Both operands are int8, the multiply uses the integer tensor cores, and the accumulator is int32. It also wins where arithmetic is the bottleneck, because the A100's INT8 rate of 624 TOPS is twice its fp16 rate of 312 TFLOP/s. The price is that activations must be quantized on the fly, and activations have outliers that weights don't.

For a reader serving a model, the split is simple. Decode is memory bound at low batch, so weight-only is the tool. Prefill is compute bound, so only a lower-precision multiply helps, and that is where INT8 on the A100, FP8 on Hopper, and FP4 on Blackwell come in. The later sections cover both.

## Scales and zero points

A scale and a zero point decide how a code maps to a weight, and where they sit decides what the kernel can do cheaply. The section answers two questions: how fine should the scale's granularity be, and where does it enter the matrix product?

### The map and its two flavors

The lab quantizes asymmetrically to unsigned 4-bit codes. Given a group's minimum $\beta_{\min}$ and maximum $\beta_{\max}$ over the real values, the scale and zero point are:

$$
\hla{s} = \frac{\beta_{\max} - \beta_{\min}}{15},
\qquad
\hlb{z} = \operatorname{round}\!\left(-\frac{\beta_{\min}}{\hla{s}}\right),
\qquad
\hat{w} = \hla{s}\,(q - \hlb{z})
$$

Here $\hla{s}$ is the real width of one code step and $\hlb{z}$ is the code that decodes to exactly 0.0. The lab always includes 0 in the range, so $\beta_{\min} \le 0 \le \beta_{\max}$ and $\hlb{z}$ lands in $0 \ldots 15$.

A *symmetric* format fixes $\hlb{z}$ in the middle, for example $z = 8$ for unsigned codes, which is the same as storing signed codes $-8 \ldots 7$ with no zero point at all. It saves the zero-point storage and one subtraction per weight. It wastes a level when the group's values aren't centered, because the range must stretch to cover the larger side. For weights, which sit close to symmetric around zero, the loss is small, which is why many int4 formats choose it. The lab uses the asymmetric form because the zero point is the interesting part of the kernel.

> [!EXAMPLE] The error you should expect from round-to-nearest
> A group of 128 normal weights has a range of about $5.2\sigma$, so the step is $5.2\sigma / 15 = 0.35\sigma$. Rounding to the nearest level leaves a residual uniform in $\pm s/2$, whose RMS is $s/\sqrt{12}$, so the relative error is $0.35/\sqrt{12} \approx 10\%$. The lab's quantizer test allows 15%.
>
> That is large. It is also what round-to-nearest gives at 4 bits, and it is the reason methods such as GPTQ and AWQ choose the codes using the layer's outputs instead. The kernel doesn't care how the codes were chosen.

### Granularity

The scale's granularity is the set of weights that share one scale. Finer scales follow the weights more closely and cost more metadata. The following table lists the options for a weight matrix of shape $(N, K)$ in the order the formats get finer:

| Granularity | Weights per scale | Typical use | Bits per weight at 4-bit codes |
|---|---|---|---|
| Per tensor | $N K$ | FP8 training recipes, simple INT8 | $4 + \varepsilon$ |
| Per channel (per output row) | $K$ | INT8 weights | $4 + 16/K$ |
| Per group along $K$ | 128 | int4 weights (GPTQ, AWQ, Marlin) | 4.125 symmetric, 4.1875 in the lab |
| Block (MX) | 32 | MXFP4 with an E8M0 scale | 4.25 |
| Block (NVFP4) | 16 | NVFP4 with an FP8 E4M3 scale | 4.5 |

The last two rows come from the OCP microscaling specification and NVIDIA's NVFP4 description, and the per-element cost counts the scale only. NVFP4 also carries one FP32 scale per tensor, which the table doesn't count. The next figure shows what finer scales buy.

```viz
18-group-scales
```

A single outlier sets the range of its own group and nothing else, which is the whole argument for grouping. With the outlier switched on, a group of 256 stretches every step, and a group of 128 or less confines the damage to one group at a price of a few hundredths of a bit per weight.

### Where the scale enters the matrix product

For an output element $Y_{mn} = \sum_k X_{mk}\,\hat{W}_{nk}$, substitute the dequantization map for a weight in group $g$ of row $n$:

$$
Y_{mn} = \sum_{g} \hla{s_{ng}} \left( \sum_{k \in g} X_{mk}\, q_{nk} \;-\; \hlb{z_{ng}} \sum_{k \in g} X_{mk} \right)
$$

Inside a group, the scale and the zero point are constants, so they leave the inner sum. The inner sums are $\sum_k X_{mk} q_{nk}$, which is a plain matrix product between the activations and the integer codes, and $\sum_k X_{mk}$, which is one number per activation row and group. A kernel can therefore feed the codes, converted to fp16, straight to the tensor cores and fix up each group's partial result with two scalars.

> [!INTUITION]
> You don't have to dequantize every weight to use the tensor core. The tensor core computes $\sum_k x_k q_k$ on raw codes. A correction of $-z \sum_k x_k$ and a multiply by $s$ then turn that partial result into the same number you'd get from fp16 weights. The correction costs one multiply-add per output element per group, not one per weight.

This factored form is why the group size matters to the kernel as well as to quality. The scale can leave the sum over $k$ only inside a group. If the scale were per tensor or per row, it would leave the whole sum, and the fix-up would happen once in the epilogue. If it varies along $K$, the kernel must apply it every time $k$ crosses a group boundary, which the lab's loop does once per iteration because its tile along $K$ is exactly one group.

## Packing two codes per byte

A 4-bit code is smaller than the smallest addressable unit, so the kernel needs a layout. The section answers what goes in each byte and why.

The lab packs along $K$. Byte $j$ of a row holds code $2j$ in its low nibble and code $2j+1$ in its high nibble:

```python
packed = codes[:, 0::2] | (codes[:, 1::2] << 4)   # (N, K // 2), uint8
low = packed & 0xF                                  # codes 0, 2, 4, ...
high = packed >> 4                                  # codes 1, 3, 5, ...
```

A row of $K = 8192$ weights is then $4096$ contiguous bytes, which a GPU reads as 256 loads of 16 bytes. Each 16-byte load, which is the widest a single thread can issue, brings 32 weights. Reading a row this way is fully coalesced, the same access pattern as a copy.

Packing along $K$ fits the product. Output element $Y_{mn}$ sums over $k$, so a thread that owns a piece of one weight row needs consecutive values of $k$ for that row and the matching activations. A layout that packed pairs along $N$ would put two different output columns' weights in one byte, and one thread would need two accumulators for what is naturally one stream.

There is one catch. After the unpack, the low nibbles hold the even-indexed weights and the high nibbles hold the odd-indexed ones, so the two halves aren't in $k$ order. The lab handles that by loading the activations at even and odd columns, so each half multiplies the right activations. Production kernels instead permute the weights offline, when they quantize, so that the unpacked order is the order the tensor core's fragment layout wants. Marlin's README says it reshuffles weights and scales offline into a layout that suits tensor-core execution.

## Unpacking fast

Turning a nibble into an fp16 number is arithmetic that happens for every weight, so its cost has to fit in the memory time. The section answers how many instructions per weight you can afford and how to stay inside that.

### The instruction budget

The kernel receives weights at the memory rate. At 1275 GB/s and 0.5234 bytes per weight (4.1875 bits), that is $1275 \times 10^9 / 0.5234 = 2.44 \times 10^{12}$ weights per second. The A100's four warp schedulers per SM can each issue one warp instruction per clock. Across 108 SMs at 1410 MHz, that is $4 \times 32 \times 108 \times 1.41 \times 10^9 = 19.5 \times 10^{12}$ thread instructions per second, or $19.5 / 2.44 = 8.0$ instruction slots per weight. If every instruction has to use the 64 FP32 lanes per SM, the rate halves to 4.0 per weight.

So the unpack, the scaling, and every load, shift, and address calculation share a budget of 4 to 8 instructions per weight. A plain dequantizer uses about five by a count of its steps. It masks or shifts to get the nibble, converts the integer to a float, subtracts the zero point, multiplies by the scale, and converts to fp16. That already fills the budget, and the kernel still has to feed the tensor cores. Dequantization is not free after all, and the more you cut, the closer the kernel gets to the memory rate.

### The factored form removes most of it

The factored form from the previous section takes the subtract and the multiply out of the per-weight path. The loop only converts a nibble to fp16: one mask or shift and one integer-to-fp16 convert, about two instructions per weight. The subtract and multiply happen per output element and per group, which is a small share of the work when a tile has $BLOCK\_N$ output columns and 128 weights per column in a group.

### The magic number

Even a convert is more than a bit trick needs. In fp16, the value 1024 has the bit pattern `0x6400`: sign 0, exponent field 25, mantissa 0. At that magnitude one unit in the last place equals 1, because the 10 mantissa bits span $1024$ to $2047$ in steps of 1. So OR a code $q < 1024$ into the mantissa, and the 16 bits `0x6400 | q` mean exactly $1024 + q$.

```python
bits = codes.to(torch.int16) | 0x6400       # 1024 + q, as raw fp16 bits
fp16 = bits.view(torch.float16) - 1024.0    # q, exactly
```

An OR and a subtract replace an integer-to-float conversion, and the subtraction can absorb the zero point: subtract $1024 + z$ and you have $q - z$ in one step. Your lab asks for `int4_to_fp16_magic`, a function that does this in torch, and its test checks that all 16 codes convert exactly.

The GPU makes this cheaper still. A 32-bit register holds two fp16 values, and the `lop3` instruction computes any logic function of three 32-bit inputs in one instruction. The mask-and-OR, $(a \,\&\, b) \,|\, c$, is one `lop3`, and its lookup-table immediate is `(0xf0 & 0xcc) | 0xaa = 0xea`: the three inputs are assigned the constant bit patterns `0xf0`, `0xcc`, and `0xaa`, and the immediate is the same expression evaluated on them. One `lop3` therefore prepares two weights.

Marlin's dequantization routine uses this idea. Given a 32-bit word of eight nibbles, it takes the nibbles at bits 0 to 3 and 16 to 19 with the mask `0x000f000f`, ORs in `0x64006400` with one `lop3`, and subtracts a packed constant of 1032 (that is, $1024 + 8$) with one `__hsub2`. The result is two signed int4 values, $q - 8$, in fp16. The nibbles at bits 4 to 7 and 20 to 23 get the mask `0x00f000f0`, which leaves $1024 + 16q$. One `__hfma2` multiplies by 1/16 and adds $-72$, which gives $64 + q - 72 = q - 8$. Two instructions per two weights, either way, which is about one instruction per weight before the scale.

> [!NOTE] You can't write lop3 in a Triton tile
> Triton exposes bitwise operations on whole tiles, and it chooses the instructions. You can write the OR-and-subtract trick with `to(tl.int16)`, an OR, and `.to(tl.float16, bitcast=True)`, and the compiler may or may not turn a pair of values into one `lop3`. `tl.inline_asm_elementwise` lets you write the PTX yourself. The lab's solution uses the plain convert, and an optional extension is to compare it with the bit-trick version.

## The fused kernel

A fused W4A16 kernel reads the packed bytes, dequantizes in registers, and multiplies, without ever writing an fp16 copy of the weights. This section builds the structure, the parallelism for a tiny $M$, and the pipeline, and says why each piece reaches the memory rate.

### Why fused

A kernel that first dequantizes the whole matrix to fp16 and then calls a GEMM does three passes over memory. It reads 35.1 MB of codes, writes 134.2 MB of fp16 weights, and reads those 134.2 MB again. That is $35.1 + 134.2 + 134.2 = 303.5$ MB, against 134.2 MB for the plain fp16 GEMM, so it is 2.26x slower than not quantizing at all. Fusing keeps the dequantized values in registers, so the only traffic is the 35.1 MB of the packed format.

### The loop

Each program computes a tile of $Y$ with `BLOCK_M` rows and `BLOCK_N` columns. Its loop runs once per group of 128 along $K$. The excerpt below is the heart of the lab's solution:

```python
for g in range(0, K // GROUP):
    x_e = tl.load(x_even_ptrs + g * GROUP * stride_xk, mask=m_mask[:, None], other=0.0)
    x_o = tl.load(x_odd_ptrs + g * GROUP * stride_xk, mask=m_mask[:, None], other=0.0)
    packed = tl.load(w_ptrs + g * HALF, mask=n_mask[:, None], other=0).to(tl.int32)
    scale = tl.load(s_ptr + offs_n * stride_gn + g, mask=n_mask, other=0.0).to(tl.float32)
    zero = tl.load(z_ptr + offs_n * stride_gn + g, mask=n_mask, other=0).to(tl.float32)

    q_lo = tl.trans((packed & 0xF).to(tl.float16))   # (64, BLOCK_N)
    q_hi = tl.trans((packed >> 4).to(tl.float16))

    part = tl.dot(x_e, q_lo)
    part = tl.dot(x_o, q_hi, part)                   # sum over the group of x_k * q_k
    x_sum = tl.sum(x_e.to(tl.float32), axis=1) + tl.sum(x_o.to(tl.float32), axis=1)
    acc += (part - x_sum[:, None] * zero[None, :]) * scale[None, :]
```

Read it in four steps. First, the loads: `x_e` and `x_o` are the activations at even and odd columns of the group, and `packed` is a tile of 64 bytes per row, 128 weights. Second, two `tl.dot` calls multiply fp16 codes, which are exact in fp16, with fp16 activations. Each accumulates in fp32. Third, `x_sum` is the sum of the group's activations, and `zero` is the zero point for this group and column. Fourth, the fix-up from the previous section converts the partial sum into the right contribution of this group.

Every load carries a mask, so $M$ and $N$ can be any size. $K$ must be a multiple of 128, because a group is the unit of the loop. With $M = 1$, `BLOCK_M` is still 16, the smallest tile `tl.dot` accepts, and the masked rows load zeros. The kernel multiplies 16 rows where one is real. At $M = 16$ the padded tensor-core work for the 8192 x 8192 matrix is $2 \times 16 \times 8192^2 / 312 \times 10^{12} = 6.9\ \mu s$, a quarter of the 27.6 $\mu s$ memory time, so the padding is hidden behind the loads.

### Parallelism at tiny M

At $M \le 16$ the grid has one row of programs, $N / BLOCK\_N$ of them. With $BLOCK\_N = 16$ that is 512 programs for $N = 8192$, about 4.7 per SM on 108 SMs. A larger tile gives fewer programs, and with $BLOCK\_N = 128$ there are 64, fewer than the 108 SMs, so some SMs sit idle and the kernel can't use the full bandwidth. That is why the lab's config list includes small `BLOCK_N` values, and why production kernels also split $K$ across programs and add the partial sums with atomics or a second pass.

Memory-level parallelism sets a second requirement. Little's law says bytes in flight equal bandwidth times latency. With the 466.3-cycle HBM latency that the Luo microbenchmark paper measured on an A100 PCIe 40GB, which is 330.7 ns at 1410 MHz, 1275 GB/s needs $1275 \times 10^9 \times 330.7 \times 10^{-9} = 422$ KB in flight across the GPU, or about 3.9 KB per SM. Each program must therefore keep several tiles of bytes in flight, which is what `num_stages` controls: Triton issues the loads for later iterations while the current one computes.

### What Marlin adds

Marlin is the best-known fused W4A16 kernel. Its paper reports speedups close to the 4x maximum up to batch sizes of 16 to 32, a gradual decline at batch sizes of 64 to 128, and up to 2.8x end to end in vLLM. The paper's abstract doesn't name the GPU, and the README reports its peak microbenchmark on an NVIDIA A10 and states that the kernel targets compute capability 8.0 and later, so treat the numbers as "reported by the authors" and don't transfer absolute figures to the A100.

The README lists the techniques that get a kernel from "correct" to "near the memory rate":

- Activations come from L2 and are reused from registers across the weight tiles.
- Weight loads use asynchronous copies with a cache hint that marks the data as evictable, because each weight is read once.
- Shared-memory loads are double-buffered to overlap with the tensor-core work.
- Weights and scales are reshuffled offline into the order the tensor cores consume, so no shuffling happens at run time.
- Several warps of a block work on partial sums of the same output tile, and a striped partitioning spreads each SM's share of the work across column slices.
- A global reduction happens in the output buffer, using fp16 downcasts of fp32 accumulators that stay in L2.

Each item answers something this chapter derived. The cache hints and async copies keep the bytes in flight high. The offline reshuffle removes shuffles from the instruction budget. The striped partition and the reduction address the grid-size problem at tiny $M$.

## Where the gain ends

The byte saving helps only while the kernel stays memory bound. The section answers at which batch size the 4-bit kernel stops being faster than fp16.

For a weight matrix $(N, K)$ and $M$ activation rows, the FLOPs are $2MNK$ and the weight bytes are $b\,NK$, where $b$ is bytes per weight. The intensity is therefore $2M/b$ FLOP per byte, ignoring the small activation traffic. A kernel is compute bound once that exceeds the ridge point $I^{\ast}$:

$$
\boxed{M^{\ast} = \frac{\hla{I^{\ast}}\, \hlb{b}}{2}}
$$

For fp16 weights, $b = 2$, so $M^{\ast} = I^{\ast} = 244.7$. For the lab's int4 layout, $b = 0.5234$, so $M^{\ast} = 244.7 \times 0.5234 / 2 = 64.0$. The 4-bit kernel runs out of memory-bound headroom 3.8x sooner, because each weight byte now carries 3.8x more arithmetic.

> [!EXAMPLE] Time at M of 64, 128, and 256 for the 8192 x 8192 matrix
> The roofline time is the larger of the bytes over 1275 GB/s and the FLOPs over 312 TFLOP/s, with activations and outputs counted as $2MK + 2MN$ bytes.
>
> | M | fp16 | W4A16 | Predicted speedup |
> |---|---|---|---|
> | 64 | 106.9 $\mu s$ (memory) | 29.2 $\mu s$ (memory) | 3.66x |
> | 128 | 108.6 $\mu s$ (memory) | 55.1 $\mu s$ (compute) | 1.97x |
> | 256 | 111.8 $\mu s$ (memory) | 110.1 $\mu s$ (compute) | 1.02x |
>
> At 128 rows, the FLOPs take $2 \times 128 \times 8192^2 / 312 \times 10^{12} = 55.1\ \mu s$. At 256 they take $110.1\ \mu s$, which is nearly the fp16 memory time, so the two kernels converge.

```viz
18-w4a16-roofline
```

This is a prediction, and the lab measures the real curve. It ignores two real costs: the dequantization instructions, which can leave the kernel compute bound by the CUDA cores before the tensor cores saturate, and the tensor-core padding at small $M$. It agrees in shape with Marlin's reported curve, which is near the maximum to batch 16 or 32 and falls gradually beyond.

The practical reading is a rule. W4A16 is the right kernel for decode and for small-batch serving. For prefill, where $M$ is thousands of rows, it moves the same FLOPs and adds unpack work, so the kernel doesn't win. A serving stack picks the kernel by batch size. Reaching a lower arithmetic cost needs a lower-precision multiply, which is the subject of the next two sections.

## INT8 tensor cores

An INT8 tensor core multiplies int8 tiles and accumulates into int32, at twice the fp16 rate on the A100. This section covers what it computes and why its scale handling differs from the weight-only case.

The A100's datasheet rates the INT8 tensor core at 624 TOPS dense. The per-SM arithmetic matches: each SM does 4096 INT8 operations per clock, so $108 \times 4096 \times 1.41 \times 10^9 = 623.7$ TOPS. The warp-level instruction is `mma.sync.m16n8k32` with `s8` operands, which does $16 \times 8 \times 32 = 4096$ multiply-adds, or 8192 operations. On an A100 PCIe 40GB, the Luo microbenchmark paper measured 607.6 TOPS for it, 97% of the rating. The ridge point rises with the rate: $624 / 1.935 = 322.5$ INT8 operations per byte against the rated bandwidth.

To use the INT8 core, the whole product must be integer. Write the activation as $x_k = \hla{s_x} q_{x,k}$ and the weight as $W_{nk} = \hlb{s_{w,n}} q_{w,nk}$, with one scale per activation tensor (or row) and one per output channel. Both scales are independent of $k$, so they leave the sum:

$$
Y_{mn} = \hla{s_x}\, \hlb{s_{w,n}} \sum_{k} q_{x,mk}\, q_{w,nk}
$$

The sum is an integer matrix product, which the tensor core computes into int32. The kernel's *dequantize epilogue* then multiplies each accumulator by $\hla{s_x}\hlb{s_{w,n}}$ once, converts to fp16, and stores. That is one float multiply per output element, with no per-weight work in the loop at all.

The accumulator has room. Each product is at most $127 \times 127 = 16{,}129$, so with $K = 8192$ the worst-case sum is $8192 \times 16{,}129 = 132{,}128{,}768$, against an int32 range of $2^{31} = 2{,}147{,}483{,}648$. That is 16.3x of headroom, so the int32 accumulator can't overflow at this size.

> [!WARNING] A group scale breaks the integer inner loop
> A group scale depends on $k$, so it can't leave the sum over $K$. The kernel would have to convert the int32 partial sum to float, scale it, and add it every 128 steps. That's possible, and it is the same fix-up this chapter does in the W4A16 loop, but it adds float work to a loop whose entire point was integer throughput. W8A8 kernels therefore use per-tensor or per-channel scales, and the outlier handling of LLM.int8() and SmoothQuant exists because those coarse scales are fragile on activations.

## FP8 on Hopper

Hopper adds FP8 tensor cores, and FP8 changes the scaling problem because the format is floating point. This section is prose. The A100 has no FP8 tensor cores: FP8 `mma.sync` starts at compute capability 8.9 and `wgmma` at 9.0, and `torch._scaled_mm` fails on the lab card for that reason.

Two FP8 formats exist. E4M3 has 4 exponent bits and 3 mantissa bits, a maximum of 448 and no infinity, and it suits weights and activations. E5M2 has 5 exponent bits and 2 mantissa bits, a maximum of 57,344, and its wider range suits gradients. The OCP FP8 specification defines both, and the Micikevicius paper describes how to use them in training.

FP8 spends bits on an exponent, so each value has the same relative precision: 8 steps per octave for E4M3, a worst-case relative rounding error of $2^{-4} = 6.25\%$. An outlier costs an exponent increment and not the mantissa of its neighbors. The range is still narrow, so a tensor needs a scale to fit inside 448. A per-tensor scale is $\text{amax} / 448$, and the kernel multiplies the matmul result by the product of the two scales in the epilogue, as in the INT8 case.

On an H100 SXM the FP8 tensor core is rated at 1978.9 TFLOP/s dense, twice the 989.4 of bfloat16, and its ridge point against 3.35 TB/s is $1978.9 / 3.35 = 590.7$ FLOP per byte. The Luo paper measured 1448 of 1513 TFLOP/s on an H800 PCIe, 96% of that card's rating, with `wgmma`.

### Fine-grained scaling

A per-tensor scale has the outlier problem that per-tensor int8 has. The DeepSeek-V3 report, which trained a 671B-parameter model in FP8 on H800 GPUs, uses finer scales. Activations get one scale per 1 x 128 tile, so per token and per 128 channels. Weights get one scale per 128 x 128 block. It uses E4M3 for all tensors.

The report also names a hardware limit: FP8 GEMM accumulation on the H800 retains only about 14 bits. Its fix is to copy partial results to the CUDA cores for fp32 accumulation every 128 elements of $K$, which the authors say equals 4 `wgmma` instructions. The scale granularity and the promotion interval are linked. The two line up: one scale covers each 128-element segment between promotions, so the kernel can apply the scale during the same fp32 accumulation.

That is the same shape as the W4A16 loop: a low-precision dot over one group, then a scaled fp32 accumulate.

## Block-scaled formats on Blackwell

Blackwell takes the group scale into the hardware. The *microscaling* (MX) formats from the OCP standard attach one 8-bit power-of-two scale, type E8M0, to each block of 32 elements. The following table gives the element formats and their cost, counting the scale:

| Format | Block | Scale | Bits per element, with scale |
|---|---|---|---|
| MXFP8 | 32 | E8M0 | 8 + 8/32 = 8.25 |
| MXFP6 | 32 | E8M0 | 6 + 8/32 = 6.25 |
| MXFP4 (E2M1) | 32 | E8M0 | 4 + 8/32 = 4.25 |
| NVFP4 (E2M1) | 16 | FP8 E4M3, plus one FP32 per tensor | 4 + 8/16 = 4.5 |

The FP4 element, E2M1, has one sign bit, 2 exponent bits, and 1 mantissa bit. Its positive values are 0, 0.5, 1, 1.5, 2, 3, 4, and 6, so the block scale does nearly all the work of fitting the data. NVFP4 uses a smaller block of 16 and a scale with a mantissa, so its scales are finer than MXFP4's, at 0.25 bits more per element.

The fifth-generation tensor core on Blackwell (compute capability 10.0) issues these through `tcgen05.mma`, which has kinds for the block-scaled types (`mxf8f6f4`, `mxf4`, and `mxf4nvf4`), and the instruction takes the block scales as operands. The group-scale fix-up that your W4A16 loop does with fp32 FMAs is part of the instruction. Hardware that applies the scale is what makes 16-element blocks affordable.

The rates are rated per GPU in an 8-GPU HGX B200 at 1,000 W: 4,500 TFLOP/s dense for FP8 and 9,000 for FP4, against 8 TB/s of bandwidth. The ridge points are $4500 / 8 = 562.5$ and $9000 / 8 = 1125$ FLOP per byte. NVIDIA's architecture blog lists 5 and 10 PFLOP/s dense for FP8 and NVFP4 on a higher-power SKU, so the ridge points depend on which part you have.

In Triton, `tl.dot_scaled` exposes block-scaled products with E8M0 scales on a group of 32 and the formats E2M1, E4M3, E5M2, bf16, and fp16. It is experimental. PyTorch 2.14 describes `torch.float4_e2m1fn_x2`, which stores two 4-bit values per byte, as a shell dtype with limited operator support. You can read it as the same packing as the lab's, standardized.

## What goes wrong

- **Packing along the wrong axis.** If the packed pairs lie along $N$, one thread needs two weight rows' worth of accumulators, and the activation reuse the kernel relies on disappears. The lab's test checks the nibble order directly: code $2j$ low, code $2j+1$ high.
- **Forgetting that the nibbles are out of order.** After an unpack, the low nibbles are the even-indexed weights. Multiplying them by the first half of the activations is silently wrong and produces plausible-looking output with an error of order 1. Load the activations at even and odd columns, or permute the weights offline.
- **A tile of 16 rows that isn't masked.** `BLOCK_M` is 16 even when $M$ is 1. Rows 1 to 15 must load zeros and must not store. The lab tests $M = 1$, 3, and 17 to catch a missing mask.
- **A grid that is too small.** A large `BLOCK_N` at $M = 1$ launches fewer programs than the card has SMs. The kernel is correct and slow. Check the number of programs before you blame the unpack.
- **Counting only the codes.** Effective bandwidth counts codes, scales, and zero points. The lab's `weight_gbs_m1` divides all three tensors' bytes by the time, so it compares fairly with a copy at 1275 GB/s.
- **Timing with a warm L2.** The 4-bit tensors for an 8192 x 8192 matrix total 35.1 MB, which fits the 40 MiB L2. Without flushing the cache between calls, the kernel appears to beat the bandwidth. The lab calls `cuda_time(..., flush_l2=True)`.
- **Quantizing to the wrong zero.** If the range excludes 0, the zero point falls outside $0 \ldots 15$ and the clamp quietly shifts every code. The lab includes 0 in the range, and a test checks that 0.0 round-trips exactly.
- **Trusting the weight error.** A quantizer that passes the weight-error test can still hurt the model, because the error correlates with the activations. The companion course measures layer-output error, logit correlation, and perplexity, in that order of cost.

> [!RECAP]
> - Decode time is weight bytes over bandwidth. For an 8192 x 8192 matrix, int4 with group size 128, an fp16 scale, and a uint8 zero point takes 35.1 MB against 134.2 MB, a ratio of 3.82, and the roofline predicts the same ratio in time.
> - A scale can leave the sum over $K$ only inside a group, so a W4A16 kernel converts codes to fp16, runs the tensor core, and fixes up each group with $s\,(\text{partial} - z \sum x)$.
> - Packing two codes per byte along $K$ gives coalesced 16-byte loads, and the unpack has an instruction budget of 4 to 8 per weight. The factored form and the `0x6400` magic number keep it near 1 to 2.
> - Fusing keeps dequantized weights in registers. Dequantizing to an fp16 buffer first moves 303.5 MB and is slower than not quantizing.
> - The 4-bit kernel is memory bound up to about $M = 64$, against 245 for fp16. Prefill needs lower-precision arithmetic: INT8 on the A100, FP8 on Hopper, and block-scaled FP8, FP6, and FP4 on Blackwell.

## Check your understanding

> [!QUESTION] The lab's format costs 4.1875 bits per weight. Why is the byte ratio 3.82 and not 4?
> Each group of 128 weights stores an fp16 scale (16 bits) and a uint8 zero point (8 bits), which is $24/128 = 0.1875$ bits per weight on top of the 4-bit code. The fp16 weight is 16 bits, so the ratio is $16 / 4.1875 = 3.82$. A symmetric format without the zero point costs 4.125 bits and reaches 3.88.

> [!QUESTION] Why can't a per-group scale leave the sum over K in an INT8 tensor-core GEMM, and what does the kernel do about it?
> The tensor core accumulates $\sum_k q_x q_w$ in int32 across all of $K$, and a scale that changes every 128 steps can't be applied to the finished sum. The kernel would have to stop every 128 steps, convert the int32 partial sum to float, scale it, and add it to an fp32 accumulator. That keeps the math right but adds float work to the integer loop, so INT8 kernels use per-tensor or per-channel scales instead.

> [!QUESTION] What does the bit pattern 0x6400 | q mean as an fp16 number, for q from 0 to 15?
> `0x6400` is the fp16 value 1024, whose mantissa has 10 bits and whose unit in the last place is 1. Setting the low bits to $q$ adds $q$ ulps, so the number is $1024 + q$ exactly. Subtracting 1024, or $1024 + z$ to fold in the zero point, leaves $q$ or $q - z$ in fp16.

> [!QUESTION] At what batch size does W4A16 stop being memory bound on the A100, for the lab's layout and the 1275 GB/s ceiling?
> The intensity is $2M/b$ with $b = 0.5234$ bytes per weight, and the ridge point is 244.7 FLOP per byte. Setting them equal gives $M^{\ast} = 244.7 \times 0.5234 / 2 = 64$. For fp16, $b = 2$ and $M^{\ast} = 245$. Between 64 and 245, the 4-bit kernel is compute bound and fp16 isn't, so the speedup falls toward 1.

> [!QUESTION] A teammate dequantizes the whole int4 matrix to fp16 and then calls cuBLAS. Why is this slower than the fp16 baseline at M of 1?
> It reads 35.1 MB of codes, writes 134.2 MB of fp16 weights, and reads them back in the GEMM, a total of 303.5 MB against 134.2 MB for the plain fp16 GEMM. At the same bandwidth, that is 2.26x slower. A fused kernel dequantizes in registers and moves only the 35.1 MB.

> [!QUESTION] The A100 has no FP8 tensor cores. Can an FP8 weight-only kernel still help there?
> It can save bytes, because an FP8 weight is 1 byte, but the multiply must run in fp16 or bf16 after an unpack in software. That gives the memory saving of int8 with a more expensive unpack, so int8 is the better choice on Ampere. The compute speedup of FP8 needs `mma` on E4M3 or E5M2 operands, which starts at compute capability 8.9 and `wgmma` at 9.0.

> [!QUESTION] How many bits per element does NVFP4 cost, and why does it use a block of 16 where MXFP4 uses 32?
> NVFP4 stores a 4-bit E2M1 value plus one FP8 E4M3 scale per 16 elements, so $4 + 8/16 = 4.5$ bits, ignoring its single FP32 per-tensor scale. The smaller block follows the data's local range more closely, and the E4M3 scale has a mantissa where MXFP4's power-of-two E8M0 scale doesn't, so the scale itself rounds less. MXFP4 costs 4.25 bits.

> [!QUESTION] At M = 1 the kernel loads a tile of 16 activation rows. Is the 15-row waste a problem?
> The extra rows load zeros and add tensor-core work, but the 8192 x 8192 example needs only $2 \times 16 \times 8192^2 / 312 \times 10^{12} = 6.9\ \mu s$ at the fp16 peak, against a memory time of 27.6 $\mu s$. The tensor cores finish in a quarter of the memory time, so the padding is hidden.

## Lab

> [!TRY] W4A16, a fused dequantize-GEMM
> Open the lab `18-w4a16`. You write a quantizer, a packer, and a Triton kernel for the format in this chapter, and the tests check them against an fp32 reference.
>
> 1. `quantize_int4`, `pack_int4`, `unpack_int4`, and `dequantize_int4`: asymmetric uint4 codes, one fp16 scale and one uint8 zero point per row and group of 128 along $K$, two codes per byte with the even index in the low nibble.
> 2. `int4_to_fp16_magic`: convert codes 0 to 15 to fp16 with the `0x6400` trick in torch.
> 3. `w4a16_kernel` and `w4a16_matmul`: compute $Y = X\,\text{dequant}(W)^{\top}$ for $X$ of shape $(M, K)$ in fp16, for $M$ from 1 to 1000, including $N$ and $M$ that aren't multiples of the tile.
>
> Passing means every correctness check agrees with the fp32 reference within a relative error of $2 \times 10^{-3}$ and the kernel is at least 30% of the byte-ratio bound (1.15x) faster than fp16 `torch.matmul` at $M = 1$ on an 8192 x 8192 matrix. The lab reports `bytes_ratio`, `weight_gbs_m1` (the effective bandwidth of codes, scales, and zero points), `speedup_m1`, `speedup_m16`, and `tflops_m1024`.
>
> Before you run it, predict `speedup_m1` and `weight_gbs_m1` from this chapter's byte count and the 1275 GB/s ceiling. A kernel far below the prediction usually has too few programs or too much unpack work. The first call compiles and autotunes, which takes a minute.

## Further reading

- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743) and the [Marlin repository](https://github.com/IST-DASLab/marlin). Read the `dequant` routine in `marlin/marlin_cuda_kernel.cu` against the magic-number section.
- [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978): where group-wise int4 weights come from.
- [LLM.int8()](https://arxiv.org/abs/2208.07339) and [SmoothQuant](https://arxiv.org/abs/2211.10438): activation outliers and W8A8.
- [The Case for 4-bit Precision](https://arxiv.org/abs/2212.09720): why 4-bit weights are the usual operating point.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) and the [OCP 8-bit Floating Point Specification](https://opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): fine-grained FP8 scaling and the accumulation interval.
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) and [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- [torchao](https://github.com/pytorch/ao): reference quantization APIs and Triton kernels.
- The companion course's [quantization chapter](https://qwen.ifkash.dev/c/18-quantization): the error bound, granularity, and the KV cache. Next in this course, [chapter 19](/c/19-training-kernels) turns to training kernels.
