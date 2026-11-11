---
title: FP8, MXFP4, and NVFP4
slug: 33-fp8-and-fp4
part: "Part 4 — Kernels for ML"
summary: Encode FP8 and FP4 numbers bit by bit, scale them per tensor, per block, and per microscaling group, round them without bias, and measure what scale granularity and accumulator width do to a GEMM.
minutes: 120
gpu: true
objectives:
  - "Decode and encode E4M3 (the fn variant), E5M2, and E2M1 bit by bit with round-to-nearest-even and saturation, and match torch's fp8 casts exactly."
  - "Compute the scales of per-tensor, per-channel, 1x128 tile, 128x128 block, MX (E8M0 on 32), and NVFP4 (E4M3 on 16 plus an fp32 tensor scale) quantization."
  - "Explain why stochastic rounding is unbiased, and why round-to-nearest is not."
  - "Choose between delayed and current scaling, and predict when a delayed scale saturates."
  - "Bound the error of a block-scaled GEMM, and say why a narrow tensor-core accumulator needs fp32 promotion every 128 elements of K."
lab: 33-fp8-fp4
---

# FP8, MXFP4, and NVFP4

> [!TLDR]
> - FP8 and FP4 are tiny floats. E4M3 has 4 exponent bits and 3 mantissa bits and tops out at 448. E5M2 reaches 57,344 with 2 mantissa bits. E2M1 holds only $0, 0.5, 1, 1.5, 2, 3, 4, 6$ and their negatives.
> - A scale does most of the work. Per-tensor scales break when one outlier stretches the range, so recipes scale per 1x128 tile, per 128x128 block, per 32 elements (MX, with a power-of-two E8M0 scale), or per 16 elements (NVFP4, with an E4M3 scale plus one fp32 per tensor).
> - Round to nearest, ties to even, is biased on a coarse grid for a sum of many values. ==Stochastic rounding is unbiased==, at the price of noise.
> - A block-scaled GEMM multiplies codes, then applies the scales per K block. A tensor-core accumulator that keeps about 14 bits needs an fp32 promotion every 128 elements of $K$, or its error grows with $K$.
> - The lab writes every piece on a CPU, bit-exact against torch's casts, and checks a Triton block-scaled GEMM on the A100 against the CPU emulation.

An FP8 matmul looks like a free doubling of throughput: half the bytes, twice the tensor-core rate. The catch is the numbers. An 8-bit float has 3 mantissa bits, so the rounding error of every value is up to 6.25%, and the range is narrow enough that one outlier can push the rest of a tensor into the subnormals. FP4 has one mantissa bit and 8 positive values. Every system that trains or serves in these formats is a set of decisions about scales, rounding, and accumulation, and the bugs hide in those decisions.

[Chapter 9](/c/09-floating-point-on-the-gpu) gave you the format table and the idea of a block scale. [Chapter 18](/c/18-quantized-kernels) showed where a scale enters a GEMM and described FP8 and FP4 hardware in prose. This chapter goes one level down. You write the encoder that a hardware converter implements, the scale rules of the four recipes in use, a stochastic rounder, and an emulated FP8 GEMM, and you measure the error of each choice.

Everything but one kernel runs on a CPU. The A100 has no FP8 or FP4 tensor cores, so the lab emulates the formats with plain arithmetic, which is also how you test any real FP8 kernel: against a model you can read.

## Before you start

**The format table.** [Chapter 9](/c/09-floating-point-on-the-gpu) lists the limits of every format. You need three rows again here. E4M3 has a largest value of 448 and a smallest subnormal of $2^{-9}$. E5M2 has 57,344 and $2^{-16}$. E2M1 has 6 and 0.5. This chapter doesn't repeat the table. It explains where those numbers come from.

**Where a scale enters a GEMM.** [Chapter 18](/c/18-quantized-kernels) showed that a scale can leave a sum over $K$ only inside the block that shares it. A per-tensor scale leaves the whole sum and is applied once in the epilogue. A per-block scale is applied each time the loop crosses a block boundary. This chapter builds on that fact.

**Weight quantization for serving.** GPTQ, AWQ, and KV-cache quantization choose which values to quantize and how to compensate for the error. That is serving work, and the companion course's [quantization chapter](https://qwen.ifkash.dev/c/18-quantization) owns it. This chapter owns the formats, the scales, and the kernel-level numerics those methods stand on.

## Reading an 8-bit float

This section answers how eight bits become a number, and how a number becomes eight bits.

An FP8 byte is a sign bit, an exponent field of $e$ bits, and a mantissa field of $m$ bits. Chapter 9's decoding rule applies unchanged: a normal number is $(-1)^S \cdot 2^{E - \text{bias}} \cdot (1 + M/2^m)$, and a field $E = 0$ is a subnormal, $(-1)^S \cdot 2^{1 - \text{bias}} \cdot M/2^m$. The two formats differ in how they spend the top exponent:

| | E4M3 (fn variant) | E5M2 |
|---|---|---|
| Layout S/E/M | 1/4/3 | 1/5/2 |
| Bias | 7 | 15 |
| Top exponent field | Finite numbers, except one NaN | Infinity and NaN, as in IEEE |
| Infinity | None | `0x7C` and `0xFC` |
| NaN | `0x7F` and `0xFF` | `0x7D` to `0x7F` and the negative codes |
| Largest finite | `0x7E`: $2^{8} \times 1.75 = 448$ | `0x7B`: $2^{15} \times 1.75 = 57{,}344$ |
| Smallest normal | $2^{-6}$ | $2^{-14}$ |
| Smallest subnormal | $2^{-9}$ | $2^{-16}$ |

The "fn" in torch's `float8_e4m3fn` means finite: the format gives up infinity so that the top exponent can hold the 7 numbers from 256 to 448, and keeps only the all-ones pattern, `S.1111.111`, as NaN. The OCP FP8 specification defines both formats, and torch implements the same patterns, which the lab confirms on all 256 codes of each.

> [!EXAMPLE] Decode three bytes
> - `0x7E` is `0 1111 110` in E4M3. The exponent field is 15, so the scale is $2^{15-7} = 256$. The mantissa is $110_2 = 6$, so the significand is $1 + 6/8 = 1.75$. The value is 448.
> - `0x01` is `0 0000 001`. The exponent field is 0, so the byte is subnormal: $2^{1-7} \times 1/8 = 2^{-9} = 0.001953125$.
> - `0x7C` is `0 11111 00` in E5M2: exponent field 31, mantissa 0, so it's infinity. The same byte in E4M3 is $2^{8} \times 1.5 = 384$.

The last bullet carries a warning. A byte has no type. A buffer of FP8 data is meaningless until you know which format wrote it.

### Which one for what

E4M3 has 3 mantissa bits and a narrow range: the ratio of its largest value to its smallest subnormal is $448 / 2^{-9} = 229{,}376$. E5M2 has 2 mantissa bits and a ratio of $57{,}344 / 2^{-16} \approx 3.76 \times 10^{9}$. A weight or an activation has a modest range once you scale it, so E4M3's extra bit of precision pays. A gradient spans many orders of magnitude, so E5M2's range matters more than its precision. The [FP8 formats paper](https://arxiv.org/abs/2209.05433) recommends this split: E4M3 for the forward pass and E5M2 for gradients.

The DeepSeek-V3 report makes a different choice. It uses E4M3 for every tensor, including gradients, and relies on fine-grained scales to supply the range that E5M2 would. A later section explains why that works.

## Rounding to the grid

This section answers how a float32 value becomes the nearest FP8 or FP4 value, and why the tie rule matters.

The picture: every format is a ladder of representable values. Inside one binade, $[2^e, 2^{e+1})$, the rungs are $2^{e-m}$ apart. Below the smallest normal number, the exponent stops falling, so the gap stays at $2^{\text{emin} - m}$ and the rungs are evenly spaced down to zero. To round, find the binade, divide by the gap, round to an integer, and multiply back:

$$
\operatorname{round}(x) = \hla{g} \cdot \operatorname{RNE}\!\left(\frac{x}{\hla{g}}\right), \qquad \hla{g} = 2^{\max(\lfloor \log_2 |x| \rfloor,\; \text{emin}) - m}
$$

Here $\hla{g}$ is the gap of $x$'s binade and $\operatorname{RNE}$ rounds to the nearest integer, sending an exact tie to the even one. The even integer is the one whose mantissa bit is 0, which is why the rule is called ties to even.

> [!EXAMPLE] Round three values to E4M3
> - $0.3$ is in $[0.25, 0.5)$, so $g = 2^{-2-3} = 0.03125$. Then $0.3 / 0.03125 = 9.6$, which rounds to 10, giving $0.3125$. The byte is `0 0101 010` = `0x2A`.
> - $17$ is in $[16, 32)$, so $g = 2$. Then $17/2 = 8.5$ is a tie, and RNE picks 8, giving 16. Round-half-up gives 18. The neighbours are 16 with mantissa 0 and 18 with mantissa 1.
> - $19 / 2 = 9.5$ is a tie too, and RNE picks 10, giving 20. Half-up also gives 20, so one tie in two agrees. The wrong rule is wrong only half the time, which is why a test needs many ties.

An exact tie is common. Every midpoint between two FP8 values is exactly representable in float32, and a value that was itself quantized earlier, such as the output of another low-precision op, lands on grids that share many midpoints. The tie rule is not a corner case.

### Why ties to even

Round-half-up sends every tie in one direction. Add up many tied values and the errors share a sign. RNE sends half of them down and half up, because the even neighbour alternates along the ladder. The expected error of a tie is zero, so the sum of many ties has no drift. IEEE 754 chose RNE as the default for the same reason, and the converters in GPU hardware follow it.

### Overflow: saturate or not

A value above the largest finite number overflows, and the format decides what comes out. Rounding runs as if the exponent were unbounded, and the result is compared with the maximum:

- **E4M3.** The largest value is 448, and the next rung would be 480. The midpoint 464 is a tie that goes to 448, the even mantissa. Anything above 464 rounds to 480, which doesn't exist, so a non-saturating conversion returns NaN. Torch 2.11 gives `0x7E` for 464 and `0x7F` for 465. Torch 2.15 saturates an E4M3 overflow to 448 instead, so check the version you run before you rely on either.
- **E5M2.** The largest value is 57,344 and the next rung is 65,536. The midpoint 61,440 is a tie, and it goes to the even mantissa, which is 65,536's. A plain cast returns infinity.
- **Saturating.** Hardware has a mode that clamps to the largest finite value instead. PTX calls it `satfinite`. A quantizer for training should saturate, because a single NaN in an activation poisons the whole step.

The lab's `encode_fp8` takes `saturate=False` for the cast behavior and `saturate=True` for the clamp, and the tests compare it with torch for in-range values and with the rule above for overflow.

> [!WARNING] A plain cast to E4M3 turns 500 into NaN
> In torch 2.11, `torch.tensor(500.).to(torch.float8_e4m3fn)` is NaN, and in torch 2.15 it is 448. A quantizer that scales by amax divided by 448 never produces a value past 448 in exact arithmetic, but float32 rounding in the division can put the largest element a hair above it. Clamp, or use a saturating conversion, every time.

## The 4-bit element, E2M1

This section answers what you can say with four bits. It's a small enough format to write down in full.

E2M1 has 1 sign bit, 2 exponent bits, and 1 mantissa bit, with a bias of 1. The exponent field 0 is the subnormal: $2^{0} \times M/2$, giving 0 and 0.5. The fields 1 to 3 are normal numbers $2^{E-1} (1 + M/2)$. There is no infinity and no NaN, so the top exponent holds finite numbers. The 8 positive codes are:

| Code (E, M) | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|---|---|---|---|---|---|---|---|---|
| Value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

Bit 3 is the sign, so code 8 is negative zero and code 15 is $-6$. The gaps are 0.5 up to 2, then 1 from 2 to 4, then 2 from 4 to 6. Even the largest gap is a third of the largest value, and the largest value is 6.

Ties follow the same rule as FP8. The midpoint 0.75 sits between code 001 (0.5, odd) and code 010 (1.0, even), so RNE returns 1.0. The midpoint 5 sits between code 110 (4) and code 111 (6), and the even code is 110, so RNE returns 4. The midpoint 0.25 returns 0, and 1.25 returns 1. The lab checks a table of these, and a half-up rounder fails it on 0.25 and 1.25.

Two E2M1 codes fit in one byte, which is how every FP4 tensor is stored. [Chapter 18](/c/18-quantized-kernels) packed int4 the same way, with the even index in the low nibble.

## Stochastic rounding

This section answers how to round so that the average of many roundings is the true value.

The picture: a value $x$ lies between two rungs $\text{lo}$ and $\text{hi}$. Round to nearest always picks the closer one, so a value at 0.3 on a grid with rungs at 0 and 0.5 always becomes 0.5. Stochastic rounding picks $\text{hi}$ with a probability equal to how far $x$ is along the gap:

$$
P(\text{round up}) = \hla{p} = \frac{x - \text{lo}}{\text{hi} - \text{lo}}
$$

The expected value of the result is $\text{lo}(1 - \hla{p}) + \text{hi}\,\hla{p} = \text{lo} + \hla{p}(\text{hi} - \text{lo}) = x$. That is the whole proof: the rounding is ==unbiased by construction==. A value already on the grid has $\hla{p} = 0$ or 1 and doesn't change.

```viz
33-rounding-grid
```

Move $x$ along the grid. The readout shows the round-to-nearest result, the probability of rounding up, and the mean of the stochastic result, which always equals $x$.

> [!EXAMPLE] Sum 1000 copies of 0.3 on the E2M1 grid
> Round to nearest turns every copy into 0.5, so the sum is 500 against a true 300. The error is 200, and it grows linearly with the count. Stochastic rounding returns 0.5 with probability $0.3/0.5 = 0.6$ and 0 otherwise. Each draw has mean 0.3 and variance $p(1-p)(\text{hi}-\text{lo})^2 = 0.24 \times 0.25 = 0.06$, so the standard deviation of one draw is 0.245. The sum of 1000 draws has mean 300 and a standard deviation of $0.245 \sqrt{1000} = 7.7$. The error grows like $\sqrt{n}$ and not like $n$.

The price is noise. The variance of one rounding is at most $(\text{hi} - \text{lo})^2/4$, which is larger than the mean squared error of round to nearest when $x$ is uniformly placed. Stochastic rounding wins when many roundings accumulate into a sum or an update, and it loses when each value is used once. That is why it appears on gradients and weight updates in low-precision training, where a small update to a weight can be smaller than half a rung and would otherwise vanish. Round to nearest suits the forward pass. This chapter states the pattern, and the lab tests the unbiasedness. The choice of which tensors in a recipe use it is a recipe question that the papers in the further reading answer.

### The generator is part of the kernel

Stochastic rounding needs one random number per element. In a real kernel, that number comes from a counter-based generator such as Philox, keyed by the element's index and a seed, so that the result doesn't depend on which thread handles it. The lab uses a seeded `torch.Generator`, and the test asserts that the same seed returns the same bits. An unseeded rounder makes a training run unrepeatable in the way chapter 9's `atomicAdd` does.

## Scales: what they are for

This section answers why a tensor needs a scale and what it costs to choose one badly.

An FP8 format covers a fixed range. A tensor that lives elsewhere needs a multiplier to move into it. With a scale $s$, you store $q = \operatorname{round}(x / s)$ and recover $\hat{x} = s \cdot q$. The usual choice is $s = \text{amax} / q_{\max}$, so the largest element lands on the largest code: $s = \text{amax}/448$ for E4M3 and $s = \text{amax} / 6$ for E2M1.

The scale's granularity decides how many values share one amax. Each choice trades storage and kernel complexity against sensitivity to outliers:

| Granularity | Values per scale | Scale overhead | Where it's used |
|---|---|---|---|
| Per tensor | The whole tensor | Negligible | FP8 training in the original recipes |
| Per channel (row or column) | One row or column | $1/K$ or $1/N$ per element | Weight scales in serving |
| 1x128 tile | 128 along $K$, one row | 32 bits per 128 elements | DeepSeek-V3 activations |
| 128x128 block | 16,384 | 32 bits per 16,384 | DeepSeek-V3 weights |
| MX block | 32 | 8 bits per 32 | MXFP8, MXFP6, MXFP4 |
| NVFP4 block | 16 | 8 bits per 16, plus one fp32 | NVFP4 |

> [!EXAMPLE] One outlier and the rest of the row
> A row holds the value 1.0 and the outlier $10^{5}$. A per-tensor E4M3 scale is $10^{5} / 448 = 223.2$. The 1.0 maps to $1/223.2 = 0.00448$, which is below the smallest normal number $2^{-6} = 0.0156$. In the subnormal range the gap is $2^{-9} = 0.00195$, so 0.00448 is 2.29 gaps and rounds to 2 gaps, $0.003906$. Dequantized, 1.0 returns as $0.872$, an error of 12.8%. With an outlier of $10^{3}$ the scale is 2.23, the 1.0 maps to 0.448, which is normal, and it returns as 0.977, an error of 2.3%. The lab computes both.

The figure measures this on a matrix.

```viz
33-scale-outliers
```

Per-tensor E4M3 holds a relative error of about 2.6% through an outlier of $10^{3}$, reaches 4.8% at $10^{4}$ and 48% at $10^{5}$, and loses every small value at $10^{6}$. A scale per 1x128 tile stays at 2.6%, because the outlier rows keep their own scales. The numbers come from `scripts/figures/33-fp8-and-fp4.py`, which uses the lab's functions on a seeded matrix.

### Delayed and current scaling

A scale needs an amax, and the amax of a tensor is known only after the op that produced it has run. Two policies resolve that:

- **Current scaling** (also called online or just-in-time) computes the amax of the tensor about to be quantized, scales, and quantizes. The scale is exact. The cost is an extra pass over the tensor to find the amax, before the cast can start. DeepSeek-V3 uses it, computing the amax for each 1x128 tile or 128x128 block online.
- **Delayed scaling** uses the amax history of earlier steps, for example the maximum over a window of recent steps. The cast needs no pass over the current tensor, and the new amax is recorded for later. The DeepSeek report describes this as the practice in tensor-wise frameworks.

The failure of delayed scaling is easy to state. When the tensor grows past the window's maximum, $x / s$ exceeds 448 and saturates.

> [!EXAMPLE] A tensor that grows 4x
> The history is $[1.0, 0.9, 1.0]$, so the delayed scale is $1.0/448$. The new tensor has amax 4.0. Its largest element maps to $4.0 \times 448 = 1792$, which saturates to 448 and returns as 1.0, an error of 3.0 on a value of 4.0. Current scaling uses $4.0/448$, maps it to 448 exactly, and returns 4.0. A $-3.0$ in the same tensor returns as $-1.0$ under the delayed scale. The lab asserts both.

Fine-grained scales make delayed scaling less attractive, because there are too many scales to track in a history, and current scaling on a small tile is cheap because the tile is already in registers or shared memory.

## Block scaling in DeepSeek-V3

This section answers what the most-cited FP8 training recipe does, and what it implies for the kernel.

The DeepSeek-V3 report scales activations in 1x128 tiles, which is per token per 128 channels, and weights in 128x128 blocks. Every tensor uses E4M3. The authors attribute the all-E4M3 choice to the fine-grained scales, which share exponent bits among the elements of a group, so each group effectively has its own range.

Chapter 18 gave the structural reason for the shapes. In $C = A B$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$, a scale can leave the sum over $K$ only for elements that share it along $K$. An activation tile 1x128 shares a scale across 128 consecutive $k$ for one row $i$. A weight block 128x128 shares one across 128 consecutive $k$ and 128 columns $j$. Write $\hla{s^A_{i,b}}$ for the activation scale of row $i$ and K block $b$, and $\hlb{s^B_{b,c}}$ for the weight scale of K block $b$ and column block $c$:

$$
C_{ij} = \sum_{b} \hla{s^A_{i,b}} \, \hlb{s^B_{b,\,c(j)}} \sum_{k \in \text{block } b} Q^A_{ik}\, Q^B_{kj}
$$

The inner sum runs on raw codes and is exactly what a tensor core computes. The scales apply once per K block, to a partial result. That is why the K extent of both shapes is the same 128: one pair of scales covers one inner sum. It is also why 128x128 for weights is cheap: the kernel loads one weight scale per $(b, c)$ and one activation scale per $(i, b)$, and applies their product to a $128$-element partial result.

### The cost of an fp32 promotion

The multiply by the two scales and the accumulation into the total happen in fp32 on the CUDA cores, not on the tensor core. The tensor core accumulates only inside a block. That split is intentional, and the next section explains why.

## What a block-scaled GEMM gets wrong

This section answers how large the error of the product is, and where it comes from. It has two parts: quantization error and accumulation error.

### Quantization error

Within the normal range, rounding to $m$ mantissa bits changes a value by a relative amount of at most $u = 2^{-(m+1)}$. For E4M3 that is $u = 2^{-4} = 6.25\%$. If $\hat{a}_k = a_k(1 + \delta_k)$ and $\hat{b}_k = b_k(1 + \epsilon_k)$ with $|\delta_k|, |\epsilon_k| \le u$, the error of a dot product satisfies:

$$
\left| \sum_k \hat{a}_k \hat{b}_k - \sum_k a_k b_k \right| \le (2u + u^2) \sum_k |a_k b_k|
$$

The bound is relative to $\sum_k |a_k b_k|$, not to the result. When the terms cancel, the result is smaller than that sum and the relative error of the result is larger. In practice the errors have random signs, so they add like $\sqrt{K}$ and not like $K$. The lab's seeded Gaussian input, a $70 \times 300$ by $300 \times 130$ product with two outlier rows, gives a mean relative error per row of 3.4% with 1x128 and 128x128 scales on the rows without an outlier. That is below the bound of $2u = 12.5\%$ and in line with errors that cancel.

Outside the normal range, the relative bound doesn't hold. A value that falls into the subnormals has an absolute error of up to half a subnormal gap, times the scale. Per-tensor scaling pushes small values there when one element is large, which is the 12.8% in the earlier example and the 40% mean error of the lab's per-tensor result with $10^{5}$ outlier rows. The lab asserts that block scaling is under 5% and at least four times better than per-tensor on that input.

### Accumulation error

The second error is in the sum. A tensor core multiplies FP8 values exactly, and it accumulates the products in a register of limited width. The DeepSeek-V3 report states that FP8 GEMM accumulation on H800 GPUs retains only around 14 bits, well below fp32. A running sum with a 14-bit significand has a unit roundoff of $2^{-14}$, and Chapter 9's stall argument applies: once the sum is large against the next term, small terms lose their low bits.

The report's fix is promotion. Partial results are copied to CUDA cores for fp32 accumulation every $N_C = 128$ elements of $K$, which the authors report as the minimal interval that meaningfully improves precision. The report says 128 elements equal 4 `wgmma` instructions, which implies a K slice of 32 elements per FP8 instruction. The promotion interval and the scale block are the same 128, so one promotion step applies the scale as well.

The lab models this. The model is not the hardware's exact behavior, which the report describes only as "around 14 bits". It's this: the accumulator adds one 32-element slice, then rounds to 14 significant bits. Every $N_C$ elements, the accumulator is added to an fp32 total and restarts from zero. With $N_C \ge K$ there is no promotion.

The worst-case error follows from the model. Each rounding step errs by at most half a unit in the 14th bit, a relative $2^{-14}$ of the accumulator. Take $K = 8192$ with every product near 1, so a slice adds about 32:

- **No promotion.** The accumulator after step $t$ is about $32t$, and there are 256 steps. The worst-case error is $2^{-14} \times 32 \times (1 + 2 + \dots + 256) = 2^{-14} \times 32 \times 32{,}896 = 64.2$, which is 0.78% of the result 8192.
- **Promotion every 128.** Each chunk has four steps with accumulators of about 32, 64, 96, and 128. The error per chunk is at most $2^{-14} \times 320 = 0.0195$, and there are 64 chunks, for a worst case of 1.25. That is 0.015% of the result, 51 times smaller.

The lab measures both on a seeded input of positive values in $[0.5, 1.5)$: $7 \times 10^{-4}$ without promotion and $1.5 \times 10^{-5}$ with it, relative to the largest output. Both are under their worst cases because the rounding errors partly cancel, and the ratio is 46 against the worst-case ratio of 51. Treat the model as a teaching tool: it shows that the error scales with how long the narrow accumulator runs, and promotion shortens the run.

> [!KEY] The error depends on how long the narrow accumulator runs
> Quantization error is set by the format and the scale granularity. Accumulation error is set by the number of elements added before an fp32 promotion. Fix one without the other and the product still errs.

## NVFP4 and MXFP4

This section answers how the two 4-bit recipes compute their scales, and why the second is more accurate on the same bits.

### MX: a power-of-two scale on 32 elements

The OCP microscaling specification, as described in the Rouhani paper, defines the conversion in three steps. Take the block's amax. Set the shared exponent to $\lfloor \log_2(\text{amax}) \rfloor - e_{\max}$, where $e_{\max}$ is the exponent of the element format's largest normal number. Divide every element by $2^{\text{shared exponent}}$ and round to the element format, clamping an overflow to the largest value:

$$
X = 2^{\hla{\lfloor \log_2 \text{amax} \rfloor - e_{\max}}}, \qquad P_i = \operatorname{round}_{\text{E2M1}}\!\left(\frac{V_i}{X}\right)
$$

For E2M1, $e_{\max} = 2$, because the largest value is $6 = 1.5 \times 2^{2}$. The scale is stored as an E8M0 byte, an 8-bit exponent with no sign and no mantissa. Its value is $2^{c - 127}$ for a byte $c$ from 0 to 254, and $c = 255$ is NaN. The lab checks the bias against torch's `float8_e8m0fnu`.

The floor puts the block's amax in $[4, 8)$ after scaling, which is the top binade of E2M1 and a little beyond it. Values in $(6, 8)$ clamp to 6. That bounds the error of one element at a quarter of the block's amax, from two sources. A clamped amax of nearly 8 returns as 6, an error of $1/4$ of 8. And a value between 4 and 6 errs by up to 1 on a scale where the amax is at least 4, which is also a quarter.

> [!WARNING] ceil is not floor
> Computing the exponent with $\lceil \log_2 \rceil$ moves the block's amax into $(2, 4]$ after scaling. Nothing clamps, but the top binade of the format, which holds 4 and 6, is never used, and every value has a coarser grid. The specification uses floor, and the lab's checks assert the exponent of a known block: amax 5.2 must give exponent 0, and a ceil gives 1.

> [!EXAMPLE] The block from chapter 9
> The block $[0.3, -1.7, 5.2, 0.05]$ has amax 5.2, so $\lfloor \log_2 5.2 \rfloor = 2$, and the exponent is $2 - 2 = 0$, a scale of 1 and an E8M0 byte of 127. The elements round on the E2M1 grid to $0.5, -1.5, 6, 0$. The 5.2 clamps nowhere: it rounds to 6, because it is closer to 6 than to 4.

### NVFP4: an E4M3 scale on 16 elements

NVIDIA's NVFP4 uses blocks of 16 and an E4M3 scale, which has a mantissa, so the scale can be any of the E4M3 values and not just a power of two. A tensor-level fp32 scale keeps the block scales inside E4M3's range. The NVIDIA post describes the pieces, 16-element blocks with E4M3 scales and one fp32 scale per tensor, without printing the formulas. The lab uses the common convention:

$$
s_{T} = \frac{\text{amax}_T}{448 \times 6}, \qquad s_{b} = \operatorname{E4M3}\!\left(\frac{\text{amax}_b}{6\, s_T}\right), \qquad q_i = \operatorname{E2M1}\!\left(\frac{x_i}{s_b\, s_T}\right)
$$

The tensor scale $\hla{s_T}$ sends the tensor's largest block scale to 448, the top of E4M3's range. The block scale $\hlb{s_b}$ is stored in 8 bits, and the element is divided by the product of both.

> [!EXAMPLE] The same block in both formats
> A tensor has amax 100, so $s_T = 100/2688 = 0.0372$. One block holds $[2.4, 0.9, -1.3, 0.2]$ with amax 2.4.
> - **NVFP4.** The raw block scale is $2.4 / (6 \times 0.0372) = 10.75$, which E4M3 rounds to 11 (byte `0x53`). The effective scale is $11 \times 0.0372 = 0.4092$. The block returns as $[2.455, 0.818, -1.228, 0.205]$. The largest error is 0.082.
> - **MXFP4.** The exponent is $\lfloor \log_2 2.4 \rfloor - 2 = -1$, a scale of 0.5. The 2.4 maps to 4.8, which is nearer 4 than 6, so it returns as 2.0. The block returns as $[2.0, 1.0, -1.5, 0.25]$. The largest error is 0.4.
>
> The power-of-two scale can't place the amax near the top of the grid. The mantissa in the NVFP4 scale can.

The NVIDIA post reports the same effect on its illustrative example: an average MSE of 0.72 with an E8M0 scale and 0.08 with an E4M3 scale. The lab measures it on seeded Gaussian data, where NVFP4's mean squared error is 0.0089 against 0.0132 for MXFP4. The post gives the cost of the better scales: NVFP4 spends 4.5 bits per element against MXFP4's 4.25, counting the block scale and not the tensor scale.

### Where NVFP4 breaks

The two-level scheme has a failure mode that follows from the formula. The tensor scale is set by the largest element of the tensor, so a huge outlier makes $s_T$ large and every other block's raw scale small. Substituting $s_T$ gives a raw block scale of $448 \times \text{amax}_b / \text{amax}_T$. It falls into E4M3's subnormals, whose gap is $2^{-9}$, once the tensor's amax exceeds a block's by more than $448 \times 512 = 229{,}376$. The figure's seeded matrix shows it. With outlier rows at $10^{5}$, NVFP4's error rises from 9.5% to 14.1%, and at $10^{6}$ every normal block's scale underflows to zero and the error is 100%. MXFP4 stays at 11.5% and a 1x128 E4M3 tile stays at 2.6%, because their scales don't depend on a tensor-wide maximum.

Real tensors rarely have a $10^{5}$ outlier within a single tensor, and recipes that train in NVFP4 add measures of their own. The point for you is the mechanism: a second-level scale buys range for the first level, and a first level with a finite range can still underflow.

## The kernels

This section answers how these numerics map to code that runs on the lab A100 and on a Hopper GPU.

### What the A100 can run

The A100 (compute capability 8.0) has no FP8 or FP4 tensor core. FP8 `mma.sync` starts at 8.9 and FP8 `wgmma` at 9.0, as [chapter 18](/c/18-quantized-kernels) says. Native `torch._scaled_mm` fails on the lab card. Two things still run on it.

The first is storage conversion. `torch.float8_e4m3fn` exists as a dtype, and casting to it and back gives the round trip a Hopper kernel would see in its inputs. The lab's CPU emulation computes that round trip with its own encoder, and it equals the cast bit for bit.

The second is an emulated block-scaled GEMM in Triton. Every E4M3 value is exactly representable in fp16, so the kernel can hold the quantized values in fp16 tensors, multiply them on the A100's fp16 tensor cores with an fp32 accumulator, and apply the scales in fp32. The structure is the DeepSeek structure of the previous sections:

```python
for kb in range(0, tl.cdiv(K, BLOCK_K)):          # BLOCK_K = 128
    a = tl.load(...)                              # (BLOCK_M, 128) fp16 codes
    b = tl.load(...)                              # (128, BLOCK_N) fp16 codes
    partial = tl.dot(a, b, out_dtype=tl.float32)  # the "tensor core" part
    sa = tl.load(sa_ptr + rm * stride_sam + kb * stride_sak)
    sb = tl.load(sb_ptr + kb * stride_sbk + (rn // 128) * stride_sbn)
    total += partial * sa[:, None] * sb[None, :]  # promotion with the scales
```

This kernel proves the control flow and the scale indexing. It doesn't prove the speed. The products are exact in fp16, and the A100's accumulator is fp32 within a tile, so its numerical error is smaller than the Hopper FP8 path and the lab compares it with the CPU emulation within a relative $10^{-4}$.

The block sizes need care. The lab's 128-element K step is one scale block, and the `rn // 128` index maps a 64-wide output tile to the 128x128 weight block that contains it. A tile that straddled two blocks would need two scales per tile.

### What Hopper and Blackwell add

On a Hopper GPU, `torch._scaled_mm` takes E4M3 or E5M2 operands and scale tensors and returns bf16 or fp16. The second operand must be column-major, which is why a call passes `b.t()` of a row-major $(N, K)$ tensor:

```python
out = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a, scale_b, out_dtype=torch.bfloat16)
```

The lab's last check calls it with per-tensor scalar scales when the GPU has compute capability 8.9 or higher and records the error against an fp32 reference as `scaled_mm_max_rel_err`. On the A100 it prints a skip message and passes. The finer scale layouts, per row and per block, depend on the PyTorch and cuBLAS versions, so read the documentation of your build.

On Blackwell, the `tcgen05.mma` instruction takes the block scales as operands, in the kinds `mxf8f6f4`, `mxf4`, and `mxf4nvf4`. The fp32 promotion that a Hopper kernel runs on the CUDA cores becomes part of the instruction, which is what makes 16-element scale blocks affordable. [Chapter 18](/c/18-quantized-kernels) describes the hardware, and Triton's experimental `tl.dot_scaled` exposes the MX forms. Nothing on the lab A100 runs them.

### Reading the speedup honestly

FP8 doubles the dense tensor-core rate relative to bf16 on Hopper and halves the bytes. A kernel gets the doubling only if the epilogue, the promotion, and the scale loads don't cost more than the multiply saves. The promotion on CUDA cores is the first suspect: it is $2 \times$ BLOCK_M $\times$ BLOCK_N fp32 operations per 128 elements of $K$ against $2 \times 128 \times$ BLOCK_M $\times$ BLOCK_N FLOP of multiply, so it is $1/128$ of the arithmetic, small on paper and larger in practice because it competes for issue slots. The lab doesn't measure this on a Hopper card. It records a Triton time on the A100 for reference, not as a result about FP8.

## What goes wrong

- **Round-half-up.** The encoder agrees with torch on every representable value and on random data, and it disagrees on exact ties. A test with only random inputs almost never hits a tie. The lab builds every midpoint between adjacent values, in both formats, and a value a hair above and below each.
- **A ceil in the E8M0 exponent.** The code runs and the error is slightly larger, because the top binade of E2M1 goes unused. Only a test that checks the exponent of a known block finds it.
- **Subnormal inputs to the scale.** An all-zero block has amax 0 and $\log_2 0 = -\infty$. The lab's exponent function returns $-127$, and the quantizer must still divide by a nonzero scale.
- **A NaN from an unsaturated cast.** An E4M3 value past 464 becomes NaN. One NaN in an activation spreads through the matmul and the loss.
- **Delayed scaling after a distribution shift.** The first steps after a learning-rate change or a data change can raise the amax by a factor, and the delayed scale clips the largest values for as many steps as the window remembers the old maximum.
- **Forgetting that NVFP4 has two scales.** Dequantizing with the block scale alone returns values off by the tensor scale, a factor of 0.0372 in the example above. The tensor scale lives in a separate fp32 and is easy to drop when a tensor is saved.
- **A promotion that doesn't reset the accumulator.** The fp32 total then counts earlier chunks again, and the output is far too large. The lab's check on the accumulator catches it.
- **Comparing against the wrong reference.** Compare an emulated GEMM with an fp64 matmul of the unquantized inputs to measure total error, and with an fp32 matmul of the dequantized inputs to isolate accumulation error. Mixing the two hides which one you changed.

> [!RECAP]
> - E4M3 (fn) has bias 7, no infinity, the single NaN `0x7F`, and a maximum of 448. E5M2 is IEEE-style with a maximum of 57,344. E2M1 has the 8 positive values 0, 0.5, 1, 1.5, 2, 3, 4, 6.
> - Round to nearest even finds the binade, divides by the gap, and rounds ties to the even mantissa. A plain E4M3 cast turns anything above 464 into NaN, so quantizers saturate.
> - Stochastic rounding goes up with probability $(x - \text{lo})/(\text{hi} - \text{lo})$, which makes its mean exact. Round to nearest drifts when many values round the same way.
> - A scale's granularity sets how far one outlier reaches: per-tensor E4M3 degrades once the outlier is about $10^{4}$ times the rest, and 1x128 tiles don't. MX uses a floor-based power-of-two scale on 32, and NVFP4 an E4M3 scale on 16 plus an fp32 tensor scale.
> - A block-scaled GEMM applies scales once per K block, to an exact partial sum. A tensor-core accumulator with about 14 bits needs an fp32 promotion every 128 elements of $K$, or its error grows with $K$.

## Check your understanding

> [!QUESTION] What does E4M3 (fn) return for the byte `0x2A`, and what does `0x7F` mean?
> `0x2A` is `0 0101 010`. The exponent field is 5, so the scale is $2^{5-7} = 0.25$, and the significand is $1 + 2/8 = 1.25$. The value is 0.3125. `0x7F` is `0 1111 111`, the only NaN pattern of the fn variant: the top exponent otherwise holds finite numbers up to 448.

> [!QUESTION] Round 17 and 19 to E4M3 with ties to even. What does round-half-up give?
> Both are ties, with a gap of 2. RNE gives 16 and 20, because 16 has mantissa 0 and 20 has mantissa 2, the even ones. Half-up gives 18 and 20. The rules agree on 19 and differ on 17, so a test needs many ties to tell them apart.

> [!QUESTION] Why does a plain cast of 464 to E4M3 give 448, and 465 give NaN?
> Rounding runs as if the exponent had no limit. The next rung above 448 would be 480, and 464 is the midpoint, a tie that goes to the even mantissa, which is 448's. Above 464 the nearest rung is 480, which E4M3 doesn't have, so the cast returns NaN. A saturating conversion returns 448 for both.

> [!QUESTION] Where does an E2M1 value of 4.8 land with round to nearest, and what is the stochastic probability of 6?
> The neighbours are 4 and 6, and 4.8 is 0.8 from 4 and 1.2 from 6, so round to nearest returns 4. The stochastic rounder returns 6 with probability $0.8/2 = 0.4$, so its mean is $4 \times 0.6 + 6 \times 0.4 = 4.8$.

> [!QUESTION] The E8M0 exponent of an MXFP4 block with amax 7.9 is 0. Where does the 7.9 land, and what is the worst error in that block?
> The 7.9 divides by 1 and is above the largest value, 6, so it clamps to 6, an error of 1.9. That is the clamp that makes the worst case a quarter of the amax, $0.25 \times 7.9 = 1.975$. A larger error would need a value above 8, which the floor excludes.

> [!QUESTION] An NVFP4 tensor has amax 2688. What is the tensor scale, and what raw scale does a block with amax 3 get?
> $s_T = 2688 / 2688 = 1$. The raw block scale is $3 / (6 \times 1) = 0.5$, which E4M3 holds exactly. The block's elements divide by 0.5, so the 3 maps to 6.

> [!QUESTION] A delayed scale was computed from a history with maximum 2. The next tensor has amax 8. What happens to the largest element?
> The scale is $2/448$, so $8 / (2/448) = 1792$, which saturates to 448. The element returns as 2, an error of 6 on a value of 8. The same tensor under current scaling returns 8.

> [!QUESTION] Why is the promotion interval 128, the same number as the scale block?
> Both come from one design. The scale is constant along a block of 128 in $K$, so the kernel can apply it to a partial result that covers exactly those 128 elements. That partial result is also the span after which the narrow accumulator is copied to fp32. Applying the scale and promoting in one step costs one extra multiply per promotion.

## Lab

> [!TRY] FP8 and FP4, bit by bit
> Open `33-fp8-fp4`. In `starter.py` you write, in order: `e2m1_decode` (the first check needs only this one), `round_minifloat`, `e2m1_encode`, `decode_fp8`, `encode_fp8`, `e8m0_exponent`, `mxfp4_quantize` and its inverse, `nvfp4_quantize` and its inverse, `stochastic_round_e2m1`, `fp8_scale`, `fp8_fake_quant`, `block_scaled_gemm`, and `gemm_limited_accumulator`. The last function, `triton_block_scaled_gemm`, is a kernel.
>
> You pass when `decode_fp8` and `encode_fp8` equal torch's casts on all 256 codes of E4M3 and E5M2, on every midpoint, and on seeded random inputs, when overflow follows the format rule, when E2M1 rounds the tie table correctly, when the MXFP4 exponent of a block with amax 5.2 is 0 and its error stays within a quarter of the block amax, when NVFP4 beats MXFP4 on Gaussian data, when the stochastic mean is within five standard deviations of $x$, and when the block-scaled GEMM is under 5% error and at least four times better than per-tensor on outlier rows of $10^{5}$. The accumulator check needs promotion every 128 elements to cut the model's error at least ten times.
>
> A machine without a GPU skips the Triton check, and the A100 skips `torch._scaled_mm`. The lab reports `mxfp4_mse`, `nvfp4_mse`, `gemm_rel_err_tensor`, `gemm_rel_err_block`, `acc_err_nc128`, `acc_err_nc_k`, `triton_max_rel_err`, and `triton_ms`. Predict `acc_err_nc_k` from the worst-case arithmetic in this chapter before you run it.

## Further reading

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), Micikevicius and others, 2022: the E4M3 and E5M2 definitions and the recipe for using each in training.
- [OCP 8-bit Floating Point Specification (OFP8)](https://opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf): the bit-level definitions of both formats.
- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537), Rouhani and others, 2023: the shared-exponent conversion, the element formats, and the E8M0 scale.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), NVIDIA: 16-element blocks, E4M3 scales, and the second-level fp32 scale.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), section 3.3: 1x128 and 128x128 scaling, online quantization, and the 128-element promotion interval.
- [Quantization](https://qwen.ifkash.dev/c/18-quantization), the companion course: weight quantization, GPTQ and AWQ, and KV-cache quantization for serving.
- [GPU MODE lecture index](https://github.com/gpu-mode/lectures): the numerics and quantization sessions. The index was not fetched for this chapter, so it cites no lecture number.
- [Chapter 9](/c/09-floating-point-on-the-gpu) for the format table and the accumulation argument, and [chapter 18](/c/18-quantized-kernels) for where a scale enters a GEMM and for the Hopper and Blackwell hardware.
