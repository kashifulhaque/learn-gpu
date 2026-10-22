---
title: Floating point on the GPU
slug: 09-floating-point-on-the-gpu
part: "Part 2 — CUDA patterns"
summary: How fp32, TF32, bf16, fp16, fp8, and fp4 round, why the order of a sum changes its value, and how to write sums that are accurate, fast, and repeatable.
minutes: 120
gpu: true
objectives:
  - "Decode a number from its sign, exponent, and mantissa bits, and state the range and unit roundoff of fp32, TF32, bf16, fp16, FP8, and FP4."
  - Explain round-to-nearest-even, subnormals, and flush-to-zero, and say what --use_fast_math changes.
  - Show why a fused multiply-add can change a result, and why a GPU result and a CPU result may differ without either being wrong.
  - Predict where a low-precision running sum stalls, and compare the error of sequential, pairwise, and Kahan summation.
  - Say how TF32 and tensor-core accumulation round, and which PyTorch switches control them.
  - Explain why atomicAdd and split reductions make a sum nondeterministic, and write a fixed-order alternative.
  - Distinguish run-to-run nondeterminism from batch variance in inference.
lab: 09-numerics
---

# Floating point on the GPU

> [!TLDR]
> - A float is a sign, an exponent, and a mantissa. The mantissa width sets the
>   precision: a unit roundoff of $2^{-24}$ for fp32, $2^{-8}$ for bf16, and
>   $2^{-4}$ for FP8 E4M3. The exponent width sets the range.
> - Addition rounds, so ==the order of a sum changes its value==. A running sum
>   stalls once the next term is smaller than about $2^{-p}$ of the total, which
>   for bf16 is $1/256$.
> - Accumulate in fp32 even when the inputs are 16-bit or 8-bit. Tensor cores
>   already do, and the error that remains comes from the order of the sums.
> - A tree sum, or a Kahan compensated sum, cuts the error from growing like $n$
>   to growing like $\log n$ or not at all.
> - `atomicAdd` and split reductions add in whatever order the hardware
>   delivers, so two runs can differ in the last bits. A fixed-order tree is
>   deterministic and costs little.

Train the same model twice with the same seed on the same GPU, and the losses
drift apart in the fourth digit. Run a served model at temperature 0 and the
same prompt sometimes returns different words. Neither is a bug in your code.
Floating-point addition isn't associative, a GPU adds in an order that depends
on the schedule, and every format you use to save memory throws away bits that
a sum then cannot recover.

This chapter is the numerics that the rest of the course leans on. The
reductions of [chapter 6](/c/06-reductions) are sums, the tensor-core GEMM of
[chapter 11](/c/11-tensor-cores) multiplies narrow numbers and accumulates wide
ones, softmax and attention add up thousands of exponentials, and the quantized
kernels of [chapter 18](/c/18-quantized-kernels) are a choice of format and
scale. You can't reason about any of them without knowing what a float can and
can't hold.

You build the lab bottom-up: round-trip a value through each format and measure
its error, watch a bfloat16 sum stall, write Kahan and fixed-order tree sums in
CUDA, and count how many different answers `atomicAdd` gives you on one input.

## Before you start

**Binade.** The range $[2^e, 2^{e+1})$. Every float in one binade shares an
exponent, so the gap between neighbours is the same throughout it.

**ULP and unit roundoff.** A *unit in the last place* (ulp) is the gap between
a float and the next one. Rounding to the nearest float errs by at most half an
ulp. The companion course derives this in
[its notation chapter](https://qwen.ifkash.dev/c/00a-notation-and-prerequisites);
this chapter repeats only what the kernels need.

**Reductions and atomics.** A reduction turns many values into one, usually with
a tree of adds inside a block. An *atomic* operation reads, modifies, and writes
a memory word as one indivisible step. [Chapter 6](/c/06-reductions) builds both,
and this chapter asks what they do to the answer.

## What a float is

A floating-point number is a sign bit, an exponent field, and a mantissa field,
and the question this section answers is how those bits become a value.

Take a format with $e$ exponent bits and $m$ mantissa bits. Its bias is
$2^{e-1} - 1$, and the value of a *normal* number, one whose exponent field $E$
is neither all zeros nor all ones, is:

$$
x = (-1)^{S} \times 2^{\,\hla{E} - \text{bias}} \times \left(1 + \frac{\hlb{M}}{2^{m}}\right)
$$

Here $\hla{E}$ is the exponent field read as an integer and $\hlb{M}$ is the
mantissa field read as an integer. The leading 1 is implicit, which is why fp32
stores 23 mantissa bits and carries 24 bits of precision.

The exponent field being all zeros marks a *subnormal*, which drops the implicit
1 so that the numbers near zero stay evenly spaced:

$$
x = (-1)^{S} \times 2^{\,1 - \text{bias}} \times \frac{M}{2^{m}}
$$

Zero is the subnormal with $M = 0$. An exponent field of all ones, in an
IEEE-style format, means infinity when the mantissa is zero and NaN otherwise.

> [!EXAMPLE] Decode a bfloat16 number
> The bfloat16 bit pattern `0x4049` is `0 10000000 1001001`. The exponent field
> is $128$ and the bias is $127$, so the scale is $2^{1}$. The mantissa is
> $1001001_2 = 73$, so the significand is $1 + 73/128 = 1.5703125$. The value is
> $2 \times 1.5703125 = 3.140625$: the nearest bfloat16 to $\pi$, off by
> $9.7 \times 10^{-4}$.

### The formats a GPU kernel meets

Every format in this course is one choice of $(e, m)$. The following table lists
them with their limits, taken from IEEE 754, the OCP specifications, and the
NVIDIA PTX documentation:

| Format | Bits (S/E/M) | Largest finite | Smallest normal | Smallest subnormal | Epsilon $2^{-m}$ |
|---|---|---|---|---|---|
| fp32 | 32 (1/8/23) | $3.4028 \times 10^{38}$ | $1.1755 \times 10^{-38}$ | $1.4013 \times 10^{-45}$ | $1.1921 \times 10^{-7}$ |
| TF32 | 19 (1/8/10), in a 32-bit container | $3.4012 \times 10^{38}$ | $1.1755 \times 10^{-38}$ | $1.148 \times 10^{-41}$ | $9.766 \times 10^{-4}$ |
| bf16 | 16 (1/8/7) | $3.3895 \times 10^{38}$ | $1.1755 \times 10^{-38}$ | $9.18 \times 10^{-41}$ | $7.8125 \times 10^{-3}$ |
| fp16 | 16 (1/5/10) | 65,504 | $6.1035 \times 10^{-5}$ | $5.9605 \times 10^{-8}$ | $9.766 \times 10^{-4}$ |
| FP8 E4M3 | 8 (1/4/3) | 448 | 0.015625 | 0.001953125 | 0.125 |
| FP8 E5M2 | 8 (1/5/2) | 57,344 | $6.1035 \times 10^{-5}$ | $1.5259 \times 10^{-5}$ | 0.25 |
| FP6 E3M2 | 6 (1/3/2) | 28 | 0.25 | 0.0625 | 0.25 |
| FP6 E2M3 | 6 (1/2/3) | 7.5 | 1 | 0.125 | 0.125 |
| FP4 E2M1 | 4 (1/2/1) | 6 | 1 | 0.5 | 0.5 |

Epsilon is the gap between 1.0 and the next float. Three details in the table
matter later:

- **bf16 is fp32 with the bottom 16 bits cut off.** It has fp32's exponent, so
  it has fp32's range, and 7 mantissa bits, so it has about two decimal digits.
- **fp16 trades range for precision.** Its largest value is 65,504, so a
  gradient or an activation above that overflows to infinity. That is why fp16
  training needs *loss scaling* and bf16 training does not.
- **E4M3 reuses its top exponent for finite numbers.** The "fn" variant has no
  infinity, and only the all-ones exponent with an all-ones mantissa is NaN, so
  its largest value is $2^{8} \times 1.75 = 448$ instead of 240. E5M2 follows
  IEEE and keeps infinity and NaN.

Two integer formats complete the picture. int8 holds $-128$ to $127$ and int4
holds $-8$ to $7$. They have no exponent, so their grid is uniform, and a
*scale* maps them back to real values. [Chapter 18](/c/18-quantized-kernels)
builds a kernel around that scale.

```viz
09-format-rounding
```

Use the figure to see the spacing. Pick bf16 and move the exponent: the marks
stay the same number of mantissa steps apart, but each step is twice as wide at
every power of two. A format has a fixed *relative* precision and a variable
*absolute* one.

## Rounding

The section answers one question: when a result lands between two floats, which
one do you get?

The answer is set by the *rounding mode*. The default in IEEE 754 and on the
GPU is *round to nearest, ties to even*: take the closer float, and when the
number sits exactly halfway, take the one whose last mantissa bit is zero.
Ties to even has no bias, because half of all ties round up and half round down.

The error of one rounding is at most half an ulp. For a number in a binade with
$m$ mantissa bits, the ulp is $2^{e-m}$ and the relative error is bounded by
the *unit roundoff* $\hlc{u}$:

$$
\hlc{u} = 2^{-(m+1)}, \qquad \frac{|\mathrm{fl}(x) - x|}{|x|} \le \hlc{u}
$$

Here $\mathrm{fl}(x)$ is $x$ rounded into the format. The following table
evaluates it, and your lab measures each entry by round-tripping 65,536 values:

| Format | $m$ | Unit roundoff $2^{-(m+1)}$ | Decimal |
|---|---|---|---|
| fp32 | 23 | $2^{-24}$ | $5.96 \times 10^{-8}$ |
| TF32 | 10 | $2^{-11}$ | $4.88 \times 10^{-4}$ |
| fp16 | 10 | $2^{-11}$ | $4.88 \times 10^{-4}$ |
| bf16 | 7 | $2^{-8}$ | $3.91 \times 10^{-3}$ |
| E4M3 | 3 | $2^{-4}$ | $6.25 \times 10^{-2}$ |
| E5M2 | 2 | $2^{-3}$ | 0.125 |
| E2M1 | 1 | $2^{-2}$ | 0.25 |

The measured value for a coarse format sits a little under the table entry. The
bound is reached only where the gap is widest relative to the number, and in
$[1, 2)$ the first tie is at $1 + 2^{-(m+1)}$, not at 1. For E4M3 the worst
case is the tie at 1.0625, which errs by 0.0625 on a value of 1.0625, so the lab
reports $0.0588$ against the bound $0.0625$. The tests accept 85% to 100% of the
bound for that reason.

> [!EXAMPLE] Ties to even in bfloat16
> A bfloat16 number in $[256, 512)$ has $m = 7$, so the ulp is
> $2^{8-7} = 2$ and the neighbours of 257 are 256 and 258. Then 257 is a tie,
> and 256 has the even mantissa (`0000000`) while 258 has the odd one
> (`0000001`), so $257 \mapsto 256$. The number 259 is a tie between 258 and
> 260, and 260 is even, so $259 \mapsto 260$. Ties to even rounds one up and
> one down.

The other modes matter because hardware offers them. Round toward zero, or
*truncation*, drops the extra bits, so a positive number always rounds down and
a negative one up. Its error bound is a full ulp, not a half, and the bias is
one-sided: a large sum of truncated products drifts toward zero. CUDA exposes
all four IEEE modes as suffixes, such as `__fadd_rn` for nearest-even and
`__fadd_rz` for toward zero. TF32 conversion is where the mode becomes a real
decision, and a later section returns to it.

### Subnormals and flush to zero

Near zero, the implicit 1 would leave a gap between zero and the smallest normal
number, so the format fills it with subnormals. In fp32 the smallest normal is
$2^{-126} = 1.1755 \times 10^{-38}$, and the subnormals below it step by
$2^{-149} = 1.4013 \times 10^{-45}$ all the way down to zero. They are evenly
spaced, so a subnormal has a larger *relative* error the closer it gets to zero.

Subnormals are slow or absent on some hardware. *Flush to zero* (FTZ) treats a
subnormal input as zero and returns zero for a result that would be subnormal.
The `-ftz=true` compiler flag enables it, and `--use_fast_math` implies it. A
sum or a normalization that depends on tiny values, such as the denominator of
an attention softmax in a long sequence, can change when FTZ is on.

For the 8-bit formats, subnormals are a large part of the range. E4M3 has a
smallest normal of $2^{-6} = 0.015625$ and a smallest subnormal of
$2^{-9} = 0.001953125$, so every value below $2^{-10}$ rounds to zero, and $2^{-10}$
itself is a tie that ties to even sends to zero. Your lab checks
both.

### Range and overflow

Rounding up can overflow. In an IEEE-style format a value past the largest
finite number becomes infinity. In E4M3, a plain cast to `torch.float8_e4m3fn`
produces NaN instead, because the format has no infinity. FP8 recipes avoid both
by clamping to the maximum first, which is called *saturation*, and the lab's
`roundtrip` does the same.

The E4M3 and E5M2 pair exists because range and precision pull against each
other. The FP8 formats paper, by
[Micikevicius and others](https://arxiv.org/abs/2209.05433), recommends E4M3 for
the tensors that need precision, weights and activations, and E5M2 for
gradients, which span a wider range. Your choice per tensor is a choice of where
to spend the 8 bits.

## The low-precision zoo and block scaling

The section answers why a 4-bit float is usable at all: a *scale* shared across
a block of values moves the format's tiny range to where the data lives.

An element format on its own has a narrow dynamic range. E2M1 holds only
$0, 0.5, 1, 1.5, 2, 3, 4, 6$ and their negatives. If you multiply a block of
values by a scale chosen so that its largest value lands near 6, the 16 numbers
become a usable grid for that block. Two schemes do this:

| Scheme | Element | Block size | Scale format | Bits per element, scale included |
|---|---|---|---|---|
| MXFP8 | E4M3 or E5M2 | 32 | E8M0 (a power of two) | $8 + 8/32 = 8.25$ |
| MXFP6 | E3M2 or E2M3 | 32 | E8M0 | $6 + 8/32 = 6.25$ |
| MXFP4 | E2M1 | 32 | E8M0 | $4 + 8/32 = 4.25$ |
| NVFP4 | E2M1 | 16 | FP8 E4M3, plus one FP32 per tensor | $4 + 8/16 = 4.5$ |
| int4, group 128 | int4 | 128 | fp16 | $4 + 16/128 = 4.125$ |

The MX family is specified by the
[OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
NVFP4 uses a smaller block and a finer scale, which
[NVIDIA's NVFP4 post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
describes. The per-tensor FP32 scale in NVFP4 isn't counted in the 4.5 bits.

> [!EXAMPLE] Quantize a block to E2M1
> Take the block $[0.3, -1.7, 5.2, 0.05]$ and a scale of 1, as the MX
> specification's reference conversion would pick for a maximum of 5.2, whose
> binade starts at $2^2$ and whose element format ends at the exponent 2. Round
> each value to the grid $0, 0.5, 1, 1.5, 2, 3, 4, 6$:
>
> - $0.3 \mapsto 0.5$ (0.2 away, against 0.3 to zero).
> - $-1.7 \mapsto -1.5$ (0.2 away, against 0.3 to 2).
> - $5.2 \mapsto 6$ (0.8 away, against 1.2 to 4).
> - $0.05 \mapsto 0$.
>
> The relative errors are 67%, 12%, 15%, and 100%. A coarse grid is rough on
> small values and gentle on large ones, so what matters is whether the small
> ones carry signal.

The hardware that runs these formats is newer than the lab card:

| Architecture | Compute capability | Tensor-core formats added |
|---|---|---|
| Ampere (A100) | 8.0 | TF32, bf16, fp16 (and int8) |
| Ada | 8.9 | FP8 through `mma.sync` |
| Hopper | 9.0 | FP8 (E4M3, E5M2) through `wgmma` |
| Blackwell | 10.0 and later | FP6, FP4, block-scaled MXFP8, MXFP4, NVFP4 through `tcgen05.mma` |

The A100 has no FP8 tensor cores, and `torch._scaled_mm` requires compute
capability 8.9 or higher, so it fails on the lab card. The `torch.float8_e4m3fn`
dtype still exists as a *storage* type: you can cast to it and back, and the
round-trip error is exactly what a Hopper kernel would see in its inputs. That
is how the lab studies FP8 on an A100, by emulation with casts. Throughput is a
separate question, and [chapter 11](/c/11-tensor-cores) and
[chapter 18](/c/18-quantized-kernels) return to it.

## Fused multiply-add and the compiler

The section answers why the same expression can give two answers on one GPU
and a CPU, with no bug in either.

A *fused multiply-add* (FMA) computes $a \times b + c$ with a single rounding.
A separate multiply and add rounds twice: once for the product, once for the
sum. The GPU's arithmetic units do an FMA in one instruction, and `nvcc` fuses
`a * b + c` into one by default. A CPU reference written in NumPy usually does
not.

The difference is visible when the product has more bits than the format holds.
Let $a = 1 + 2^{-12}$ in fp32. The exact square is $1 + 2^{-11} + 2^{-24}$. The
spacing of fp32 at 1 is $2^{-23}$, so the term $2^{-24}$ is exactly half a ulp:
a tie, and ties to even drops it. The rounded product is $1 + 2^{-11}$. Now
subtract that rounded product:

$$
\underbrace{\mathrm{fl}\bigl(\mathrm{fl}(a \times a) - (1 + 2^{-11})\bigr)}_{\text{multiply, then add}} = 0
\qquad
\underbrace{\mathrm{fl}\bigl(a \times a - (1 + 2^{-11})\bigr)}_{\text{FMA}} = 2^{-24}
$$

The FMA keeps the low bits of the product that the multiply threw away, so it
returns the true answer, $2^{-24} \approx 5.96 \times 10^{-8}$. Neither result is
wrong in the IEEE sense. They are different programs. NVIDIA's
[floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html)
works the same trap in binary with a value squared minus one, and says that to
stop a fusion you use the intrinsics `__fmul_rn` and `__fadd_rn`, which `nvcc`
never merges.

### What --use_fast_math does

`--use_fast_math` is a bundle. The CUDA programming guide lists what it turns
on: `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`, and `-fmad=true`. It
also swaps library calls for the faster, less accurate intrinsics:

| Call | Default accuracy | With `--use_fast_math` |
|---|---|---|
| `expf(x)` | at most 2 ulp | becomes `__expf(x)`: at most $2 + \lfloor 1.173\,|x| \rfloor$ ulp |
| `x / y` | correctly rounded, 0 ulp | approximate: documented at 2 ulp |
| `sqrtf(x)` | 0 ulp | 1 ulp |
| a subnormal input | used as is | flushed to zero |

The bounds are the CUDA programming guide's. For $|x|$ up to 20 the `__expf`
bound is $2 + \lfloor 1.173 \times 20 \rfloor = 25$ ulp, against 2. The error
grows with $|x|$ because `__expf` computes $2^{x \log_2 e}$ with the hardware's
approximate exponential, and the multiplication by $\log_2 e$ rounds first.

That is a good trade in a softmax whose output goes to bf16, where an ulp is
$2^{-8}$ and 25 fp32 ulps are $25 \times 2^{-24} \approx 1.5 \times 10^{-6}$. It
is a poor one in a loss you compare against a reference to six digits. The
guide itself says not to use the flag when the portability of results matters.
Your lab compiles one `expf` kernel both ways and reports the measured worst
error over $[-20, 20]$ for each.

> [!WARNING] A fused CPU reference is a different program
> If you compare a CUDA kernel with a NumPy or PyTorch-on-CPU reference and the
> results differ in the last bit, check FMA first. Compile with `-fmad=false`,
> or write `__fmul_rn` and `__fadd_rn`, and see whether the gap closes. Don't
> tighten the tolerance until you know which side fused.

## Why the order of a sum matters

The section answers why adding the same numbers in two orders gives two
answers, and by how much.

Rounding happens after every add, so $(a + b) + c$ and $a + (b + c)$ round at
different points. With fp32 and $a = 10^{8}$, $b = 1$, $c = -10^{8}$:

- $(a + b) + c$: the ulp of fp32 at $10^8$ is $2^{26 - 23} = 8$, so
  $10^8 + 1 \mapsto 10^8$, and the sum is $0$.
- $(a + c) + b$: the first add is exactly 0, so the sum is $1$.

One answer is right and the other is 100% wrong. The general question is how
fast the error grows with the number of terms $n$.

### The error of a sum

Add $n$ numbers one at a time, left to right. Each add commits a relative error
of at most $u$, on a partial sum that can be as large as the total. The
standard bound, from Goldberg's
[survey of floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
and Higham's textbook analysis, is:

$$
\frac{|\hat{S} - S|}{\sum_i |x_i|} \le (n - 1)\,\hlc{u}
\quad \text{(sequential)}
\qquad
\frac{|\hat{S} - S|}{\sum_i |x_i|} \le \lceil \log_2 n \rceil\,\hlc{u}
\quad \text{(pairwise)}
$$

Here $\hat{S}$ is the computed sum and $S$ the exact one. The pairwise sum adds
neighbours, then neighbours of those, so every input passes through only
$\lceil \log_2 n \rceil$ additions. Plug in $n = 2^{20}$ values in fp32, where
$u = 2^{-24}$:

| Scheme | Bound | Value |
|---|---|---|
| Sequential fp32 | $(2^{20} - 1) \times 2^{-24}$ | $0.0625$ |
| Pairwise fp32 | $20 \times 2^{-24}$ | $1.19 \times 10^{-6}$ |

The sequential bound, a 6% error, is the worst case and rarely reached. In
practice the rounding errors have random signs, so they add like a random walk.
For values uniform in $[0, 1)$ the partial sum after $k$ terms is about $k/2$,
and each add errs by an amount whose standard deviation is $\mathrm{ulp}(S_k)/\sqrt{12}$.
Summing those variances over $2^{20}$ terms predicts a relative error of about
$1.3 \times 10^{-5}$ for the sequential sum: about 10 times the pairwise
bound. Your lab measures all three sums on exactly that input.

### The sum that stops

For a narrow accumulator the problem is not drift but a wall. A running sum $S$
and an addend $a$ round to $S$ again when $a$ is below half an ulp of $S$, and
the ulp of $S$ is about $S \cdot 2^{-(p-1)}$ in a format with $p$ significand
bits, so:

$$
\hlc{a} \le \tfrac{1}{2}\,\mathrm{ulp}(S) \implies \mathrm{fl}(S + a) = S
\qquad\text{when}\qquad
\frac{S}{\hlc{a}} \gtrsim 2^{\,p}
$$

> [!KEY] A running sum in a $p$-bit format stops at about $2^p$ addends
> If every addend is about the same size, a left-to-right sum in bf16 ($p = 8$)
> stops growing at about $2^8 = 256$ addends, in fp16 ($p = 11$) at about
> $2^{11} = 2048$, and in fp32 ($p = 24$) at about $2^{24} = 16{,}777{,}216$.
> Past that point every further term vanishes.

> [!EXAMPLE] A thousand ones in bfloat16
> Add 1 to a bfloat16 accumulator, 1024 times. The sum counts exactly from 1 to
> 256, because every integer to 256 fits in 8 significand bits. The next add
> is $256 + 1 = 257$, the tie from the rounding section, and ties to even
> returns 256. Every later add does the same, so the sum is stuck at
> $\hlc{256}$ and the true answer, 1024, is off by a factor of 4. fp32 does
> the same trick at $2^{24}$: a float32 loop that counts up by 1 stops at
> 16,777,216.

The companion course meets the same wall in
[RMSNorm](https://qwen.ifkash.dev/c/04-rmsnorm-and-residuals), where a 5120-wide
bf16 reduction would stall near 256 terms. Your lab reproduces it: the
`sum_bf16_sequential` function adds one column at a time in a bf16 accumulator,
and a row of 1024 ones must come out as exactly 256.

The error is also **one-sided**. An addend below half an ulp is dropped, and
dropping a positive addend makes the sum too small, so the mistake is a bias that averaging over layers or tokens
does not cancel.

```viz
09-summation-order
```

In the figure, switch the accumulator from bf16 to fp32 and watch the red curve
straighten. Then raise $n$ with fp32 selected: the sequential error grows with
$n$ while the tree error barely moves.

### Tree sums, and why the GPU is better than the loop

A tree reduction is a pairwise sum, and it is what a GPU does anyway. Inside a
block, each thread first adds its own strided slice, then the block halves the
number of active threads at every step, as in
[chapter 6](/c/06-reductions). No running sum grows far ahead of what is added to
it, so the stall mostly disappears.

For 5120 uniform values in bf16, the pairwise bound is
$\lceil \log_2 5120 \rceil \times 2^{-8} = 13 \times 0.0039 = 0.051$, about 5%.
That is the worst case, and the lab's check uses the tighter
$12.3 \times 2^{-8} = 0.048$ figure. The sequential sum stalls near 256 of a true
2560, a 90% error. The two are not close. A hand-written reduction should
accumulate in fp32: cast to fp32 before you add, and round once at the end.

### Kahan summation

Kahan summation keeps the low-order bits that each add loses, in a second
variable, and feeds them back into the next add. For a sum $s$ with a
compensation $c$ and the next input $x_i$:

$$
y = x_i - \hld{c}, \qquad t = s + y, \qquad \hld{c} \leftarrow (t - s) - y, \qquad s \leftarrow t
$$

The term $(t - s)$ is what the add actually contributed. Subtracting $y$ gives
the part of $y$ that the rounding lost, with its sign flipped. The next
iteration subtracts $\hld{c}$ from its input, which puts the lost bits back in.

Trace it in bfloat16, starting from $s = 256$ and adding 1 four times. The
plain sum is stuck at 256. The Kahan sum recovers:

| Step | $y$ | $t = \mathrm{fl}(s + y)$ | $c$ | $s$ |
|---|---|---|---|---|
| 1 | 1 | 256 (257 is a tie) | $(256 - 256) - 1 = -1$ | 256 |
| 2 | $1 - (-1) = 2$ | 258 | $(258 - 256) - 2 = 0$ | 258 |
| 3 | 1 | 260 (259 is a tie) | $(260 - 258) - 1 = 1$ | 260 |
| 4 | $1 - 1 = 0$ | 260 | $0$ | 260 |

After four adds $s = 260$, which is the true $256 + 4$. In step 3 the sum overshot
to 260 and $c = 1$ records the overshoot, so step 4 adds nothing.

Kahan's error bound is $2u \sum|x_i|$ plus a term of order $n u^2$, with no
factor of $n$ in the leading term. It costs four flops per element instead of
one, which is free when the kernel is memory bound, as a reduction is. Two
cautions:

- **Don't let the compiler remove it.** On a CPU, `-ffast-math` may reassociate
  $(t - s) - y$ to zero. `nvcc` fuses multiply-adds but doesn't reassociate
  additions by default, and your lab verifies that the compensation survives by
  checking the error, not by trusting the claim.
- **Compensation doesn't rescue a narrow accumulator forever.** The trace works
  because $c$ is small and exact. A tree in fp32 is simpler and nearly as
  accurate, and it parallelizes. Reach for Kahan when you cannot change the
  order, such as a single-thread loop or a streaming sum.

## Accumulation inside a tensor core

The section answers how a tensor core adds, because a GEMM is a sum of $K$
products.

A tensor core multiplies narrow inputs and accumulates in a wider format. A
bf16 or fp16 input has 8 or 11 significand bits, so a product has at most 16 or
22 bits, which fits in fp32's 24. The products are therefore exact, and the
only rounding is in the accumulation. PTX specifies the accumulator as "at
least" single precision and leaves the order and the internal rounding
unspecified, and the paper
[Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/abs/2206.02874)
probes what the hardware really does. Don't assume a tensor core adds like your
Python loop.

The accumulator type has a price in speed that is small and a price in accuracy
that is large. On an A100 PCIe, Luo and others (the 40GB card in their paper) measured `mma.sync`
m16n8k16 at 310.6 TFLOP/s with an fp16 accumulator and 303.4 TFLOP/s with an
fp32 accumulator, a 2.3% difference. A gap that small buys nothing for an
accumulator that stalls at $2^{11}$ terms. Use the fp32 accumulator.

Two library behaviours sit on top of the hardware:

- **Split-K.** A GEMM with a long $K$ and few output tiles splits $K$ across
  blocks and adds the partial results afterwards. The partials are then added in
  a second pass or with atomics, in an order that depends on the library's
  choice for that shape.
- **Reduced-precision reduction.** PyTorch's
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` is `True`
  by default, which lets a bf16 GEMM combine partial results in reduced
  precision. The PyTorch CUDA notes suggest setting it to `False` if you see
  numerical instability.

## TF32 and the matmul switches

The section answers what TF32 is, how it rounds, and which flag turns it on.

TF32 keeps fp32's 8 exponent bits and cuts the mantissa to 10 bits, so it has
fp32's range and fp16's precision: 19 meaningful bits in a 32-bit container. A
TF32 tensor core on the A100 does $108 \times 1024 \times 1.41 \times 10^9 =
155.9$ TFLOP/s, and the datasheet rates it at 156. That is 8 times the 19.5
TFLOP/s of the fp32 CUDA cores, and half of the bf16 rate of 312.

PyTorch's defaults put you on the slow side of that 8x. Per the PyTorch CUDA
notes:

| Setting | Default | Effect on an A100 |
|---|---|---|
| `torch.backends.cuda.matmul.allow_tf32` | `False` | An fp32 `matmul` runs on CUDA cores at the 19.5 TFLOP/s peak |
| `torch.backends.cudnn.allow_tf32` | `True` | fp32 convolutions use TF32 tensor cores |
| `torch.set_float32_matmul_precision` | `"highest"` | Full fp32 |

The precision setting has three levels. `"highest"` computes in fp32. `"high"`
uses TF32, or splits each fp32 number into two bf16 numbers, about 16 mantissa
bits, when a fast algorithm exists. `"medium"` uses bf16. The setting changes
the internal computation and not the output dtype, so the result is still an fp32
tensor whose low bits are noisier. [Chapter 10](/c/10-gemm-from-scratch) compares
its SGEMM against cuBLAS with TF32 off for that reason.

### Two roundings of the same format

You can't state one rounding rule for TF32. PTX defines `cvt.rna.tf32.f32` as
round to nearest with ties away from zero, and the instruction gives you a TF32
value in a 32-bit register. When a `wgmma` instruction on Hopper takes raw fp32
inputs as TF32, it truncates the low 13 bits. NVIDIA's TF32 blog says the
inputs are rounded. Each statement is true of one path.

The two differ in the bound and in the bias:

- **Rounding.** Ties away: relative error at most $2^{-11} = 4.88 \times 10^{-4}$,
  and unbiased apart from ties.
- **Truncation.** Toward zero: relative error up to $2^{-10} = 9.77 \times 10^{-4}$,
  always the same sign.

The lab's `round_tf32` implements the first with integer arithmetic on the bit
pattern. View the float as an int32, add half of the 13 dropped bits, which is
`0x1000`, and clear the low 13 bits:

```python
bits = x.contiguous().view(torch.int32)
rounded = (bits + 0x1000) & ~0x1FFF   # ties carry up: away from zero
tf32 = rounded.view(torch.float32)
```

A carry out of the kept bits increments the exponent, which is the right
answer for a mantissa of all ones. Use `& ~0x1FFF` alone and you have the
truncating path.

## Determinism

The section answers why a program that does not change can give different
numbers, and what you can do about each cause.

Float addition isn't associative, so any change in the order of a sum can change
the result. A GPU changes the order in three ways.

1. **Atomics.** `atomicAdd` on a float adds whenever the instruction arrives, and
   which block arrives first depends on scheduling.
2. **Split reductions.** A reduction or GEMM that splits its work across blocks,
   or across a number of blocks that depends on the problem shape, combines the
   partials in a library-chosen order.
3. **Algorithm selection.** cuDNN and cuBLAS pick an algorithm by shape, by
   benchmark, or by the workspace they are given, and different algorithms add in
   different orders.

### atomicAdd

The one-pass reduction of chapter 6 reduces each block with a tree and then does
a single `atomicAdd(out, block_total)`. The tree is deterministic. The
`atomicAdd` is not, because the blocks finish in an order set by DRAM timing and
the scheduler:

```cuda
if (threadIdx.x == 0) {
    atomicAdd(out, block_total);   // the order of these adds varies run to run
}
```

With 16,384 blocks, the sum is the same set of numbers added in a different order
each time. Every order has a different rounding pattern, so the last bits move.
==The result is correct to within the error bound and still not repeatable==.

Your lab counts the distinct results over 50 runs on the same input. Expect more
than one on most runs, but not a guarantee: a quiet card can land on one order
by luck, and the lab prints a note when it does. That is itself the lesson, since
a race that happens to resolve the same way every time you test it will not do
so in production.

### The fixed-order alternative

The deterministic version never lets the order depend on timing. Write the
partial sums to a buffer, one slot per block, and reduce the buffer with another
kernel in a fixed order:

```cuda
// pass 1: each block reduces its slice with a tree, then thread 0 stores it
if (threadIdx.x == 0) out[blockIdx.x] = total;
// pass 2 (and 3, ...): the same kernel, on the partials, until one remains
```

Every launch has a fixed shape and every add has a fixed partner, so two runs
agree bit for bit. The price is an extra launch or two, which costs a few
microseconds each and little else, because the second pass reads one float per
block. The lab's tree reduction is this design, and the test asserts that 20
runs on 4 million values return one bit pattern.

Other fixed-order tricks exist:

- **Sort or bucket before adding,** so equal sets give equal orders.
- **Add integers.** Convert each float to a fixed-point `int64` with a scale and
  add the integers with `atomicAdd`. Integer addition is associative, so the
  result doesn't depend on order. You pay in range and in one more rounding.
- **Let the framework do it.** `torch.use_deterministic_algorithms(True)` tells
  PyTorch to choose deterministic implementations where they exist and to raise
  an error for an operation that has none. The PyTorch
  [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
  list the affected operations. Setting
  `torch.backends.cudnn.benchmark = False` makes cuDNN pick its algorithm
  deterministically, and `torch.backends.cudnn.deterministic = True` forces
  deterministic convolutions.

### Batch invariance in inference

For a served model the main source of variation isn't run-to-run atomics. It is
that the same request returns different numbers depending on how many other
requests share its batch.

The Thinking Machines post
[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
argues that atomics are rarely used in the forward pass, so a forward pass is
deterministic run to run, but the kernels are not *batch invariant*. A
reduction in RMSNorm, a matmul, or an attention kernel chooses a different split
and a different order for a different batch size, and the batch size depends on
server load. Under that load, a request is nondeterministic from the user's side
even though each kernel is deterministic.

The authors report, on Qwen3-235B-A22B-Instruct-2507 at temperature 0, 80 unique
completions in 1000 samples of the same prompt, and 1000 identical ones with
batch-invariant kernels. Their fix is a fixed reduction order per row for
RMSNorm, one matmul configuration for every shape, and a fixed split size for
attention. They estimate about 20% slower matmuls than cuBLAS. The numbers are
theirs, measured on their setup, not on the lab card.

The lesson for a kernel writer is the same as for the lab: a reduction whose
order doesn't depend on the batch, the grid, or timing is repeatable, and you
pay for it in a lower peak. That is a design choice to make on purpose.

## What goes wrong

- **Accumulating in the storage format.** A bf16 sum in a bf16 accumulator
  stalls at 256 equal addends. Cast the inputs to fp32, accumulate there, and
  round once at the end. The lab's Part 2 shows the failure at 90% error.
- **Casting past the FP8 maximum.** A cast to `torch.float8_e4m3fn` of a value
  above the largest finite number, after rounding, returns NaN rather than 448.
  Clamp first. E5M2 returns infinity.
- **Trusting `--use_fast_math` blindly.** It flushes subnormals to zero, makes
  division approximate, and swaps `expf` for `__expf`. A kernel that passes on a
  random input can fail on one with tiny values or large exponents.
- **Comparing bit for bit across devices.** A CPU reference without FMA, a
  different thread count, and a different block size each change the order or
  the rounding. Compare with a tolerance scaled by the sum of absolute values,
  as the lab does, and not by one fixed number.
- **Seeding and calling it deterministic.** `torch.manual_seed` fixes the random
  inputs, not the order of a floating-point sum. Reproducibility needs the
  algorithm to be deterministic too.
- **Reading one clean run as proof.** An `atomicAdd` sum that matched in ten
  runs can differ on the eleventh, on another card, or under load.
- **Letting an optimizer undo your trick.** Kahan on a CPU under `-ffast-math`
  collapses to a plain sum. Test that the compensated sum is more accurate, not
  that the code contains the compensation.

> [!RECAP]
> - A float is $(-1)^S \, 2^{E - \text{bias}} (1 + M/2^m)$. The mantissa width
>   sets the unit roundoff $2^{-(m+1)}$: $2^{-24}$ for fp32, $2^{-8}$ for bf16,
>   $2^{-4}$ for E4M3. The exponent width sets the range.
> - Rounding is nearest, ties to even, except where hardware truncates: the
>   `wgmma` path on raw fp32 inputs drops the low 13 bits of a TF32 value, and
>   `cvt.rna` rounds ties away from zero.
> - `--use_fast_math` bundles flush-to-zero, approximate division, and `__expf`.
>   An FMA, which `nvcc` emits by default, rounds once instead of twice.
> - A left-to-right sum in a $p$-bit format stops at about $2^p$ equal addends.
>   A pairwise tree gives an error bound of $\lceil\log_2 n\rceil\,u$, and Kahan
>   compensation gives about $2u$.
> - Accumulate in fp32. TF32 is off by default for fp32 `matmul` in PyTorch.
> - `atomicAdd` and shape-dependent splits make sums depend on timing and on
>   batch size. A fixed-order tree is repeatable and nearly free.

## Check your understanding

> [!QUESTION] What is the largest finite value of FP8 E4M3, and why isn't it $2^{8} \times 1.875 = 480$?
> The largest value is 448, $2^{8} \times 1.75$. In the "fn" variant the top
> exponent field holds finite numbers, but the all-ones mantissa with the
> all-ones exponent is reserved for NaN, since the format has no infinity. That
> removes the mantissa $111_2$, which would have given 1.875, so the largest
> usable mantissa is $110_2 = 1.75$.

> [!QUESTION] You add 1 to a bfloat16 accumulator holding 512. What do you get, and what if the accumulator holds 255?
> The ulp in $[512, 1024)$ is $2^{9-7} = 4$, so 513 is a quarter of the way to
> 516 and rounds back to 512: the addend vanishes. At 255, which is in
> $[128, 256)$ with an ulp of 1, the sum 256 is exactly representable, so you
> get 256.

> [!QUESTION] Estimate where a sequential fp16 sum of 5000 values uniform in $[0, 1)$ stalls.
> The significand has $p = 11$ bits. In $[2048, 4096)$ the ulp is 2, so half an
> ulp is 1, and every addend below 1 rounds away. The sum therefore stops near
> $2^{11} = 2048$, while the true sum is about 2500.

> [!QUESTION] The sequential fp32 error bound for $n = 2^{20}$ is 0.0625, yet you measure about $10^{-5}$. Why isn't the bound reached?
> The bound assumes every rounding errs in the same direction by the maximum
> amount. Real rounding errors have random signs, so they add like a random
> walk. Summing $\mathrm{ulp}^2/12$ over the terms predicts a relative standard
> deviation of about $1.3 \times 10^{-5}$ for uniform data.

> [!QUESTION] Your CUDA kernel and a NumPy reference differ by 1 ulp on `a * b + c`. Name two causes and a test to tell them apart.
> The GPU fuses the multiply-add into one rounding while NumPy rounds twice, and
> a reduction order differs. Compile with `-fmad=false` or use `__fmul_rn` and
> `__fadd_rn`: if the gap closes, it was FMA. If a kernel with no reduction still
> differs, the cause is the math function, such as `--use_fast_math`'s `__expf`.

> [!QUESTION] A `scatter_add` on the GPU gives a different loss each run. What is the cause, and what are two ways to fix it?
> It uses `atomicAdd`, so duplicate indices are added in an order that varies.
> One fix is `torch.use_deterministic_algorithms(True)`, which picks a
> deterministic implementation or raises an error. Another is to sort the
> indices and reduce each run of equal keys in a fixed order, or to accumulate
> in fixed-point integers where the order cannot matter.

> [!QUESTION] Two requests with the same prompt, at temperature 0, return different text on a loaded server but identical text on an idle one. Is `atomicAdd` the cause?
> Probably not. If every kernel is deterministic run to run, load still changes
> the batch size, and a kernel that picks its reduction split by batch size adds
> in a different order. That is a lack of batch invariance, and the fix is a
> reduction order that doesn't depend on the batch.

## Lab

> [!TRY] Numerics on the GPU
> Open `09-numerics`. In `starter.py` you write four things: the format
> functions (`format_stats`, `round_tf32`, `roundtrip`, `measured_roundoff`), the
> two bfloat16 sums (`sum_bf16_sequential`, `sum_bf16_pairwise`), three CUDA
> kernels (Kahan, a fixed-order tree, and a one-pass `atomicAdd`), and two probes
> of `--use_fast_math`. You pass when the format limits match `torch.finfo`, 1024
> ones sum to exactly 256 sequentially, the Kahan sum errs by under $3 \times 10^{-7}$ on
> $2^{20}$ values, the tree sum returns one bit pattern in 20 runs, and each sum is
> correct on awkward sizes such as 1, 7, and 1,000,003.
> The lab reports `bf16_roundoff`, `e4m3_roundoff`, `bf16_seq_rel_err`,
> `seq_rel_err`, `kahan_rel_err`, `tree_rel_err`, `atomic_distinct`, `tree_gbs`,
> `expf_ulp`, and `fast_expf_ulp`. A sum reads each float once, so
> the lab sets only a loose floor for `tree_gbs`: a quarter of the 1275 GB/s copy
> ceiling, which is 0.25 x 1275 = 319 GB/s, and it reports the real figure. Predict
> `bf16_seq_rel_err` before you run it, using the stall argument.

## Further reading

- [Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point/index.html):
  FMA, order of operations, and why GPU and CPU results differ.
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html),
  David Goldberg, 1991: rounding, ulp, and cancellation.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740), Micikevicius and
  others, 2017: fp16 loss scaling and fp32 master weights.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), Micikevicius
  and others, 2022: why E4M3 for forward values and E5M2 for gradients.
- [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322),
  Kalamkar and others, 2019: why bf16 needs no loss scaling.
- [Accelerating AI Training with NVIDIA TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/):
  the TF32 layout.
- [OCP 8-bit Floating Point Specification (OFP8)](https://opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
  and the
  [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537),
  Rouhani and others, 2023.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/),
  Thinking Machines, 2025.
- [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/abs/2206.02874),
  Sun and others.
