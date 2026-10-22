---
title: Softmax and normalization
slug: 15-softmax-and-normalization
part: "Part 4 — Kernels for ML"
summary: Why softmax, LayerNorm, and RMSNorm are bandwidth-bound, how online softmax and Welford statistics keep them to the minimum number of passes, and how to derive and reduce the LayerNorm backward pass.
minutes: 150
gpu: true
objectives:
  - "Count the bytes a softmax or normalization moves, and turn the count into a time floor against the measured copy ceiling."
  - "Derive the online softmax recurrence and the merge of two partial (max, sum) states, and say why the merge is associative."
  - "Choose between one row per program, several rows per program, and a loop over chunks, from the row length and the register budget."
  - "Compute LayerNorm statistics with Welford's update and Chan's merge, and explain why E[x^2] - E[x]^2 loses digits."
  - "Derive dx, dgamma, and dbeta for LayerNorm, and write the RMSNorm variant."
  - "Reduce dgamma and dbeta across rows with a two-stage scheme, and compare it with atomics and locks."
  - "Fuse a residual add and dropout into a normalization, and predict the byte saving."
  - "Explain why bf16 kernels keep their statistics and accumulators in fp32."
lab: 15-softmax-layernorm
---

# Softmax and normalization

> [!TLDR]
> - Softmax, LayerNorm, and RMSNorm do a handful of FLOPs per element and move
>   2 to 8 bytes per element, so they're bandwidth-bound. Their speed is a byte
>   count divided by the measured copy ceiling of 1275 GB/s.
> - Subtract the row maximum before you exponentiate. Online softmax then keeps a
>   running maximum and a running sum, and rescales the sum by
>   $e^{m_{\text{old}} - m_{\text{new}}}$ whenever the maximum grows. That cuts
>   the passes over a long row from three to two.
> - LayerNorm statistics come from Welford's update, merged across chunks with
>   Chan's formula. The shortcut $E[x^2] - E[x]^2$ loses most of its digits when
>   the mean is large.
> - The backward pass is $dx = r\,(g - \hat{x}\,\overline{g\hat{x}} - \overline{g})$.
>   The gradients for the weight and bias are column sums over every row, and a
>   two-stage reduction computes them without atomics or locks.
> - Keep statistics and accumulators in fp32 even when the tensors are bf16.

Every transformer layer runs two normalizations, and every attention head runs a
softmax. None of them is much arithmetic. On the A100 they cost what it takes to
read and write the tensor, which is why a fused kernel that makes one trip over
the data beats a chain of PyTorch operations that makes eight.

This chapter builds those kernels in Triton. It starts with the byte count that
fixes the target, then derives the two algorithms that make the byte count
achievable: online softmax and Welford statistics. After that comes the backward
pass, which is where normalization kernels get hard, because the weight gradient
sums over every row in the batch.

The online softmax recurrence is also the core of
[FlashAttention](/c/16-flash-attention), and the fusion habits here carry into
[chapter 17](/c/17-fusion-and-compilers) and [chapter 19](/c/19-training-kernels).

## Before you start

**The roofline.** A kernel is *bandwidth-bound* when its arithmetic intensity, in
FLOP per byte, is below the ridge point. [Chapter 4](/c/04-roofline-and-latency)
derives the ridge point, and [chapter 3](/c/03-memory-and-bandwidth) measures the
copy ceiling. This chapter uses the measured ceiling of 1275 GB/s on the lab
A100 80GB PCIe, from the companion course's
[roofline chapter](https://qwen.ifkash.dev/c/10-roofline).

**Triton's program model.** One *program* handles one tile of data, written with
`tl.arange`, masks, and `tl.load` and `tl.store`. [Chapter 14](/c/14-triton)
introduces it. You need `tl.sum`, `tl.max`, and loops with a runtime bound.

**Reductions.** A row-wise `tl.sum` reduces across the lanes of a program with
warp shuffles and shared memory, the pattern from
[chapter 6](/c/06-reductions). You don't write that code here, but you need to
know it costs a few synchronizations per reduction.

**Floating-point formats.** bf16 keeps 7 explicit mantissa bits, so its machine
epsilon is $2^{-7} = 0.0078$. fp32 keeps 23, and its epsilon is $1.19 \times 10^{-7}$.
[Chapter 9](/c/09-floating-point-on-the-gpu) covers rounding and accumulation.

## Why these kernels wait on memory

This section answers one question: how long should a softmax or a normalization
take, before you write any code?

Count the bytes, because the arithmetic is negligible. A LayerNorm forward pass
does about four FLOP per element to normalize and scale (subtract, multiply,
multiply, add), plus a few more to gather statistics. Say ten FLOP per element, to
be generous. A bf16 element costs 2 bytes in and 2 bytes out, so the intensity is
$10 / 4 = 2.5$ FLOP per byte.

Compare that with the ridge point. These kernels run on the CUDA cores, not the
tensor cores, so the relevant peak is the fp32 rate of 19.5 TFLOP/s. Against the
rated 1935 GB/s, the ridge is $19.5 \times 10^{12} / 1.935 \times 10^{12} = 10.1$
FLOP per byte. Against the measured 1275 GB/s copy, it's 15.3. A kernel at 2.5
FLOP per byte sits well below both, so ==memory bandwidth sets the time==.

The time floor is the bytes moved divided by the bandwidth:

$$
\boxed{t_{\min} = \frac{\hla{\text{bytes moved}}}{\hlb{\beta}}}
$$

Here $\hla{\text{bytes moved}}$ counts every byte read from and written to HBM, and
$\hlb{\beta}$ is the bandwidth, which is 1275 GB/s for the measured ceiling.
Everything in the chapter is a way to shrink the numerator.

> [!EXAMPLE] The cost of three ways to run softmax
> Take a bf16 tensor of 8192 rows by 16384 columns. It holds
> $8192 \times 16384 = 134{,}217{,}728$ elements, and at 2 bytes each one *volume*
> (one full pass over the tensor) is 268.4 MB. The floor for each strategy is its
> volume count times 268.4 MB, divided by 1275 GB/s:
>
> | Strategy | Volumes moved | Bytes | Floor at 1275 GB/s |
> |---|---|---|---|
> | Eager PyTorch, one kernel per operation (8 volumes) | 5 reads, 3 writes | 2147 MB | 1.684 ms |
> | Three-pass fused kernel | 3 reads, 1 write | 1074 MB | 0.842 ms |
> | Two-pass online softmax | 2 reads, 1 write | 805 MB | 0.632 ms |
> | One pass, the row held in registers | 1 read, 1 write | 537 MB | 0.421 ms |
>
> The eager row counts `max`, `sub`, `exp`, `sum`, and `div` as five separate kernels,
> each reading its input and writing its output (a reduction writes almost
> nothing). The floors are predictions from the byte counts. Your lab reports the
> measured time.

The table also shows what each trick buys. Fusing the five kernels into one
three-pass kernel cuts the floor by 2x. Online softmax cuts it by another 1.33x. Holding
the whole row in registers reaches the 2-volume minimum, but only works while a row fits.

The same arithmetic applies to normalization. A LayerNorm forward pass over a
bf16 tensor of $16384 \times 4096$ elements (134.2 MB) reads x and writes y: two
volumes, 268.4 MB, a floor of 0.211 ms at 1275 GB/s and 0.139 ms at the rated 1935
GB/s. The weight, bias, mean, and reciprocal standard deviation add under 0.2 MB.
The backward pass reads x and dy and writes dx, three volumes, with a floor of 0.316
ms.

The companion course measured this on the lab card. In
[its Triton chapter](https://qwen.ifkash.dev/c/13-fusion-in-triton), a fused
RMSNorm over 4096 rows of 5120 bf16 values (41.9 MB) ran in 0.089 ms. That's
945 GB/s counting one read and one write, which is 74% of the copy ceiling and
49% of the rating. Treat 74% as what a well-tuned row-wise kernel reaches on this
card, not as a promise, and recall that a tensor this small partly lives in the 40
MiB L2 cache.

> [!NOTE] Why the lab uses large tensors
> A tensor under 40 MiB can stay in L2 between timing iterations, which makes
> the measured bandwidth look better than HBM can deliver. The lab's performance
> tensors are 134 MB and 268 MB, which are larger than L2, so no cache flush is
> needed.

## Softmax numerics

This section answers one question: why does every softmax subtract the maximum?

The softmax of a vector $x$ with $N$ elements is

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}.
$$

Read the formula as a probability for each element, proportional to $e^{x_i}$. The
trouble is that $e^{x}$ overflows quickly. In fp32 and bf16, which share an 8-bit
exponent, $e^{x}$ overflows to infinity for $x > \ln(3.4028 \times 10^{38}) = 88.7$.
In fp16, whose largest finite value is 65504, it overflows for $x > 11.09$. A logit
of 12 breaks an fp16 softmax, and attention logits can be larger than that.

Softmax is unchanged when you add the same constant $c$ to every input, because
$e^{x_i + c}$ is $e^{c} e^{x_i}$ and the factor cancels between numerator and
denominator. Pick $c = -\max_j x_j$:

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j.
$$

Now every exponent is at most 0, so every term is in $(0, 1]$ and nothing
overflows. The largest term is exactly $e^0 = 1$, so the denominator is at least
1 and never underflows to a division by zero. This is the *safe softmax*, and it's
what PyTorch computes.

The safe form needs the maximum before it can compute anything else. That's the
origin of the three passes of the straightforward kernel:

1. Read the row and find $m$.
2. Read the row again and sum $e^{x_j - m}$ to get the denominator $d$.
3. Read the row a third time, write $e^{x_i - m} / d$.

> [!WARNING] Overflow hides in tests with small logits
> A softmax kernel that skips the max subtraction passes every test with standard
> normal inputs, where the largest logit is around 4. It fails in production when
> a logit reaches 89. The lab's tests add 1000 to the logits to catch this.

## From three passes to one: online softmax

This section answers one question: can a kernel get the maximum and the
denominator in one pass?

The maximum changes as you read, and every term of the denominator depends on it.
Milakov and Gimelshein's
[online normalizer](https://arxiv.org/abs/1805.02867) observes that you can
repair the denominator whenever the maximum changes, so one pass produces both.

Define, for the first $j$ elements, the running maximum $\hla{m_j}$ and the running denominator
$\hlb{d_j}$:

$$
\hla{m_j} = \max_{i \le j} x_i, \qquad \hlb{d_j} = \sum_{i \le j} e^{x_i - \hla{m_j}}.
$$

The goal is a recurrence from $j - 1$ to $j$. The maximum is direct:
$m_j = \max(m_{j-1}, x_j)$. For the denominator, split off the new term and then
change the base of the old sum from $m_{j-1}$ to $m_j$:

$$
d_j = \sum_{i \le j-1} e^{x_i - m_j} + e^{x_j - m_j}
    = \Big(\sum_{i \le j-1} e^{x_i - m_{j-1}}\Big) e^{m_{j-1} - m_j} + e^{x_j - m_j}.
$$

The step multiplies and divides by $e^{m_{j-1}}$ inside the old sum. The bracket
is $d_{j-1}$, so

$$
\boxed{\hlb{d_j} = \hlb{d_{j-1}}\, e^{\hla{m_{j-1}} - \hla{m_j}} + e^{x_j - \hla{m_j}}}
$$

Read it as follows. When the maximum doesn't change, the factor is $e^0 = 1$ and
the update adds only the new term. When a new maximum arrives, the factor
$e^{m_{j-1} - m_j}$ is less than 1 and shrinks the old sum to the new scale, as if
you had used the new maximum from the start. After $N$ elements, $m_N$ is the true
maximum and $d_N$ is the true denominator.

### Blocks, not elements

A GPU kernel can't process one element at a time. It reads a block of
`BLOCK` columns, reduces the block, and merges the block's result into the running
state. A block has its own local maximum $m_b$ and local sum
$d_b = \sum_{i \in b} e^{x_i - m_b}$. To merge two states $(m_a, d_a)$ and
$(m_b, d_b)$, rescale both to the larger maximum. Write the merge as $(m, d) = (m_a, d_a) \oplus (m_b, d_b)$:

$$
m = \max(m_a, m_b), \qquad d = d_a\, e^{m_a - m} + d_b\, e^{m_b - m}.
$$

This merge operator is *associative*, because it computes the denominator for the
union of the two element sets, and union is associative. That matters in two ways.
Inside a program, Triton can merge the lanes' partial states in a tree, in any
order. Across programs, you can split one row among several programs and merge
their states afterward. [FlashAttention](/c/16-flash-attention) merges a running
state for each tile of keys, and flash-decoding merges states across programs.

The figure below steps through the recurrence on twelve logits. The second block
contains a larger value than the first, so the running denominator shrinks by
$e^{m_{\text{old}} - m_{\text{new}}}$ at that step. Change the block size and step
through again.

```viz
15-online-softmax
```

### The kernel

The lab's kernel follows the recurrence directly. Pass 1 streams the row once and
keeps two scalars, $m$ and $d$, with no row held in registers. Pass 2 re-reads the
row and writes the result:

```python
@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols

    m = float("-inf")
    d = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(y_row + cols, (tl.exp(x - m) / d).to(y_ptr.dtype.element_ty), mask=mask)
```

Three details carry the correctness:

- **The masked value is $-\infty$.** A lane past the end of the row loads
  $-\infty$, which never wins the maximum and contributes $e^{-\infty} = 0$ to the
  sum. Loading 0.0 would add $e^{0 - m}$ to the denominator for each padded lane.
- **The first rescale is harmless.** On the first block, $m = -\infty$, so
  $e^{m - m_{\text{new}}} = 0$ and the zero initial $d$ stays zero.
- **The arithmetic is fp32.** The `.to(tl.float32)` runs before the first
  subtraction, so a bf16 input still accumulates in fp32.

The cost of the extra rescale is one `exp` per block, not per element, so it's
negligible. The online version does more arithmetic per block than the three-pass
one, but the arithmetic was never the bottleneck. Milakov and Gimelshein report an
improvement of up to 1.3x for the softmax alone; the paper's hardware differs from the
lab's A100, so treat 1.3x as the paper's number and read the ratio your lab reports.
The byte count above predicts 1074 MB over 805 MB, which is 1.33x.

> [!KEY] Online softmax removes one trip over memory
> The recurrence lets one pass compute both the maximum and the denominator. A long
> row now costs two reads and one write, 3 volumes, instead of three reads and one
> write. If the second read hits L2, it costs even less.

## One row per program, or many

This section answers one question: how many rows should one program handle, and
how many columns at a time?

The choice depends on the row length $N$. The rule of thumb is to give each thread
a modest number of elements and each SM enough resident programs to hide memory
latency ([chapter 4](/c/04-roofline-and-latency)). The following table is a
starting point to autotune, not a law:

| Row length | Layout | Why |
|---|---|---|
| $N \le 256$ | A 2D tile of several rows per program | One row doesn't give a warp enough work, so lanes sit idle. |
| 256 to 16384 | One row per program, the whole row in registers, `num_warps` $\approx N/256$ up to 8 | One load, one reduction, one store. |
| $N > 16384$ | One row per program, a loop over chunks of 4096 | Holding the whole row in registers would exceed the register budget. |
| Few rows, very long | Several programs per row, merged with the $\oplus$ operator | Otherwise most SMs sit idle. |

### Short rows: several rows per program

A row of 128 elements handled by a 4-warp program uses 128 threads for 128
elements, one element each. The reduction then spends more time on shuffles than
on loads. Instead, write a tile of `[BLOCK_M, BLOCK_N]` with `BLOCK_M` rows, and
reduce along axis 1. One program now covers several rows, and each warp can take
its own row. The kernel changes from `tl.sum(x, axis=0)` on a vector to
`tl.sum(x, axis=1)` on a matrix, and the pointers gain a row index.

### Long rows: a loop over chunks

A program that holds its entire row in registers needs one fp32 register per
element, at minimum. The A100 allows 255 registers per thread, and the SM has a
file of 65,536 registers to share among resident programs. Take a row of 131072
elements and 8 warps, which is 256 threads. Each thread would hold
$131072 / 256 = 512$ values. That exceeds 255, so the compiler spills to local
memory.

Even a row that fits crowds the SM. At 16384 elements and 8 warps, each thread
holds 64 values, so a program uses $64 \times 256 = 16384$ registers at minimum.
The register file holds $65536 / 16384 = 4$ such programs per SM. A loop over
chunks of 4096 holds only 16 values per thread, and the same arithmetic gives
65536 / 4096 = 16 programs by registers, capped at 8 by the 2048-thread limit
($2048 / 256$). More resident programs hide more latency.

```viz
15-row-per-program
```

The figure above does this arithmetic for a row length and a warp count. The numbers
are minimums, because the real kernel needs registers for addresses and
temporaries too, so the true count of resident programs is at most what the
figure shows.

The chunked loop reads each row twice, once per pass. When a row is small enough that
the second read hits L2, the loop costs nothing extra. When it isn't, you pay the
third volume in the table. The companion course saw the first case: a tiled
RMSNorm with `BLOCK=2048` that re-read each 5120-element row ran in 0.090 ms
(929 GB/s), within 1% of the single-block kernel's 945 GB/s, because the row stayed
in cache.

### How many warps

More warps mean more threads to hide latency, and also more shuffles per
reduction. The companion course swept `num_warps` for 5120-wide rows: 8 warps
gave 945 GB/s, 16 gave 924, and 32 gave 878. For rows of a few thousand elements, 4
to 8 warps is the right range, and `min(max(block // 256, 1), 8)` is a sound default.
Autotune from there.

### Few rows

A softmax over 32 rows of 131072 columns, such as attention scores for a batch of
decode queries, launches 32 programs. The A100 has 108 SMs, so 32 programs occupy
at most $32 / 108 = 30\%$ of them. Each program also loops over 32 chunks of 4096 one
after another. Split each row across several programs, let each compute a partial
$(m, d)$, and merge with the $\oplus$ operator. Flash-decoding applies this idea to
attention ([chapter 16](/c/16-flash-attention)).

## LayerNorm and RMSNorm forward

This section answers one question: how does one program compute a row's mean and
variance in a single trip?

LayerNorm, from [Ba, Kiros, and Hinton](https://arxiv.org/abs/1607.06450),
normalizes each row to zero mean and unit variance, then applies a learned scale
$w$ (often called gamma) and shift $b$ (beta):

$$
\hat{x}_i = (x_i - \mu)\, r, \qquad y_i = w_i \hat{x}_i + b_i,
\qquad \mu = \tfrac{1}{N}\sum_i x_i, \quad r = \frac{1}{\sqrt{\sigma^2 + \varepsilon}}.
$$

Here $\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2$ is the variance, $\varepsilon$ is a
small constant, and $r$ is the reciprocal standard deviation, `rstd` in the code. The
kernel stores $\mu$ and $r$ for the backward pass: 8 bytes per row, which is
0.1% of a 4096-column bf16 row.

[RMSNorm](https://arxiv.org/abs/1910.07467) drops the mean and the shift. It
divides by the root mean square and applies only a scale:

$$
r = \frac{1}{\sqrt{\tfrac{1}{N}\sum_i x_i^2 + \varepsilon}}, \qquad y_i = w_i\, x_i\, r.
$$

RMSNorm needs one moment where LayerNorm needs two, so it has less to compute and
less to store. It also has no cancellation problem in the statistic, because it sums
squares that are all positive.

### Why the obvious variance fails

The two-moment shortcut computes $E[x]$ and $E[x^2]$ in one pass and sets
$\sigma^2 = E[x^2] - E[x]^2$. It's algebraically right and numerically fragile.
Take a row whose mean is 1000 and whose variance is 1. Then $E[x^2] \approx 10^6 + 1$
and $E[x]^2 = 10^6$, so the true answer is the small difference of two large
numbers.

fp32 spacing near $10^6$ is $2^{-4} = 0.0625$, because $10^6$ lies between $2^{19}$
and $2^{20}$ and fp32 keeps 23 mantissa bits. Each rounding of $E[x^2]$ can therefore
move it by about 0.03, against a true variance of 1. The subtraction cancels the leading
digits and keeps only the rounding error. The lab's test of a mean-1000 row exists
to catch this.

### Welford's update

Welford's algorithm tracks the count $n$, the mean, and $M_2 = \sum (x - \text{mean})^2$,
the sum of squared deviations from the *current* mean. For each new element:

$$
n \leftarrow n + 1, \qquad \delta = x - \text{mean}, \qquad
\text{mean} \leftarrow \text{mean} + \frac{\delta}{n}, \qquad
M_2 \leftarrow M_2 + \delta\,(x - \text{mean}).
$$

The update uses the new mean in the last product, so $M_2$ only ever adds a
non-negative quantity. Nothing subtracts two large nearly-equal numbers. A GPU
kernel can't run this one element at a time, so it uses the block form.

### Chan's merge

Given two groups with statistics $(n_a, \mu_a, M_{2a})$ and $(n_b, \mu_b, M_{2b})$,
Chan, Golub, and LeVeque's parallel formula combines them:

$$
\boxed{n = n_a + n_b, \quad \delta = \mu_b - \mu_a, \quad
\mu = \mu_a + \delta\,\frac{n_b}{n}, \quad
M_2 = M_{2a} + M_{2b} + \delta^2\,\frac{n_a n_b}{n}}
$$

The last term is the extra variance from the two groups having different means. If
$\mu_a = \mu_b$, then $\delta = 0$ and the sums of squares add directly.

> [!EXAMPLE] Merging two chunks by hand
> Chunk A is $[1, 2, 3]$: $n_a = 3$, $\mu_a = 2$, $M_{2a} = 2$. Chunk B is $[10, 12]$:
> $n_b = 2$, $\mu_b = 11$, $M_{2b} = 2$. Then $n = 5$ and $\delta = 9$, so
> $\mu = 2 + 9 \cdot 2/5 = 5.6$ and $M_2 = 2 + 2 + 81 \cdot 6/5 = 101.2$. Directly,
> the five values have mean $28/5 = 5.6$, and the squared deviations are
> $21.16 + 12.96 + 6.76 + 19.36 + 40.96 = 101.2$.

> [!DEEPDIVE] Why Chan's merge is correct
> Let $\mu = (n_a \mu_a + n_b \mu_b)/n$ be the merged mean. For group $a$,
> $\sum_a (x - \mu)^2 = M_{2a} + n_a(\mu_a - \mu)^2$, because the cross term vanishes
> when you expand around $\mu_a$. The same holds for $b$. Since
> $\mu_a - \mu = -\delta\, n_b / n$ and $\mu_b - \mu = \delta\, n_a / n$, the two
> correction terms are
>
> $$
> n_a \delta^2 \frac{n_b^2}{n^2} + n_b \delta^2 \frac{n_a^2}{n^2}
> = \delta^2 \frac{n_a n_b (n_b + n_a)}{n^2} = \delta^2 \frac{n_a n_b}{n}.
> $$
>
> Adding $M_{2a} + M_{2b}$ gives the boxed formula. Like the softmax merge, it's
> associative, so a block reduction can combine partial states in any tree shape.

### The forward kernel

The lab's forward kernel computes each chunk's mean and $M_2$ from the chunk's own
deviations, then merges the chunk into the running state with Chan's formula. It then
makes a second pass to normalize:

```python
count = 0.0
mean = 0.0
m2 = 0.0
for start in range(0, n_cols, BLOCK):
    cols = start + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    n_b = tl.minimum(n_cols - start, BLOCK).to(tl.float32)
    mean_b = tl.sum(x, axis=0) / n_b
    centered = tl.where(mask, x - mean_b, 0.0)
    m2_b = tl.sum(centered * centered, axis=0)
    n_a = count
    count = n_a + n_b
    delta = mean_b - mean
    mean = mean + delta * (n_b / count)
    m2 = m2 + m2_b + delta * delta * (n_a * n_b / count)
var = m2 / n_cols
rstd = 1.0 / tl.sqrt(var + eps)
```

The `tl.where` after the subtraction is the important line. A masked lane loaded
0.0, and $0 - \text{mean}_b$ isn't zero, so without the `where` the padded lanes add
$\text{mean}_b^2$ each to $M_2$. For a row whose length is a multiple of `BLOCK`,
this bug never fires, which is why the lab tests a row of 4097.

When the whole row fits in one block, you can skip the merge. Compute the mean with
one reduction, then the variance from the centered values in registers. The Triton
[layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
does that with a loop over blocks and a second pass for the variance. It's stable,
because it subtracts the exact mean before it squares, and it costs a second
reduction. The Welford merge handles long rows in one pass over the statistics.

## The LayerNorm backward pass

This section answers one question: given the gradient $dy$ of the loss with respect
to the output, what are $dx$, $dw$, and $db$?

Start with the parameters, which are direct. Since $y_i = w_i \hat{x}_i + b_i$, the
gradient for the bias is $dy$ itself and the gradient for the weight is $dy$ times
$\hat{x}$. Both sum over the $M$ rows, because the same $w$ and $b$ apply to every row:

$$
db_j = \sum_{\text{rows}} dy_{\cdot j}, \qquad dw_j = \sum_{\text{rows}} dy_{\cdot j}\,\hat{x}_{\cdot j}.
$$

The input gradient is harder, because every $x_i$ affects the mean and the variance,
and so every output. Define $g_i = dy_i\, w_i$, the gradient with respect to $\hat{x}_i$.
Then the result is

$$
\boxed{dx_i = \hla{r}\Big(g_i - \hlb{\overline{g}} - \hat{x}_i\, \hlc{\overline{g\hat{x}}}\Big)}
$$

where $\hlb{\overline{g}} = \frac{1}{N}\sum_j g_j$ and
$\hlc{\overline{g\hat{x}}} = \frac{1}{N}\sum_j g_j \hat{x}_j$ are two row means. Read it as three
terms: the direct path $g_i$, a correction for the mean (subtract the average
gradient), and a correction for the variance (subtract the part of the gradient that
points along $\hat{x}$).

> [!DEEPDIVE] Deriving dx
> Apply the chain rule through $\hat{x}_j = (x_j - \mu) r$. The mean depends on
> every $x_i$ with $\partial \mu / \partial x_i = 1/N$. The variance satisfies
> $\partial \sigma^2 / \partial x_i = \frac{2}{N}(x_i - \mu)$, because the
> terms that come from differentiating $\mu$ sum to zero (the deviations sum to
> zero). With $r = (\sigma^2 + \varepsilon)^{-1/2}$,
>
> $$
> \frac{\partial r}{\partial x_i} = -\tfrac{1}{2}\, r^3 \cdot \tfrac{2}{N}(x_i - \mu)
> = -\frac{r^2}{N}\,\hat{x}_i,
> $$
>
> using $r (x_i - \mu) = \hat{x}_i$. Now differentiate $\hat{x}_j$ by the product rule:
>
> $$
> \frac{\partial \hat{x}_j}{\partial x_i}
> = r\Big(\delta_{ij} - \frac{1}{N}\Big) + (x_j - \mu)\frac{\partial r}{\partial x_i}
> = r\Big(\delta_{ij} - \frac{1}{N} - \frac{\hat{x}_i \hat{x}_j}{N}\Big).
> $$
>
> Finally, $dx_i = \sum_j g_j\, \partial \hat{x}_j / \partial x_i$, which expands to
> $r\big(g_i - \frac{1}{N}\sum_j g_j - \frac{\hat{x}_i}{N}\sum_j g_j \hat{x}_j\big)$.
> Two sanity checks follow. $\sum_i dx_i = 0$, because shifting the whole row doesn't
> change $y$. And $\sum_i dx_i \hat{x}_i = 0$ for the same reason with scaling.

For RMSNorm, the mean term drops out. With $\hat{x}_i = x_i r$ and
$r = (\overline{x^2} + \varepsilon)^{-1/2}$, the same steps give
$dx_i = r\,(g_i - \hat{x}_i\,\overline{g\hat{x}})$ and $dw_j = \sum_{\text{rows}} dy_{\cdot j}\,\hat{x}_{\cdot j}$,
with no bias.

### What the kernel stores and recomputes

The backward pass needs $\hat{x}$. Saving it costs a whole tensor, so the kernel saves
only $\mu$ and $r$ (8 bytes per row) and recomputes $\hat{x} = (x - \mu) r$ from the
input x it reads anyway. That's one subtract and one multiply per element, which is
free against the bandwidth. The backward kernel then reads x and dy and writes dx:
three volumes, plus the small weight and partial-sum traffic.

The row reductions $\overline{g}$ and $\overline{g\hat{x}}$ are two more `tl.sum`
calls over registers that already hold the row. The per-row work is cheap. The
expensive part is the column sums.

## Reducing dw and db across rows

This section answers one question: how do thousands of rows add into one weight
gradient of $N$ values?

The row-wise work parallelizes across rows with no communication, but $dw_j$ and
$db_j$ are sums over all $M$ rows. A batch of 16384 tokens makes $M = 16384$, and a
million tokens in a long-context run makes $M$ a million. A naive kernel lets
every program add its row into $N$ shared addresses. The three practical options are
the following:

| Scheme | Deterministic | Extra memory | Cost and risk |
|---|---|---|---|
| `tl.atomic_add` into dw and db | No | None | Every row adds into the same $N$ addresses, so atomics serialize on them, and the float summation order varies from run to run. |
| Locks and partial buffers (the Triton tutorial) | No: the order in which rows take a lock varies | `GROUP_SIZE_M` $\times N$ per vector | A spin lock built on `atomic_cas`, plus a counter and a barrier. Error-prone. |
| Two stages, one partial row per program | Yes | $G \times N$ per vector | A second small kernel. No atomics, no locks. |

The [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
uses the second scheme: each row's program adds its partial sums into one of
`GROUP_SIZE_M` buffers chosen by `row % GROUP_SIZE_M`, guarded by a lock, and a second
kernel sums the buffers. The lab uses the third. Make the grid of the first kernel a
fixed number of programs $G$, and give each program a stride-$G$ set of rows. A
program keeps its running $dw$ and $db$ in registers across all its rows and writes one
partial row when it finishes. A second kernel then sums the $G$ partial rows:

```python
@triton.jit
def _layernorm_bwd_dx_kernel(dy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, dx_ptr,
                             dw_part_ptr, db_part_ptr, n_rows, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    groups = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for row in range(pid, n_rows, groups):
        offs = row.to(tl.int64) * n_cols + cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - tl.load(mean_ptr + row)) * tl.load(rstd_ptr + row), 0.0)
        g = dy * w
        c1 = tl.sum(xhat * g, axis=0) / n_cols
        c2 = tl.sum(g, axis=0) / n_cols
        dx = (g - xhat * c1 - c2) * tl.load(rstd_ptr + row)
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        dw_acc += dy * xhat
        db_acc += dy
    tl.store(dw_part_ptr + pid * n_cols + cols, dw_acc, mask=mask)
    tl.store(db_part_ptr + pid * n_cols + cols, db_acc, mask=mask)
```

The cost is the partial buffers. With $G = 4 \times 108 = 432$ programs and
$N = 4096$, each partial buffer is $432 \times 4096 \times 4 = 7.08$ MB in fp32, and
there are two of them, 14.2 MB. The backward pass for a $16384 \times 4096$ bf16
tensor moves 403 MB in its three volumes, so the partials add about 3.5% for the
write, and as much again when the second kernel reads them. They're small enough to
sit in the 40 MiB L2, so the re-read may not reach HBM. The lab's byte count ignores
them, so the bandwidth it reports is a slight underestimate of the real traffic.

Choosing $G$ is a trade. More programs fill the GPU better and give each program
fewer rows, but they make the partial buffers larger and the second kernel longer.
Four programs per SM is a reasonable start.

> [!TIP] Accumulate in fp32 even when dw is bf16
> The partial buffers and the register accumulators are fp32. The final stage casts
> to the weight's dtype once. If you accumulated $dw$ in bf16 across 16384 rows, the
> sum would stall once it passes 256 and the next addend is below the spacing.

## Fusing the residual add and dropout

This section answers one question: what else can a normalization kernel absorb at
no extra read?

A transformer sublayer adds its output $f$ to a residual stream $x$, then normalizes
the sum. Unfused, `s = x + f; y = norm(s)` is an elementwise add (reads x and f,
writes s) followed by a normalization (reads s, writes y). That is 5 volumes. The
fused kernel reads x and f, writes both s (the next residual) and y: 4 volumes. The byte
count predicts a $5/4 = 1.25\times$ speedup.

The companion course measured exactly this fusion on the lab card, and the prediction
failed. At 4096 rows of 5120 columns (41.9 MB), the fused kernel ran at 0.92x, slower
than unfused. At 16384 rows (167.8 MB) it ran at 1.10x, and at 32768 rows (335.5 MB)
1.16x. The explanation is the 40 MiB L2: when the tensor is small, the unfused kernel's
intermediate stays in cache and the saved trip was never a trip to HBM. The crossover
is near the size of L2. Fusion's real saving is bytes that reach HBM, not bytes a kernel
touches, so measure at sizes above 40 MiB.

Dropout fuses the same way. An unfused dropout reads x, writes the output, and writes a
mask, which is a bool tensor of 1 byte per element, half a bf16 volume. The backward
pass reads the mask back. A fused kernel generates the mask in registers with a
counter-based generator such as `tl.rand(seed, offsets)`, which uses the Philox
algorithm. It stores the seed instead of the mask and regenerates the same random
numbers in the backward pass from the same offsets. The mask tensor never touches
memory.

The same idea extends to anything elementwise that sits next to the normalization:
a bias add, a scale, a cast. Every one removes a read and a write that the kernel
would otherwise make. [Chapter 17](/c/17-fusion-and-compilers) shows the compiler
version of this, and [chapter 19](/c/19-training-kernels) applies it to the loss.

## bf16 tensors with fp32 statistics

This section answers one question: which values must stay in fp32 in a bf16 kernel?

bf16 has 8 significant bits, so a bf16 sum stalls early. Add 1 to 256 and the exact
answer, 257, falls between the representable values 256 and 258, and round-to-nearest-even
returns 256. A bf16 accumulator that sums values near 1 stops growing at 256. A
row of 4096 normalized values has a sum of squares near 4096, so a bf16 accumulator
would be off by a large factor. [Chapter 9](/c/09-floating-point-on-the-gpu) derives
this and shows the same stall for a bf16 matrix accumulator.

The rule has four parts:

1. **Load bf16, convert to fp32 immediately.** The `.to(tl.float32)` goes right after
   the `tl.load`, before any subtraction or exponentiation.
2. **Keep every reduction in fp32.** That covers the max, the sum of exponentials,
   Welford's count, mean, and $M_2$, and the row sums in the backward pass.
3. **Keep the saved statistics fp32.** The mean and `rstd` cost 8 bytes per row, so
   there's no reason to round them. Rounding `rstd` to bf16 would scale every output
   in the row by an error of up to $2^{-8}$.
4. **Round once, on the output.** Cast to bf16 at the final `tl.store`, so each value
   rounds one time instead of after every step.

The weights $w$ and $b$ may be bf16 or fp32 depending on the framework. The kernel
upcasts either to fp32 for the multiply. Their gradients are the exception that
matters: sum them in fp32 across rows, as in the previous section, and cast the total
once.

## What goes wrong

- **Skipping the max.** A softmax without the subtraction works on small logits and
  returns `inf` or `NaN` on large ones. Test with logits of 1000.
- **The wrong `other=`.** Use $-\infty$ for a max or a softmax load and 0.0 for a sum.
  Swap them and a padded lane wins the maximum or adds to the sum.
- **Masked lanes after a subtraction.** `x - mean` isn't zero where the load was
  masked to zero. Wrap it in `tl.where(mask, ..., 0.0)` before you square or sum it.
- **Raw moments.** $E[x^2] - E[x]^2$ gives wrong variances when the mean is large
  relative to the standard deviation, and it can even go negative. Use Welford or
  the two-pass form.
- **Forgetting to rescale.** In online softmax, adding the new block's terms to an
  old $d$ without multiplying by $e^{m_{\text{old}} - m_{\text{new}}}$ is correct only
  when the maximum doesn't change. A test where the maximum sits in the last block
  catches this.
- **A row index that overflows int32.** `row * n_cols` exceeds $2^{31} = 2.1 \times 10^9$
  for a tensor of more than 2.1 billion elements, such as a million tokens by 4096
  columns. Convert `row` to int64 before the multiply.
- **Atomics for dw and db.** They give a different result on every run, because
  floating-point addition isn't associative. A training run that you can't reproduce
  is hard to debug.
- **Miscounting the bytes.** A backward pass moves 3 volumes, not 2. Dividing 2
  volumes by the time overstates the bandwidth by half.
- **Timing a tensor that fits in L2.** A tensor under 40 MiB can stay in cache between
  iterations. Time a tensor several times larger than L2.

> [!RECAP]
> - Softmax and normalization move 2 to 8 volumes of the tensor and do almost no
>   arithmetic, so their floor is bytes divided by 1275 GB/s, and fusion lowers the
>   byte count.
> - Subtract the row maximum. Online softmax keeps a running maximum $m$ and
>   denominator $d$, and multiplies $d$ by $e^{m_{\text{old}} - m_{\text{new}}}$ when the
>   maximum grows. Its merge is associative.
> - Use one row per program while the row fits in registers, a loop over chunks when
>   it doesn't, and a multi-row tile for short rows.
> - Compute variance with Welford's update and Chan's merge, never with
>   $E[x^2] - E[x]^2$. RMSNorm needs only one moment.
> - $dx = r\,(g - \overline{g} - \hat{x}\,\overline{g\hat{x}})$, and $dw$ and $db$ are
>   column sums that a two-stage reduction computes deterministically.
> - In bf16 kernels, convert on load, reduce in fp32, store fp32 statistics, and round once on the output.

## Check your understanding

> [!QUESTION] At what logit does a plain `exp` overflow in fp16, and why does subtracting the maximum prevent it?
> fp16's largest finite value is 65504, so $e^{x}$ overflows for $x > \ln 65504 =
> 11.09$. After subtracting the row maximum every exponent is at most 0, so every
> term lies in $(0, 1]$, and the largest term is exactly 1. Softmax is unchanged
> because the shift cancels between numerator and denominator.

> [!QUESTION] For an $8192 \times 16384$ bf16 tensor, what is the time floor of the three-pass fused softmax and of the online softmax at 1275 GB/s?
> One volume is $134{,}217{,}728 \times 2 = 268.4$ MB. The three-pass kernel moves 3
> reads and 1 write, 4 volumes, which is 1074 MB and a floor of 0.842 ms. Online
> softmax moves 3 volumes, 805 MB, with a floor of 0.632 ms. The ratio is 4/3 = 1.33.

> [!QUESTION] Two partial states are $(m_a, d_a) = (2, 3)$ and $(m_b, d_b) = (5, 2)$. What is the merged state?
> The merged maximum is 5. The merged denominator is
> $3 e^{2 - 5} + 2 e^{5 - 5} = 3 \times 0.0498 + 2 = 2.149$.

> [!QUESTION] A row has mean 1000 and variance 1 in fp32. Why does $E[x^2] - E[x]^2$ give a poor variance?
> $E[x^2]$ is about $10^6$, where fp32 values are spaced $2^{-4} = 0.0625$ apart. A single
> rounding of $E[x^2]$ can shift it by about 0.03, and the subtraction cancels the
> leading digits and leaves that error beside a true variance of 1. Welford's update
> adds only non-negative deviations from the running mean, so it doesn't cancel.

> [!QUESTION] Why does a whole-row-in-registers softmax fail for a row of 131072 elements with 8 warps?
> Eight warps are 256 threads, so each thread holds $131072 / 256 = 512$ values. The A100
> allows 255 registers per thread, so the compiler spills to local memory. A loop over
> chunks of 4096 holds 16 values per thread instead.

> [!QUESTION] Why does the lab compute dw and db in two stages instead of with `tl.atomic_add`?
> Every row adds into the same $N$ addresses, so atomics contend on them, and the
> order of additions varies between runs, so the float result isn't reproducible. In
> the two-stage scheme each program owns one partial row and the second kernel sums
> them in a fixed order, so the result is the same on every run.

> [!QUESTION] The byte count predicts a 1.25x speedup for fusing a residual add into LayerNorm. Why might you measure less on a 42 MB tensor?
> The unfused kernels write an intermediate that the next kernel reads. The A100 has a 40 MiB L2 cache, so
> for a tensor near that size the intermediate may never reach HBM, and the saved trip
> costs little. The companion course measured 0.92x at 41.9 MB and 1.10x at 167.8 MB.

> [!QUESTION] Write the RMSNorm backward formula for $dx$.
> With $\hat{x}_i = x_i r$ and $g_i = dy_i w_i$, it's
> $dx_i = r\,(g_i - \hat{x}_i\,\overline{g\hat{x}})$, where $\overline{g\hat{x}}$ is the mean of
> $g_j \hat{x}_j$ over the row. The mean term of LayerNorm drops out because RMSNorm
> never subtracts a mean.

## Lab

> [!TRY] Online softmax and LayerNorm in Triton
> Fill in four Triton kernels in `starter.py`: the online softmax, the LayerNorm
> forward pass with a Welford merge, and the two kernels of the LayerNorm backward
> pass. The tests check them against a float64 PyTorch reference, and wrap your
> forward and backward in a `torch.autograd.Function` to compare with the gradients
> that `torch.autograd` computes. Shapes include a $1 \times 1$ row, rows that aren't
> multiples of the block size, a 70000-column row, logits near 1000, a mean of 1000,
> and bf16.
>
> Your lab reports `softmax_gbs`, `softmax_vs_torch`, `layernorm_fwd_gbs`, and
> `layernorm_bwd_gbs`. The roofline predicts floors of 0.421 ms for the 268 MB softmax
> test (2 volumes), 0.211 ms for the LayerNorm forward test, and 0.316 ms for the
> backward test, all at 1275 GB/s. To pass, the kernels must reach 40%, 40%, and 30%
> of that ceiling. Triton compiles on first use, so the first call is slow.

## Further reading

- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867):
  Milakov and Gimelshein's one-pass running maximum and sum.
- [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html):
  a row-wise softmax with a persistent loop over rows and its bandwidth comparison.
- [Triton layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html):
  forward and backward kernels, including the locked partial-sum reduction for dw.
- [Layer Normalization](https://arxiv.org/abs/1607.06450) and
  [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467): the
  definitions this chapter differentiates.
- [Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance):
  Welford's update, Chan's merge, and their citations.
- [Liger Kernel](https://arxiv.org/abs/2410.10989): production Triton norm kernels
  for LLM training.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html):
  why normalization and softmax are bandwidth-bound.
- [Fusion in Triton](https://qwen.ifkash.dev/c/13-fusion-in-triton) in the companion
  course: the measured RMSNorm and the L2 crossover this chapter quotes.
