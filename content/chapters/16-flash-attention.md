---
title: FlashAttention
slug: 16-flash-attention
part: "Part 4 — Kernels for ML"
summary: How FlashAttention avoids the N x N score matrix in the forward and the backward pass, what FlashAttention-2 changed, and how causal masking, grouped heads, Hopper, and Blackwell reshape the kernel.
minutes: 180
gpu: true
objectives:
  - "Count the memory traffic of standard attention and say why it is bound by bandwidth, not by arithmetic."
  - "Write the forward pass as a tiled loop with an online softmax, and say why it saves one number per row, the logsumexp."
  - "Derive the backward pass: recompute P from the logsumexp, then dV, dP, dS, dK, and dQ, including the delta = rowsum(dO * O) identity."
  - "Compare the two ways to accumulate dQ, atomic adds and a separate kernel, and say what each costs."
  - "Explain causal block skipping, the load imbalance it creates, and how grouped-query attention changes the backward indexing."
  - "Describe what FlashAttention-2, -3, and -4 each changed, and which hardware limit drove each change."
  - "Write FlashAttention-2 forward and backward in Triton and check it against float32 PyTorch and torch.autograd."
lab: 16-flash-attention
---

# FlashAttention

> [!TLDR]
> - Standard attention writes the $N \times N$ score matrix to memory and reads it
>   back four times. At $N = 8192$ and $d = 128$, one head's matrix is 134.2 MB and
>   the pass is bound by bandwidth, with an intensity of 63 FLOP per byte against
>   a ridge point of 161.
> - FlashAttention computes the scores tile by tile with an online softmax, keeps
>   the running state in registers, and saves one number per query row, the
>   logsumexp, so the matrix never has an address.
> - The backward pass recomputes each tile of $P$ from $Q$, $K$, and the
>   logsumexp, then applies five matmuls. A preprocessing step,
>   $\delta = \operatorname{rowsum}(dO \circ O)$, replaces a full pass over each row.
> - FlashAttention-2 changed the loop order, parallelized over the sequence, and
>   cut non-matmul work. FlashAttention-3 and -4 then overlap softmax with the
>   tensor cores on Hopper and Blackwell, where the arithmetic units no longer
>   scale together.
> - The lab writes the forward and backward kernels in Triton, causal and not,
>   and reports TFLOP/s against PyTorch's `scaled_dot_product_attention`.

Attention is the one operation in a transformer whose intermediate result grows
with the square of the sequence length. Everything else, the projections, the MLP,
and the norms, produces activations that grow with the number of tokens. Attention
produces one score for every pair of tokens, and the obvious implementation
stores all of them.

FlashAttention is the kernel that doesn't. It computes exactly the same function,
up to rounding, and it runs several times faster because it moves far fewer bytes
through HBM. Training depends on it twice over: the backward pass has the same
$N \times N$ problem as the forward pass, and a training step that couldn't fit
the scores in memory couldn't run at long context at all.

The companion course wrote the forward pass for inference in
[its chapter 14](https://qwen.ifkash.dev/c/14-flash-attention), and derived the
online softmax from a four-score example. This chapter takes that as given and
goes further: why the traffic matters on this hardware, what the second version
changed, the whole backward pass, and how the kernel bends around the Hopper and
Blackwell GPUs.

## Before you start

**Attention for one head.** Queries $Q$, keys $K$, and values $V$ each have shape
$(N, d)$, where $N$ is the sequence length and $d$ the head dimension. The scores
are $S = QK^{\top} / \sqrt{d}$, the probabilities are the row-wise softmax
$P = \operatorname{softmax}(S)$, and the output is $O = PV$. A batch and several
heads run as independent copies.

**The online softmax.** [Chapter 15](/c/15-softmax-and-normalization) built it:
one pass over a row keeps a running maximum $m$ and a running sum $\ell$ of
$e^{s - m}$, and when the maximum rises you multiply the old sum by
$e^{m_{\text{old}} - m_{\text{new}}}$. This chapter extends it from a sum to a
sum of value vectors.

**The roofline and tensor cores.** [Chapter 4](/c/04-roofline-and-latency) defines
the ridge point, 161 FLOP per byte on the A100 80GB PCIe (312 TFLOP/s over 1935
GB/s). [Chapter 11](/c/11-tensor-cores) covers the tensor cores that `tl.dot`
reaches. A tile matmul runs there at up to 312 TFLOP/s in bfloat16.

**Triton basics.** [Chapter 14](/c/14-triton) covers `tl.load` and `tl.store`
with masks, `tl.dot`, and the `tl.program_id` launch grid. This chapter's kernels
use nothing else.

## What standard attention costs

The standard implementation is slow because of the bytes it moves, not the FLOPs
it does, and this section counts both for one head. Take $N = 8192$, $d = 128$,
and bfloat16, two bytes per element.

Count the arithmetic first. Each of the two matmuls does $2N^2 d$ FLOPs, so one
head costs:

$$
4N^2 d = 4 \times 8192^2 \times 128 = 34.36 \times 10^{9}\ \text{FLOP}.
$$

At the A100's 312 TFLOP/s, that takes at least $34.36 \times 10^{9} / 312 \times 10^{12} = 0.110$ ms.

Now count the bytes. The score matrix has $8192^2 = 67.1$ million entries, or
134.2 MB. The standard path makes four passes over it: write $S$, read $S$ for the
softmax, write $P$, and read $P$ for the multiply with $V$. That is 536.9 MB of
traffic. The four tensors $Q$, $K$, $V$, and $O$ add $4 \times 8192 \times 128 \times 2 = 8.4$ MB.

> [!EXAMPLE] The standard path is bound by bandwidth
> Total traffic is $536.9 + 8.4 = 545.3$ MB, of which 98.5% is the score matrix.
>
> - Arithmetic intensity: $34.36 \times 10^{9} / 545.3 \times 10^{6} = 63.0$ FLOP/byte,
>   below the ridge point of 161.
> - Memory floor at the measured copy ceiling of 1275 GB/s (measured in
>   learn-inference, chapter 10): $545.3 \times 10^{6} / 1.275 \times 10^{12} = 0.428$ ms.
> - Compute floor: 0.110 ms.
>
> The memory floor is 3.9 times the compute floor. The tensor cores wait.

This count is a lower bound for a real framework. PyTorch's unfused path also runs
the scale, the mask, and the softmax as separate kernels, and it often computes
the softmax in float32, which doubles the width of the matrix. Capacity is the
second problem: the score matrix for 8 heads at this length is 1.07 GB in
bfloat16 for one sequence, and it scales with $N^2$.

==The cost of standard attention is the matrix, not the matmuls==. The two matmuls
are the useful work, and they would run at tensor-core speed if the data were on
the chip. FlashAttention's whole job is to make it be there.

## The tiled forward pass

The forward pass answers one question: how do you produce $O$ without ever holding
a whole row of scores? The answer is to process the keys in blocks and carry a
small running state from block to block.

For one query row, keep three things as you stream over key blocks $j = 1, \dots, T$:
the running maximum $\hla{m}$, the running sum $\hlb{\ell}$ of $e^{s - m}$, and the
unnormalized output $\hlc{o}$, a vector of length $d$. When block $j$ arrives with
scores $S_j$, the update, from chapter 15 extended to values, is:

$$
\boxed{
\begin{aligned}
\hla{m_{\text{new}}} &= \max\!\big(\hla{m},\ \operatorname{rowmax}(S_j)\big), \qquad
\hld{\alpha} = e^{\hla{m} - \hla{m_{\text{new}}}}, \qquad
P_j = e^{S_j - \hla{m_{\text{new}}}}, \\
\hlb{\ell} &\leftarrow \hld{\alpha}\,\hlb{\ell} + \operatorname{rowsum}(P_j), \qquad
\hlc{o} \leftarrow \hld{\alpha}\,\hlc{o} + P_j V_j .
\end{aligned}}
$$

Here $\hld{\alpha} \in (0, 1]$ converts the old totals to the new maximum. After the
last block, divide once: $O = \hlc{o} / \hlb{\ell}$. The result is exact. Only the
order of the floating-point additions differs from the standard path.

Each program in the kernel owns a tile of $B_M$ query rows. It loads that tile of
$Q$ once and keeps it in registers, then loops over key tiles of $B_N$ rows:

```python
for start_n in range(0, hi, BLOCK_N):
    k = tl.load(...)                       # (HEAD_DIM, BLOCK_N), loaded transposed
    v = tl.load(...)                       # (BLOCK_N, HEAD_DIM)
    s = tl.dot(q, k) * qk_scale            # (BLOCK_M, BLOCK_N), in registers
    s = tl.where(keep, s, float("-inf"))   # padding and causal mask
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new
```

The kernel works in base 2. It folds $\log_2 e$ into the scale once, so
`qk_scale = sm_scale * log2(e)`, and it calls `exp2`, which maps to a single fast
hardware instruction. The score tile $S$ and probability tile $P$ live in
registers and are never stored. The accumulator `acc` stays in float32 even when
the inputs are bfloat16, because the running sum would otherwise stop growing
once it passed 256 times the next term ([chapter 9](/c/09-floating-point-on-the-gpu)).

### The one number the forward pass saves

The backward pass needs to rebuild $P$, and it can't if the forward pass kept only
$O$. The forward pass therefore saves one extra value per query row: the
*logsumexp*, written $\hla{L}$. It is the row's maximum plus the log of its sum:

$$
\hla{L_i} = \hla{m_i} + \log \hlb{\ell_i} = \log \sum_j e^{S_{ij}} .
$$

The log of the normalizer is all you need, because $P_{ij} = e^{S_{ij} - L_i}$. One
float32 per row costs $4N$ bytes per head, 32 KiB at $N = 8192$, against 134.2 MB for
the matrix it replaces. The kernel stores `lse = (m_i + log2(l_i)) * ln(2)`, in
natural-log units, so it matches `torch.logsumexp` in the lab's test.

```viz
16-tile-grid
```

> [!KEY] The forward pass trades a matrix for a vector
> The output $O$ and a vector $L$ of $N$ floats are everything the forward pass
> writes. $P$ is never stored, and the backward pass rebuilds each tile of it when
> it needs one.

## How much traffic remains

The tiled kernel doesn't make the traffic vanish, and this section counts what is
left. Each program reads its $Q$ tile once, writes its $O$ tile once, and streams
all of $K$ and $V$ past it. With $N / B_M$ query tiles, $K$ and $V$ are read that
many times.

Per head, in elements, the traffic is:

$$
\underbrace{2Nd}_{Q,\ O}\ +\ \frac{N}{B_M}\cdot\underbrace{2Nd}_{K,\ V\ \text{per tile}} .
$$

With $B_M = 128$, $N = 8192$, $d = 128$, the key and value reads are $64 \times 2 \times 8192 \times 128 = 134.2$ million elements, or 268.4 MB. With the 4.2 MB
for $Q$ and $O$ ($2 \times 8192 \times 128 \times 2$ bytes), the total is 272.6 MB.

That is half of the standard path's 545.3 MB, not 64 times less. The paper states the
result as $\Theta(N^2 d^2 / M)$ for $M$ elements of on-chip memory: the tiled
kernel is still quadratic in $N$, and ==its advantage is the factor $M / d^2$==, not
a change of order. Taken literally, the factor is modest.

What closes the gap is the L2 cache. All 64 programs for one head read the same $K$
and $V$, 4.2 MB per head at this size in total, and the A100 has 40 MiB of L2.
Programs that run at the same time hit the cache for nearly every read after the
first. Only the first read of each tile comes from HBM, which brings the traffic
toward $4Nd$ elements, or 8.4 MB.

> [!EXAMPLE] Where the kernel lands on the roofline
> Intensity is $34.36 \times 10^{9}$ FLOP over the bytes that reach HBM:
>
> | Case | HBM bytes | Intensity (FLOP/byte) | Memory floor at 1275 GB/s |
> |---|---|---|---|
> | Standard attention | 545.3 MB | 63.0 | 0.428 ms |
> | Tiled, every $K$ and $V$ read from HBM | 272.6 MB | 126.0 | 0.214 ms |
> | Tiled, L2 absorbs the repeats | 8.4 MB | 4096 | 0.007 ms |
>
> Only the last row is far above the ridge point of 161. The kernel is
> compute-bound when its $K$ and $V$ tiles stay in L2, and the lab measures how
> close it gets.

```viz
16-io-traffic
```

The cache argument has a limit: it needs a head's $K$ and $V$ to fit in L2, and the
programs that share them to run together. At $N = 8192$ and $d = 128$ they are 4.2
MB per head, but a long-context run with $N = 131072$ has 67 MB per head, which
exceeds 40 MiB. At that point the repeats reach HBM, and a larger query tile or
a different schedule is the fix.

## What FlashAttention-2 changed

The first version, FlashAttention, already avoided the matrix. FlashAttention-2
made the same algorithm run about 2 times faster, and the paper reports 50 to 73%
of the A100's peak for the forward pass and up to 225 TFLOP/s end to end
(measured on an A100, per [the paper](https://arxiv.org/abs/2307.08691)). Four changes
account for it.

**1. The loop order.** The first version looped over key blocks on the outside and
query blocks on the inside. Every outer iteration touched every row's output, so
the running $\hlc{o}$, $\hla{m}$, and $\hlb{\ell}$ had to be written to HBM and read
back. FlashAttention-2 swapped the loops: the outer loop is over query tiles, so
one tile's state stays in registers for its whole pass, and $O$ is written once.

**2. Parallelism over the sequence.** The first version launched one thread block
per (batch, head) pair. With a batch of 1 and 8 heads, that is 8 programs on an
A100 with 108 SMs, which leaves 93% of the machine idle. Because the outer loop is
now over query tiles, which are independent, the grid becomes
$\lceil N / B_M \rceil \times B \times H$, which is $64 \times 8 = 512$ programs
at $N = 8192$ and $B_M = 128$. Long sequences with small batches, the training
case that motivated the change, now fill the GPU.

**3. Fewer non-matmul FLOPs.** The A100 does 312 TFLOP/s of bfloat16 matmul but
only 19.5 TFLOP/s of float32 arithmetic on its CUDA cores, so a matmul FLOP
costs 1/16 of a non-matmul FLOP ($312 / 19.5 = 16.0$). The version-2 kernel trims the
latter:

- It keeps the output unnormalized and divides by $\hlb{\ell}$ once at the end,
  instead of rescaling by $1/\hlb{\ell}$ each iteration.
- It saves the single logsumexp $\hla{L}$ instead of both $\hla{m}$ and $\hlb{\ell}$.
- It skips fully masked blocks entirely in causal attention.

**4. Warp partitioning.** A thread block has several warps, and each warp must
own part of the tile. The first version split $K$ and $V$ across warps, so each
warp produced a partial result for the same output rows, and the warps had to
write those partials to shared memory and add them. The second version splits $Q$
across warps and gives every warp the whole $K$ and $V$ tile, so each warp's rows
are independent and no cross-warp reduction is needed.

> [!NOTE] What Triton does for you
> You never write the warp partition in Triton. The compiler chooses how to lay
> a `tl.dot` tile across the warps, and `num_warps` sets how many. The parallel
> grid, the loop order, and the deferred division are your decisions, and they are
> the ones that matter most.

## The backward pass

The backward pass answers a different question: given the upstream gradient $dO$,
what are $dQ$, $dK$, and $dV$, without the matrix? The answer rebuilds $P$ tile by
tile and applies the chain rule to each tile. This is the part most treatments
compress, so this section goes one matmul at a time.

Write the forward pass as three steps and differentiate them in reverse. The
scale $1/\sqrt{d}$ is written $\sigma$.

1. $S = \sigma Q K^{\top}$
2. $P = \operatorname{softmax}(S)$, row by row
3. $O = PV$

Start at the output. The gradient flows back through $O = PV$ in two directions,
to $V$ and to $P$:

$$
\boxed{dV = P^{\top}\, dO}, \qquad dP = dO\, V^{\top} .
$$

Both are matmuls. $dV$ is a sum over the query rows, so a key tile's $dV$ collects
a contribution from every query tile. Next comes the softmax, whose Jacobian gives
the gradient of each row of $S$:

$$
dS_{ij} = P_{ij}\Big(dP_{ij} - \hld{\delta_i}\Big), \qquad \hld{\delta_i} = \sum_k P_{ik}\, dP_{ik} .
$$

The row sum $\hld{\delta_i}$ spans the whole row of $P$, which a tiled kernel
doesn't have. The identity in the next subsection removes it. Last, the scores
$S = \sigma Q K^{\top}$ send the gradient to the two inputs:

$$
\boxed{dQ = \sigma\, dS\, K}, \qquad \boxed{dK = \sigma\, dS^{\top} Q} .
$$

Five matmuls: $dV$, $dP$, $dQ$, $dK$, and the recomputation of $S$ to rebuild $P$.
The forward pass has two.

### The delta identity

The row sum $\delta_i = \sum_k P_{ik} dP_{ik}$ looks like it needs the whole row,
but it doesn't. Substitute $dP_{ik} = dO_i \cdot V_k$ and move the sum inside:

$$
\hld{\delta_i} = \sum_k P_{ik}\,(dO_i \cdot V_k) = dO_i \cdot \Big(\sum_k P_{ik} V_k\Big) = \boxed{dO_i \cdot O_i} .
$$

So $\hld{\delta}$ is the dot product of two rows you already have, $dO$ and $O$.
A small kernel computes it for every row before the main backward loop starts:
$\delta = \operatorname{rowsum}(dO \circ O)$, one float32 per row. This is the
*preprocessing* step. It reads $O$ and $dO$ once, which costs $2Nd$ elements, and it
removes a full extra pass over $K$ and $V$.

> [!EXAMPLE] Check the identity on one row
> Take one query with $P = [0.25,\ 0.75]$, values $V = [1,\ 3]$ (a single column),
> and upstream gradient $dO = 2$.
>
> - The output is $O = 0.25 \times 1 + 0.75 \times 3 = 2.5$.
> - The score gradients are $dP_k = dO \cdot V_k = [2,\ 6]$.
> - The direct sum is $\delta = 0.25 \times 2 + 0.75 \times 6 = 5.0$.
> - The identity gives $dO \cdot O = 2 \times 2.5 = 5.0$.
> - Then $dS = P \odot (dP - \delta) = [0.25 \times (2 - 5),\ 0.75 \times (6 - 5)] = [-0.75,\ 0.75]$.
>
> The two ways to get $\delta$ agree, and $dS$ sums to zero, as the gradient of a
> softmax must.

> [!DEEPDIVE] Why the softmax Jacobian has this form
> For one row, $P_j = e^{S_j} / \sum_k e^{S_k}$. Differentiate with respect to
> $S_i$. The numerator gives $\partial P_j / \partial S_i = P_j (\mathbb{1}[i = j] - P_i)$.
> Contract that with an upstream gradient $dP$:
>
> $$
> dS_i = \sum_j dP_j\, P_j\,(\mathbb{1}[i = j] - P_i) = P_i\, dP_i - P_i \sum_j P_j\, dP_j = P_i\,(dP_i - \delta) .
> $$
>
> The Jacobian is $\operatorname{diag}(P) - PP^{\top}$. It is symmetric, and its
> rows sum to zero, which is why $dS$ sums to zero and why adding a constant to a
> whole row of scores changes nothing.

### Recomputing P

The backward pass needs $P$ for $dV$ and for $dS$, and it has no matrix to read. It
rebuilds one tile at a time from the saved logsumexp:

$$
P_{ij} = e^{\,\sigma\, Q_i \cdot K_j\, -\, \hla{L_i}} .
$$

This is the one matmul the backward pass repeats from the forward pass, $QK^{\top}$,
and it needs no running maximum, because $\hla{L_i}$ already folds in the
normalizer. The recomputation spends tensor-core FLOPs, which are cheap, to avoid
$N^2$ bytes, which are not.

### The loop structure and the dQ problem

A backward kernel has to decide which index is the outer loop, and no choice is
free. $dK$ and $dV$ are sums over query tiles. $dQ$ is a sum over key tiles. One
loop order can keep one family of accumulators in registers, but not both.

| Outer loop | Stays in registers | Reduction that goes to HBM |
|---|---|---|
| Key tile $j$ | $dK_j$, $dV_j$ | $dQ_i$, from every key tile |
| Query tile $i$ | $dQ_i$ | $dK_j$ and $dV_j$, from every query tile |

FlashAttention-2 chooses the first row. Each program holds one key tile with its
$dK$ and $dV$ accumulators in float32 and loops over all the query tiles. $dQ$
then has a problem: every program adds a contribution to every $dQ_i$ tile, and
programs run concurrently. There are two fixes.

**Atomic adds.** The reference FlashAttention-2 kernel keeps a float32 $dQ$ buffer
in HBM, and each program adds its contribution with `atomicAdd`. It needs one
backward kernel and five matmuls. The cost is nondeterminism: floating-point
addition isn't associative, and the atomics land in whatever order the scheduler
picks, so two runs differ in the last bits. It also adds HBM traffic for the
float32 buffer and a conversion pass at the end.

**A second kernel.** A separate kernel loops with the query tile on the outside,
keeps $dQ_i$ in registers, and writes it once. It recomputes $S$ and $dP$ again,
two extra matmuls, so the total is seven instead of five. In exchange the result is
deterministic and needs no atomics.

The lab takes the second route, and reports its throughput with the five-matmul
FLOP count, so its reported TFLOP/s has a ceiling of $312 \times 5 / 7 = 223$.
FlashAttention-4 attacks the same cost on Blackwell with a different tool, which a
later section describes.

> [!NOTE] Deterministic training
> Some training runs must reproduce bit for bit. The atomic version of the
> backward pass can't promise that, and PyTorch's attention backends expose a
> deterministic mode for it. The separate-kernel design gives it for free.

### The dK and dV kernel

The inner loop of the key-tile kernel is the five matmuls written out. The lab
works in transposed tiles, with keys as rows, so that $dV$ and $dK$ are plain
`tl.dot` calls on $P^{\top}$ and $dS^{\top}$:

```python
st  = tl.dot(k, tl.trans(q)) * qk_scale          # S^T, (BLOCK_N, BLOCK_M)
pt  = tl.exp2(st - lse[None, :] * 1.4426950408889634)   # P^T from the saved LSE
pt  = tl.where(keep, pt, 0.0)                    # zero padding and causal-masked entries
dv += tl.dot(pt.to(do.dtype), do)                # dV  += P^T dO
dpt = tl.dot(v, tl.trans(do))                    # dP^T = V dO^T
dst = pt * (dpt - delta[None, :])                # dS^T = P^T o (dP^T - delta)
dk += tl.dot(dst.to(q.dtype), q)                 # dK  += dS^T Q
```

Both `dk` and `dv` are float32 tiles that never leave registers until the loop ends.
Then the kernel multiplies `dk` by $\sigma$ and stores both. The `dq` kernel is the
mirror image: its tile holds $Q$ and $dO$, it streams $K$ and $V$, and it
accumulates `dq += tl.dot(ds, tl.trans(k))`.

Notice the mask is applied to $P$ after the exponential, not to $S$ before it. An
out-of-range key loads as zeros, so its score is 0 and its $P$ is $e^{-L}$, which
isn't zero. Setting $P$ to zero removes the key's contribution to $dV$, $dK$, and
$dS$ at once.

### What the backward pass costs

Count the FLOPs the way the papers do. The forward pass is $4N^2 d$. The backward
pass is five matmuls of $2N^2 d$ each, so $10 N^2 d$, which is 2.5 times the
forward. The A100 has a compute floor for the backward pass of
$85.9 \times 10^{9} / 312 \times 10^{12} = 0.275$ ms per head at $N = 8192$,
$d = 128$ without a mask, against 0.110 ms for the forward pass. Training spends
$3.5 \times$ the forward's tensor-core time per attention layer in all, before
counting the recompute that activation checkpointing adds.

> [!WARNING] Backward register pressure
> The key-tile kernel holds $K$, $V$, $dK$, and $dV$. With a 64-row tile and
> $d = 128$, $dK$ and $dV$ in float32 are $64 \times 128 \times 4 = 32$ KiB each, and $K$
> and $V$ in bfloat16 are 16 KiB each, a total of 96 KiB per program before any
> temporary. With 8 warps, that is 96 registers per thread of the 255 available,
> and an A100 SM's 256 KB register file holds only two such programs at once.
> A larger tile spills. The lab's launcher tries a smaller configuration when a
> launch asks for more shared memory than the SM has.

## Causal masking and block skipping

Causal attention lets query $i$ see only keys $j \le i$, and the saving comes from
never computing the tiles above the diagonal, not from masking them. This section
shows how the kernel finds them.

Masking the scores after the matmul saves nothing, because you've already paid. The
kernel instead bounds the key loop. Query tile $i$ covers rows up to
$(i + 1) B_M - 1$, so no key past that can matter:

```python
hi = N
if CAUSAL:
    hi = tl.minimum(N, (pid_m + 1) * BLOCK_M)
for start_n in range(0, hi, BLOCK_N):
    ...
```

> [!EXAMPLE] Tiles computed at $N = 8192$
> With $B_M = 128$ and $B_N = 64$ there are $8192 / 128 = 64$ query tiles and
> $8192 / 64 = 128$ key tiles, so a full pass has $64 \times 128 = 8192$ tile pairs.
> Query tile $i$ runs $2(i + 1)$ key tiles, so a causal pass runs:
>
> $$
> \sum_{i=0}^{63} 2(i + 1) = 2 \cdot \frac{64 \times 65}{2} = 4160 \ \text{tile pairs},
> $$
>
> which is 50.8% of 8192. The extra 0.8% is the diagonal tiles, which straddle the
> boundary and still need the element-wise mask.

Causal masking has two consequences the formula hides. First, only the diagonal
tiles need the `tl.where` mask, and a tuned kernel runs the tiles below the diagonal
in a separate loop with no mask at all. The lab's kernel applies the mask to every
tile, which is simpler and a little slower. Second, rows of tiles now have
different lengths: the last query tile does 64 times the work of the first
at $B_M = B_N$. Programs finish at different times, and a program that starts last
but runs longest leaves the GPU half empty at the end. A tuned kernel launches the long
rows first, so the short ones fill the gaps.

The backward key-tile kernel inverts the bound. Key tile $j$ is seen only by query
tiles whose rows are at least $jB_N$, so its loop starts at
`lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M` instead of 0. The counting works out the
same way: about half the tile pairs.

## Grouped-query and multi-query attention

Grouped-query attention (GQA) shares each key-value head among several query heads,
and the kernel handles it by indexing, not by copying. With $H$ query heads and
$H_{KV}$ key-value heads, the group size is $g = H / H_{KV}$, and query head $h$ reads
KV head $\lfloor h / g \rfloor$. Multi-query attention (MQA) is the case $H_{KV} = 1$.

> [!EXAMPLE] The KV cache shrinks by the group size
> With 32 query heads, 8 KV heads ($g = 4$), $d = 128$, and bfloat16, one layer
> stores $2 \times 8 \times 128 \times 2 = 4096$ bytes per token for $K$ and $V$.
> Full multi-head attention would store $2 \times 32 \times 128 \times 2 = 16{,}384$
> bytes, four times as much.

The forward pass changes by one line: `hk = h // GROUP`, used to find the $K$ and
$V$ base pointers. A kernel that instead calls `repeat_interleave` first writes $g$
copies of $K$ and $V$ to HBM, which keeps the arithmetic and gives away the bandwidth
saving.

The backward pass is where GQA bites, because $dK$ and $dV$ for a shared head are a
sum over all $g$ query heads that read it. The lab's key-tile kernel is launched over
$B \times H_{KV}$ programs and loops over the group inside:

```python
for g in range(GROUP):
    h = hk * GROUP + g          # one of the query heads sharing this KV head
    for start_m in range(lo, N, BLOCK_M):
        ...                      # accumulate into the same dk, dv registers
```

The sum happens in registers in float32, so GQA needs no atomics. The query-tile
kernel is unchanged, since each $dQ$ row belongs to exactly one query head.

## FlashAttention-3 on Hopper

On the H100, the FlashAttention-2 kernel reaches only about 35% of the GPU's peak,
according to the [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608). The
Hopper tensor cores run faster than the A100's, at 989.4 TFLOP/s of dense bfloat16
on an H100 SXM5, and a kernel that worked on Ampere leaves them idle between
operations. FlashAttention-3 reports 1.5 to 2.0 times the speed of
FlashAttention-2 on an H100, up to about 740 TFLOP/s in FP16 (75% of the peak) and
close to 1.2 PFLOP/s in FP8.

Four ideas drive it, and the first three use a Hopper feature
[chapter 12](/c/12-pipelining-and-async-copies) introduced.

**Warp specialization.** Some warps become *producers* that only issue TMA copies
of $K$ and $V$ tiles into shared memory, and the rest become *consumers* that only
issue `wgmma` matmuls and do the softmax. The two groups run asynchronously and
coordinate through barriers, and the kernel moves registers from the producers to
the consumers, which need more of them.

**Ping-pong scheduling.** The exponentials in the softmax run on a unit with far
less throughput than the tensor cores. Per element, the two matmuls do $4d = 512$
FLOPs at $d = 128$, which takes $512 / 989.4 \times 10^{12} = 0.52$ ps of the
tensor cores' time. The same element needs one exponential, and NVIDIA's figure for Hopper's
exponential throughput is 4.5 trillion per second for the whole GPU, which is 0.22 ps. If the two don't
overlap, softmax adds 43% to the matmul time. FlashAttention-3 runs two warpgroups
and schedules them so that one does its softmax while the other does its matmuls,
and the roles swap each iteration.

**Intra-warpgroup overlap.** Within one warpgroup, the kernel issues the matmul
$QK^{\top}$ for the next key tile asynchronously, and then does the softmax for
the current tile while that matmul runs. It holds the scores for two tiles in
registers at once, which costs registers, and it hides the softmax behind the
tensor cores.

**FP8 with incoherent processing.** FP8 doubles the tensor-core rate to 1978.9
TFLOP/s dense on the H100 SXM5, but FP8 has few bits, and attention has outlier
features. The paper quantizes per block and multiplies $Q$ and $K$ by a random
orthogonal matrix, a Hadamard transform with random signs, before quantizing. The
transform spreads an outlier across many entries without changing $QK^{\top}$, and it
costs $O(d \log d)$ per token. The paper reports 2.6 times lower numerical error than
a baseline FP8 attention.

> [!NOTE] The labs can't run this
> The lab A100 has neither TMA nor `wgmma` nor FP8 tensor cores. This section and the
> next are prose: the numbers are the papers' own, measured on the H100 and B200
> the papers name.

## FlashAttention-4 on Blackwell

Blackwell breaks the balance FlashAttention-3 relied on. The
[FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) and
[its blog post](https://tridao.me/blog/2026/flash4/) describe *asymmetric hardware
scaling*: from H100 to B200 the bfloat16 tensor-core throughput grows from about 1
to 2.25 PFLOP/s, while the exponential units and shared-memory bandwidth stay almost
flat.

The blog does the per-SM arithmetic for a tile with $M = N = d = 128$:

| Resource (one B200 SM) | Rate | Forward cycles | Backward cycles (1-CTA) |
|---|---|---|---|
| Tensor cores | 8192 ops per cycle | 1024 | 2560 |
| Exponential unit | 16 ops per cycle | 1024 | 1024 |
| Shared memory | 128 bytes per cycle | 768 | 3328 |

In the forward pass the tensor cores and the exponentials tie at 1024 cycles, so any
exponential not overlapped with a matmul is lost time. In the backward pass shared
memory is the longest at 3328 cycles, so the backward pass is bound by shared-memory
traffic, not arithmetic. FlashAttention-4 attacks each:

- **Asynchronous MMA and larger tiles.** The `tcgen05` instructions
  ([chapter 12](/c/12-pipelining-and-async-copies)) issue from one thread and write
  to tensor memory, so the kernel pipelines the matmuls and the softmax with 128 by
  256 tiles, twice the largest Hopper atom.
- **Software exponentials.** Part of the $2^x$ work moves from the exponential unit to
  the FMA units. It uses a polynomial approximation after range reduction, a cubic
  with the coefficients the blog lists, so both units work at once.
- **Conditional rescaling.** The accumulator is rescaled by $\hld{\alpha}$ only when the
  row maximum rises by more than a threshold, since the final normalization uses
  the true statistics anyway. A dedicated correction warpgroup keeps the rescale off
  the critical path.
- **A 2-CTA backward pass.** A pair of CTAs share one MMA of $M = 256$, each staging
  half of one operand, which roughly halves that operand's shared-memory traffic. The
  $dQ$ reduction exchanges half of $dS$ through distributed shared memory so each CTA
  owns half the rows, which halves the atomic adds to global memory. A deterministic
  mode serializes those reductions with a lock and reaches about 85 to 90% of the
  nondeterministic throughput, per the blog.

The paper reports up to 1613 TFLOP/s in bfloat16 on a B200, about 71% of its peak, and
up to 1.3 times cuDNN 9.13 and 2.7 times Triton (the blog quotes 1605 TFLOP/s, and
notes that newer cuDNN releases perform similarly). The kernels are written in the
CuTe DSL ([chapter 13](/c/13-cutlass-and-cute)), with compile times the authors give as 20 to 30
times faster than C++ templates.

Compare the three generations by what limited each:

| Version | Hardware | What limited the previous kernel | Reported result |
|---|---|---|---|
| FlashAttention-2 | A100 | Wasted parallelism and non-matmul FLOPs | 50 to 73% of peak forward, 225 TFLOP/s end to end |
| FlashAttention-3 | H100 | Matmul and softmax not overlapped | up to 740 TFLOP/s FP16 (75%), about 1.2 PFLOP/s FP8 |
| FlashAttention-4 | B200 | Exponentials and shared memory | up to 1613 TFLOP/s BF16 (71%) |

## Attention when there is one query

Decoding has $N_q = 1$, and a kernel that parallelizes over query tiles has one tile
to give. During autoregressive decoding the new token's query attends to a long
cache, and the kernel has the grid $1 \times B \times H$. For one sequence with 32
heads, that is 32 programs on 108 SMs, each reading a long KV cache serially.

*Flash-decoding* ([PyTorch blog](https://pytorch.org/blog/flash-decoding/)) splits
the work along the key axis. Each of $S$ programs handles a slice of the cache and
returns a partial output $O_s$ and its logsumexp $L_s$. A small second kernel merges
them, and the logsumexp is what makes it exact:

$$
L = \log \sum_s e^{L_s}, \qquad O = \sum_s e^{L_s - L}\, O_s .
$$

Each partial $O_s$ is already normalized within its slice, and the weight
$e^{L_s - L}$ is the slice's share of the full softmax mass. Splitting 32 heads
four ways gives 128 programs, enough to fill the A100. The same $L$ that the forward
pass saved for the backward pass is the merge key here. The companion course covers
the decode path in its [paged attention chapter](https://qwen.ifkash.dev/c/15-paged-attention).

## Using it from PyTorch

PyTorch exposes these kernels through `torch.nn.functional.scaled_dot_product_attention`
(SDPA), which picks a backend per call. On an A100 with bfloat16, head dimension 64 or
128, and a causal or no mask, it can choose the FlashAttention-2 kernel, which is what
the lab compares against. An arbitrary additive mask moves it to a slower backend.

For a mask or score modification that isn't causal, such as a sliding window or ALiBi,
[FlexAttention](https://pytorch.org/blog/flexattention/) compiles the modification
into a FlashAttention-style kernel, so you write a small function and get the tiled
loop. [Chapter 17](/c/17-fusion-and-compilers) covers the compiler machinery behind it.

## What goes wrong

- **Masking $S$ instead of $P$ in the backward pass.** An out-of-range key loads as
  zeros, so $e^{s - L}$ isn't zero. Without `tl.where(keep, p, 0.0)` after the
  exponential, phantom keys leak into $dV$, $dK$, and $dS$. The error grows as
  the sequence length moves away from a multiple of the block size, which is why the
  lab tests lengths of 1, 100, 130, 257, and 333.
- **Forgetting the scale in the gradients.** $dQ$ and $dK$ carry the factor
  $\sigma = 1/\sqrt{d}$ because $S = \sigma Q K^{\top}$, and $dV$ doesn't. A kernel
  missing it passes a shape check and misses by a factor of $\sqrt{d}$.
- **Mixing log bases.** The kernel works in base 2 and the saved logsumexp is in
  natural log, or the reverse. If the backward kernel's $P$ is wrong, check the units of
  $L$ before the matmuls.
- **Computing $\delta$ from a rounded $O$.** $\delta = dO \cdot O$ uses the stored $O$,
  rounded to bfloat16. It's what the reference does, and it's consistent with
  recomputing from the stored value. Computing it from a float32 $O$ you don't keep is
  a different, slightly wrong, number.
- **Atomics and reproducibility.** A dQ built with atomic adds differs from run to
  run in the last bits. A test that compares two runs bit for bit fails, and the cause
  isn't a bug.
- **bfloat16 accumulators.** Keeping `acc`, `dk`, or `dv` in bfloat16 saves registers
  and loses the sum at long context.
- **Out of shared memory.** A larger tile or more pipeline stages exceeds the 163 KB
  per block the A100 allows, and Triton raises `OutOfResources` at launch. The lab's
  launcher catches it and retries with smaller tiles.

> [!RECAP]
> - Standard attention moves the $N \times N$ matrix through HBM four times and is
>   bound by bandwidth: 63 FLOP/byte at $N = 8192$, $d = 128$, against a ridge point
>   of 161.
> - The tiled forward pass keeps $\hla{m}$, $\hlb{\ell}$, and $\hlc{o}$ in registers
>   and saves only the logsumexp $\hla{L}$, so $P$ has no address. Traffic stays
>   quadratic, and the L2 absorbs the re-reads of $K$ and $V$.
> - The backward pass recomputes $P = e^{\sigma QK^{\top} - L}$ and applies $dV$, $dP$,
>   $dS$, $dK$, and $dQ$, with $\hld{\delta} = \operatorname{rowsum}(dO \circ O)$
>   replacing a pass over each row. It costs 2.5 times the forward FLOPs.
> - $dQ$ needs either atomic adds or a second kernel that recomputes two matmuls.
>   Causal masking skips about half the tiles, and GQA sums $dK$ and $dV$ over the
>   query heads of a group.
> - FlashAttention-2 reordered the loops and parallelized the sequence;
>   FlashAttention-3 and -4 overlap the softmax with the tensor cores on Hopper and
>   Blackwell, where the exponential and shared-memory rates lag the matmul rate.

## Check your understanding

> [!QUESTION] The tiled kernel re-reads K and V once per query tile and still moves fewer bytes than standard attention. Why isn't that a contradiction?
> Standard attention moves four passes over an $N^2$ matrix, 536.9 MB at
> $N = 8192$, $d = 128$. The tiled kernel moves 272.6 MB if every re-read reaches
> HBM, and about 8.4 MB if the L2 serves the re-reads. The $N^2$ term is what the
> tiling removes.

> [!QUESTION] Why does the backward pass save the logsumexp and not $P$?
> $P$ has $N^2$ entries and the logsumexp has $N$. Given $L$, any tile of $P$ is
> $e^{\sigma Q_i \cdot K_j - L_i}$, which costs one matmul that the tensor cores do
> cheaply. Saving $P$ would put the matrix back in memory.

> [!QUESTION] Prove that $\sum_k P_{ik}\, dP_{ik} = dO_i \cdot O_i$.
> Substitute $dP_{ik} = dO_i \cdot V_k$ and move $dO_i$ out of the sum:
> $\sum_k P_{ik} (dO_i \cdot V_k) = dO_i \cdot \sum_k P_{ik} V_k = dO_i \cdot O_i$. A
> small kernel computes it from $O$ and $dO$ once, so the main loop never needs the
> whole row of $P$.

> [!QUESTION] Using atomic adds for dQ is faster than a second kernel. When would you still choose the second kernel?
> When you need reproducible gradients. Atomic float32 adds happen in scheduler
> order, and addition isn't associative, so two runs differ in the last bits. The
> second kernel is deterministic. It also avoids a float32 $dQ$ buffer in HBM and
> its conversion pass. The price is two more matmuls, seven in all instead of five.

> [!QUESTION] In the causal backward pass, how many query tiles does the last key-tile program visit, compared with the first?
> With $B_M = B_N = T$ tiles per side, key tile $j$ visits the query tiles from $j$
> to the end, which is $T - j$ of them. The first key tile visits all $T$, and the last
> visits 1. Early programs do the most work, the mirror image of the forward pass.

> [!QUESTION] An engineer replaces `repeat_interleave` on $K$ and $V$ with GQA indexing in the forward kernel and sees no memory saving on the KV cache. Why?
> The indexing saves bandwidth and avoids writing copies to HBM in the kernel, but
> the cache's size is set by how many KV heads the model stores. If the model still
> stores $H$ KV heads and repeats them in the cache, the cache is the same size.
> The cache shrinks only when $H_{KV}$ is stored and read directly.

> [!QUESTION] On a B200 the forward pass ties tensor cores and exponentials at 1024 cycles per tile. What does FlashAttention-4's software exponential buy?
> If both units are equally busy, the pass is limited by whichever is slower the
> moment they don't overlap perfectly. Computing part of $2^x$ on the FMA units
> takes load off the 16-per-cycle exponential unit, so the exponential time drops
> below the tensor-core time and the tensor cores become the bottleneck again.

## Lab

> [!TRY]
> Write FlashAttention-2 in Triton: the forward pass (returning $O$ and the
> logsumexp), the $\delta$ preprocessing kernel, the key-tile kernel for $dK$ and
> $dV$, and the query-tile kernel for $dQ$. The starter has the launch code and five
> TODOs. You pass when the forward and backward results match a float32 PyTorch
> reference and `torch.autograd` across seven shapes, causal and not, for head
> dimensions 64 and 128, including lengths 1, 100, 130, 257, and 333 and grouped
> heads, when the backward pass is deterministic, and when the forward pass's peak
> memory excludes the score matrix. The lab then reports TFLOP/s for the forward and
> backward passes next to `scaled_dot_product_attention` at $N = 4096$. It
> requires at least 0.33 times the vendor forward and 0.2 times the vendor backward.

The lab's performance figures are measurements, so the chapter makes predictions and
you check them:

- **Forward floor.** At $B = 4$, $H = 16$, $N = 4096$, $d = 128$ with a causal mask,
  the pass does $0.5 \times 4 \times 4 \times 16 \times 4096^2 \times 128 = 275$ GFLOP.
  At 312 TFLOP/s the floor is 0.88 ms. Your kernel will land well above it. At 8192 tokens, learn-inference's
  Triton forward reached 27% of peak on this card and the vendor kernel 43% (derived in
  [its chapter 14](https://qwen.ifkash.dev/c/14-flash-attention)), so expect tens of percent,
  not ninety.
- **Backward ceiling.** The lab counts $2.5\times$ the forward FLOPs but does seven
  matmuls, so its reported TFLOP/s can't exceed $312 \times 5 / 7 = 223$.
- **Head dimension.** $d = 64$ has half the arithmetic per score as $d = 128$, so the
  exponentials and masks matter more, and you can check whether your TFLOP/s
  falls below the $d = 128$ figure.

## Further reading

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): tiling, online softmax, and the IO analysis.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): the loop order, sequence parallelism, and warp partitioning this chapter describes.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): asynchrony and low precision on Hopper.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) and [Tri Dao's blog post](https://tridao.me/blog/2026/flash4/): the per-SM resource arithmetic on Blackwell.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) and [Self-attention does not need O(n^2) memory](https://arxiv.org/abs/2112.05682): the recurrence, derived independently.
- [The Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html): forward and backward in Triton.
- [FlexAttention](https://pytorch.org/blog/flexattention/) and the [flash-attention repository](https://github.com/Dao-AILab/flash-attention).
- The companion course: [chapter 14, FlashAttention for inference](https://qwen.ifkash.dev/c/14-flash-attention).
