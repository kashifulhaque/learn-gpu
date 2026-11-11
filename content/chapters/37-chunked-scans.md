---
title: Chunked scans for linear attention
slug: 37-chunked-scans
part: "Part 4 — Kernels for ML"
summary: Write a linear recurrence as an associative scan, turn decayed linear attention into a chunked algorithm with a masked matmul inside each chunk and a scan over chunk states between them, keep the decay in log space, and size the chunk from a FLOP and byte model.
minutes: 150
gpu: true
objectives:
  - "Write the linear recurrence h_t = a_t h_{t-1} + b_t as a scan over pairs, state the composition operator, and prove that it is associative but not commutative."
  - "Scan a first-order recurrence in parallel with Hillis-Steele, say what it costs in work and depth, and say why a kernel does most of the work sequentially inside a thread."
  - "Write linear attention as a matrix-valued recurrence S_t = g_t S_{t-1} + k_t v_t^T, and name the scalar-decay and per-channel-decay forms."
  - "Derive the chunked algorithm: the intra-chunk term as a masked matmul, the chunk states as a scan, and the output as an intra term plus an inter term."
  - "Compute the cumulative decay in log space, keep every exponent at or below zero, and explain why a cumulative product divided by another one overflows or gives NaN."
  - "Compare the FLOPs and the memory of the chunked form with softmax attention as a function of the chunk size C and the sequence length T, and find the chunk size that minimizes a roofline time."
lab: 37-chunked-scan
---

# Chunked scans for linear attention

> [!TLDR]
> - A recurrence $h_t = a_t h_{t-1} + b_t$ is a scan over pairs $(a, b)$. The operator $(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)$ is associative, so a GPU can regroup the work, but it isn't commutative, so the order of the operands matters.
> - Decayed linear attention keeps a matrix state, $S_t = g_t S_{t-1} + k_t v_t^{\top}$, and reads it with $o_t = q_t S_t$. Run one token at a time, it is a chain of small matrix-vector products that uses no tensor cores.
> - The chunked algorithm cuts the sequence into chunks of $C$ tokens. Inside a chunk, the output is a masked matmul, which tensor cores run. Between chunks, the states pass through a scan. Each output is an ==intra-chunk term plus an inter-chunk term==.
> - Keep the cumulative decay in log space and build the decay matrix from differences $b_i - b_j$ with $i \ge j$. Then every exponent is at most zero. A cumulative product divided by another one underflows to $0 / 0$ after 46 steps in float32 when $g = 0.1$.
> - Chunked FLOPs grow with $C$ and the bytes of chunk states shrink with $1/C$. A roofline model on the A100 puts the best size at 512 for $d_k = d_v = 128$, and it ignores shared memory and parallelism, so a measured sweep decides.

The scan in [chapter 7](/c/07-scan-and-histogram) adds numbers. A scan can do much more than that: any operator that is associative works, and one of them runs a recurrence in parallel. That recurrence is the heart of a family of sequence models, from linear attention to the state-space layers. Each token updates a fixed-size state, and the next token reads it. The state doesn't grow with the sequence, so decoding costs the same at token 100,000 as at token 100.

Training is where the trouble starts. The update is sequential, and a naive loop over tokens does small matrix-vector products that leave the tensor cores idle. The fix is the one [chapter 16](/c/16-flash-attention) used for attention: cut the work into tiles, and make each tile a matmul. This chapter builds the kernel that does it, the chunked scan, and the lab checks it against a one-token-at-a-time reference.

> [!NOTE] Who owns the models
> This chapter owns the kernel. The model side, meaning why linear attention and state-space layers exist, how they compare with softmax attention on language tasks, and the duality between them, belongs to the ml course's state-space models lesson. The chapter keeps its model section short and links out. Everything after that section is about how to make the computation fast and numerically safe.

## Before you start

**Scan.** [Chapter 7](/c/07-scan-and-histogram) defined the inclusive and exclusive scan, measured work and depth, and built Kogge-Stone, Blelloch, and the decoupled look-back scan. This chapter reuses the vocabulary and the Kogge-Stone shape, and it changes the operator from addition to something richer.

**Triton.** [Chapter 14](/c/14-triton) wrote a matmul with plain pointer arithmetic, `tl.dot`, and `tl.arange`. The lab kernel uses the same pieces plus `tl.cumsum`. A `tl.dot` on float32 inputs uses TF32 unless you ask for full precision, which matters for the tolerance below.

**FlashAttention.** [Chapter 16](/c/16-flash-attention) tiled attention so that the score matrix never reached HBM, and it skipped the blocks that causal masking zeroes out. The intra-chunk step here is the same causal tile, with a decay mask in place of the softmax.

**Floating point.** [Chapter 9](/c/09-floating-point-on-the-gpu) gave the exponent ranges. A float32 holds values down to about $1.4 \times 10^{-45}$ and a float64 down to about $4.9 \times 10^{-324}$, counting subnormals. Below that, a value is zero.

**The copy ceiling.** A device-to-device copy measured 1275 GB/s on the lab A100, against a rating of 1935 GB/s (learn-inference, chapter 10). This chapter turns byte counts into time predictions at that figure. A prediction is not a lab result.

## A recurrence is a scan

This section answers whether a loop that carries a value from one step to the next can run in parallel. It can, when the update has one particular shape.

Take the first-order linear recurrence, with a starting value $h_0 = 0$:

$$
h_t = \hla{a_t}\, h_{t-1} + \hlb{b_t}
$$

Here $\hla{a_t}$ is how much of the previous value survives, and $\hlb{b_t}$ is what step $t$ adds. With $a_t = 1$ it is a running sum, the scan of chapter 7. With $a_t = 0.9$ it is an exponential moving average. A decay that depends on the token, $a_t = g(x_t)$, is the recurrence inside every layer in this chapter.

### The trick: a step is a function

The picture is this: step $t$ is a small function of $h$, namely $h \mapsto a_t h + b_t$, and running the recurrence composes those functions in order. The function is described by two numbers, so you can carry the pair $(a_t, b_t)$ around in place of the step.

Composing two steps gives another step of the same kind. Run $(a_1, b_1)$ first and $(a_2, b_2)$ second, and an input $h$ becomes:

$$
a_2\,(a_1 h + b_1) + b_2 = \hla{(a_1 a_2)}\, h + \hlb{(a_2 b_1 + b_2)}
$$

So the composed step is the pair $(a_1 a_2,\; a_2 b_1 + b_2)$. Write that as an operator, with the earlier step on the left:

$$
\boxed{(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)}
$$

The operator takes two steps and returns the one step that does both. The scan of these pairs, from the first step to step $t$, is the pair $(A_t, B_t)$ with $h_t = A_t h_0 + B_t$, and with $h_0 = 0$ the answer is $B_t$.

### A worked example with real numbers

Take $a = [0.5, 0.5, 2, 1]$ and $b = [1, 2, 1, 3]$. The loop gives $h_1 = 1$, then $h_2 = 0.5 \times 1 + 2 = 2.5$, then $h_3 = 2 \times 2.5 + 1 = 6$, then $h_4 = 1 \times 6 + 3 = 9$.

Now use the operator, and group the first two steps and the last two steps separately:

- $(0.5, 1) \circ (0.5, 2) = (0.25,\; 0.5 \times 1 + 2) = (0.25, 2.5)$
- $(2, 1) \circ (1, 3) = (2,\; 1 \times 1 + 3) = (2, 4)$
- $(0.25, 2.5) \circ (2, 4) = (0.5,\; 2 \times 2.5 + 4) = (0.5, 9)$

The last pair has $B = 9$, which is $h_4$. The two groups could run on different threads, and the third line joins them. That regrouping is what a parallel scan needs.

### Why the grouping doesn't matter

A parallel scan regroups the work, so the operator must be *associative*: $(x \circ y) \circ z = x \circ (y \circ z)$. Check it with three pairs:

$$
\begin{aligned}
(x \circ y) \circ z &= (a_1 a_2,\; a_2 b_1 + b_2) \circ (a_3, b_3) = (a_1 a_2 a_3,\; a_3 a_2 b_1 + a_3 b_2 + b_3) \\
x \circ (y \circ z) &= (a_1, b_1) \circ (a_2 a_3,\; a_3 b_2 + b_3) = (a_1 a_2 a_3,\; a_2 a_3 b_1 + a_3 b_2 + b_3)
\end{aligned}
$$

The two sides are equal. The reason is that function composition is associative, and $\circ$ is composition in disguise. ==The operator is not commutative==: $(2, 3) \circ (4, 5) = (8, 17)$, while $(4, 5) \circ (2, 3) = (8, 13)$. A scan keeps the operands in order, with the earlier step on the left, and getting the order wrong is a silent bug, because the wrong operator $(a_1 a_2,\; a_1 b_2 + b_1)$ is itself associative. The lab's first check calls `combine((2, 3), (4, 5))` and expects $(8, 17)$.

> [!NOTE] The identity
> The pair $(1, 0)$ is the identity: $(1, 0) \circ (a, b) = (a, b)$. A scan needs an identity for its exclusive form and for padding the last tile, as [chapter 7](/c/07-scan-and-histogram) did with 0 for addition.

### Hillis-Steele over the operator

The parallel scan of chapter 7 carries over with `+` replaced by $\circ$. Kogge-Stone, which that chapter introduced, is also called Hillis-Steele. At step $d = 1, 2, 4, \ldots$, element $i \ge d$ becomes $x_{i-d} \circ x_i$, computed from the values before the step. After $\lceil \log_2 T \rceil$ steps, element $i$ holds the composition of steps $0$ through $i$.

```python
def scan_parallel(a, b):
    A, B = a.clone(), b.clone()
    d = 1
    while d < A.shape[0]:
        na, nb = combine((A[:-d], B[:-d]), (A[d:], B[d:]))   # earlier on the left
        A = torch.cat([A[:d], na])
        B = torch.cat([B[:d], nb])
        d *= 2
    return B                                                  # h_t, from h_{-1} = 0
```

Every step reads the old arrays and writes new ones, so the `torch.cat` copies avoid the in-place hazard that a GPU thread block avoids with `__syncthreads()`. The work is about $T \log_2 T$ operators, against $T - 1$ for the loop. Blelloch's up-sweep and down-sweep do $2(T-1)$ operators in $2\log_2 T$ steps, as in chapter 7, and they work for this operator too. The lab implements Hillis-Steele because it's shorter, and it works for any $T$, not only a power of two.

## A first-order scan in a kernel

This section answers how a real kernel runs that scan, and what it costs. The short answer is that the kernel is a chapter 7 scan with a pair as its element, and it's bound by memory.

A device-wide scan over pairs has the same three levels as chapter 7's scan over numbers:

1. **Inside a thread.** Each thread owns a short run of consecutive steps, say 4 to 16, and combines them in order with the sequential recurrence. This is the cheapest part: no communication, and the registers hold the running pair.
2. **Inside a block.** The threads' totals are scanned with a warp scan, using `__shfl_up_sync` on pairs, and a shared-memory handoff between warps. The combine is two multiplies and an add instead of one add.
3. **Across blocks.** Each block needs the composition of every block before it. Chapter 7's decoupled look-back works unchanged: a block publishes its aggregate pair with a flag, and a later block combines the pairs it finds.

Triton exposes the same idea as `tl.associative_scan(inputs, axis, combine_fn)`. Its combine function takes two tuples and returns one. A sketch of the pair operator in that form:

```python
@triton.jit
def _combine(a1, b1, a2, b2):
    return a1 * a2, a2 * b1 + b2

# inside a kernel, with a and b as one-dimensional tiles of a block's steps:
# _, h = tl.associative_scan((a, b), axis=0, combine_fn=_combine)
```

The sketch scans one tile. A recurrence longer than a tile still needs the cross-tile step above. The lab doesn't use `tl.associative_scan`, because the interesting scan in this chapter has matrices for its $b$ values, and the next sections show why a different structure wins there.

### What it costs

A scan over $T$ fp32 elements reads $a_t$ and $b_t$ and writes $h_t$: 12 bytes per element, and 3 FLOPs of arithmetic for each composition: two multiplies and an add. The intensity is far below the A100's ridge point, so the roofline from [chapter 4](/c/04-roofline-and-latency) says the kernel is memory bound. For $T = 2^{26}$ elements the traffic is $12 \times 2^{26} = 805{,}306{,}368$ bytes, and at the 1275 GB/s copy ceiling the floor is 0.632 ms. That is a prediction from the byte count. A look-back scan adds a little traffic for the published aggregates, so the lab measures how close a real kernel gets.

The selective state-space layer of Mamba needs a scan of the same kind. There the recurrence runs independently on every channel and state dimension, so $a_t$ and $b_t$ carry extra dimensions and the pair operator acts element by element. The next section is about the case where $b_t$ is a matrix, and that is where the structure changes.

## Linear attention as a matrix recurrence

This section says what the recurrence looks like in a sequence-model layer. It is short on purpose: the ml course's state-space models lesson explains why these layers work, and this chapter takes the formulas as given.

Softmax attention at token $t$ compares the query with every earlier key, normalizes, and mixes the values. *Linear attention* drops the softmax, so the sum can be reorganized. With queries and keys of width $d_k$ and values of width $d_v$:

$$
o_t = \sum_{j \le t} (q_t^{\top} k_j)\, v_j = q_t^{\top} \underbrace{\sum_{j \le t} k_j v_j^{\top}}_{S_t}
$$

The matrix $S_t$ has shape $d_k \times d_v$, and it grows by one rank-one term per token: $S_t = S_{t-1} + k_t v_t^{\top}$. The layer is an RNN whose state is a matrix. The cost per token is $2 d_k d_v$ FLOPs to update $S$ and $2 d_k d_v$ to read it, and the state has a fixed size. For $d_k = d_v = 128$ in fp32, the state is $128 \times 128 \times 4 = 65{,}536$ bytes, 64 KiB, per head. A softmax-attention KV cache for one head at $T = 4096$ in bf16 holds $4096 \times 2 \times 128 \times 2 = 2{,}097{,}152$ bytes, 2 MiB, and it keeps growing.

### Forgetting: the decay

A plain sum never forgets, and that hurts quality. The usual fix multiplies the old state by a decay $g_t \in (0, 1]$ before adding the new token:

$$
S_t = \hla{g_t}\, S_{t-1} + k_t v_t^{\top}, \qquad o_t = q_t^{\top} S_t
$$

The decay $\hla{g_t}$ comes in two shapes, and the difference decides the kernel:

| Form | Decay $g_t$ | Examples | Consequence for the kernel |
|---|---|---|---|
| Scalar per head | One number per token and head | RetNet (a constant per head), Mamba-2 | The decay factors out of the matmuls as a scalar mask. |
| Per-channel | A vector of $d_k$ numbers, applied as $\mathrm{diag}(\alpha_t)$ | Gated linear attention (GLA) | The decay depends on the row of $S$, so the mask has an extra dimension. |

The lab and the rest of the chapter use the scalar form. The per-channel form follows the same three steps, and the numerics section says what changes. For the models themselves, see the ml course's state-space models lesson, and the sources in the further reading for the abstracts of the papers. This chapter doesn't derive them or compare their quality.

### Why the loop is slow

Run the recurrence one token at a time and you do, per token, a rank-one update of a $d_k \times d_v$ matrix and a vector-matrix product. Both read the whole state and do two FLOPs per element, so the arithmetic intensity is about 1 FLOP per 2 bytes of fp32 state, well below the ridge point of chapter 4. If you store every $S_t$ for the backward pass, the memory is $T \times d_k \times d_v$ elements: 256 MiB per head at $T = 4096$ and $d_k = d_v = 128$ in fp32. None of it uses a tensor core, and the tensor cores are where an A100's 312 TFLOP/s (dense bf16) lives.

You can also unroll the recurrence into a *parallel* form. With the cumulative decay $b_t = \sum_{m \le t} \log g_m$, the output is $o = \left((Q K^{\top}) \odot D\right) V$ with $D_{ij} = e^{b_i - b_j}$ for $j \le i$. That form is three matmuls and no loop, but it builds a $T \times T$ matrix, which is the cost the whole family of models is trying to avoid. The chunked algorithm sits between the two forms.

## The chunked algorithm

This section answers how to get matmuls without a $T \times T$ matrix. Cut the sequence into chunks of $C$ tokens, use the parallel form inside each chunk, and pass a state between chunks.

The picture is a two-level schedule. Inside a chunk, every token can see every earlier token of the same chunk, so a small causal, decayed attention handles that. Everything before the chunk reaches it as one number of state: $S$ at the end of the previous chunk, which already summarizes all earlier tokens. The chunk's output is the sum of the two views.

### The three pieces

Number the tokens of one chunk $i = 1, \ldots, C$. Let $b_i = \sum_{m \le i} \log g_m$ be the cumulative log decay *inside the chunk*, so $b_i \le 0$ and it decreases. Let $S_{\text{in}}$ be the state entering the chunk. The three pieces follow from unrolling the recurrence from the chunk's start:

$$
S_i = e^{\hla{b_i}} S_{\text{in}} + \sum_{j \le i} e^{\hla{b_i - b_j}}\, k_j v_j^{\top}
$$

Reading it: the entering state decays by everything that happened since the chunk began, and each token $j$ in the chunk decays by the steps after it, up to $i$. Multiply by $q_i^{\top}$ to get the output at position $i$:

$$
\boxed{o_i = \underbrace{e^{b_i}\, q_i^{\top} S_{\text{in}}}_{\text{inter-chunk}} \;+\; \underbrace{\sum_{j \le i} e^{b_i - b_j}\,(q_i^{\top} k_j)\, v_j}_{\text{intra-chunk}}}
$$

The inter-chunk term is a matmul of the chunk's queries with one state. The intra-chunk term is $\left((Q K^{\top}) \odot D\right) V$ on a $C \times C$ block, the same thing as the parallel form, restricted to the chunk. The state that leaves the chunk, $S_{\text{out}}$, is the same unrolled sum evaluated at $i = C$:

$$
S_{\text{out}} = \underbrace{e^{b_C}}_{\hla{A}}\, S_{\text{in}} + \underbrace{\sum_{j \le C} e^{b_C - b_j}\, k_j v_j^{\top}}_{\hlb{U}}
$$

This is again $S \leftarrow A\, S + U$ with a scalar $A = e^{b_C}$ and a matrix $U$, one step of the first-order recurrence at the top of the chapter, taken per chunk instead of per token. ==Chunk states are scanned with the same operator.== $U$ depends only on the chunk's own keys and values, so every chunk computes its $(A, U)$ in parallel, with no knowledge of the other chunks.

### The three phases

That structure gives three phases, and each one is a good GPU workload:

1. **Chunk states.** For every chunk in parallel, compute $A_c$ and $U_c = (K \odot w)^{\top} V$ with weights $w_j = e^{b_C - b_j}$. That's one $d_k \times C$ by $C \times d_v$ matmul per chunk.
2. **State scan.** Scan the $N = \lceil T / C \rceil$ pairs $(A_c, U_c)$ with the operator of the first section, with $a$ a scalar and $b$ a matrix. The result is the state that enters each chunk, $S_{\text{in}, c}$. The scan is short: $N$ is $T / C$, so 64 for $T = 4096$ and $C = 64$.
3. **Outputs.** For every chunk in parallel, compute the intra term as a masked matmul and the inter term as one matmul against $S_{\text{in}, c}$, and add them.

The lab builds exactly this decomposition in torch: `chunk_states`, then `entering_states` over `scan_parallel`, then `chunked_linear_attention`.

> [!KEY] The scan has moved up a level
> A token-level scan over $T$ steps with $d_k \times d_v$ state becomes a chunk-level scan over $T / C$ steps with the same state, and the work inside each chunk becomes matmuls. The recurrence is still there, but the sequential part is $C$ times shorter and the rest runs on tensor cores.

### A worked example with real numbers

Take a scalar state to keep the arithmetic visible: $d_k = d_v = 1$, a decay of $g = 0.5$ at every step, $T = 6$, and $C = 3$. The inputs are $q = [1, 2, 1, 1, 3, 1]$, $k = [1, 1, 2, 1, 1, 2]$, and $v = [1, 2, 1, 3, 1, 1]$. The token-by-token loop $S_t = 0.5\, S_{t-1} + k_t v_t$, $o_t = q_t S_t$ gives:

| $t$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $S_t$ | 1 | 2.5 | 3.25 | 4.625 | 3.3125 | 3.65625 |
| $o_t$ | 1 | 5 | 3.25 | 4.625 | 9.9375 | 3.65625 |

Now the chunked way. For chunk 1, the cumulative decays are $e^{b} = [0.5, 0.25, 0.125]$, so $A_1 = 0.125$. The update is $U_1 = 1 \times 0.25 \times 1 + 1 \times 0.5 \times 2 + 2 \times 1 \times 1 = 3.25$. The chunk enters with $S_{\text{in}} = 0$, so its outputs are the intra terms alone: $[1, 5, 3.25]$, which match the loop.

Chunk 2 enters with $S_{\text{in}} = A_1 \times 0 + U_1 = 3.25$. Its inter terms are $e^{b_i}\, q_i\, S_{\text{in}} = [0.5 \times 1,\; 0.25 \times 3,\; 0.125 \times 1] \times 3.25 = [1.625,\; 2.4375,\; 0.40625]$. Its intra terms, from $q = [1, 3, 1]$, $k = [1, 1, 2]$, and $v = [3, 1, 1]$, are $[3,\; 7.5,\; 3.25]$. The sums are $[4.625, 9.9375, 3.65625]$, equal to the loop's last three outputs. The chunk's own update is $U_2 = 3.25$ again, and the final state $0.125 \times 3.25 + 3.25 = 3.65625$ matches $S_6$.

Two things in the example are worth carrying away. First, the inter term for position $i$ uses the decay up to *and including* step $i$, $e^{b_i}$. That position-$i$ decay is the line a plausible implementation gets off by one, using $e^{b_{i-1}}$, and the lab's boundary check catches it. Second, $A_1 = 0.125$ is the product of the three decays in the chunk, the number that carries the state across the boundary.

## Where the matmuls are

This section answers why the chunked form is faster than the loop, in terms of what each piece does on the hardware. The answer is that every piece except the decay mask is a plain matmul.

| Piece | Shape | FLOPs per chunk | Runs on |
|---|---|---|---|
| Scores $Q K^{\top}$ | $(C \times d_k)(d_k \times C)$ | $2 C^2 d_k$ | Tensor cores |
| Masked scores times $V$ | $(C \times C)(C \times d_v)$ | $2 C^2 d_v$ | Tensor cores |
| Inter term $Q\, S_{\text{in}}$ | $(C \times d_k)(d_k \times d_v)$ | $2 C d_k d_v$ | Tensor cores |
| State update $(K \odot w)^{\top} V$ | $(d_k \times C)(C \times d_v)$ | $2 C d_k d_v$ | Tensor cores |
| Decay mask $D$ | $C \times C$ | About $C^2$ exponentials | Elementwise |

Two of these pieces already appeared in other chapters. The first two rows are chapter 16's attention tile, with a decay mask in place of the softmax: there is no max, no running denominator, and no rescaling, because the mask is a fixed function of the decays. The other rows are ordinary tensor-core matmuls, the subject of chapters 11 and 14. The mask is the only new elementwise work.

The count over a whole sequence follows from the table. With $N = T / C$ chunks and every block counted in full, the FLOPs are:

$$
\boxed{F_{\text{chunk}} = T\,\bigl(2C\,(d_k + d_v) + 4\,d_k d_v\bigr)}
$$

The first term grows with $C$, because each chunk's $C \times C$ block holds $C^2$ pairs and there are $T / C$ of them. The second term is independent of $C$: it is the state read and the state update, $4 d_k d_v$ per token. Causal masking wastes half of each full block, and the kernel computes the full block anyway, because a tensor-core tile can't skip individual elements. That's the price of the matmul.

## Keeping the decay finite

This section answers why the obvious way to build the decay mask produces NaN, and what to do instead. Every exponent in the algorithm should be at most zero.

The mask entry is $D_{ij} = e^{b_i - b_j} = \prod_{m = j+1}^{i} g_m$ for $j \le i$. The most direct way to compute it is with cumulative products: let $P_i = \prod_{m \le i} g_m$ and set $D_{ij} = P_i / P_j$. In float64 that looks harmless, and it fails on long chunks.

### The cumulative product underflows

Each $g_m < 1$ shrinks $P$. After enough steps the product is below the smallest number the format holds, and it becomes exactly 0. Then $P_i / P_j$ is $0 / 0$, which is NaN. The lab's figure and the following table come from running `torch.cumprod` on a constant decay:

| Decay $g$ | float32: $P = 0$ from step | float64: $P = 0$ from step |
|---|---|---|
| 0.1 | 46 | 324 |
| 0.5 | 150 | 1075 |
| 0.9 | 987 | none by step 4096 |
| 0.99 | none by step 4096 | none by step 4096 |

A decay of 0.5 per token looks mild, and a float32 chunk of 150 tokens already breaks. The lab tests a stronger decay, $e^{-6} \approx 0.0025$ per step, and there a float64 cumulative product reaches zero well inside a chunk of 200.

```viz
37-decay-underflow
```

### The other direction overflows

The failure has a twin. If you compute $e^{b_i - b_j}$ for *every* pair and then multiply by the lower-triangular mask, the entries above the diagonal have $j > i$, so $b_i - b_j > 0$. With a decay of $e^{-6}$ per step over 200 tokens, the exponent reaches $6 \times 199 = 1194$, and $e^{1194}$ overflows a float64 to $\infty$. Then $\infty \times 0$ is NaN. The mask came too late.

The fix is to mask the *exponent*, not the result: replace the above-diagonal entries of $b_i - b_j$ with $-\infty$ before calling `exp`, and $e^{-\infty} = 0$ exactly. The lab's `decay_matrix` does this. Its check uses a decay of $e^{-6}$ over 200 tokens: the masked-exponent form returns a finite matrix with $D_{ij} = e^{-6(i-j)}$, and the same input through a cumulative-product ratio puts 2926 NaN entries in float64.

### The rule

Keep $b$, the cumulative *log* decay, and form only differences of it. Every term of the algorithm has an exponent of the form $b_i - b_j$ with $i \ge j$, or $b_C - b_j$, or $b_i$ itself, and each one is at most zero:

| Where | Exponent | Sign |
|---|---|---|
| Decay mask $D_{ij}$ | $b_i - b_j$, $j \le i$ | $\le 0$ |
| Inter term | $b_i$ | $\le 0$ |
| State update weights | $b_C - b_j$ | $\le 0$ |
| Chunk decay $A$ | $b_C$ | $\le 0$ |

So every factor lies in $(0, 1]$. Nothing can overflow, and a factor that underflows to 0 is the right answer: that token's contribution really has decayed away. The cumulative sum $b$ of log decays is also better conditioned than the product: it's a sum of bounded negative numbers, so it only grows linearly in the chunk length.

> [!WARNING] The per-channel case
> For a per-channel decay $\mathrm{diag}(\alpha_t)$, the intra-chunk term can't use one $C \times C$ mask. A common way to keep it a matmul is to scale the queries by $e^{b_i}$ and the keys by $e^{-b_j}$, and the $e^{-b_j}$ factor is the positive exponent that overflows. Implementations that take this route bound the factor by working in smaller sub-chunks, or by subtracting a per-sub-chunk reference. The scalar form in this chapter doesn't have the problem, because its mask is built from differences directly.

### Precision

Keep the decay and the state in float32, even when $q$, $k$, and $v$ are bf16. The state sums $T$ rank-one terms with decay, and a bf16 accumulator would add the bias that [chapter 9](/c/09-floating-point-on-the-gpu) described for long sums. The lab's CPU checks run in float64 so that the comparison with the reference isolates the algorithm from the rounding, and the GPU check uses float32 inputs with a tolerance of $2 \times 10^{-3}$ of the output's scale.

## What chunking costs and saves

This section answers how the chunked form compares with softmax attention in FLOPs and in memory, and how the chunk size sets the balance. The model is arithmetic, not a measurement.

### FLOPs against softmax attention

Causal softmax attention, counting the lower triangle only, costs $T^2 (d_k + d_v)$ FLOPs: $T^2 / 2$ pairs, each costing $2 d_k$ for the score and $2 d_v$ for the mix. The chunked form costs $F_{\text{chunk}}$ from above. The ratio of per-token costs, $T (d_k + d_v)$ against $2C(d_k + d_v) + 4 d_k d_v$, crosses 1 at:

$$
\boxed{T^{\ast} = 2C + \frac{4\, d_k d_v}{d_k + d_v}}
$$

For $d_k = d_v = 128$ the second term is 256, so with $C = 64$ the crossover is $T^{\ast} = 384$ tokens. Above that the chunked form does fewer FLOPs, and the gap widens linearly. The lab's `crossover_length` returns this number. For $d_k = d_v = 128$, the table shows the FLOPs:

| $T$ | Softmax (causal) | Chunked, $C = 64$ | Ratio |
|---|---|---|---|
| 4,096 | 4.295 GFLOP | 0.403 GFLOP | 10.7 |
| 16,384 | 68.719 GFLOP | 1.611 GFLOP | 42.7 |
| 65,536 | 1099.512 GFLOP | 6.442 GFLOP | 170.7 |

Softmax attention's FLOPs grow with $T^2$ and the chunked form's with $T$. The table counts per head and per sequence, for the forward pass only. A FlashAttention kernel also avoids the HBM traffic of the score matrix, as chapter 16 showed, so the FLOP ratio is not a time ratio: both sides are limited by how well they use the hardware.

### Memory

Three quantities matter:

- **State at inference.** The recurrent form carries one $d_k \times d_v$ state per head: 64 KiB in fp32 for 128 by 128. It doesn't grow with $T$. A KV cache grows by $2 d \times 2$ bytes per token per head in bf16.
- **Chunk states in the three-phase form.** Phase 1 writes $N$ states, and phase 3 reads them back. For $T = 4096$ and $C = 64$ that's $64 \times 64\ \text{KiB} = 4\ \text{MiB}$ per head, against 256 MiB for the token-by-token form that stores every state. For $C = 256$ it is 1 MiB.
- **Intermediate scores.** The $C \times C$ block lives in registers or shared memory, never in HBM. At $C = 128$ in fp32 it's $128 \times 128 \times 4 = 65{,}536$ bytes, 64 KiB, a large share of the 164 KiB of shared memory on an A100 SM, so a bigger chunk eventually costs occupancy.

### The chunk-size trade-off

Pull the two pieces together. The FLOPs rise with $C$: the intra-chunk block is $C \times C$. The state traffic falls with $1/C$: there are fewer states to write and read. If the three-phase form moves $q$, $k$, $v$, and $o$ once in bf16, $2 (d_k + d_v)$ elements per token, and each of the $N$ states once out and once back in fp32, the bytes are:

$$
B_{\text{chunk}} = 2\,T\,(d_k + d_v) \cdot 2 \;+\; 2 \cdot \frac{T}{C} \cdot d_k d_v \cdot 4
$$

The roofline time is the larger of $F_{\text{chunk}} / 312\ \text{TFLOP/s}$ and $B_{\text{chunk}} / 1275\ \text{GB/s}$. The first rises with $C$ and the second falls, so the larger of the two has a minimum where they cross.

```viz
37-chunk-tradeoff
```

For $d_k = d_v = 128$, this model puts the crossing at $C = 512$ for each of $T = 4096$, $16{,}384$, and $65{,}536$. At $T = 16{,}384$ and $C = 64$, the byte time is 39.476 microseconds against a FLOP time of 5.162 microseconds, so the chunked kernel at that size is memory bound by a factor of about 7.6, and the larger chunks win in the model because they move fewer states.

Read that result with its limits. The model ignores several things that favor a smaller chunk: a $512 \times 512$ fp32 score block is 1 MiB and doesn't fit in the 164 KiB of shared memory on an SM, the mask's exponentials are not tensor-core work, and the parallelism across chunks is $T / C$ programs per head, so a larger $C$ leaves fewer programs to fill 108 SMs. The model's job is to show *why* there's a trade-off and which way each knob moves it, not to name the winner. The lab's `best_chunk` returns the model's minimum, and the Triton sweep measures the real one.

## The Triton kernel in the lab

This section answers what the lab's GPU kernel looks like. It's the simplest correct form, and the next section says what it leaves out.

The kernel runs one program per head. Each program loops over that head's chunks in order and carries the state $S$ in registers, so the "scan over chunk states" is the loop itself, and phase 1 and phase 3 fuse into one pass. For each chunk it does the following:

```python
@triton.jit
def _chunk_fwd_kernel(q_ptr, k_ptr, v_ptr, lg_ptr, o_ptr, T,
                      DK: tl.constexpr, DV: tl.constexpr, C: tl.constexpr):
    head = tl.program_id(0).to(tl.int64)
    offs_c = tl.arange(0, C)
    causal = offs_c[:, None] >= offs_c[None, :]
    s = tl.zeros((DK, DV), dtype=tl.float32)            # the state, in registers
    for start in range(0, T, C):
        # ... load q, k, v tiles and the C log decays, masked at the end of the sequence ...
        b = tl.cumsum(lg, axis=0)                       # cumulative log decay in the chunk
        b_last = tl.sum(lg, axis=0)
        diff = tl.where(causal, b[:, None] - b[None, :], -float("inf"))
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * tl.exp(diff)
        o = tl.dot(scores, v, input_precision="ieee")                    # intra
        o += tl.exp(b)[:, None] * tl.dot(q, s, input_precision="ieee")   # inter
        # ... store o ...
        kd = k * tl.exp(b_last - b)[:, None]
        s = s * tl.exp(b_last) + tl.dot(tl.trans(kd), v, input_precision="ieee")
```

Four details carry the numerics from the previous section:

- **Masked exponent.** `tl.where(causal, b_i - b_j, -inf)` before `tl.exp`, the same fix as the CPU `decay_matrix`.
- **Ragged tail.** Positions at or beyond $T$ load zeros for $k$, $v$, and the log decay, so a padded token adds nothing to the state and multiplies it by $e^{0} = 1$. The last chunk needs no special case.
- **Full precision dots.** `input_precision="ieee"` keeps float32 products at float32. Without it, `tl.dot` on float32 inputs uses TF32, which rounds the operands to a 10-bit mantissa, and the comparison with a float64 reference would fail at the lab's tolerance.
- **Sizes.** The tile dimensions of `tl.dot` must be at least 16, so $d_k$, $d_v$, and $C$ are powers of two of 16 or more.

### What this kernel leaves out

One program per head means $H$ programs. With 16 heads, 16 of the A100's 108 SMs have work, about 15%, however long the sequence is. The three-phase decomposition removes that limit: phases 1 and 3 launch $H \times N$ programs, and only the short state scan is sequential. A production kernel also fuses what it can, keeps the states in a separate buffer between kernels, and has a backward pass that recomputes the intra-chunk block, like chapter 16's backward pass. The lab's kernel shows the arithmetic of one chunk, and its sweep shows how $C$ moves the time, but it isn't a tuned kernel, and the sweep result is a measurement of this kernel only.

## What goes wrong

- **The operands in the wrong order.** $(a_1 a_2,\; a_1 b_2 + b_1)$ is also associative, so an associativity test passes. Only a comparison with the sequential loop fails. Test on a pair where the order matters, such as $(2, 3) \circ (4, 5) = (8, 17)$.
- **A decay that includes the token itself.** The recurrence adds token $t$ after the decay, so the token doesn't decay at its own step. Applying $g_t$ to $k_t v_t^{\top}$ as well changes every output by a factor that is invisible in a test with $g = 1$.
- **The inter term decayed by the wrong boundary.** The inter-chunk output at position $i$ needs $e^{b_i}$, which includes step $i$. Using $e^{b_{i-1}}$, the decay before position $i$, is off by one position. The first chunk is unaffected, because its entering state is zero, so test several chunks. The same shift on the state update, $e^{b_C - b_{j-1}}$, shows up in the state that enters the next chunk.
- **Cumulative-product ratios.** $P_i / P_j$ gives $0 / 0$ once $P$ underflows: after 46 steps in float32 at $g = 0.1$. Build $D$ from differences of the log decay.
- **Masking after the exponent.** $e^{b_i - b_j}$ for $j > i$ is huge and positive, and $\infty \times 0$ is NaN. Mask the exponent with $-\infty$ first.
- **A chunk size that doesn't divide $T$.** The last chunk is short. Loops that assume $T / C$ is an integer drop or duplicate tokens. Test with $C = 7$ and $T = 50$.
- **TF32 in the GPU dot.** A float32 `tl.dot` rounds its inputs to TF32 by default. The result is wrong in the third decimal against a float64 reference, with no error message.
- **bf16 state.** Accumulating the state in bf16 loses small contributions over a long sequence. Keep the state and the decay in float32.
- **Reading the model as a measurement.** The cost model says memory bound and a best $C$ of 512 for $d_k = d_v = 128$. It leaves out the shared memory limit and the parallelism. Measure the sweep before you pick $C$.

> [!RECAP]
> - A recurrence $h_t = a_t h_{t-1} + b_t$ is a scan over pairs with the associative, non-commutative operator $(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2,\; a_2 b_1 + b_2)$. Hillis-Steele scans it in $\lceil \log_2 T \rceil$ steps.
> - Decayed linear attention is that recurrence with a matrix state, $S_t = g_t S_{t-1} + k_t v_t^{\top}$ and $o_t = q_t S_t$. Run per token, it has low arithmetic intensity and uses no tensor cores.
> - The chunked algorithm computes the intra-chunk term as a masked matmul, passes chunk states through a scan with the same operator, and adds the inter-chunk term $e^{b_i}\, q_i^{\top} S_{\text{in}}$.
> - Build the decay from differences of the cumulative log decay and mask the exponent before `exp`. Then every exponent is at most zero, and nothing overflows or becomes $0 / 0$.
> - Chunked FLOPs are $T(2C(d_k + d_v) + 4 d_k d_v)$ against $T^2 (d_k + d_v)$ for causal softmax, crossing at $T^{\ast} = 2C + 4 d_k d_v / (d_k + d_v)$, 384 tokens for $C = 64$ and $d_k = d_v = 128$.
> - The chunk size trades intra-chunk FLOPs, which grow with $C$, against state traffic, which shrinks with $1/C$. The model finds the crossing, and the lab's sweep measures it.

## Check your understanding

> [!QUESTION] Compose $(0.5, 1) \circ (0.5, 2)$, then show with the pair $(1, 0)$ why a scan can pad its last tile.
> $(0.5, 1) \circ (0.5, 2) = (0.25,\; 0.5 \times 1 + 2) = (0.25, 2.5)$. The pair $(1, 0)$ is the identity: $(1, 0) \circ (a, b) = (a,\; a \cdot 0 + b) = (a, b)$, and $(a, b) \circ (1, 0) = (a,\; 1 \cdot b + 0) = (a, b)$. A padded slot that holds the identity changes no composition, just as 0 does for a sum.

> [!QUESTION] The operator $(a_1, b_1) \circ' (a_2, b_2) = (a_1 a_2,\; a_1 b_2 + b_1)$ is associative. Why does a scan with it give the wrong answer?
> Associativity only lets you regroup, and it says nothing about which function runs first. This operator applies step 2 before step 1: for $(2, 3)$ and $(4, 5)$ it gives $(8, 13)$, where the recurrence needs $(8, 17)$. The scan result at step $t$ is then not $h_t$. A comparison with the sequential loop catches it, and an associativity test doesn't.

> [!QUESTION] For $T = 4096$, $C = 64$, and $d_k = d_v = 128$, how many chunk states does the three-phase form store per head, and how many bytes is that in fp32?
> $N = 4096 / 64 = 64$ states, each $128 \times 128 \times 4 = 65{,}536$ bytes. The total is $64 \times 65{,}536 = 4{,}194{,}304$ bytes, 4 MiB, against 256 MiB for storing every per-token state.

> [!QUESTION] In the worked example, chunk 2 enters with $S_{\text{in}} = 3.25$. Compute the inter-chunk term of its first position, and say which boundary decay it uses.
> The first position has $q = 1$ and $b_1 = \log 0.5$, so the term is $e^{b_1}\, q\, S_{\text{in}} = 0.5 \times 1 \times 3.25 = 1.625$. It uses the decay up to and including the position's own step, $e^{b_i}$. A version that used $e^{b_{i-1}}$ would give $1 \times 3.25 = 3.25$ and an output of 6.25 instead of 4.625.

> [!QUESTION] A chunk has 200 tokens with $g = 0.5$ each, in float32. Why does a cumulative-product decay matrix fail, and which numbers in the log-space version stay finite?
> $P_i = 0.5^i$ reaches the float32 floor of about $1.4 \times 10^{-45}$ at step 150, where the table says the product becomes 0. After that, $P_i / P_j$ is $0 / 0$ for $j \ge 150$. In log space, $b_i = -0.6931\, i$ is a modest negative number, at most about 139 in magnitude, and each mask entry $e^{b_i - b_j}$ with $j \le i$ is in $(0, 1]$. An entry that is too small to represent underflows to 0, which is the correct value.

> [!QUESTION] At what sequence length does the chunked form with $C = 128$ and $d_k = d_v = 64$ do fewer FLOPs than causal softmax attention?
> $T^{\ast} = 2C + 4 d_k d_v / (d_k + d_v) = 256 + 4 \times 64 \times 64 / 128 = 256 + 128 = 384$ tokens. Above 384 the chunked form's FLOPs per token, $2C(d_k + d_v) + 4 d_k d_v = 32{,}768 + 16{,}384 = 49{,}152$, are below softmax's $T (d_k + d_v) = 128\, T$.

> [!QUESTION] The lab's kernel runs 16 heads at any sequence length and uses about 15% of an A100. How does the three-phase form fix that?
> The kernel launches one program per head, so 16 programs fill 16 of 108 SMs. Phases 1 and 3 of the three-phase form have no dependence between chunks, so they launch $H \times N$ programs, for example $16 \times 128 = 2048$ at $T = 8192$ and $C = 64$. Only the scan over the $N$ chunk states is sequential, and it's short.

## Lab

> [!TRY] Chunked scans for linear attention
> Open `37-chunked-scan`. Check 1 needs only `combine`: return $(a_1 a_2,\; a_2 b_1 + b_2)$. Then check associativity on random triples, write `scan_sequential` and a Hillis-Steele `scan_parallel`, and write the sequential reference `linear_attention_sequential`. Build `decay_matrix` in log space, `chunk_states`, `entering_states` over your parallel scan, and `chunked_linear_attention`, which must equal the reference for several chunk sizes, including one that doesn't divide $T$, and under a decay that overflows a cumulative product. The last CPU piece is the cost model: `chunk_flops`, `chunk_bytes`, `best_chunk`, and `crossover_length`. Nine checks run in torch float64 on a CPU. The tenth is the Triton chunked forward against the float64 reference at chunk sizes 16, 32, 64, and 128. It runs only on a machine with an A100 and Triton, and it's skipped otherwise.

The lab reports these metrics, all from the Triton kernel:

- `ms_c16`, `ms_c32`, `ms_c64`, `ms_c128`: the median time of the chunked forward pass for 16 heads, $T = 8192$, and $d_k = d_v = 64$, at each chunk size. They are measurements of the lab's one-program-per-head kernel, so they don't say how a tuned three-phase kernel behaves.
- `best_chunk`: the chunk size with the smallest of those times. A machine without a GPU records `skipped (no GPU)`.

The cost-model check passes for any correct model, and it doesn't compare the model with the measured sweep. Put the two side by side yourself, and ask which of the model's omissions explains the gap.

## Further reading

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), Katharopoulos, Vyas, Pappas, and Fleuret. Rewrites attention with kernel feature maps and uses the associativity of the matrix product to make the cost linear in sequence length, which exposes the recurrent form.
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635), Yang, Wang, Shen, Panda, and Kim. A linear-attention algorithm with a data-dependent gate that trades memory movement against parallelizability, with the FlashLinearAttention implementation. Read it for the per-channel decay this chapter only names.
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060), Dao and Gu. Connects state-space models and attention through structured matrices, and the abstract reports a core layer 2 to 8 times faster than Mamba's. I read the abstract, not the algorithm sections.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), Gu and Dao. Input-dependent state-space parameters rule out a convolution, so the authors designed a hardware-aware parallel algorithm in recurrent mode. This is the first-order scan of the second section, applied per channel.
- [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621), Sun and colleagues. Defines parallel, recurrent, and chunkwise recurrent computation modes for one layer, and the chunkwise mode is the schedule of this chapter.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention). A library of Triton implementations of linear attention and related layers, including RetNet, GLA, and Mamba-2 layers. Read its kernels after the lab, to see the three-phase form and the backward pass that the lab leaves out.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures). The index lists lectures 20 and 21, "Scan Algorithm" by Izzat El Haj, and lecture 24, "Scan at the Speed of Light" by Jake Hemstad and Georgii Evtushenko. I read the titles in the index, not the slides.
- [Chapter 7](/c/07-scan-and-histogram), for the scan this chapter generalizes. [Chapter 16](/c/16-flash-attention), for the tile that the intra-chunk step reuses. [Chapter 14](/c/14-triton), for the Triton constructs the kernel uses. The ml course's state-space models lesson, for the models themselves.
