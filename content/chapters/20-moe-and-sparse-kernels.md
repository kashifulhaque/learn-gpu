---
title: Mixture-of-experts and sparse kernels
slug: 20-moe-and-sparse-kernels
part: "Part 4 — Kernels for ML"
summary: Why a loop over experts wastes the GPU, how to sort tokens by expert and run every expert's matmul in one grouped GEMM launch, and when 2:4 structured sparsity pays and when it doesn't.
minutes: 100
gpu: true
objectives:
  - "Describe a mixture-of-experts layer (router, top-k, gate weights, experts) and compute how many tokens an expert needs before its matmul is compute bound on an A100."
  - "Explain why a Python loop over experts is slow: launch overhead, tiny grids, tile padding, and load imbalance."
  - "Build the token permutation with a histogram, an exclusive scan, and a stable sort, and count the bytes that the gather, the scatter, and the combine move."
  - "Write a Triton grouped GEMM whose program ids map to (expert, tile) pairs, and size its launch grid without knowing the per-expert counts."
  - "Fuse the gate weights into the GEMM's epilogue, and say what the unpermute and combine still cost."
  - "Compare padded and exact grouped GEMM, the block-sparse formulation in MegaBlocks, and expert parallelism with its all-to-all."
  - "Explain why unstructured sparsity rarely speeds up a GPU, and describe the 2:4 format, its metadata, and its measured speedups."
lab: 20-grouped-gemm
---

# Mixture-of-experts and sparse kernels

> [!TLDR]
> - A mixture-of-experts (MoE) layer sends each token to the top few of many expert MLPs. The arithmetic per token stays small, but the matmuls become many, small, and uneven, and a loop that launches one matmul per expert leaves the GPU idle.
> - Sort the tokens by expert with a histogram, an exclusive scan, and a stable sort. Each expert then owns a contiguous range of rows, and one *grouped GEMM* launch multiplies every range by its expert's weights.
> - A grouped GEMM maps each program id to an (expert, tile) pair. The grid is sized with the bound $\lceil R / B_M \rceil + E$ for $R$ rows, $E$ experts, and row-tile height $B_M$, and spare programs exit at once.
> - Fuse the gate weights into the GEMM's epilogue, and the cost that remains is moving rows: gather, scatter, and combine.
> - Unstructured sparsity rarely pays on a GPU, because the dense tensor cores out-run the CUDA cores by 16 times. 2:4 structured sparsity uses sparse tensor cores rated at 2x dense and reaches about 1.3x to 1.6x in PyTorch on an A100.

A dense transformer spends the same arithmetic on every token. A mixture-of-experts transformer breaks that rule on purpose. Mixtral 8x7B holds about 47 billion parameters, but each token uses about 13 billion of them ([Jiang et al., 2024](https://arxiv.org/abs/2401.04088)). That ratio buys model capacity at a fixed per-token cost.

The cost moves from FLOP to kernels. In a dense layer, one big matmul covers all tokens. In an MoE layer, each expert sees only the tokens that chose it, and how many that is changes on every batch. A kernel that handles this well uses the GPU almost as efficiently as a dense layer, and a loop that handles it badly runs at a small fraction of the card's speed. This chapter builds the good kernel from pieces you already have: the histogram and scan from [chapter 7](/c/07-scan-and-histogram), the tiled matmul from [chapter 14](/c/14-triton), and the roofline from [chapter 4](/c/04-roofline-and-latency).

The second half asks a related question: when you zero out half of a weight matrix, can the GPU use the zeros? For almost every sparsity pattern the answer is no. For one pattern, 2:4, the answer is a qualified yes, and you see why.

## Before you start

**Top-k.** For each row of a matrix of scores, top-k returns the $k$ largest entries and their column indices. `torch.topk` does it, and the router uses it once per token.

**GEMM FLOP count.** Multiplying an $M \times K$ matrix by a $K \times N$ matrix costs $2MNK$ FLOP: one multiply and one add per term. A tile of the output is the unit of work that one thread block, or one Triton program, computes ([chapter 14](/c/14-triton)).

**Ridge point.** The A100 80GB PCIe is rated at 312 TFLOP/s for bfloat16 and 1935 GB/s of memory bandwidth, so its ridge point is $312 \times 10^{12} / 1.935 \times 10^{12} = 161$ FLOP per byte ([chapter 4](/c/04-roofline-and-latency)). A kernel below that intensity waits on memory.

**Histogram and exclusive scan.** A histogram counts how many items fall in each bin. An exclusive scan turns those counts into the starting offset of each bin. [Chapter 7](/c/07-scan-and-histogram) builds both on the GPU, and this chapter uses them to find where each expert's rows start.

**Launch overhead.** A kernel launch costs a few microseconds of host time. [Chapter 0](/c/00-what-you-are-building) quotes about 5 µs of queue time on the lab A100, so a kernel that finishes in a microsecond is launch bound.

## Anatomy of an MoE layer

An MoE layer replaces the single feed-forward network (FFN) of a transformer block with $E$ parallel FFNs, called *experts*, and a small *router* that picks which experts see each token.

The router is one linear layer. It maps a token's hidden vector, of size $d$, to $E$ scores called *logits*. The layer keeps the $k$ highest scores for the token and ignores the rest, which is *top-k routing* ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)). A softmax over the $k$ kept logits gives the *gate weights* $g_1, \dots, g_k$, which sum to 1. Mixtral picks $k = 2$ of $E = 8$ this way.

Each chosen expert applies its own FFN to the token, and the layer returns the gate-weighted sum:

$$
y = \sum_{j=1}^{k} \hla{g_j} \cdot \hlb{\text{FFN}_{e_j}}(x)
$$

Here $\hla{g_j}$ is the gate weight for the $j$-th chosen expert, and $\hlb{\text{FFN}_{e_j}}$ is that expert's MLP. The token's work is $k$ expert FFNs, not $E$.

An expert in a modern model is a gated MLP with three weight matrices: two of shape $d \times f$ (gate and up projections) and one of shape $f \times d$ (the down projection). The lab simplifies to one matrix per expert, a $D \times N$ weight, because the kernel structure is identical and a full MLP repeats it three times.

### What one expert costs

Take Mixtral 8x7B's shapes: $d = 4096$ and $f = 14336$, per its [published configuration](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json). One expert has $3 \times 4096 \times 14336 = 176{,}160{,}768$ parameters. In bfloat16 that is 352.3 MB of weights. Each token that passes through the expert costs $2 \times 176.2 \times 10^{6} = 352.3$ MFLOP.

Now ask how busy a matmul is when an expert receives $n$ tokens. The matmul reads the whole weight matrix once, whatever $n$ is, and does $2nP$ FLOP for $P$ parameters. In bfloat16 the weights are $2P$ bytes, so:

$$
I = \frac{2 n P \ \text{FLOP}}{2 P \ \text{bytes}} = \boxed{n \ \text{FLOP per byte}}
$$

The intensity of an expert's matmul equals its token count, when the weights dominate the traffic. Setting it equal to the ridge point gives the threshold: an expert reaches the compute-bound side of the roofline at about 161 tokens.

> [!KEY] An expert needs about 161 tokens to be compute bound
> With bfloat16 weights, an expert's intensity is $n$ FLOP/byte for $n$ tokens, so it crosses the A100's ridge point of 161 near $n = 161$. With fewer tokens, the expert's matmul reads weights faster than it can use them.

The batch size that gives every expert 161 tokens depends on the layout. With $B$ tokens, top-$k$ routing, and $E$ experts under perfect balance, each expert gets $kB/E$ tokens.

| Layout | Tokens per expert | Batch for 161 tokens per expert |
|---|---|---|
| Mixtral-like: 8 experts, top-2 | $B/4$ | 644 |
| 64 experts, top-8 | $B/8$ | 1288 |
| 256 experts, top-8 | $B/32$ | 5152 |

The last two rows are example layouts, in the style of fine-grained designs such as [DeepSeek-V3](https://arxiv.org/abs/2412.19437), which activates 37 billion of 671 billion parameters per token. They are not that model's exact configuration. The trend is the point: finer experts need larger batches to keep each matmul compute bound. During decode, with a batch of a few dozen sequences, almost every expert matmul is memory bound, and the kernel's job is to read each active expert's weights once at full bandwidth.

## Why a loop over experts is slow

The obvious implementation loops over experts in Python: select the tokens for expert $e$, multiply by its weights, and add the result back. It's correct, and it fails on the GPU in four separate ways.

```python
def moe_loop(x, w, ids, gates):
    out = torch.zeros(x.shape[0], w.shape[2], dtype=x.dtype, device=x.device)
    for e in range(w.shape[0]):
        rows, slot = torch.where(ids == e)          # host sync: the size is data dependent
        if rows.numel() == 0:
            continue
        y = (x[rows] @ w[e]) * gates[rows, slot, None].to(x.dtype)
        out.index_add_(0, rows, y)
    return out
```

**Launch overhead.** Each pass of the loop launches several kernels: the comparison, the `where`, the gather, the matmul, the scale, and the `index_add_`. Consider 64 experts and 8192 rows, about 128 rows per expert, with $1024 \times 1024$ weights. One expert's matmul is $2 \times 128 \times 1024 \times 1024 = 268$ MFLOP, which takes $268 \times 10^{6} / 312 \times 10^{12} = 0.86\ \mu\text{s}$ at the rated peak. A launch costs about 5 µs. The matmul is launch bound by a factor of nearly 6 before it starts.

**Host synchronization.** `torch.where` returns a tensor whose length depends on the data, so PyTorch must copy the count to the host and wait. The host can't queue the next expert until the GPU finishes the last one, and the GPU sits idle during each wait.

**Tiny grids.** A $128 \times 1024$ output cut into $128 \times 128$ tiles has $1 \times 8 = 8$ tiles. The A100 has 108 SMs, so one expert's matmul keeps 8 of them busy, 7.4% of the machine. The other 100 SMs idle until the next launch.

**Load imbalance.** Routers don't send equal numbers of tokens to every expert. A popular expert gets several times the mean, and a rare one gets nothing. A fixed-size buffer per expert either overflows or wastes space, which brings the capacity factor and token dropping that [Switch Transformers](https://arxiv.org/abs/2101.03961) introduced and that MegaBlocks removes (below).

The figure shows how skew in the load changes the picture. Raise the skew and the number of experts under 161 rows rises, even though the mean stays fixed.

```viz
20-routing-load
```

The fix has three parts, and the rest of the chapter builds them in order. Sort the tokens so each expert's rows are contiguous. Launch one kernel that covers all the experts' tiles. Fuse the scaling and, where you can, the gather and scatter into that kernel.

## Sorting tokens by expert

The kernel needs one input shape: an array $X_s$ of rows sorted by expert, and an array of offsets that says where each expert's rows start. Producing them is a counting sort, and you've already built its parts.

The router gives you `ids`, a $T \times k$ matrix of expert ids. Flatten it to a vector of $A = Tk$ *assignments*, where assignment $a = t \cdot k + j$ is token $t$'s $j$-th choice. Then three steps produce the permutation:

1. **Histogram.** Count the assignments per expert: `counts[e]` is the number of rows for expert $e$.
2. **Exclusive scan.** Turn the counts into offsets: `offsets[0] = 0` and `offsets[e + 1] = offsets[e] + counts[e]`. The offsets have $E + 1$ entries, and `offsets[E]` equals $A$.
3. **Placement.** Each assignment goes to row `offsets[e] + rank`, where `rank` is its position among the assignments with the same expert id.

In PyTorch the first two steps are two calls, and the third is a stable sort:

```python
def expert_offsets(expert_ids, num_experts):
    counts = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)   # histogram
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=expert_ids.device)
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)                # scan
    return offsets

order = torch.argsort(flat_ids, stable=True)         # order[p] = assignment placed at row p
x_sorted = x.index_select(0, order // k)             # gather the token rows
gates_sorted = gates.reshape(-1).index_select(0, order)
```

A hand-written CUDA version does the same with the kernels of [chapter 7](/c/07-scan-and-histogram): a privatized histogram, a device-wide exclusive scan, then a scatter kernel in which each thread computes `rank` with an atomic increment on a per-expert cursor. That atomic makes the order inside an expert depend on thread timing, so the result isn't reproducible from run to run. A stable sort keeps assignments in token order inside each expert, which makes the whole layer deterministic. The lab uses the stable sort, and its test checks that two runs agree bit for bit.

### What moving the rows costs

The permutation isn't free, because every row of $X$ is copied once into $X_s$, and every output row is copied back. Take the lab's large case: 4096 tokens, top-2, $D = 2048$, $N = 4096$, bfloat16, so $A = 8192$ rows. The matmul does $2 \times 8192 \times 2048 \times 4096 = 137.4$ GFLOP, which takes at least $137.4 \times 10^{9} / 312 \times 10^{12} = 440\ \mu\text{s}$ at the rated peak.

Use the 1275 GB/s copy ceiling that learn-inference [measured on this card](https://qwen.ifkash.dev/c/10-roofline) as the memory rate, and count every byte read and written:

| Step | Bytes | Floor at 1275 GB/s |
|---|---|---|
| Gather $X_s$: read and write $8192 \times 2048 \times 2$ B each way | 67.1 MB | 52.6 µs |
| Scatter the outputs back: $8192 \times 4096 \times 2$ B each way | 134.2 MB | 105.3 µs |
| Combine: read 67.1 MB of rows, write $4096 \times 4096 \times 2$ B = 33.6 MB | 100.7 MB | 78.9 µs |
| Total of the three | 302 MB | 236.8 µs |

These are floors from the bytes, not measurements. They add up to 54% of the matmul's own floor, so the data movement around a grouped GEMM isn't a rounding error. It is the main reason production kernels fuse it.

Two fusions remove most of it. A kernel can *gather inside the GEMM*: each program loads its A rows through an index array, `x[sorted_token_id]`, instead of from a pre-gathered copy, which deletes the first row of the table. A kernel can also *scatter inside the epilogue*: each program stores its output row at the row's final position in a $T \times k \times N$ buffer, which deletes the second row and leaves only the $k$-way sum. The [vLLM fused MoE kernel](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) works this way, with sorted token indices passed to the GEMM. The lab keeps the explicit gather and scatter so each step is testable on its own, and fusing them is a good extension once the lab passes.

> [!TIP] Count bytes before you fuse
> Before you fuse a copy into a kernel, write down the bytes it moves and divide by 1275 GB/s. If the number is 1% of the matmul's floor, leave it alone. If it's 50%, as the table's total is, fuse it.

## Grouped GEMM: one launch, many problems

A *grouped GEMM* runs $E$ independent matmuls, with different row counts but a common $K$ and $N$, in a single kernel launch. Row tile by row tile, it covers the sorted array $X_s$, and each tile multiplies by the weights of whichever expert owns it.

The core question is the *tile scheduler*: given a program id, which expert and which rows does it compute? A dense GEMM answers with arithmetic, because every tile has the same shape. A grouped GEMM must look at the offsets.

### The launch grid

Cut each expert's rows into tiles of $B_M$ rows. Expert $e$ with $c_e$ rows needs $\lceil c_e / B_M \rceil$ tiles, and the last tile may be short. The launch has to be sized before the kernel runs, and the counts live on the device. Reading them to the host would force a synchronization, the cost the loop paid.

The fix is an upper bound that doesn't need the counts. For $R$ total rows and $E$ experts:

$$
\sum_{e} \left\lceil \frac{c_e}{B_M} \right\rceil \le \boxed{\left\lceil \frac{R}{B_M} \right\rceil + E}
$$

Launch that many programs along the row axis. The programs past the real tile count find no expert and exit at once. The number of wasted programs is at most $E$, which is small next to the real work.

> [!DEEPDIVE] Why the bound holds
> For each expert, $\lceil c_e / B_M \rceil \le (c_e + B_M - 1) / B_M$. Summing over the $E$ experts, and using $\sum_e c_e = R$, gives
>
> $$
> \sum_e \left\lceil \frac{c_e}{B_M} \right\rceil \le \frac{R + E (B_M - 1)}{B_M} = \frac{R}{B_M} + E - \frac{E}{B_M} \le \frac{R}{B_M} + E .
> $$
>
> The left side is an integer, and the right side is at most $\lceil R / B_M \rceil + E$, so the bound holds. It's tight in spirit: each expert can waste up to $B_M - 1$ rows of its last tile, and the $+E$ pays for that.

A worked example makes it concrete. Take counts `[0, 5, 64, 1, 130]` and $B_M = 64$. Expert 0 has no tile, experts 1, 2, and 3 have one each, and expert 4 has three, so there are 6 real tiles. The bound is $\lceil 200 / 64 \rceil + 5 = 9$, so the launch has 9 programs on the row axis and 3 exit at once. Those 6 tiles hold $6 \times 64 = 384$ rows of capacity for 200 real rows, so 47.9% of the tile rows are padding. The lab's `tile_schedule`, `max_m_tiles`, and `tile_waste` compute exactly these numbers.

```viz
20-tile-scheduler
```

### Finding the expert for a program id

Program `pid_m` has to find the expert whose tile range contains it. The lab's kernel walks the offsets with a running total of tiles. For each expert $e$, it computes the tile count from `offsets[e + 1] - offsets[e]`, checks whether `pid_m` falls in the range $[\text{running}, \text{running} + \text{tiles})$, and records the expert and the running total at which its range started.

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)

expert = pid_m * 0 - 1          # int32 tensors, so the loop-carried types agree
first_tile = pid_m * 0
running = pid_m * 0
for e in range(0, E):
    seg_start = tl.load(offsets_ptr + e)
    seg_end = tl.load(offsets_ptr + e + 1)
    tiles = tl.cdiv(seg_end - seg_start, BLOCK_M)
    mine = (pid_m >= running) & (pid_m < running + tiles)
    expert = tl.where(mine, e, expert)
    first_tile = tl.where(mine, running, first_tile)
    running += tiles
valid = expert >= 0
expert = tl.maximum(expert, 0)
```

This is a linear scan over $E$ entries, and every program does it. For $E = 8$ or $64$ it costs 16 to 128 cached scalar loads (two per expert), which is small next to a K loop of dozens of tensor-core steps. For hundreds of experts you'd precompute a table that maps each tile number to its expert, as the vLLM kernel does, or binary search the running totals.

### The tile computation

Once the program knows its expert, the rest is the tiled matmul of [chapter 14](/c/14-triton), with three differences. The rows come from a segment, so the mask checks `offs_m < row_end` where `row_end` is the expert's end offset. The B pointer includes an expert stride, `w_ptr + expert * stride_we`. And a program that found no expert must load nothing and store nothing.

```python
row_begin = tl.load(offsets_ptr + expert) + (pid_m - first_tile) * BLOCK_M
row_end = tl.load(offsets_ptr + expert + 1)
offs_m = row_begin + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
mask_m = (offs_m < row_end) & valid
mask_n = offs_n < N

x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
w_ptrs = (w_ptr + expert.to(tl.int64) * stride_we
          + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

k_iters = tl.where(valid, tl.cdiv(K, BLOCK_K), 0)    # an idle program runs no iterations
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, k_iters):
    k_left = K - k * BLOCK_K
    a = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_left), other=0.0)
    b = tl.load(w_ptrs, mask=(offs_k[:, None] < k_left) & mask_n[None, :], other=0.0)
    acc = tl.dot(a, b, acc)
    x_ptrs += BLOCK_K * stride_xk
    w_ptrs += BLOCK_K * stride_wk
```

Three details carry the correctness. Masking the rows with the expert's end offset stops a tile from reading the next expert's first rows, which would otherwise be silently wrong for experts whose count isn't a multiple of $B_M$. Masking the K axis with `K - k * BLOCK_K` makes a $D$ that `BLOCK_K` doesn't divide correct. And `other=0.0` is the identity for a dot product, so masked lanes add nothing.

The `expert.to(tl.int64)` cast matters for large models. An index like `expert * stride_we` can pass $2^{31}$ for a stack of big experts, and a 32-bit multiply would wrap around to a wrong address.

### Padded or exact

The kernel above is an *exact* grouped GEMM: tiles are cut inside each expert's range, the last tile is masked, and nothing is copied or padded in memory. The alternative is the *padded* layout that some production kernels use. They pad each expert's segment to a multiple of $B_M$ rows during the sort, fill the gap with a sentinel index, and then every tile is full and belongs to exactly one expert. The scheduler becomes a lookup: a table `expert_of_tile[tile]` written by the sort step.

| | Exact (the lab) | Padded (a lookup table) |
|---|---|---|
| Extra memory | none | up to $E (B_M - 1)$ rows of padding |
| Masking | a row mask on every load and store | a mask only on the sentinel rows |
| Scheduler | a loop or search over offsets | one load from a table |
| Sort step | histogram and scan only | histogram, scan, and a table of tile owners |

Neither wins everywhere. The padding costs the same wasted tile rows in both: the exact kernel computes the masked rows and throws the results away, and the padded kernel computes the sentinel rows. What changes is the bookkeeping. The lab's `tile_waste` function measures the waste for a given split, and the figure lets you move $B_M$ and the skew to see it. A smaller $B_M$ wastes fewer rows, but a smaller tile reuses each loaded weight tile for fewer rows, so each tile does less work per byte it loads. The kernel's tile height is a tuning knob, and the lab's tests time two choices.

## Fusing the gate weights

The last step of the layer multiplies each expert output by its gate weight and sums the $k$ results. The multiplication can happen anywhere before the sum, and the cheapest place is the grouped GEMM's *epilogue*, the code after the K loop.

In the epilogue, `acc` still holds the float32 tile. The program loads one gate weight per row, scales the rows, and only then rounds to bfloat16 and stores:

```python
if HAS_GATE:
    gate = tl.load(gate_ptr + offs_m, mask=mask_m, other=0.0)    # gates in sorted order
    acc = acc * gate[:, None]
out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
```

This costs one extra load of $B_M$ floats per tile, against a K loop that reads thousands. Without the fusion the gate multiply is a separate elementwise kernel that reads and writes all $A \times N$ output elements, which is another 134.2 MB, or 105 µs of floor on the lab's large case. Scaling in float32 before the rounding also gives a slightly smaller error than rounding first and then scaling.

In a full gated MLP the same epilogue slot has two more uses. The gate and up projections' epilogue can apply the activation and multiply the two halves, so the intermediate tensor is written once. The down projection's epilogue applies the gate weight. [Chapter 17](/c/17-fusion-and-compilers) covers why this kind of epilogue fusion pays.

### Unpermute and combine

After the GEMM, `out_sorted` is in expert order. Assignment `order[p]` produced row $p$, so the inverse permutation restores token order:

```python
out_flat = torch.empty_like(out_sorted)
out_flat[order] = out_sorted                              # scatter: undo the sort
y = out_flat.view(T, k, -1).float().sum(dim=1).to(out_sorted.dtype)   # combine the k rows
```

The scatter writes each row exactly once, because `order` is a permutation, so the result is deterministic. A faster-looking alternative is `index_add_` on the token index, which skips the intermediate buffer, but on a GPU it uses atomic adds, and the order of float additions then varies from run to run. A sum over the $k$ axis of a contiguous $T \times k \times N$ tensor has no such problem.

> [!WARNING] Atomic combine breaks reproducibility
> Summing the $k$ expert outputs with `index_add_` or `atomicAdd` gives results that differ in the last bits between runs, because floating-point addition isn't associative ([chapter 9](/c/09-floating-point-on-the-gpu)). For training runs you need to reproduce, store each row at its final position and sum in a fixed order.

## MegaBlocks and the block-sparse view

A grouped GEMM handles any split of rows, but the first systems for MoE training didn't use it. They fixed a *capacity* per expert, the most tokens an expert will take, and ran a batched matmul over $E$ equal-size buffers. A token routed to a full expert is *dropped*, and a buffer that isn't full is padded with zeros. [Switch Transformers](https://arxiv.org/abs/2101.03961) tuned the capacity factor to trade dropped tokens against wasted compute.

[MegaBlocks](https://arxiv.org/abs/2211.15841) removes both problems with a reformulation. Stack all the experts' weight matrices side by side, and the layer becomes one large matmul in which each token row is non-zero only in the column blocks of its own expert. That matrix has a *block-sparse* structure: it is built from fixed-size blocks, and the blocks that hold zeros are never computed. The paper's custom GPU kernels run the block-sparse products, and no token is dropped, because the sparse layout can describe any split. The abstract reports end-to-end training speedups of up to 40% over Tutel and 2.4x over Megatron-LM, without naming the hardware, so treat the ratios as the paper's and not as a prediction for the lab card.

The two formulations compute the same thing. A grouped GEMM is the block-sparse product with the sparsity pattern stored as row ranges, and its tile scheduler is the block-sparse kernel's iteration over non-zero blocks. The block-sparse view is useful because it shows why dropless routing is possible: padding is a property of the tile size, not of the layer.

## Expert parallelism and the all-to-all

When the experts don't fit on one GPU, you place different experts on different GPUs. This is *expert parallelism*. A token's chosen experts may live on other ranks, so the layer needs two exchanges: a *dispatch* that sends each row to the rank that owns its expert, and a *combine* that sends the output rows back. Each exchange is an *all-to-all*, in which every rank sends a different slice to every other rank. [Chapter 21](/c/21-collective-communication) builds the collectives, and [chapter 22](/c/22-overlap-and-parallelism) shows how to overlap them with compute.

The permutation of this chapter is the local half of dispatch. Sorting by expert id also sorts by destination rank, because experts are assigned to ranks in contiguous groups. The counts from the histogram become the send sizes of the all-to-all.

Estimate the cost with the bandwidth term only. Take an 8-GPU A100 SXM node, 4096 tokens per rank, top-2, $d = 4096$, bfloat16. Each rank holds $8192 \times 4096 \times 2 = 67.1$ MB of rows to send, and $7/8$ of them leave the rank, which is 58.7 MB. NVLink 3 is rated at 600 GB/s counting both directions, which is 300 GB/s one way. The floor is $58.7 \times 10^{6} / 300 \times 10^{9} = 196\ \mu\text{s}$ per exchange, and 391 µs for dispatch and combine together.

Compare that with the experts' compute. A Mixtral-sized expert costs 352 MFLOP per row, so 8192 rows per rank cost $2.89 \times 10^{12}$ FLOP, which is 9.25 ms at 312 TFLOP/s. Communication is 4% of the compute floor, before any overlap.

The general form of that comparison is the *FLOP per communicated byte*. Per row, the layer moves $2d$ bytes out and $2d$ back, and does $6df$ FLOP, so the ratio is $6df / 4d = 1.5 f$ FLOP per byte. The node's balance is $312 \times 10^{12} / 300 \times 10^{9} = 1040$ FLOP per byte. Communication dominates only when $1.5 f$ falls below about 1040, which is an expert width $f$ below roughly 700. Fine-grained experts with $f = 2048$ give 3072, still above the balance, which is why large MoE systems can hide most of the all-to-all with careful overlap. The estimate ignores load imbalance across ranks, which makes the busiest rank the bottleneck, so treat it as a best case.

## Unstructured sparsity rarely pays on a GPU

Pruning removes weights, and a pruned matrix has fewer FLOP to do. The GPU, however, rarely gets faster, for two reasons that the specs make plain.

The first is the tensor cores. A dense bfloat16 matmul runs on tensor cores at 312 TFLOP/s. An irregular sparse matmul can't use them, because a tensor core multiplies a dense tile, so it runs on the CUDA cores at 19.5 TFLOP/s for float32. The ratio is $312 / 19.5 = 16$. To break even on arithmetic alone, a sparse kernel must skip at least $15/16$ of the work, which is 93.75% sparsity, before it spends anything on indexing.

The second is memory. A sparse format such as compressed sparse row stores an index beside each value and reads the dense operand through those indices, which makes the loads scattered and uncoalesced ([chapter 3](/c/03-memory-and-bandwidth)). The kernel's intensity drops, and the bandwidth it achieves drops with it.

So a matrix pruned to 50% or 80% at random positions runs no faster on the GPU than the dense matmul, and often slower. The accuracy-preserving sparsity levels of trained networks sit well below 94%. The way out is to make the pattern regular enough that the hardware can use it.

## 2:4 structured sparsity

NVIDIA's Ampere tensor cores support one regular pattern in hardware. In every group of 4 consecutive values along the reduction dimension, at most 2 are non-zero. That is *2:4 structured sparsity*, and it fixes the sparsity at 50% ([Mishra et al., 2021](https://arxiv.org/abs/2104.08378)).

The format stores only the kept values, and a small piece of *metadata* that says where they were. For each kept value, 2 bits give its position, 0 to 3, inside the group of 4. A 16-bit group of 4 values takes $4 \times 16 = 64$ bits dense. Compressed, it takes two values and two indices, $2 \times 16 + 2 \times 2 = 36$ bits, a ratio of $36/64 = 0.5625$. The metadata overhead is $4/32 = 12.5\%$ of the kept values for 16-bit data.

| Property | 2:4 on the A100 |
|---|---|
| Pattern | at most 2 non-zero values in every 4 along the reduction dimension |
| Formats | fp16, bf16, and int8 at 2:4. TF32 uses a 1:2 pattern. |
| Storage | 36 bits per 4 fp16 values, 0.5625 of dense |
| PTX instruction | `mma.sp`, for example m16n8k16 for fp16 |
| Rated peak | 624 TFLOP/s for fp16 and bf16, twice the dense 312 |
| Measured instruction rate | 622.8 TFLOP/s sparse against 310.6 dense, for m16n8k16 fp16 on an A100 ([Luo et al.](https://arxiv.org/abs/2402.13499)) |

The sparse tensor core reads the compressed values and metadata, picks the matching entries of the other operand with the metadata, and does half the multiplications in the same time. Because the instruction itself runs at twice the rate, a perfect kernel would approach 2x. That's the claim NVIDIA makes: "twice the math throughput of dense matrix units."

### What speedups to expect

A whole GEMM doesn't reach the instruction's 2x, because the rest of the kernel doesn't speed up: loading the dense operand, the epilogue, and the launch. Published results on the A100 show the gap:

- A ViT-L-sized layer with $M \times N \times K = 13008 \times 4096 \times 1024$ takes 538 µs dense and 387 µs with the 2:4 kernel in the [PyTorch team's report](https://pytorch.org/blog/accelerating-neural-network-training/), a speedup of 1.39x. The layer does $2 \times 13008 \times 4096 \times 1024 = 109.1$ GFLOP, so the dense run sustains 203 TFLOP/s and the sparse run is equivalent to 282 TFLOP/s of dense work, 45% of the 624 TFLOP/s sparse peak.
- The same report quotes up to 1.6x for a matmul with cuSPARSELt, and 1.3x for the forward and backward passes of ViT-L's MLP layers.
- End to end, it reports 6% for DINOv2 ViT-L and 10% for Segment Anything inference, because the matmuls are only part of the model.
- The [PyTorch tutorial](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html) reports that BERT-base on an A100 is slower with 2:4 sparsity at batch 4 (about 0.7x in its uncompiled table) and about 1.2x faster at batch 16. At small batches the sparse kernel is slower than the dense one.

The compression step is a cost to pay once. The report measured 380 µs with cuSPARSELt and 36 µs with PyTorch's own kernel, for the same ViT-L shape. For inference weights, you compress once at load time. For training, where the weights change every step, you re-prune and re-compress each step, so the faster kernel matters.

### Using it from PyTorch

PyTorch exposes the format through `torch.sparse.to_sparse_semi_structured`. The weight must be a 2D CUDA tensor in fp16, bf16, or int8, and its sizes must be multiples of 64 for 16-bit data, per the [API documentation](https://docs.pytorch.org/docs/2.14/sparse.html). The feature is a prototype, so check it on your PyTorch and GPU build.

```python
from torch.sparse import to_sparse_semi_structured

w_pruned = prune_2_4(w)                         # keep the 2 largest |values| of every 4
w_sparse = to_sparse_semi_structured(w_pruned)  # compress: values plus metadata
y = torch.nn.functional.linear(x, w_sparse)     # dispatches to a sparse tensor-core GEMM
```

Pruning a trained dense model to 2:4 loses accuracy unless you fine-tune. The recipe in Mishra et al. is to train dense, prune to 2:4 by magnitude, and retrain with the same hyperparameters while the pattern is held fixed. The lab's `prune_2_4` does the magnitude step, and the lab times the compressed matmul so that you see the speedup, or its absence, on your own card.

> [!NOTE] Sparsity and MoE are different axes
> MoE is *activation* sparsity chosen per token: which experts run. 2:4 is *weight* sparsity chosen once: which entries exist. They compose in principle, since an expert's weights can be 2:4 and run on sparse tensor cores, but a grouped GEMM with sparse operands is a separate kernel that this lab doesn't build.

## What goes wrong

- **A loop-carried type error in the scheduler.** Writing `expert = -1` before the loop makes a Python constant, and Triton rejects a loop that reassigns it with a tensor. Start from `pid_m * 0 - 1`, an int32 tensor.
- **Reading past an expert's rows.** A mask on `offs_m < total_rows` instead of `offs_m < row_end` lets a short tile read and write the next expert's rows, and the output is wrong only for counts that aren't multiples of $B_M$. Test with counts like 1, 5, and 130.
- **A grid sized from the counts.** Calling `.item()` or `.tolist()` on the counts to size the grid puts back the host synchronization you removed. Use the bound $\lceil R / B_M \rceil + E$.
- **Early `return` for idle programs.** An idle program that still runs the K loop wastes tensor-core time, and a data-dependent `return` is easy to get wrong. Fold validity into the masks and the loop's trip count instead.
- **A 32-bit offset overflow.** `expert * stride_we` can exceed $2^{31}$ with large experts. Cast the expert id to int64 before the multiply.
- **An empty expert or a one-token batch.** A zero count gives zero tiles, and `bincount` with `minlength` still returns $E$ entries. Test a batch where most experts are empty.
- **An atomic combine.** `index_add_` makes the sum order vary, so runs differ in the last bits. Use the fixed-order sum over the $k$ axis.
- **Timing the loop unfairly, in either direction.** The naive loop's host synchronizations dominate it, so a comparison against it overstates the gain from the GEMM. The lab also times a loop over the sorted segments, which isolates the GEMM, and reports both.
- **A 2:4 matrix with the wrong shape.** Sizes that aren't multiples of 64, a float32 weight, or a CPU tensor make `to_sparse_semi_structured` raise. The lab catches the error and reports 0.0, but your own code must handle it.

> [!RECAP]
> - An expert's matmul has an intensity of about $n$ FLOP/byte for $n$ tokens, so it needs about 161 tokens to be compute bound on an A100. Few tokens per expert means memory bound, and tiny grids mean idle SMs.
> - Sort the assignments by expert with a histogram, an exclusive scan, and a stable sort. Each expert then owns a contiguous range of rows given by `offsets`.
> - A grouped GEMM maps a program id to an (expert, tile) pair, with a grid of $\lceil R / B_M \rceil + E$ row programs. Masks on the expert's end offset make ragged tiles exact, and idle programs run no iterations.
> - Fuse the gate weights into the epilogue. What remains, the gather, the scatter, and the combine, is about 237 µs of byte floor on the lab's large case against 440 µs of matmul floor.
> - Expert parallelism adds two all-to-alls. Their cost against compute is set by $1.5 f$ FLOP per communicated byte against the machine's balance.
> - Unstructured sparsity needs about 94% sparsity to break even against the tensor cores. 2:4 sparsity runs on sparse tensor cores rated at 2x, and published A100 speedups are 1.3x to 1.6x per matmul and far less end to end.

## Check your understanding

> [!QUESTION] An A100 expert has bfloat16 weights and receives 40 tokens. Is its matmul compute bound or memory bound, and what fraction of the 312 TFLOP/s can it reach?
> The intensity is about $n = 40$ FLOP per byte, below the ridge point of 161, so it's memory bound. The roofline caps it at $40 / 161 = 25\%$ of the peak, about 77 TFLOP/s, and the real kernel reaches less than that ceiling.

> [!QUESTION] With 8 experts, top-2 routing, and balanced load, how many tokens must a batch contain for each expert to see 161 tokens? Redo it for 256 experts and top-8.
> Each expert gets $kB/E$ tokens. For $E = 8$ and $k = 2$ that's $B/4$, so $B = 644$. For $E = 256$ and $k = 8$ it's $B/32$, so $B = 5152$. Finer experts need much larger batches, which is why decode is memory bound.

> [!QUESTION] Why does the grouped GEMM launch $\lceil R / B_M \rceil + E$ programs along the row axis, and what happens to the extra ones?
> The kernel's grid must be fixed before the per-expert counts are known on the host. Each expert can waste up to $B_M - 1$ rows in its last tile, and the $+E$ covers that worst case. Programs past the real tile count find no expert, run zero K iterations, store nothing, and exit.

> [!QUESTION] For counts `[0, 5, 64, 1, 130]` and $B_M = 128$, how many tiles are launched and what fraction of tile rows is padding?
> The tiles are 0, 1, 1, 1, and $\lceil 130/128 \rceil = 2$, so 5 tiles with $5 \times 128 = 640$ rows of capacity for 200 real rows. The padding is $1 - 200/640 = 68.75\%$, worse than the 47.9% at $B_M = 64$, because bigger tiles waste more on small experts.

> [!QUESTION] Why does the lab sort with a stable sort instead of an atomic counter per expert?
> An atomic cursor assigns row positions in the order threads happen to run, so the order inside an expert changes between runs, and so can the float sum. A stable sort keeps token order inside each expert, which makes the layer deterministic. The cost is a sort instead of a histogram-and-scatter, a small price next to the matmul.

> [!QUESTION] A 50%-sparse weight matrix pruned at random positions runs on the CUDA cores in a sparse kernel. How sparse must it be to match the dense bfloat16 tensor-core matmul on arithmetic alone?
> The tensor cores run at 312 TFLOP/s against 19.5 TFLOP/s for float32 CUDA cores, a ratio of 16. The sparse kernel must skip $15/16$ of the work, so about 93.75% sparsity, before the cost of indices and scattered loads. A 50% matrix is far from it.

> [!QUESTION] The 2:4 instruction runs at 622.8 TFLOP/s against 310.6 dense, yet PyTorch measures 1.39x on a ViT-L layer. Where does the rest go?
> The instruction rate is 2x, but the kernel still loads the dense operand at the same bandwidth, runs the same epilogue, and pays the same launch and tail effects. The layer's dense run sustains 203 TFLOP/s and the sparse run is equivalent to 282 TFLOP/s, 45% of the sparse peak. Only the tensor-core part of the time shrinks.

## Lab

> [!TRY] Grouped GEMM for mixture-of-experts
> Fill in `route_topk`, `expert_offsets`, `sort_by_expert`, `tile_schedule`, `max_m_tiles`, `tile_waste`, the Triton `grouped_gemm_kernel`, and `prune_2_4`. The given `moe_forward` chains them: route with top-2, sort with a histogram and a scan, run the grouped GEMM with the gates fused, and unpermute and combine.
>
> You pass when the CPU-only parts match their specifications, and the layer matches a float32 per-expert reference in fp16 and bf16 on awkward shapes: one token, a $D$ of 33, six empty experts out of eight, and ragged tiles. The grouped kernel must reach at least 20% of the 312 TFLOP/s peak on the large case, and at least half the speed of a loop of cuBLAS matmuls over the sorted segments in each case. The roofline predicts the large case at 440 µs or more, and your lab measures the real figure.
>
> The lab reports `grouped_tflops_big`, `loop_tflops_big`, `speedup_vs_loop_big`, `grouped_tflops_many`, `speedup_vs_loop_many`, `speedup_vs_loop_skewed`, `max_over_mean_load_skewed`, `moe_forward_ms`, `naive_moe_ms`, and `speedup_end_to_end`. The optional 2:4 part reports `sparse24_dense_ms`, `sparse24_sparse_ms`, and `sparse24_speedup`, and never fails the lab. If the speedup reads 0.0, the log says why. Allow a minute for the first Triton compile.

## Further reading

- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841): the block-sparse formulation and why token dropping is unnecessary.
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538): the sparsely gated mixture-of-experts layer and top-k gating.
- [Switch Transformers](https://arxiv.org/abs/2101.03961): the capacity factor and the load-balancing loss.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088): the open top-2 architecture whose shapes this chapter uses.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437): fine-grained experts and expert-parallel communication.
- [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): the 2:4 format, its metadata, and the training recipe.
- [Accelerating neural network training with 2:4 sparsity](https://pytorch.org/blog/accelerating-neural-network-training/): PyTorch's API and the measured A100 speedups.
- [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html): a persistent-kernel scheduler for independent GEMMs.
- [cuSPARSELt](https://docs.nvidia.com/cuda/cusparselt/): the library behind sparse tensor-core GEMMs.
- [vLLM fused MoE kernels](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe): production Triton MoE kernels with token permutation.
- [Fusion in Triton](https://qwen.ifkash.dev/c/13-fusion-in-triton) in the companion course: the epilogue fusion pattern this chapter reuses.
