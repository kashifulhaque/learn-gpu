---
title: Collective communication
slug: 21-collective-communication
part: "Part 5 — Many GPUs"
summary: The six collectives that data, tensor, and expert parallelism are built from, the alpha-beta cost model, ring all-reduce derived and bounded, trees for latency, how NCCL chooses, and how NVLink, PCIe, and InfiniBand set the ceiling.
minutes: 100
gpu: true
objectives:
  - "Name the six collectives (broadcast, reduce, all-reduce, reduce-scatter, all-gather, all-to-all), say which parallelism strategy uses each, and prove that all-reduce is a reduce-scatter followed by an all-gather."
  - "Write the alpha-beta cost model, $T = \\alpha + n/B$, and compute the half-bandwidth message size $\\alpha B$ for a link."
  - "Derive the ring all-reduce time $2(p-1)\\alpha + 2\\frac{p-1}{p}\\frac{n}{B}$, and explain why each rank must send at least $2\\frac{p-1}{p}n$ bytes."
  - "Compute where a tree beats a ring, and explain what NCCL's double binary tree, its LL, LL128, and Simple protocols, and its automatic algorithm choice are for."
  - "Convert nccl-tests' algbw to busbw, and use busbw to compare a measurement with a link's per-direction ceiling."
  - "Use per-direction bandwidth for NVLink and PCIe, and say why a datasheet's bidirectional total gives a time that is half the truth."
  - "Implement ring all-reduce from isend and irecv across processes, and check it against dist.all_reduce."
lab: 21-ring-allreduce
---

# Collective communication

> [!TLDR]
> - Every way of splitting a model across GPUs ends in a *collective*: an operation that all ranks call together. Data parallelism all-reduces gradients, tensor parallelism all-reduces activations, FSDP all-gathers weights and reduce-scatters gradients, and expert parallelism uses all-to-all.
> - The cost of one message is $T = \alpha + n/B$: a fixed latency $\alpha$ plus size over per-direction bandwidth $B$. A ring all-reduce over $p$ ranks takes $2(p-1)\alpha + 2\frac{p-1}{p}\frac{n}{B}$, and the bandwidth term is optimal: no algorithm sends fewer bytes per rank.
> - Rings are bandwidth-optimal and slow for small messages, because latency grows with $p$. Trees have logarithmic latency. NCCL picks among them, and among three protocols, by message size and topology.
> - nccl-tests reports *busbw*, the measured rate times $2(p-1)/p$ for an all-reduce, so a number is comparable with the link's rate. On an 8-GPU A100 node the ceiling is 300 GB/s per direction, not the 600 GB/s NVIDIA prints, because that is the sum of both directions.
> - The lab implements the ring with `isend` and `irecv` across processes, checks it against `dist.all_reduce`, and codes the cost model.

A single GPU runs out of two things: memory and time. A 7-billion-parameter model trained with Adam needs more than one 80 GB card for its weights, gradients, and optimizer state. A model served to many users needs more memory bandwidth than one card has. The answer is more GPUs, and the moment you have two, the question changes from "how fast is the kernel?" to "how long does the data take to get from one card to the other?"

That time is often the whole story. On the lab A100, a matrix multiplication can run at a large fraction of 312 TFLOP/s, but a gradient that has to cross a PCIe link moves at 31.5 GB/s per direction. This chapter gives you the vocabulary for those transfers, a cost model that predicts them within a small factor, and the algorithm, the ring, that nearly every framework uses for the biggest one.

You write the ring yourself in the lab, from point-to-point sends and receives. The lab runs on CPU tensors with the gloo backend, because the lab machine has one GPU and the ring's logic doesn't depend on what the tensors are made of. The cost model, which is where the hardware numbers come in, you check against the arithmetic in this chapter.

## Before you start

**Rank.** A *rank* is one process in a group, numbered 0 to $p-1$, usually one per GPU. A *collective* is an operation every rank in the group calls, such as "sum your tensor with everyone else's." A *point-to-point* operation is a `send` from one rank and a matching `recv` on another.

**Latency and bandwidth.** *Latency* is the time before the first byte arrives. *Bandwidth* is the bytes per second after that. A transfer's time is both, as the model below makes precise.

**Per direction, not total.** A link such as NVLink or PCIe carries data both ways at once. NVIDIA quotes NVLink and PCIe figures as the sum of both directions, so "NVLink 600 GB/s" on an A100 means 300 GB/s each way. Every formula in this chapter needs the rate at which *one rank sends*, which is the per-direction figure. When the chapter says $B$, it means per direction.

**Decimal units.** Bandwidth is decimal, as in the earlier chapters: 300 GB/s is $3 \times 10^{11}$ bytes per second. A message size written as 1 GiB is $2^{30} = 1{,}073{,}741{,}824$ bytes.

**Where this fits.** The earlier chapters taught what limits a kernel on one GPU: memory bandwidth against arithmetic. This chapter does the same for data moving between GPUs, where the "memory" is a link. [Chapter 22](/c/22-overlap-and-parallelism) shows how to hide that movement behind computation. The companion course covers [tensor parallelism](https://qwen.ifkash.dev/c/19-tensor-parallelism) from the model's side. Here we look at the collective underneath.

## Why every parallelism strategy needs a collective

Splitting a model across GPUs always leaves some tensor that every rank needs in full, or a sum that every rank holds a part of, so a collective follows. This section answers one question: which collective does each strategy call, and on how many bytes?

Take four strategies, each seen from the kernel's side.

- **Data parallelism.** Each rank holds a full copy of the model and trains on a different slice of the batch. After the backward pass, each rank has gradients computed from its own slice, and the optimizer needs the average. That is an *all-reduce* on the gradients: every rank contributes a tensor, and every rank receives the sum.
- **Tensor parallelism.** Each rank holds a slice of every weight matrix. After a row-parallel matrix multiplication, each rank has a partial sum of the layer's output, and the next layer needs the complete sum. That is an all-reduce on the activations, twice per transformer layer in the Megatron-LM scheme.
- **Fully sharded data parallelism (FSDP, and ZeRO stage 3).** Each rank stores only a shard of the weights. Before a layer runs, the ranks *all-gather* its weights, and after the backward pass, they *reduce-scatter* its gradients so each rank keeps the sum for its own shard.
- **Expert parallelism.** In a mixture-of-experts layer ([chapter 20](/c/20-moe-and-sparse-kernels)), each rank holds some experts, and each token must travel to the rank holding the expert the router chose. Sending a different piece of data from every rank to every other rank is an *all-to-all*.

Put numbers on the first one. A 7-billion-parameter model has $7 \times 10^{9} \times 2 = 14 \times 10^{9}$ bytes of gradients in bfloat16. We'll derive the cost formula below, and for now take its result: an all-reduce over 8 ranks sends $2 \cdot \frac{7}{8} = 1.75$ times the data size per rank. That is $1.75 \times 14 \times 10^{9} = 24.5 \times 10^{9}$ bytes on every rank's outgoing link.

> [!EXAMPLE] Gradients of a 7B model over two links
> At 300 GB/s per direction, the A100 NVLink rate, 24.5 GB takes $24.5 \times 10^{9} / 300 \times 10^{9} = 81.7$ ms. At 31.5 GB/s, the PCIe Gen4 x16 rate, it takes $24.5 \times 10^{9} / 31.5 \times 10^{9} = 778$ ms. The same gradients, a factor of 9.5 apart, set by nothing but the link.
>
> Both figures assume the all-reduce runs after the backward pass finishes and that nothing overlaps. Chapter 22 shows how to start the all-reduce on early layers' gradients while later layers are still computing.

So a collective's cost depends on two things you can read off the machine: how many bytes each rank must send, and how fast one rank can send them. The rest of the chapter is about getting both numbers right.

## The six collectives

A small set of collectives covers nearly every pattern, and they relate to each other by identities you can use to reason about cost. The section answers one question: what does each collective do, and how many bytes does it move?

Let there be $p$ ranks and an array of $n$ bytes. The table lists what each rank holds before and after. "Per-rank traffic" is the bytes one rank sends in a bandwidth-optimal algorithm, written as a multiple of $n$ with $n$ the full array's size.

| Collective | Before | After | Per-rank traffic | Used by |
|---|---|---|---|---|
| Broadcast | Rank 0 has the array | Every rank has it | about $n$ | Loading weights, sharing a seed |
| Reduce | Every rank has an array | Rank 0 has their sum | about $n$ | Gathering a metric on one rank |
| All-reduce | Every rank has an array | Every rank has their sum | $2\frac{p-1}{p}n$ | Data parallel gradients, tensor parallel activations |
| Reduce-scatter | Every rank has an array | Rank $r$ has chunk $r$ of the sum | $\frac{p-1}{p}n$ | FSDP gradients |
| All-gather | Rank $r$ has chunk $r$ | Every rank has all chunks | $\frac{p-1}{p}n$ | FSDP weights |
| All-to-all | Rank $r$ has $p$ chunks, one for each rank | Rank $r$ has chunk $r$ from every rank | $\frac{p-1}{p}n$ | Expert parallel token routing |

The traffic column follows from counting. In an all-gather, each rank starts with $n/p$ bytes and must end with $n$, so it must *receive* $n - n/p = \frac{p-1}{p}n$ bytes. In an all-to-all, each rank keeps one of its $p$ chunks and sends the other $p-1$, each $n/p$ bytes, so it sends $\frac{p-1}{p}n$. The nccl-tests documentation uses the same factors, which we use again in the section on bus bandwidth.

### The identity: all-reduce is a reduce-scatter plus an all-gather

The two halves of an all-reduce are themselves collectives. A reduce-scatter leaves rank $r$ with the fully summed chunk $r$, and an all-gather then copies every rank's chunk to everyone. Together they produce the sum on every rank, which is what an all-reduce does.

$$
\text{all-reduce} = \text{all-gather} \circ \text{reduce-scatter}
$$

The identity does real work in three places:

- **It gives the all-reduce's cost.** Each half sends $\frac{p-1}{p}n$ bytes per rank, so the whole sends $2\frac{p-1}{p}n$, the factor in the table.
- **It is why FSDP costs what data parallelism costs.** FSDP reduce-scatters the gradients and all-gathers the weights. Those are the two halves of the all-reduce that data parallelism runs, so FSDP moves comparable gradient bytes while storing only a shard per rank. The extra weight all-gathers before the forward and backward passes are what it pays for the memory it saves.
- **It is the plan for the ring.** The ring algorithm below runs the two halves back to back.

In PyTorch the six collectives are `dist.broadcast`, `dist.reduce`, `dist.all_reduce`, `dist.reduce_scatter_tensor`, `dist.all_gather_into_tensor`, and `dist.all_to_all_single`. On GPU tensors with the NCCL backend, each launches a kernel on a CUDA stream, so it is asynchronous in the same sense as every kernel in [chapter 0](/c/00-what-you-are-building).

```python
import torch
import torch.distributed as dist

# Run one process per GPU, for example with torchrun.
dist.init_process_group("nccl")
rank = dist.get_rank()
x = torch.full((4,), float(rank), device="cuda")

dist.all_reduce(x)        # every rank now holds the sum over ranks
# For 4 ranks: x == [6, 6, 6, 6], because 0 + 1 + 2 + 3 = 6.
```

> [!NOTE] Send and receive are not collectives
> `dist.send` and `dist.recv` move data between two ranks and involve nobody else. The lab builds a collective out of them. In production, you rarely write that yourself, because NCCL's own all-reduce is faster, but pipeline parallelism passes activations between stages with exactly this pair.

## The alpha-beta model

A model of one message, with two parameters, predicts most of what you will see. The section answers one question: how long does it take to send $n$ bytes from one rank to another?

Send $n$ bytes over a link. The first byte arrives after a fixed delay $\hla{\alpha}$, which covers the software, the wire, and the hardware handshake. The remaining bytes arrive at the link's bandwidth $\hlb{B}$, in bytes per second and per direction. This is Hockney's model, and Thakur, Rabenseifner, and Gropp build their MPICH collective cost analysis on it.

$$
\boxed{T(n) = \hla{\alpha} + \frac{n}{\hlb{B}}}
$$

The $\hla{\alpha}$ term doesn't depend on size, and the $\frac{n}{\hlb{B}}$ term does. For small messages the first term dominates, and for large ones the second does. The two are equal at the *half-bandwidth message size*, where the transfer reaches half the link's peak rate:

$$
n_{1/2} = \alpha B
$$

For a message of size $n$, the fraction of the peak bandwidth you achieve is $n / (n + \alpha B)$, which is 50% at $n = n_{1/2}$ and approaches 100% as $n$ grows.

> [!EXAMPLE] Half-bandwidth sizes for four links
> Take $\alpha = 5\ \mu\text{s}$. This is an assumption: docs/FACTS.md marks it unconfirmed, and the companion course uses 5 to 10 µs for a small all-reduce. The lab fits the real value for its processes. With the per-direction rates from FACTS:
>
> | Link | $B$ per direction | $n_{1/2} = \alpha B$ |
> |---|---|---|
> | PCIe Gen4 x16 | 31.5 GB/s | $5 \times 10^{-6} \times 31.5 \times 10^{9} = 157.5$ KB |
> | InfiniBand NDR port | 50 GB/s | 250 KB |
> | NVLink 3 (A100) | 300 GB/s | 1.5 MB |
> | NVLink 4 (H100) | 450 GB/s | 2.25 MB |
>
> A faster link needs a larger message to reach the same fraction of its peak. That is why a latency that is unimportant on PCIe can dominate on NVLink.

Latency has a hardware floor and a software component. NCCL's three protocols, which come up below, are different trades between them. In the Demystifying NCCL paper's Table I, the per-hop latency is about 6 µs for the Simple protocol, 1 µs for LL, and 2 µs for LL128. The paper's table doesn't name the hardware, so treat those figures as the right order of magnitude and not as numbers for one machine. They bracket the 5 µs we assume.

### Where the model stops being true

The model assumes one message at a time on an otherwise idle link. Three things break that, and each is worth knowing when a measurement disagrees with the prediction:

- **Contention.** Two collectives sharing a link split its bandwidth. So do two ranks sending through one switch port.
- **Latency isn't constant.** $\alpha$ depends on the protocol, the transport (NVLink, PCIe, or the network), and whether the data crosses a node boundary.
- **Bandwidth isn't constant either.** A link reaches its peak only for messages well above $n_{1/2}$, and no link delivers its full datasheet rate.

The model is a floor and a shape, not a forecast. It tells you what the cost can't be below, and how it scales with size and with $p$.

## Ring all-reduce

The ring is the algorithm that makes an all-reduce's bandwidth term optimal, and it is simple enough to write in 20 lines. This section answers one question: how do $p$ ranks sum an array so that each sends only about $2n$ bytes in total?

Arrange the ranks in a ring, so rank $r$ sends to $(r+1) \bmod p$ and receives from $(r-1) \bmod p$. Split each rank's array into $p$ chunks of $n/p$ bytes. The algorithm has two phases, both $p-1$ steps long.

**Reduce-scatter.** In step $s$, rank $r$ sends chunk $(r-s) \bmod p$ to its successor and receives chunk $(r-s-1) \bmod p$ from its predecessor, then adds the received chunk into its own copy. The chunk a rank sends is the one it just finished accumulating, so partial sums travel around the ring and grow by one rank's contribution per hop. After $p-1$ steps, every chunk has passed through every rank, and rank $r$ holds the fully reduced chunk $(r+1) \bmod p$.

**All-gather.** Now rank $r$ owns one finished chunk. In step $s$, it sends chunk $(r+1-s) \bmod p$ and receives chunk $(r-s) \bmod p$, and this time it overwrites its copy instead of adding. After $p-1$ steps, every rank holds every finished chunk.

Step through it with the figure. Pick the number of ranks, press **Play** or step with the arrows, and watch each cell count how many ranks' contributions it holds.

```viz
21-ring-steps
```

Every cell in the figure starts at 1, the rank's own data. A cell reaches "all" when it holds the sum of all $p$ ranks' contributions. After the reduce-scatter's last step, each rank has exactly one such cell, on a diagonal. After the all-gather's last step, every cell is complete.

### The cost, step by step

Count what one rank does. Each of the $2(p-1)$ steps sends one chunk of $n/p$ bytes. Each step costs one message by the alpha-beta model, so the time is the number of steps times the cost of a chunk-sized message:

$$
T_{\text{ring}} = 2(p-1)\left(\hla{\alpha} + \frac{n/p}{\hlb{B}}\right) = \boxed{2(p-1)\,\hla{\alpha} + 2\,\frac{p-1}{p}\,\frac{n}{\hlb{B}}}
$$

Read the two terms separately. The latency term, $2(p-1)\hla{\alpha}$, grows with the number of ranks, since each rank takes part in $2(p-1)$ steps. The bandwidth term, $2\frac{p-1}{p}\frac{n}{\hlb{B}}$, grows toward $2n/B$ as $p$ grows. Its factor $2(p-1)/p$ is 1.000, 1.500, 1.750, and 1.875 for $p = 2, 4, 8, 16$. Doubling the number of GPUs from 8 to 16 raises the bandwidth term by only $1.875 / 1.75 = 7\%$, which is why the ring scales so well in bandwidth.

> [!KEY] The ring's bandwidth term is almost independent of $p$
> Each rank sends $2\frac{p-1}{p}n$ bytes, which never exceeds $2n$, however many ranks join. Adding GPUs adds latency, $2(p-1)\alpha$, but barely adds bandwidth cost. A large all-reduce on 1,000 GPUs sends 1.998 times the array per rank, against 1.0 times for two GPUs, which is why the algorithm is the default for big messages.

### Implementing it with isend and irecv

Here is the reduce-scatter as the lab's solution writes it, with the unusual cases kept in: chunk sizes differ when $n$ isn't a multiple of $p$, and a chunk is empty when $n < p$.

```python
def ring_reduce_scatter(x, rank, world_size):
    p = world_size
    bounds = ring_chunks(x.numel(), p)           # p (start, end) pairs, sizes differ by <= 1
    nxt, prv = (rank + 1) % p, (rank - 1) % p
    for step in range(p - 1):
        s_lo, s_hi = bounds[(rank - step) % p]       # the chunk I send
        r_lo, r_hi = bounds[(rank - step - 1) % p]   # the chunk I receive
        reqs, tmp = [], None
        if s_hi > s_lo:
            reqs.append(dist.isend(x[s_lo:s_hi], nxt))
        if r_hi > r_lo:
            tmp = torch.empty(r_hi - r_lo, dtype=x.dtype, device=x.device)
            reqs.append(dist.irecv(tmp, prv))
        for req in reqs:
            req.wait()
        if tmp is not None:
            x[r_lo:r_hi] += tmp
    return (rank + 1) % p
```

Three details matter. The send and the receive are posted together with the non-blocking `isend` and `irecv`, and only then waited on. If every rank called a blocking `send` first, each would wait for a receive that no rank has posted, and the ring would deadlock. The receive goes into a scratch buffer, because the incoming chunk must be added to the local one. And an empty chunk is skipped on both ends, which each side can tell from the chunk sizes alone.

The all-gather has the same shape, with `irecv` writing straight into the slice of `x` and no addition. Your lab asks for both, then calls them in sequence.

### Why this is the best possible

The ring's bandwidth term is not just good. It's the smallest any algorithm can have, in the model where every rank has its own link. Patarasuk and Yuan prove the bound for clusters of workstations, and a counting argument gives its flavor.

Take one element in the array. The sum needs $p-1$ additions to combine $p$ contributions, and then all $p$ ranks need the result. In the best case, almost every one of those operations moves data to a different rank, so each element needs about $2(p-1)$ transfers. Across all $n$ bytes of the array, that is about $2(p-1)\,n$ byte-transfers. Each rank has an outgoing link of rate $B$, and the $p$ links work in parallel, so the best time is the total volume over the total rate:

$$
t \ge \frac{2(p-1)\,n}{p\,B}
$$

That is the ring's bandwidth term. The nccl-tests documentation gives the same argument and calls its result the *bus bandwidth*, which is the subject of a later section.

The bound has a precise scope: it assumes each rank sends on its own link and that data moves only between ranks. A switch that reduces data in the network, such as SHARP, steps outside that model, which is how those algorithms beat $2n/B$. This course doesn't quote their performance, because docs/FACTS.md has no confirmed figures for them.

## What a collective costs on real links

Now apply the model to the links in docs/FACTS.md. This section answers one question: how long do the all-reduces of a real training step take on each link?

### A 1 GiB all-reduce

Start with the size we use throughout the lab: 1 GiB, which is $1.0737 \times 10^{9}$ bytes. For 8 ranks, the bandwidth term has a factor of $2 \cdot 7/8 = 1.75$, so each rank sends $1.75 \times 1.0737 \times 10^{9} = 1.879 \times 10^{9}$ bytes. The latency term is $2 \times 7 = 14$ steps of $\alpha$, or 70 µs at 5 µs. The table gives the lab's five cases.

| Case | $p$ | $B$ per direction | Bandwidth term | Latency term | Total |
|---|---|---|---|---|---|
| A100 NVLink 3 | 8 | 300 GB/s | 6.26 ms | 70 µs | 6.33 ms |
| A100 NVLink bridge | 2 | 300 GB/s | 3.58 ms | 10 µs | 3.59 ms |
| H100 NVLink 4 | 8 | 450 GB/s | 4.18 ms | 70 µs | 4.25 ms |
| 8 GPUs over PCIe Gen4 | 8 | 31.5 GB/s | 59.65 ms | 70 µs | 59.72 ms |
| 16 GPUs, one NDR port per ring link | 16 | 50 GB/s | 40.27 ms | 150 µs | 40.42 ms |

Three observations follow. The link sets the time: PCIe is 9.5 times slower than NVLink 3 for the same data. The latency term is under 2% of the total for a 1 GiB message on every row, so for large messages only $B$ matters. And the 16-GPU row is slower than any 8-GPU NVLink row because its ring crosses an InfiniBand-class link. The ring is as slow as its slowest link, which is the reason for the hierarchical algorithm in the topology section.

Your lab reports these five predictions, with the same 5 µs, as `pred_ring_8x_nvlink3_ms`, `pred_ring_8x_pcie4_ms`, `pred_ring_8x_nvlink4_ms`, `pred_ring_2x_nvlink3_ms`, and `pred_ring_16x_ib_ndr_ms`.

> [!WARNING] The 600 and 64 on the datasheet are totals
> NVIDIA's A100 datasheet lists NVLink at 600 GB/s and PCIe Gen4 at 64 GB/s. Both are the *sum of both directions*. A ring rank sends on its outgoing link while it receives on its incoming one, so the rate that bounds it is half: 300 and 31.5 GB/s. Dividing by 600 gives a time half the truth. The companion course's [chapter 19](https://qwen.ifkash.dev/c/19-tensor-parallelism) divides by the quoted totals for its two-GPU all-reduce, so its communication times are about half the per-direction figures. Use 300 and 31.5 here.

### A tensor parallel layer

A smaller, more frequent message shows the other regime. Take 4096 tokens of a 4096-wide bfloat16 hidden state, the activation from chapter 0, which is $4096 \times 4096 \times 2 = 33{,}554{,}432$ bytes. One tensor parallel all-reduce on 8 A100s over NVLink 3 sends $1.75 \times 33.55 \times 10^{6} = 58.7 \times 10^{6}$ bytes per rank:

$$
t = \frac{58.72 \times 10^{6}}{300 \times 10^{9}} + 14 \times 5\ \mu\text{s} = 195.7\ \mu\text{s} + 70\ \mu\text{s} = 265.7\ \mu\text{s}.
$$

A transformer layer has two of them, so about 531 µs per layer, per forward pass, before any overlap. The latency is already 26% of each all-reduce. Over PCIe the bandwidth term alone is $58.72 \times 10^{6} / 31.5 \times 10^{9} = 1.86$ ms, seven times as long as the NVLink total.

Now decode. One token of 4096 bfloat16 values is 8192 bytes. The ring's bandwidth term is $1.75 \times 8192 / 300 \times 10^{9} = 47.8$ ns, and its latency term is still 70 µs. The message is so small that the ring spends 99.9% of its time on $\alpha$. This is the regime that trees, and NCCL's low-latency protocols, exist for.

## Trees and the latency problem

A ring is the wrong algorithm for small messages, because its latency term is linear in $p$. This section answers one question: what do you use instead, and when does it win?

The ring's $2(p-1)$ steps are sequential. A tree reduces to a root in $\lceil \log_2 p \rceil$ levels and broadcasts back down in as many more, so its latency term is $2\lceil\log_2 p\rceil\,\alpha$. For $p = 8$ that is 6 messages against the ring's 14.

NCCL doesn't use one plain binary tree, because half of the ranks are leaves, and a leaf sends data up but never forwards any, which wastes its outgoing link. NVIDIA's NCCL 2.4 blog post describes the *double binary tree*: two trees laid out so that no rank is an interior node in both and at most one is a leaf in both. Each tree handles half of the data, so every rank's links are kept busy, and the pair is as efficient as a ring in bytes sent and received, according to the post. The same post reports up to 180 times lower latency than a ring at 24,576 GPUs on the Summit supercomputer.

### An idealized tree model

We model the tree's cost the same way, with a latency term of $2\lceil\log_2 p\rceil$ messages and a bandwidth term that approaches $2n/B$ when the tree is pipelined:

$$
T_{\text{tree}} \approx 2\lceil\log_2 p\rceil\,\hla{\alpha} + \frac{2n}{\hlb{B}}
$$

This is a model, not NCCL's code: a real tree is pipelined in chunks, its constant factors depend on the protocol, and NCCL uses its own tuning tables to choose. The model still shows the trade. The tree's bandwidth term is $2n/B$ against the ring's $2\frac{p-1}{p}n/B$, slightly worse, and its latency is logarithmic in $p$ instead of linear.

Set the two equal and solve for the size where they cross. The $2n/B$ terms differ by $\frac{2n}{B}\cdot\frac{1}{p}$, and the latency terms differ by $2\alpha\,(p - 1 - \lceil\log_2 p\rceil)$, so:

$$
\boxed{n^{\ast} = \alpha\,B\,p\,\bigl(p - 1 - \lceil\log_2 p\rceil\bigr)}
$$

Below $n^{\ast}$ the tree is faster in the model, and above it the ring is. The lab asks you to derive this from the two functions and to code it as `crossover_bytes`.

> [!EXAMPLE] Where the crossover sits
> With $\alpha = 5\ \mu\text{s}$:
>
> - 8 ranks on NVLink 3: $n^{\ast} = 5 \times 10^{-6} \times 300 \times 10^{9} \times 8 \times (8 - 1 - 3) = 48 \times 10^{6}$ bytes, or 45.8 MiB. Your lab reports this as `crossover_mib_8x_nvlink3`.
> - 8 ranks over PCIe Gen4: $5 \times 10^{-6} \times 31.5 \times 10^{9} \times 8 \times 4 = 5.04$ MB. A slower link has a smaller crossover, because bytes are expensive and latency matters less.
> - 16 ranks on NVLink 3: $5 \times 10^{-6} \times 300 \times 10^{9} \times 16 \times (16 - 1 - 4) = 264 \times 10^{6}$ bytes, or 252 MiB.
>
> The 8 KiB decode all-reduce from the last section is far below all three, so the model puts it on the tree: $6 \times 5 = 30\ \mu\text{s}$ against the ring's 70 µs.

Try it yourself. The figure plots both models against message size. Move the link, the number of ranks, and $\alpha$, and watch the crossover move.

```viz
21-allreduce-cost
```

The model says the crossover grows quickly with $p$: it is proportional to $p\,(p - 1 - \lceil\log_2 p\rceil)$, roughly $p^2$. At 64 ranks on NVLink 3 the formula gives 5.47 GB. NCCL doesn't pick algorithms from this idealized formula, so don't read that as a threshold you can rely on. Read it as the direction: more ranks push more of the size range toward trees.

## How NCCL chooses

NCCL is the library behind `dist.all_reduce` on GPU tensors, and it makes three choices for every call. This section answers one question: what is NCCL deciding, and how can you see or override it?

**The algorithm.** For an all-reduce, the Demystifying NCCL paper lists Ring, Tree, and several specialized ones: CollNet (which uses SHARP, in-network reduction, on supporting networks) and NVLS (which uses NVLink SHARP on NVSwitch systems). NCCL also has newer algorithms, such as PAT, which this course names but doesn't analyze. The default is automatic, chosen from the message size, the topology, and tuning tables. `NCCL_ALGO` forces one, which is useful for experiments and risky in production.

**The protocol.** Each algorithm runs over one of three protocols, which trade latency against bandwidth. The paper's Table I summarizes them:

| Protocol | How the receiver knows data arrived | Payload | Bandwidth | Per-hop latency |
|---|---|---|---|---|
| Simple | Memory fences after a whole chunk | Whole chunks | Near peak | about 6 µs |
| LL (low latency) | A flag sent with the data | 4 bytes of data and a 4-byte flag | 25% to 50% of peak | about 1 µs |
| LL128 | A flag in each 128-byte unit | 120 bytes of data and 8 bytes of flag | about 95% of peak | about 2 µs |

Read the rows as a story. Simple is careful and slow to start: it waits for a fence before it hands over a chunk, so it wins on large messages and loses on small ones. LL puts a flag next to every 4 bytes, so the receiver can spin on the flag and proceed immediately, at the price of half the payload. LL128 does the same in 128-byte units, which keeps 120 of every 128 bytes as data, and so suits NVLink. `NCCL_PROTO` forces one.

**The channels.** NCCL splits one collective into *channels* that run in parallel. A NCCL kernel is an ordinary CUDA kernel with one block per channel, and the paper describes the grid as `(nChannels, 1, 1)`. So communication takes streaming multiprocessors away from compute while it runs. You met SMs in [chapter 1](/c/01-the-gpu-at-a-glance), and a collective competes with your matmul for them, which chapter 22 returns to when it overlaps the two.

Inside a channel, the ring is built from a few primitives. The paper names `send`, `recv`, `recvReduceSend` (receive a chunk, add it into the local one, and forward the result), `recvCopySend`, and `recvReduceCopySend`. Those are the two phases of the lab's ring, fused into single steps, and they run pipelined over slots: a chunk moves on while the next is still arriving, which is how NCCL fills a link without waiting for a whole message.

> [!NOTE] What this course doesn't claim
> docs/FACTS.md has no confirmed NCCL bus bandwidth from a primary source. Forum posts and vendor blogs quote 230 to 240 GB/s for 8 A100s on NVSwitch and 450 to 480 GB/s for 8 H100s, but those are unverified, so this chapter doesn't use them. It computes the ceiling instead and tells you to measure the real value.

## Algorithm bandwidth and bus bandwidth

A measured all-reduce time means little until you turn it into a rate that you can compare with the hardware. This section answers one question: how does nccl-tests report a collective's speed, and what number should it approach?

The nccl-tests repository measures NCCL collectives and reports two rates. The *algorithm bandwidth*, $\text{algbw}$, is the size divided by the time:

$$
\text{algbw} = \frac{S}{t}
$$

It is what you'd compute without thinking, and it has a flaw: the same hardware gives a different $\text{algbw}$ at different $p$. An all-reduce on 8 ranks moves more bytes per rank than one on 2, so its $\text{algbw}$ is lower even if the links are used equally well. The *bus bandwidth* corrects for it:

$$
\text{busbw} = \text{algbw} \times \text{factor}(p)
$$

The factor is chosen so that $\text{busbw}$ estimates the rate at which each link is actually busy, which you can compare with the link's peak. The factors, from the nccl-tests documentation, are:

| Collective | Factor |
|---|---|
| All-reduce | $2(p-1)/p$ |
| Reduce-scatter | $(p-1)/p$ |
| All-gather | $(p-1)/p$ |
| All-to-all | $(p-1)/p$ |
| Broadcast | 1 |
| Reduce | 1 |

> [!EXAMPLE] From a time to a busbw
> Suppose 8 ranks all-reduce 1 GB in 10 ms. Then $\text{algbw} = 10^{9} / 0.010 = 100$ GB/s, and $\text{busbw} = 100 \times 1.75 = 175$ GB/s. The lab's `bus_bandwidth_gbs` computes exactly this, and its test uses these numbers.
>
> The 175 GB/s is below the 300 GB/s per-direction ceiling for an A100 on NVSwitch. It says the all-reduce used 58% of what one rank's link can send. A $\text{busbw}$ of 300 would mean the ring kept the link saturated.

This gives a ceiling you can state without a measurement. For an all-reduce, the ring's time is at least $2\frac{p-1}{p}\,n / B$, so $\text{busbw}$ is at most $B$: 300 GB/s per direction for NVLink 3, 450 for NVLink 4, 31.5 for PCIe Gen4. A real NCCL run lands below those, and how far below depends on the protocol, the message size, and the topology. To find out for your machine, run `all_reduce_perf` from nccl-tests over a sweep of sizes, for example `-b 8 -e 1G -f 2 -g 8` on an 8-GPU node, and compare the plateau of $\text{busbw}$ with $B$.

## Topology: what the links are

The numbers above come from the wires. This section answers one question: what links connect GPUs, and how do their rates compare?

| Link | Total, both directions | Per direction | Source |
|---|---|---|---|
| PCIe Gen4 x16 (A100 host link) | 63.0 GB/s (datasheet prints 64) | 31.5 GB/s | A100 datasheet; per-direction derived from 16 GT/s $\times$ 128/130 $\times$ 16 lanes / 8 |
| PCIe Gen5 x16 (H100 host link) | 126 GB/s (NVIDIA prints 128) | 63 GB/s | Hopper whitepaper |
| NVLink 3 (A100, 12 links) | 600 GB/s | 300 GB/s | A100 datasheet |
| NVLink 4 (H100, 18 links) | 900 GB/s | 450 GB/s | Hopper whitepaper |
| NVLink 5 (B200, 18 links) | 1,800 GB/s | 900 GB/s | NVIDIA NVLink page |
| InfiniBand NDR port (Quantum-2) | 400 Gb/s each way | 50 GB/s | NVIDIA Quantum-2 page; 400 / 8 |

Two facts about this table organize the rest of the section. NVLink's per-GPU rate has grown from 300 to 450 to 900 GB/s per direction over three generations, so each generation moves the crossover and the half-bandwidth size up with it. And the step from NVLink to the network is large: a 50 GB/s InfiniBand port is one sixth of NVLink 3's 300 GB/s per direction, and one eighteenth of NVLink 5's 900.

**NVLink and NVSwitch.** NVLink is a direct link between GPUs. With only a few GPUs you can wire them point to point, and the A100 PCIe card the lab runs on can take a bridge to exactly one other card. For eight GPUs, NVIDIA puts *NVSwitch* chips in the node, so any GPU can talk to any other at full NVLink rate. The NVLink Switch generation for H100 lists 900 GB/s per GPU, and for B200 1,800 GB/s per GPU, with 130 TB/s across a 72-GPU rack. Those are totals, so divide by two for $B$.

**PCIe.** A node whose GPUs have no NVLink talks through PCIe and the CPU's root complex, at 31.5 GB/s per direction on Gen4, shared with every other device on the bus. This is the 59.65 ms row in the table above.

**InfiniBand and RoCE.** Between nodes, GPUs talk through network cards. InfiniBand NDR runs at 400 Gb/s per port, which is 50 GB/s each way, and a ConnectX-7 card carries one or two such ports. RoCE (RDMA over Converged Ethernet) offers the same remote-memory-access model over Ethernet. For both, NCCL can use GPUDirect RDMA, where the network card reads and writes GPU memory directly, so the data doesn't detour through host memory.

**Rail-optimized fabrics.** A cluster usually connects GPU $i$ of every node to the same leaf switch, called a *rail*. Traffic between GPUs with the same index crosses one switch, and NCCL uses the node's NVLink for everything else. The point is to give each GPU its own network path instead of funneling all eight through one card.

### Why the topology changes the algorithm: hierarchical all-reduce

A flat ring over two 8-GPU nodes pays for the slow link at every step. The 16-GPU row of the earlier table, at 40.42 ms, shows the cost: each step is paced by the InfiniBand edges, so the whole ring runs at 50 GB/s.

A hierarchical all-reduce does better by using each level for what it is good at. Take a hypothetical pair of nodes, each with 8 GPUs on NVLink 3 and one 50 GB/s NDR port per GPU, so the eight rails are independent:

1. **Reduce-scatter inside each node.** Each GPU ends up with the node's sum of $n/8$ of the array. Per GPU, that is $\frac{7}{8}n / 300\ \text{GB/s} = 3.13$ ms for $n = 1$ GiB.
2. **All-reduce across nodes.** GPU $i$ of node 0 all-reduces its $n/8$ slice with GPU $i$ of node 1, over its own rail. With two ranks, each sends its slice once: $n/8 = 134.2$ MB at 50 GB/s, or 2.68 ms.
3. **All-gather inside each node.** The mirror of step 1, 3.13 ms.

The total is about $3.13 + 2.68 + 3.13 = 8.94$ ms, against 40.27 ms for the flat ring's bandwidth term, because only $n/8$ of the data crosses the slow link and all eight links cross at once. The sum ignores latency and any overlap between the phases, and the 8.94 ms assumes the rails don't contend. It is a model's prediction, and NCCL's own hierarchical algorithms are more refined, but the structure is the lesson: do the heavy movement on the fast level, and send across the slow level only what you must.

> [!DEEPDIVE] How the balance shifts with more nodes
> In the two-node example, the inter-node step moves $n/8$ per GPU while each intra-node step moves $7n/8$. The intra-node steps carry 7 times the data and take only 1.17 times as long (3.13 against 2.68 ms), because NVLink 3 is 6 times as fast as an NDR port. The levels are about equally loaded.
>
> More nodes change the balance. With $m$ nodes, the cross-node all-reduce on $n/8$ bytes sends $2\frac{m-1}{m}\cdot\frac{n}{8}$ per GPU, which approaches $n/4$ as $m$ grows. That is 268 MB at 1 GiB, or 5.37 ms on a 50 GB/s port, still far below the flat ring's 40 ms. What grows with $m$ is the latency: $2(m-1)\alpha$ for a ring across nodes, or $2\lceil\log_2 m\rceil\,\alpha$ for a tree, which is where a large cluster pays and why NCCL's tree matters there.

## What goes wrong

The lab's mistakes are mostly about ordering, sizes, and units. The mistakes that follow you to production are about the numbers.

- **Blocking send in every rank.** If each rank calls `dist.send` to its successor first, every rank waits for a receive that nobody has posted, and the ring hangs. Post `isend` and `irecv` together, then wait on both. The lab's harness turns a hang into a timeout, so you see an error instead of a stalled job.
- **Writing the receive into the buffer you are sending.** In the reduce-scatter, the chunk you send is not the chunk you receive, but both live in `x`. Receive into a scratch buffer and add after the wait, or you overwrite data still in flight.
- **Wrong chunk indices.** An off-by-one in $(r - s) \bmod p$ gives plausible but wrong sums. The lab's exact test on integer-valued data catches this, because any summation order gives the same answer there. A test on random floats alone would hide it behind rounding error.
- **Sizes that don't divide.** A length-1 array on 4 ranks has three empty chunks, and a length that isn't a multiple of $p$ has chunks of unequal size. Compute bounds with `divmod` and skip empty sends and receives on both ends.
- **Using the datasheet total as $B$.** Dividing by 600 GB/s for an A100 or 64 GB/s for PCIe Gen4 understates the time by 2. Use 300 and 31.5.
- **Comparing algbw across rank counts.** An 8-rank $\text{algbw}$ is lower than a 2-rank one on the same hardware. Compare $\text{busbw}$.
- **Reading a CPU measurement as a link measurement.** The lab's timings use gloo on CPU tensors in processes sharing one machine's memory. Its fitted $\alpha$ and $B$ describe that machine's inter-process copies, and not NVLink. Use them to check your model, and don't use them to rate a GPU.
- **Quoting a small-message time with a large-message model.** At 8 KiB the bandwidth term is 48 ns and the latency term is 70 µs. A model with $\alpha = 0$ predicts a time 1,500 times too small.

> [!RECAP]
> - Every multi-GPU strategy ends in a collective: all-reduce for data and tensor parallelism, reduce-scatter and all-gather for FSDP, all-to-all for experts. An all-reduce is a reduce-scatter followed by an all-gather.
> - One message costs $\alpha + n/B$, with $B$ the per-direction rate. The half-bandwidth size $\alpha B$ is 1.5 MB on NVLink 3 and 157.5 KB on PCIe Gen4, at an assumed 5 µs.
> - A ring all-reduce takes $2(p-1)\alpha + 2\frac{p-1}{p}\frac{n}{B}$. Its bandwidth term is optimal, so large messages want a ring, and its latency term grows with $p$, so small messages want a tree.
> - On 8 A100s over NVLink 3, 1 GiB takes 6.33 ms by the model. Over PCIe Gen4 it takes 59.72 ms. The lab reports both, and a real NCCL run lands somewhere above them.
> - nccl-tests' busbw is algbw times $2(p-1)/p$ for an all-reduce, and its ceiling is $B$ per direction, never the quoted two-direction total.
> - NCCL chooses the algorithm and the protocol, and a collective runs as a kernel that takes SMs from your compute.

## Check your understanding

> [!QUESTION] Why does an all-reduce of $n$ bytes over $p$ ranks send $2\frac{p-1}{p}n$ bytes per rank in a ring?
> It runs a reduce-scatter and an all-gather, each of $p-1$ steps that send one chunk of $n/p$ bytes. That is $2(p-1)\cdot n/p$ bytes per rank. For 8 ranks it is $1.75n$, and as $p$ grows it approaches $2n$.

> [!QUESTION] An 8-GPU A100 node all-reduces 1 GiB. Compute the bandwidth term of the ring at 300 GB/s per direction, and say what mistake gives 3.1 ms.
> $1.75 \times 1.0737 \times 10^{9} / 300 \times 10^{9} = 6.26$ ms. Using the datasheet's 600 GB/s, which is the total of both directions, gives 3.13 ms, half the truth.

> [!QUESTION] A tensor parallel all-reduce moves 8 KiB across 8 ranks, with $\alpha = 5\ \mu\text{s}$ and $B = 300$ GB/s. Does the ring or the idealized tree win, and by how much?
> The tree wins. The ring takes $14 \times 5 = 70\ \mu\text{s}$ plus 48 ns of bandwidth. The tree takes $6 \times 5 = 30\ \mu\text{s}$ plus $2 \times 8192 / 300 \times 10^{9} = 55$ ns. Both are dominated by latency, and the tree is 2.3 times faster, which is $14/6$.

> [!QUESTION] Where is the crossover size for 8 ranks on PCIe Gen4 with $\alpha = 5\ \mu\text{s}$, and why is it smaller than on NVLink 3?
> $n^{\ast} = \alpha B p (p - 1 - \lceil\log_2 p\rceil) = 5 \times 10^{-6} \times 31.5 \times 10^{9} \times 8 \times 4 = 5.04$ MB. It is smaller because it is proportional to $B$: on a slow link the bandwidth term outweighs latency at a smaller size, so the ring takes over sooner. NVLink 3 gives 48 MB.

> [!QUESTION] Eight ranks all-reduce 2 GB in 20 ms. What are algbw and busbw, and what do they say about an A100 NVSwitch node?
> $\text{algbw} = 2 \times 10^{9} / 0.020 = 100$ GB/s, and $\text{busbw} = 100 \times 1.75 = 175$ GB/s. The ceiling is 300 GB/s per direction, so the run used 58% of one rank's link. The gap is for the protocol, the algorithm, and the kernel's use of SMs to explain, and a measurement tells you which.

> [!QUESTION] Why can't a ring over two 8-GPU nodes use the NVLink rate, even though most of its links are NVLink?
> Every step of the ring moves one chunk across every edge at once, so the step lasts as long as its slowest edge. The edges that cross between nodes run at the network's 50 GB/s, and the ring is paced by them. A hierarchical all-reduce avoids this by moving only $n/8$ per GPU across the network, after a reduce-scatter inside each node.

> [!QUESTION] What do LL and LL128 give up, and what do they buy?
> LL sends 4 bytes of data with a 4-byte flag, so it delivers 25% to 50% of peak bandwidth, and in return the receiver can proceed without a memory fence, for about 1 µs per hop against about 6 µs for Simple. LL128 sends 120 bytes of data per 128, reaching about 95% of peak at about 2 µs per hop, and suits NVLink. These are the paper's figures, with hardware unnamed.

## Lab

> [!TRY] Ring all-reduce and the cost model
> Fill in the alpha-beta functions (`transfer_time`, `ring_allreduce_time`, `tree_allreduce_time`, `crossover_bytes`, `bus_bandwidth_gbs`, and `fit_alpha_beta`) and the ring (`ring_chunks`, `ring_reduce_scatter`, `ring_all_gather`, and `ring_allreduce`).
>
> The tests spawn 2, 3, and 4 processes with `torch.multiprocessing` on the gloo backend with CPU tensors, so this lab doesn't use the GPU. You pass when your ring equals `dist.all_reduce` exactly on integer-valued data and within float32 rounding on random data, for sizes that include 1, 3, and 7, which are smaller than or not a multiple of the number of ranks. You also pass when each rank sends exactly $2\frac{p-1}{p}$ of the array, which the test counts by wrapping `dist.isend`, and when your cost functions reproduce this chapter's arithmetic: 6.26 ms, 59.65 ms, and a 70 µs latency term.
>
> The lab reports the predictions `pred_ring_8x_nvlink3_ms`, `pred_ring_8x_pcie4_ms`, `pred_ring_8x_nvlink4_ms`, `pred_ring_2x_nvlink3_ms`, `pred_ring_16x_ib_ndr_ms`, and `crossover_mib_8x_nvlink3`, and the measurements `fit_alpha_us`, `fit_bandwidth_gbs`, `ring_ms`, `builtin_ms`, `ring_busbw_gbs`, `builtin_busbw_gbs`, `ring_over_builtin`, and `model_over_measured`. The measurements are for inter-process copies on the lab machine, so compare them with your model, not with NVLink. Allow about half a minute, since each of the five jobs starts a few processes.

## Further reading

- [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf): Patarasuk and Yuan's proof that the ring is bandwidth-optimal.
- [Optimization of Collective Communication Operations in MPICH](https://www.cs.umd.edu/class/fall2019/cmsc714/readings/Thakur-MPICH.pdf): the alpha-beta cost of each collective and algorithm.
- [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/): Gibiansky's step-by-step ring all-reduce.
- [nccl-tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): the definitions of algbw and busbw and each collective's factor.
- [Demystifying NCCL](https://arxiv.org/abs/2507.04786): ring, tree, and the Simple, LL, and LL128 protocols, with the per-hop latencies this chapter cites.
- [Massively Scale Your Deep Learning Training with NCCL 2.4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/): the double binary tree and why latency matters at scale.
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html): the API, the algorithms, and `NCCL_ALGO` and `NCCL_PROTO`.
- [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html): `send`, `recv`, `isend`, `irecv`, and the collectives the lab uses.
- [NVIDIA NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/): the per-GPU totals for each generation. Halve them for the per-direction rate.
- [How to Scale Your Model: Collectives](https://jax-ml.github.io/scaling-book/): all-reduce, all-gather, and reduce-scatter costs on rings and tori, from the JAX side.
- [Tensor parallelism](https://qwen.ifkash.dev/c/19-tensor-parallelism) in the companion course: where the all-reduce sits inside a transformer layer. Its link bandwidths are totals, so halve them.
