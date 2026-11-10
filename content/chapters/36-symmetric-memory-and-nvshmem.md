---
title: Symmetric memory, NVSHMEM, and fused communication
slug: 36-symmetric-memory-and-nvshmem
part: "Part 5 — Many GPUs"
summary: One-sided put and get on a symmetric heap, the fence and quiet rules that keep a signal behind its data, and how a signalled all-gather feeds a GEMM tile loop, with an alpha-beta model of when fusing beats separate collectives.
minutes: 110
gpu: true
objectives:
  - "Contrast one-sided put and get with two-sided send and receive, and say which side of a one-sided transfer is passive."
  - "Translate a (PE, offset) pair on a symmetric heap into a remote address, and say why nvshmem_malloc must run on every PE with the same arguments."
  - "State what nvshmem_putmem, nvshmem_getmem, nvshmem_put_signal, nvshmem_fence, and nvshmem_quiet each guarantee, and place a fence so a signal cannot overtake its data."
  - "Name the PyTorch symmetric memory entry points at a conceptual level, and say that the module is in alpha."
  - "Compare copy-engine communication with SM-driven communication, and say what each costs the compute kernel."
  - "Order the tiles of an all-gather plus GEMM by when each chunk's signal arrives, and say why no tile may read a chunk before its signal."
  - "Evaluate an alpha-beta model of serial, overlapped, and fused execution, and say when fusing wins and when it does not."
lab: 36-symmetric-memory
---

# Symmetric memory, NVSHMEM, and fused communication

> [!TLDR]
> - In two-sided communication, a send on one rank pairs with a receive on another. In one-sided communication, one rank writes into or reads from another rank's memory, and the other rank runs no matching call.
> - A symmetric heap gives every processing element (PE) an allocation at the same offset. An address on a remote PE is the remote heap's base plus that offset, so a (PE, offset) pair names a location anywhere in the job.
> - A put returns before its data arrives. A fence orders earlier operations ahead of later ones to the same PE without waiting. A quiet waits until your own operations complete. A signal that follows its data needs a fence, or the signal can arrive first.
> - Communication that a kernel drives with its own SMs avoids a launch per transfer and costs compute SMs. Communication that a copy engine drives costs no SMs and costs a launch. A fused all-gather plus GEMM signals each chunk, and the matching tile waits on that signal.
> - The alpha-beta model puts a number on the trade. For a 128 MiB all-gather feeding a GEMM on 8 GPUs, it predicts 866.978 µs serial, 737.038 µs with eight separate chunked collectives, and 535.494 µs fused. The latencies in that prediction are assumed, and the lab's two-GPU run is the only measurement.

[Chapter 21](/c/21-collective-communication) prices a collective with $\alpha + n/B$. [Chapter 22](/c/22-overlap-and-parallelism) ends with a section called "Overlap inside a kernel" that says tensor parallelism has no independent work to hide its all-reduce behind, and that the way out is to cut the matmul into tiles and communicate each tile as it finishes. That section names the idea and stops. This chapter builds the machinery underneath it: memory that every GPU can address, writes that go straight into a peer's memory, and flags that tell a consumer when data has landed.

The machinery matters because the dependent part of a tensor-parallel layer sits on the critical path. A stream cannot hide an all-reduce whose result the next matmul needs. A kernel that sends each output tile the moment it exists, and a kernel that starts each input tile the moment it arrives, can. The mechanism is a shared heap, one-sided writes, and a flag.

The lab builds that mechanism in Python on a CPU: a symmetric heap on simulated PEs with put, signal, fence, and quiet, and a network model that delivers operations in an adversarial order, so a missing fence shows up as stale data. On top you write an all-gather from puts, a chunked all-gather plus GEMM that never reads an unsignaled chunk, and the alpha-beta model. A separate two-GPU check times a chunked NCCL all-gather plus matmul, and it skips without two GPUs.

## Before you start

**A PE.** A *PE*, or processing element, is the NVSHMEM word for one rank of the job. This chapter says PE for the one-sided material and rank for NCCL and `torch.distributed`. [Chapter 21](/c/21-collective-communication) is where ranks and collectives are defined.

**Streams and engines.** [Chapter 22](/c/22-overlap-and-parallelism) explains that a copy engine moves bytes without using the SMs. Its "Overlap inside a kernel" section is the one this chapter expands, and you need its claim that a communication kernel takes SMs and memory bandwidth from a compute kernel.

**The alpha-beta model.** [Chapter 21](/c/21-collective-communication) gives the time of one message as $\alpha + n/B$, and [chapter 25](/c/25-ddp-and-zero) uses it to price buckets. This chapter reuses it and adds a per-chunk signal latency. The latencies are assumed values, not measurements, and the chapter says so each time.

**What you are not rebuilding.** Ring attention and context parallelism are owned by the companion course, learn-inference, at [qwen.ifkash.dev](https://qwen.ifkash.dev). They use the same send-a-chunk-while-computing idea on attention, and this chapter does not teach them. Expert-parallel dispatch appears once as a motivating example, not as a lesson.

## Two-sided and one-sided communication

This section answers what a one-sided transfer removes compared with the send and receive you used in [chapter 21](/c/21-collective-communication).

Picture two people exchanging a package. In the two-sided version, one person hands it over and the other must be at the door to take it. Both have to call the exchange, and each call has to name the other. In the one-sided version, the sender has a key to a locker in the receiver's building and puts the package in. The receiver does nothing at that moment. It learns the package arrived only if someone tells it.

In code, two-sided is `dist.send(tensor, dst)` on one rank and `dist.recv(tensor, src)` on another. The pair has to match: same size, same tag order, and a receive posted for every send. A mismatch hangs. This is the contract behind every NCCL collective's internals, and behind the pipeline-parallel sends of [chapter 22](/c/22-overlap-and-parallelism).

One-sided communication names two operations. A *put* copies bytes from your local memory into a location on a remote PE. A *get* copies bytes from a location on a remote PE into your local memory. The remote PE is *passive*: its code makes no call, and its kernel keeps running. The NVSHMEM [remote memory access documentation](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html) describes the interface as one-sided, with the target passive and communication decoupled from synchronization.

That decoupling is the hard part, and it is the reason the rest of the chapter exists. A receive tells the receiver when the data is there. A put does not. The receiver needs another channel, a flag, to learn that the data arrived. Two-sided communication bundles the transfer and the notification. One-sided communication separates them, and you rebuild the notification yourself.

| Property | Two-sided send and receive | One-sided put and get |
|---|---|---|
| Who calls | Both ranks, with matching arguments | The initiator only |
| Who learns the data arrived | The receiver, when its receive returns | Nobody, unless a flag says so |
| Typical hazard | A mismatched pair hangs | A flag that arrives before its data |
| What the receiver can do meanwhile | Block in the receive, or poll a handle | Keep computing |
| Natural fit | Collectives, pipeline stages | Fine-grained, irregular, or fused transfers |

The right column has one cost and one gift. The cost is the notification you must write. The gift is that a kernel can issue a put from inside its own code, per tile, with no host call. That is what makes fusing communication into a kernel possible.

> [!KEY] One-sided separates the transfer from the notification
> A put moves bytes and says nothing to the target. The target learns about the data through a signal that you order behind the put.

## The symmetric heap

This section answers how a PE can name a location in another PE's memory without a pointer from that PE.

Picture each PE as a building with the same floor plan. Room 3 is in the same place in every building. To leave a note in room 3 of building 5, you do not need building 5 to tell you where room 3 is. You know the floor plan, and you know where building 5 starts.

That is a *symmetric heap*. Every PE reserves a region of GPU memory of the same size and lays it out the same way. When a program allocates an object, the allocation lands at the same offset on every PE. NVSHMEM's `nvshmem_malloc` returns the symmetric address of a block from that heap. The [NVSHMEM memory management documentation](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html) says the allocation is collective: every PE must call it, with identical arguments, and the behavior is undefined if the arguments differ. The call is semantically equivalent to a barrier on exit, so the memory on every other PE is usable once your call returns.

The identical-arguments rule is what keeps the layout symmetric. If PE 0 allocates 4 KiB and then 8 KiB, and PE 1 allocates 8 KiB and then 4 KiB, offset 0 means different objects on the two PEs. A put to "offset 0" lands in the wrong object. Symmetry is a contract on the sequence of allocations, not something the runtime can repair.

### Address translation

Each PE's heap starts at its own virtual base address, and the bases differ. An object at offset $o$ on PE $p$ therefore sits at

$$
\mathrm{addr}(p, o) = \hla{\mathrm{base}_p} + \hlb{o}
$$

where $\hla{\mathrm{base}_p}$ is where PE $p$'s heap starts in the address space the sender uses, and $\hlb{o}$ is the byte offset the allocator returned. To reach the same object on another PE, keep the offset and swap the base. A local address $a$ on PE $s$ becomes $a - \mathrm{base}_s + \mathrm{base}_d$ on PE $d$. That subtraction and addition is the whole of address translation.

> [!EXAMPLE] The same object on PE 1 and PE 3
> The lab's simulator gives PE $p$ the base $\mathtt{0x7F0000000000} + p \times 2^{32}$ and stores float32 elements, 4 bytes each. Element 3 on PE 1 sits at $\mathtt{0x7F0000000000} + 2^{32} + 12$. Translate that address to PE 3. The offset into the heap is 12 bytes, so the answer is $\mathtt{0x7F0000000000} + 3 \times 2^{32} + 12$. The offset did not change. Only the base did.
>
> An address that falls outside PE 1's heap has no meaning to translate, and the lab's `translate` raises `ValueError` for it.

On NVLink-connected GPUs, the runtime can map a peer's heap into the local address space, and a put becomes a store through that mapping. This is peer-to-peer access. When the peer is across a network, the runtime goes through a different path. The NVSHMEM documentation names the NVLink peer path and the remote path as separate algorithms in its tile interface, and it says a put's source or destination can be device shared memory when the devices are connected by a peer-to-peer transport. This chapter works with the NVLink case, which is what a two-GPU node has.

## The operations

This section answers what each NVSHMEM call guarantees, so you can predict what a program does.

The following table lists the calls this chapter uses. The names and semantics come from the NVSHMEM documentation pages for [remote memory access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html), [signaling and ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html), and [memory management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html).

| Call | What it does | When it returns |
|---|---|---|
| `nvshmem_malloc(size)` | Collective allocation from the symmetric heap | After every PE has allocated; an implied barrier |
| `nvshmem_putmem(dest, source, nbytes, pe)` | Copies `nbytes` from local `source` to `dest` on PE `pe` | When the source has been read locally; delivery can come later |
| `nvshmem_putmem_nbi(...)` | The same, nonblocking | After initiation; complete only after `nvshmem_quiet` |
| `nvshmem_getmem(dest, source, nbytes, pe)` | Copies from `source` on PE `pe` into local `dest` | When the data is in `dest` |
| `nvshmem_put_signal(...)` | A put, then an atomic update of a 64-bit signal word on the target | Same as a put; the signal update indicates delivery of the data |
| `nvshmem_signal_wait_until(...)` | Waits until a local signal word meets a condition | When the condition holds |
| `nvshmem_fence()` | Orders earlier operations to a PE before later ones to the same PE | Immediately; it does not wait for completion |
| `nvshmem_quiet()` | Waits until all operations you issued are complete and visible | When everything you issued has completed |

The documentation gives the signal operators as `NVSHMEM_SIGNAL_SET`, which writes a 64-bit value atomically, and `NVSHMEM_SIGNAL_ADD`, which adds to it. A signal word is a `uint64_t` in symmetric memory. The calls also come in variants for other issuers: the documentation lists `_block` and `_warp` forms that a group of threads calls together, and `_on_stream` forms that the host enqueues on a CUDA stream.

The subtlety is in the "when it returns" column. A blocking put returns once the source data has been read on the local PE, which means you may reuse your source buffer. It does not mean the data is in the remote memory. The lab's simulator models this. `put` appends to a list of in-flight operations and returns, and nothing arrives until the network delivers it.

### What the simulator models

The simulator models the rules, not the speed. A `put` and a `signal` go into `pending`, and the network delivers one operation at a time in an order a policy chooses: first in, first out, reverse, or `signals_first`, which delivers signals before data whenever the rules allow. A real NVLink fabric usually delivers a put and the signal behind it in order, so a program that omits the fence often appears to work. The `signals_first` policy is the worst case the documentation permits, and a program that is correct under it is correct on any legal network.

## Fence, quiet, and the signal that overtakes its data

This section answers which call you need to keep a flag behind the data it announces, and why `quiet` does not do the same job.

Picture a mailroom. You drop off a parcel and then a note that says "the parcel is ready." The mailroom may sort the note first, so the recipient can read the note while the parcel is still on a shelf. You can fix that two ways. You can stand at the desk until the parcel is confirmed delivered, and then drop off the note. Or you can ask the clerk to promise that the note never moves ahead of the parcel, and walk away. The first is `quiet`. The second is `fence`.

The [NVSHMEM ordering documentation](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html) defines them precisely:

- `nvshmem_fence` guarantees that operations on symmetric data that you issued to a particular PE before the call are delivered before any later operations to that same PE. It does not guarantee completion.
- `nvshmem_quiet` guarantees that all symmetric-data operations you issued are complete, so their updates are visible at the destination PEs. It is local and non-collective, and it does not notify other PEs.

The difference is waiting. A fence returns at once and constrains the network. A quiet blocks you until the network is done. A fence is the cheaper tool when you only need an order, and a quiet is the tool when you need completion, for example before you reuse a destination buffer or before you exit.

The signal needs an order, not a completion. You want the data delivered before the flag, and you have no reason to wait. The pattern is put, fence, signal:

```c
nvshmem_putmem_nbi(remote_buf, local_chunk, nbytes, peer);
nvshmem_fence();                    // data to `peer` is ordered before what follows to `peer`
signal_set(remote_flag, 1, peer);  // an atomic 64-bit update, NVSHMEM_SIGNAL_SET
```

`nvshmem_put_signal` packages this: the documentation says its signal update indicates delivery of the corresponding data. It also says that without a memory-ordering operation, a plain put followed by a put with signal does not guarantee the plain put's data has been delivered. The packaged call orders its own data against its own signal. It does not order your other puts.

### The epoch model

The lab models a fence with an *epoch*. Every operation carries the epoch of its (source, destination) pair at the moment it is issued. A fence raises the epoch of every pair that starts at the fencing PE by one. The network may deliver an operation only when no operation from the same source to the same destination with a smaller epoch is still in flight. That is the entire rule, and it is enough to reproduce the documented behavior.

> [!EXAMPLE] Three operations, two epochs
> PE 0 puts three floats to PE 1, fences, and signals PE 1. The put has epoch 0. The fence raises the (0, 1) epoch to 1, so the signal has epoch 1. Under `signals_first`, the network prefers a signal, but the signal's epoch is 1 and a put with epoch 0 is still in flight, so the signal is held back. The put goes first, then the signal. PE 1 sees the flag only after the data.
>
> Remove the fence and both operations have epoch 0. Nothing holds the signal back, `signals_first` delivers it first, and PE 1 sees the flag next to three zeros. This is the plausible wrong implementation the lab calls `signal_without_fence`.

> [!WARNING] A fence on the wrong side does nothing
> A fence orders the operations of the PE that calls it. A consumer that calls `nvshmem_fence` before it reads has not ordered anything about the producer's puts. The producer fences. The consumer waits on the signal. The lab's fence check fails a program that puts and signals with no fence, and it passes the put, fence, signal sequence.

Quiet has a different failure, and it is expensive instead of wrong. A quiet after every put works, because it completes the data before the signal goes out. It also stalls the issuing thread for the whole round trip of every transfer, and the sender can no longer pipeline the next chunk behind the current one. A quiet is correct and slow. A fence is correct and cheap. A missing fence is fast and wrong.

| Sequence | Is the signal behind the data? | What it costs |
|---|---|---|
| put, signal | No: the signal can arrive first | Nothing, and a stale read |
| put, fence, signal | Yes | No wait; the network holds the signal back |
| put, quiet, signal | Yes | The issuer waits for the put to complete |
| put_signal | Yes, for that put's own data | No wait |

## PyTorch symmetric memory

This section answers how the same idea appears in PyTorch, and which names are safe to rely on.

PyTorch exposes symmetric memory as a module, `torch.distributed._symmetric_memory`. The [symmetric memory documentation](https://docs.pytorch.org/docs/2.14/symmetric_memory.html) states that the module is in alpha and that API changes are possible. The leading underscore is the same warning. Treat what follows as a map of the ideas, and read the page for the version you run before you write code against a name.

The page documents these entry points, which map onto the NVSHMEM vocabulary:

- `empty` allocates a tensor backed by a symmetric memory allocation, like `torch.empty`.
- `rendezvous` establishes the allocation across a group of ranks and returns a handle. The page says every rank must call `empty` and `rendezvous` in the same order, which is the same identical-sequence contract as `nvshmem_malloc`.
- `get` performs a one-sided copy from a peer's symmetric allocation into a local tensor.
- `is_nvshmem_available` reports whether NVSHMEM is usable at runtime, and `set_backend` selects the backend: the page lists `"NVSHMEM"`, `"CUDA"`, and `"NCCL"`.
- The handle exposes `buffer_ptrs`, which are pointers to each peer's buffer, `multicast_ptr`, and `signal_pad_ptrs`. A *signal pad* is, in the page's words, a peer-to-peer-accessible memory region used for synchronization. It plays the role of the flags in this chapter.

The page also lists ready-made operators under `torch.ops.symm_mem`: `one_shot_all_reduce`, `two_shot_all_reduce_`, `multimem_all_reduce_`, `multimem_all_gather_out`, and variable-split all-to-alls named `all_to_all_vdev`, `all_to_all_vdev_2d`, and `all_to_all_vdev_2d_offset`. The page describes the all-to-alls as built for MoE dispatch and combine.

Two limits are worth stating. The page does not document a fused all-gather plus matmul operator, and this chapter does not name one. The page's matmul example places the output in symmetric memory and does not fuse communication into the compute. And the multimem operators need hardware support: the page says NVLink SHARP on NVIDIA GPUs, which the NVLS section below explains.

[Chapter 22](/c/22-overlap-and-parallelism) cites PyTorch's asynchronous tensor parallelism post, which builds on the `SymmetricMemory` abstraction to let GPUs write into each other's memory through peer-to-peer access. That is the production use of what you simulate here. The lab does not call these functions, because they need a GPU, and it does not depend on their signatures.

## Who moves the bytes

This section answers what hardware carries a transfer, and what each choice takes from the compute kernel.

Two engines can move bytes between GPUs. A *copy engine* is a DMA unit on the GPU that copies memory without running a kernel on the SMs. [Chapter 22](/c/22-overlap-and-parallelism) introduced it for host-to-device copies. An *SM-driven* transfer is a kernel in which threads issue loads and stores to a peer's mapped memory, or call a put from device code. The NVSHMEM `_block` and `_warp` variants are the second kind: a block or a warp of a running kernel moves the bytes.

NCCL's collectives are kernels. [Chapter 22](/c/22-overlap-and-parallelism) says so: an all-reduce on a side stream takes SMs from the compute stream's GEMM, and the two also share HBM bandwidth. That is the cost of SM-driven communication, and it applies whether the kernel is NCCL's or yours.

| Property | Copy engine | SM-driven |
|---|---|---|
| Takes SMs from compute | No | Yes, for the threads that move bytes |
| Launch per transfer | One host call or stream operation | None from inside a kernel; one launch if it is a standalone kernel |
| Can start from inside a compute kernel | No | Yes |
| Can fuse a reduction or a reorder with the transfer | No: it copies | Yes: the threads can add or reshape |
| Granularity that stays efficient | Large transfers, where the launch is amortized | Small transfers, per tile |

The lab's model makes the trade two numbers. `launch_s` is the cost of an extra kernel or stream operation per chunk, and the course's measurement from learn-inference is about 5 microseconds per launch call on the lab A100, per [docs/FACTS.md](../../docs/FACTS.md). `sm_steal` is the fraction of the compute rate lost while SMs communicate, and the lab assumes 0.10. That 10 percent is an assumed value, not a measurement. A copy engine sets `sm_steal` to 0 and pays `launch_s` per chunk. A fused kernel pays `sm_steal` and no launch.

Neither choice dominates. A large transfer that the compute kernel does not depend on belongs on a copy engine: the launch is a rounding error against the bytes, and the SMs stay on the GEMM. A transfer that a tile of the GEMM waits on, or that the GEMM produces tile by tile, belongs inside the kernel: a launch per tile would cost more than the tile.

## An all-gather from puts

This section answers how to build an all-gather out of the operations above, and what it costs compared with the ring.

Picture four PEs, each holding one chunk of five floats. The target is that every PE ends with all four chunks, 20 floats, in PE order. Each PE writes its chunk into a slot in every other PE's buffer. The slot is the chunk's origin: PE $s$'s chunk goes at element $s \times 5$. Because the heap is symmetric, the slot is the same offset on every PE, and the sender computes the destination address without asking.

Each sender then signals each receiver. A receiver needs to know, per source, that the chunk landed, so it has one signal word per source. PE $d$ waits on the words of the three PEs it expects data from, then reads its buffer. The steps for one source-destination pair are the put, the fence, and the signal from the previous section.

The count is small enough to check by hand. Four PEs make $4 \times 3 = 12$ puts and 12 signals. Each put carries 5 floats, 20 bytes, so each PE receives $3 \times 20 = 60$ bytes. That is $(p-1)/p$ of the 80-byte result, the same fraction the ring all-gather of [chapter 21](/c/21-collective-communication) moves per rank. The bytes are the same. What changes is the schedule. The ring sends to one neighbor in $p-1$ dependent steps, so it pays the latency term $(p-1)\alpha$. The put version sends to every peer at once, which on an all-to-all fabric such as NVSwitch has no dependent steps between transfers.

> [!INTUITION] The ring is a latency choice, not a bandwidth one
> Both schedules move $(p-1)/p$ of the result per rank. A ring keeps each link busy with one neighbor and pays for its steps in latency. Direct puts to every peer need a fabric where every pair can talk at full rate.

The reference all-gather the lab compares against is `torch.cat(chunks)`. Your puts are correct when every PE's buffer equals that concatenation under all three delivery policies. The lab also includes `all_gather_no_wait`, the plausible wrong version that reads its buffer without waiting on the signals. Under first-in first-out delivery nothing has arrived yet, so it reads zeros and fails the equality.

The consumer's wait is a spin. In the simulator, `wait_until` calls the network's `step` until the word has the value you want. When `step` has nothing left to deliver and the word still differs, no operation can change it, so the wait raises `SignalNotReady`. On a GPU, the same situation is a hang, and the simulator's exception is the debugging gift.

```python
def all_gather_with_puts(chunks, policy="fifo"):
    n, c = len(chunks), chunks[0].numel()
    heap = SymmetricHeap(n, n * c + n, policy)
    buf, sig = heap.malloc(n * c), heap.malloc(n)
    for pe in range(n):                       # each PE writes its own slot locally
        heap.mem[pe, buf + pe * c : buf + (pe + 1) * c] = chunks[pe]
    for src in range(n):
        for dst in range(n):
            if dst != src:
                heap.put(src, dst, buf + src * c, chunks[src])
                heap.fence(src)               # data ahead of the signal
                heap.signal(src, dst, sig + src, 1.0)
    outs = []
    for pe in range(n):
        for src in range(n):
            if src != pe:
                heap.wait_until(pe, sig + src, 1.0)
        outs.append(heap.get(pe, pe, buf, n * c))
    return outs
```

## Fusing an all-gather into a GEMM

This section answers how a GEMM tile loop consumes an all-gather as it arrives, and in what order the tiles become ready.

Start with the problem. In tensor parallelism, the activation $A$ is sharded by rows across PEs, and a layer needs the full $A$ times a weight shard $B_p$ on each PE. The unfused program all-gathers $A$ and then multiplies. The multiply waits for the whole gather, and the link idles during the multiply. That is the serial time in the model.

Cut $A$ into the $p$ chunks it already has: chunk $s$ is the rows PE $s$ owns. The product $A B_p$ is a stack of row blocks, and row block $s$ equals chunk $s$ times $B_p$. Row block $s$ needs only chunk $s$, so it can run as soon as chunk $s$ is local. The GEMM is a loop over tiles, and each tile depends on exactly one chunk.

That is a producer-consumer structure. The producers are the puts: each PE writes its chunk into every peer, fences, and signals. The consumer is the tile loop on each PE: before it multiplies chunk $s$, it waits on the signal for chunk $s$. In symbols, tile $i$ may start only when the signal of its chunk has been set:

$$
\hla{\mathrm{start}(\text{tile } i)} \;\ge\; \hlb{\mathrm{arrive}(\text{chunk } i)}
$$

Here $\hla{\mathrm{start}}$ is the time the tile's first load issues, and $\hlb{\mathrm{arrive}}$ is the time its chunk's signal becomes visible. The inequality is what the wait enforces. Without the wait, nothing enforces it, and a tile can read a chunk that is still on its way.

### The order the tiles become ready

The local chunk is ready at time zero, since it is already in the PE's own memory. Remote chunks arrive later. A good consumer starts with what is ready, and does not wait on a chunk in a fixed order that may not match arrival.

The lab fixes one order, a ring order. PE $r$ computes its own chunk first, then the chunks of PE $r+1$, $r+2$, and so on, modulo $p$. For four PEs the orders are:

| PE | Tile order |
|---|---|
| 0 | 0, 1, 2, 3 |
| 1 | 1, 2, 3, 0 |
| 2 | 2, 3, 0, 1 |
| 3 | 3, 0, 1, 2 |

The stagger has a purpose. If every PE started its remote work with chunk 0, every receiver would read from PE 0 first, and PE 0's outgoing link would be the hot spot. With the ring order, each PE's first remote chunk comes from a different source, so the sources send in parallel. Production kernels choose the order from the topology, and the principle is the same: compute what is local, then what arrives next.

The lab's `ag_gemm` logs each step as `("wait", pe, chunk)` or `("compute", pe, chunk)`. A PE's own chunk has no wait. The audit in the tests walks the log and flags any compute on a remote chunk with no earlier wait by that PE. A correct run produces no flags and an output equal to `torch.cat(shards) @ weights[pe]` on every PE. The wrong version, `eager_ag_gemm`, computes tiles in the same order without waiting. Its log has compute events with no waits, the audit flags them, and its output differs from the reference because the chunks it read were zeros.

```python
for chunk in tile_order(pe, n):
    if chunk != pe:
        heap.wait_until(pe, sig + chunk, 1.0)     # the tile starts only after its signal
        log.append(("wait", pe, chunk))
    tile = heap.get(pe, pe, buf + chunk * rows * k, rows * k).reshape(rows, k)
    out[chunk * rows : (chunk + 1) * rows] = tile @ weights[pe]
    log.append(("compute", pe, chunk))
```

On a GPU the structure is the same, with the pieces mapped onto hardware. Some thread blocks, or a warp group, play the producer and issue puts and signals. Other blocks play the consumer: each spins on its chunk's flag, with an acquire load so that reading the flag orders the later reads of the data, and then runs its tile. The consumer's wait is the `nvshmem_signal_wait_until` of the earlier table, or a spin on a signal-pad word in PyTorch's terms.

[Flux](https://arxiv.org/abs/2406.06858), which [chapter 22](/c/22-overlap-and-parallelism) cites, builds this on CUTLASS by mapping each tile of the GEMM and its communication to one thread block of a fused kernel. The authors argue that decomposing at the stream level leaves precise timing out of reach and underuses the GPU when the pieces are small. That claim is what the model below tests. Ring attention applies the same send-a-chunk-while-computing schedule to attention's key and value blocks. The companion course learn-inference teaches it, and this chapter stops here.

## In-switch reductions

This section answers what an NVLink switch can do that a plain link cannot.

A ring all-reduce adds partial sums on the GPUs, one hop at a time. On NVSwitch-based systems the switch itself can add. NVIDIA calls the feature NVLink SHARP, and the NCCL algorithm built on it NVLS. Each GPU writes its contribution toward a multicast address, the switch reduces the incoming values, and the result returns to the GPUs. The step count that feeds the latency term stops depending on $p$, and the SMs stop doing the additions, so the compute kernel keeps more of them. PyTorch's page ties this to its API: `multimem_all_reduce_` and `multimem_all_gather_out` exist, need NVLink SHARP on NVIDIA GPUs, and the handle's `multicast_ptr` is the address a kernel stores to.

This chapter gives no NVLS speedup number. The sources read for it give none in a form the course can cite, and a bytes-on-the-wire count depends on how the reduce-scatter and all-gather phases map onto the switch, which the chapter did not verify. [docs/FACTS.md](../../docs/FACTS.md) lists an NVLink bridge for two PCIe A100s and NVLink Switch 4 for 8-GPU systems, so the two-GPU lab cannot run it. In-switch reduction is prose and a pointer here.

## An alpha-beta model of overlap

This section answers when fusing communication into the compute kernel beats leaving it as separate collectives, and when neither beats the serial program.

The model prices one all-gather feeding one GEMM, per GPU. Start with the two totals. The all-gather moves $(p-1)/p$ of the gathered bytes $n$ over a link of bandwidth $B$, and pays $\alpha$ for each of its $p-1$ ring steps. The GEMM takes its FLOP count over the peak rate:

$$
\hla{T_{\mathrm{comm}}} = (p-1)\,\alpha + \frac{p-1}{p}\frac{n}{B},
\qquad
\hlb{T_{\mathrm{comp}}} = \frac{F}{P}
$$

$\hla{T_{\mathrm{comm}}}$ is the all-gather time and $\hlb{T_{\mathrm{comp}}}$ is the GEMM time at the peak rate $P$ for $F$ FLOPs. The serial program takes their sum.

Now cut the work into $c$ chunks and pipeline them. Each chunk's communication stage is followed by its compute stage, and chunk $j+1$'s communication overlaps chunk $j$'s compute. The time of a two-stage pipeline of $c$ chunks, with per-chunk communication $t_m$ and per-chunk compute $t_p$, is the first stage, plus $c-1$ rounds of the slower stage, plus the last compute:

$$
\boxed{T(c) = t_m + (c-1)\max(t_m,\, t_p) + t_p}
$$

This is the pipeline formula of [chapter 22](/c/22-overlap-and-parallelism), with two stages. The two execution styles differ in what goes into $t_m$ and $t_p$:

- **Overlapped, with separate collectives.** Every chunk is its own all-gather, so $t_m = (p-1)\alpha + \tfrac{p-1}{p}\tfrac{n/c}{B}$. Every chunk's GEMM is its own kernel, so $t_p = \tfrac{T_{\mathrm{comp}}}{c}(1+s) + \ell$, where $s$ is the SM share lost to communication and $\ell$ is the launch cost.
- **Fused.** One kernel signals each chunk, so $t_m = \sigma + \tfrac{p-1}{p}\tfrac{n/c}{B}$, where $\sigma$ is the latency of a put and its signal. There is no per-chunk launch, so $t_p = \tfrac{T_{\mathrm{comp}}}{c}(1+s)$.

The model has two assumptions to flag. The fused kernel pays one signal latency per chunk where a ring pays $p-1$ steps, because direct puts need no dependent steps. And the lab's $\alpha = 5$ µs comes from [docs/FACTS.md](../../docs/FACTS.md) as an assumed value, $\sigma = 2$ µs and $s = 0.10$ are the lab's assumed values, and $\ell = 5$ µs is the launch figure from learn-inference. The model is a way to reason about the trade. It is not a measurement of any kernel.

> [!EXAMPLE] A 128 MiB all-gather feeding a GEMM on 8 GPUs
> Take $p = 8$, NVLink 3 at $B = 300$ GB/s per direction, and 312 TFLOP/s dense bf16 from [docs/FACTS.md](../../docs/FACTS.md). The activation is $8192 \times 8192$ in bf16, so $n = 134{,}217{,}728$ bytes, which is 128 MiB. Each GPU multiplies it by an $8192 \times 1024$ weight shard, so $F = 2 \times 8192 \times 8192 \times 1024 = 1.374 \times 10^{11}$. `scripts/figures/36-symmetric-memory.py` prints every number below.
>
> The all-gather is $7 \times 5$ µs $= 35$ µs of latency plus $0.875 \times 134{,}217{,}728 / 300 \times 10^{9} = 391.468$ µs of transfer, so 426.468 µs. The GEMM is $1.374 \times 10^{11} / 312 \times 10^{12} = 440.509$ µs. The serial program takes 866.978 µs. The two halves are nearly equal, which is the case where overlap pays most.
>
> Eight chunks, separate collectives: each chunk's communication is $35 + 391.468/8 = 83.934$ µs. Each chunk's compute is $440.509/8 \times 1.1 + 5 = 65.570$ µs. The time is $83.934 + 7 \times 83.934 + 65.570 = 737.038$ µs. Fused with eight chunks: communication is $2 + 48.934 = 50.934$ µs and compute is $60.570$ µs, so $50.934 + 7 \times 60.570 + 60.570 = 535.494$ µs. The fused kernel is 1.62 times faster than serial and 1.38 times faster than the separate version.

The sweep shows how the chunk count moves the answer. Smaller chunks give the pipeline more rounds, so the exposed first stage shrinks. They also multiply the per-chunk fixed costs.

```viz
36-overlap-sweep
```

For this layer, the separate-collective version is fastest at 4 chunks among the powers of two the script tries, at 657.608 µs. Past that, its $(p-1)\alpha$ and launch per chunk dominate: at 64 chunks it takes 2644.04 µs, three times the serial time. The fused version keeps improving to 32 chunks, at 498.794 µs, because its per-chunk fixed cost is a 2 µs signal. Its floor is the compute time with the SM share added, $440.509 \times 1.1 = 484.560$ µs, and 32 chunks lands 14.2 µs above it.

Case B in the figure is a $256 \times 8192$ activation, 4 MiB gathered, against the same weight. The all-gather is 47.233 µs and the GEMM is 13.766 µs, so the serial time is 60.999 µs. At 4 chunks, the separate-collective version takes 161.019 µs, which is 2.6 times the serial time, and the fused version takes 24.019 µs. At 64 chunks even the fused version, at 140.47 µs, loses to serial. A small problem has so little work per chunk that any per-chunk fixed cost swamps it, and the best chunk count falls toward 1.

The model gives a rule for when each style wins:

1. **Serial wins** when the problem is so small that every per-chunk cost exceeds the work it hides. The lab's `overlap_model` returns `serial` for case A at one chunk and for case B at 64.
2. **Separate collectives win** over serial when the collective is large enough to amortize $(p-1)\alpha$ and a launch per chunk, and the compute is large enough to hide it. A moderate chunk count captures most of the gain.
3. **Fusion wins** when the chunk has to be small, because the dependent work is fine-grained or because a coarse chunk would leave a long exposed tail, and the per-chunk cost of a separate collective would be prohibitive. Its price is the SM share and the engineering of a fused kernel.

Neither style beats the larger of the two stage times. The bound is $\max(T_{\mathrm{comm}}, T_{\mathrm{comp}})$, which is 440.509 µs in case A. Overlap converts a sum into something near a maximum, and it can never go below the maximum.

## A motivating example: expert dispatch

A mixture-of-experts layer sends each token to the GPUs that hold its experts, an all-to-all whose sizes depend on the router. [Chapter 22](/c/22-overlap-and-parallelism) covers it as a collective. The one-sided view fits: a kernel on the sending GPU computes where each token goes and puts it there, with no host round trip to exchange counts. [DeepEP](https://github.com/deepseek-ai/DeepEP) is a library of such dispatch and combine kernels. Its README lists Hopper or newer GPUs and says NVSHMEM is no longer a dependency, so treat it as evidence that these kernels are worth writing at the level of puts and flags, not as a reference for NVSHMEM's API. PyTorch's `all_to_all_vdev_2d`, which its page describes as converting rank-major input to expert-major output for MoE dispatch, is the same idea. This chapter does not teach dispatch.

## What the two-GPU check measures

When the machine has two GPUs, the lab starts two NCCL processes and times an all-gather followed by a matmul two ways, with 2048 rows per rank, $K = 8192$, and 4096 output columns in bf16. The serial version gathers the whole activation and multiplies. The chunked version splits each rank's rows into four slices, issues four asynchronous all-gathers, and multiplies each slice after waiting on that slice's work handle, so the multiply on chunk $j$ runs while NCCL gathers chunk $j+1$.

This is the separate-collective style, not a fused kernel. NCCL runs on SMs, as [chapter 22](/c/22-overlap-and-parallelism) says, and the chunked version pays a launch per chunk. The lab ships no fused kernel, and the chapter predicts no time for one. [docs/FACTS.md](../../docs/FACTS.md) marks it unconfirmed whether the lab's two-GPU endpoint has an NVLink bridge. Without one, the link is PCIe Gen4 at 31.5 GB/s per direction and the transfer dominates.

Correctness is the hard check: the chunked output must match the serial output within a relative error of $2 \times 10^{-2}$ in bf16. Timing is a metric with a loose bound, that the chunked version takes at most 1.5 times the serial time. The check reports `gpu_serial_ms`, `gpu_overlap_ms`, and `gpu_overlap_speedup`. With fewer than two GPUs it prints a skip, records no timing, and passes, and a deployment without `RUNPOD_ENDPOINT_2XA100` does not send it to the single A100. This chapter quotes no measured speedup, because none has been run.

## What goes wrong

- **A signal without a fence.** Put, then signal, with nothing between. On a real fabric the signal usually arrives second, so the program passes its tests, and on another run it reads stale data. Put the fence between them, or use `nvshmem_put_signal` for that put's own data.
- **A quiet where a fence would do.** The program is correct and the issuing thread waits for the full round trip of every chunk. A fence gives the order without the wait.
- **A fence on the consumer.** A fence orders the calling PE's own operations. The producer must fence, and the consumer waits on the flag.
- **Reading before the wait.** A tile that reads a remote chunk before its signal reads zeros, or last iteration's data, and no error is raised. The lab's audit flags a compute event on a remote chunk with no earlier wait.
- **Different allocation sequences on different PEs.** Symmetry is a contract on the order and size of allocations. A PE that skips an allocation shifts every later offset, and puts land in the wrong object.
- **Reusing a flag or a buffer too early.** A flag still at 1 from the last iteration satisfies the next wait at once, and the next iteration's put can overwrite a chunk a slow tile still reads. Write an iteration counter into the flag, and double-buffer the destination.
- **Chunks too small.** Case B in the figure is the example: past a few chunks, per-chunk fixed costs exceed the work. Sweep the chunk count and keep the fastest.
- **Trusting an alpha API, or the model as a measurement.** Check the PyTorch page for your version before you copy a name. The latencies $\alpha$, $\sigma$, and $s$ are assumed, so the model ranks strategies and a measurement says how fast one is.

> [!RECAP]
> - A put or get is one-sided: only the initiator calls. A blocking put returns once its source is read, and delivery can come later. The target learns of data through a flag.
> - A symmetric heap puts every allocation at the same offset on every PE, so the address on PE $d$ is the base of PE $d$ plus the offset. `nvshmem_malloc` is collective and needs identical arguments.
> - A fence orders your earlier operations to a PE before your later ones, with no wait. A quiet waits for completion. Put, fence, signal keeps a flag behind its data.
> - A copy engine costs a launch and no SMs. An SM-driven transfer costs SMs and no launch, and can start from inside a kernel. A fused all-gather plus GEMM waits on each chunk's signal, and runs the local chunk first.
> - For a 128 MiB all-gather feeding a GEMM on 8 GPUs, the model predicts 866.978 µs serial, 737.038 µs overlapped with eight chunks, and 535.494 µs fused. A small problem, 4 MiB gathered, is faster serial or fused than with separate collectives. The latencies are assumed.
> - PyTorch's symmetric memory module is in alpha. In-switch reductions need NVLink SHARP, and the lab runs none.

## Check your understanding

> [!QUESTION] PE 0 puts three floats to PE 1, then signals PE 1, with no fence, and the network delivers signals first. What does PE 1 read when it sees the flag, and what changes it?
> PE 1 reads three zeros, or whatever was in the buffer before. Both operations share epoch 0, so nothing holds the signal back. A fence between the put and the signal raises the epoch, and the network may deliver the signal only after the put.

> [!QUESTION] What is the difference between `nvshmem_fence` and `nvshmem_quiet`, and which one belongs between a put and its signal?
> A fence orders your earlier operations to a PE before your later ones to the same PE and does not wait. A quiet waits until all your operations are complete and visible. The signal needs an order, so a fence is enough. A quiet is correct too, and it stalls the issuer.

> [!QUESTION] Element 5 on PE 2 sits at address $a$. Where is the same object on PE 6, with 4-byte elements and PE $p$'s base at $\text{base}_0 + p \cdot 2^{32}$?
> The heap offset is 20 bytes and does not change. The address is $\text{base}_0 + 6 \cdot 2^{32} + 20$, which is $a + 4 \cdot 2^{32}$.

> [!QUESTION] Four PEs each hold a 5-float chunk and gather with puts. How many puts and signals are there, and how many bytes does each PE receive?
> There are $4 \times 3 = 12$ puts and 12 signals. Each put is 20 bytes, so each PE receives 60 bytes, which is $3/4$ of the 80-byte result, the same $(p-1)/p$ fraction a ring all-gather moves per rank.

> [!QUESTION] In the fused all-gather plus GEMM, why does PE 2 of 4 compute chunk 2 first, and in what order does it take the rest?
> Chunk 2 is its own and is in local memory at time zero, so its tile needs no wait. The rest follow in ring order: 3, 0, 1. The stagger spreads the sources, so every PE's first remote read comes from a different PE.

> [!QUESTION] Eight chunks of the 128 MiB example, separate collectives: what are the per-chunk communication time, the per-chunk compute time, and the total?
> Communication is $35 + 391.468/8 = 83.934$ µs. Compute is $440.509/8 \times 1.1 + 5 = 65.570$ µs. The total is $83.934 + 7 \times 83.934 + 65.570 = 737.038$ µs, against 866.978 µs serial.

> [!QUESTION] The model says case B, a 4 MiB gather, is slower with four separate chunked collectives than serial. Why?
> The all-gather is 47.233 µs and the GEMM 13.766 µs. Four chunks pay $4 \times 35$ µs of ring latency and four launches, more than the work they hide: 161.019 µs against 60.999 µs serial. The fused version pays 2 µs per chunk and takes 24.019 µs. The latencies are the lab's assumed values.

> [!QUESTION] Why can no overlap strategy beat 440.509 µs in case A?
> Overlap hides the shorter stage behind the longer one, so the best case is the longer stage's time. The GEMM is the longer stage at 440.509 µs. Communication takes 426.468 µs. The fused version's floor is higher, 484.560 µs, because communicating SMs slow the GEMM by the assumed 10 percent.

## Lab

> [!TRY] A symmetric heap and a fused all-gather plus GEMM
> Implement `SymmetricHeap`, `all_gather_with_puts`, `tile_order`, `ag_gemm`, and `overlap_model` in `starter.py`. Everything runs on CPU tensors and needs no GPU.
>
> You pass the first check when `address(1, 3)` is the base plus $2^{32}$ plus 12. You pass the heap checks when `translate` swaps the base, `malloc` returns one shared offset and raises when the heap is full, a put arrives only after its issuer's quiet, and put, fence, signal keeps the data ahead of the signal under the `signals_first` policy while `signal_without_fence` does not. You pass the all-gather check when four PEs equal `torch.cat` under three delivery policies, and the chunked GEMM check when every PE equals `A @ B`, no tile reads an unsignaled chunk, and each PE starts with its own chunk. You pass the model check when the four reference cases, with 866.978, 737.038, and 535.494 µs for case A at eight chunks, match to 0.002 µs and pick the right winner.
>
> A separate check runs only when the machine has two GPUs. It times a serial and a chunked NCCL all-gather plus matmul, requires matching output and a chunked time of at most 1.5 times the serial one, and reports `gpu_serial_ms`, `gpu_overlap_ms`, and `gpu_overlap_speedup`. On any other machine it prints a skip and records no timing. Allow a minute for the two-GPU check's process startup.

The slide deck is [Symmetric memory, NVSHMEM, and fused communication](/slides/36-symmetric-memory-and-nvshmem).

## Further reading

- [NVSHMEM remote memory access](https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html): `nvshmem_putmem`, `nvshmem_getmem`, the nonblocking forms, the `_block`, `_warp`, and `_on_stream` variants, and the statement that delivery order is not guaranteed.
- [NVSHMEM signaling and ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html) and [ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html): `nvshmem_put_signal`, `NVSHMEM_SIGNAL_SET` and `NVSHMEM_SIGNAL_ADD`, and the exact guarantees of `nvshmem_fence` and `nvshmem_quiet`.
- [NVSHMEM memory management](https://docs.nvidia.com/nvshmem/api/latest/gen/api/memory.html): `nvshmem_malloc` as a collective call with identical arguments and an implied barrier.
- [PyTorch symmetric memory](https://docs.pytorch.org/docs/2.14/symmetric_memory.html): `empty`, `rendezvous`, `get`, signal pads, the `torch.ops.symm_mem` operators, and the alpha-state warning. Read it for your version before you write code.
- [Flux](https://arxiv.org/abs/2406.06858) and [PyTorch async tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487): communication overlap inside and around matmul kernels, as chapter 22 summarizes them.
- [DeepEP](https://github.com/deepseek-ai/DeepEP): expert-parallel dispatch and combine kernels. Its README says NVSHMEM is no longer a dependency.
- [GPU MODE, lecture 67: NCCL and NVSHMEM](https://github.com/gpu-mode/lectures), Jeff Hammond. The lecture index lists this title. This chapter did not read the lecture.
- [GPU MODE, lecture 78: Iris, Multi-GPU Programming in Triton](https://github.com/gpu-mode/lectures), Muhammad Awad, Muhammad Osama, and Brandon Potter, and [lecture 79: Mirage (MPK), Compiling LLMs into Mega Kernels](https://github.com/gpu-mode/lectures), Mengdi Wu and Xinhao Cheng. The lecture index lists these titles. This chapter did not read the lectures.
- [Chapter 21](/c/21-collective-communication), [chapter 22](/c/22-overlap-and-parallelism), and [chapter 25](/c/25-ddp-and-zero): the cost model, the overlap section this chapter expands, and the bucketed overlap you write on gloo. The companion course, learn-inference, owns ring attention and context parallelism.
