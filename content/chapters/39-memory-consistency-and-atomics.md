---
title: Memory consistency and atomics
slug: 39-memory-consistency-and-atomics
part: "Part 2 — CUDA patterns"
summary: Amdahl's law and work/span for a reduction and a scan, why a GPU needs a memory model, the PTX scopes and semantics behind cuda::atomic_ref and __threadfence, message passing and litmus tests, a lock-free CAS structure, and a grid barrier from atomics next to cooperative groups.
minutes: 150
gpu: true
objectives:
  - "Apply Amdahl's law and Brent's bound to a reduction and to a scan, and say when Kogge-Stone's bound beats Blelloch's."
  - "Explain why a GPU needs a memory model: ordinary loads and stores can reorder and need not become visible to another SM in program order."
  - "Name the PTX scopes (cta, cluster, gpu, sys) and the semantics (relaxed, acquire, release, acq_rel), and map them to cuda::thread_scope and cuda::memory_order."
  - "Write flag-and-data message passing with a release store and an acquire load, or with fences, and say why a relaxed flag store does not order the data."
  - "Read the message-passing, store-buffering, and load-buffering litmus tests and say which outcomes sequential consistency, relaxed accesses, release and acquire, and fences each allow."
  - "Write a lock-free push and pop with compare-and-swap, and explain the ABA problem."
  - "Build a reusable grid barrier from a generation counter, and explain why cooperative_groups grid sync needs a cooperative launch with every block resident."
lab: 39-memory-model
---

# Memory consistency and atomics

> [!TLDR]
> - Amdahl's law caps the speedup of a program with serial fraction $s$ at $1/s$. Work and span give the sharper tool: a greedy schedule on $p$ processors takes at most $W/p + S$ steps, for work $W$ and span $S$. For $2^{20}$ values a tree reduction has $W = 1{,}048{,}575$ and $S = 20$.
> - A GPU doesn't promise that two threads see each other's writes in program order. The PTX memory model says exactly what it does promise, in terms of *scopes* (`cta`, `cluster`, `gpu`, `sys`) and *semantics* (`relaxed`, `acquire`, `release`, `acq_rel`).
> - To pass data with a flag, store the data, then store the flag with release semantics. Load the flag with acquire semantics, then load the data. A relaxed flag store orders nothing, and a release store without a matching acquire load orders nothing either.
> - `cuda::atomic_ref<T, Scope>` and `cuda::memory_order` are the C++ spelling of those PTX rules. `__threadfence()` is a device-scope fence, and `volatile` makes a load or store happen but orders nothing else.
> - A grid barrier built from a generation counter is correct only when every block is resident, because a resident block waits for a block that can't start. `cooperative_groups` grid sync gets that guarantee from a cooperative launch.

[Chapter 6](/c/06-reductions) ended its reduction with a pattern that you copied without a proof: write the partial sum, call `__threadfence()`, then increment a counter. [Chapter 7](/c/07-scan-and-histogram) went further and let one block wait inside a kernel for another block's result. Both depend on a promise that the hardware makes under conditions this course hasn't stated yet. This chapter states them.

The failures are the worst kind. A kernel with a missing release passes every test on a quiet GPU and returns a stale value once in a million launches. A barrier built from a counter works for one round and hangs on the second. Neither shows up in a profile, and neither fails the same way twice.

The chapter also reaches back to the CPU course material that CMU's 15-418 and Stanford's CS149 build on: Amdahl's law, work and span, memory consistency, and lock-free structures. It keeps what carries over to a GPU. It skips cache coherence protocols, because a GPU's ordinary loads aren't kept coherent across SMs and the model is written to work without it. The lab's CPU checks are pure Python and run anywhere. Two kernel checks run only on a GPU.

## Before you start

**Atomics.** An atomic operation reads, modifies, and writes one memory location as a single step that no other thread can interleave with. [Chapter 6](/c/06-reductions) used `atomicAdd` to combine partial sums, and [chapter 7](/c/07-scan-and-histogram) used it to number tiles. This chapter adds the question chapter 6 left open: what other writes does an atomic make visible?

**Resident blocks.** A block is *resident* on an SM from the moment the SM accepts it until it finishes, and a resident block is never paused to make room. [Chapter 2](/c/02-the-execution-model) computes how many fit: 864 blocks of 256 threads on the A100's 108 SMs when the kernel uses at most 32 registers per thread. The barrier section depends on this number.

**Work and depth.** *Work* is the total number of operations, and *depth*, also called *span*, is the longest chain of dependent operations. [Chapter 7](/c/07-scan-and-histogram) defines both for a scan. This chapter turns them into a bound.

**Decoupled look-back.** [Chapter 7](/c/07-scan-and-histogram) builds a one-pass scan in which a block waits for flags that earlier blocks publish. This chapter explains the memory-ordering rules that protocol relies on and doesn't teach the protocol again.

## What parallel speedup can reach

This section answers how much faster $p$ processors can make a program, and what limits the answer. You can compute the limit before you write any kernel.

Picture a program as a line of work with a gate in it. Some stretch of the line must run one step at a time, and the rest splits across any number of workers. Adding workers shrinks the stretch that splits. It never shrinks the stretch that doesn't.

Say a fraction $\hla{f}$ of the one-processor running time can use $p$ processors, and the remaining $1-f$ can't. The time on $p$ processors is $(1-f) + f/p$ of the original, so the speedup is:

$$
\boxed{\text{speedup} = \frac{1}{(1-\hla{f}) + \hla{f}/\hlb{p}}}
$$

Here $\hla{f}$ is the parallel fraction and $\hlb{p}$ is the processor count. As $p$ grows without bound, the speedup approaches $1/(1-f)$. That is *Amdahl's law*.

> [!EXAMPLE] A program that is 95% parallel
> With $f = 0.95$ and $p = 8$, the speedup is $1/(0.05 + 0.11875) = 5.926$. With $p = 108$, one per SM on the A100, it is $1/(0.05 + 0.008796) = 17.01$. With $p = 221{,}184$, the resident threads of chapter 2's example, it is 19.998. The limit is $1/0.05 = 20$, and the last 3 points of speedup cost 220,000 processors. A 99% parallel program on 108 processors reaches 52.17, not 99.

Amdahl's law treats the serial fraction as given. For an algorithm you can say more, because the serial part is the longest chain of dependent steps, and you can count it. This is the *work and span* model. The *work* $\hla{W}$ is the number of operations, and the *span* $\hlb{S}$ is the length of the longest dependent chain. One processor takes $W$ steps. Infinitely many processors take $S$ steps.

*Brent's bound* connects them. A greedy scheduler never leaves a processor idle while an operation is ready. On $p$ processors it takes at most:

$$
\boxed{T_p \le \frac{\hla{W}}{p} + \hlb{S}}
$$

No scheduler beats $\max(W/p,\, S)$, so the true time sits between $\max(W/p, S)$ and $W/p + S$. The bound is within a factor of two of optimal, whatever the algorithm. You can read it as Amdahl's law with the serial fraction replaced by something you can count.

### Apply it to a reduction and a scan

A tree reduction of $n$ values does $n - 1$ additions, and its span is $\lceil \log_2 n \rceil$ steps ([chapter 6](/c/06-reductions)). For $n = 2^{20}$ that is $W = 1{,}048{,}575$ and $S = 20$.

Chapter 7 gives the two scan networks. Kogge-Stone does $n - 2^k$ additions at step $k$ and takes $\log_2 n$ steps. For $n = 2^{20}$ its work is $\sum_{k=0}^{19}(2^{20} - 2^k) = 20 \cdot 2^{20} - (2^{20} - 1) = 19{,}922{,}945$ and its span is 20. Blelloch does $2(n-1) = 2{,}097{,}150$ additions in $2 \log_2 n = 40$ steps. Kogge-Stone does 9.5 times Blelloch's work for half its span.

Set the two bounds equal to find where Kogge-Stone wins:

$$
\frac{19{,}922{,}945}{p} + 20 = \frac{2{,}097{,}150}{p} + 40 \implies p = \frac{17{,}825{,}795}{20} \approx 891{,}290
$$

Below about 891,290 processors, Blelloch's bound is lower. On the A100 at 221,184 resident threads, the bounds are 24.74 steps for the reduction, 110.07 for Kogge-Stone, and 49.48 for Blelloch. Those figures are arithmetic and not times, since a step has no duration here.

```viz
39-brent-bound
```

Drag $p$ in the figure and watch the lines. Work dominates at small $p$, so the algorithm with less work wins. Span dominates at large $p$, so the algorithm with the shorter chain wins. Chapter 7 found the same crossover inside one block, where a barrier costs more than an addition.

### What the bound leaves out

Work and span count operations and ignore memory. A reduction of $2^{26}$ floats is limited by the 0.2105 ms it takes to read 256 MiB, as [chapter 6](/c/06-reductions) computed, and its 20 steps of span are irrelevant to that floor. Use work and span to choose between algorithms that move the same bytes, such as the two scans. Use the roofline from [chapter 4](/c/04-roofline-and-latency) to find out whether memory decides the time at all.

The bound also assumes that the processors can communicate for free. The rest of this chapter is about what communication costs, in correctness rather than time: when one thread writes a value, which threads see it, and when.

## Why a GPU needs a memory model

This section answers a question that sounds like it has an obvious answer: if one thread writes a value and another reads it afterward, why doesn't the second thread see it?

Picture two blocks on two SMs. Block A writes `data = 42` and then `flag = 1`. Block B spins until it reads `flag == 1`, then reads `data`. In a single-threaded program that order is a guarantee. Between two SMs it isn't, for two separate reasons.

**The compiler reorders and caches.** An optimizer is free to treat a plain load as something that can't change underneath it. It hoists the load of `flag` out of the spin loop, because no code in the loop writes `flag`, and the loop never ends. It also reorders the two stores, because they touch different addresses and one thread can't tell. Both transformations are legal for a program with no other threads.

**The hardware reorders and buffers.** Each SM has its own L1 cache, and a load that hits it doesn't consult other SMs. A store can wait in a buffer while a later store to a different address goes ahead. The CUDA programming guide describes the visible effect: a fence changes the *order* of memory operations, and it takes `volatile` to make a load or store actually reach memory ([CUDA Programming Guide, C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)).

A *memory consistency model* is the contract that says which orders a program can observe. It has to be weak enough that the hardware can reorder and buffer, because that is where the speed comes from, and strong enough that you can write a correct flag. The contract for NVIDIA GPUs is the memory consistency model chapter of the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), and the C++ memory model that CUDA offers maps onto it. Lustig, Sahasrabuddhe, and Giroux give the formal analysis in [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model).

The PTX chapter doesn't describe caches. It defines what a program can observe and leaves the machinery unstated, so the cache picture above is an explanation of why the contract is weak. Don't use it to predict a result. Use the rules in the next section.

> [!KEY] Without a synchronizing operation, order and visibility are both unspecified
> A plain store followed by a plain store to another address can become visible in either order. A plain load in a loop can be executed once. A memory model names the operations that remove that freedom, and you pay for exactly the ordering you ask for.

## Scopes and semantics in PTX

This section answers how PTX says who must see an operation and what order they must see it in. You choose both, separately, on every synchronizing operation.

### Scope: which threads

A *scope* is the set of threads that an operation is ordered with respect to. PTX defines four. The warp isn't one of them ([PTX ISA, memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)).

| PTX scope | Threads it covers | libcu++ name |
|---|---|---|
| `.cta` | The threads of the same block | `cuda::thread_scope_block` |
| `.cluster` | The threads of the same thread block cluster | `cuda::thread_scope_cluster` |
| `.gpu` | The threads on the same device | `cuda::thread_scope_device` |
| `.sys` | Every thread in the program, host threads and other devices included | `cuda::thread_scope_system` |

A cluster exists on Hopper and later, and libcu++ widens `cluster` to `device` before sm_90 ([libcu++ memory model](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/memory_model.rst)). The A100 has no clusters, so the lab uses `gpu` scope, written `thread_scope_device` in C++. A wider scope costs more, because the operation has to be visible to more of the machine. Pick the narrowest scope that includes every thread that communicates.

### Semantics: which order

The *semantics* of a memory operation say what it orders against other operations of the same thread. PTX lists the following ([PTX ISA, memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)).

| Semantics | What it means |
|---|---|
| `.weak` | A plain `ld` or `st`. No ordering, and a data race if another thread writes the location. |
| `.relaxed` | Atomic at its scope, and nothing else. It orders nothing relative to other locations. |
| `.acquire` | A load that no later operation of this thread can move before. |
| `.release` | A store that no earlier operation of this thread can move after. |
| `.acq_rel` | Both, for a read-modify-write such as `atom`, and for a fence. |
| `.volatile` | Treated like a relaxed operation at system scope, with extra constraints. |

A *fence* is a memory operation that orders without reading or writing. PTX has `fence.acq_rel`, which gives release-then-acquire ordering, and `fence.sc`, which also puts every `fence.sc` into one total order that all threads agree on. That last property is what store buffering needs, as a later section shows.

An acquire load or release store carries a scope in its name, for example `ld.acquire.gpu` and `st.release.gpu`. The instruction forms are in the PTX ISA. Which compute capability introduced each form isn't something this chapter confirms, so the lab only needs the A100 to accept the C++ spelling in the next section.

### When a release meets an acquire

A release store and an acquire load create ordering only when they meet. PTX's definition is built from *morally strong* operations: two operations are morally strong when each is strong, each one's scope includes the other's thread, they use the same proxy, and they cover exactly the same bytes. A *release pattern* is a release operation, or a release fence followed by a strong write. An *acquire pattern* is an acquire operation, or a strong read followed by an acquire fence. When an acquire pattern reads a value that a release pattern wrote, the release *synchronizes with* the acquire, and every write the releasing thread made before the release is visible to every read the acquiring thread makes after the acquire.

Two parts of that definition catch people. The first is scope: a release at `cta` scope read by an acquire in another block doesn't synchronize, because the acquirer isn't in the releaser's scope. libcu++ says it plainly. A block-scoped release store read by a device-scoped acquire load in another block is a data race, which is undefined behavior ([libcu++ memory model](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/memory_model.rst)). The second is that both halves must exist. A release with no acquire on the reading side orders nothing for the reader, and the lab's tests make you demonstrate that.

## The C++ spelling

This section answers how you write those rules in CUDA C++. Use `cuda::atomic_ref` and a fence from libcu++, and treat `volatile` and the older fence intrinsics as legacy spellings of a weaker idea.

`cuda::atomic_ref<T, Scope>` from `<cuda/atomic>` wraps an existing object so that accesses through it are atomic at the scope you name. `Scope` is a `cuda::thread_scope` and defaults to `thread_scope_system`. It offers the standard `atomic_ref` operations plus `fetch_min` and `fetch_max`. The type must be at most 16 bytes, the object must be aligned to its size, and the device must be Pascal (compute capability 6.0) or newer. Types smaller than 4 bytes are emulated and much slower, and an object must not be referenced at the same time by `atomic_ref`s with different scopes ([libcu++ atomic_ref](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/synchronization_primitives/atomic_ref.rst)).

```cuda
#include <cuda/atomic>

__device__ void publish(int* data, int* flag, int value) {
    cuda::atomic_ref<int, cuda::thread_scope_device> f(*flag);
    data[0] = value;                                // plain store of the payload
    f.store(1, cuda::std::memory_order_release);    // st.release at gpu scope
}

__device__ int consume(const int* data, int* flag) {
    cuda::atomic_ref<int, cuda::thread_scope_device> f(*flag);
    while (f.load(cuda::std::memory_order_acquire) == 0) { }   // ld.acquire at gpu scope
    return data[0];                                 // sees the payload
}
```

The memory orders are the C++ ones: `relaxed`, `acquire`, `release`, `acq_rel`, and `seq_cst`. The libcu++ memory model page names `memory_order_release` and `memory_order_acquire` in its examples, and the lab's CUDA source writes `cuda::std::memory_order_*`. This chapter doesn't claim a particular instruction for each pairing on the A100. Read the SASS ([chapter 5](/c/05-profiling)) if the exact instruction matters to you.

### Fences and `volatile`

`__threadfence_block()`, `__threadfence()`, and `__threadfence_system()` are the older fence intrinsics, at block, device, and system scope. The programming guide describes them by the write ordering they give: after `__threadfence()`, no device thread observes the writes made after the call as happening before the writes made before it. The page recommends `cuda::atomic_thread_fence` with the matching scope instead ([CUDA Programming Guide, C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)). Which PTX fence `__threadfence()` compiles to isn't stated on that page, so this chapter treats it as a device-scope fence and relies only on the ordering the guide describes.

A fence orders. It doesn't make anything visible by itself, and the guide says that visibility is what `volatile` is for. `volatile` makes the compiler issue the access every time and not cache it in a register, which is why chapter 7's look-back sketch reads the status slot through a `volatile` pointer. In PTX terms it's close to a relaxed system-scope access. That is enough to stop the spin loop from collapsing into one load, and it isn't enough to order the payload that the flag guards.

| You write | What you get |
|---|---|
| A plain load or store | No ordering, and a data race if two threads touch the location |
| `volatile` | The access happens each time. Nothing else is ordered |
| `atomic_ref` with `relaxed` | Atomic at its scope. Nothing else is ordered |
| `atomic_ref` with `release` or `acquire` | The ordering of the previous section, at the scope you chose |
| A fence plus a relaxed access | The same ordering, at the cost of ordering everything around it |

A `__threadfence()` after the payload and before a relaxed flag store is correct, and it orders every earlier write, not only the payload. A release store orders the same writes, so the two forms cost about the same in a kernel that writes one flag. Choose the release store when you can, because it states which store is the flag.

## Message passing

This section answers the question the whole chapter hangs on: how do you hand data from one thread to another with a flag, and what exactly goes wrong if you leave out a half?

The picture is two threads and two locations. The *producer* writes `data`, then `flag`. The *consumer* reads `flag`, and if it sees 1, reads `data`. The consumer expects 42. This is the *message passing* pattern, MP for short, and chapter 6's last-block reduction and chapter 7's look-back are both instances.

In words, the pattern needs the producer's two stores to become visible in order, and the consumer's two loads to happen in order. The producer's half is a release on the flag. The consumer's half is an acquire on the flag. Each half alone leaves the other thread free to reorder.

Take the smallest case, with `data` and `flag` both starting at 0. The producer runs `data = 42; flag = 1`. The consumer runs `r0 = flag; r1 = data`. If every access is relaxed, hardware or compiler can swap the producer's stores, or the consumer's loads, and the consumer can finish with `r0 = 1, r1 = 0`: it saw the flag and missed the data. The four possible pairs of final values are $(r_0, r_1) = (0,0), (0,1), (1,0), (1,1)$. Sequential consistency allows three of them. The pair $(1, 0)$ is the *weak outcome*.

Make the flag store a release and the flag load an acquire, and the PTX model forbids $(1, 0)$. The producer's release can't move before the data store. The consumer's acquire can't move after the data load. When the consumer's load returns the value the producer's store wrote, the release synchronizes with the acquire, and the data store is visible. The consumer finishes with $(0,0)$, $(0,1)$, or $(1,1)$ and never $(1,0)$.

```cuda
// Producer, one thread of block A
data[i] = 42;
flag_ref.store(1, cuda::std::memory_order_release);

// Consumer, one thread of block B
int r0 = flag_ref.load(cuda::std::memory_order_acquire);
int r1 = (r0 == 1) ? data[i] : -1;   // r1 is 42 whenever r0 is 1
```

### The fence version

The PTX specification gives the pattern with fences. T1 writes `data`, issues `fence.sys`, and does a relaxed `.sys` write to `flag`. T2 does a relaxed `.sys` read of `flag`, issues `fence.sys`, and reads `data`. The specification states that if T2 reads 1 from `flag`, it reads 1 from `data` ([PTX ISA, memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)). The programming guide has the same shape in CUDA C++: a writer that stores `X`, fences, and stores `Y`, and a reader that loads `Y`, fences, and loads `X`, so that the reader can't see the new `Y` with the old `X`.

```cuda
// Writer                           // Reader
data[i] = 42;                       int r0 = flag_volatile;
__threadfence();                    __threadfence();
flag_volatile = 1;                  int r1 = (r0 == 1) ? data[i] : -1;
```

Both fences are needed, and neither is there to make a value visible. The writer's fence orders the data store before the flag store. The reader's fence orders the flag load before the data load. Chapter 6's last-block reduction has the writer's fence and reads the partials with `__ldcg`. A release on the counter increment and an acquire on the read of it state the same hand-off in the model's own terms.

### What the half-pattern loses

The lab asks you to prove each half is needed, with the enumerator you write. The following four variants of MP give the following sets of allowed outcomes, where $(1,0)$ is the one you want to rule out.

| Flag store | Flag load | Allows $(1, 0)$? |
|---|---|---|
| relaxed | relaxed | Yes |
| release | relaxed | Yes, the consumer's loads reorder |
| relaxed | acquire | Yes, the producer's stores reorder |
| release | acquire | No |

A relaxed flag store doesn't order the data store before it. That is the single most common mistake with flags on a GPU, because a relaxed atomic looks like synchronization: it is atomic, it is visible, and it carries no ordering for anything else.

## Litmus tests

This section answers how to tell what a memory model allows without running a GPU a billion times. A *litmus test* is a tiny program, two or three threads and a few locations, together with a question about its final register values. The answer is a set of outcomes the model allows.

The three classic tests each pit one pair of reorderings. In each, every location starts at 0, and $r_0$ and $r_1$ are registers.

| Test | Thread 0 | Thread 1 | Outcome to rule out | What reorders to cause it |
|---|---|---|---|---|
| MP, message passing | `x = 1; y = 1` | `r0 = y; r1 = x` | $r_0 = 1, r_1 = 0$ | Stores swap, or loads swap |
| SB, store buffering | `x = 1; r0 = y` | `y = 1; r1 = x` | $r_0 = 0, r_1 = 0$ | A load passes an earlier store |
| LB, load buffering | `r0 = x; y = 1` | `r1 = y; x = 1` | $r_0 = 1, r_1 = 1$ | A store passes an earlier load |

Under sequential consistency, one global order of all accesses that respects each thread's program order, each rule-out holds. In SB, one of the two stores comes first in the global order, so the other thread's load sees it. That makes at least one of $r_0, r_1$ equal to 1. Under a weaker model each outcome becomes possible when the right pair reorders.

```viz
39-litmus
```

The figure runs the lab's enumerator. Switch the model and watch the amber square. The sets in the figure come from a simplified model, one shared memory in which an access takes effect at once and the only freedom is reordering. It isn't the PTX axiomatic model, and it omits scopes, proxies, and the other behaviors a real GPU's memory system can show.

### What each fix removes

Release and acquire fix MP. A release store can't move before earlier accesses, and an acquire load can't move after later ones, so the weak outcome is gone. They also fix LB: with acquire loads and release stores, no thread's store can pass its own earlier load, so both threads can't read the other's write.

They don't fix SB. A release store orders what is *before* it, and an acquire load orders what is *after* it. In SB the load comes after the store, and the load may still move before it, so $(0, 0)$ remains. The C++ memory model has the same property: SB needs sequential consistency for both accesses or a full fence. In PTX it is `fence.sc`. The specification says that if `fence.sc` is replaced by `fence.acq_rel` in its store-buffering test, the outcome where both reads see 0 isn't guaranteed ([PTX ISA, memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)).

For LB, PTX deliberately permits $(1, 1)$ when the loads and stores have no dependency between them, and it forbids the outcome when a true data dependency links each load to the later store, so values can't appear from nowhere. The lab's model has no dependencies, so it allows the outcome for relaxed accesses and removes it with acquire loads and release stores, which is stricter than relaxed alone.

> [!NOTE] A GPU run can't prove an outcome impossible
> The lab's CUDA kernel runs MP millions of times and counts $(1, 0)$. A count of zero with a release and an acquire is what the model predicts, and it is evidence. A count of zero with relaxed accesses proves nothing, since the weak outcome is allowed and may be rare. The lab reports the relaxed count as a metric and never requires it to be nonzero. Alglave and colleagues' paper on GPU weak behaviours (see Further reading) tested real GPUs this way. Use the model to decide what's allowed, and use the hardware only to look for a counterexample.

## Atomics and a lock-free structure

This section answers how a group of threads can update a shared structure without a lock. The tool is *compare-and-swap*, CAS: `compare_exchange(expected, desired)` writes `desired` only if the location still holds `expected`, and reports whether it did. A CAS loop reads the location, computes a new value, and retries if another thread got there first.

Picture a stack of nodes, kept in an array so that a node is an index and `-1` is null. The stack's `head` holds the index of the top node. Pushing node $n$ points $n$'s `next` at the current head, then CAS-es the head from that value to $n$. If another thread pushed in between, the CAS fails, reloads, and tries again.

```cuda
struct Stack {
    int head;      // index of the top node, or -1
    int next[N];   // next[i] is the node below i
};

__device__ void push(Stack* s, int node) {
    cuda::atomic_ref<int, cuda::thread_scope_device> head(s->head);
    int old = head.load(cuda::std::memory_order_relaxed);
    do {
        s->next[node] = old;                       // written before the CAS publishes node
    } while (!head.compare_exchange_weak(old, node,
                 cuda::std::memory_order_release,  // success: publish next[node]
                 cuda::std::memory_order_relaxed)); // failure: just reload old
}

__device__ int pop(Stack* s) {
    cuda::atomic_ref<int, cuda::thread_scope_device> head(s->head);
    int old = head.load(cuda::std::memory_order_acquire);
    while (old != -1 &&
           !head.compare_exchange_weak(old, s->next[old],
                 cuda::std::memory_order_acquire,
                 cuda::std::memory_order_relaxed)) { }
    return old;                                    // -1 when empty
}
```

The release on a successful push and the acquire on pop are message passing again. The data is `next[node]`, the flag is the head, and a pop that reads the new head must see the `next` that was written for it. A relaxed CAS here would let a popper read a stale `next[node]` and follow it into the wrong node.

`compare_exchange_weak` may fail even when the value matches, which is why it sits in a loop. On failure it writes the value it found into `old`, so the loop needs no separate reload.

### The ABA problem

The pop above has a flaw that no memory order fixes. Say a pop reads `old = A` and `s->next[A] = B`, then stalls. Other threads pop `A`, pop `B`, and push `A` back, so the stack is now `A` over some other node `C`. The stalled pop wakes, sees the head is still `A`, and its CAS succeeds. It installs `B` as the head, but `B` is no longer in the stack. The value went from A to B to A, and CAS can't tell. This is the *ABA problem*.

The standard fix is a version tag. Pack a counter beside the index into one 64-bit word, `{index, version}`, and bump the version on every successful push and pop. A CAS on the pair fails when the version moved, even if the index came back. `atomic_ref` supports 8-byte objects at any scope, so the pair fits ([libcu++ atomic_ref](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/synchronization_primitives/atomic_ref.rst)). The alternative is never to reuse a node while any thread might still hold its index, which a kernel with a fixed array of nodes can arrange by allocating each node once.

A GPU rarely wants this structure. Thousands of threads CAS one head, one wins per round, and the rest retry, so the structure serializes. A lock-free stack is the right idea for a bounded number of producers, and the wrong one for a warp-wide push. When every thread needs a slot, a single `atomicAdd` on a counter gives each a distinct index with no retry, and that is what chapter 7 does to number tiles.

## A grid-wide barrier

This section answers how to make every block in a kernel wait for every other block, and why this needs a guarantee that an ordinary launch doesn't give.

`__syncthreads()` synchronizes one block. Nothing in the launch syntax synchronizes a grid, because blocks are scheduled as resources free up and a block that hasn't started can't wait. Splitting a kernel into two launches is the usual barrier, at about 5 µs per launch on the lab A100 ([learn-inference chapter 12](https://qwen.ifkash.dev/c/12-your-first-cuda-kernel)). Inside one kernel, a persistent kernel that loops over phases needs a barrier of its own.

Picture a turnstile that lets the crowd through only when the last person arrives. Each block adds one to an arrival counter. The last block to arrive opens the turnstile, and the others spin until it opens. To use it again, the counter must go back to 0, and that is where the design can fail.

In words, the barrier needs three things. A block reads the *generation* number before it arrives, so it can tell when its own round has been released. The last block resets the arrival counter, then adds one to the generation, which releases everybody. Waiters spin on the generation, never on the counter. Here is the barrier with $N$ blocks in $\hla{\text{count}}$ and $\hlb{\text{gen}}$:

```cuda
__device__ void grid_barrier(unsigned* count, unsigned* gen, unsigned num_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> c(*count);
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> g(*gen);
        unsigned my_gen = g.load(cuda::std::memory_order_acquire);        // 1. read the generation first
        if (c.fetch_add(1, cuda::std::memory_order_acq_rel) == num_blocks - 1) {
            c.store(0, cuda::std::memory_order_relaxed);                  // 2. reset: nobody can re-arrive yet
            g.fetch_add(1, cuda::std::memory_order_release);              // 3. release everyone
        } else {
            while (g.load(cuda::std::memory_order_acquire) == my_gen) { } // wait for the generation to move
        }
    }
    __syncthreads();
}
```

The memory orders do the same job as in message passing. Each block's `fetch_add` is `acq_rel`, so the last block acquires every earlier block's writes. Its release of the generation publishes all of them, together with the counter reset, to every waiter's acquire. Every block's writes from before the barrier are visible to every block after it.

### The classic reuse bug

Two plausible barriers fail, and the lab's simulator catches both.

The first waits on the counter. The last block resets `count` to 0, and the waiters spin until `count == 0`. On the first round it works. A fast block leaves the barrier and arrives at the next one, adding 1 to `count`, before a slow waiter has looked again. The slow waiter sees 1 and spins forever, because `count` will never be 0 again until the slow waiter's own block arrives. The counter was reset before every block had read it, and the barrier deadlocks on the second round.

The second has no generation. The last block sets a `go` flag, and waiters spin on it. Nothing clears the flag, so on the second round it is already 1 and every block passes immediately. A block reads a neighbor's data before the neighbor has written it. This version doesn't hang. It returns wrong answers.

The generation fixes both: a waiter compares against the value it read *before* arriving, so a stale release can't satisfy it and a reset can't hide it. The lab's `generation_barrier` is this algorithm, driven by a seeded scheduler that interleaves the simulated blocks one atomic operation at a time.

### Why every block must be resident

The barrier above is correct only if all `num_blocks` blocks are running at once. A block that spins on `gen` holds its SM slot until the barrier opens, and the hardware never evicts it ([chapter 2](/c/02-the-execution-model)). Suppose the grid has 8 blocks and room for 4. The first 4 arrive, then spin. The other 4 are waiting for a slot that no spinning block will give up. The barrier needs all 8, so it never opens. The lab's simulator reports this as a deadlock.

That is why `cooperative_groups`' grid synchronization needs a cooperative launch. `cg::this_grid().sync()` is a barrier like the one above, and it is only safe if the whole grid is resident. The programming guide says that synchronizing the entire grid requires the `cudaLaunchCooperativeKernel` runtime launch API, and that the launch is atomic: if the call succeeds, the requested number of blocks launches on the device ([CUDA Programming Guide, Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)). A grid that can't be placed whole fails the launch call instead of hanging. Query `cudaDevAttrCooperativeLaunch` with `cudaDeviceGetAttribute` to check that the device supports it.

```cuda
int per_sm = 0, device = 0, sms = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, 0);
cudaGetDevice(&device);
cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
int max_blocks = per_sm * sms;      // the largest grid a cooperative launch accepts

void* args[] = { &count, &gen, &flags, &bad, &rounds };
cudaLaunchCooperativeKernel((void*)kernel, max_blocks, threads, args, 0, stream);
```

For chapter 2's kernel of 256 threads and at most 32 registers per thread, that is $108 \times 8 = 864$ blocks. The lab computes `fits_cooperative_launch` for that case and checks that 864 blocks fit and 865 don't. On the A100 the lab runs the atomic barrier on twice the SM count, a grid it expects to be resident, and the cooperative barrier on the co-resident maximum.

Decoupled look-back avoids this requirement by a different route. A block takes its tile number from an atomic counter when it starts, so it only ever waits for tiles that already started ([chapter 7](/c/07-scan-and-histogram)). Its waiting is deadlock-free without a cooperative launch, and it relies on the hardware starting blocks in launch order. A grid barrier is the stronger tool: it waits for blocks that haven't started, so it needs residency.

## Check a litmus test on a CPU

This section answers how the lab's enumerator works, so you can write it. It is a search over executions of a small operational model, and it is about 40 lines.

The model has one shared memory and executes one instruction at a time, from any thread. Within a thread the instructions may run out of program order, subject to a rule that says which pairs must stay in order:

| Earlier | Later | Stays in order? |
|---|---|---|
| Anything | Anything, under sequential consistency | Yes |
| Either is a fence | | Yes |
| Same address | Same address | Yes |
| An acquire load | Anything | Yes |
| Anything | A release store | Yes |
| Anything else | | No |

An execution picks any thread's next ready instruction, where *ready* means every instruction it must stay behind has run. The search records the registers once every instruction of every thread has run. A set of seen states stops it from revisiting a state, which keeps three-instruction litmus tests to a few hundred states.

The model reproduces the table of this chapter. MP with relaxed accesses allows all four pairs, and with release and acquire it allows three. SB with release and acquire still allows $(0, 0)$, and with a fence it doesn't. LB with relaxed accesses allows $(1, 1)$, and with acquire loads and release stores it doesn't. The lab checks all of these, and it checks that the memory orders you choose for the CUDA kernel are ones your own enumerator proves safe.

The model is a teaching tool. It has no scopes, no multi-copy atomicity question, no dependencies, and no proxies. A real proof about a kernel needs the PTX model, or a tool built on it, and the lab's enumerator only shows you why release and acquire have the shape they do.

## What goes wrong

The mistakes below are the ones the lab and the A100 are most likely to expose.

- **A relaxed flag store.** `flag.store(1, relaxed)` is atomic and visible, and it doesn't order the data before it. The consumer can see the flag and read stale data. The fix is `release` on the store and `acquire` on the load.
- **Half a pair.** A release store read by a plain or relaxed load orders nothing for the reader, and an acquire load of a relaxed store orders nothing for the writer. Both sides need their half.
- **A scope that's too narrow.** A `thread_scope_block` release read by a thread in another block doesn't synchronize. It's a data race. Pick `thread_scope_device` for anything that crosses blocks.
- **`volatile` as synchronization.** `volatile` keeps the compiler from hoisting a load out of a spin loop. It doesn't order the payload. Use it for a spin that a fence or an acquire already protects, and nowhere else.
- **A fence on one side only.** The writer's fence orders its stores, and the reader's loads are still free to swap unless the reader fences or acquires.
- **A relaxed test that passes.** A kernel that never shows the weak outcome in a million runs may still be wrong. The model decides correctness, and a run can only look for a counterexample.
- **A barrier on the counter.** Spinning on the arrival counter deadlocks on round 2, and a flag that nothing clears lets round 2 through early. Read a generation before you arrive.
- **A barrier over a grid that doesn't fit.** More blocks than can be resident deadlocks, and a cooperative launch rejects the grid instead. Size the grid from the occupancy API.
- **ABA on a CAS.** A CAS that compares only an index can succeed on a recycled node. Tag the word with a version, or never reuse a node.
- **An atomic on a hot address.** All the threads of a grid CAS one location, and they take turns. If every thread needs a distinct slot, use one `atomicAdd` and let each take a different result.

> [!RECAP]
> - Amdahl's law bounds speedup by $1/(1-f)$, and Brent's bound, $T_p \le W/p + S$, refines it with work and span. Work and span choose between algorithms that move the same bytes, and the roofline says whether memory decides.
> - Plain loads and stores have no cross-thread order. A memory model, in PTX, gives each synchronizing operation a scope and a semantics.
> - A release store read by an acquire load synchronizes, and each half alone orders nothing. The same pairing is a pair of fences around relaxed accesses.
> - MP needs release and acquire, LB needs them too, and SB needs `fence.sc`. A run on a GPU can find a counterexample and never proves an outcome impossible.
> - A CAS structure needs release on publish, acquire on consume, and protection from ABA. A generation counter makes a barrier reusable, and every block of the grid must be resident or the barrier deadlocks.

## Check your understanding

> [!QUESTION] A program is 90% parallel. What speedup does Amdahl's law give on 108 processors, and what is the limit?
> $1/(0.1 + 0.9/108) = 1/(0.1 + 0.008333) = 9.231$. The limit is $1/0.1 = 10$. The 108 processors reach 92% of it, and no processor count can pass 10.

> [!QUESTION] A reduction of $2^{20}$ values has work 1,048,575 and span 20. What does Brent's bound give on 1,024 processors, and what is the lower bound?
> The upper bound is $1{,}048{,}575/1024 + 20 = 1{,}044.0$ steps. The lower bound is $\max(1{,}048{,}575/1024,\, 20) = 1{,}024$ steps. The true time lies between them, and at this $p$ work dominates, so the two are within 2%.

> [!QUESTION] Why does Kogge-Stone's Brent bound lose to Blelloch's at 1,024 processors and win at a million?
> Kogge-Stone does 19,922,945 additions to Blelloch's 2,097,150, and has half the span. At $p = 1{,}024$ the work term decides: 19,476 against 2,088 steps. The bounds meet near $p = 891{,}290$, and at $p = 2^{20}$ Kogge-Stone's 39 beats Blelloch's 42.

> [!QUESTION] A producer writes `data = 42` and then `flag.store(1, relaxed)`. A consumer spins on `flag.load(acquire)`, then reads `data`. Can the consumer read a stale `data`?
> Yes. The acquire load orders the consumer's side, and nothing orders the producer's: the relaxed flag store can become visible before the data store does. The pair needs a release on the store, or a fence between the producer's two stores.

> [!QUESTION] Why does release and acquire forbid the weak outcome of MP and not the weak outcome of SB?
> A release orders accesses before it, and an acquire orders accesses after it. In MP the two stores need to stay in order, and the two loads do too, which is exactly what release and acquire give. In SB each thread stores and then loads a different address, and the load may move before the store, because the store is before the load and neither a release nor an acquire pins that pair. Only a full fence, `fence.sc` in PTX, does.

> [!QUESTION] A block-scoped release store is read by an acquire load in another block. What does the model say?
> The release doesn't synchronize with the acquire, because the acquiring thread isn't in the release's scope. libcu++ calls the combination a data race, which is undefined behavior. Use device scope when the threads are in different blocks.

> [!QUESTION] Why does a barrier that waits for the arrival counter to reach 0 hang on its second use?
> The last block resets the counter and the waiters spin for 0. A fast block that leaves and arrives again adds 1 before a slow waiter looks, so the slow waiter sees 1 and never sees 0. A generation counter avoids it: each waiter compares against the generation it read before arriving, so a later round can't hide its release.

> [!QUESTION] A grid of 1,000 blocks of 256 threads uses at most 32 registers per thread on an A100. Can it use `grid.sync()`?
> No. The co-resident limit is 864 blocks, from 8 per SM and 108 SMs, and 1,000 exceeds it. A cooperative launch rejects the grid, and a hand-built barrier would deadlock because 136 blocks never start while the resident ones wait for them.

> [!QUESTION] A CAS-based pop reads `next[A] = B` and stalls. The stack changes to hold `A` over another node. Why can the CAS still succeed, and how do you stop it?
> The head is `A` again, and a CAS compares only that value. It installs `B`, which is no longer in the stack. A version tag in the same 64-bit word makes the CAS fail, because the version moved while the index came back.

## Lab

> [!TRY] Work and span, litmus tests, and a barrier
> Implement the pieces in `starter.py`, in order. The CPU checks are pure Python and run anywhere.
>
> You pass the first checks when `amdahl_speedup(0.95, 8)` is 5.926, when `brent_bounds(1000, 10, 8)` is $(125, 135)$, and when the work and span of $2^{26}$ values are $(67{,}108{,}863, 26)$ for a reduction, and those of 256 values are $(1793, 8)$ for Kogge-Stone, $(510, 16)$ for Blelloch, and $(255, 255)$ for a serial loop. You pass the model checks when your `allowed_outcomes` gives MP three outcomes under sequential consistency and four with relaxed accesses, removes the weak outcome with a release store and an acquire load, and keeps it when only one side has an order. SB with release and acquire must still allow $(0, 0)$, and SB with a fence on each side must not.
>
> You pass the barrier checks when your `generation_barrier` finishes three rounds with no early exit across 480 seeded interleavings, and when 8 blocks with room for 4 deadlock in the simulator. `mp_orders` returns the flag's store and load orders, and your own enumerator must forbid the weak outcome for them.
>
> The last two checks need a GPU. One runs a CUDA message-passing kernel for 13.1 million trials, with your orders and with relaxed accesses, and requires 0 forbidden outcomes with yours. The other runs a grid barrier built from `atomic_ref` and `cooperative_groups` grid sync for 20 rounds each, and requires 0 early exits. The relaxed run's weak count and the co-resident block count are reported as metrics. A machine without a GPU skips both checks, and the first compile takes 30 to 60 seconds.

The slide deck is [Memory consistency and atomics](/slides/39-memory-consistency-and-atomics).

## Further reading

- [PTX ISA, memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): scopes, semantics, morally strong operations, release and acquire patterns, `fence.sc`, and the message-passing, store-buffering, and load-buffering examples this chapter follows.
- [libcu++ memory model](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/memory_model.rst) and [`cuda::atomic_ref`](https://github.com/NVIDIA/cccl/blob/main/docs/libcudacxx/extended_api/synchronization_primitives/atomic_ref.rst): the thread scopes, the scope-inclusion rule for release and acquire, and the type requirements.
- [CUDA Programming Guide, C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html): the memory fence functions, `volatile`, and the writer and reader example with a fence on each side.
- [CUDA Programming Guide, Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html): `this_grid()`, the requirement to launch with `cudaLaunchCooperativeKernel`, and `cudaDevAttrCooperativeLaunch`.
- [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model), Lustig, Sahasrabuddhe, and Giroux, 2019: the axiomatic model, a mapping from a scoped C++ model, and a Coq soundness proof.
- *GPU concurrency: weak behaviours and programming assumptions*, Alglave and others, ASPLOS 2015: litmus tests run on real GPUs. This chapter cites it from memory and didn't read the paper.
- [Stanford CS149 schedule](https://gfxcourses.stanford.edu/cs149/fall24): the schedule lists "Lock Implementations, Fine-Grained Synchronization and Lock-Free Programming" and "Relaxed Consistency + Domain-Specific Programming Systems". The CPU side of this chapter's topics.
- [CMU 15-418 schedule](https://www.cs.cmu.edu/~418/schedule.html): lecture 16, "Fine-Grained Synchronization, Lock-Free Programming", and lectures 20 and 21, both titled "Memory Consistency". The lecture numbers come from that page.
- [Chapter 6](/c/06-reductions), [chapter 7](/c/07-scan-and-histogram), and [chapter 2](/c/02-the-execution-model): the last-block reduction, the look-back protocol, and the residency rules this chapter's barrier needs.
