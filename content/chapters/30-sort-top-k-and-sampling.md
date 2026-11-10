---
title: Radix sort, top-k, and sampling
slug: 30-sort-top-k-and-sampling
part: "Part 2 — CUDA patterns"
summary: Build a stable LSD radix sort from the histogram and scan of chapter 7, merge two sorted lists in parallel with merge path, sort a tile with a bitonic network, then select top-k and top-p without a full sort and sample reproducibly with a counter-based Philox generator.
minutes: 150
gpu: true
objectives:
  - "Build an LSD radix sort from a histogram, an exclusive scan, and a stable scatter, and say why every pass must be stable."
  - "Map signed integers and floats to unsigned keys that sort in the same order, and handle -0.0 and NaN with a defined rule."
  - "Find the cut points of a parallel merge with a diagonal binary search (merge path), and say why the work per thread is equal."
  - "Write a bitonic sorting network over a shared-memory tile and count its stages and compare-exchanges."
  - "Select the top-k elements by radix select, with a tie rule that returns exactly k, and compute a top-p nucleus without sorting the vocabulary."
  - "Draw reproducible samples from a counter-based generator, Philox, whose output depends on a seed, a step, and a row id and on nothing else."
lab: 30-radix-sort-top-k
---

# Radix sort, top-k, and sampling

> [!TLDR]
> - Radix sort is [chapter 7's counting sort](/c/07-scan-and-histogram) repeated once per digit, lowest digit first. It works only because each pass is ==stable==: equal digits keep their order, so the next pass can't undo the last one.
> - A float or a signed integer isn't an unsigned key. Flip the sign bit of an `int32`. For a `float32`, flip the sign bit of a positive value and every bit of a negative one. Then `-inf < -1 < -0.0 <= 0.0 < 1 < inf` in key order.
> - Merge path finds where to cut two sorted lists so that every thread merges the same number of outputs. A binary search along an anti-diagonal finds each cut in $O(\log n)$ steps.
> - Top-k doesn't need a sort. Radix select narrows the k-th largest key one digit at a time, in 4 passes over a 32-bit key, and a tie rule picks which tied elements to keep. Top-p works the same way, with a mass instead of a count.
> - A counter-based generator such as Philox turns `(seed, step, row id)` into random bits with no state to carry. A sampled token then depends on those three numbers and on nothing else, so a batch can reorder without changing any result.

The last two chapters gave you the pieces. [Chapter 6](/c/06-reductions) turned many numbers into one, and [chapter 7](/c/07-scan-and-histogram) kept every running total and counted values into bins. This chapter builds the operation that people reach for next and that the GPU does badly if you write it the obvious way: ordering things.

Sorting hides inside more of a model server than the word suggests. Top-k and top-p sampling pick a few tokens out of a vocabulary of about $10^5$. A mixture-of-experts layer groups tokens by expert, which is a sort on a few bits. Beam search keeps the best few of a much larger candidate set. Nearest-neighbor retrieval keeps the k closest vectors. All of these need an order, or only a threshold in an order, and you can compute the threshold much more cheaply than the order.

The chapter builds three things. First the sorts: the radix sort that production libraries use for large arrays, merge path for merging in parallel, and a bitonic network for a small tile in shared memory. Then selection, which drops the full sort. Last, a random draw that a seed makes reproducible.

> [!NOTE] Who owns sampling
> This chapter owns the kernels: sort, select, top-k, top-p, and a seeded draw. Sampling inside an inference engine, such as how the serving loop calls it per step or how speculative decoding verifies draft tokens, belongs to [learn-inference](https://qwen.ifkash.dev/c/11-sampling). That course derives the sampling rules and explains where they sit in a decode step. Here you build the GPU pieces and keep them correct.

## Before you start

**Reductions and scan.** [Chapter 6](/c/06-reductions) built the block-wide sum from warp shuffles and shared memory. [Chapter 7](/c/07-scan-and-histogram) built the exclusive scan and the privatized histogram, and it ended with a CPU counting sort. You need both. Radix sort is a histogram, a scan, and a scatter, repeated.

**Stability.** A sort is *stable* when elements with equal keys keep their input order. `torch.sort(x, stable=True)` guarantees it. The default `torch.sort` doesn't, and the [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.sort.html) shows the indices of equal elements coming out scrambled without the flag.

**Floating-point bits.** A `float32` is one sign bit, eight exponent bits, and 23 fraction bits. The value 1.0 is `0x3F800000`, and -1.0 is `0xBF800000`. Two zeros exist, `0x00000000` and `0x80000000`, and they compare equal. A NaN has an exponent of all ones and a nonzero fraction, and it doesn't compare equal to anything, itself included. [Chapter 9](/c/09-floating-point-on-the-gpu) covers the formats in depth.

**The copy ceiling.** A device-to-device copy measured 1275 GB/s on the lab A100, against a rating of 1935 GB/s (learn-inference, chapter 10). This chapter judges every sort and select against that figure by counting bytes.

## What you do with an order

This section answers why a GPU course needs a sort at all, and which of the sorts' relatives are cheaper.

Three questions come up, and each has a cheaper answer than the full sort:

| Question | Output you need | Cheapest tool in this chapter |
|---|---|---|
| Put every element in order | A permutation of $n$ items | Radix sort |
| Which $k$ elements are largest | A set of $k$, optionally in order | Radix select |
| Which tokens hold the first $p$ of the mass | A mask over $n$ items | Threshold search on the mass |

A sort answers all three, and it costs the most. For a 32-bit key, the radix sort of this chapter reads and writes every key four times. Selecting the top-k reads every key at most four times and writes $k$ of them. When you only need a threshold, you pay for a threshold. The full sort stays worth knowing, because the cheaper tools are built from the same parts, and because some jobs need the whole order, such as grouping a million tokens by expert.

## Keys that sort as unsigned integers

This section answers how to feed a signed integer or a float to a sort that compares only unsigned bits.

A radix sort looks at a key one digit at a time, from its bits. It has no comparison, so it can't ask whether $-3 < 2$. It sees only that the key's bit pattern, read as an unsigned number, is smaller or larger. The fix is a map from every value to an unsigned key, chosen so that the order of the keys is the order of the values. ==The map must be monotone and, for a sort that returns the values, invertible.==

### Signed integers

An `int32` stores two's complement. The value $-1$ is `0xFFFFFFFF`, which read as unsigned is the largest key, and $0$ is `0x00000000`, the smallest. Every negative number has its top bit set, and every non-negative number has it clear, so the negatives land above the positives. Flip the top bit and the two halves swap:

$$
\text{key}(x) = x \oplus \hla{\texttt{0x80000000}}
$$

Here $\hla{\texttt{0x80000000}}$ is the sign bit, and $\oplus$ is exclusive or. Inside each half, two's complement is already monotone, so the order inside each half holds. After the flip, $-2^{31}$ maps to 0, $-1$ to `0x7FFFFFFF`, $0$ to `0x80000000`, and $2^{31}-1$ to `0xFFFFFFFF`. The inverse is the same XOR.

### Floating-point numbers

A `float32` is sign-and-magnitude, not two's complement, and that changes the fix. For a *positive* float, the bits as an unsigned integer already sort correctly: a larger exponent is a larger number, and for equal exponents a larger fraction is. So a positive float needs only the sign-bit flip, which moves every positive above every negative. A *negative* float has the opposite problem. A larger magnitude has a larger bit pattern, but a more negative value. Invert every bit and the order reverses.

$$
\text{key}(x) = \begin{cases} \text{bits}(x) \oplus \hla{\texttt{0x80000000}} & \text{if the sign bit of } x \text{ is } 0 \\ \text{bits}(x) \oplus \hlb{\texttt{0xFFFFFFFF}} & \text{if the sign bit of } x \text{ is } 1 \end{cases}
$$

The positive case flips one bit, $\hla{\texttt{0x80000000}}$. The negative case flips all 32, $\hlb{\texttt{0xFFFFFFFF}}$. To invert it, read the key's top bit. If it's set, the value was positive, and you undo the single flip. Otherwise undo the full inversion. [CUB's radix sort documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html) describes the same pair of transformations: positive floats have the sign bit flipped, and negative floats have all bits inverted.

> [!EXAMPLE] Six probe values
> `scripts/figures/30-sort-top-k-and-sampling.py` prints the keys. Each row is the bit pattern, then the key:
>
> | Value | Bits | Key |
> |---|---|---|
> | $-\infty$ | `0xFF800000` | `0x007FFFFF` |
> | $-1.0$ | `0xBF800000` | `0x407FFFFF` |
> | $-0.0$ | `0x80000000` | `0x7FFFFFFF` |
> | $+0.0$ | `0x00000000` | `0x80000000` |
> | $1.0$ | `0x3F800000` | `0xBF800000` |
> | $+\infty$ | `0x7F800000` | `0xFF800000` |
>
> The keys increase down the table, so $-\infty < -1 < -0.0 \le 0.0 < 1 < \infty$ holds in key order. The one place the table is strict where the numbers are equal is the pair of zeros: `0x7FFFFFFF` is less than `0x80000000`.

A common mistake is to flip every bit of a positive float and only the sign bit of a negative one: the transform in reverse. It still looks plausible on a few inputs, but it sends 1.0 to `0x407FFFFF` and sorts every positive below every negative. The lab's second check hard-codes the table above and catches it.

### Zeros and NaNs

The key map orders $-0.0$ strictly before $+0.0$. A sort that matches `torch.sort` can't do that, because the two compare equal and a stable sort must keep their input order. The fix is a *sort key*: map both zeros to the key of $+0.0$. Then the pair ties, and stability decides their order.

NaN needs a rule too. IEEE 754 leaves NaN unordered, and `torch.sort` puts NaN after $+\infty$. A positive NaN such as `0x7FC00000` has the key `0xFFC00000`, above $+\infty$'s key. A NaN with the sign bit set, `0xFFC00000`, has the key `0x003FFFFF`, below $-\infty$. The sort would scatter NaNs to both ends. The rule that matches torch maps every NaN, whatever its sign or payload, to the largest key, `0xFFFFFFFF`.

> [!WARNING] A key that inverts exactly isn't always a sort key
> `float32_to_key` is bit-exact and invertible, so the round trip gives back the same bits, including the sign of a zero. `float32_sort_key` isn't invertible: it merges the zeros and the NaNs. Sort by the sort key, and get the values back by indexing the original array with the sorted indices, not by inverting the key.

CUB takes a different position. Its documentation says positive and negative zero are equivalent and keep their input order, and that NaNs get no special handling and sort by their bit patterns after the transformation. If your data can hold NaN, decide the rule before you pick the library, and clamp or mask NaNs yourself when the rule matters.

## One radix pass is a counting sort

This section answers what one pass does, and why a sequence of such passes sorts the whole key.

Chapter 7 built a counting sort of small keys: histogram the digits, take the exclusive scan of the counts, and scatter each element to the next free slot of its digit. That is one radix pass. The new part is how to call it on a key that has more bits than you can afford a bin for.

A 32-bit key has $2^{32}$ values, and no histogram that large fits anywhere. Split the key into digits of $r$ bits. A digit has $2^{r}$ values, and 8 bits gives 256 bins, which fits in shared memory. Sort by the *lowest* digit first, then the next, up to the highest. That order is *least significant digit first*, LSD. Each pass is a stable counting sort on one digit.

```viz
30-radix-pass
```

The figure runs two passes over eight 4-bit keys with a 2-bit digit. The first pass reads the keys left to right: `13a` has the low digit 1, and its bin starts at slot 0, so it goes to slot 0. `6b` has the low digit 2, and bin 2 starts at slot 4, so it goes to slot 4. The histogram is $[0, 4, 3, 1]$, the exclusive scan is $[0, 0, 4, 7]$, and the destinations are $[0, 4, 1, 2, 7, 5, 6, 3]$. After the pass the keys are sorted by their low two bits.

The second pass sorts by the high digit. The keys `6b` and `6g` are equal, so they have the same high digit, 1. The pass places them by their order in its input, and that order came from the first pass, which kept `b` before `g`. The output has `6b` at slot 3 and `6g` at slot 4. The whole sort is correct: `1h, 3e, 5d, 6b, 6g, 9c, 13a, 14f`.

### Why stability is the whole proof

The sort is correct because of one property, and you can see it from the last pass backward. After the final pass, the array is ordered by the highest digit. Within one value of that digit, the order is whatever the previous pass left, and that order is by the next digit, and so on. This is an induction on passes:

$$
\hla{\text{after pass } t}: \quad \text{sorted by } (d_t, d_{t-1}, \ldots, d_0) \text{ lexicographically}
$$

Here $d_0$ is the lowest digit. Pass $t$ sorts by $d_t$ and, because it's stable, leaves elements with equal $d_t$ in the order $\hla{\text{pass } t-1}$ gave them, which is sorted by $(d_{t-1}, \ldots, d_0)$. After the last pass the order is by all digits, which is by the key. Without stability the induction breaks at the first tie, and the earlier work is lost.

> [!KEY] Stability is what lets passes compose
> A radix pass that scatters equal digits in any order other than their input order sorts correctly by that digit and loses everything the lower digits did. The full sort is correct only if every pass is stable.

The lab catches the usual way to write an unstable pass: walk the input forward, but fill each bin from its end. The first 1-digit of the lab's example input, `[3, 1, 2, 1, 0, 3, 2, 1]`, lands in the last slot of bin 1 instead of the first. The stable result carries the indices $[4, 1, 3, 7, 2, 6, 0, 5]$, and the unstable one reverses each group of equal digits.

Here is the pass in Python, the shape the lab asks for:

```python
def radix_pass(keys, idx, shift, bits):
    nbins = 1 << bits
    mask = nbins - 1
    digits = [(k >> shift) & mask for k in keys]
    counts = [0] * nbins
    for d in digits:
        counts[d] += 1
    offsets, running = [0] * nbins, 0
    for d in range(nbins):                 # exclusive scan of the histogram
        offsets[d], running = running, running + counts[d]
    out_keys, out_idx = [0] * len(keys), [0] * len(keys)
    for pos, d in enumerate(digits):       # input order keeps the pass stable
        slot = offsets[d]
        offsets[d] += 1
        out_keys[slot], out_idx[slot] = keys[pos], idx[pos]
    return out_keys, out_idx
```

A full sort calls it $32/r$ times with `shift = 0, r, 2r, ...`, and the indices that come out are `torch.sort(x, stable=True).indices`. Carrying the indices through every pass, instead of sorting the values alone, is what gives you the permutation. It's also how a key-value sort moves a payload.

## From one pass to a GPU radix sort

This section answers how to run the pass on thousands of threads without losing stability, and what it costs in bytes.

The CPU pass walks the input in order and increments one running offset per digit. That is a sequential dependence, and a GPU can't use it directly. The fix is the one that chapter 7 used for a device-wide scan: split the work into tiles, count each tile on its own, scan the counts, then let each tile scatter with its own starting offsets. The pass has three phases.

1. **Count.** Each thread block takes one tile of keys and histograms the digit into shared memory, as in chapter 7's privatized histogram. The result is a table $H[d][b]$, the number of keys with digit $d$ in tile $b$.
2. **Scan.** Take the exclusive scan of $H$ in digit-major order: all tiles of digit 0 first, then all tiles of digit 1, and so on. The entry for $(d, b)$ is the first output slot for the keys of digit $d$ in tile $b$.
3. **Scatter.** Each block ranks each of its keys among the keys with the same digit in its own tile, in tile order, and writes the key to the entry for its $(d, b)$ plus that rank.

The digit-major scan order is the stability argument for a GPU. Every key of digit $d$ in tile 0 lands before every key of digit $d$ in tile 1, and within a tile the rank is in input order. The sequence matches the sequential pass exactly.

> [!EXAMPLE] Two tiles of four keys and two bins
> Tile 0 holds the digits $[1, 0, 1, 1]$ and tile 1 holds $[0, 1, 0, 0]$. The count table has $H[0] = [1, 3]$ for the tiles' digit-0 counts and $H[1] = [3, 1]$. Flatten it in digit-major order to $[1, 3, 3, 1]$. The exclusive scan is $[0, 1, 4, 7]$. So digit 0 of tile 0 starts at slot 0, digit 0 of tile 1 at slot 1, digit 1 of tile 0 at slot 4, and digit 1 of tile 1 at slot 7. The total, 8, is the array's length. Digit 0 fills slots 0 to 3, with tile 0's key first. Digit 1 fills slots 4 to 7, with tile 0's three keys first.

The count phase is the only kernel the lab asks you to write on the GPU, and it's chapter 7's privatized histogram with one change: the bin index is `(key >> shift) & mask`.

```cuda
__global__ void radix_hist_kernel(const uint32_t* keys, int* hist, int64_t n, int shift, int nbins) {
  extern __shared__ int local[];
  for (int b = threadIdx.x; b < nbins; b += blockDim.x) local[b] = 0;
  __syncthreads();
  const uint32_t mask = (uint32_t)(nbins - 1);
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    atomicAdd(&local[(keys[i] >> shift) & mask], 1);
  __syncthreads();
  for (int b = threadIdx.x; b < nbins; b += blockDim.x)
    if (local[b] != 0) atomicAdd(&hist[b], local[b]);
}
```

The key type is `uint32_t`, so the shift is logical. If you read an `int32` and shift it, the compiler uses an arithmetic shift that copies the sign bit into the top digit, and a negative key gets the wrong bin. The lab tests a negative key at `shift = 24`.

### How much memory the sort moves

A radix sort is a streaming algorithm, so the roofline from [chapter 4](/c/04-roofline-and-latency) is the right way to ask how fast it can go. Count the bytes. The global histogram of a digit doesn't depend on the order of the keys, so one read of every key builds the histograms of all four 8-bit digits. Then suppose each of the four passes reads every key once and writes every key once. For $n$ 4-byte keys, that is

$$
\text{bytes} = \hla{4n} + \hlb{4 \times (4n + 4n)} = 36n
$$

where $\hla{4n}$ is the up-front histogram read and $\hlb{4 \times 8n}$ is four passes of one read and one write. With 4-byte indices carried along, each pass moves $16n$ bytes and the total is $68n$.

> [!EXAMPLE] 2^26 keys on the lab A100
> Take $n = 2^{26} = 67{,}108{,}864$ keys. The key-only sort moves $36n = 2.416$ GB, and at the 1275 GB/s copy ceiling the floor is 1.895 ms. The key-and-index sort moves $68n = 4.563$ GB, with a floor of 3.579 ms. These are predictions from the byte count, not measurements. A real sort also moves the scan tables and spends time in the ranking step, so it lands above them. The lab measures only the histogram kernel, and it reports GB/s.

The three-phase scheme above reads the keys twice per pass, once to count and once to scatter. That is $4 \times (4n + 8n) = 48n$ bytes for four passes, a floor of 2.526 ms for the same $2^{26}$ keys. The *Onesweep* sort removes the extra read. It fuses the scan with the scatter using a single-pass prefix sum, the decoupled look-back from chapter 7's device scan, so each pass reads the keys once. The paper reports 29.4 giga-keys per second for 256 million random 32-bit keys on an NVIDIA A100, about 1.5 times the speed of CUB's earlier radix sort ([Adinets and Merrill, 2022](https://arxiv.org/abs/2206.01784)). That figure is from the paper's A100, not from the lab card, so don't compare it with the floor above.

### Choosing the digit width

A wider digit means fewer passes and a bigger histogram. With $r = 8$, a 32-bit key takes 4 passes, and 256 counters fit in shared memory with room to spare. With $r = 11$ it takes 3 passes, and 2048 counters use 8 KiB per block. With $r = 16$ it takes 2 passes, but 65,536 counters is 256 KiB, more than the 164 KiB an A100 SM has. A pass costs mostly memory traffic, so fewer passes win until the histogram stops fitting. The lab's histogram takes `bits` up to 11 for that reason.

Narrow keys need fewer passes: token ids below 65,536 take two 8-bit passes, not four. CUB exposes this as `begin_bit` and `end_bit`, and warns that its key transformations run before the truncation. A sort of expert ids needs one pass of a few bits, which is chapter 7's counting sort.

## Merge path: split a merge into equal pieces

This section answers how to merge two sorted lists with many threads, when the merge looks like a single sequential walk.

Merging is the step in a merge sort, and it's how you combine two sorted runs. The serial merge keeps one pointer in each list and repeatedly takes the smaller head. That is $n - 1$ comparisons in a chain, and each step needs the previous pointers. A parallel merge has to cut the output into pieces that don't depend on each other.

The obvious cut fails. Give each thread a slice of $a$ and the matching slice of $b$, and the values decide which slice matches: one thread's slice of $b$ can be empty while another's holds half the list. The work is unbalanced, and a warp runs at the speed of its slowest lane.

*Merge path* cuts the *output* instead. To give $T$ threads equal work, thread $t$ produces output positions $tL$ to $(t+1)L - 1$ for $L = (|a| + |b|)/T$. Each thread needs one number: how many of the first $tL$ outputs came from $a$. Call it $i$. The count from $b$ is $j = tL - i$, and the thread starts its merge at $(i, j)$.

### The diagonal binary search

Picture a grid with $a$ along one axis and $b$ along the other. A merge is a monotone path from the corner $(0, 0)$ to $(|a|, |b|)$: take a step along $a$ when you take from $a$, and along $b$ when you take from $b$. The first $d$ outputs end on the anti-diagonal $i + j = d$. The path crosses each anti-diagonal exactly once, and the crossing is the answer you need.

Which point on the diagonal is it? A cut $(i, j)$ is the right one when every element taken from $a$ is no later in the merge than the next element of $b$, and every element taken from $b$ is no later than the next element of $a$. Walk along the diagonal from one end to the other and the first condition flips exactly once, from false to true, so a binary search finds the flip. Search $i$ between $\max(0, d - |b|)$ and $\min(d, |a|)$, which keeps both $i$ and $j = d - i$ inside their lists:

$$
\text{if } a[\hla{m}] \le b[d - 1 - \hla{m}] \text{ then } \hla{i} > m, \text{ otherwise } \hla{i} \le m
$$

Here $\hla{m}$ is the probe. The elements $a[m]$ and $b[d-1-m]$ are the last candidates on either side of the probe, and the comparison says which one comes first in the merge. When $a[m]$ goes first, the cut takes at least $m + 1$ elements from $a$, so move the lower bound to $m + 1$. Otherwise move the upper bound to $m$. The search ends in $\lceil \log_2(\min(d, |a|) - \max(0, d - |b|) + 1) \rceil$ steps, and each step is two loads.

> [!EXAMPLE] Cutting a merge of 9 elements into three pieces
> Take $a = [1, 3, 5, 7, 9]$ and $b = [2, 3, 4, 8]$. The merge has 9 outputs, so split it at diagonals 0, 3, 6, and 9. The script prints each cut:
>
> | Diagonal | From $a$ | From $b$ |
> |---|---|---|
> | 0 | 0 | 0 |
> | 3 | 2 | 1 |
> | 6 | 3 | 3 |
> | 9 | 5 | 4 |
>
> Thread 1 starts at $(2, 1)$ and merges until $(3, 3)$. It reads $a[2] = 5$ and $b[1], b[2] = 3, 4$, and writes $3, 4, 5$. Every thread merges exactly three outputs, whatever the values are.

The comparison uses less-than-or-equal on purpose. When $a[m]$ equals $b[d-1-m]$, $a$ goes first, so equal elements from $a$ come before equal elements from $b$ in the output. That is the stable merge rule, and it's the same rule the sequential merge uses. In the example, the two 3s are $a_1$ and $b_1$. The first three outputs are $1, 2, 3_a$, so diagonal 3 takes two elements from $a$ (the 1 and the 3) and one from $b$ (the 2). If you search with strict less-than, the cut at diagonal 3 takes one from $a$ and two from $b$, which puts $3_b$ ahead of $3_a$. The lab's sixth check compares every diagonal of 40 random pairs against a brute-force stable merge, and it also checks which list each output came from.

The device function is short:

```cuda
__device__ int merge_path(const int* a, int na, const int* b, int nb, int diag) {
  int lo = max(0, diag - nb), hi = min(diag, na);
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (a[mid] <= b[diag - 1 - mid]) lo = mid + 1; else hi = mid;
  }
  return lo;   // the number of the first `diag` outputs that come from a
}
```

### Where merge path goes

A thread block merges a tile of outputs in two steps. It runs the diagonal search once per tile, at the tile's two edges, and the slices of $a$ and $b$ between those cuts go to shared memory. Then each thread runs a second, shorter search inside the tile, and merges its few outputs from shared memory.

A GPU merge sort uses this at every level: sort tiles with a bitonic network, then merge pairs of runs with merge path, and repeat for $\log_2$ of the number of tiles levels. For 32-bit keys the radix sort makes a fixed four passes over the data, so it needs fewer passes than a merge sort. Merge path earns its place when the keys have a comparator but no digits.

The same diagonal search balances sparse matrix-vector products, where the two lists are the row offsets and the nonzero indices, and the balance problem is the same. Merrill and Garland's merge-based sparse matrix-vector product applies it to that problem.

## Bitonic sort in shared memory

This section answers how to sort a few thousand elements that already sit in shared memory, with a schedule that doesn't depend on the data.

A *sorting network* is a fixed list of compare-exchange operations: look at positions $i$ and $j$, and put the smaller value in $i$. The list is the same for every input, so no thread branches on the data to decide what to do, and every thread in a warp does the same instructions. That is why a GPU likes them. *Bitonic sort* is a network built from one idea: a *bitonic sequence* rises and then falls, and a single pass of compare-exchanges at distance half the length splits it into two bitonic sequences, with every element of the first half no larger than every element of the second.

The network has two nested loops. The outer loop doubles the block size $k = 2, 4, 8, \ldots, n$, and sorts each block of $k$ elements. The inner loop halves the distance $j = k/2, k/4, \ldots, 1$, and compare-exchanges the pairs $i$ and $i \oplus j$. The direction of a block alternates: where bit $k$ of the index is 0 the block sorts ascending, and where it's 1 it sorts descending, so two neighbors form a bitonic sequence for the next block size to merge.

```viz
30-bitonic-network
```

Run the figure from stage 0 to stage 6. The input is $[7, 3, 6, 1, 5, 8, 2, 4]$. Stage 1 compares neighbors with alternating directions and gives $[3, 7, 6, 1, 5, 8, 4, 2]$. After stage 3 the left four are ascending and the right four are descending, $[1, 3, 6, 7, 8, 5, 4, 2]$, which is bitonic. The last three stages, $k = 8$ with $j = 4, 2, 1$, merge it into $[1, 2, 3, 4, 5, 6, 7, 8]$.

### The cost of a network

For $n = 2^{m}$ elements, the outer loop has $m$ block sizes, and block size $2^{s}$ takes $s$ inner stages. Every stage does $n/2$ compare-exchanges:

$$
\text{stages} = \hla{\frac{m(m+1)}{2}}, \qquad \text{compare-exchanges} = \hlb{\frac{n}{2}} \cdot \frac{m(m+1)}{2}
$$

The stage count $\hla{m(m+1)/2}$ is the number of times threads must synchronize, and $\hlb{n/2}$ is the number of threads that are busy in each. For $n = 8$, $m = 3$ and the network has 6 stages and 24 compare-exchanges, which the figure shows. For $n = 1024$, it has 55 stages and 27,648 compare-exchanges. For $n = 2^{20}$ it has 210 stages and 110,100,480.

A comparison sort needs only about $n \log_2 n$ comparisons, which is 10,240 for $n = 1024$. The network does 27,648, 2.7 times as many. The extra work is the price of a schedule with no data-dependent control flow, and it's a good price when $n$ is small enough to live in shared memory and a sync costs less than a divergent branch.

### The kernel

Each thread owns one pair per stage. To find the pair, insert a zero bit at the position of $j$ into the thread index $t$, which gives $i$, and add $j$:

```cuda
// 1024 keys per block, 512 threads. Each thread handles one compare-exchange per stage.
__global__ void bitonic_tile_kernel(uint32_t* keys) {
  __shared__ uint32_t s[1024];
  const int t = threadIdx.x;
  s[t] = keys[blockIdx.x * 1024 + t];
  s[t + 512] = keys[blockIdx.x * 1024 + t + 512];
  __syncthreads();
  for (int k = 2; k <= 1024; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int i = 2 * t - (t & (j - 1));   // t with a zero bit inserted at the bit of j
      const int p = i + j;
      const bool up = (i & k) == 0;
      const uint32_t x = s[i], y = s[p];
      if ((x > y) == up) { s[i] = y; s[p] = x; }
      __syncthreads();                        // every stage reads what the last one wrote
    }
  }
  keys[blockIdx.x * 1024 + t] = s[t];
  keys[blockIdx.x * 1024 + t + 512] = s[t + 512];
}
```

The sort runs on unsigned keys, so the key transform from earlier in the chapter applies, and the compare has no NaN case. The tile is 4 KiB of shared memory. For $j < 32$ the stride between consecutive threads' `i` can put two lanes on one bank, so a tuned kernel handles those last five stages inside a warp with shuffles, as the scan in chapter 7 did.

A tile sorted this way is a sorted run. A full sort of $n$ elements sorts tiles of 1024 with this kernel and merges the runs with merge path, or feeds a radix sort a tile-sorted input. A top-k sort of a small array, such as 1024 candidates, stops here.

## Top-k by radix select

This section answers how to find the $k$ largest elements without ordering the other $n - k$.

Top-k asks for a set, and a threshold defines a set: the $k$ largest elements are those at or above the $k$-th largest value. Finding that one value is *selection*, and a histogram finds it by elimination. You don't need the order of the keys. You need to know which digit holds the $k$-th largest key, and the histogram tells you that.

The procedure walks the digits from the highest to the lowest, with 8-bit digits and 4 passes for a 32-bit key. Keep a *candidate set* and a count $\hla{r}$ of how many of the top $k$ are still unplaced, starting at $r = k$.

1. Histogram the current digit over the candidates.
2. Scan the counts from the largest digit down. If a digit's count is less than $r$, all of its candidates are among the top $r$, so subtract its count from $r$ and move down.
3. The first digit whose count is at least $r$ holds the $r$-th largest. Keep only its candidates, and go to the next digit.

After the last digit, the candidates are all the elements equal to the threshold key, and $r$ is how many of them the top-k still needs.

> [!EXAMPLE] Top 4 of eight 8-bit keys with 4-bit digits
> Take the keys `5A 3C 5F 12 5A 7B 5C 0E` at indices 0 to 7, and $k = 4$, so $r = 4$ at the start.
>
> Pass 1 reads the high hex digit of each key: $[5, 3, 5, 1, 5, 7, 5, 0]$. The histogram has one key in each of digits 0, 1, 3, and 7, and four keys in digit 5. Scan from the top. Digit 7 holds one candidate, less than $r = 4$, so subtract and $r$ becomes 3. Digit 6 is empty. Digit 5 holds four, at least $r = 3$, so the third-largest key is in digit 5. The candidates are now `5A`, `5F`, `5A`, and `5C`, at indices 0, 2, 4, and 6.
>
> Pass 2 reads the low digit of the candidates: A, F, A, C. Scan from the top. Digit F holds one, less than $r = 3$, so $r$ becomes 2. Digits E and D are empty. Digit C holds one, less than 2, so $r$ becomes 1. Digit B is empty. Digit A holds two, at least 1, and the threshold key is `5A`.
>
> The keys above the threshold are `7B`, `5F`, and `5C`, at indices 5, 2, and 6. One more element is needed, and two keys equal `5A`, at indices 0 and 4. The lowest index, 0, wins. The top 4 are the elements at indices 5, 2, 6, and 0.

### The tie rule

The last step is a decision, and an implementation has to make it on purpose. When the $k$-th value is tied with elements that fall outside the top $k$, you can't keep all of them: the result would have more than $k$ elements. You must pick among them, and the pick should be a rule, so two runs return the same set.

The rule in this chapter and in the lab is *lowest index first*. It matches a stable descending sort, so `torch.sort(x, descending=True, stable=True).indices[:k]` is the reference. [PyTorch's `torch.topk`](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html) does not promise that: its documentation says the indices of tied elements aren't guaranteed to be stable and may differ between calls. So the lab compares the values with `torch.topk`, which must match exactly, and the indices with the stable sort, which pins the rule.

CUB has a device-wide top-k, `cub::DeviceTopK`, which its documentation says is based on an algorithm called AIR TopK. The documentation describes the full contract as deterministic with ties resolved toward the smaller index, and it says the released mode returns an unordered subset that may differ between runs when the $k$-th value is tied. Read the contract of the version you use.

The wrong implementation the lab catches here is to keep every element at or above the threshold. It returns the right threshold and a set with more than $k$ elements when the boundary is tied. The test uses a vector of rounded normal values with heavy ties, and a vector of all zeros.

### What it costs

Each pass reads the candidates once. If you recompute the digit of every key on each pass, and filter by the digits you've already fixed, each pass reads all $n$ keys: 4 passes of $4n$ bytes is $16n$ bytes. For $2^{26}$ keys that is 1.074 GB and a floor of 0.842 ms at 1275 GB/s, a bit under half the radix sort's 1.895 ms floor. If you compact the candidates after the first pass, the later passes read far less, since a threshold usually leaves few candidates.

For sampling, the shape is different. A row of logits is one vocabulary, about $10^5$ floats, which is 0.4 MB. A block can pull it into shared memory or L2 once and run all four passes there. The cost of the select is then the passes' synchronizations, not memory traffic, and one block per row is the usual design: the batch is the parallel dimension.

## Top-p without sorting the vocabulary

This section answers how to find the nucleus of a distribution with the same search idea, and where the rounding can break it.

Chapter 7 showed top-p as a scan: sort the probabilities in descending order, take the exclusive scan, and keep the tokens whose preceding mass is below $p$. The sort is the expensive part, and the rule says nothing about the order of the discarded tokens. The kept set is also a threshold set: it is all tokens with a probability above some value $\hla{t^{\ast}}$, plus some of the tokens equal to it. So search for $\hla{t^{\ast}}$.

Define the mass at or above a threshold $t$:

$$
M(t) = \sum_{i \,:\, \pi_i \ge t} \pi_i
$$

where $\pi_i$ is the probability of token $i$. $M$ never increases as $t$ rises. The threshold is the largest $t$ with $M(t) \ge p$:

$$
\hla{t^{\ast}} = \max \{\, t : M(t) \ge p \,\}
$$

Every token above $\hla{t^{\ast}}$ stays, because $M$ just above $t^{\ast}$ is less than $p$, so the mass strictly above is less than $p$. A token equal to $t^{\ast}$ stays while the kept mass is still below $p$, and you take those in index order.

Non-negative floats have bit patterns that increase with their values, so you can binary-search the 31 low bits of $t$ as an integer: guess a pattern, sum the probabilities at or above it, and compare with $p$. That is 31 passes over the probabilities, each a masked reduction, and no sort. A radix-style search does the same with 4 passes of a histogram of the mass by digit.

> [!EXAMPLE] Four equal probabilities and p = 0.6
> Take $\pi = [0.25, 0.25, 0.25, 0.25]$ and $p = 0.6$. The sort-based rule keeps a token when the mass before it is below 0.6. The exclusive scan is $[0, 0.25, 0.5, 0.75]$, so tokens 0, 1, and 2 stay and token 3 goes. The threshold search finds $t^{\ast} = 0.25$, because $M(0.25) = 1.0 \ge 0.6$ and nothing larger exists. No token is strictly above it, so the fix-up takes the tied tokens in index order while the kept mass is below 0.6: tokens 0, 1, and 2. The kept mass is 0.75. Both methods agree, and both depend on the tie rule. Without it, a threshold test alone would keep all four.

Take the chapter 7 example too. The sorted probabilities $[0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02]$ with $p = 0.85$ keep four tokens, because the exclusive scan is $[0, 0.50, 0.70, 0.82, 0.90, \ldots]$ and the first four entries are below 0.85. The threshold search lands on $t^{\ast} = 0.08$: $M(0.08) = 0.90 \ge 0.85$ and $M(0.12) = 0.82 < 0.85$. The token with probability 0.08 is the one that crosses the threshold, and it stays.

The wrong implementation is to use the inclusive scan, keeping a token only when the mass through it is at most $p$. On this example it keeps three tokens, not four. The lab's eighth check runs the example in a shuffled order and expects four.

### Rounding can break the search

The sum over a masked set adds probabilities in an order that depends on the thread layout. Floating-point addition isn't associative ([chapter 9](/c/09-floating-point-on-the-gpu)), so two runs can disagree about whether $M(t) \ge p$ when the two sides differ by a few units in the last place. The result is a nucleus that changes by one token between runs, or a mass that isn't monotone in $t$, which breaks the binary search. The [FlashInfer team notes](https://flashinfer.ai/2025/03/10/sampling.html) that parallel prefix sums may not be monotonic, and that this can produce invalid tokens unless handled with care.

Two habits keep it safe. Accumulate in float64, or in a fixed tree order, so the sum is the same on every run. And have the lab-style tests use probabilities that are multiples of $1/1024$, where every sum is exact in float64, so a failing check is a logic error and not a rounding one.

### Sorting-free sampling in production

FlashInfer takes a different route. It draws a token by inverse transform sampling, ignoring tokens at or below a pivot, then raises the pivot to the sampled token's probability and checks whether the remaining mass still meets the top-p target. If not, it repeats, inside one fused kernel. The blog reports that on one H100 under vLLM, the sort-free kernels cut overall sampling time by more than 50% across three models. That figure is from their hardware and setup, not from the lab card.

Pick threshold search when you need the mask itself, for example to renormalize the kept probabilities. Pick the pivot loop when the next step is a draw.

## Seeded sampling with a counter-based generator

This section answers how to make the final random draw reproducible when thousands of threads draw at once.

A sampler needs one uniform random number per row of a batch. The usual generator, a *stateful* one, keeps a state and updates it on every call: the next number depends on the state the last call left. On a GPU that raises a question the generator can't answer on its own. If each thread owns a state, the numbers a row sees depend on which thread handled it, and a different batch size or block shape gives a different result from the same seed. If all threads share one state, they serialize.

A *counter-based* generator has no state to carry. It is a pure function: it takes a *counter* and a *key* and returns random bits. The key is the seed. The counter is whatever you decide it means, and each distinct counter gives an independent block of random bits. [Salmon, Moraes, Dror, and Shaw's SC'11 paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) introduced the family, which includes Philox and Threefry, and the construction is why a GPU kernel can ask for the bits of any position directly. The cuRAND device API offers Philox as `Philox_4x32_10`, with a seed, a subsequence, and an offset that together locate a position in the sequence ([cuRAND device API](https://docs.nvidia.com/cuda/curand/device-api-overview.html)). Triton's `tl.rand` takes a scalar seed and a block of offsets and returns uniform `float32` values ([`tl.rand`](https://triton-lang.org/main/python-api/generated/triton.language.rand.html)).

### Philox-4x32-10

The name is its shape: a counter of four 32-bit words, a key of two, and ten rounds. A round multiplies two of the counter words by two fixed constants. Each 64-bit product splits into a high and a low half. The halves, the other two counter words, and the key mix into the next counter by exclusive or. Then the key advances by two fixed Weyl constants. With $\hla{M_0}$ and $\hlb{M_1}$ the multipliers, a round maps $(c_0, c_1, c_2, c_3)$ with key $(k_0, k_1)$ to:

$$
(c_0', c_1', c_2', c_3') = \big(\text{hi}(\hlb{M_1} c_2) \oplus c_1 \oplus k_0,\ \text{lo}(\hlb{M_1} c_2),\ \text{hi}(\hla{M_0} c_0) \oplus c_3 \oplus k_1,\ \text{lo}(\hla{M_0} c_0)\big)
$$

The multiplication spreads each word's bits upward, the XOR with the other words and the key folds the pieces together, and ten rounds, the setting in the name, finish the mixing. A multiplication is cheap on a GPU, and the whole function uses registers and no memory.

```python
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF

def philox4x32_10(counter, key):
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK, p1 & MASK, ((p0 >> 32) ^ c3 ^ k1) & MASK, p0 & MASK
        k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
    return c0, c1, c2, c3
```

The lab checks this function against two known-answer vectors from the Random123 library's test suite. For counter $(0, 0, 0, 0)$ and key $(0, 0)$, the output is `0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8`. A wrong constant or a swapped word changes every output bit, so a known-answer test is the right first check for a hand-written generator.

### From bits to a token

Turn one 32-bit word into a uniform number in $[0, 1)$ by keeping its top 24 bits: $u = (\text{word} \gg 8) / 2^{24}$. A `float32` has a 24-bit significand, so every such $u$ is exactly representable, and $u$ never reaches 1.0. Dividing the whole word by $2^{32} - 1$ in `float32` rounds the largest words up to 1.0, which picks a token past the end of the vocabulary.

Then invert the cumulative distribution: the token is the first index whose cumulative probability exceeds $u$ times the row total. Multiplying by the total makes the rule work for unnormalized rows and for a mass that rounds to just under or over 1.

> [!EXAMPLE] One draw, by hand
> The first word for counter $(0,0,0,0)$ and key $(0,0)$ is `0x6627E8D5`. Shift out the low 8 bits to get `0x6627E8`, which is 6,694,888. Divide by $2^{24} = 16{,}777{,}216$ and $u = 0.399046$. Take the distribution $[0.5, 0.25, 0.125, 0.125]$, with cumulative sums $[0.5, 0.75, 0.875, 1.0]$. The first cumulative sum above 0.399 is the first, 0.5, so the token is 0. A uniform of 0.6 would give token 1.

Gumbel-max is the other common route: add the noise $-\log(-\log u_i)$ to each logit and take the argmax. It needs one uniform per vocabulary entry and no cumulative sum, and each $u_i$ comes straight from the counter $(\text{row}, \text{step}, i)$.

### What goes in the counter

The counter is the design decision. Put the *row id* in one word and the *step* in another, and a draw depends on `(seed, step, row id)` and on nothing else: not the batch position, not the thread, not the block size. Two rows that share a probability vector get different draws because their row ids differ. One row gets a fresh draw at every step because the step changes. Move a request to a different slot in the batch, and its tokens don't change, because its row id stays with it.

The lab's `sample_categorical` builds the counter as `(row_id, 0, step, 0)`, with the seed split into the two key words. Its tests draw the same rows in a permuted order with the matching row ids and require the same tokens. A sampler that ignores the row id gives every row the same uniform, so a 64-row batch of a 1000-token uniform distribution returns one token 64 times. The test requires at least 32 distinct tokens. It then draws 4000 rows from $[0.5, 0.25, 0.125, 0.125]$ and requires a chi-square statistic of at most 16.27, the 0.1% critical value for three degrees of freedom.

> [!WARNING] A fixed seed fixes the random number, not the probabilities
> If the logits change in their last bits, the cumulative sum moves, and a draw near a boundary picks a different token. Two batch sizes can use two matrix-multiply kernels whose float sums differ. The generator is reproducible. Whether the model is is a separate question, and it's [learn-inference's](https://qwen.ifkash.dev/c/11-sampling) territory. PyTorch's [randomness notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html) warn along the same lines: a seed doesn't guarantee identical results between CPU and GPU runs.

## What goes wrong

- **An unstable pass.** Equal digits come out in reverse input order, and the sort is wrong only on inputs with ties and only at the second pass. Test with a pass on `[3, 1, 2, 1, 0, 3, 2, 1]` and check the indices, not just the keys.
- **Flipping every bit of a positive float.** The sign handling is reversed, so every positive sorts below every negative. Check `1.0` against `0xBF800000`.
- **Sorting by the raw float key.** A NaN with the sign bit set sorts below $-\infty$, and $-0.0$ sorts before $+0.0$, which breaks the stable order of an equal pair. Use the sort key.
- **A signed shift in the kernel.** `int >> 24` copies the sign bit, so a negative key's top digit is wrong. Read the keys as `uint32_t`. The lab tests a negative key at `shift = 24`.
- **Tile-major scan order.** Scanning the count table tile by tile instead of digit by digit breaks stability, even when every tile ranks its keys correctly.
- **Ties in the merge search.** A strict less-than sends equal elements from $b$ ahead of equal elements from $a$. The merge is still sorted, but it isn't stable.
- **A top-k that keeps all ties.** The threshold is right, and the set has more than $k$ elements. Cut the tied elements at $r$ by a rule.
- **Inclusive scan for top-p.** The token that crosses $p$ is dropped, and the nucleus is one token short. Compare the mass *before* a token with $p$.
- **A shared uniform.** A counter without a row id, or without a step, gives every row, or every step, the same draw.
- **A bitonic sort at a size that isn't a power of two.** The network assumes $n = 2^m$. Pad with the largest key, `0xFFFFFFFF`, and drop the padding after the sort.

> [!RECAP]
> - A radix sort is chapter 7's counting sort, run once per digit from the lowest, and it's correct only because every pass is stable.
> - Signed integers flip the sign bit to become unsigned keys. Floats flip the sign bit when positive and every bit when negative. A sort key also merges the zeros and sends every NaN last.
> - A GPU radix pass counts per tile, scans the counts in digit-major order, and scatters with a per-tile rank. For $2^{26}$ keys, four passes move at least 2.416 GB, a 1.895 ms floor at 1275 GB/s.
> - Merge path cuts the output, not the inputs: a binary search on each anti-diagonal gives every thread the same number of outputs, and less-than-or-equal keeps the merge stable.
> - A bitonic network sorts $n = 2^m$ elements in $m(m+1)/2$ stages with no data-dependent branches, and it fits a shared-memory tile.
> - Top-k and top-p need a threshold, not an order. Radix select finds it in four histogram passes, and the tie rule decides which tied elements stay.
> - A counter-based generator such as Philox makes a draw a function of `(seed, step, row id)`, so a batch can reorder without changing a token.

## Check your understanding

> [!QUESTION] Why does flipping the sign bit sort `int32` correctly, and why isn't that enough for a float?
> Two's complement is monotone inside the negatives and inside the non-negatives, and the sign bit puts the negatives above the positives. Flipping it swaps the halves. A float is sign-and-magnitude, so a negative float with a larger magnitude has a larger bit pattern but a smaller value. Its bits must all be inverted to reverse their order.

> [!QUESTION] The first pass of a radix sort on `[3, 1, 2, 1, 0, 3, 2, 1]` with a 2-bit digit gives the histogram $[1, 3, 2, 2]$. What are the exclusive scan and the destination of the second element?
> The exclusive scan is $[0, 1, 4, 6]$. The second element is a 1, and bin 1 starts at slot 1, so it goes to slot 1. The next 1 goes to slot 2, and the third 1 to slot 3.

> [!QUESTION] Two tiles have digit counts $H[0] = [2, 1]$ and $H[1] = [1, 2]$, as digit-by-tile tables. What are the starting slots for the four (digit, tile) pairs?
> Flatten in digit-major order to $[2, 1, 1, 2]$. The exclusive scan is $[0, 2, 3, 4]$. Digit 0 of tile 0 starts at slot 0, digit 0 of tile 1 at slot 2, digit 1 of tile 0 at slot 3, and digit 1 of tile 1 at slot 4. The total, 6, is the number of keys.

> [!QUESTION] For $a = [1, 3, 5, 7, 9]$ and $b = [2, 3, 4, 8]$, what does the merge-path search return at diagonal 6, and what do the first six outputs look like?
> It returns 3: three of the first six outputs come from $a$, and three from $b$. The first six outputs are $1, 2, 3_a, 3_b, 4, 5$. Elements of $a$ are 1, 3, and 5.

> [!QUESTION] How many stages does a bitonic network use for 1024 elements, and why is the count independent of the data?
> $m = 10$, so $m(m+1)/2 = 55$ stages. The compare-exchange pairs depend only on the stage's $k$ and $j$ and the index, never on the values, and the direction depends on a bit of the index. That is what lets every thread run the same instructions.

> [!QUESTION] Take the eight keys in this chapter's top-k example with $k = 2$. What is the threshold, and which indices do you return?
> The top two keys are `7B` at index 5 and `5F` at index 2. After pass 1, digit 7 has one candidate, so $r$ drops to 1, and digit 5 holds four, so the threshold is in digit 5. Pass 2 over A, F, A, C finds F first, which holds one candidate, at least $r = 1$. The threshold is `5F`, and the set above it is just `7B`. One tied element is needed, and `5F` has one. The result is indices 5 and 2.

> [!QUESTION] With $\pi = [0.375, 0.25, 0.1875, 0.0625, 0.0625, 0.0625, 0]$ and $p = 0.8$, how many tokens does top-p keep?
> Three. The exclusive scan of the sorted probabilities is $[0, 0.375, 0.625, 0.8125, \ldots]$, and only the first three entries are below 0.8. The threshold search gives $t^{\ast} = 0.1875$, since $M(0.1875) = 0.8125 \ge 0.8$ and $M(0.25) = 0.625 < 0.8$. The tokens above it have mass 0.625, which is less than 0.8, so the single token equal to 0.1875 is kept too.

> [!QUESTION] Why does a counter-based sampler need the step in its counter, and what happens to a batch whose rows are reordered?
> Without the step, every step of a row gets the same uniform number, so a row with an unchanged distribution draws the same token forever. With the row id and step in the counter, a draw depends only on `(seed, step, row id)`. Reordering the rows moves each row's output with it and changes no token, as long as each row keeps its id.

## Lab

> [!TRY] Sort, select, and sample
> Open `30-radix-sort-top-k`. Check 1 needs only `int32_to_key`. Then write `float32_to_key` and `key_to_float32`, the sort key `float32_sort_key`, and one stable `radix_pass`. Build `radix_sort` from the pass, and check it against `torch.sort(stable=True)` for values and indices. Add `merge_path_partition` and `merge_path_merge`, then `radix_select_topk` with the lowest-index tie rule, then `top_p_mask` without a full sort. The ninth check is `philox4x32_10` against two known answers and the seeded `sample_categorical`. Nine checks run on the CPU with torch and pure Python. The tenth is a CUDA kernel: `radix_histogram`, the digit histogram of an `int32` tensor, compared with `torch.bincount`. It compiles with `nvcc` on the first run, which takes 30 to 90 seconds, and it's skipped on a machine without a GPU.

The lab reports two metrics:

- `hist_gbs`: the histogram's read rate, counting the $4n$ bytes of keys, for $2^{24}$ keys with a cold L2. The check passes at 20% of the 1275 GB/s copy ceiling, which is 255 GB/s. The ceiling is the byte-count prediction, and the lab measures how close you get.
- `hist_gkeys_per_s`: keys counted per nanosecond. A machine without a GPU records `skipped (no CUDA)` for `hist_gbs`.

## Further reading

- [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0), Hwu, Kirk, and Hajj, fourth edition. The chapters on merge and on sorting are the textbook treatment of merge path and of radix sort on a GPU.
- [Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784), Adinets and Merrill. A single-pass prefix sum lowers the memory reads and writes per digit. The abstract reports 29.4 giga-keys per second for 256 million random 32-bit keys on an A100.
- [CUB `DeviceRadixSort`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceRadixSort.html). The stability guarantee, the key transformations for signed and floating-point keys, `begin_bit` and `end_bit`, and what it does with zeros and NaNs.
- [CUB `DeviceTopK`](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceTopK.html). A device-wide top-k based on AIR TopK, with a contract for ties and ordering that depends on the mode you request.
- [Sorting-free GPU kernels for LLM sampling](https://flashinfer.ai/2025/03/10/sampling.html), FlashInfer. Rejection sampling with a pivot for top-k and top-p, a dual-pivot variant, and the note that float prefix sums may not be monotonic.
- [Parallel Random Numbers: As Easy as 1, 2, 3](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf), Salmon, Moraes, Dror, and Shaw. The paper behind Philox and Threefry. The PDF could not be read when this chapter was written, so the chapter doesn't quote it. The lab's known-answer vectors check the constants instead.
- [cuRAND device API](https://docs.nvidia.com/cuda/curand/device-api-overview.html). `Philox_4x32_10` with a seed, a subsequence, and an offset.
- [GPU MODE, lecture 19: Data Processing on GPUs](https://github.com/gpu-mode/lectures), Devavret Makkar. The lecture index lists lectures 20 and 21 as the scan sessions and lecture 24 as "Scan at the Speed of Light", which is the look-back scan that Onesweep builds on. I read the titles in the index, not the slides.
- [Chapter 7](/c/07-scan-and-histogram), for the scan, the histogram, and the counting sort this chapter repeats. [Chapter 20](/c/20-moe-and-sparse-kernels), for the token permutation that uses a counting sort. [learn-inference's sampling chapter](https://qwen.ifkash.dev/c/11-sampling), for sampling inside a serving loop.
