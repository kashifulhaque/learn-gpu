---
title: Sparse and irregular work
slug: 32-sparse-and-irregular-work
part: "Part 2 — CUDA patterns"
summary: Store a sparse matrix in COO, CSR, ELL, or a hybrid, split an SpMV so every warp gets equal work with merge path, traverse a graph with a frontier and direction optimization, and hand out uneven tasks from an atomic queue.
minutes: 100
gpu: true
objectives:
  - "Compute the byte cost of COO, CSR, ELL, and hybrid storage for a matrix, and say which one a power-law matrix breaks."
  - "Write CSR SpMV as thread-per-row, warp-per-row, and merge-path kernels, and say where each loses on a power-law matrix."
  - "Measure load imbalance as max over mean work per warp, and predict it for each SpMV scheme from the row lengths."
  - "Run a frontier-based BFS with atomics, and apply the direction-optimizing switch rule to choose between top-down and bottom-up rounds."
  - "Hand out uneven tasks from a persistent kernel with an atomic work counter, and bound its finish time."
lab: 32-spmv-bfs
---

# Sparse and irregular work

> [!TLDR]
> - A sparse matrix stores only its nonzeros, and the format decides how many bytes that takes. CSR costs $4(m+1) + 8z$ bytes for $m$ rows and $z$ nonzeros in float32. ELL pads every row to the longest, and on a power-law matrix that costs 135 times more than CSR.
> - SpMV does 2 FLOP per nonzero and moves about 9 bytes per nonzero. It is bandwidth bound by a factor near 46 on an A100, so the work is keeping the memory system busy, not the arithmetic.
> - Thread-per-row lets the longest row in a warp set the warp's time. Warp-per-row wastes lanes on short rows. On the lab's seeded matrix the max-over-mean work per warp is 11.01, 1.96, and 1.002 for thread-per-row, warp-per-row, and merge path.
> - Merge path gives every worker an equal share of rows plus nonzeros, found by a binary search on a diagonal, so a row of any length never stalls a warp. A BFS does the same job with a frontier and an atomic claim, and direction optimization swaps to a bottom-up round when the frontier is large.
> - A persistent kernel pulls tasks from an atomic counter. On the lab's skewed task set it finishes in 119 time units where static assignment takes 800.

Everything so far in Part 2 had a regular shape. A reduction sums $n$ numbers, a scan prefixes them, and a transpose moves a rectangle. Every thread knows its work from its index, and every warp does the same amount. Sparse data breaks that. A row of a sparse matrix, a vertex of a graph, or an expert's batch of tokens has a length that nobody knows until the data arrives, and the lengths of real data are rarely close to each other.

That matters for machine learning more often than the word *sparse* suggests. A graph neural network aggregates over neighbors with an SpMV. A recommender looks up embeddings for a ragged set of ids. A mixture-of-experts layer, from [chapter 20](/c/20-moe-and-sparse-kernels), routes a different number of tokens to each expert. In each case the arithmetic is simple and the question is how to split uneven work evenly across 108 SMs that hold up to 6,912 resident warps.

This chapter builds three answers in the order they were found. Give each thread a row, and watch the longest row set the time. Give each warp a row, and watch short rows waste lanes. Then split the work, not the rows, so the shares come out equal. A BFS and a work queue then apply the same idea to a graph and to a pile of tasks.

## Before you start

You need three tools from earlier chapters:

- **A warp reads 32-byte sectors.** A load is cheap when the 32 lanes read adjacent addresses and expensive when each lane lands in its own sector. ([Chapter 3](/c/03-memory-and-bandwidth).)
- **A warp reduces with shuffles.** `__shfl_down_sync` halves the offset from 16 to 1 and leaves the sum in lane 0. The warp-per-row kernel below reuses that tree. ([Chapter 6](/c/06-reductions).)
- **An atomic claims a slot, and a scan turns counts into offsets.** An `atomicAdd` returns the old value, so each caller gets a unique number. An exclusive scan of row lengths is how `row_ptr` is built. ([Chapter 7](/c/07-scan-and-histogram).)

## Three formats and what they cost in bytes

This section answers one question: how many bytes does each sparse format store, and which matrices break it?

Picture a mostly empty grid with a few dots in it. A sparse format is a rule for writing down only the dots. The simplest rule writes down each dot's row, column, and value. That is *COO*, coordinate format, and it stores three arrays of length $z$, the number of nonzeros.

COO repeats every row index once per entry in the row. *CSR*, compressed sparse row, removes the repetition. It sorts the entries by row and then column, and it replaces the $z$ row indices with $m+1$ offsets, `row_ptr`. Row $i$ owns the half-open range `row_ptr[i]` to `row_ptr[i+1]` of the column and value arrays. An empty row has two equal offsets. *ELL*, ELLPACK, goes the other way: it pads every row to the length of the longest, $w$, and stores two dense $m \times w$ arrays, which gives the kernel a regular shape.

Take the six-row matrix whose rows hold 3, 0, 1, 5, 0, and 1 entries, so $z = 10$. Its `row_ptr` is `[0, 3, 3, 4, 9, 9, 10]`. COO stores 10 triples of 12 bytes, 120 bytes. CSR stores 7 offsets and 10 pairs, $7 \times 4 + 10 \times 8 = 108$ bytes. ELL pads every row to 5 entries and stores $6 \times 5 \times 8 = 240$ bytes, of which 20 entries are padding.

With 4-byte indices and 4-byte values, the three costs are

$$
\boxed{B_{\text{COO}} = 12\,z, \qquad B_{\text{CSR}} = 4(m+1) + 8\,z, \qquad B_{\text{ELL}} = 8\,m\,w}
$$

Read the formulas as three questions: does the cost follow $z$ (COO), follow $z$ plus a small per-row term (CSR), or follow the longest row (ELL)? Only ELL's cost depends on the worst row, and that is its weakness.

The lab's seeded matrix shows how large the weakness is. It has 4,096 rows and 4,096 columns, 32,838 nonzeros, a mean row of 8.02 entries, and a longest row of 1,153. The row lengths follow a heavy-tailed Pareto distribution, which is the shape of a social graph's degrees or an embedding table's popularity. Most rows are short, and a few are enormous.

| Format | Formula | Bytes on the lab matrix | Against CSR |
|---|---|---|---|
| COO | $12z$ | 394,056 | 1.41x |
| CSR | $4(m+1) + 8z$ | 279,092 | 1.00x |
| ELL | $8mw$, with $w = 1153$ | 37,781,504 | 135.4x |
| Hybrid | ELL of width 4, plus COO overflow | 383,288 | 1.37x |

ELL stores 135 times the bytes of CSR because one row of 1,153 entries forces 4,095 other rows to carry 1,153 slots each. That is the reason ELL exists only for matrices whose rows have similar lengths, such as a finite-difference stencil or a mesh with bounded degree.

The *hybrid* format, from [Bell and Garland's 2008 report](https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf), keeps ELL's regular shape for the typical rows and spills the rest into COO. Their rule, which the lab implements as `hyb_width`, adds a $K$-th ELL column while at least one third of the rows have $K$ or more entries. On the six-row example that gives $K=3$: two of six rows reach 3 entries, which is exactly one third, and only one row reaches 4. The ELL part costs $6 \times 3 \times 8 = 144$ bytes, the two overflow entries of row 3 go to COO at 12 bytes each, and the total is 168 bytes. On the lab matrix the rule gives $K=4$ and 383,288 bytes.

> [!NOTE] Build CSR from COO by sorting and summing
> Data usually arrives as COO, with entries in any order and sometimes a repeated coordinate. Build CSR in three steps: add the values of repeated `(row, col)` pairs, sort by row and then column, and take a running sum of the per-row counts to get `row_ptr`. PyTorch's `coalesce()` does the first two steps, and `to_sparse_csr()` does the third. The lab's first check compares your `coo_to_csr` with that pair of calls.

A dense matrix would take $4mn$ bytes. The lab matrix is $4096 \times 4096$, so dense is 67,108,864 bytes, and CSR's 279,092 is 0.4% of that. CSR is the format the rest of the chapter uses. `torch.sparse` stores it, cuSPARSE reads it, and a graph is exactly a CSR matrix with one row per vertex and the neighbors in the columns.

## SpMV moves about 9 bytes per nonzero

This section answers one question: what limits a sparse matrix-vector product, and what time does that predict?

The product is $y = Ax$, with $A$ a sparse $m \times n$ matrix in CSR. Every nonzero does one multiply and one add, 2 FLOP. To do that it reads a column index (4 bytes), a value (4 bytes), and one element of $x$ (4 bytes) picked by the column index. The rows write $m$ outputs and read $m+1$ offsets.

Put numbers on the lab's GPU matrix. It is generated the same way as the small one, with 524,288 rows, 524,288 columns, and rows capped at 65,536 entries. It has 5,183,261 nonzeros, a mean row of 9.89 entries, and a longest row of 65,536. If the vector $x$ is read once from DRAM and cached after that, the compulsory traffic is

$$
B = 4(m+1) + 8z + 4n + 4m = 47{,}757{,}548\ \text{bytes},
$$

which is 9.21 bytes per nonzero. The work is $2z = 10{,}366{,}522$ FLOP, so the arithmetic intensity is $2z/B = 0.217$ FLOP per byte.

Compare that with the machine. The A100's float32 ridge point is $19.5\ \text{TFLOP/s} / 1.935\ \text{TB/s} = 10.1$ FLOP per byte, and 0.217 is 46 times below it. SpMV is a bandwidth problem. At the 1275 GB/s copy ceiling measured in learn-inference (its chapter 10), the floor is $47{,}757{,}548 / 1.275 \times 10^{12} = 0.0375$ ms. This is a prediction from the specs, and the lab measures the real figure. The number to keep is the shape of the result: a good SpMV kernel runs at a large fraction of copy bandwidth, and every kernel below is judged by how close it gets.

Two things push a real kernel above that floor. The gathers `x[col]` are not coalesced, because the columns of one row are arbitrary, so each gather can pull in a full 32-byte sector for a 4-byte value. The vector here is 2 MiB, which fits in the 40 MiB L2, so most gathers hit the cache, but a larger vector would not. The second cost is the topic of the rest of the chapter. Even when the bytes are right, the warps may not finish together, and the kernel runs as long as the last warp.

## Thread per row

This section answers one question: what happens when each thread owns one row?

Picture 32 lanes of a warp standing in a row, each holding one matrix row. Every lane loops over its own entries. The warp's 32 lanes execute the same instruction together, so the loop runs for as many iterations as the longest of the 32 rows, and lanes whose rows ended sit idle. This is the *scalar* kernel in Bell and Garland's terms.

```cuda
__global__ void tpr_kernel(int n_rows, const int* rp, const int* ci,
                           const float* v, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float acc = 0.0f;
    for (int k = rp[row]; k < rp[row + 1]; ++k)
        acc += v[k] * x[ci[k]];
    y[row] = acc;
}
```

The kernel is the shortest of the three, and it has two costs on a skewed matrix. The first is the memory pattern. At iteration $t$, lane $l$ reads `v[rp[row_l] + t]`. The 32 addresses are separated by the row lengths, about 32 bytes apart when the mean row is 8 entries. Each lane lands in its own 32-byte sector, where chapter 3's coalesced load would have used 4 sectors for 32 floats. Later iterations may find their sector still in L1, so the cost is cache pressure and more sectors in flight, not a flat 8 times the traffic.

The second cost is divergence in time, and it is the bigger one. The warp's iteration count is the maximum row length in its 32 rows, so the warp runs as long as its longest row. Define the work of a warp as that iteration count. On the lab matrix, 128 warps own 32 consecutive rows each. A warp that holds the 1,153-entry row runs 1,153 iterations. The mean warp runs 104.7. The warp that holds the longest row sets the kernel's time, while a typical warp finishes in a tenth of it and then waits.

Counting useful lane-steps against the 32 lanes times the iterations each warp runs gives the *lane use*. For thread-per-row on the lab matrix it is $32{,}838 / (32 \times 13{,}404) = 0.077$: about 8% of the lane slots hold real work. The remaining 92% either idle behind a longer row or run a lane whose row ended.

## Warp per row

This section answers one question: what does a whole warp on one row fix, and what does it break?

Give a whole warp one row. The 32 lanes read 32 consecutive entries of that row, which is one coalesced 128-byte load of values and one of column indices, and then reduce their partial sums with a shuffle tree. This is the *vector* kernel in Bell and Garland, and it is chapter 6's warp reduction applied once per row.

```cuda
__global__ void wpr_kernel(int n_rows, const int* rp, const int* ci,
                           const float* v, const float* x, float* y) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int warps = (gridDim.x * blockDim.x) >> 5;
    for (int row = warp; row < n_rows; row += warps) {
        float acc = 0.0f;
        for (int k = rp[row] + lane; k < rp[row + 1]; k += 32)
            acc += v[k] * x[ci[k]];
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) y[row] = acc;
    }
}
```

A row of 1,153 entries now takes $\lceil 1153 / 32 \rceil = 37$ iterations instead of 1,153, and the loads are coalesced. A grid-stride loop over rows deals row $i$ to warp $i \bmod W$, so no warp is pinned to a region of the matrix.

The kernel loses in two places. The first is short rows. A row of 3 entries uses 3 of 32 lanes for one iteration, then runs the 5-step shuffle tree for a sum of 3 numbers. The matrix's mean row is 8.02 entries, and 97% of its rows have 32 or fewer, so most iterations use a fraction of a warp. The lane use is $32{,}838 / (32 \times 4{,}499) = 0.228$, three times better than thread-per-row and still short of full.

The second loss is the long row. A warp on a row of 1,153 entries still runs 37 iterations while a typical warp, with its 32 short rows of at least one iteration each, runs about 32. On the lab matrix the work per warp ranges from 32 to 69, so the max over mean is 1.96, and the 65,536-entry row of the GPU matrix would take one warp 2,048 iterations. Warp-per-row spreads a row across 32 lanes, not across the machine, so a single enormous row is still a serial bottleneck.

> [!KEY] Each fixed assignment has a matrix that beats it
> Thread-per-row loses when row lengths vary inside a warp. Warp-per-row loses when rows are shorter than 32 and when one row is far longer than the rest. A kernel that assigns work by row can't do better than the row-length distribution allows. Fixing that means assigning work by something other than rows.

## Measuring load imbalance

This section answers one question: how do you put one number on how unevenly a kernel splits its work?

Count the work each warp does, in iterations of that warp's loop. Then divide the largest count by the mean:

$$
\boxed{\text{imbalance} = \frac{\max_w W_w}{\tfrac{1}{P}\sum_w W_w}}
$$

Here $W_w$ is the work of warp $w$ and $P$ is the number of warps. The value 1 means every warp does the same work and nothing waits. A value of 11 means the slowest warp does 11 times the average, so on a machine that runs all warps at once the kernel takes 11 times as long as perfect balance, and 91% of the warp-time is spent waiting.

The reason this is the right ratio is the roofline for waiting. All warps start together, the kernel ends when the last one does, and the mean is the time the kernel would take with perfect sharing. The ratio is the slowdown from imbalance alone, before any memory effect.

Here are the three schemes on the lab's 4,096-row matrix with 128 warps. The model is the one the lab's `warp_work` implements: thread-per-row takes the maximum row length in each group of 32 rows, warp-per-row sums $\lceil \text{length}/32 \rceil$ over its rows with a minimum of 1, and merge path takes $\lceil \text{share}/32 \rceil$.

| Scheme | Max iterations | Mean iterations | Max / mean | Lane use |
|---|---|---|---|---|
| Thread per row | 1,153 | 104.7 | 11.01 | 0.077 |
| Warp per row | 69 | 35.2 | 1.96 | 0.228 |
| Merge path | 10 | 9.98 | 1.002 | n/a |

The counts are a lockstep model, not a measurement, and the absolute numbers across schemes aren't comparable: a merge-path iteration does a different amount of work from a warp-per-row iteration. The ratios within a row of the table are the point. Use the model before you write the kernel to see whether imbalance is your problem.

```viz
32-warp-work
```

The figure draws all 128 warps for each scheme. Thread-per-row is a few spikes over a low floor, and every spike is a row of hundreds of entries. Warp-per-row has a floor of 32 because each warp owns 32 rows and each row costs at least one iteration. Merge path is a flat line.

## Merge path

This section answers one question: how do you split an SpMV so every warp gets the same amount of work, whatever the row lengths?

The idea, from Merrill and Garland's merge-based SpMV, is to split the work and not the rows. Picture two sorted lists being merged. List A is the *row end offsets*, `row_ptr[1:]`, one entry per row. List B is the nonzero indices $0, 1, 2, \ldots, z-1$. Merging them walks a path through a grid whose horizontal axis counts rows finished and whose vertical axis counts nonzeros consumed. At each step, the path takes a row end when that row's end offset is at most the next nonzero's index, and it takes a nonzero otherwise.

That path has $m + z$ steps, and each step is one unit of work: either "finish a row" (write an output) or "consume a nonzero" (one multiply-add). A step on the path is exactly the work that has to be done, and the path doesn't care how the nonzeros are grouped into rows. Cut the path into $P$ equal pieces of $\lceil (m+z)/P \rceil$ steps, give one piece to each worker, and every worker has equal work by construction.

The remaining problem is to find where piece $w$ starts without walking the path. The point after $d$ steps lies on the *diagonal* $i + j = d$, where $i$ is the number of rows finished and $j$ is the number of nonzeros consumed. A binary search finds the one point of that diagonal that lies on the path.

Take the six-row example, `row_ptr = [0, 3, 3, 4, 9, 9, 10]`, with $m + z = 16$ steps, split among 4 workers of 4 steps each. The cut points at diagonals 0, 4, 8, 12, and 16 are, as (rows finished, nonzeros consumed), $(0,0)$, $(1,3)$, $(3,5)$, $(3,9)$, and $(6,10)$. Each piece covers exactly 4 steps:

| Worker | From | To | Rows finished | Nonzeros consumed |
|---|---|---|---|---|
| 0 | (0, 0) | (1, 3) | 1 | 3 |
| 1 | (1, 3) | (3, 5) | 2 | 2 |
| 2 | (3, 5) | (3, 9) | 0 | 4 |
| 3 | (3, 9) | (6, 10) | 3 | 1 |

Worker 2 finishes no row at all, and worker 3 finishes three, one of them empty. Row 3, with its five entries at positions 4 to 8, is split across workers 1, 2, and 3. Splitting two rows per worker would hand worker 1 the whole of row 3 and leave worker 2 with one entry in total. The nonzero counts per worker, 3, 2, 4, and 1, are uneven, but the steps per worker are not, and a step is what costs time.

The binary search is short. On diagonal $d$, the row count $i$ lies between $\max(0, d - z)$ and $\min(d, m)$. Compare row end offset $A_{\text{mid}}$ with the nonzero index $B$ that the path would pair against it, which is $d - 1 - \text{mid}$:

$$
\boxed{\text{row } \text{mid} \text{ is behind the diagonal} \iff \texttt{row\_ptr}[\text{mid}+1] \le d - 1 - \text{mid}}
$$

If the row is behind, the search moves right, and otherwise left. It takes $\lceil \log_2 \min(d, m) \rceil$ steps, which is about 19 for the GPU matrix's half million rows. The lab's `merge_path_coordinate` is this search, and its check compares every boundary with a brute-force walk of the path.

```python
def merge_path_coordinate(row_ptr, diagonal):
    n_rows, nnz = len(row_ptr) - 1, row_ptr[-1]
    lo, hi = max(0, diagonal - nnz), min(diagonal, n_rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if row_ptr[mid + 1] <= diagonal - 1 - mid:
            lo = mid + 1
        else:
            hi = mid
    return lo, diagonal - lo
```

Each worker then walks its piece. At every step, if the current row's end offset is at most the current nonzero index, the row ends: write the running sum to `y[i]`, reset it, and move to the next row. Otherwise add `v[j] * x[ci[j]]` and move to the next nonzero. A worker's last row may continue into the next worker's piece, as row 3 does above, so the worker adds its leftover running sum to `y` at the end.

```cuda
for (long long d = d0; d < d1; ++d) {
    if (i < n_rows && rp[i + 1] <= j) {
        atomicAdd(&y[i], running);   // row i ends here
        running = 0.0f;
        ++i;
    } else {
        running += v[j] * x[ci[j]];  // one nonzero
        ++j;
    }
}
if (i < n_rows && running != 0.0f) atomicAdd(&y[i], running);   // carry-out
```

The kernel uses `atomicAdd` for every row end, which is the simplest correct choice: a row that spans workers receives one add from each, and `y` starts at zero. The cost is that float addition isn't associative, so a spanning row's last bits can vary from run to run. A production kernel writes interior rows with a plain store and sends only the boundary rows' carry-outs to a second pass or an atomic. The lab's tolerance allows for the reordering.

The lab check for imbalance asks for merge path under 1.05, and the model gives 1.002. The last worker holds a shorter piece when $m+z$ isn't a multiple of $P$, which is the only imbalance left. Merge path loses nowhere on balance, and it pays elsewhere: two binary searches per worker, an atomic per row end, and a loop with a data-dependent branch on every step. For a matrix with uniform short rows, such as a stencil, ELL or a plain row kernel might win, because they skip the searches and the atomics.

> [!WARNING] Splitting by rows alone, or by nonzeros alone, is the plausible wrong merge path
> An equal number of rows per worker gives the worker with the long row most of the nonzeros. An equal number of nonzeros per worker gives one worker every empty row. Both hand out a count of the wrong thing. On the six-row example, two rows per worker gives the workers 3, 6, 1, and 0 nonzeros, a max over mean of 2.4. The lab compares your partition with a brute-force walk of the path.

## Where the formats and kernels fit together

This section answers one question: which format and kernel should you pick for which matrix?

| Matrix | Row lengths | Format and kernel | Why |
|---|---|---|---|
| Finite-difference stencil, bounded-degree mesh | Nearly equal | ELL, one thread per row, column-major | Padding is small, and lane $l$ reads adjacent addresses |
| Rows of about 32 or more entries | Moderate spread | CSR, warp per row | Coalesced loads, and the shuffle tree is amortized |
| Mostly short rows, a few very long | Power law | CSR with merge path, or a hybrid | Balance does not depend on any row |
| Graph adjacency | Power law, often empty rows | CSR with merge path or frontier-based traversal | Same as above, and rows are vertices |

ELL's regular shape comes with a layout trick. Store the $m \times w$ arrays column-major, so the $k$-th entries of consecutive rows are adjacent. Then at iteration $k$, thread-per-row's 32 lanes read 32 adjacent floats, which is the coalesced pattern from chapter 3. ELL turns the thread-per-row kernel's worst memory pattern into its best one, as long as the padding stays small.

The decision isn't permanent. The row-length histogram is cheap to compute with chapter 7's histogram, and a library can choose per matrix. The lab doesn't ask you to choose. It asks you to measure the cost of each choice on one matrix.

## Frontier-based BFS with atomics

This section answers one question: how do you traverse a graph in parallel when each step's work depends on the step before?

A *breadth-first search*, BFS, labels every vertex with its distance from a source. Picture dropping a stone in water. The first ring is the source's neighbors, the second ring is their unvisited neighbors, and so on. The set of vertices in the current ring is the *frontier*. Each round expands the frontier to the next one, and the rounds are sequential: you can't expand ring 3 before ring 2 is complete. Inside a round, every frontier vertex expands independently, which is where the parallelism is.

A graph in CSR is an SpMV matrix with the neighbors in the columns. For vertex $u$, `col_idx[row_ptr[u] : row_ptr[u+1]]` lists its neighbors. A top-down round loops over the frontier and, for each neighbor $v$ that has no level yet, sets `level[v] = depth + 1` and appends $v$ to the next frontier. The lab's reference BFS from vertex 0 on a 1,520-vertex preferential-attachment graph reaches 1,500 of its vertices in 4 levels. The other 20 vertices form a detached path and stay at level $-1$.

The trouble is that two frontier vertices can share a neighbor. If both threads read `level[v] == -1` and both write, then $v$ lands in the next frontier twice. The fix is an atomic claim. `atomicCAS(&level[v], -1, depth + 1)` sets the level only if it is still $-1$ and returns the old value, so exactly one thread sees $-1$. That thread then takes a slot in the next frontier with `atomicAdd` on a counter.

```cuda
__global__ void bfs_top_down(const int* rp, const int* ci, int* level,
                             const int* frontier, int frontier_size,
                             int* next, int* next_size, int depth) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frontier_size) return;
    int u = frontier[t];
    for (int k = rp[u]; k < rp[u + 1]; ++k) {
        int v = ci[k];
        if (atomicCAS(&level[v], -1, depth + 1) == -1)   // one winner per vertex
            next[atomicAdd(next_size, 1)] = v;           // unique slot
    }
}
```

The two atomics are the two jobs from chapter 7: the compare-and-swap decides who wins, and the add hands out a slot, the same trick as a histogram bin's counter. The host launches one kernel per level, reads `next_size`, and swaps the two frontier buffers.

This kernel has the same load imbalance as thread-per-row SpMV, because thread $t$ expands one vertex whose degree may be 85 or 3. The cures are the ones already seen. A warp per frontier vertex handles hubs, and merge path over the frontier's concatenated edge lists balances a mixed frontier. On a power-law graph the hubs are exactly the high-degree vertices that every frontier eventually contains, which is why the imbalance comes back.

The check against a plain queue BFS is the hard test in the lab. Your `bfs_levels_frontier` has to give the same level for every vertex as a one-vertex-at-a-time queue, including the 20 unreached vertices.

## Direction optimization

This section answers one question: when does a top-down round do more work than it needs to?

Look at the lab graph's rounds. The frontier at level 3 has 766 vertices, with 3,479 edges. A top-down round examines all 3,479 edges, almost all of them leading to a vertex that is already visited: only 63 vertices remain unvisited. The round does a great deal of work to find a handful of new vertices. Beamer, Asanović, and Patterson's [direction-optimizing BFS](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.html) turns the round around.

In a *bottom-up* round, every unvisited vertex scans its neighbors and stops at the first one that is in the frontier. If it finds one, it joins level $\text{depth}+1$. The round writes only to the scanning vertex itself, so it needs no atomics: each unvisited vertex has one writer. And a vertex stops after its first hit, so the cost on a graph with a big frontier is small. The technical report describes the trade: top-down costs in proportion to the frontier, bottom-up costs in proportion to the unvisited vertices divided by the frontier size, and the two are complementary. Bottom-up needs the graph to be undirected, so a vertex's neighbors are also the vertices that point to it, and it wants the frontier as a bitmap for a constant-time membership test.

Here are the per-level costs on the lab graph, each computed from the same state. The *edges examined* for top-down is $m_f$, the sum of the frontier's degrees, and for bottom-up it is the count of edges the unvisited vertices scan.

| Level | Frontier | Top-down edges | Bottom-up edges | Cheaper |
|---|---|---|---|---|
| 0 | 1 | 85 | 8,028 | Top-down |
| 1 | 85 | 992 | 4,659 | Top-down |
| 2 | 605 | 4,282 | 1,094 | Bottom-up |
| 3 | 766 | 3,479 | 81 | Bottom-up |
| 4 | 43 | 144 | 38 | Bottom-up |

Top-down wins while the frontier is small, bottom-up wins in the middle, and at the end the frontier is small again but the few unvisited vertices make bottom-up cheap anyway. The cheapest choice at every level totals 85 + 992 + 1,094 + 81 + 38 = 2,290 edges, against 8,982 for top-down alone. A rule has to pick without seeing the other column.

```viz
32-bfs-directions
```

The rule the lab implements compares two quantities that a BFS knows before it runs the round. Let $m_f$ be the sum of the frontier's degrees, $m_u$ the sum of degrees of the vertices not yet visited, $n_f$ the frontier size, and $n$ the vertex count. From top-down, switch to bottom-up when the frontier has a large share of the unexplored edges. From bottom-up, switch back when the frontier shrinks:

$$
\text{top-down} \to \text{bottom-up} \text{ when } \hla{\alpha\, m_f > m_u}, \qquad \text{bottom-up} \to \text{top-down} \text{ when } \hlb{\beta\, n_f < n}
$$

Read it as two thresholds. $\hla{\alpha\, m_f > m_u}$ says that once the frontier's edges are within a factor $\hla{\alpha}$ of everything left, the round is going to touch most of the graph, so scan from the other side. $\hlb{\beta\, n_f < n}$ says that once the frontier is under $n/\hlb{\beta}$ vertices, top-down is cheap again. The two constants depend on the graph. Beamer's report also controls the switch with cutoffs on the current and predicted frontier size, and the lab's $\alpha$ and $\beta$ are one simple form of that.

The constants matter, and the lab shows how much. With $\beta = 18$, sweep $\alpha$ on the lab graph:

| $\alpha$ | Directions by level | Edges examined |
|---|---|---|
| 1, 2, 4, 8 | top-down, top-down, bottom-up, bottom-up, top-down | 2,396 |
| 15, 30 | top-down, bottom-up, bottom-up, bottom-up, top-down | 6,063 |

With $\alpha \le 8$ the rule picks the cheaper direction at four of the five levels and examines 2,396 edges, 3.75 times fewer than top-down's 8,982. The count of 2,396 is 106 above the per-level minimum of 2,290, because at level 4 the rule has already switched back to top-down, which costs 144 edges where bottom-up would cost 38. At $\alpha = 15$ the rule switches one level early, at level 1, where bottom-up costs 4,659 against top-down's 992, and examines 6,063 edges. A threshold tuned on one graph is not portable, and the lab's default of 4 is a choice for this graph, not a constant from a paper. Treat $\alpha$ as a parameter to sweep on your own data.

> [!TIP] Count edges before you time kernels
> The edge counts above come from a Python loop and need no GPU. When a traversal is slow, count the edges each direction would examine per level first. If the counts say bottom-up saves a factor of 3, a GPU kernel can at best deliver that factor, and if they say it saves nothing, no kernel will.

## A persistent kernel and an atomic work queue

This section answers one question: when you can't predict how long a task takes, how do you keep every SM busy until the end?

Picture 8 checkout lanes and 64 customers. Assigning customer $i$ to lane $i \bmod 8$ in advance is *static assignment*, and it works when every customer takes the same time. If every eighth customer has a cart of 100 items and the others have 1, lane 0 gets all the big carts and finishes at 800 while the other lanes finish at 8. A shared queue fixes that. Each lane takes the next customer when it is free, and the heavy carts spread across lanes.

On a GPU, the shared queue is a single counter in global memory and the lanes are *persistent* thread blocks. A normal kernel launches one block per task and lets the hardware scheduler place them. A persistent kernel launches only as many blocks as fit on the GPU at once, about one wave, and each block loops: take a task number from the counter with `atomicAdd`, do the task, and repeat until the counter passes the task count.

```cuda
__device__ int queue_head;   // zeroed before the launch

__global__ void persistent(int n_tasks /* , task data */) {
    __shared__ int task;
    while (true) {
        if (threadIdx.x == 0) task = atomicAdd(&queue_head, 1);
        __syncthreads();                    // everyone sees the same task
        if (task >= n_tasks) break;
        run_task(task);                     // the block does this task together
        __syncthreads();                    // before thread 0 overwrites `task`
    }
}
```

One thread fetches the number and a barrier shares it, so a block of 256 threads does one atomic per task, not 256. The counter hands out tasks in order and the earliest-free block takes the next one, which is a list-scheduling policy and has a clean bound. If the tasks cost $c_i$, the finish time on $W$ workers satisfies

$$
\boxed{T \;\le\; \frac{\sum_i c_i}{W} + c_{\max}\left(1 - \frac{1}{W}\right)}
$$

The first term is perfect sharing, and the second is the price of the last task starting late. The reasoning is short. When the last-finishing task starts, every worker has been busy the whole time, so that start time is at most $(\sum c_i - c_{\text{last}})/W$, and the task then runs for $c_{\text{last}}$. The bound says the queue is never worse than perfect sharing plus almost one largest task.

The lab simulates this on CPU with a heap in place of the GPU. The task set has 64 tasks, with every eighth costing 100 and the rest costing 1. With 8 workers, the static assignment gives worker 0 all eight heavy tasks, for a finish time of 800. The bound for the dynamic queue is $(800 + 56)/8 + 100 \times 7/8 = 194.5$. The dynamic queue's finish time is 119, a factor 6.7 better than static, and under the bound.

The queue has costs of its own. Every `atomicAdd` on one address serializes in the L2, so a queue of very short tasks makes the counter the bottleneck. The usual fix is to take a chunk of tasks per fetch, for example `atomicAdd(&queue_head, 16)`, which trades some end-of-run imbalance for fewer atomics. A persistent kernel also keeps its blocks resident, which means other kernels can't run until it finishes, and it has to avoid blocks that wait on each other. The [Triton grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html), which [chapter 20](/c/20-moe-and-sparse-kernels) points to, is a persistent kernel, and the work queue above is a dynamic way to hand out its tiles.

> [!KEY] Count work, hand it out when it's ready
> Both merge path and the work queue turn an uneven distribution into equal shares. Merge path does it before the launch by computing the shares from the row offsets. The queue does it during the run by letting workers ask for more. Use merge path when the costs are known up front, as row lengths are. Use the queue when they aren't, as with a recursive task or an unknown-length decode.

## Where 2:4 sparsity fits

This section answers one question: is a pruned weight matrix an SpMV?

No. A pruned weight matrix with a regular 2:4 pattern, at most 2 nonzeros in every group of 4, is *structured* sparsity, and the Ampere sparse tensor cores read it directly from a compressed form with 2-bit metadata. There is no `row_ptr` and no gather, and the hardware does the selection. Unstructured sparsity, the kind CSR handles, rarely beats a dense tensor-core matmul unless more than 93.75% of the weights are zero, as [chapter 20](/c/20-moe-and-sparse-kernels) works out.

So the kernels in this chapter fit data that is sparse because of its nature, such as graphs, adjacency, and ragged lookups, and not weights that were pruned on purpose. Chapter 20 owns the 2:4 format, its metadata, and its measured speedups. This chapter doesn't repeat them.

## What goes wrong

- **The row end off by one.** The row's range is half-open, so the loop runs to `row_ptr[i+1]` and not to `row_ptr[i+1] - 1`. The off-by-one drops the last entry of every row. The result is wrong only a little on a long row and exactly 0 on a row of one entry, which is why the lab's SpMV check includes single-entry rows.
- **Duplicate coordinates left in a COO matrix.** `coo_to_csr` has to sum repeated `(row, col)` pairs. A build that keeps both entries gives two entries in one column, and the SpMV still adds them correctly, but the matrix no longer equals what `coalesce()` returns.
- **Splitting by rows, or by nonzeros, and calling it merge path.** Both give equal counts of one thing and uneven work. The lab's brute-force partition check catches it.
- **A bottom-up round that labels vertices as it goes.** If a vertex joins the frontier the moment it finds a parent, and any labeled neighbor counts as a parent, a chain of vertices joins one round and the levels come out too small. Label after the scan, and accept only a neighbor at the current depth.
- **Bottom-up on a directed graph.** Bottom-up asks whether a vertex's neighbors are in the frontier, which finds a frontier vertex that points to it only if the edges are stored both ways.
- **One `alpha` for every graph.** The sweep above moves the cost from 2,396 to 6,063 edges on a 1,520-vertex graph. Sweep it on your own graph.
- **Comparing SpMV times to the dense floor.** The roofline floor of 0.0375 ms for the lab's GPU matrix counts $x$ once. A kernel can be near it only when the gathers hit the cache, so a matrix with a much larger $x$ has a higher floor.
- **A persistent kernel that spins on a dependency.** A block that waits for a task another block must finish can deadlock when not every block is resident. Keep tasks independent.

> [!RECAP]
> - CSR costs $4(m+1) + 8z$ bytes. ELL costs $8mw$ and breaks on a power-law matrix, where $w$ is the longest row. A hybrid keeps ELL for typical rows and spills the rest to COO.
> - SpMV does 2 FLOP per nonzero for about 9 bytes, an intensity of 0.217 FLOP per byte. The GPU matrix's floor is 0.0375 ms at the 1275 GB/s copy ceiling, a prediction and not a measurement.
> - Imbalance is max over mean work per warp. On the lab matrix it is 11.01 for thread-per-row, 1.96 for warp-per-row, and 1.002 for merge path.
> - Merge path cuts the $m + z$ steps of the row-end and nonzero merge into equal pieces, found by a binary search on each diagonal, and fixes up rows that span workers.
> - A frontier BFS claims a vertex with `atomicCAS` and a slot with `atomicAdd`. Direction optimization examines 2,396 edges instead of 8,982 on the lab graph when $\alpha \le 8$.
> - An atomic work counter finishes within $\sum c_i / W + c_{\max}(1 - 1/W)$, which is 119 against a static 800 on the lab's task set.

## Check your understanding

> [!QUESTION] A matrix has 1,000 rows. One row holds 999 entries, and the other 999 rows hold 1 entry each. What is the ELL storage against CSR?
> $z = 999 + 999 = 1998$ and $w = 999$. ELL stores $8 \times 1000 \times 999 = 7{,}992{,}000$ bytes. CSR stores $4 \times 1001 + 8 \times 1998 = 19{,}988$ bytes. ELL is 400 times larger, because one row sets the width of all 1,000.

> [!QUESTION] The rows of a matrix hold 3, 0, 1, 5, 0, and 1 entries. What is `row_ptr`, and what does a loop to `row_ptr[i+1] - 1` return for row 5?
> `row_ptr` is `[0, 3, 3, 4, 9, 9, 10]`. Row 5 owns the range 9 to 10, one entry. A loop to `row_ptr[6] - 1 = 9` covers the range 9 to 9, which is empty, so the row returns 0 and the entry is dropped.

> [!QUESTION] Why does thread-per-row finish later than warp-per-row on a power-law matrix, even if both read the same bytes?
> A thread-per-row warp runs as long as the longest row among its 32. The lab's matrix has a 1,153-entry row, so one warp runs 1,153 iterations against a mean of 104.7, a max over mean of 11. A warp-per-row warp splits a long row over 32 lanes, so the same row costs 37 iterations, and the ratio is 1.96. The bytes read are the same, and the time is set by the last warp.

> [!QUESTION] Where does the merge-path cut for 4 workers on `row_ptr = [0, 3, 3, 4, 9, 9, 10]` fall at diagonal 8, and why is the search's comparison `row_ptr[mid+1] <= d - 1 - mid`?
> It falls at 3 rows finished and 5 nonzeros consumed, $(3, 5)$, which adds to 8. The comparison asks whether row mid ends at or before the nonzero that the path would pair against it on that diagonal, which is nonzero $d - 1 - \text{mid}$. If it does, the row is behind the diagonal and the search moves right.

> [!QUESTION] Which splits give workers 3, 6, 1, and 0 nonzeros on that six-row matrix, and what is their max over mean?
> Two rows per worker: rows 0 and 1 hold 3, rows 2 and 3 hold 6, rows 4 and 5 hold 1, and the fourth worker has none. The max is 6 and the mean is 2.5, so the ratio is 2.4. Merge path gives every worker 4 steps.

> [!QUESTION] Why does a bottom-up BFS round not need atomics, when a top-down round does?
> In a top-down round, two frontier vertices can reach the same unvisited neighbor, so the claim must be atomic. In a bottom-up round each unvisited vertex scans its own neighbors and writes only its own level, so there is one writer per vertex.

> [!QUESTION] On the lab graph at level 3, top-down examines 3,479 edges and bottom-up examines 81. Why is bottom-up so cheap there?
> Only 63 vertices are unvisited, and a vertex stops scanning at its first frontier neighbor. With 766 of the 1,520 vertices in the frontier, most unvisited vertices find one on the first or second edge.

> [!QUESTION] Eight workers run 64 tasks, every eighth costing 100 and the rest costing 1. What does the bound promise, and why can't a static assignment promise it?
> The sum is $8 \times 100 + 56 \times 1 = 856$, so the bound is $856/8 + 100 \times 7/8 = 194.5$. The queue can't exceed it. A static assignment depends on which tasks land on which worker, and the stride-8 pattern puts every heavy task on worker 0, for a finish time of 800.

## Lab

> [!TRY] Build the formats, the merge-path split, and BFS
> Open `32-spmv-bfs`. `coo_to_csr` is the one function the first check needs. Then implement `spmv_csr`, the byte counts and `hyb_width`, `merge_path_coordinate` and `merge_path_partition`, `spmv_merge_path`, `warp_work`, `bfs_levels_frontier`, `bottom_up_step`, `should_switch`, `bfs_direction_optimizing`, and `simulate_work_queue` in plain Python. Each check compares your result with a reference: torch's coalesced CSR, `torch.sparse.mm`, a brute-force walk of the merge path, and a plain queue BFS. The last part is CUDA. Write the warp-per-row kernel and the merge kernel in the starter's source, and the thread-per-row kernel is given. On a machine without a GPU the CUDA checks print a skip message.
>
> You pass when every CPU check passes, and on a GPU when all three kernels match the float64 product of a 524,288-row power-law matrix within a float32 bound. The roofline predicts about 0.0375 ms at best for that matrix, and your lab measures the real figure.

The lab reports these metrics:

- `imbalance_thread_per_row`, `imbalance_warp_per_row`, and `imbalance_merge_path`: max over mean work per warp from your `warp_work`, on the 4,096-row matrix.
- `bfs_edges_top_down` and `bfs_edges_direction_optimizing`: the edges a top-down BFS and your direction-optimizing BFS examine on the lab graph.
- `tpr_gbs`, `wpr_gbs`, `merge_gbs`, and `torch_gbs`: the GPU matrix's 47,757,548 bytes divided by each kernel's time, with a cold L2.
- `tpr_over_wpr_ms`: thread-per-row's time over warp-per-row's. The check fails only if thread-per-row takes less than 0.8 times warp-per-row's time, which would suggest the warp-per-row kernel is wrong. The expected value is above 1.

A machine without a GPU records `skipped (no GPU)` for the last five. The first compile takes 30 to 60 seconds.

## Further reading

- [Efficient Sparse Matrix-Vector Multiplication on CUDA](https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf), Nathan Bell and Michael Garland, NVIDIA technical report NVR-2008-004. The COO, CSR, ELL, and hybrid formats, the scalar and vector CSR kernels, and the hybrid's one-third rule for the ELL width. The measurements are from a GeForce GTX 280, not an A100.
- Merge-based parallel sparse matrix-vector multiplication, Duane Merrill and Michael Garland, SC16. The merge-path SpMV this chapter's partition follows. The chapter describes the method from the paper's idea and its own derivation, and the paper's page wasn't fetched for the chapter.
- [Searching for a Parent Instead of Fighting Over Children](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.html), Scott Beamer, Krste Asanović, and David Patterson, UC Berkeley technical report EECS-2011-117. The bottom-up step, the queue-to-bitmap conversion, and the cutoff rule for switching direction.
- [GPU MODE, lecture 11: Sparsity](https://github.com/gpu-mode/lectures), Jesse Cai. The lecture index titles this session Sparsity. Chapter 20 draws on it for structured sparsity, and it isn't an SpMV or graph lecture.
- Programming Massively Parallel Processors, 4th edition, chapters 14 and 15, on sparse matrix computation and graph traversal. Not re-fetched for this chapter.
- [Chapter 20](/c/20-moe-and-sparse-kernels), for 2:4 structured sparsity and the grouped GEMM's scheduler, and [chapter 7](/c/07-scan-and-histogram), for the atomics and the scan this chapter builds on.
