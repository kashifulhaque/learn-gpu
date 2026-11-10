"""Lab 27 solution: the fixed kernels and the sanitizer-report tools."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Part 1. The five kernels, fixed. Each string defines a kernel and the host
# function the tests call. The tests compile each with -lineinfo, so a
# sanitizer report carries a source line.
# ---------------------------------------------------------------------------

# Bug 1: the grid rounds up, so the last block has threads past the end.
# Fix: guard both the read and the write with i < n.
OOB_CUDA = r"""
#include <torch/extension.h>

__global__ void scale_shift_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 2.0f * x[i] + 1.0f;
    }
}

torch::Tensor scale_shift(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto y = torch::empty_like(x);
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_shift_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "scale_shift launch failed");
    return y;
}
"""

# Bug 2: a tree reduction with no barrier between steps. Step k reads values
# that other threads wrote in step k - 1. Fix: __syncthreads() after each step.
RACE_CUDA = r"""
#include <torch/extension.h>

__global__ void block_sum_kernel(const float* x, float* out) {
    __shared__ float s[256];
    int t = threadIdx.x;
    s[t] = x[blockIdx.x * 256 + t];
    __syncthreads();
    for (int d = 128; d > 0; d >>= 1) {
        if (t < d) {
            s[t] += s[t + d];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x] = s[0];
    }
}

torch::Tensor block_sums(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.numel() % 256 == 0, "length must be a multiple of 256");
    int blocks = static_cast<int>(x.numel() / 256);
    auto out = torch::empty({blocks}, x.options());
    block_sum_kernel<<<blocks, 256>>>(x.data_ptr<float>(), out.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "block_sums launch failed");
    return out;
}
"""

# Bug 3: the unrolled last warp reads values that other lanes wrote one line
# earlier, and relies on the lanes running in lockstep. Since Volta they need
# not. Fix: read into a register, __syncwarp(), write, __syncwarp().
WARP_CUDA = r"""
#include <torch/extension.h>

__global__ void warp_block_sum_kernel(const float* x, float* out) {
    __shared__ float s[128];
    int t = threadIdx.x;
    s[t] = x[blockIdx.x * 128 + t];
    __syncthreads();
    if (t < 64) {
        s[t] += s[t + 64];
    }
    __syncthreads();
    if (t < 32) {
        float v = s[t] + s[t + 32];
        for (int d = 16; d > 0; d >>= 1) {
            __syncwarp();
            s[t] = v;
            __syncwarp();
            v += s[t + d];
        }
        if (t == 0) {
            out[blockIdx.x] = v;
        }
    }
}

torch::Tensor warp_block_sums(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.numel() % 128 == 0, "length must be a multiple of 128");
    int blocks = static_cast<int>(x.numel() / 128);
    auto out = torch::empty({blocks}, x.options());
    warp_block_sum_kernel<<<blocks, 128>>>(x.data_ptr<float>(), out.data_ptr<float>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "warp_block_sums launch failed");
    return out;
}
"""

# Bug 4: threads past the end return before __syncthreads(), so the barrier is
# not reached by the whole block. Fix: let every thread reach the barrier, and
# guard only the loads and the store.
SYNC_CUDA = r"""
#include <torch/extension.h>

template <int B>
__global__ void tail_pair_sum_kernel(const float* x, float* out, int n) {
    __shared__ float tile[B];
    int t = threadIdx.x;
    int i = blockIdx.x * B + t;
    tile[t] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    if (i < n) {
        out[i] = tile[t] + ((t + 1 < B) ? tile[t + 1] : 0.0f);
    }
}

torch::Tensor tail_pair_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto out = torch::empty_like(x);
    int grid = (n + 127) / 128;
    tail_pair_sum_kernel<128><<<grid, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "tail_pair_sum launch failed");
    return out;
}
"""

# Bug 5: the tile has one halo cell on each side, and the kernel never writes
# them. Fix: thread 0 loads the left halo and the last thread loads the right.
INIT_CUDA = r"""
#include <torch/extension.h>

template <int B>
__global__ void stencil3_kernel(const float* x, float* out, int n) {
    __shared__ float tile[B + 2];
    int t = threadIdx.x;
    int i = blockIdx.x * B + t;
    tile[t + 1] = (i < n) ? x[i] : 0.0f;
    if (t == 0) {
        tile[0] = (i > 0) ? x[i - 1] : 0.0f;
    }
    if (t == B - 1) {
        tile[B + 1] = (i + 1 < n) ? x[i + 1] : 0.0f;
    }
    __syncthreads();
    if (i < n) {
        out[i] = tile[t] + tile[t + 1] + tile[t + 2];
    }
}

torch::Tensor stencil3(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous());
    int n = static_cast<int>(x.numel());
    auto out = torch::empty_like(x);
    int grid = (n + 127) / 128;
    stencil3_kernel<128><<<grid, 128>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stencil3 launch failed");
    return out;
}
"""


# ---------------------------------------------------------------------------
# Part 2. Index and hazard reasoning in plain Python.
# ---------------------------------------------------------------------------


def min_grid(n: int, block_dim: int) -> int:
    """Blocks needed to give every one of n elements a thread."""
    return -(-n // block_dim)


def oob_indices(n: int, block_dim: int, grid_dim: int, guarded: bool) -> list[int]:
    """Global thread indices at or past n that touch memory.

    An unguarded kernel lets every launched thread with index >= n read and
    write. A guarded kernel (`if (i < n)`) lets none of them.
    """
    if guarded:
        return []
    return list(range(n, grid_dim * block_dim))


def barrier_arrivals(n: int, block_dim: int, block_idx: int, early_return: bool) -> tuple[int, bool]:
    """How many threads of one block reach a __syncthreads(), and whether that is divergent.

    With `early_return`, a thread whose global index is >= n leaves before the
    barrier. A barrier is divergent when some, but not all, of the block's
    threads arrive: 0 arrivals is a block that left as a whole.
    """
    if not early_return:
        return block_dim, False
    arriving = min(block_dim, max(0, n - block_idx * block_dim))
    return arriving, 0 < arriving < block_dim


def shared_hazards(block_dim, barriers, lockstep=False, warp_size=32):
    """Read-after-write hazards in a shared-memory tree reduction.

    Phase 0: thread t writes s[t]. Phase k (k = 1 .. log2(block_dim)): with
    d = block_dim >> k, thread t < d reads s[t] and s[t + d], then writes s[t].
    `barriers[k]` is a barrier between phase k and phase k + 1; a bool applies
    to every boundary. A hazard is (address, writer, reader): the reader reads
    an address whose latest earlier write came from another thread, with no
    barrier between the two phases. With `lockstep=True`, threads in one warp
    are treated as ordered, which no GPU since Volta guarantees.
    Returns the hazards as a sorted list of (address, writer, reader).
    """
    steps = block_dim.bit_length() - 1
    if isinstance(barriers, bool):
        barriers = [barriers] * steps
    found = set()
    for phase in range(1, steps + 1):
        d = block_dim >> phase
        for reader in range(d):
            address = reader + d
            # The latest write to s[address] came from thread `address` in the
            # previous phase, because that thread was active there.
            writer = address
            if any(barriers[phase - 1 : phase]):
                continue
            if lockstep and writer // warp_size == reader // warp_size:
                continue
            found.add((address, writer, reader))
    return sorted(found)


# ---------------------------------------------------------------------------
# Part 3. Read a compute-sanitizer report.
# ---------------------------------------------------------------------------

_PREFIX = re.compile(r"^=+ ?")
_MEMCHECK = re.compile(r"Invalid __(\w+)__ (read|write) of size (\d+) bytes")
_INITCHECK = re.compile(r"Uninitialized __(\w+)__ memory (read|write) of size (\d+) bytes")
_SYNCCHECK = re.compile(r"Barrier error detected\.")
_RACE = re.compile(
    r"(Error|Warning): (\(Warp Level Programming\) )?Potential (RAW|WAR|WAW) hazard detected"
    r" at __shared__ (0x[0-9a-fA-F]+) in block \((\d+),(\d+),(\d+)\)"
)
_AT = re.compile(r"at (.+?)\+0x[0-9a-fA-F]+ in (\S+):(\d+)$")
_BY = re.compile(r"by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)")
_AFTER = re.compile(r"and is (\d+) bytes (?:after|before) the nearest allocation")
_ACCESS = re.compile(r"(Write|Read) Thread \((\d+),(\d+),(\d+)\) at 0x[0-9a-fA-F]+ in (\S+?):(\d+):(.+)$")
_SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+errors?")


def error_count(text: str):
    """The N in the `ERROR SUMMARY: N errors` line, or None when the report has none."""
    match = _SUMMARY.search(text)
    return int(match.group(1)) if match else None


def _kernel_name(signature: str) -> str:
    name = signature.strip()
    if name.startswith("void "):
        name = name[5:]
    name = name.split("(")[0]
    return name.split("<")[0].strip()


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _triple(a, b, c):
    return (int(a), int(b), int(c))


def _blocks(text: str):
    block = []
    for raw in text.splitlines():
        line = _PREFIX.sub("", raw).rstrip()
        if line.strip() == "COMPUTE-SANITIZER":
            continue
        if line.strip():
            block.append(line.strip())
        elif block:
            yield block
            block = []
    if block:
        yield block


def parse_report(text: str) -> list[dict]:
    """One dict per finding, in the order the tool printed them."""
    findings = []
    for block in _blocks(text):
        head = block[0]
        item = {
            "tool": None, "kind": None, "severity": "error", "kernel": None, "file": None,
            "line": None, "thread": None, "block": None, "size": None, "bytes_after": None,
            "other_line": None, "other_thread": None, "warp_level": False,
        }
        if m := _MEMCHECK.match(head):
            item.update(tool="memcheck", kind=f"invalid-{m.group(1)}-{m.group(2)}", size=int(m.group(3)))
        elif m := _INITCHECK.match(head):
            item.update(tool="initcheck", kind=f"uninitialized-{m.group(1)}-{m.group(2)}", size=int(m.group(3)))
        elif _SYNCCHECK.match(head):
            item.update(tool="synccheck", kind="divergent-barrier")
        elif m := _RACE.match(head):
            item.update(
                tool="racecheck",
                kind=f"{m.group(3).lower()}-hazard",
                severity=m.group(1).lower(),
                warp_level=bool(m.group(2)),
                block=_triple(*m.group(5, 6, 7)),
            )
        else:
            continue
        accesses = []
        for line in block[1:]:
            if m := _AT.match(line):
                item.update(kernel=_kernel_name(m.group(1)), file=_basename(m.group(2)), line=int(m.group(3)))
            elif m := _BY.match(line):
                item.update(thread=_triple(*m.group(1, 2, 3)), block=_triple(*m.group(4, 5, 6)))
            elif m := _AFTER.match(line):
                item["bytes_after"] = int(m.group(1))
            elif m := _ACCESS.match(line):
                accesses.append(m)
        if accesses:
            first = accesses[0]
            item.update(
                thread=_triple(*first.group(2, 3, 4)),
                file=_basename(first.group(5)),
                line=int(first.group(6)),
                kernel=_kernel_name(first.group(7)),
            )
            if len(accesses) > 1:
                second = accesses[1]
                item.update(other_thread=_triple(*second.group(2, 3, 4)), other_line=int(second.group(6)))
        findings.append(item)
    return findings


OUT_OF_BOUNDS_WRITE = "out-of-bounds write"
OUT_OF_BOUNDS_READ = "out-of-bounds read"
SHARED_RACE = "shared-memory race"
MISSING_SYNCWARP = "missing __syncwarp"
DIVERGENT_SYNCTHREADS = "divergent __syncthreads"
UNINITIALIZED_SHARED = "uninitialized shared read"
UNINITIALIZED_GLOBAL = "uninitialized global read"


def classify(finding: dict) -> str:
    """Name the bug behind one finding."""
    tool = finding["tool"]
    if tool == "memcheck":
        return OUT_OF_BOUNDS_WRITE if finding["kind"].endswith("write") else OUT_OF_BOUNDS_READ
    if tool == "racecheck":
        return MISSING_SYNCWARP if finding["warp_level"] else SHARED_RACE
    if tool == "synccheck":
        return DIVERGENT_SYNCTHREADS
    return UNINITIALIZED_SHARED if "shared" in finding["kind"] else UNINITIALIZED_GLOBAL


def is_clean(text: str) -> bool:
    """True when the run has a summary of zero errors and no finding at any severity."""
    return error_count(text) == 0 and not parse_report(text)
