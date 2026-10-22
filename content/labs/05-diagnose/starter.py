"""Lab 05: Diagnose three slow kernels.

`fixture_slow.py`, next to this file, holds three slow pieces of code. Do not
edit it. Open it, run each one under a profiler, say what is wrong from the
evidence, and write a fixed version here. The tests import the functions below.

Work in this order:

1. `hot_kernel`: find the GPU kernel that dominates a call, with torch.profiler.
2. `fast_scale`: slow kernel 1 reads memory the wrong way round.
3. `clip_to_max_abs`: slow function 2 stalls the GPU with host round trips.
4. `fast_branchy`: slow kernel 3 leaves half of every warp idle.
5. `parse_ptxas_info`, `count_sass`, `max_load_bits`: read what the compiler printed.

The CUDA kernels below start as copies of the slow ones, so the lab passes its
correctness checks and fails only its speed checks until you fix them.
"""

import torch
from lab_common import load_cuda
from torch.profiler import ProfilerActivity, profile  # noqa: F401

# ---------------------------------------------------------------------------
# Part 1: find the hot kernel
# ---------------------------------------------------------------------------


def hot_kernel(fn) -> str:
    """Run `fn()` once under torch.profiler and return the name of the GPU
    kernel with the most device time of its own.

    Call `fn()` once first, outside the profiler, so compilation and lazy
    initialization stay out of the trace. Then profile one call with
    `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`, synchronize
    inside the `with` block, and look at `prof.key_averages()`. Each row has a
    `key` (the name) and `self_device_time_total` (microseconds of device time
    that row spent itself, which is zero for CPU-only rows). Return the key of
    the row with the largest self device time.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: the three fixes
# ---------------------------------------------------------------------------

CPP = """
torch::Tensor fast_scale(torch::Tensor x);
torch::Tensor fast_branchy(torch::Tensor x);
"""

CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Kernel 1. TODO: this is the slow mapping. Neighboring threads (tid, tid + 1)
// read addresses `cols` floats apart. Change the mapping so that a warp's 32
// threads read 32 consecutive floats, and write 32 consecutive floats.
__global__ void fast_scale_kernel(const float* x, float* out, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (tid >= n) return;
    int row = tid % rows;
    int col = tid / rows;
    int i = row * cols + col;
    out[i] = x[i] * 2.0f + 1.0f;
}

torch::Tensor fast_scale(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "x must be a contiguous 2-D float32 CUDA tensor");
    auto out = torch::empty_like(x);
    int rows = (int)x.size(0), cols = (int)x.size(1);
    int n = rows * cols;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fast_scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

__device__ __forceinline__ float path_a(float v) {
    for (int k = 0; k < 256; ++k) v = fmaf(v, 0.999f, 0.001f);
    return v;
}
__device__ __forceinline__ float path_b(float v) {
    for (int k = 0; k < 256; ++k) v = v * v * 0.5f + 0.25f;  // a different body, so nvcc cannot merge the two paths
    return v;
}

// Kernel 3. TODO: even and odd lanes of the same warp take different paths, so
// the warp runs path A with half its lanes idle, then path B with the other half
// idle. Restructure so every lane is active in every pass. The results must stay
// identical: out[i] is path_a(x[i]) for even i and path_b(x[i]) for odd i, and
// the array length n may be odd or 1.
__global__ void fast_branchy_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (i & 1) ? path_b(x[i]) : path_a(x[i]);
}

torch::Tensor fast_branchy(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 1 && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "x must be a contiguous 1-D float32 CUDA tensor");
    auto out = torch::empty_like(x);
    int n = (int)x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;  // TODO: adjust if a thread now owns more than one element
    fast_branchy_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


def _module():
    return load_cuda("diagnose_fast", CUDA, CPP, ["fast_scale", "fast_branchy"])


def fast_scale(x: torch.Tensor) -> torch.Tensor:
    """2 * x + 1 for a contiguous 2-D float32 CUDA tensor, with coalesced access."""
    return _module().fast_scale(x)


def fast_branchy(x: torch.Tensor) -> torch.Tensor:
    """Same result as `fixture_slow.slow_branchy`, without lane-level divergence."""
    return _module().fast_branchy(x)


def clip_to_max_abs(tensors: list, thresh: float) -> list:
    """Scale each tensor so its largest absolute value is at most `thresh`.

    A tensor whose largest absolute value is already at most `thresh` comes back
    unchanged. Return a new list; don't modify the inputs. The tensors are
    float32 CUDA tensors, possibly of different sizes, possibly all zeros.

    TODO: this copy of the slow version reads every maximum back to the host with
    `.item()`. Remove every synchronization. The test runs your function under
    `torch.cuda.set_sync_debug_mode("error")`, which raises on a sync.
    """
    out = []
    for t in tensors:
        m = t.abs().max().item()
        out.append(t * (thresh / m) if m > thresh else t.clone())
    return out


# ---------------------------------------------------------------------------
# Part 3: read compiler output
# ---------------------------------------------------------------------------


def parse_ptxas_info(text: str) -> dict:
    """Parse the first kernel in the text `nvcc -Xptxas -v` prints.

    Return {"registers": int, "smem_bytes": int, "spill_stores": int,
    "spill_loads": int}. The relevant lines look like
        64 bytes stack frame, 48 bytes spill stores, 52 bytes spill loads
        Used 255 registers, 8192 bytes smem, 380 bytes cmem[0]
    A field the text doesn't mention is 0 (for example, no "bytes smem").
    """
    raise NotImplementedError


def count_sass(sass: str) -> dict:
    """Count the instructions in `cuobjdump -sass` text by opcode base.

    An instruction line looks like
        /*0070*/                   LDG.E.128 R4, [R2.64] ;
        /*0050*/               @P0 EXIT ;
    after an address in /* */ and an optional predicate (@P0, @!P1). The opcode
    is the first word; its base is the part before the first dot, so LDG.E.128
    counts under "LDG". Lines that hold only an encoding, like `/* 0x000fd0... */`,
    are not instructions. Return {"LDG": 2, "FFMA": 4, ...}.
    """
    raise NotImplementedError


def max_load_bits(sass: str) -> int:
    """The widest per-thread global load in `sass`, in bits.

    LDG.E.128 is 128, LDG.E.64 is 64, a plain LDG.E is 32, and no LDG at all is 0.
    """
    raise NotImplementedError
