"""Lab 05: Diagnose three slow kernels. Worked solution."""

import re

import torch
from lab_common import load_cuda
from torch.profiler import ProfilerActivity, profile

# ---------------------------------------------------------------------------
# Part 1: find the hot kernel with torch.profiler
# ---------------------------------------------------------------------------


def _self_device_us(event) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def hot_kernel(fn) -> str:
    """Run `fn` once under torch.profiler and return the name of the GPU kernel
    with the most device time of its own."""
    fn()
    torch.cuda.synchronize()  # warm up, so compilation and lazy init stay out of the trace
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = list(prof.key_averages())
    best = max(events, key=_self_device_us)
    if _self_device_us(best) <= 0:
        raise RuntimeError("the profiler recorded no GPU time")
    return best.key


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

// Fix 1: consecutive threads read consecutive floats, so a warp's 32 loads are
// 128 contiguous bytes and a warp's 32 stores are the same.
__global__ void fast_scale_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * 2.0f + 1.0f;
}

torch::Tensor fast_scale(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "x must be a contiguous 2-D float32 CUDA tensor");
    auto out = torch::empty_like(x);
    int n = (int)x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fast_scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

__device__ __forceinline__ float path_a(float v) {
    for (int k = 0; k < 256; ++k) v = fmaf(v, 0.999f, 0.001f);
    return v;
}
__device__ __forceinline__ float path_b(float v) {
    for (int k = 0; k < 256; ++k) v = fmaf(v, 0.998f, -0.002f);
    return v;
}

// Fix 3: each thread owns an (even, odd) pair. Every lane runs path A, then
// every lane runs path B, so no pass has idle lanes.
__global__ void fast_branchy_kernel(const float* x, float* out, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i0 = 2 * t;
    if (i0 < n) out[i0] = path_a(x[i0]);
    int i1 = i0 + 1;
    if (i1 < n) out[i1] = path_b(x[i1]);
}

torch::Tensor fast_branchy(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 1 && x.is_contiguous() && x.scalar_type() == torch::kFloat32,
                "x must be a contiguous 1-D float32 CUDA tensor");
    auto out = torch::empty_like(x);
    int n = (int)x.numel();
    int pairs = (n + 1) / 2;
    int threads = 256;
    int blocks = (pairs + threads - 1) / threads;
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
    """Same result as slow_branchy, without lane-level divergence."""
    return _module().fast_branchy(x)


def clip_to_max_abs(tensors: list, thresh: float) -> list:
    """Scale each tensor so its largest absolute value is at most `thresh`,
    without a single host-device synchronization."""
    if len(tensors) > 1 and len({t.numel() for t in tensors}) == 1 and tensors[0].numel() > 0:
        flat = torch.stack([t.reshape(-1) for t in tensors])
        m = flat.abs().amax(dim=1, keepdim=True)
        scale = (thresh / m).clamp(max=1.0)  # a zero maximum gives inf, which clamps to 1
        out = flat * scale
        return [out[i].reshape(t.shape) for i, t in enumerate(tensors)]
    out = []
    for t in tensors:
        m = t.abs().amax()
        out.append(t * (thresh / m).clamp(max=1.0))
    return out


# ---------------------------------------------------------------------------
# Part 3: read compiler output
# ---------------------------------------------------------------------------


def parse_ptxas_info(text: str) -> dict:
    """Parse the first kernel in `nvcc -Xptxas -v` output.

    Returns {"registers": int, "smem_bytes": int, "spill_stores": int,
    "spill_loads": int}. A field the text doesn't mention is 0.
    """

    def first(pattern: str) -> int:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0

    return {
        "registers": first(r"Used (\d+) registers"),
        "smem_bytes": first(r"(\d+) bytes smem"),
        "spill_stores": first(r"(\d+) bytes spill stores"),
        "spill_loads": first(r"(\d+) bytes spill loads"),
    }


_SASS_LINE = re.compile(
    r"^\s*/\*[0-9a-fA-F]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Za-z0-9_]+)*)"
)


def _sass_instructions(sass: str):
    for line in sass.splitlines():
        m = _SASS_LINE.match(line)
        if m:
            yield m.group(1), [p for p in m.group(2).split(".") if p]


def count_sass(sass: str) -> dict:
    """Count instructions in `cuobjdump -sass` text by opcode base, for example
    "LDG.E.128" counts under "LDG"."""
    counts: dict = {}
    for base, _ in _sass_instructions(sass):
        counts[base] = counts.get(base, 0) + 1
    return counts


_WIDTHS = {"128": 128, "64": 64, "U8": 8, "S8": 8, "U16": 16, "S16": 16}


def max_load_bits(sass: str) -> int:
    """The widest per-thread global load in `sass`: 128 for LDG.E.128, 64 for
    LDG.E.64, 32 for a plain LDG.E, and 0 when there is no global load."""
    widest = 0
    for base, mods in _sass_instructions(sass):
        if base != "LDG":
            continue
        bits = 32
        for mod in mods:
            if mod in _WIDTHS:
                bits = _WIDTHS[mod]
        widest = max(widest, bits)
    return widest
