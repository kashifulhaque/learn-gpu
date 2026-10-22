"""Lab 09: Numerics on the GPU, worked solution.

Four parts:

1. Format round trips, in PyTorch: what each format can and can't hold.
2. Accumulation order, in PyTorch: the bfloat16 stall and the pairwise cure.
3. Summation kernels, in CUDA: Kahan, a fixed-order tree, and atomicAdd.
4. Compiler flags: what --use_fast_math does to expf and to subnormals.
"""

import torch
import torch.nn.functional as F
from lab_common import load_cuda

# (exponent bits, mantissa bits, finite_top_exponent). E4M3 ("fn") reuses its
# top exponent for finite values and has no infinity; the others follow IEEE.
FORMATS = {
    "fp32": (8, 23, False),
    "tf32": (8, 10, False),
    "bf16": (8, 7, False),
    "fp16": (5, 10, False),
    "e4m3": (4, 3, True),
    "e5m2": (5, 2, False),
}


# ---------------------------------------------------------------------------
# Part 1: format round trips
# ---------------------------------------------------------------------------


def format_stats(exp_bits: int, man_bits: int, finite_top_exponent: bool = False) -> dict:
    """Return the limits of a binary floating-point format.

    Keys: "max", "min_normal", "min_subnormal", "epsilon", all Python floats.
    """
    bias = 2 ** (exp_bits - 1) - 1
    if finite_top_exponent:
        # The all-ones exponent holds finite numbers; only the all-ones
        # mantissa there is NaN, so the top mantissa is one step lower.
        top_exponent = 2**exp_bits - 1 - bias
        largest_significand = 2.0 - 2.0 ** (1 - man_bits)
    else:
        top_exponent = 2**exp_bits - 2 - bias
        largest_significand = 2.0 - 2.0 ** (-man_bits)
    return {
        "max": largest_significand * 2.0**top_exponent,
        "min_normal": 2.0 ** (1 - bias),
        "min_subnormal": 2.0 ** (1 - bias - man_bits),
        "epsilon": 2.0 ** (-man_bits),
    }


def round_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values to TF32 (10 mantissa bits), ties away from zero.

    This is what `cvt.rna.tf32.f32` does. The result stays a float32 tensor
    whose low 13 mantissa bits are zero.
    """
    bits = x.contiguous().view(torch.int32)
    # Half of 2**13 is 0x1000. Adding it and clearing the low 13 bits rounds
    # the magnitude to nearest, and a tie carries up, so ties go away from zero.
    rounded = (bits + 0x1000) & ~0x1FFF
    return rounded.view(torch.float32)


def roundtrip(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize float32 `x` to `fmt` and return it as float32.

    `fmt` is a key of FORMATS. For "e4m3" and "e5m2", saturate: values beyond
    the largest finite value become that value, not NaN or infinity.
    """
    x = x.float()
    if fmt == "fp32":
        return x.clone()
    if fmt == "tf32":
        return round_tf32(x)
    if fmt == "bf16":
        return x.to(torch.bfloat16).float()
    if fmt == "fp16":
        return x.to(torch.float16).float()
    if fmt in ("e4m3", "e5m2"):
        dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
        limit = torch.finfo(dtype).max
        return x.clamp(-limit, limit).to(dtype).float()
    raise ValueError(f"unknown format {fmt!r}")


def measured_roundoff(fmt: str, n: int = 1 << 16) -> float:
    """Largest relative round-trip error over n evenly spaced values in [1, 2).

    Every value in [1, 2) is a normal number in every format here, so the
    result is the format's unit roundoff, 2**-(m + 1), seen from below.
    """
    x = 1.0 + torch.arange(n, dtype=torch.float64) / n
    x32 = x.float()
    back = roundtrip(x32, fmt).double()
    return ((back - x32.double()).abs() / x32.double()).max().item()


# ---------------------------------------------------------------------------
# Part 2: accumulation order
# ---------------------------------------------------------------------------


def sum_bf16_sequential(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of `x` (rows, n) left to right in a bfloat16 accumulator.

    Round the inputs to bfloat16, then add one column at a time so every
    addition is rounded to bfloat16. Return a float32 tensor of shape (rows,).
    """
    v = x.to(torch.bfloat16)
    acc = torch.zeros(v.shape[0], dtype=torch.bfloat16, device=v.device)
    for i in range(v.shape[1]):
        acc = acc + v[:, i]
    return acc.float()


def sum_bf16_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of `x` (rows, n) with a binary tree of bfloat16 additions.

    Add neighbouring pairs, then pairs of those, until one value is left.
    Pad an odd-length row with a zero. Every addition rounds to bfloat16.
    Return a float32 tensor of shape (rows,).
    """
    v = x.to(torch.bfloat16)
    while v.shape[1] > 1:
        if v.shape[1] % 2:
            v = F.pad(v, (0, 1))
        v = v[:, 0::2] + v[:, 1::2]
    return v[:, 0].float()


# ---------------------------------------------------------------------------
# Part 3 and 4: CUDA kernels
# ---------------------------------------------------------------------------

CPP_DECLARATIONS = """
torch::Tensor seq_sum(torch::Tensor x);
torch::Tensor kahan_sum(torch::Tensor x);
torch::Tensor tree_sum(torch::Tensor x);
torch::Tensor atomic_sum(torch::Tensor x);
torch::Tensor exp_f32(torch::Tensor x);
torch::Tensor scale_f32(torch::Tensor x, double s);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int ITEMS = 4;  // elements each thread adds before the tree starts

// One thread, left to right: the textbook order, and the slowest and least
// accurate of the four. Given.
__global__ void seq_sum_kernel(const float* x, float* out, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += x[i];
    }
    *out = s;
}

// One thread, left to right, with a running compensation c that holds the
// low-order bits each addition lost.
__global__ void kahan_sum_kernel(const float* x, float* out, int n) {
    float s = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < n; ++i) {
        float y = x[i] - c;    // the addend, corrected by the last error
        float t = s + y;       // big + small: the low bits of y are lost here
        c = (t - s) - y;       // what was added, minus what we meant to add
        s = t;
    }
    *out = s;
}

// Sum THREADS values held in shared memory with a binary tree. The pairing is
// fixed by the thread index, so the result never depends on timing. Every
// thread must call this; thread 0 (and all the others) read the total.
__device__ float block_tree(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        }
        __syncthreads();
    }
    return smem[0];
}

// Each block reduces THREADS * ITEMS consecutive elements to one partial.
__global__ void tree_pass_kernel(const float* in, float* out, int n) {
    __shared__ float smem[THREADS];
    long long base = (long long)blockIdx.x * THREADS * ITEMS;
    float s = 0.0f;
    for (int k = 0; k < ITEMS; ++k) {
        long long i = base + (long long)k * THREADS + threadIdx.x;
        if (i < n) {
            s += in[i];
        }
    }
    float total = block_tree(s, smem);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = total;
    }
}

// One pass: each block reduces its slice with the tree, then adds its total
// to one float in global memory with atomicAdd. The adds land in whatever
// order the blocks finish.
__global__ void atomic_sum_kernel(const float* x, float* out, int n) {
    __shared__ float smem[THREADS];
    int i = blockIdx.x * THREADS + threadIdx.x;
    float v = (i < n) ? x[i] : 0.0f;
    float total = block_tree(v, smem);
    if (threadIdx.x == 0) {
        atomicAdd(out, total);
    }
}

__global__ void exp_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = expf(x[i]);
    }
}

// The scale arrives as a runtime argument so the compiler can't fold x * 1.0f.
__global__ void scale_kernel(const float* x, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * s;
    }
}

torch::Tensor seq_sum(torch::Tensor x) {
    x = x.contiguous();
    auto out = torch::zeros({1}, x.options());
    seq_sum_kernel<<<1, 1>>>(x.data_ptr<float>(), out.data_ptr<float>(), (int)x.numel());
    return out;
}

torch::Tensor kahan_sum(torch::Tensor x) {
    x = x.contiguous();
    auto out = torch::zeros({1}, x.options());
    kahan_sum_kernel<<<1, 1>>>(x.data_ptr<float>(), out.data_ptr<float>(), (int)x.numel());
    return out;
}

// Repeat the pass until one value is left. Every launch has a fixed shape, so
// the whole reduction is deterministic.
torch::Tensor tree_sum(torch::Tensor x) {
    auto cur = x.contiguous();
    int n = (int)cur.numel();
    if (n == 0) {
        return torch::zeros({1}, x.options());
    }
    while (true) {
        int per_block = THREADS * ITEMS;
        int blocks = (n + per_block - 1) / per_block;
        auto out = torch::empty({blocks}, x.options());
        tree_pass_kernel<<<blocks, THREADS>>>(cur.data_ptr<float>(), out.data_ptr<float>(), n);
        cur = out;
        n = blocks;
        if (n == 1) {
            break;
        }
    }
    return cur;
}

torch::Tensor atomic_sum(torch::Tensor x) {
    x = x.contiguous();
    auto out = torch::zeros({1}, x.options());
    int n = (int)x.numel();
    if (n == 0) {
        return out;
    }
    int blocks = (n + THREADS - 1) / THREADS;
    atomic_sum_kernel<<<blocks, THREADS>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

torch::Tensor exp_f32(torch::Tensor x) {
    x = x.contiguous();
    auto out = torch::empty_like(x);
    int n = (int)x.numel();
    int blocks = (n + 255) / 256;
    exp_kernel<<<blocks, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

torch::Tensor scale_f32(torch::Tensor x, double s) {
    x = x.contiguous();
    auto out = torch::empty_like(x);
    int n = (int)x.numel();
    int blocks = (n + 255) / 256;
    scale_kernel<<<blocks, 256>>>(x.data_ptr<float>(), (float)s, out.data_ptr<float>(), n);
    return out;
}
"""

_FUNCTIONS = ["seq_sum", "kahan_sum", "tree_sum", "atomic_sum", "exp_f32", "scale_f32"]


def _module(fast_math: bool = False):
    return load_cuda("numerics_lab", CUDA_SRC, CPP_DECLARATIONS, _FUNCTIONS, fast_math=fast_math)


def seq_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """One thread, left to right, float32. Returns a one-element CUDA tensor."""
    return _module().seq_sum(x)


def kahan_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """One thread, left to right, with Kahan compensation."""
    return _module().kahan_sum(x)


def tree_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """A fixed-order tree reduction. Deterministic."""
    return _module().tree_sum(x)


def atomic_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """Per-block tree totals combined with atomicAdd. Order varies run to run."""
    return _module().atomic_sum(x)


def _ulp_error(y: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|y - exact| in units of the float32 spacing at `exact`."""
    exact32 = exact.float()
    _, exponent = torch.frexp(exact32)  # exact32 = m * 2**e with m in [0.5, 1)
    ulp = torch.ldexp(torch.ones_like(exact32), exponent - 24).double()
    return (y.double() - exact).abs() / ulp


def expf_max_ulp_error(fast_math: bool) -> float:
    """Largest error of CUDA's expf, in float32 ulps, over [-20, 20].

    Compile the module with `fast_math` on or off, evaluate 2**20 evenly
    spaced inputs on the GPU, and compare with float64 `torch.exp`.
    """
    x = torch.linspace(-20.0, 20.0, 1 << 20, device="cuda", dtype=torch.float32)
    y = _module(fast_math).exp_f32(x)
    exact = torch.exp(x.double())
    return _ulp_error(y, exact).max().item()


def subnormal_survives(fast_math: bool) -> bool:
    """Whether a subnormal float32 times 1.0 on the GPU stays nonzero."""
    x = torch.tensor([1e-40], device="cuda", dtype=torch.float32)
    y = _module(fast_math).scale_f32(x, 1.0)
    return bool(y.item() != 0.0)
