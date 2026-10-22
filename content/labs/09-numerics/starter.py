"""Lab 09: Numerics on the GPU.

Four parts. Each one pairs a number you can compute by hand with a measurement
the tests make:

1. Format round trips, in PyTorch: what each format can and can't hold.
2. Accumulation order, in PyTorch: the bfloat16 stall and the pairwise cure.
3. Summation kernels, in CUDA: Kahan, a fixed-order tree, and atomicAdd.
4. Compiler flags: what --use_fast_math does to expf and to subnormals.

Parts 1 and 2 are pure PyTorch and run on any device. Parts 3 and 4 compile
CUDA with nvcc, which takes 30 to 60 seconds the first time.
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

    Use the value rule from the chapter. With bias = 2**(exp_bits - 1) - 1, a
    normal number is 2**(E - bias) * (1 + M / 2**man_bits), and a subnormal
    (E = 0) is 2**(1 - bias) * (M / 2**man_bits). Epsilon is the gap between
    1.0 and the next number, 2**-man_bits.

    For an IEEE-style format the top exponent field (all ones) is reserved for
    infinity and NaN, so the largest finite number uses the field below it.
    When `finite_top_exponent` is True, as for E4M3, the all-ones exponent holds
    finite numbers too, and only the all-ones mantissa there is NaN. Check your
    result against the tests: E4M3's maximum is 448, E5M2's is 57344.
    """
    raise NotImplementedError


def round_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round float32 values to TF32 (10 mantissa bits), ties away from zero.

    This is what `cvt.rna.tf32.f32` does. The result stays a float32 tensor
    whose low 13 mantissa bits are zero.

    Do it with integer arithmetic on the bit pattern: `x.view(torch.int32)`,
    add half of the dropped range (0x1000), then clear the low 13 bits
    (`& ~0x1FFF`) and view the result as float32 again. A tie carries into the
    kept bits, which sends it away from zero.
    """
    raise NotImplementedError


def roundtrip(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Quantize float32 `x` to `fmt` and return it as float32.

    `fmt` is a key of FORMATS. Use a real cast for bf16, fp16, and the two fp8
    formats (`torch.bfloat16`, `torch.float16`, `torch.float8_e4m3fn`,
    `torch.float8_e5m2`), `round_tf32` for tf32, and a copy for fp32.

    For "e4m3" and "e5m2", saturate: a value beyond the largest finite value
    (`torch.finfo(dtype).max`) becomes that value, not NaN or infinity. A cast
    to float8_e4m3fn turns an overflow into NaN, so clamp first.
    """
    raise NotImplementedError


def measured_roundoff(fmt: str, n: int = 1 << 16) -> float:
    """Largest relative round-trip error over n evenly spaced values in [1, 2).

    Build x = 1 + k / n for k = 0 .. n - 1 as float32, round-trip it through
    `fmt`, and return max(|roundtrip(x) - x| / x), computed in float64. Every
    value in [1, 2) is a normal number in every format here, so the result
    should land just under the unit roundoff 2**-(m + 1), where m is the number
    of mantissa bits.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2: accumulation order
# ---------------------------------------------------------------------------


def sum_bf16_sequential(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of `x` (rows, n) left to right in a bfloat16 accumulator.

    Round the inputs to bfloat16, then add one column at a time so every
    addition is rounded to bfloat16: `acc = acc + v[:, i]` with `acc` a
    bfloat16 tensor. Return a float32 tensor of shape (rows,).

    Predict first: what does a row of 1024 ones sum to?
    """
    raise NotImplementedError


def sum_bf16_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of `x` (rows, n) with a binary tree of bfloat16 additions.

    Add neighbouring pairs (`v[:, 0::2] + v[:, 1::2]`), then pairs of those,
    until one value is left. Pad an odd-length row with a zero first
    (`F.pad(v, (0, 1))`). Every addition rounds to bfloat16. Return a float32
    tensor of shape (rows,). The input can have a single column.
    """
    raise NotImplementedError


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

// One thread, left to right: the textbook order, and the least accurate of
// the sums you compare. Given.
__global__ void seq_sum_kernel(const float* x, float* out, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += x[i];
    }
    *out = s;
}

// TODO 1: Kahan summation. One thread, left to right, with a running
// compensation `c` that holds the low-order bits each addition lost:
//   y = x[i] - c;   t = s + y;   c = (t - s) - y;   s = t;
// Write the total to *out. (nvcc does not reassociate float arithmetic, so the
// compensation survives optimization.)
__global__ void kahan_sum_kernel(const float* x, float* out, int n) {
    *out = 0.0f;  // replace this
}

// TODO 2: sum THREADS values held in shared memory with a binary tree, and
// return the total to every thread. Store v in smem[threadIdx.x], then halve
// `offset` from THREADS / 2 down to 1: threads below `offset` add
// smem[threadIdx.x + offset] into their own slot. Put a __syncthreads() after
// the store and after every step, and make sure every thread calls this
// function (a __syncthreads() that only some threads reach hangs the block).
__device__ float block_tree(float v, float* smem) {
    return 0.0f;  // replace this
}

// TODO 3: each block reduces THREADS * ITEMS consecutive elements to one
// partial sum. Thread t adds the ITEMS elements at base + k * THREADS + t for
// k = 0 .. ITEMS - 1 (skip indices >= n), calls block_tree, and thread 0 writes
// the block total to out[blockIdx.x]. Use no atomics: the order of every add
// must depend only on the indices.
__global__ void tree_pass_kernel(const float* in, float* out, int n) {
    if (threadIdx.x == 0) {
        out[blockIdx.x] = 0.0f;  // replace this
    }
}

// TODO 4: one pass with atomics. Each thread loads x[blockIdx.x * THREADS +
// threadIdx.x] (0 past the end), the block reduces with block_tree, and thread
// 0 adds the block total to *out with atomicAdd. The adds land in whatever
// order the blocks finish.
__global__ void atomic_sum_kernel(const float* x, float* out, int n) {
    // replace this
}

// Part 4's kernels are given.
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

// ---- Host wrappers. Given; read them to see how the kernels are launched. ----

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
// the whole reduction is deterministic if the kernel is.
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
    """Compile CUDA_SRC once per flag setting."""
    return load_cuda("numerics_lab", CUDA_SRC, CPP_DECLARATIONS, _FUNCTIONS, fast_math=fast_math)


def seq_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """One thread, left to right, float32. Returns a one-element CUDA tensor. Given."""
    return _module().seq_sum(x)


def kahan_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """One thread, left to right, with Kahan compensation. Given once TODO 1 is done."""
    return _module().kahan_sum(x)


def tree_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """A fixed-order tree reduction. Deterministic. Given once TODOs 2 and 3 are done."""
    return _module().tree_sum(x)


def atomic_sum_cuda(x: torch.Tensor) -> torch.Tensor:
    """Per-block tree totals combined with atomicAdd. Given once TODO 4 is done."""
    return _module().atomic_sum(x)


def _ulp_error(y: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|y - exact| in units of the float32 spacing at `exact`. Given."""
    exact32 = exact.float()
    _, exponent = torch.frexp(exact32)  # exact32 = m * 2**e with m in [0.5, 1)
    ulp = torch.ldexp(torch.ones_like(exact32), exponent - 24).double()
    return (y.double() - exact).abs() / ulp


def expf_max_ulp_error(fast_math: bool) -> float:
    """Largest error of CUDA's expf, in float32 ulps, over [-20, 20].

    TODO 5: build x = torch.linspace(-20, 20, 1 << 20) on the GPU in float32,
    run `_module(fast_math).exp_f32(x)` (the flag recompiles the same source with
    or without --use_fast_math), and compare with float64 `torch.exp(x.double())`
    using `_ulp_error(y, exact).max().item()`.
    """
    raise NotImplementedError


def subnormal_survives(fast_math: bool) -> bool:
    """Whether a subnormal float32 times 1.0 on the GPU stays nonzero.

    TODO 6: put 1e-40 (smaller than the smallest normal float32, 1.18e-38) in a
    one-element float32 CUDA tensor, call `_module(fast_math).scale_f32(x, 1.0)`,
    and return whether the single output value is nonzero.
    """
    raise NotImplementedError
