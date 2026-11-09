"""Lab 26 solution.

The CPU functions are the tensor-map address, the PTX 128-byte swizzle,
and the wgmma descriptor. The kernels are CUDA for sm_90a. A machine
without an H100 skips them and does not compile for sm_80.
"""

from __future__ import annotations


def box_byte_offset(
    origin_col: int,
    origin_row: int,
    local_col: int,
    local_row: int,
    elem_bytes: int,
    stride_bytes: int,
) -> int:
    """Byte offset from the tensor base. Dimension 0 is the contiguous column."""
    col = origin_col + local_col
    row = origin_row + local_row
    return col * elem_bytes + row * stride_bytes


def box_load(
    matrix: list[list[int]],
    origin_col: int,
    origin_row: int,
    box_cols: int,
    box_rows: int,
) -> list[list[int]]:
    """Box values, with 0 written for every coordinate outside the matrix."""
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    out: list[list[int]] = []
    for local_row in range(box_rows):
        line: list[int] = []
        for local_col in range(box_cols):
            row = origin_row + local_row
            col = origin_col + local_col
            if row < 0 or col < 0 or row >= height or col >= width:
                line.append(0)
            else:
                line.append(matrix[row][col])
        out.append(line)
    return out


def swizzle_128b_byte(row: int, chunk: int) -> int:
    """Physical byte of one 16-byte chunk. Chunk XOR row, then times 16."""
    physical = chunk ^ (row & 7)
    return row * 128 + physical * 16


def choose_swizzle(elem_bytes: int, box_inner: int) -> str:
    """The swizzle span that equals the box's inner width, in bytes."""
    inner = elem_bytes * box_inner
    if inner % 16 != 0:
        return "illegal"
    if inner == 128:
        return "128B"
    if inner == 64:
        return "64B"
    if inner == 32:
        return "32B"
    return "none"


def expect_tx_bytes(tiles: list[tuple[int, int, int]]) -> int:
    """Sum of rows * cols * elem_bytes over the tiles."""
    total = 0
    for rows, cols, elem_bytes in tiles:
        total += rows * cols * elem_bytes
    return total


def wgmma_descriptor(start: int, leading: int, stride: int, layout_type: int) -> int:
    """64-bit descriptor. layout_type 1 is the 128-byte swizzle, not 3."""
    if layout_type not in (0, 1, 2, 3):
        raise ValueError(f"expected layout_type 0, 1, 2, or 3, actual {layout_type}")

    def enc(value: int) -> int:
        if value < 0 or value % 16 != 0:
            raise ValueError(f"expected a non-negative multiple of 16, actual {value}")
        return (value & 0x3FFFF) >> 4

    return enc(start) | (enc(leading) << 16) | (enc(stride) << 32) | (layout_type << 62)


def predicted_ms(m: int, n: int, k: int) -> float:
    """H100 SXM5 bf16 roofline, in milliseconds. The lab measures the kernel."""
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    peak = 989.4e12
    bandwidth = 3350e9
    if flops / nbytes >= peak / bandwidth:
        return flops / peak * 1e3
    return nbytes / bandwidth * 1e3


def _skip_reason() -> str | None:
    try:
        import torch
    except ImportError:
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            "torch is not installed, so there is no H100 to compile them on."
        )
    if not torch.cuda.is_available():
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            "No CUDA GPU is visible. They need an H100 (sm_90a) and are not sent to the A100 endpoint."
        )
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        return (
            "SKIP: the Hopper TMA copy and the wgmma kernel were not run. "
            f"The visible GPU is sm_{major}{minor}, not sm_90. "
            "They are not sent to the A100 endpoint."
        )
    return None


_CPP = """
torch::Tensor tma_copy_cuda(
    torch::Tensor src,
    int64_t origin_col,
    int64_t origin_row,
    int64_t box_cols,
    int64_t box_rows,
    int64_t swizzle_bytes);
torch::Tensor wgmma_64x8x16_cuda(torch::Tensor a, torch::Tensor b);
"""

# Real sm_90a CUDA. The TMA kernel issues one cp.async.bulk.tensor and waits
# on an mbarrier. The wgmma kernel stages both operands in the K-major
# no-swizzle core-matrix layout and issues one m64n8k16. A fused GEMM is
# those two issue sequences back to back. Chapter 12 owns the ring that
# repeats them. Callers pass arch="sm_90a".
_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace {

using EncodeFn = CUresult (*)(
    CUtensorMap*,
    CUtensorMapDataType,
    unsigned int,
    void*,
    const unsigned long long*,
    const unsigned long long*,
    const unsigned int*,
    const unsigned int*,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

EncodeFn encode_tiled() {
    static EncodeFn fn = nullptr;
    if (fn != nullptr) {
        return fn;
    }
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(lib != nullptr, "libcuda.so.1 did not open");
    fn = reinterpret_cast<EncodeFn>(dlsym(lib, "cuTensorMapEncodeTiled"));
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled is missing from libcuda");
    return fn;
}

__device__ uint32_t shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ void mbarrier_init(uint32_t addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ void mbarrier_arrive_expect(uint32_t addr, int bytes) {
    asm volatile(
        "{\n\t"
        ".reg .b64 state;\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n\t"
        "}\n" :: "r"(addr), "r"(bytes));
}

__device__ int mbarrier_try(uint32_t addr, int phase) {
    int ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n"
        : "=r"(ready)
        : "r"(addr), "r"(phase));
    return ready;
}

// The map describes the tensor. The coordinates are the box origin.
// Thread 0 is the producer: arrive.expect_tx, then one bulk tensor copy.
// Every thread waits. A fused GEMM would hand this barrier to a warpgroup
// that then issues wgmma. Chapter 12 owns that ring. This kernel stops at
// the copy and unswizzles the box.
__global__ void tma_box_kernel(
    const CUtensorMap* map,
    __nv_bfloat16* dst,
    int origin_col,
    int origin_row,
    int box_cols,
    int box_rows,
    int swizzle_bytes) {
    __shared__ __align__(8) uint64_t barrier;
    __shared__ __align__(1024) __nv_bfloat16 smem[512];
    uint32_t bar = shared_u32(&barrier);
    uint32_t smem_addr = shared_u32(smem);
    int nbytes = box_rows * box_cols * (int)sizeof(__nv_bfloat16);
    if (threadIdx.x == 0) {
        mbarrier_init(bar, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mbarrier_arrive_expect(bar, nbytes);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_addr), "l"(map), "r"(origin_col), "r"(origin_row), "r"(bar)
            : "memory");
    }
    int ready = 0;
    for (int spin = 0; spin < 10000000 && ready == 0; ++spin) {
        ready = mbarrier_try(bar, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    int n = box_rows * box_cols;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int row = i / box_cols;
        int col = i - row * box_cols;
        int logical = col * (int)sizeof(__nv_bfloat16);
        int physical = logical;
        if (swizzle_bytes == 128) {
            int chunk = logical >> 4;
            int within = logical & 15;
            physical = ((chunk ^ (row & 7)) << 4) | within;
        }
        dst[i] = smem[row * box_cols + physical / 2];
    }
}

// K-major, no swizzle, bf16. T = 8. LBO = 128 bytes, SBO = 256 bytes.
// offset in elements = (mn % 8) * 8 + (mn / 8) * 128 + (k % 8) + (k / 8) * 64.
__device__ int kmajor_offset(int mn, int k) {
    return (mn & 7) * 8 + (mn >> 3) * 128 + (k & 7) + (k >> 3) * 64;
}

__device__ uint64_t gmma_desc(uint32_t start, int leading, int stride, int layout_type) {
    uint64_t enc_start = (uint64_t)((start & 0x3FFFF) >> 4);
    uint64_t enc_lead = (uint64_t)((leading & 0x3FFFF) >> 4);
    uint64_t enc_stride = (uint64_t)((stride & 0x3FFFF) >> 4);
    return enc_start | (enc_lead << 16) | (enc_stride << 32) | ((uint64_t)layout_type << 62);
}

__global__ void wgmma_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    float* out) {
    __shared__ __align__(256) __nv_bfloat16 smem_a[1024];
    __shared__ __align__(256) __nv_bfloat16 smem_b[256];
    for (int row = threadIdx.x; row < 64; row += blockDim.x) {
        for (int k = 0; k < 16; ++k) {
            smem_a[kmajor_offset(row, k)] = a[row * 16 + k];
        }
    }
    for (int col = threadIdx.x; col < 8; col += blockDim.x) {
        for (int k = 0; k < 16; ++k) {
            smem_b[kmajor_offset(col, k)] = b[k * 8 + col];
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

    uint64_t desc_a = gmma_desc(shared_u32(smem_a), 128, 256, 0);
    uint64_t desc_b = gmma_desc(shared_u32(smem_b), 128, 256, 0);
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, p, %7, %8, %9, %10;\n\t"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b), "r"(0), "n"(1), "n"(1), "n"(0), "n"(0));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float regs[4] = {d0, d1, d2, d3};
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int v0 = v & 1;
        int v1 = v >> 1;
        int row = (lane >> 2) + 16 * warp + 8 * v1;
        int col = 2 * (lane & 3) + v0;
        out[row * 8 + col] = regs[v];
    }
}

}  // namespace

torch::Tensor tma_copy_cuda(
    torch::Tensor src,
    int64_t origin_col,
    int64_t origin_row,
    int64_t box_cols,
    int64_t box_rows,
    int64_t swizzle_bytes) {
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.scalar_type() == torch::kBFloat16, "src must be bfloat16");
    TORCH_CHECK(src.dim() == 2 && src.is_contiguous(), "src must be a contiguous matrix");
    TORCH_CHECK(box_rows > 0 && box_cols > 0, "the box must be non-empty");
    TORCH_CHECK(box_rows * box_cols <= 512, "this kernel holds at most 512 bf16 elements");
    TORCH_CHECK(swizzle_bytes == 0 || swizzle_bytes == 128, "swizzle must be 0 or 128");
    if (swizzle_bytes == 128) {
        TORCH_CHECK(box_cols * 2 == 128, "a 128-byte swizzle needs a 64-element bf16 inner box");
    }
    const int64_t rows = src.size(0);
    const int64_t cols = src.size(1);
    TORCH_CHECK(cols % 8 == 0, "the row stride must be a multiple of 16 bytes");
    auto* ptr = src.data_ptr();
    TORCH_CHECK(reinterpret_cast<uintptr_t>(ptr) % 16 == 0, "the base pointer must be 16-byte aligned");

    c10::cuda::CUDAGuard guard(src.device());
    alignas(64) CUtensorMap host_map;
    unsigned long long global_dim[2] = {
        static_cast<unsigned long long>(cols),
        static_cast<unsigned long long>(rows)};
    unsigned long long global_stride[1] = {
        static_cast<unsigned long long>(cols * sizeof(__nv_bfloat16))};
    unsigned int box_dim[2] = {
        static_cast<unsigned int>(box_cols),
        static_cast<unsigned int>(box_rows)};
    unsigned int elem_stride[2] = {1u, 1u};
    CUtensorMapSwizzle swizzle = swizzle_bytes == 128
        ? CU_TENSOR_MAP_SWIZZLE_128B
        : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult status = encode_tiled()(
        &host_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        ptr,
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS, "cuTensorMapEncodeTiled returned ", static_cast<int>(status));

    auto map = torch::empty(
        {static_cast<int64_t>(sizeof(CUtensorMap))},
        torch::TensorOptions().dtype(torch::kUInt8).device(src.device()));
    cudaMemcpy(map.data_ptr(), &host_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    auto dst = torch::empty(
        {box_rows, box_cols},
        torch::TensorOptions().dtype(torch::kBFloat16).device(src.device()));
    tma_box_kernel<<<1, 128>>>(
        reinterpret_cast<const CUtensorMap*>(map.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
        static_cast<int>(origin_col),
        static_cast<int>(origin_row),
        static_cast<int>(box_cols),
        static_cast<int>(box_rows),
        static_cast<int>(swizzle_bytes));
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return dst;
}

torch::Tensor wgmma_64x8x16_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
    TORCH_CHECK(a.is_contiguous() && a.sizes() == torch::IntArrayRef({64, 16}), "a must be (64, 16)");
    TORCH_CHECK(b.is_contiguous() && b.sizes() == torch::IntArrayRef({16, 8}), "b must be (16, 8)");
    c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty(
        {64, 8},
        torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
    wgmma_kernel<<<1, 128>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        out.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}
"""


_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        from lab_common import load_cuda

        _MODULE = load_cuda(
            "hopper_gemm_26",
            _CUDA,
            _CPP,
            ["tma_copy_cuda", "wgmma_64x8x16_cuda"],
            arch="sm_90a",
        )
    return _MODULE


def tma_copy(src, origin_col: int, origin_row: int, box_cols: int, box_rows: int, swizzle: str):
    """Copy one box with TMA. `swizzle` is `none` or `128B`."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    swizzle_bytes = {"none": 0, "128B": 128}[swizzle]
    return _module().tma_copy_cuda(src, origin_col, origin_row, box_cols, box_rows, swizzle_bytes)


def wgmma_64x8x16(a, b):
    """One m64n8k16 wgmma. Returns fp32 of shape (64, 8)."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        return None
    return _module().wgmma_64x8x16_cuda(a, b)
