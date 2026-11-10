"""CUDA sources for lab 39. tests.py compiles them on a GPU.

The message-passing kernel takes the memory orders for the flag from your
mp_orders(). The grid-barrier kernel builds the same generation barrier you
wrote in Python out of cuda::atomic_ref, and runs it next to
cooperative_groups' grid sync.
"""

from __future__ import annotations

FUNCTIONS = ["mp_litmus", "coresident_blocks", "grid_barrier"]

CPP_SRC = """
torch::Tensor mp_litmus(int64_t pairs, int64_t launches, bool ordered);
int64_t coresident_blocks(int64_t threads);
torch::Tensor grid_barrier(int64_t blocks, int64_t threads, int64_t rounds, bool cooperative);
"""

_TEMPLATE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

constexpr int STRIDE = 32;  // ints between two pairs: 128 bytes, one cache line each

// Message passing. Even blocks are producers and the odd block next to each is
// the consumer. Pair p writes data[p] = 1 and then flag[p] = 1. The consumer
// spins on the flag, then reads the data. stats[0] counts consumers that saw
// the flag, stats[1] counts those that then read 0 from the data, which is
// the outcome the model forbids when the flag is released and acquired, and
// stats[2] counts consumers that gave up.
template <bool ORDERED>
__global__ void mp_kernel(int* data, int* flag, unsigned long long* stats, int pairs, int spin_limit) {
    int pair = (blockIdx.x / 2) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    cuda::atomic_ref<int, cuda::thread_scope_device> x(data[pair * STRIDE]);
    cuda::atomic_ref<int, cuda::thread_scope_device> y(flag[pair * STRIDE]);
    if ((blockIdx.x & 1) == 0) {
        x.store(1, cuda::std::memory_order_relaxed);
        if (ORDERED) {
            y.store(1, cuda::std::memory_order_@STORE@);
        } else {
            y.store(1, cuda::std::memory_order_relaxed);
        }
    } else {
        int seen = 0;
        for (int i = 0; i < spin_limit && !seen; ++i) {
            if (ORDERED) {
                seen = y.load(cuda::std::memory_order_@LOAD@);
            } else {
                seen = y.load(cuda::std::memory_order_relaxed);
            }
        }
        if (seen) {
            int v = x.load(cuda::std::memory_order_relaxed);
            atomicAdd(&stats[0], 1ull);
            if (v == 0) atomicAdd(&stats[1], 1ull);
        } else {
            atomicAdd(&stats[2], 1ull);
        }
    }
}

torch::Tensor mp_litmus(int64_t pairs, int64_t launches, bool ordered) {
    auto ints = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto longs = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto data = torch::zeros({pairs * STRIDE}, ints);
    auto flag = torch::zeros({pairs * STRIDE}, ints);
    auto stats = torch::zeros({3}, longs);
    const int threads = 128;
    const int pair_blocks = static_cast<int>((pairs + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* counters = reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>());
    for (int64_t t = 0; t < launches; ++t) {
        data.zero_();
        flag.zero_();
        if (ordered) {
            mp_kernel<true><<<pair_blocks * 2, threads, 0, stream>>>(
                data.data_ptr<int>(), flag.data_ptr<int>(), counters, static_cast<int>(pairs), 1 << 12);
        } else {
            mp_kernel<false><<<pair_blocks * 2, threads, 0, stream>>>(
                data.data_ptr<int>(), flag.data_ptr<int>(), counters, static_cast<int>(pairs), 1 << 12);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return stats;
}

// A generation-counter barrier for a grid whose blocks are all resident.
// Thread 0 of each block arrives. The spin is bounded so a mistake ends the
// kernel instead of hanging the lab. It returns false when the bound runs out.
__device__ bool atomic_grid_barrier(unsigned* count, unsigned* gen, unsigned num_blocks) {
    __syncthreads();
    bool ok = true;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> c(*count);
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> g(*gen);
        unsigned my_gen = g.load(cuda::std::memory_order_acquire);          // before arriving
        if (c.fetch_add(1, cuda::std::memory_order_acq_rel) == num_blocks - 1) {
            c.store(0, cuda::std::memory_order_relaxed);                    // safe: nobody re-arrives until gen moves
            g.fetch_add(1, cuda::std::memory_order_release);                // the release
        } else {
            long spins = 0;
            while (g.load(cuda::std::memory_order_acquire) == my_gen) {
                if (++spins > (1L << 26)) { ok = false; break; }
            }
        }
    }
    __syncthreads();
    return ok;
}

// Round r: block b writes flags[r][b], waits at the barrier, then each thread
// reads one other block's flag for the same round. A zero is a block that
// passed the barrier before its neighbour arrived.
template <bool COOP>
__global__ void barrier_kernel(unsigned* count, unsigned* gen, int* flags, int* bad, int rounds) {
    const unsigned num_blocks = gridDim.x;
    for (int r = 0; r < rounds; ++r) {
        if (threadIdx.x == 0) flags[r * num_blocks + blockIdx.x] = 1;
        bool ok = true;
        if (COOP) {
            cg::this_grid().sync();
        } else {
            ok = atomic_grid_barrier(count, gen, num_blocks);
        }
        if (!ok && threadIdx.x == 0) atomicAdd(bad, 1000000);
        unsigned other = (blockIdx.x + threadIdx.x) % num_blocks;
        if (flags[r * num_blocks + other] != 1) atomicAdd(bad, 1);
        __syncthreads();
    }
}

int64_t coresident_blocks(int64_t threads) {
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, barrier_kernel<true>, static_cast<int>(threads), 0));
    int device = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return static_cast<int64_t>(per_sm) * sms;
}

torch::Tensor grid_barrier(int64_t blocks, int64_t threads, int64_t rounds, bool cooperative) {
    auto ints = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto count = torch::zeros({1}, ints);
    auto gen = torch::zeros({1}, ints);
    auto flags = torch::zeros({rounds * blocks}, ints);
    auto bad = torch::zeros({1}, ints);
    unsigned* count_ptr = reinterpret_cast<unsigned*>(count.data_ptr<int>());
    unsigned* gen_ptr = reinterpret_cast<unsigned*>(gen.data_ptr<int>());
    int* flags_ptr = flags.data_ptr<int>();
    int* bad_ptr = bad.data_ptr<int>();
    int rounds_i = static_cast<int>(rounds);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cooperative) {
        void* args[] = {&count_ptr, &gen_ptr, &flags_ptr, &bad_ptr, &rounds_i};
        cudaError_t err = cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(barrier_kernel<true>), dim3(static_cast<unsigned>(blocks)),
            dim3(static_cast<unsigned>(threads)), args, 0, stream);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();  // clear the launch error so later calls still work
            TORCH_CHECK(false, "cooperative launch failed: ", cudaGetErrorString(err));
        }
    } else {
        barrier_kernel<false><<<static_cast<unsigned>(blocks), static_cast<unsigned>(threads), 0, stream>>>(
            count_ptr, gen_ptr, flags_ptr, bad_ptr, rounds_i);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return bad;
}
"""


def build_source(store_order: str, load_order: str) -> str:
    """Fill the flag's memory orders into the message-passing kernel."""
    if store_order not in ("relaxed", "release"):
        raise ValueError(f"a store order must be 'relaxed' or 'release', not {store_order!r}")
    if load_order not in ("relaxed", "acquire"):
        raise ValueError(f"a load order must be 'relaxed' or 'acquire', not {load_order!r}")
    return _TEMPLATE.replace("@STORE@", store_order).replace("@LOAD@", load_order)
