"""Recorded compute-sanitizer reports for lab 27.

Each constant is the text one tool prints for one of the five buggy kernels in
`starter.py`. The reader's parser reads them on any machine, with no GPU.

These reports are written in the format that the NVIDIA Compute Sanitizer
documentation describes, and they are trimmed to the first findings, as
`--print-limit` does. They are not captured from a run on the lab A100, so
treat the addresses and byte offsets as representative. The tool, the kind of
error, the kernel, the source line, and the thread and block coordinates are
the facts the lab grades. The reports use these thread coordinates:

- `OOB`: 1000 floats, 256 threads per block, 4 blocks. Thread 1000 is block 3,
  thread 232, because 3 * 256 + 232 = 1000.
- `BLOCK_RACE`: 256 threads in the tree reduction.
- `WARP_RACE`: 128 threads; the unrolled last warp.
- `TAIL_SYNC`: 1000 floats, 128 threads per block. The tail block is block 7,
  and its first thread index is 896, so 104 threads reach the barrier.
- `HALO_INIT`: 128 threads and a tile of 130 floats.
"""

# One allocation of 4000 bytes (1000 floats). Thread 1000 reads, and then
# writes, 0 bytes past the end. Twenty-four threads (1000 to 1023) each make
# a read and a write, so the tool counts 48 errors and prints two.
OOB = """\
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at scale_shift_kernel(const float *, float *, int)+0x70 in /tmp/lab27/oob.cu:11
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f3a5ce00fa0 is out of bounds
=========     and is 0 bytes after the nearest allocation at 0x7f3a5ce00000 of size 4,000 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2b8316]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========     Host Frame: scale_shift(at::Tensor) [0x4a1f]
=========                in /tmp/torch_extensions/lab27_oob/lab27_oob.so
=========
========= Invalid __global__ write of size 4 bytes
=========     at scale_shift_kernel(const float *, float *, int)+0x90 in /tmp/lab27/oob.cu:11
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f3a5ce01fa0 is out of bounds
=========     and is 0 bytes after the nearest allocation at 0x7f3a5ce01000 of size 4,000 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2b8316]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========
========= Target application returned an error
========= ERROR SUMMARY: 48 errors
"""

# Phase k reads s[t + d] that another thread wrote in phase k - 1. Thread 64
# wrote s[64] and thread 0 reads it, with no barrier between the two steps.
BLOCK_RACE = """\
========= COMPUTE-SANITIZER
========= Error: Potential RAW hazard detected at __shared__ 0x100 in block (0,0,0) :
=========     Write Thread (64,0,0) at 0x1f0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Read Thread (0,0,0) at 0x1d0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Current Value : 0, Incoming Value : 3
=========
========= Error: Potential RAW hazard detected at __shared__ 0x104 in block (0,0,0) :
=========     Write Thread (65,0,0) at 0x1f0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Read Thread (1,0,0) at 0x1d0 in /tmp/lab27/race.cu:17:block_sum_kernel(const float *, float *)
=========     Current Value : 0, Incoming Value : 5
=========
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
========= ERROR SUMMARY: 2 errors
"""

# Inside one warp, lane 16 wrote s[16] and lane 0 reads it with no __syncwarp
# between them. The tool reports this at warning level, and the error count
# stays 0. A parser that trusts the summary alone calls this report clean.
WARP_RACE = """\
========= COMPUTE-SANITIZER
========= Warning: (Warp Level Programming) Potential RAW hazard detected at __shared__ 0x40 in block (0,0,0) :
=========     Write Thread (16,0,0) at 0x2a0 in /tmp/lab27/warp.cu:21:warp_block_sum_kernel(const float *, float *)
=========     Read Thread (0,0,0) at 0x2c0 in /tmp/lab27/warp.cu:22:warp_block_sum_kernel(const float *, float *)
=========     Current Value : 0, Incoming Value : 7
=========
========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)
========= ERROR SUMMARY: 0 errors
"""

# Block 7 holds elements 896 to 1023, and elements 1000 and up do not exist.
# Threads 104 to 127 returned before the barrier. The kernel is a template, so
# its name carries the template argument.
TAIL_SYNC = """\
========= COMPUTE-SANITIZER
========= Barrier error detected. Divergent thread(s) in warp
=========     at void tail_pair_sum_kernel<128>(const float *, float *, int)+0x110 in /tmp/lab27/sync.cu:14
=========     by thread (96,0,0) in block (7,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2b8316]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========
========= ERROR SUMMARY: 1 error
"""

# The two halo cells, tile[0] and tile[129], are never written. Thread 0 reads
# tile[0]. Thread 127 reads tile[129]. Every one of the 8 blocks does both.
HALO_INIT = """\
========= COMPUTE-SANITIZER
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at void stencil3_kernel<128>(const float *, float *, int)+0x1c0 in /tmp/lab27/init.cu:15
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x0
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2b8316]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at void stencil3_kernel<128>(const float *, float *, int)+0x1c0 in /tmp/lab27/init.cu:15
=========     by thread (127,0,0) in block (0,0,0)
=========     Address 0x204
=========     Saved host backtrace up to driver entry point at kernel launch time
=========
========= ERROR SUMMARY: 16 errors
"""

# What a clean run prints.
CLEAN = """\
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
"""

ALL = {
    "OOB": OOB,
    "BLOCK_RACE": BLOCK_RACE,
    "WARP_RACE": WARP_RACE,
    "TAIL_SYNC": TAIL_SYNC,
    "HALO_INIT": HALO_INIT,
    "CLEAN": CLEAN,
}
