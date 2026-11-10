"""Two-GPU plumbing for lab 36. You don't edit this file.

`launch_overlap` starts two NCCL processes and times an all-gather followed by
a matmul two ways: serially, and as a chunked pipeline in which the matmul on
chunk j runs while the all-gather of chunk j + 1 is in flight. The tests call
it only when the machine has two GPUs.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M_PER_RANK = 2048  # rows of A on each rank
K = 8192
N = 4096
CHUNKS = 4


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serial(shard, weight, world):
    gathered = torch.empty(world * shard.shape[0], shard.shape[1], dtype=shard.dtype, device=shard.device)
    dist.all_gather_into_tensor(gathered, shard)
    return gathered @ weight


def _chunked(shard, weight, world, chunks):
    """All-gather `chunks` row slices asynchronously, and multiply each as it lands."""
    rows = shard.shape[0]
    step = rows // chunks
    slices = [shard[j * step : (j + 1) * step].contiguous() for j in range(chunks)]
    buffers = [torch.empty(world * step, shard.shape[1], dtype=shard.dtype, device=shard.device) for _ in slices]
    works = [dist.all_gather_into_tensor(buffers[j], slices[j], async_op=True) for j in range(chunks)]
    out = torch.empty(world * rows, weight.shape[1], dtype=shard.dtype, device=shard.device)
    for j in range(chunks):
        works[j].wait()  # the current stream waits for chunk j only
        tile = buffers[j] @ weight
        for rank in range(world):
            out[rank * rows + j * step : rank * rows + (j + 1) * step] = tile[rank * step : (rank + 1) * step]
    return out


def _timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _worker(rank, world, port, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
        shard = torch.randn(M_PER_RANK, K, device="cuda", dtype=torch.bfloat16, generator=gen)
        weight = torch.randn(K, N, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(99)) / (K**0.5)
        want = _serial(shard, weight, world)
        got = _chunked(shard, weight, world, CHUNKS)
        err = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        serial_ms = _timed(lambda: _serial(shard, weight, world))
        overlap_ms = _timed(lambda: _chunked(shard, weight, world, CHUNKS))
        if rank == 0:
            with open(out_path, "w") as handle:
                json.dump(
                    {
                        "rel_err": err / scale,
                        "serial_ms": serial_ms,
                        "overlap_ms": overlap_ms,
                        "gathered_bytes": world * M_PER_RANK * K * 2,
                    },
                    handle,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch_overlap(timeout: float = 240.0) -> dict:
    """Run the serial and chunked all-gather plus matmul on two GPUs."""
    descriptor, out_path = tempfile.mkstemp(suffix=".json")
    os.close(descriptor)
    try:
        context = mp.spawn(_worker, args=(2, _free_port(), out_path), nprocs=2, join=False)
        deadline = time.monotonic() + timeout
        finished = False
        while not finished and time.monotonic() < deadline:
            finished = context.join(timeout=max(0.1, deadline - time.monotonic()))
        if not finished:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("the two-GPU overlap run did not finish")
        with open(out_path) as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
