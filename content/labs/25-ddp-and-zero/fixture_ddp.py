"""Process-group plumbing for the bucketed DDP lab. You don't edit this file.

`launch` starts processes on the gloo backend with CPU tensors. A deadlock
in collective order shows up as a timeout. `launch_nccl` is the two-GPU
sweep, and the tests call it only when the machine has two GPUs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import tempfile
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _load(path: str):
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in module.parameters()])


class Toy(nn.Module):
    """Four parameters whose byte sizes are 64, 64, 64, and 192.

    A 128-byte cap therefore makes three buckets: the last parameter alone,
    the middle two together, and the first alone.
    """

    def __init__(self) -> None:
        super().__init__()
        self.p0 = nn.Parameter(torch.empty(16))
        self.p1 = nn.Parameter(torch.empty(16))
        self.p2 = nn.Parameter(torch.empty(16))
        self.p3 = nn.Parameter(torch.empty(48))
        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        total = self.p0.sum() + self.p1.sum() + self.p2.sum() + self.p3.sum()
        return scale * total


class _Counter:
    """Records dist.all_reduce calls and which phase issued them."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.phase = "other"
        self._real = dist.all_reduce

    def install(self) -> None:
        def wrapped(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            self.calls.append({"async_op": bool(async_op), "phase": self.phase})
            return self._real(tensor, op=op, group=group, async_op=async_op)

        dist.all_reduce = wrapped

    def remove(self) -> None:
        dist.all_reduce = self._real


def _adam(flat: torch.Tensor, mean: float, steps: int, lr: float) -> torch.Tensor:
    """Two Adam steps on a constant gradient, matching Zero1Adam's defaults."""
    grad = torch.full_like(flat, mean)
    moment = torch.zeros_like(flat)
    second = torch.zeros_like(flat)
    out = flat.clone()
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        moment = beta1 * moment + (1 - beta1) * grad
        second = beta2 * second + (1 - beta2) * grad * grad
        mhat = moment / (1 - beta1**step)
        vhat = second / (1 - beta2**step)
        out = out - lr * mhat / (vhat.sqrt() + eps)
    return out


def _job_ddp(rank, world, sub, job):
    del job
    torch.manual_seed(1000 + rank)
    model = Toy()
    wrapper = sub.BucketedDDP(model, 128)
    flat = _flat(model)
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    init_err = max((item - gathered[0]).abs().max().item() for item in gathered)

    counter = _Counter()
    counter.install()
    try:
        counter.phase = "backward"
        wrapper(torch.tensor(float(rank + 1))).backward()
        counter.phase = "finish"
        wrapper.finish_gradient_synchronization()
        counter.phase = "other"
    finally:
        counter.remove()

    mean = (world + 1) / 2
    grad_err = 0.0
    for param in model.parameters():
        if param.grad is None:
            grad_err = float("inf")
            break
        grad_err = max(grad_err, (param.grad - mean).abs().max().item())
    err = torch.tensor([grad_err], dtype=torch.float64)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)

    torch.optim.SGD(model.parameters(), lr=0.25).step()
    flat = _flat(model)
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    step_err = max((item - gathered[0]).abs().max().item() for item in gathered)
    return {
        "init_err": init_err,
        "grad_err": err.item(),
        "step_err": step_err,
        "calls": len(counter.calls),
        "async_calls": sum(1 for call in counter.calls if call["async_op"]),
        "backward_calls": sum(1 for call in counter.calls if call["phase"] == "backward"),
    }


def _job_zero(rank, world, sub, job):
    del job
    torch.manual_seed(7)
    model = Toy()
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)
    initial = _flat(model).clone()
    opt = sub.Zero1Adam(model.parameters(), lr=0.1)

    reduce_scatter_calls = 0
    all_gather_calls = 0
    real_reduce = dist.reduce_scatter_tensor
    real_gather = dist.all_gather_into_tensor
    real_reduce_list = dist.reduce_scatter
    real_gather_list = dist.all_gather

    def reduce_scatter_tensor(*args, **kwargs):
        nonlocal reduce_scatter_calls
        reduce_scatter_calls += 1
        return real_reduce(*args, **kwargs)

    def reduce_scatter(*args, **kwargs):
        nonlocal reduce_scatter_calls
        reduce_scatter_calls += 1
        return real_reduce_list(*args, **kwargs)

    def all_gather_into_tensor(*args, **kwargs):
        nonlocal all_gather_calls
        all_gather_calls += 1
        return real_gather(*args, **kwargs)

    def all_gather(*args, **kwargs):
        nonlocal all_gather_calls
        all_gather_calls += 1
        return real_gather_list(*args, **kwargs)

    dist.reduce_scatter_tensor = reduce_scatter_tensor
    dist.reduce_scatter = reduce_scatter
    dist.all_gather_into_tensor = all_gather_into_tensor
    dist.all_gather = all_gather
    try:
        for _ in range(2):
            model.zero_grad()
            model(torch.tensor(float(rank + 1))).backward()
            opt.step()
    finally:
        dist.reduce_scatter_tensor = real_reduce
        dist.reduce_scatter = real_reduce_list
        dist.all_gather_into_tensor = real_gather
        dist.all_gather = real_gather_list

    state_bytes = (
        opt.exp_avg.numel() * opt.exp_avg.element_size()
        + opt.exp_avg_sq.numel() * opt.exp_avg_sq.element_size()
    )
    packed = torch.tensor(
        [float(state_bytes), float(reduce_scatter_calls), float(all_gather_calls)],
        dtype=torch.float64,
    )
    # Min across ranks: a rank that skipped a collective cannot hide on rank 0.
    dist.all_reduce(packed, op=dist.ReduceOp.MIN)
    got = _flat(model)
    ref = _adam(initial, (world + 1) / 2, steps=2, lr=0.1)
    weight_err = (got - ref).abs().max()
    dist.all_reduce(weight_err, op=dist.ReduceOp.MAX)
    return {
        "state_bytes": int(packed[0].item()),
        "reduce_scatter": int(packed[1].item()),
        "all_gather": int(packed[2].item()),
        "weight_err": weight_err.item(),
        "numel": int(initial.numel()),
        "world": world,
    }


_JOBS = {"ddp": _job_ddp, "zero": _job_zero}


def _gloo_worker(rank, world, port, job, out_path, submission_path):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:
        try:
            submission = _load(submission_path)
            result = _JOBS[job["name"]](rank, world, submission, job)
        except Exception:
            result = {"error": traceback.format_exc()}
        if rank == 0:
            with open(out_path, "w") as handle:
                json.dump(result, handle)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch(world_size: int, job: dict, submission_path: str, timeout: float = 60.0) -> dict:
    """Run `job` in `world_size` gloo processes and return rank 0's result."""
    descriptor, out_path = tempfile.mkstemp(suffix=".json")
    os.close(descriptor)
    try:
        context = mp.spawn(
            _gloo_worker,
            args=(world_size, _free_port(), job, out_path, submission_path),
            nprocs=world_size,
            join=False,
        )
        deadline = time.monotonic() + timeout
        finished = False
        try:
            while not finished and time.monotonic() < deadline:
                finished = context.join(timeout=max(0.1, deadline - time.monotonic()))
        except Exception as exc:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(str(exc)[-1500:]) from None
        if not finished:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError(
                f"{world_size} ranks did not finish in {timeout:.0f} s. "
                "A mismatched collective order is the usual cause."
            )
        with open(out_path) as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _fit_alpha_beta(sizes, times) -> tuple[float, float]:
    """Ordinary least squares for t = alpha + n / B. Returns alpha seconds and B in GB/s."""
    count = len(sizes)
    mean_x = sum(sizes) / count
    mean_y = sum(times) / count
    var = sum((x - mean_x) ** 2 for x in sizes)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(sizes, times)) / var
    alpha = mean_y - slope * mean_x
    return alpha, 1.0 / slope / 1e9


def _nccl_worker(rank, world, port, sizes, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        points = []
        for numel in sizes:
            buf = torch.ones(numel, device=rank)
            for _ in range(3):
                dist.all_reduce(buf)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            iters = 8
            start.record()
            for _ in range(iters):
                dist.all_reduce(buf)
            end.record()
            torch.cuda.synchronize()
            if rank == 0:
                points.append((numel * buf.element_size(), start.elapsed_time(end) / 1e3 / iters))
        if rank == 0:
            alpha, bandwidth = _fit_alpha_beta([n for n, _ in points], [t for _, t in points])
            largest_n, largest_t = points[-1]
            # nccl-tests busbw for all-reduce: algbw * 2 (world - 1) / world.
            algbw = largest_n / largest_t / 1e9
            busbw = algbw * 2 * (world - 1) / world
            with open(out_path, "w") as handle:
                json.dump(
                    {
                        "alpha_s": alpha,
                        "bandwidth_gbs": bandwidth,
                        "busbw_gbs": busbw,
                        "largest_bytes": largest_n,
                    },
                    handle,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch_nccl(timeout: float = 120.0) -> dict:
    """Time NCCL all-reduce on two GPUs and fit alpha and beta. Needs two GPUs."""
    sizes = [2**18, 2**20, 2**22]
    descriptor, out_path = tempfile.mkstemp(suffix=".json")
    os.close(descriptor)
    try:
        context = mp.spawn(
            _nccl_worker,
            args=(2, _free_port(), sizes, out_path),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + timeout
        finished = False
        while not finished and time.monotonic() < deadline:
            finished = context.join(timeout=max(0.1, deadline - time.monotonic()))
        if not finished:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("the NCCL sweep did not finish")
        with open(out_path) as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
