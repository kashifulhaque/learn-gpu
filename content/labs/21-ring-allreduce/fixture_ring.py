"""Process-group plumbing for the ring all-reduce lab. You don't edit this file.

`launch` starts `world_size` processes with torch.multiprocessing, connects
them with the gloo backend (CPU tensors, so it runs anywhere), and runs one
job in every rank. Rank 0 writes the job's result to a JSON file, and `launch`
returns it. A deadlock in your send and receive order shows up as a timeout
here instead of a hung lab.
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def _make_input(rank: int, n: int, kind: str) -> torch.Tensor:
    """Each rank's data. 'int' is integer valued, so any summation order is exact."""
    g = torch.Generator().manual_seed(1000 + rank)
    if kind == "int":
        return torch.randint(-8, 9, (n,), generator=g).to(torch.float32)
    return torch.randn(n, generator=g)


class _Counter:
    """Counts the bytes this rank sends through dist.isend and dist.send."""

    def __init__(self) -> None:
        self.bytes = 0
        self._isend, self._send = dist.isend, dist.send

    def __enter__(self):
        def isend(tensor, *a, **k):
            self.bytes += tensor.numel() * tensor.element_size()
            return self._isend(tensor, *a, **k)

        def send(tensor, *a, **k):
            self.bytes += tensor.numel() * tensor.element_size()
            return self._send(tensor, *a, **k)

        dist.isend, dist.send = isend, send
        return self

    def __exit__(self, *exc):
        dist.isend, dist.send = self._isend, self._send


def _job_correctness(rank, world, sub, job):
    out = {"cases": []}
    for n in job["sizes"]:
        for kind in ("int", "normal"):
            x = _make_input(rank, n, kind)
            want = x.clone()
            dist.all_reduce(want)
            got = x.clone()
            with _Counter() as counter:
                sub.ring_allreduce(got, rank, world)
            err = (got - want).abs().max().item() if n else 0.0
            err_t = torch.tensor([err, float(counter.bytes)])
            worst = err_t.clone()
            dist.all_reduce(worst, op=dist.ReduceOp.MAX)
            lowest = err_t.clone()
            dist.all_reduce(lowest, op=dist.ReduceOp.MIN)
            out["cases"].append(
                {
                    "n": n,
                    "kind": kind,
                    "max_err": worst[0].item(),
                    "max_bytes": int(worst[1].item()),
                    "min_bytes": int(lowest[1].item()),
                }
            )
    # Reduce-scatter on its own: the rank must own chunk (rank + 1) % world.
    n = job["sizes"][-1]
    x = _make_input(rank, n, "int")
    want = x.clone()
    dist.all_reduce(want)
    got = x.clone()
    owned = sub.ring_reduce_scatter(got, rank, world)
    lo, hi = sub.ring_chunks(n, world)[owned]
    ok = int(owned == (rank + 1) % world and torch.equal(got[lo:hi], want[lo:hi]))
    flag = torch.tensor([ok])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    out["reduce_scatter_ok"] = bool(flag.item())
    return out


def _job_timing(rank, world, sub, job):
    n = job["n"]
    x = _make_input(rank, n, "normal")
    ring_t, built_t = [], []
    for i in range(job["warmup"] + job["iters"]):
        y = x.clone()
        dist.barrier()
        t0 = time.perf_counter()
        sub.ring_allreduce(y, rank, world)
        t1 = time.perf_counter()
        z = x.clone()
        dist.barrier()
        t2 = time.perf_counter()
        dist.all_reduce(z)
        t3 = time.perf_counter()
        if i >= job["warmup"]:
            ring_t.append(t1 - t0)
            built_t.append(t3 - t2)
    t = torch.tensor([_median(ring_t), _median(built_t)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return {"ring_s": t[0].item(), "builtin_s": t[1].item(), "n": n, "world": world}


def _job_pingpong(rank, world, sub, job):
    points = []
    for n_bytes in job["sizes"]:
        buf = torch.zeros(max(1, n_bytes // 4))
        times = []
        for i in range(job["warmup"] + job["iters"]):
            dist.barrier()
            t0 = time.perf_counter()
            if rank == 0:
                dist.send(buf, 1)
                dist.recv(buf, 1)
            else:
                dist.recv(buf, 0)
                dist.send(buf, 0)
            t1 = time.perf_counter()
            if i >= job["warmup"]:
                times.append((t1 - t0) / 2)  # one way
        points.append((buf.numel() * 4, _median(times)))
    return {"points": points}


_JOBS = {"correctness": _job_correctness, "timing": _job_timing, "pingpong": _job_pingpong}


def _worker(rank, world, port, job, out_path):
    # The ranks only talk to each other on this machine. Pin gloo to the loopback
    # interface instead of letting it resolve the hostname, which in a container
    # can point at an address no other rank can reach.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    # A spawned child starts with a fresh import state; the submission sits
    # beside this file, so make sure that directory is searched.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=job.get("dist_timeout", 60)),
    )
    try:
        sub = importlib.import_module("submission")
        result = _JOBS[job["name"]](rank, world, sub, job)
        if rank == 0:
            with open(out_path, "w") as f:
                json.dump(result, f)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def launch(world_size: int, job: dict, timeout: float = 120.0) -> dict:
    """Run `job` in `world_size` processes; return rank 0's result, or raise."""
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        ctx = mp.spawn(
            _worker,
            args=(world_size, _free_port(), job, out_path),
            nprocs=world_size,
            join=False,
        )
        deadline = time.monotonic() + timeout
        finished = False
        try:
            # ProcessContext.join returns after any one rank exits, so loop.
            while not finished and time.monotonic() < deadline:
                finished = ctx.join(timeout=max(0.1, deadline - time.monotonic()))
        except Exception as exc:  # a rank raised; surface the first traceback
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(str(exc)[-1500:]) from None
        if not finished:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            raise TimeoutError(
                f"{world_size} ranks did not finish in {timeout:.0f} s. "
                "A deadlock is the usual cause: two ranks that both block in send."
            )
        with open(out_path) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
