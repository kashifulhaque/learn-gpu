"""Lab 25 solution: bucketed overlap and a ZeRO-1 Adam."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def assign_buckets(
    param_numels: list[int], bucket_cap_bytes: int, dtype_bytes: int = 4
) -> list[list[int]]:
    """Group parameter indices into buckets, last parameter first."""
    buckets: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for index in range(len(param_numels) - 1, -1, -1):
        nbytes = param_numels[index] * dtype_bytes
        if current and current_bytes + nbytes > bucket_cap_bytes:
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(index)
        current_bytes += nbytes
    if current:
        buckets.append(current)
    return buckets


class BucketedDDP:
    """Average gradients with one asynchronous all-reduce per bucket."""

    def __init__(self, module: nn.Module, bucket_cap_bytes: int) -> None:
        self.module = module
        self.params = [p for p in module.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("BucketedDDP needs at least one parameter")
        dtype_bytes = self.params[0].element_size()
        self.buckets = assign_buckets(
            [p.numel() for p in self.params], bucket_cap_bytes, dtype_bytes
        )
        self.param_bucket: dict[int, int] = {}
        for bucket_index, indices in enumerate(self.buckets):
            for index in indices:
                self.param_bucket[id(self.params[index])] = bucket_index
        with torch.no_grad():
            for param in self.params:
                dist.broadcast(param, src=0)
        self.handles: list[tuple] = []
        self.pending: list[int] = []
        self.ready: list[bool] = []
        self.next_bucket = 0
        for param in self.params:
            param.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param: torch.Tensor) -> None:
        bucket_index = self.param_bucket[id(param)]
        self.pending[bucket_index] -= 1
        if self.pending[bucket_index] == 0:
            self.ready[bucket_index] = True
            self._drain()

    def _drain(self) -> None:
        while self.next_bucket < len(self.buckets) and self.ready[self.next_bucket]:
            self._launch(self.next_bucket)
            self.next_bucket += 1

    def _launch(self, bucket_index: int) -> None:
        flats = [self.params[index].grad.reshape(-1) for index in self.buckets[bucket_index]]
        # Clone so the all-reduce buffer stays alive until finish, and so
        # dividing it does not change .grad before the collective completes.
        flat = torch.cat(flats).clone()
        handle = dist.all_reduce(flat, async_op=True)
        self.handles.append((handle, bucket_index, flat))

    def forward(self, *args, **kwargs):
        self.pending = [len(bucket) for bucket in self.buckets]
        self.ready = [False] * len(self.buckets)
        self.next_bucket = 0
        self.handles = []
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        world = dist.get_world_size()
        for handle, bucket_index, flat in self.handles:
            handle.wait()
            flat.div_(world)
            offset = 0
            for index in self.buckets[bucket_index]:
                grad = self.params[index].grad
                count = grad.numel()
                grad.copy_(flat[offset : offset + count].view_as(grad))
                offset += count
        self.handles.clear()


class Zero1Adam:
    """Adam whose moments cover 1/world of the flattened parameters."""

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        numel = sum(p.numel() for p in self.params)
        if self.world == 0 or numel % self.world != 0:
            raise ValueError(
                f"flattened parameter count {numel} must be divisible by world size {self.world}"
            )
        self.numel = numel
        self.shard = numel // self.world
        self.lo = self.rank * self.shard
        self.exp_avg = torch.zeros(self.shard, dtype=torch.float32)
        self.exp_avg_sq = torch.zeros(self.shard, dtype=torch.float32)

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).float() for p in self.params])

    def _flatten_grads(self) -> torch.Tensor:
        return torch.cat([p.grad.detach().reshape(-1).float() for p in self.params])

    def _write_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for param in self.params:
            count = param.numel()
            param.data.copy_(flat[offset : offset + count].view_as(param).to(dtype=param.dtype))
            offset += count

    def step(self) -> None:
        grads = self._flatten_grads()
        shard_grad = torch.empty(self.shard, dtype=torch.float32)
        dist.reduce_scatter_tensor(shard_grad, grads)
        shard_grad.div_(self.world)

        self.t += 1
        beta1, beta2 = self.betas
        self.exp_avg.mul_(beta1).add_(shard_grad, alpha=1 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(shard_grad, shard_grad, value=1 - beta2)
        mhat = self.exp_avg / (1 - beta1**self.t)
        vhat = self.exp_avg_sq / (1 - beta2**self.t)
        updated = self._flatten_params()[self.lo : self.lo + self.shard]
        updated = updated - self.lr * mhat / (vhat.sqrt() + self.eps)

        full = torch.empty(self.numel, dtype=torch.float32)
        dist.all_gather_into_tensor(full, updated.contiguous())
        self._write_params(full)
