"""Lab 25: bucketed data-parallel training and a ZeRO-1 Adam.

You write three things. `assign_buckets` splits parameters into gradient
buckets. `BucketedDDP` all-reduces each bucket asynchronously, from a
post-accumulate gradient hook, while backward is still running.
`Zero1Adam` keeps Adam's moments for one shard of the parameters and
rebuilds the full parameter vector with a reduce-scatter and an all-gather.

The tests start two processes, connect them with the gloo backend on CPU
tensors, and call your code inside each rank. The GPU is not required for
those checks. A separate NCCL sweep runs only when the machine has two GPUs.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def assign_buckets(
    param_numels: list[int], bucket_cap_bytes: int, dtype_bytes: int = 4
) -> list[list[int]]:
    """Group parameter indices into buckets, walking from the last index to the first.

    A parameter's size is `numel * dtype_bytes`. Add it to the current bucket
    when it fits. When it does not fit, close the current bucket and start a
    new one. A parameter larger than `bucket_cap_bytes` is a bucket by itself,
    which is the case of an empty current bucket that cannot fit the parameter
    either. Indices inside a bucket stay in the order you visit them, so they
    decrease. The first bucket you close holds the last parameters, because
    that is the bucket backward fills first.

    For `[16, 16, 16, 48]`, a cap of 128, and 4 bytes per element, return
    `[[3], [2, 1], [0]]`.
    """
    # TODO
    raise NotImplementedError


class BucketedDDP:
    """A module wrapper that averages gradients in buckets during backward.

    Construct it after `dist.init_process_group`. The constructor broadcasts
    every parameter from rank 0, builds buckets with `assign_buckets`, and
    registers a `register_post_accumulate_grad_hook` on each parameter.

    `forward` resets the per-bucket ready counts, then calls the wrapped
    module. Each hook marks its parameter ready. When every parameter in a
    bucket is ready, and every earlier bucket has already launched, launch
    one `dist.all_reduce` of that bucket's flattened gradients with
    `async_op=True`. Launch in bucket-index order so every rank issues the
    same collectives in the same order.

    `finish_gradient_synchronization` waits for those all-reduces, divides
    each reduced buffer by the world size, and copies it back into the
    parameter `.grad` tensors. Call it after `backward` and before the
    optimizer step. Every parameter must receive a gradient.
    """

    def __init__(self, module: nn.Module, bucket_cap_bytes: int) -> None:
        # TODO
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # TODO
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # TODO
        raise NotImplementedError


class Zero1Adam:
    """Adam with optimizer state for one shard of the flattened parameters.

    Flatten `params` in the order you are given them. Rank `r` of `world`
    owns the slice `[r * shard, (r + 1) * shard)`, where
    `shard = numel / world`. Raise `ValueError` when `world` does not divide
    the flattened length.

    Store `exp_avg` and `exp_avg_sq` as float32 tensors of length `shard`
    and nothing else of that length. `step` reduce-scatters the flattened
    gradients, divides the shard by the world size, applies one Adam update
    (betas default to `(0.9, 0.999)`, `eps` to `1e-8`, step count starting
    at 1), and all-gathers the updated parameters back into `params`.
    """

    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        # TODO
        raise NotImplementedError

    def step(self) -> None:
        # TODO
        raise NotImplementedError
