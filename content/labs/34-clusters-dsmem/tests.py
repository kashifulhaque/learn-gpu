"""Tests for lab 34.

The first eight checks need no GPU. They model a thread block cluster in
Python: rank and peer addressing, a DSMEM histogram compared with
torch.bincount, the exit rule, a cluster reduction, how many clusters fit on
an H100, and the L2 traffic that TMA multicast removes. The last two checks
run sm_90 kernels and need an H100. A machine without one skips them. A
skipped kernel is not a failure, and the job is not sent to the A100.

global_index_as_rank is a plausible wrong peer_block. It returns
target_rank, as if the rank were an index into the grid. Block 5 in clusters
of 4 addressing rank 2 reaches block 6, and the wrong answer is block 2,
which sits in another cluster.

forgot_final_sync is a plausible wrong exit_steps. It lets every block exit
at its own finish step. A slower peer's remote write then lands on shared
memory that no longer exists, so the update is lost.

total_slots_over_cluster is a plausible wrong max_active_clusters. It
divides all the SM slots by the cluster size and ignores that a cluster must
fit inside one GPC. With clusters of 3 it returns 44, and 42 fit.

divide_everything_by_cluster is a plausible wrong multicast_l2_bytes. It
divides both operands by the cluster size. Multicast saves only the shared
operand, so the saving is (cluster_size - 1) / cluster_size of that operand.
"""

from __future__ import annotations

import random

from lab_common import Checks

NAMES = (
    "cluster_rank",
    "peer_block",
    "bin_owner",
    "simulate_dsmem_histogram",
    "exit_steps",
    "simulate_cluster_reduction",
    "blocks_per_sm",
    "max_active_clusters",
    "multicast_l2_bytes",
    "cluster_histogram",
    "global_atomic_histogram",
    "cluster_reduce",
    "block_atomic_reduce",
)

# Illustrative H100 SXM5 split: 8 GPCs and 132 SMs are published, the per-GPC
# counts are not. This split is an assumption that sums to 132.
GPC_SMS = (16, 16, 16, 16, 16, 16, 18, 18)
MIB = 1024 * 1024


def _skip_reason() -> str | None:
    msg = "SKIP: the cluster kernels were not run. "
    try:
        import torch
    except ImportError:
        return msg + "torch is not installed, so there is no H100 to compile them on."
    if not torch.cuda.is_available():
        return msg + "No CUDA GPU is visible. They need an H100 (sm_90) and are not sent to the A100 endpoint."
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        return msg + f"The visible GPU is sm_{major}{minor}, not sm_90. They are not sent to the A100 endpoint."
    return None


def _call(fn, *args):
    """Return (value, None), or (None, error text) when the reader's code raises."""
    try:
        return fn(*args), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    reader = submission

    # Check 1 calls only cluster_rank. Implementing that one function passes it.
    def rank_inside_cluster():
        cases = (((0, 4), 0), ((5, 4), 1), ((7, 4), 3), ((13, 8), 5), ((9, 1), 0))
        for args, want in cases:
            got, err = _call(reader.cluster_rank, *args)
            if err:
                return False, f"expected {want} for block {args[0]} in clusters of {args[1]}, actual {err}"
            if got != want:
                return False, f"expected {want} for block {args[0]} in clusters of {args[1]}, actual {got}"
        return True, "expected rank 1 for block 5 in clusters of 4, actual 1"

    c.check("the rank is the block's index inside its cluster", rank_inside_cluster)

    def peer():
        # global_index_as_rank fails here: block 5, rank 2 is block 6, not block 2.
        cases = (((5, 4, 2), 6), ((13, 8, 0), 8), ((0, 4, 3), 3), ((6, 4, 1), 5), ((17, 2, 0), 16))
        for args, want in cases:
            got, err = _call(reader.peer_block, *args)
            if err:
                return False, f"expected block {want} for {args}, actual {err}"
            if got != want:
                return False, f"expected block {want} for block {args[0]}, cluster {args[1]}, rank {args[2]}, actual block {got}"
        return True, "expected block 6 for block 5, cluster 4, rank 2, actual block 6"

    c.check("map_shared_rank reaches the block with that rank in your own cluster", peer)

    def owner():
        cases = (((1000, 251), (3, 247)), ((0, 251), (0, 0)), ((251, 251), (1, 0)), ((250, 251), (0, 250)))
        for args, want in cases:
            got, err = _call(reader.bin_owner, *args)
            if err:
                return False, f"expected {want} for bin {args[0]}, actual {err}"
            if tuple(got) != want:
                return False, f"expected (rank, offset) {want} for bin {args[0]} with {args[1]} bins per rank, actual {tuple(got)}"
        return True, "expected rank 3, offset 247 for bin 1000, actual rank 3, offset 247"

    c.check("a bin belongs to one rank, at one offset in its slice", owner)

    def histogram():
        import torch

        rng = random.Random(34)
        torch.manual_seed(34)
        configs = (
            (1001, 4, 8, 5000),
            (64, 8, 16, 3000),
            (10, 3, 6, 200),
            (5, 16, 16, 100),
        )
        for num_bins, cluster, blocks, n in configs:
            skew = [min(num_bins - 1, int(num_bins * rng.random() ** 4)) for _ in range(n)]
            want = torch.bincount(torch.tensor(skew), minlength=num_bins).tolist()
            out, err = _call(reader.simulate_dsmem_histogram, skew, num_bins, cluster, blocks)
            if err:
                return False, f"expected a histogram of {n} values in {num_bins} bins, actual {err}"
            got = list(out["hist"])
            if got != want:
                at = next(i for i in range(num_bins) if got[i] != want[i])
                return False, (
                    f"expected {want[at]} in bin {at} (torch.bincount), actual {got[at]} "
                    f"with {num_bins} bins, clusters of {cluster}"
                )
            bins_per_rank = -(-num_bins // cluster)
            remote = sum(
                1 for i, v in enumerate(skew) if (v // bins_per_rank) != (i % blocks) % cluster
            )
            if out["remote"] != remote or out["local"] != n - remote:
                return False, f"expected {remote} remote updates, actual {out['remote']}"
            atomics = (blocks // cluster) * num_bins
            if out["global_atomics"] != atomics:
                return False, f"expected {atomics} global atomics, actual {out['global_atomics']}"
        return True, "expected torch.bincount for every config, actual a match"

    c.check("the DSMEM histogram equals torch.bincount", histogram)

    def exit_rule():
        # forgot_final_sync fails here: with the sync, every block exits at step 9.
        finish = [3, 9, 5, 6]
        writes = [(8, 0), (9, 2)]
        for sync, want in ((True, [9, 9, 9, 9]), (False, [3, 9, 5, 6])):
            got, err = _call(reader.exit_steps, list(finish), sync)
            if err:
                return False, f"expected {want} with final_sync={sync}, actual {err}"
            if list(got) != want:
                return False, f"expected exit steps {want} with final_sync={sync}, actual {list(got)}"
        safe = reader.exit_steps(list(finish), True)
        lost = sum(1 for step, rank in writes if step > safe[rank])
        if lost:
            return False, f"expected 0 lost remote writes with the final sync, actual {lost}"
        return True, "expected 0 lost writes with the sync and 2 without, actual 0 and 2"

    c.check("a final cluster.sync keeps shared memory alive for slower peers", exit_rule)

    def reduction():
        rng = random.Random(34)
        for cluster in (1, 2, 3, 5, 8, 16):
            clusters = 3
            partials = [rng.randrange(-50, 50) for _ in range(clusters * cluster)]
            want = [sum(partials[i * cluster : (i + 1) * cluster]) for i in range(clusters)]
            levels = (cluster - 1).bit_length()
            out, err = _call(reader.simulate_cluster_reduction, list(partials), cluster)
            if err:
                return False, f"expected sums {want} for clusters of {cluster}, actual {err}"
            sums, syncs = out
            if list(sums) != want:
                return False, f"expected cluster sums {want} for clusters of {cluster}, actual {list(sums)}"
            if syncs != 1 + levels:
                return False, f"expected {1 + levels} cluster.sync calls for clusters of {cluster}, actual {syncs}"
        return True, "expected exact sums and 1 + ceil(log2(size)) syncs, actual a match"

    c.check("the cluster reduction is exact for any cluster size", reduction)

    def fit():
        # total_slots_over_cluster fails here: clusters of 3 fit 42 times, not 44.
        occupancy = (
            ((256, 32, 0), 8),
            ((1024, 64, 0), 1),
            ((128, 32, 48 * 1024), 4),
            ((128, 32, 100 * 1024), 2),
            ((128, 32, 228 * 1024), 0),
        )
        for args, want in occupancy:
            got, err = _call(reader.blocks_per_sm, *args)
            if err:
                return False, f"expected {want} blocks per SM for {args}, actual {err}"
            if got != want:
                return False, f"expected {want} blocks per SM for threads, regs, smem {args}, actual {got}"
        clusters = (
            ((1, 1, False), 132),
            ((2, 1, False), 66),
            ((3, 1, False), 42),
            ((4, 1, False), 32),
            ((5, 1, False), 24),
            ((8, 1, False), 16),
            ((16, 1, False), 0),
            ((16, 1, True), 8),
            ((17, 1, True), 0),
            ((4, 2, False), 66),
        )
        for (size, per_sm, opt_in), want in clusters:
            got, err = _call(reader.max_active_clusters, size, GPC_SMS, per_sm, opt_in)
            if err:
                return False, f"expected {want} clusters of {size}, actual {err}"
            if got != want:
                return False, (
                    f"expected {want} clusters of {size} with {per_sm} block(s) per SM "
                    f"and non-portable={opt_in}, actual {got}"
                )
        return True, "expected 42 clusters of 3 on the 8-GPC split, actual 42"

    c.check("clusters pack inside GPCs, so fewer fit than total slots / size", fit)

    def multicast():
        # divide_everything_by_cluster fails here: (2, 1) saves a third of the traffic.
        shape = (4096, 4096, 4096, 128, 256, 2)
        cases = (
            ((1, 1), 1536 * MIB, 1536 * MIB),
            ((2, 1), 1536 * MIB, 1024 * MIB),
            ((2, 2), 1536 * MIB, 768 * MIB),
            ((4, 1), 1536 * MIB, 768 * MIB),
        )
        for (cm, cn), base, multi in cases:
            got, err = _call(reader.multicast_l2_bytes, *shape, cm, cn)
            if err:
                return False, f"expected ({base}, {multi}) for a {cm} by {cn} cluster, actual {err}"
            if tuple(got) != (base, multi):
                return False, f"expected (baseline, multicast) = ({base}, {multi}) bytes for a {cm} by {cn} cluster, actual {tuple(got)}"
        # A grid that is not a multiple of the cluster: 3 tiles along M, clusters of 2.
        got, err = _call(reader.multicast_l2_bytes, 384, 256, 64, 128, 256, 2, 2, 1)
        if err:
            return False, f"expected (147456, 114688) for a ragged grid, actual {err}"
        if tuple(got) != (147456, 114688):
            return False, f"expected (147456, 114688) bytes for a ragged grid, actual {tuple(got)}"
        return True, "expected 1024 MiB with a 2 by 1 cluster, a saving of 1/3, actual 1024 MiB"

    c.check("multicast removes (size - 1) / size of the shared operand's L2 loads", multicast)

    _gpu_section(c, reader)
    return c.finish()


def _gpu_section(c: Checks, reader) -> None:
    """Kernel checks. Skipped, not failed, when this machine has no H100."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("hist_speedup_vs_global_atomics", "skipped (no H100)")
        c.metric("reduce_gbs", "skipped (no H100)")
        return

    import torch

    from lab_common import bandwidth_gbs, cuda_time

    torch.manual_seed(34)

    def hist_case():
        for num_bins, n in ((1001, (1 << 20) + 37), (3, 5000)):
            x = (torch.rand(n, device="cuda") ** 4 * num_bins).int().clamp_(max=num_bins - 1)
            want = torch.bincount(x.long(), minlength=num_bins)
            got, err = _call(reader.cluster_histogram, x, num_bins)
            if err:
                return False, f"expected a histogram of {n} values in {num_bins} bins, actual {err}"
            if got is None or tuple(got.shape) != (num_bins,):
                shown = None if got is None else tuple(got.shape)
                return False, f"expected shape ({num_bins},), actual {shown}"
            bad = (got.long() != want).nonzero(as_tuple=False)
            if bad.numel():
                at = int(bad[0])
                return False, f"expected {int(want[at])} in bin {at} (torch.bincount), actual {int(got[at])}"
        num_bins, n = 1024, 1 << 24
        x = (torch.rand(n, device="cuda") ** 4 * num_bins).int().clamp_(max=num_bins - 1)
        t_cluster = cuda_time(lambda: reader.cluster_histogram(x, num_bins), warmup=5, iters=20)
        t_global = cuda_time(lambda: reader.global_atomic_histogram(x, num_bins), warmup=5, iters=20)
        speedup = t_global / t_cluster
        c.metric("hist_speedup_vs_global_atomics", round(speedup, 3))
        return True, (
            f"expected bin counts equal to torch.bincount, actual a match. On a skewed 2^24-element input "
            f"the cluster kernel took {t_cluster:.4f} ms and global atomics took {t_global:.4f} ms. "
            "No pass/fail threshold: this run records the metric."
        )

    c.check("the cluster histogram kernel equals torch.bincount", hist_case)

    def reduce_case():
        n = (1 << 22) + 5
        x = torch.rand(n, device="cuda")
        want = float(x.double().sum())
        got, err = _call(reader.cluster_reduce, x)
        if err:
            return False, f"expected {want:.1f}, actual {err}"
        if got is None:
            return False, f"expected {want:.1f}, actual None"
        actual = float(got.reshape(-1)[0])
        if abs(actual - want) > 1e-4 * abs(want):
            return False, f"expected {want:.1f}, actual {actual:.1f} (relative tolerance 1e-4)"
        big = torch.rand(1 << 26, device="cuda")
        t_cluster = cuda_time(lambda: reader.cluster_reduce(big), warmup=5, iters=20)
        t_block = cuda_time(lambda: reader.block_atomic_reduce(big), warmup=5, iters=20)
        gbs = bandwidth_gbs(big.numel() * 4, t_cluster)
        c.metric("reduce_gbs", round(gbs, 1))
        return True, (
            f"expected {want:.1f}, actual {actual:.1f}. On 2^26 floats the cluster reduction took "
            f"{t_cluster:.4f} ms and the one-atomic-per-block version took {t_block:.4f} ms. "
            "No pass/fail threshold: this run records the metric."
        )

    c.check("the cluster reduction kernel matches an fp64 sum", reduce_case)
