"""Tests for lab 14: a Triton matmul with grouped ordering and autotuning."""

import ast
import inspect

import torch
from lab_common import A100, Checks, close, cuda_time, require_cuda, tflops

# (M, N, K): from a single element to ragged sizes that no block size divides.
SHAPES = [
    (1, 1, 1),
    (1, 64, 64),
    (7, 5, 3),
    (127, 129, 65),
    (130, 257, 100),
    (256, 256, 256),
    (333, 1024, 1000),
    (1000, 1000, 1000),
]

# The tolerance is relative to the largest reference value. A correctly rounded
# fp16 result is within 2^-11 (4.9e-4) of its value and a bf16 result within
# 2^-8 (3.9e-3), so these leave a factor of 8 and 2.6 for summation order.
REL_TOL = {torch.float16: 4e-3, torch.bfloat16: 1e-2}


def _reference(a, b, bias=None, activation=None):
    """The float32 answer, from the same rounded inputs."""
    torch.backends.cuda.matmul.allow_tf32 = False
    out = a.float() @ b.float()
    if bias is not None:
        out = out + bias.float()
    if activation == "relu":
        out = torch.relu(out)
    elif activation == "silu":
        out = torch.nn.functional.silu(out)
    return out


def _compare(actual, expected, dtype):
    tol = REL_TOL[dtype] * max(expected.abs().max().item(), 1.0)
    return close(actual, expected, tol)


def _uses_block_ptr(submission) -> bool:
    """True when the submission's code calls the deprecated tl.make_block_ptr."""
    path = inspect.getsourcefile(submission)
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "make_block_ptr":
            return True
        if isinstance(node, ast.Name) and node.id == "make_block_ptr":
            return True
    return False


def _check_python_parts(c, submission):
    """The program-order helpers are plain Python and need no GPU."""
    grid_cases = [(1, 1, 1), (5, 7, 3), (9, 9, 3), (10, 3, 4), (4, 4, 8), (8, 1, 3), (13, 6, 5)]

    def permutation():
        for pm, pn, g in grid_cases:
            seen = [submission.grouped_tile(p, pm, pn, g) for p in range(pm * pn)]
            expected = {(m, n) for m in range(pm) for n in range(pn)}
            if len(set(seen)) != pm * pn or set(seen) != expected:
                return False, f"{pm} x {pn} tiles with group_m={g}: the program ids don't visit every tile exactly once"
        return True, f"{len(grid_cases)} grids, each tile visited once"

    c.check("grouped_tile visits every tile exactly once, including a short last group", permutation)

    def row_major():
        for pm, pn in [(4, 6), (1, 5), (7, 1)]:
            for p in range(pm * pn):
                if tuple(submission.grouped_tile(p, pm, pn, 1)) != (p // pn, p % pn):
                    return False, f"group_m=1 is not row-major at pid {p} of a {pm} x {pn} grid"
        return True, ""

    c.check("grouped_tile with group_m=1 is row-major", row_major)

    def column_inside_group():
        got = [tuple(submission.grouped_tile(p, 9, 9, 3)) for p in range(4)]
        return got == [(0, 0), (1, 0), (2, 0), (0, 1)], f"9 x 9, group_m=3, pids 0 to 3: {got}"

    c.check("grouped_tile walks down a group's rows before moving right", column_inside_group)

    def tutorial_counts():
        row = submission.unique_tile_loads(9, 9, 9, 1, 9)
        grouped = submission.unique_tile_loads(9, 9, 9, 3, 9)
        return (row, grouped) == (90, 54), f"first 9 programs of a 9 x 9 grid: row-major {row}, group_m=3 {grouped} (expected 90 and 54)"

    c.check("unique_tile_loads reproduces the tutorial's 90 versus 54", tutorial_counts)

    def clamp():
        every = submission.unique_tile_loads(3, 4, 2, 2, 10_000)
        return every == (3 + 4) * 2, f"all tiles of a 3 x 4 grid: {every}, expected 14"

    c.check("unique_tile_loads clamps n_programs to the number of tiles", clamp)


def _check_configs(c, submission):
    import triton

    def valid():
        configs = submission.autotune_configs()
        if len(configs) < 4:
            return False, f"{len(configs)} configs; give the autotuner at least 4 to choose from"
        for cfg in configs:
            if not isinstance(cfg, triton.Config):
                return False, f"{cfg!r} is not a triton.Config"
            kw = cfg.kwargs
            missing = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"} - set(kw)
            if missing:
                return False, f"a config is missing {sorted(missing)}"
            bm, bn, bk = kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"]
            for name, v, lo in (("BLOCK_M", bm, 32), ("BLOCK_N", bn, 32), ("BLOCK_K", bk, 16)):
                if v < lo or v & (v - 1):
                    return False, f"{name}={v} must be a power of two of at least {lo}"
            smem = (bm * bk + bk * bn) * 2 * cfg.num_stages
            if smem > A100.SMEM_PER_BLOCK_MAX_BYTES:
                return False, f"{kw} with {cfg.num_stages} stages needs {smem} bytes of shared memory; the limit is {A100.SMEM_PER_BLOCK_MAX_BYTES}"
            acc_regs = bm * bn / (A100.WARP_SIZE * cfg.num_warps)
            if acc_regs > A100.MAX_REGISTERS_PER_THREAD:
                return False, f"{kw} with {cfg.num_warps} warps needs {acc_regs:.0f} accumulator registers per thread; the limit is {A100.MAX_REGISTERS_PER_THREAD}"
        return True, f"{len(configs)} configs, each within the shared-memory and register limits"

    c.check("autotune_configs returns valid triton.Config objects that fit the A100", valid)


def run(submission):
    c = Checks()
    if not c.require(
        submission, "grouped_tile", "unique_tile_loads", "autotune_configs", "matmul", "matmul_fixed"
    ):
        return c.finish()

    _check_python_parts(c, submission)
    c.check(
        "the code doesn't call the deprecated tl.make_block_ptr",
        lambda: (not _uses_block_ptr(submission), "use pointer arithmetic with masks instead"),
    )
    _check_configs(c, submission)

    device = require_cuda()
    props = torch.cuda.get_device_properties(0)
    print(f"      device = {props.name}", flush=True)
    if props.multi_processor_count != A100.SM_COUNT:
        print(f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs); the ratios below still hold.", flush=True)

    torch.manual_seed(0)

    # Correctness over awkward shapes, in both 16-bit formats.
    for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        shapes = SHAPES if dtype == torch.float16 else [(1, 1, 1), (127, 129, 65), (333, 1024, 1000)]
        for m, n, k in shapes:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            c.check(
                f"{label} matmul {m} x {n} x {k}",
                lambda a=a, b=b, dtype=dtype: _compare(submission.matmul(a, b).float(), _reference(a, b), dtype),
            )

    def transposed_b():
        a = torch.randn(127, 65, device=device, dtype=torch.float16)
        b = torch.randn(129, 65, device=device, dtype=torch.float16).t()  # shape (65, 129), strides (1, 65)
        return _compare(submission.matmul(a, b).float(), _reference(a, b), torch.float16)

    c.check("a transposed B (non-unit stride along K) gives the right answer", transposed_b)

    def output_is_16_bit():
        a = torch.randn(33, 40, device=device, dtype=torch.bfloat16)
        b = torch.randn(40, 17, device=device, dtype=torch.bfloat16)
        out = submission.matmul(a, b)
        return out.dtype == torch.bfloat16 and out.shape == (33, 17), f"got {out.dtype} {tuple(out.shape)}"

    c.check("the output has the input dtype and shape (M, N)", output_is_16_bit)

    def every_group_size():
        a = torch.randn(1000, 100, device=device, dtype=torch.float16)
        b = torch.randn(100, 1000, device=device, dtype=torch.float16)
        ref = _reference(a, b)
        for g in (1, 3, 8):
            ok, detail = _compare(submission.matmul_fixed(a, b, 128, 128, 32, g, 4, 4).float(), ref, torch.float16)
            if not ok:
                return False, f"group_m={g}: {detail}"
        return True, "group_m = 1, 3, and 8 on an 8 x 8 grid of tiles, so the last group is short for 3"

    c.check("matmul_fixed is correct for every group size, including a short last group", every_group_size)

    # Optional: the fused epilogue.
    if getattr(submission, "SUPPORTS_EPILOGUE", False):
        for m, n, k in [(127, 129, 65), (512, 512, 512)]:
            a = torch.randn(m, k, device=device, dtype=torch.float16)
            b = torch.randn(k, n, device=device, dtype=torch.float16)
            bias = torch.randn(n, device=device, dtype=torch.float16)
            for act in (None, "relu", "silu"):
                c.check(
                    f"epilogue: bias + {act or 'no activation'}, {m} x {n} x {k}",
                    lambda a=a, b=b, bias=bias, act=act: _compare(
                        submission.matmul(a, b, bias=bias, activation=act).float(),
                        _reference(a, b, bias, act),
                        torch.float16,
                    ),
                )
        c.metric("epilogue_checked", 1)
    else:
        print("      note: SUPPORTS_EPILOGUE is False, so the optional fused bias and activation aren't tested.", flush=True)
        c.metric("epilogue_checked", 0)

    # Performance. The roofline puts a 4096^3 fp16 matmul at 2*4096^3 = 137.4
    # GFLOP, which is 0.44 ms at the 312 TFLOP/s peak. The Triton tutorial
    # reports about 220 TFLOP/s on an A100 and parity with cuBLAS at this size,
    # so half of torch.matmul is a loose bar for a correct tiled kernel.
    def performance():
        n = 4096
        a = torch.randn(n, n, device=device, dtype=torch.float16)
        b = torch.randn(n, n, device=device, dtype=torch.float16)
        flops = 2 * n**3
        ms_torch = cuda_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
        ms_mine = cuda_time(lambda: submission.matmul(a, b), warmup=5, iters=20)
        t_torch, t_mine = tflops(flops, ms_torch), tflops(flops, ms_mine)
        c.metric("torch_tflops_4096", round(t_torch, 1))
        c.metric("tflops_4096", round(t_mine, 1))
        c.metric("pct_of_torch_4096", round(100 * t_mine / t_torch, 1))
        c.metric("pct_of_peak_4096", round(100 * t_mine / A100.TENSOR_FP16_TFLOPS, 1))
        c.add(
            "the measured rate is below the tensor-core peak",
            t_mine <= 1.1 * A100.TENSOR_FP16_TFLOPS,
            f"{t_mine:.1f} TFLOP/s against a {A100.TENSOR_FP16_TFLOPS:.0f} peak; a higher figure means the timing is wrong",
        )
        c.add(
            "at least 50% of torch.matmul at 4096 x 4096 x 4096",
            t_mine >= 0.5 * t_torch,
            f"{t_mine:.1f} versus {t_torch:.1f} TFLOP/s",
        )

        # Grouped against row-major order, with one fixed tile so only the order changes.
        ms_row = cuda_time(lambda: submission.matmul_fixed(a, b, 128, 128, 32, 1, 4, 4), warmup=5, iters=20)
        ms_grp = cuda_time(lambda: submission.matmul_fixed(a, b, 128, 128, 32, 8, 4, 4), warmup=5, iters=20)
        c.metric("group_speedup", round(ms_row / ms_grp, 3))
        print("      group_speedup is the time with group_m=1 divided by the time with group_m=8. "
              "It's reported, not graded: see the chapter for what the arithmetic predicts.", flush=True)
        del a, b

        # Ragged sizes: no pointer is 16-byte aligned past the first row, and cuBLAS has to cope too.
        mr, nr, kr = 4095, 4097, 4093
        a = torch.randn(mr, kr, device=device, dtype=torch.float16)
        b = torch.randn(kr, nr, device=device, dtype=torch.float16)
        flops_r = 2 * mr * nr * kr
        ms_torch_r = cuda_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
        ms_mine_r = cuda_time(lambda: submission.matmul(a, b), warmup=5, iters=20)
        c.metric("tflops_ragged", round(tflops(flops_r, ms_mine_r), 1))
        c.metric("pct_of_torch_ragged", round(100 * ms_torch_r / ms_mine_r, 1))

    try:
        performance()
    except Exception as exc:  # noqa: BLE001 - an unfinished kernel must fail the lab, not crash the harness
        c.add("performance run", False, f"{type(exc).__name__}: {exc}")
    return c.finish()
