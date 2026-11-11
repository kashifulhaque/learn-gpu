"""Tests for lab 41: the capstone kernel leaderboard.

The first six checks run on any machine, in plain Python and torch-CPU. The last
four need a CUDA GPU and are skipped, with a message, without one.

Plausible wrong implementations that these checks catch:

- unfused_traffic: traffic that counts the MLP's hidden layer as written and
  read back. That is the eager baseline's traffic, not the fused kernel's floor.
- hbm_rating_floor: speed_of_light that divides by the rated 1935 GB/s, not the
  measured 1275 GB/s copy ceiling, so every memory floor comes out 34% too low.
- sum_not_max: speed_of_light that adds the compute and memory times.
- mean_not_median: robust_time that averages, so one 30 ms outlier moves it.
- no_warmup_drop: robust_time that keeps the first launches in the sample.
- linear_score: score that divides (ref - time) by (ref - floor), which
  disagrees with the log form at every point except the two ends.
- any_tie_accepted: topk_matches that compares only the values, and accepts a
  tied element that the stable sort would not pick.
- ordered_compare: topk_matches that compares the indices in order, and rejects
  a kernel that returns them unsorted.
- first_launch_timing: a kernel timed once, including compilation. The GPU
  timing check uses warmup and the L2 flush, and fails a time below the rated
  hardware limit, which is what a skipped synchronize or a cached input gives.
"""

from __future__ import annotations

import math

from lab_common import A100, Checks

CPU_NAMES = ("traffic", "speed_of_light", "robust_time", "score", "geomean", "topk_matches")
GPU_NAMES = ("grayscale_blur", "topk_rows", "mlp_block")

# Hand-derived traffic: (flops, bytes) per task and shape.
#   grayscale_blur (H, W): 22 FLOP and 16 bytes (12 in, 4 out) per pixel.
#   topk (rows, n, k): 4 rows n bytes in, 12 rows k bytes out (4 value, 8 index).
#   mlp (M, d, h): 4 M d h FLOP; 2 (2 M d + 2 d h + h + d) bytes.
TRAFFIC_CASES = (
    ("grayscale_blur", (3, 5), (330, 240)),
    ("topk", (2, 10, 3), (0, 152)),
    ("mlp", (1, 2, 3), (24, 42)),
    ("grayscale_blur", (4096, 4096), (369_098_752, 268_435_456)),
    ("topk", (4096, 32000, 8), (0, 524_681_216)),
    ("mlp", (4096, 1024, 4096), (68_719_476_736, 33_564_672)),
)
# Floors in ms from the cases above. Memory at 1275 GB/s, compute at the task's peak.
#   gray: 268,435,456 / 1.275e12 = 0.2105376 ms (compute 22 HW / 19.5e12 = 0.0189 ms)
#   topk: 524,681,216 / 1.275e12 = 0.4115147 ms
#   mlp:  68,719,476,736 / 312e12 = 0.2202547 ms (memory 0.0263 ms)
FLOOR_CASES = (
    ("grayscale_blur", (4096, 4096), 0.2105376125),
    ("topk", (4096, 32000, 8), 0.4115146793),
    ("mlp", (4096, 1024, 4096), 0.2202547331),
)
PEAK_TFLOPS = {"grayscale_blur": A100.FP32_TFLOPS, "topk": A100.FP32_TFLOPS, "mlp": A100.TENSOR_BF16_TFLOPS}


def _fail(want, got):
    return False, f"expected {want}, actual {got}"


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# The tests' own copies of the arithmetic, so the GPU checks don't depend on the reader's code.
def _traffic(task, shape):
    if task == "grayscale_blur":
        h, w = shape
        return 22 * h * w, 16 * h * w
    if task == "topk":
        rows, n, k = shape
        return 0, 4 * rows * n + 12 * rows * k
    m, d, h = shape
    return 4 * m * d * h, 2 * (2 * m * d + 2 * d * h + h + d)


def _floor_ms(task, shape, bw_gbs):
    flops, nbytes = _traffic(task, shape)
    return max(flops / (PEAK_TFLOPS[task] * 1e12), nbytes / (bw_gbs * 1e9)) * 1e3


def _score(time_ms, floor_ms, ref_ms):
    s = math.log(ref_ms / time_ms) / math.log(ref_ms / floor_ms)
    return min(1.0, max(0.0, s))


def run(submission):
    c = Checks()
    if not c.require(submission, *CPU_NAMES):
        return c.finish()
    s = submission

    # 1. The first check needs only traffic.
    def traffic_counts():
        for task, shape, want in TRAFFIC_CASES:
            got, err = _call(s.traffic, task, shape)
            if err:
                return _fail(f"{want} for {task}{shape}", err)
            if tuple(int(v) for v in got) != want:
                # unfused_traffic fails here on the MLP: it adds 4 M h bytes for the hidden layer.
                return _fail(f"(flops, bytes) = {want} for {task}{shape}", tuple(got))
        _, err = _call(s.traffic, "softmax", (4, 4))
        if err is None or "ValueError" not in err:
            return _fail("a ValueError for an unknown task", err or "no error")
        return True, "expected 33,564,672 bytes for the (4096, 1024, 4096) MLP, actual 33,564,672"

    c.check("traffic counts the ideal bytes and FLOPs of each task", traffic_counts)

    # 2. Roofline floors.
    def floors():
        for task, shape, want in FLOOR_CASES:
            got, err = _call(s.speed_of_light, task, shape)
            if err:
                return _fail(f"{want:.6f} ms for {task}{shape}", err)
            # hbm_rating_floor gives 0.1387 ms for the first case, and sum_not_max gives 0.2466 ms for the last.
            if abs(got - want) > 1e-5:
                return _fail(f"{want:.6f} ms for {task}{shape}", f"{got:.6f} ms")
        return True, "expected 0.220255 ms for the MLP, the compute floor, actual 0.220255"

    c.check("speed_of_light is the larger of the compute and memory floors, at the copy ceiling", floors)

    # 3. The timing harness's discipline.
    def timing():
        times = [50.0, 9.0, 1.2, 1.0, 1.1, 1.0, 30.0, 1.05]
        got, err = _call(s.robust_time, times, 2)
        if err:
            return _fail("1.075", err)
        # mean_not_median gives 5.9, and no_warmup_drop gives 1.15.
        if abs(got - 1.075) > 1e-9:
            return _fail("1.075 after dropping 2 warmup times and taking the median", got)
        got, err = _call(s.robust_time, [7.0, 3.0, 5.0], 0)
        if err or abs(got - 5.0) > 1e-9:
            return _fail("5.0 for [7, 3, 5] with no warmup", err or got)
        got, err = _call(s.robust_time, [9.0, 1.0, 2.0, 3.0], 1)
        if err or abs(got - 2.0) > 1e-9:
            return _fail("2.0 for the times [1, 2, 3] that remain", err or got)
        _, err = _call(s.robust_time, [1.0, 2.0], 2)
        if err is None or "ValueError" not in err:
            return _fail("a ValueError when warmup drops every time", err or "no error")
        return True, "expected 1.075, actual 1.075"

    c.check("robust_time drops the warmup iterations and reports the median", timing)

    # 4. The score and the aggregate.
    def scoring():
        # ref 4, floor 1: time 2 is halfway on a log scale (0.5). A linear form gives 2/3.
        cases = ((2.0, 1.0, 4.0, 0.5), (4.0, 1.0, 4.0, 0.0), (1.0, 1.0, 4.0, 1.0), (8.0, 1.0, 4.0, 0.0), (0.5, 1.0, 4.0, 1.0))
        for t, fl, ref, want in cases:
            got, err = _call(s.score, t, fl, ref)
            if err:
                return _fail(f"{want} for time {t}, floor {fl}, ref {ref}", err)
            if abs(got - want) > 1e-9:
                return _fail(f"{want} for time {t}, floor {fl}, ref {ref}", got)
        for bad in ((0.0, 1.0, 4.0), (1.0, 4.0, 4.0), (1.0, 0.0, 4.0)):
            _, err = _call(s.score, *bad)
            if err is None or "ValueError" not in err:
                return _fail(f"a ValueError for score{bad}", err or "no error")
        got, err = _call(s.geomean, [1.0, 0.25])
        if err or abs(got - 0.5) > 1e-9:
            return _fail("0.5 for [1.0, 0.25]", err or got)
        got, err = _call(s.geomean, [0.8, 0.0, 1.0])
        if err or got != 0.0:
            return _fail("0.0 when one score is 0", err or got)
        _, err = _call(s.geomean, [])
        if err is None or "ValueError" not in err:
            return _fail("a ValueError for an empty list", err or "no error")
        return True, "expected 0.5 for time 2 against reference 4 and floor 1, actual 0.5"

    c.check("score is a log-headroom fraction and geomean combines shapes", scoring)

    # 5. The correctness grader for top-k.
    def tie_rule():
        import torch

        x = torch.tensor([[3.0, 1.0, 3.0, 3.0, 0.0], [5.0, 5.0, 4.0, 4.0, 4.0]])
        cases = (
            (torch.tensor([[2, 0], [1, 0]]), True, "the lowest tied indices, in any order"),
            (torch.tensor([[0, 3], [0, 1]]), False, "a tied index that is not the lowest"),
            (torch.tensor([[0, 2], [0, 2]]), False, "row 1 keeps index 2, which is not among the two largest"),
            (torch.tensor([[0, 0], [0, 1]]), False, "a repeated index"),
        )
        for idx, want, why in cases:
            got, err = _call(s.topk_matches, x, idx, 2)
            if err:
                return _fail(f"{want} for {why}", err)
            # any_tie_accepted says True for the second case; ordered_compare says False for the first.
            if bool(got) != want:
                return _fail(f"{want} for {why}: {idx.tolist()}", got)
        x2 = torch.tensor([[1.0, 4.0, 4.0, 4.0, 2.0]])
        got, err = _call(s.topk_matches, x2, torch.tensor([[1, 2, 3]]), 3)
        if err or not got:
            return _fail("True for k = 3 when every tied 4 is kept", err or got)
        return True, "expected [3, 1, 3, 3, 0] with k = 2 to keep indices {0, 2}, actual {0, 2}"

    c.check("topk_matches compares sets under the lowest-index tie rule", tie_rule)

    # 6. The grader's own references, against values worked out by hand. Runs on the fixture only.
    def references():
        import torch
        import fixture_capstone as fx

        ones = torch.ones(3, 3, 3)
        out = fx.grayscale_blur_ref(ones)
        want = torch.tensor([[9, 12, 9], [12, 16, 12], [9, 12, 9]], dtype=torch.float64) / 16.0
        if (out - want).abs().max().item() > 1e-12:
            return _fail(want.tolist(), out.tolist())
        x = torch.tensor([[3.0, 1.0, 3.0, 3.0, 0.0]])
        values, indices = fx.topk_ref(x, 2)
        if indices.tolist() != [[0, 2]] or values.tolist() != [[3.0, 3.0]]:
            return _fail("values [3, 3] at indices [0, 2]", (values.tolist(), indices.tolist()))
        one = torch.ones(1, 1, dtype=torch.bfloat16)
        zero = torch.zeros(1, dtype=torch.bfloat16)
        y = fx.mlp_ref(one, one, zero, one, zero)
        if abs(y.item() - 0.8413447460685429) > 1e-12:
            return _fail("gelu(1) = 0.841345", y.item())
        y = fx.mlp_ref(one, one, -one[0], one, one[0] * 2)
        if abs(y.item() - 2.0) > 1e-12:
            return _fail("gelu(0) + 2 = 2 when b1 = -1", y.item())
        return True, "expected a corner of 9/16 on an all-ones image, actual 0.5625"

    c.check("the grader's references match hand-derived values", references)

    _gpu_section(c, s)
    return c.finish()


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible. The kernels need an A100 (sm_80)"
    try:
        import triton  # noqa: F401
    except ImportError:
        pass  # A CUDA kernel through load_cuda does not need Triton.
    return None


def _gpu_section(c: Checks, s) -> None:
    reason = _skip_reason()
    metric_names = (
        "gray_gbs", "gray_speedup", "gray_score", "topk_gbs", "topk_speedup", "topk_score",
        "mlp_tflops", "mlp_speedup", "mlp_score", "capstone_score",
    )
    if reason is not None:
        print(f"SKIP: the GPU correctness and timing checks were not run, because {reason}.", flush=True)
        for name in metric_names:
            c.metric(name, "skipped (no GPU)")
        return
    if not c.require(s, *GPU_NAMES):
        return
    import torch

    import fixture_capstone as fx
    from lab_common import bandwidth_gbs, cuda_time, tflops

    dev = "cuda"
    good: set[str] = set()

    def gray_ok():
        # Sizes that are not multiples of any tile: 1 by 1, one row, one column, and odd shapes.
        for i, (h, w) in enumerate(((1, 1), (1, 37), (37, 1), (5, 7), (33, 1000), (1023, 1025))):
            x = fx.make_image(h, w, seed=i, device=dev)
            try:
                got = s.grayscale_blur(x)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                return _fail(f"a ({h}, {w}) float32 image", f"{type(exc).__name__}: {exc}")
            want = fx.grayscale_blur_ref(x)
            if tuple(got.shape) != (h, w):
                return _fail(f"shape {(h, w)}", tuple(got.shape))
            err = (got.double() - want).abs().max().item()
            # border_replicates: a blur that clamps coordinates reads 1.0 where the reference reads 9/16.
            if err > 1e-5:
                return _fail(f"max abs error at most 1e-5 for {(h, w)}", f"{err:.3e}")
        good.add("grayscale_blur")
        return True, "expected max abs error at most 1e-5 on six odd shapes, actual within tolerance"

    c.check("grayscale_blur matches the float64 reference on odd shapes", gray_ok)

    def topk_ok():
        k = 8
        cases = ((1, 8, False), (3, 1000, False), (5, 1000, True), (17, 32000, True), (64, 32000, False), (2, 100, "constant"))
        for i, (rows, n, mode) in enumerate(cases):
            x = fx.make_logits(rows, n, seed=100 + i, ties=(mode is True))
            if mode == "constant":
                x = torch.zeros(rows, n)
            x = x.to(dev)
            try:
                values, indices = s.topk_rows(x, k)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                return _fail(f"k = {k} of ({rows}, {n})", f"{type(exc).__name__}: {exc}")
            want_values, _ = fx.topk_ref(x, k)
            if tuple(values.shape) != (rows, k) or tuple(indices.shape) != (rows, k):
                return _fail(f"shapes {(rows, k)}", (tuple(values.shape), tuple(indices.shape)))
            if not torch.equal(values.float().cpu(), want_values.cpu()):
                return _fail(f"values equal to the sorted top {k} for {(rows, n)}", "values differ")
            if not s.topk_matches(x.cpu(), indices.cpu(), k):
                # keep_all_ties returns the right threshold and the wrong tied elements on the tie cases.
                return _fail(f"the lowest-index tied elements for {(rows, n)}", "a different index set")
        good.add("topk")
        return True, "expected exact values and the lowest-index tie rule on six cases, actual equal"

    c.check("topk_rows matches the stable-sort reference, ties included", topk_ok)

    def mlp_ok():
        for i, (m, d, h) in enumerate(((1, 64, 96), (100, 200, 300), (257, 192, 520), (130, 48, 33))):
            args = fx.make_mlp(m, d, h, seed=200 + i, device=dev)
            try:
                got = s.mlp_block(*args)
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                return _fail(f"(M, d, h) = {(m, d, h)}", f"{type(exc).__name__}: {exc}")
            want = fx.mlp_ref(*args)
            if tuple(got.shape) != (m, d):
                return _fail(f"shape {(m, d)}", tuple(got.shape))
            rel = (got.double() - want).abs().max().item() / want.abs().max().item()
            # drops_b1 and gelu_after_gemm give a relative error far above 2e-2 with these biases.
            if rel > 2e-2:
                return _fail(f"max error at most 2e-2 of max|y| for {(m, d, h)}", f"{rel:.3e}")
        good.add("mlp")
        return True, "expected max error at most 2% of max|y| on four odd shapes, actual within tolerance"

    c.check("mlp_block matches the float64 reference with bf16 inputs", mlp_ok)

    def timing():
        notes = []
        ok = True
        scores = []
        specs = {
            "grayscale_blur": ("gray", fx.GRAY_SHAPE, lambda: (fx.make_image(*fx.GRAY_SHAPE, device=dev),), s.grayscale_blur, fx.eager_grayscale_blur),
            "topk": ("topk", fx.TOPK_SHAPE, lambda: (fx.make_logits(*fx.TOPK_SHAPE[:2], device=dev), fx.TOPK_SHAPE[2]), s.topk_rows, fx.eager_topk),
            "mlp": ("mlp", fx.MLP_SHAPE, lambda: fx.make_mlp(*fx.MLP_SHAPE, device=dev), s.mlp_block, fx.eager_mlp),
        }
        keys = {"grayscale_blur": "grayscale_blur", "topk": "topk", "mlp": "mlp"}
        for task, (tag, shape, make, fn, eager) in specs.items():
            if keys[task] not in good:
                notes.append(f"{task}: not timed, the kernel is not correct")
                ok = False
                continue
            args = make()
            ms = cuda_time(lambda: fn(*args), warmup=5, iters=20, flush_l2=True)
            ref_ms = cuda_time(lambda: eager(*args), warmup=3, iters=10, flush_l2=True)
            flops, nbytes = _traffic(task, shape)
            floor = _floor_ms(task, shape, A100.COPY_CEILING_GBS)
            physical = _floor_ms(task, shape, A100.HBM_BW_GBS)
            sc = _score(ms, floor, ref_ms) if ref_ms > floor else 0.0
            scores.append(sc)
            c.metric(f"{tag}_ms", round(ms, 4))
            c.metric(f"{tag}_speedup", round(ref_ms / ms, 3))
            c.metric(f"{tag}_score", round(sc, 3))
            if task == "mlp":
                c.metric("mlp_tflops", round(tflops(flops, ms), 1))
                c.metric("mlp_fraction_of_peak", round(tflops(flops, ms) / A100.TENSOR_BF16_TFLOPS, 3))
            else:
                gbs = bandwidth_gbs(nbytes, ms)
                c.metric(f"{tag}_gbs", round(gbs, 1))
                if task == "grayscale_blur":
                    c.metric("gray_fraction_of_ceiling", round(gbs / A100.COPY_CEILING_GBS, 3))
                else:
                    c.metric("topk_torch_topk_ms", round(cuda_time(lambda: torch.topk(args[0], args[1]), warmup=3, iters=10, flush_l2=True), 4))
            if ms < 0.95 * physical:
                ok = False
                notes.append(f"{task}: expected at least {0.95 * physical:.4f} ms, the rated limit, actual {ms:.4f} ms. Check the timing")
            elif ms > 2.0 * ref_ms:
                ok = False
                notes.append(f"{task}: expected at most {2.0 * ref_ms:.4f} ms, twice the eager baseline, actual {ms:.4f} ms")
            else:
                notes.append(f"{task}: {ms:.4f} ms against {ref_ms:.4f} ms eager and a {floor:.4f} ms floor")
        if len(scores) == 3:
            c.metric("capstone_score", round(math.exp(sum(math.log(max(v, 1e-9)) for v in scores) / 3), 3) if all(v > 0 for v in scores) else 0.0)
        return ok, "; ".join(notes)

    c.check("timing is honest: no kernel beats the rated limit or runs 2x slower than eager", timing)
