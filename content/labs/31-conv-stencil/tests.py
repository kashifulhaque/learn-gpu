"""Tests for lab 31: convolution, stencils, and implicit GEMM.

The first eight checks run on any machine, in plain Python and torch-CPU. The
last two need a CUDA GPU and skip with a message without one.

Plausible wrong implementations that these checks catch:

- flipped_filter: conv2d_im2col that flips the filter, which is the textbook
  convolution. PyTorch computes cross-correlation, and an asymmetric filter
  shows the difference.
- negative_index_wraps: halo_tile or the implicit-GEMM tile that reads
  x[-1] at the top-left border. Python and torch wrap a negative index to the
  last row, so the halo holds real pixels, not zeros.
- tap_major_rows: im2col with rows ordered tap-major (i, j, c) instead of
  channel-major (c, i, j), which is a valid layout for NHWC and not the
  layout of F.unfold.
- roll_wraps_boundary: a stencil built from torch.roll, which wraps the
  boundary around the grid instead of leaving it unchanged.
- ceil_pool: a fused pool that keeps a partial window at an odd edge.
"""

from __future__ import annotations

from lab_common import A100, Checks

CPU_NAMES = (
    "conv_out_size", "halo_overhead", "active_fraction", "halo_tile", "im2col",
    "im2col_blowup", "conv2d_im2col", "implicit_gemm_coords", "implicit_gemm_tile",
    "conv2d_implicit", "stencil7_reference", "conv_relu_maxpool",
)
GPU_NAMES = ("cuda_source", "CPP_DECLARATIONS")

# Odd shapes: (N, C, H, W, K, R, S, stride, padding, dilation).
CONV_CASES = (
    (1, 1, 5, 5, 1, 3, 3, 1, 1, 1),
    (2, 3, 7, 9, 4, 3, 3, 1, 1, 1),
    (1, 2, 8, 6, 3, 3, 2, 2, 1, 1),
    (2, 3, 11, 10, 5, 5, 3, 2, 2, 1),
    (1, 2, 13, 9, 2, 3, 3, 1, 2, 2),
    (3, 1, 6, 7, 2, 1, 1, 1, 0, 1),
    (1, 4, 9, 9, 3, 2, 4, 3, 0, 1),
)


def _fail(want, got):
    return False, f"expected {want}, actual {got}"


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def run(submission):
    import torch
    import torch.nn.functional as F

    c = Checks()
    if not c.require(submission, *CPU_NAMES, *GPU_NAMES):
        return c.finish()
    s = submission
    torch.manual_seed(31)

    # 1. The first check needs only conv_out_size.
    def out_size():
        cases = (
            ((5, 3, 1, 0, 1), 3),
            ((5, 3, 1, 1, 1), 5),
            ((8, 3, 2, 1, 1), 4),
            ((9, 3, 1, 2, 2), 9),
            ((224, 7, 2, 3, 1), 112),
            ((10, 4, 3, 0, 1), 3),
        )
        for args, want in cases:
            got, err = _call(s.conv_out_size, *args)
            if err:
                return _fail(f"{want} for {args}", err)
            if got != want:
                return _fail(f"{want} for (size, kernel, stride, padding, dilation) = {args}", got)
            # Cross-check against torch's own shape.
            size, k, st, pd, dl = args
            ref = F.conv1d(torch.zeros(1, 1, size), torch.zeros(1, 1, k), stride=st, padding=pd, dilation=dl).shape[-1]
            if ref != want:
                return _fail(f"torch to give {want}", ref)
        return True, "expected 112 for a 224 input, 7 filter, stride 2, padding 3, actual 112"

    c.check("the output size follows stride, padding, and dilation", out_size)

    # 2. Halo arithmetic.
    def halo():
        cases = (
            (s.halo_overhead, (16, 3), 1.265625),
            (s.halo_overhead, (16, 5), 1.5625),
            (s.halo_overhead, (8, 7), (14 / 8) ** 2),
            (s.halo_overhead, (8, 3, 3), (10 / 8) ** 3),
            (s.active_fraction, (16, 3), (14 / 16) ** 2),
            (s.active_fraction, (32, 5), (28 / 32) ** 2),
            (s.active_fraction, (8, 9), 0.0),
        )
        for fn, args, want in cases:
            got, err = _call(fn, *args)
            if err:
                return _fail(f"{want:.6f} for {fn.__name__}{args}", err)
            if abs(got - want) > 1e-9:
                return _fail(f"{want:.6f} for {fn.__name__}{args}", f"{got:.6f}")
        return True, "expected 1.265625 loads per output for a 16 tile and a 3 filter, actual 1.265625"

    c.check("halo overhead and active threads follow the tile and filter sizes", halo)

    # 3. A tile at the image border. negative_index_wraps fails here.
    def tile():
        x = torch.arange(1.0, 21.0).reshape(4, 5)
        cases = (
            (-1, -1, 3, 3),
            (2, 3, 4, 4),
            (-2, 4, 3, 3),
            (3, -1, 3, 2),
            (1, 1, 2, 2),
            (10, 10, 2, 2),
        )
        for row0, col0, h, w in cases:
            want = torch.zeros(h, w)
            for i in range(h):
                for j in range(w):
                    r, cc = row0 + i, col0 + j
                    if 0 <= r < 4 and 0 <= cc < 5:
                        want[i, j] = x[r, cc]
            got, err = _call(s.halo_tile, x, row0, col0, h, w)
            if err:
                return _fail(f"{want.tolist()}", err)
            if tuple(got.shape) != (h, w) or not torch.equal(got, want):
                return _fail(f"{want.tolist()} for the tile at ({row0}, {col0})", got.tolist())
        return True, "expected zeros above and left of the image for the tile at (-1, -1), actual zeros"

    c.check("a halo tile at the image border reads zeros, not wrapped pixels", tile)

    # 4. im2col against F.unfold, and the blow-up.
    def lowering():
        for N, C, H, W, _K, R, S, st, pd, dl in CONV_CASES:
            x = torch.randn(N, C, H, W)
            want = F.unfold(x, (R, S), dilation=dl, padding=pd, stride=st)
            got, err = _call(s.im2col, x, R, S, st, pd, dl)
            if err:
                return _fail(f"shape {tuple(want.shape)}", err)
            if tuple(got.shape) != tuple(want.shape):
                return _fail(f"shape {tuple(want.shape)} for {(N, C, H, W)}, filter {(R, S)}", tuple(got.shape))
            if not torch.equal(got, want):
                bad = (got != want).nonzero()[0].tolist()
                return _fail(f"{want[tuple(bad)].item():.4f} at {bad}", f"{got[tuple(bad)].item():.4f}")
        cases = (
            (((1, 3, 56, 56), (3, 3), 1, 1, 1), 9.0),
            (((1, 3, 224, 224), (7, 7), 2, 3, 1), 12.25),
            (((8, 16, 32, 32), (1, 1), 1, 0, 1), 1.0),
        )
        for (shape, kernel, st, pd, dl), want in cases:
            got, err = _call(s.im2col_blowup, shape, kernel, st, pd, dl)
            if err:
                return _fail(f"{want} for {shape}, filter {kernel}", err)
            if abs(got - want) > 1e-9:
                return _fail(f"{want} for {shape}, filter {kernel}", got)
        return True, "expected 9.0 times the input for a 3 by 3 filter at stride 1, actual 9.0"

    c.check("im2col equals F.unfold, and a 3 by 3 filter blows the input up 9 times", lowering)

    # 5. Conv by im2col GEMM. flipped_filter fails here.
    def via_gemm():
        for N, C, H, W, K, R, S, st, pd, dl in CONV_CASES:
            x = torch.randn(N, C, H, W)
            w = torch.randn(K, C, R, S)
            b = torch.randn(K)
            want = F.conv2d(x, w, b, stride=st, padding=pd, dilation=dl)
            got, err = _call(s.conv2d_im2col, x, w, b, st, pd, dl)
            if err:
                return _fail(f"shape {tuple(want.shape)}", err)
            if tuple(got.shape) != tuple(want.shape):
                return _fail(f"shape {tuple(want.shape)}", tuple(got.shape))
            e = (got - want).abs().max().item()
            if e > 1e-4:
                return _fail(f"max error below 1e-4 for {(N, C, H, W, K, R, S, st, pd, dl)}", f"{e:.3e}")
        # A filter that is not symmetric shows the flip: one tap, top-left.
        x = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        w = torch.zeros(1, 1, 3, 3)
        w[0, 0, 0, 0] = 1.0
        got, err = _call(s.conv2d_im2col, x, w, None, 1, 1, 1)
        if err:
            return _fail("the tap at the top-left to read the pixel above and left", err)
        want = F.conv2d(x, w, padding=1)
        if not torch.equal(got, want):
            return _fail(f"{want.flatten().tolist()} (no flip)", got.flatten().tolist())
        return True, "expected F.conv2d to match on 7 odd shapes, and no filter flip, actual match"

    c.check("conv by im2col and one GEMM equals F.conv2d, with no filter flip", via_gemm)

    # 6. The implicit-GEMM index map, entry by entry.
    def index_map():
        for N, C, H, W, K, R, S, st, pd, dl in CONV_CASES:
            x = torch.randn(N, C, H, W)
            cols = F.unfold(x, (R, S), dilation=dl, padding=pd, stride=st)  # (N, CRS, PQ)
            P = (H + 2 * pd - dl * (R - 1) - 1) // st + 1
            Q = (W + 2 * pd - dl * (S - 1) - 1) // st + 1
            M, Kg = N * P * Q, C * R * S
            # The virtual A matrix is (M, Kg): row m = (n, p, q), column k.
            for layout in ("nchw", "nhwc"):
                for m in range(M):
                    n, pq = divmod(m, P * Q)
                    for k in range(Kg):
                        if layout == "nchw":
                            col = k
                        else:
                            rs, ch = divmod(k, C)
                            col = ch * R * S + rs
                        want_val = cols[n, col, pq].item()
                        got, err = _call(s.implicit_gemm_coords, m, k, (N, C, H, W), (R, S), st, pd, dl, layout)
                        if err:
                            return _fail(f"a coordinate or None for m={m}, k={k}", err)
                        if got is None:
                            val = 0.0
                        else:
                            n2, c2, h2, w2 = got
                            if not (0 <= h2 < H and 0 <= w2 < W and 0 <= c2 < C and 0 <= n2 < N):
                                return _fail("a coordinate inside the tensor, or None", got)
                            val = x[n2, c2, h2, w2].item()
                        if val != want_val:
                            return _fail(
                                f"{want_val:.4f} at m={m}, k={k} ({layout}, {(N, C, H, W)})",
                                f"{val:.4f} from {got}",
                            )
        # The tile gather and the tiled conv, with tiles that do not divide M or K.
        for N, C, H, W, K, R, S, st, pd, dl in CONV_CASES:
            x = torch.randn(N, C, H, W)
            w = torch.randn(K, C, R, S)
            want = F.conv2d(x, w, stride=st, padding=pd, dilation=dl)
            got, err = _call(s.conv2d_implicit, x, w, st, pd, dl, 7, 5)
            if err:
                return _fail(f"shape {tuple(want.shape)}", err)
            if tuple(got.shape) != tuple(want.shape) or (got - want).abs().max().item() > 1e-4:
                e = (got - want).abs().max().item() if tuple(got.shape) == tuple(want.shape) else float("nan")
                return _fail(f"F.conv2d within 1e-4 for {(N, C, H, W, K, R, S, st, pd, dl)}", f"error {e:.3e}")
        return True, "expected None for every padded entry and x[n, c, h, w] elsewhere, actual that"

    c.check("the implicit-GEMM index map matches im2col entry by entry, padding included", index_map)

    # 7. The 7-point stencil against a plain loop. roll_wraps_boundary fails here.
    def stencil():
        def loop(x, c0, c1):
            Z, Y, X = x.shape
            out = [[[x[z, y, xx].item() for xx in range(X)] for y in range(Y)] for z in range(Z)]
            for z in range(1, Z - 1):
                for y in range(1, Y - 1):
                    for xx in range(1, X - 1):
                        out[z][y][xx] = c0 * x[z, y, xx].item() + c1 * (
                            x[z - 1, y, xx].item() + x[z + 1, y, xx].item()
                            + x[z, y - 1, xx].item() + x[z, y + 1, xx].item()
                            + x[z, y, xx - 1].item() + x[z, y, xx + 1].item()
                        )
            return torch.tensor(out, dtype=torch.float64)

        for shape in ((5, 6, 7), (3, 3, 3), (2, 4, 4), (4, 1, 5), (6, 5, 3)):
            x = torch.randn(*shape, dtype=torch.float64)
            want = loop(x, 0.5, 1.0 / 12)
            got, err = _call(s.stencil7_reference, x, 0.5, 1.0 / 12)
            if err:
                return _fail(f"shape {shape}", err)
            if tuple(got.shape) != shape or (got - want).abs().max().item() > 1e-12:
                e = (got - want).abs().max().item() if tuple(got.shape) == shape else float("nan")
                return _fail(f"the loop result for shape {shape}", f"max error {e:.3e}")
        return True, "expected the loop result on 5 shapes, with the boundary copied, actual match"

    c.check("the 7-point stencil leaves the boundary alone and matches a plain loop", stencil)

    # 8. Fused conv + ReLU + maxpool. ceil_pool fails here.
    def fused():
        cases = (
            (1, 1, 6, 6, 2, 3, 1, 1, 2),
            (2, 3, 9, 9, 4, 3, 1, 1, 2),
            (2, 2, 11, 10, 3, 3, 2, 1, 2),
            (1, 3, 12, 12, 5, 5, 1, 2, 3),
        )
        for N, C, H, W, K, R, st, pd, pool in cases:
            x = torch.randn(N, C, H, W)
            w = torch.randn(K, C, R, R)
            b = torch.randn(K)
            want = F.max_pool2d(F.relu(F.conv2d(x, w, b, stride=st, padding=pd)), pool)
            got, err = _call(s.conv_relu_maxpool, x, w, b, st, pd, pool)
            if err:
                return _fail(f"shape {tuple(want.shape)}", err)
            if tuple(got.shape) != tuple(want.shape):
                return _fail(f"shape {tuple(want.shape)} for {(N, C, H, W)}, pool {pool}", tuple(got.shape))
            e = (got - want).abs().max().item()
            if e > 1e-4:
                return _fail(f"max error below 1e-4 for {(N, C, H, W, K, R, st, pd, pool)}", f"{e:.3e}")
        return True, "expected max_pool2d(relu(conv2d)) on odd conv outputs, actual match"

    c.check("the fused conv + ReLU + maxpool equals the unfused torch ops", fused)

    _gpu_section(c, s)
    return c.finish()


# ---------------------------------------------------------------------------
# GPU checks
# ---------------------------------------------------------------------------


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU is visible"
    return None


def _gpu_section(c: Checks, s) -> None:
    reason = _skip_reason()
    if reason is not None:
        msg = f"SKIP: the CUDA conv2d and stencil checks were not run, because {reason}."
        print(msg, flush=True)
        c.metric("conv_ms", "skipped (no GPU)")
        c.metric("stencil_coarse_gbs", "skipped (no GPU)")
        return

    import torch
    import torch.nn.functional as F

    from lab_common import bandwidth_gbs, close, cuda_time, load_cuda

    torch.manual_seed(31)
    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    try:
        module = load_cuda("lab31_conv_stencil", s.cuda_source(), s.CPP_DECLARATIONS, ["conv2d_tiled", "stencil7"])
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[-2000:])
        return
    print("      the CUDA source compiles", flush=True)

    def conv_gpu():
        # Filter sides 1 to 15, and shapes that are not multiples of the 16 tile.
        shapes = ((1, 1, 1, 3), (2, 37, 53, 5), (1, 16, 16, 3), (3, 100, 31, 7), (1, 65, 17, 15), (2, 33, 64, 1))
        for N, H, W, K in shapes:
            x = torch.randn(N, H, W, device="cuda")
            w = torch.randn(K, K, device="cuda")
            want = F.conv2d(x[:, None].double(), w[None, None].double(), padding=K // 2)[:, 0].float()
            got = module.conv2d_tiled(x, w)
            torch.cuda.synchronize()
            ok, detail = close(got, want, 1e-4)
            if not ok:
                return False, f"expected max error below 1e-4 for {(N, H, W)} with a {K}x{K} filter, actual {detail}"
        record = conv_time()
        return True, f"{len(shapes)} shapes, filter sides 1 to 15, against a float64 F.conv2d; {record}"

    def conv_time():
        x = torch.randn(8, 2048, 2048, device="cuda")
        w = torch.randn(5, 5, device="cuda")
        ms = cuda_time(lambda: module.conv2d_tiled(x, w), warmup=3, iters=20)
        ref = cuda_time(lambda: F.conv2d(x[:, None], w[None, None], padding=2), warmup=3, iters=20)
        c.metric("conv_ms", round(ms, 3))
        c.metric("torch_conv_ms", round(ref, 3))
        return f"recorded {ms:.3f} ms for the lab kernel and {ref:.3f} ms for F.conv2d, not graded"

    c.check("the tiled conv2d with a constant-memory filter equals F.conv2d", conv_gpu)

    def stencil_gpu():
        for shape in ((5, 6, 7), (17, 33, 65), (40, 31, 70), (2, 9, 9), (33, 8, 129)):
            x = torch.randn(*shape, device="cuda")
            want = s.stencil7_reference(x.cpu(), 0.5, 1.0 / 12).cuda()
            for variant in (0, 1):
                got = module.stencil7(x, 0.5, 1.0 / 12, variant)
                torch.cuda.synchronize()
                ok, detail = close(got, want, 1e-5)
                if not ok:
                    return False, f"expected the reference for shape {shape}, variant {variant}, actual {detail}"
        shape = (256, 256, 256)
        x = torch.randn(*shape, device="cuda")
        # Read once, write once: the same bytes as a copy.
        nbytes = 2 * x.numel() * 4
        res = {}
        for variant, name in ((0, "naive"), (1, "coarse")):
            ms = cuda_time(lambda v=variant: module.stencil7(x, 0.5, 1.0 / 12, v), warmup=3, iters=20, flush_l2=True)
            res[name] = bandwidth_gbs(nbytes, ms)
            c.metric(f"stencil_{name}_gbs", round(res[name], 1))
        frac = res["coarse"] / A100.COPY_CEILING_GBS
        c.metric("stencil_fraction_of_ceiling", round(frac, 3))
        floor = 0.30
        return frac >= floor, (
            f"expected at least {floor:.0%} of the {A100.COPY_CEILING_GBS:.0f} GB/s copy ceiling for the coarsened "
            f"kernel, actual {res['coarse']:.0f} GB/s ({frac:.0%}); the naive kernel ran at {res['naive']:.0f}"
        )

    c.check("both stencil kernels equal the reference, and the coarsened one reaches 30% of the copy ceiling", stencil_gpu)

    # Optional: an implicit-GEMM conv in Triton. It records a metric and never fails the lab.
    try:
        import triton  # noqa: F401
    except ImportError:
        c.metric("triton_max_err", "skipped (no triton)")
        return
    try:
        x = torch.randn(2, 3, 19, 21, device="cuda")
        w = torch.randn(5, 3, 3, 3, device="cuda")
        got = s.conv2d_triton(x, w, 2, 1)
        want = F.conv2d(x.double(), w.double(), stride=2, padding=1).float()
        err = (got - want).abs().max().item()
        # tl.dot defaults to TF32 on Ampere, so the tolerance is loose.
        c.metric("triton_max_err", round(err, 5))
        print(f"      optional Triton implicit GEMM: max error {err:.2e} (TF32 tolerance about 1e-2)", flush=True)
    except NotImplementedError:
        c.metric("triton_max_err", "skipped (not implemented)")
    except Exception as exc:  # noqa: BLE001
        c.metric("triton_max_err", f"failed: {type(exc).__name__}")
