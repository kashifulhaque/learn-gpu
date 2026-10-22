import torch
from lab_common import A100, Checks, close, cuda_time, require_cuda, tflops

# Odd sizes exercise the padding; the multiples of 128 exercise full tiles.
SHAPES = [(1, 1, 1), (16, 16, 16), (17, 33, 19), (100, 200, 300), (128, 128, 32), (257, 129, 65), (512, 512, 512), (1000, 1000, 1000)]
PERF_SHAPE = (4096, 4096, 4096)


def _reference(a, b):
    # fp32 matmul of the bf16 values. TF32 is off by default in PyTorch, so this
    # is an fp32 GEMM on the CUDA cores: every bf16 product is exact in fp32.
    return a.float() @ b.float()


def _tolerance(ref):
    # Tensor-core accumulation order differs from the reference's, so allow a
    # small relative error. A wrong fragment index is off by O(1) of the scale.
    return 1e-3 * max(1.0, ref.abs().max().item())


def run(submission):
    c = Checks()
    if not c.require(submission, "wmma_gemm", "mma_gemm", "percent_of_torch"):
        return c.finish()
    device = require_cuda()
    torch.manual_seed(0)

    c.check(
        "percent_of_torch divides and scales by 100",
        lambda: (
            abs(submission.percent_of_torch(150.0, 200.0) - 75.0) < 1e-6,
            f"got {submission.percent_of_torch(150.0, 200.0)}, expected 75.0",
        ),
    )

    for name in ("wmma_gemm", "mma_gemm"):
        fn = getattr(submission, name)
        for m, n, k in SHAPES:
            a = torch.randn(m, k, device=device).to(torch.bfloat16)
            b = torch.randn(k, n, device=device).to(torch.bfloat16)
            ref = _reference(a, b)

            def check(fn=fn, a=a, b=b, ref=ref, m=m, n=n):
                out = fn(a, b)
                if out.dtype != torch.float32:
                    return False, f"output dtype {out.dtype}, expected torch.float32"
                if tuple(out.shape) != (m, n):
                    return False, f"output shape {tuple(out.shape)}, expected {(m, n)}"
                return close(out, ref, _tolerance(ref))

            c.check(f"{name} is correct for M={m}, N={n}, K={k}", check)

    # Structured input catches a transposed or swapped fragment that random data can hide:
    # A is a row index ramp, B is the identity, so C must equal A.
    def identity_check(fn):
        a = (torch.arange(64, device=device).float().view(64, 1) + torch.arange(64, device=device).float().view(1, 64) / 64).to(torch.bfloat16)
        eye = torch.eye(64, device=device, dtype=torch.bfloat16)
        return close(fn(a, eye), a.float(), 0.0)

    c.check("wmma_gemm times the identity returns A", lambda: identity_check(submission.wmma_gemm))
    c.check("mma_gemm times the identity returns A", lambda: identity_check(submission.mma_gemm))

    # Performance. The ceiling is the dense bf16 peak; the floor for "using the
    # tensor cores at all" is the fp32 CUDA-core peak, 19.5 TFLOP/s.
    m, n, k = PERF_SHAPE
    a = torch.randn(m, k, device=device).to(torch.bfloat16)
    b = torch.randn(k, n, device=device).to(torch.bfloat16)
    flops = 2.0 * m * n * k

    torch_ms = cuda_time(lambda: torch.matmul(a, b), warmup=5, iters=20)
    torch_tf = tflops(flops, torch_ms)
    c.metric("torch_tflops", round(torch_tf, 1))

    for name, key in (("wmma_gemm", "wmma"), ("mma_gemm", "mma")):
        fn = getattr(submission, name)
        try:
            ms = cuda_time(lambda fn=fn: fn(a, b), warmup=5, iters=20)
        except Exception as exc:  # noqa: BLE001
            c.add(f"{name} runs at 4096 x 4096 x 4096", False, f"{type(exc).__name__}: {exc}")
            continue
        tf = tflops(flops, ms)
        c.metric(f"{key}_tflops", round(tf, 1))
        c.metric(f"{key}_pct_torch", round(submission.percent_of_torch(tf, torch_tf), 1))
        c.add(
            f"{name} beats the {A100.FP32_TFLOPS} TFLOP/s fp32 CUDA-core peak",
            tf > A100.FP32_TFLOPS,
            f"{tf:.1f} TFLOP/s; the dense bf16 tensor-core peak is {A100.TENSOR_BF16_TFLOPS:.0f}. "
            "A figure below 19.5 means the tensor cores are not doing the work.",
        )
    return c.finish()
