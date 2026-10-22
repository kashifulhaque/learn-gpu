import torch
from lab_common import A100, Checks, bandwidth_gbs, cuda_time, load_cuda, require_cuda

VARIANTS = {"naive": 0, "tiled": 1, "padded": 2, "swizzled": 3}
LAYOUTS = ("plain", "padded", "swizzled")

# Shapes that are not multiples of the tile (32 for float32, 64 for bfloat16),
# plus a single element, single rows and columns, and shapes one off a multiple.
SHAPES = [
    (1, 1), (1, 37), (37, 1), (31, 33), (32, 32), (33, 31), (64, 64), (65, 129),
    (100, 300), (257, 3), (1000, 1000), (1023, 2049), (2048, 64),
]

# The performance shape. 8192 x 8192 float32 is 268 MB each way, far above the
# 40 MiB L2, so no cache flush is needed. A transpose reads every element once
# and writes it once, so it moves 2 * M * N * element_bytes, the same as a copy.
PERF_SHAPE = (8192, 8192)
RAGGED_SHAPE = (8191, 8193)

# Thresholds are fractions of the measured copy ceiling (1275 GB/s on the lab
# A100), and they are loose: a good kernel reaches most of the ceiling.
MIN_FRACTION_FP32 = 0.50
MIN_FRACTION_BF16 = 0.40


def _model_checks(c, s):
    c.check(
        "bank: 32 banks of 4 bytes",
        lambda: (
            [s.bank(a) for a in (0, 3, 4, 124, 128, 132)] == [0, 0, 1, 31, 0, 1],
            f"banks of 0, 3, 4, 124, 128, 132: {[s.bank(a) for a in (0, 3, 4, 124, 128, 132)]}",
        ),
    )
    cases = [
        ("32 consecutive floats", [4 * i for i in range(32)], 1),
        ("32 consecutive bfloat16", [2 * i for i in range(32)], 1),
        ("all lanes read one word (broadcast)", [64] * 32, 1),
        ("stride of 2 floats", [8 * i for i in range(32)], 2),
        ("stride of 32 floats", [128 * i for i in range(32)], 32),
        ("two words in one bank, one broadcast", [0] * 31 + [128], 2),
    ]
    for name, addresses, expected in cases:
        c.check(
            f"conflict_degree: {name} is {expected}",
            lambda a=addresses, e=expected: (s.conflict_degree(a) == e, f"got {s.conflict_degree(a)}"),
        )

    for eb in (4, 2):
        t = s.tile_dim(eb)
        for layout in LAYOUTS:

            def injective(layout=layout, eb=eb, t=t):
                seen = {s.smem_index(layout, r, col, eb) for r in range(t) for col in range(t)}
                limit = s.tile_bytes(layout, eb) // eb
                ok = len(seen) == t * t and min(seen) >= 0 and max(seen) < limit
                return ok, f"{len(seen)} distinct indices, largest {max(seen)}, array holds {limit}"

            c.check(f"smem_index is one-to-one inside the array ({layout}, {eb}-byte)", injective)

    c.check(
        "tile_bytes: 4096 plain, 4224 padded (float32); 8192 and 8448 (bfloat16)",
        lambda: (
            [s.tile_bytes("plain", 4), s.tile_bytes("padded", 4), s.tile_bytes("plain", 2), s.tile_bytes("padded", 2)]
            == [4096, 4224, 8192, 8448],
            f"got {[s.tile_bytes('plain', 4), s.tile_bytes('padded', 4), s.tile_bytes('plain', 2), s.tile_bytes('padded', 2)]}",
        ),
    )

    expected = {"plain": 32, "padded": 1, "swizzled": 1}
    for eb in (4, 2):
        t = s.tile_dim(eb)
        row_starts = (0,) if eb == 4 else (0, 32)
        for layout in LAYOUTS:

            def worst(layout=layout, eb=eb, t=t, row_starts=row_starts):
                degrees = [s.column_read_degree(layout, eb, col, r0) for col in range(t) for r0 in row_starts]
                return max(degrees)

            c.check(
                f"column read is {expected[layout]}-way ({layout}, {eb}-byte)",
                lambda w=worst: (w() == expected[layout], f"worst case over columns: {w()}-way"),
            )


def _correctness(c, module, dtype, name):
    for vname, v in VARIANTS.items():
        def run_all(v=v):
            for m, n in SHAPES:
                x = torch.randn(m, n, device="cuda").to(dtype)
                keep = x.clone()
                out = module.transpose(x, v)
                torch.cuda.synchronize()
                if tuple(out.shape) != (n, m) or not out.is_contiguous():
                    return False, f"shape {tuple(out.shape)} for input {(m, n)}"
                want = keep.t().contiguous()
                # A transpose only moves bits, so the match must be exact.
                if not torch.equal(out.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                                   want.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)):
                    bad = (out != want).sum().item()
                    return False, f"{bad} wrong elements for shape {(m, n)}"
                if not torch.equal(x, keep):
                    return False, f"the input changed for shape {(m, n)}"
            return True, f"{len(SHAPES)} shapes"

        c.check(f"{vname} transpose is exact for {name}", run_all)


def _gbs(fn, x):
    ms = cuda_time(fn, warmup=3, iters=20)
    return bandwidth_gbs(2 * x.numel() * x.element_size(), ms)


def run(submission):
    c = Checks()
    needed = ("bank", "conflict_degree", "smem_index", "tile_bytes", "column_read_degree",
              "tile_dim", "cuda_source", "CPP_DECLARATIONS")
    if not c.require(submission, *needed):
        return c.finish()

    print("Part 1: the bank model (no GPU needed)", flush=True)
    try:
        _model_checks(c, submission)
    except NotImplementedError:
        c.add("the bank model is implemented", False, "a function still raises NotImplementedError")
        return c.finish()

    print("Part 2: the CUDA kernels", flush=True)
    require_cuda()
    props = torch.cuda.get_device_properties(0)
    if props.multi_processor_count != A100.SM_COUNT:
        print(f"      note: this is not the course's A100 ({A100.SM_COUNT} SMs); thresholds assume one.", flush=True)
    torch.manual_seed(0)

    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    try:
        module = load_cuda("lab08_transpose", submission.cuda_source(), submission.CPP_DECLARATIONS, ["transpose"])
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[-2000:])
        return c.finish()
    c.add("the CUDA source compiles", True)

    _correctness(c, module, torch.float32, "float32")
    _correctness(c, module, torch.bfloat16, "bfloat16")
    if not all(r["passed"] for r in c.results):
        print("Skipping the timings until every correctness check passes.", flush=True)
        return c.finish()

    print(f"Timing at {PERF_SHAPE[0]} x {PERF_SHAPE[1]}", flush=True)
    results = {}
    for dtype, tag, floor in ((torch.float32, "fp32", MIN_FRACTION_FP32), (torch.bfloat16, "bf16", MIN_FRACTION_BF16)):
        x = torch.randn(*PERF_SHAPE, device="cuda").to(dtype)
        dst = torch.empty_like(x)
        copy = _gbs(lambda: dst.copy_(x), x)
        c.metric(f"{tag}_copy_gbs", round(copy, 1))
        for vname, v in VARIANTS.items():
            gbs = _gbs(lambda v=v: module.transpose(x, v), x)
            results[(tag, vname)] = gbs
            c.metric(f"{tag}_{vname}_gbs", round(gbs, 1))
        for vname in ("padded", "swizzled"):
            gbs = results[(tag, vname)]
            c.check(
                f"{vname} {tag} transpose reaches {floor:.0%} of the copy ceiling",
                lambda g=gbs, f=floor: (
                    g >= f * A100.COPY_CEILING_GBS,
                    f"{g:.0f} GB/s is {g / A100.COPY_CEILING_GBS:.0%} of {A100.COPY_CEILING_GBS:.0f}; "
                    f"the copy ran at {copy:.0f}",
                ),
            )
        del x, dst

    x = torch.randn(*RAGGED_SHAPE, device="cuda")
    ragged = _gbs(lambda: module.transpose(x, 2), x)
    c.metric("fp32_ragged_padded_gbs", round(ragged, 1))
    c.check(
        "a shape that is not a multiple of the tile stays within a factor of two of the aligned one",
        lambda: (
            ragged >= 0.5 * results[("fp32", "padded")],
            f"{ragged:.0f} GB/s against {results[('fp32', 'padded')]:.0f} GB/s at {PERF_SHAPE}",
        ),
    )

    best = max(results[("fp32", "padded")], results[("fp32", "swizzled")])
    c.metric("best_fraction_of_ceiling", round(best / A100.COPY_CEILING_GBS, 3))
    naive, tiled = results[("fp32", "naive")], results[("fp32", "tiled")]
    print(f"      float32: padded is {results[('fp32', 'padded')] / tiled:.2f}x the unpadded tile and "
          f"{results[('fp32', 'padded')] / naive:.2f}x the naive kernel", flush=True)
    return c.finish()
