# Labs

Each chapter except 23 has one lab, in `content/labs/<lab id>/`. The lab ids are
in [CURRICULUM.md](CURRICULUM.md). A lab runs on a RunPod A100 80GB (sm_80),
streams its output to the browser, and reports whether its checks passed and
what it measured. The reader edits `starter.py`; the worker runs `tests.py`
against the result.

## Layout

| File | Contents |
|---|---|
| `meta.json` | The id, title, brief, GPU, timeout, metric names, and hints the app shows. |
| `starter.py` | What the reader edits. Its docstrings and comments are the specification. |
| `tests.py` | `run(submission)`, which exercises the reader's module and returns a result. |
| `solution.py` | The worked solution. It must pass `tests.py`. |
| `fixture_*.py` | Optional. Copied beside the submission, so the reader's code and the tests can import it. |

### meta.json

```json
{
  "id": "03-bandwidth-probe",
  "title": "Bandwidth probe",
  "brief": "One or two sentences on what the reader builds.",
  "gpu": "A100-80GB",
  "needs_gpu": true,
  "timeout": 900,
  "metrics": ["copy_gbs", "vector_gbs"],
  "hints": ["Plain strings, shown one at a time."]
}
```

`id` equals the directory name. There is no `needs_weights`: no lab downloads
model weights. A lab that compiles CUDA needs 30 to 60 seconds for `nvcc` the
first time, so give it a `timeout` of at least 900 and tell the reader in a hint.
`metrics` lists the keys `tests.py` reports with `Checks.metric`.

## Writing `tests.py`

`tests.py` defines `run(submission)` and returns `Checks.finish()`:

```python
from lab_common import A100, Checks, bandwidth_gbs, close, cuda_time, load_cuda, require_cuda


def run(submission):
    c = Checks()
    if not c.require(submission, "copy_kernel_source"):
        return c.finish()
    device = require_cuda()
    ...
    c.check("the copy is correct", lambda: close(out, x, 0))
    c.metric("copy_gbs", round(gbs, 1))
    return c.finish()
```

The harness imports the reader's file as `submission`, then calls `run`. A crash
in the reader's code, such as an illegal memory access, kills only the lab's
subprocess, and the reader sees a clear message.

### The helpers in `lab_common`

Import them with `from lab_common import ...`. The worker puts
`content/labs/` on `PYTHONPATH`, so this works in `tests.py`, `starter.py`, and
`solution.py`.

| Name | Signature | Does |
|---|---|---|
| `Checks` | `Checks()` with `.add(name, passed, detail="")`, `.check(name, fn, detail="")`, `.metric(key, value)`, `.require(submission, *names)`, `.finish()` | Collects results and metrics, and prints `[PASS]` and `[FAIL]` lines. |
| `close` | `close(actual, expected, tol=1e-5) -> (bool, str)` | Compares tensors by maximum absolute error. |
| `require_cuda` | `require_cuda() -> torch.device` | Returns the CUDA device, or raises with a clear message. |
| `pick_device` | `pick_device() -> torch.device` | CUDA if present, else CPU, for tests that check mathematics. |
| `cuda_time` | `cuda_time(fn, warmup=10, iters=50, *, flush_l2=False) -> float` | Median milliseconds of one call to `fn`, measured with CUDA events. `flush_l2=True` starts every call with a cold L2. |
| `bandwidth_gbs` | `bandwidth_gbs(num_bytes, ms) -> float` | GB/s, decimal. Count every byte read and written. |
| `tflops` | `tflops(flops, ms) -> float` | TFLOP/s. |
| `load_cuda` | `load_cuda(name, cuda_src, cpp_src, functions, *, fast_math=False, extra_cuda_cflags=(), verbose=False)` | Compiles with `torch.utils.cpp_extension.load_inline`, once per distinct source. Flags: `-O3 -arch=sm_80`, plus `--use_fast_math` only on request. `cpp_src` declares each function; `cuda_src` defines them. |
| `A100` | constants object | See below. |

`A100` holds the figures every lab and chapter share, each commented with its
source in `lab_common.py`:

| Name | Value |
|---|---|
| `SM_COUNT` | 108 |
| `HBM_BW_GBS` | 1935.0 (rated, 80GB PCIe) |
| `COPY_CEILING_GBS` | 1275.0 (measured copy, learn-inference chapter 10) |
| `L2_BYTES`, `L2_MB` | 40 MiB |
| `SMEM_PER_SM_BYTES` | 164 KiB (163 KiB per block) |
| `TENSOR_BF16_TFLOPS`, `TENSOR_FP16_TFLOPS` | 312.0 dense |
| `TENSOR_TF32_TFLOPS` | 156.0 dense |
| `FP32_TFLOPS` | 19.5 |
| `BOOST_MHZ` | 1410 |
| `REGISTERS_PER_SM`, `MAX_THREADS_PER_SM`, `MAX_BLOCKS_PER_SM`, `WARP_SIZE`, ... | The compute capability 8.0 limits. |

If a lab needs a constant that isn't there, add it to `lab_common.py` with a
comment that names its source, in the same change as the lab that needs it.

### What a test asserts

- **Correctness is a hard check.** Compare against a PyTorch or float64
  reference with a tolerance justified by the data type, and test sizes that
  aren't multiples of the block size.
- **Performance is a metric, and a loose threshold.** Report the number with
  `Checks.metric`, and fail only when the kernel is far from what the roofline
  predicts. Set a threshold as a fraction of a *derived* ceiling, such as 60% of
  `A100.COPY_CEILING_GBS`, never as a figure you have not measured. The lab can
  also land on the SXM4 card, which is rated higher, so thresholds leave room.
- **Time honestly.** Use `cuda_time`, flush the L2 when the data fits in it and
  the lab is about DRAM, and count every byte read and written.
- **Compile once.** Call `load_cuda` for each source; it caches.

Because lab authors can't run the GPU either, write the tests defensively and
keep thresholds generous. [STYLE.md](../content/STYLE.md#accuracy) explains why.

## Checking a lab

```bash
.venv/bin/python scripts/check_labs.py            # every lab
.venv/bin/python scripts/check_labs.py 03-bandwidth-probe
```

Every lab gets a static check without a GPU: `meta.json` has the fields the app
reads, the three files parse, `tests.py` defines `run`, and every name imported
from `lab_common` exists. A lab marked `"device_agnostic": true` runs its tests
on a CPU too, for labs that check mathematics. Labs that need CUDA are otherwise
skipped without a GPU, and reported as skipped. To run them, use a GPU host with
the course's worker image, or open the lab in the app.

## The worker

`gpu/runpod_worker/` builds on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`: a
CUDA *devel* image with `nvcc`, plus `ninja`, a pinned `triton`, and the matching
torch. It sets `TORCH_CUDA_ARCH_LIST=8.0` and a persistent
`TORCH_EXTENSIONS_DIR`. A change to `lab_common.py` or any lab reaches the GPU
only when the image is rebuilt.
