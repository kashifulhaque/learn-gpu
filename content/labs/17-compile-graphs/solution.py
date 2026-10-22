"""Lab 17 solution: fusion, torch.compile, CUDA graphs, and custom ops."""

import torch
import triton
import triton.language as tl
from torch.library import custom_op, triton_op, wrap_triton

BLOCK = 1024

# ---------------------------------------------------------------------------
# Part 1: a chain of six pointwise ops
# ---------------------------------------------------------------------------


def chain_eager(x: torch.Tensor) -> torch.Tensor:
    """Six pointwise ops. Eager PyTorch launches one kernel for each."""
    t = x * 1.5
    t = t + 0.5
    t = torch.sin(t)
    t = t * t
    t = t - 0.25
    return torch.relu(t)


@triton.jit
def _chain_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = x * 1.5
    t = t + 0.5
    t = tl.sin(t)
    t = t * t
    t = t - 0.25
    t = tl.maximum(t, 0.0)
    tl.store(y_ptr + offs, t, mask=mask)


def fused_chain(x: torch.Tensor) -> torch.Tensor:
    """The same six ops in one Triton kernel: one read and one write per element."""
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    grid = (triton.cdiv(n, BLOCK),)
    _chain_kernel[grid](x, y, n, BLOCK=BLOCK)
    return y


def compile_chain():
    """Return chain_eager compiled by torch.compile, with no graph breaks allowed."""
    return torch.compile(chain_eager, fullgraph=True)


def inductor_code(fn, x: torch.Tensor) -> str:
    """Compile `fn`, run it once on `x`, and return the Python source Inductor wrote."""
    import torch._dynamo as dynamo
    from torch._inductor import config as inductor_config
    from torch._inductor.utils import run_and_get_code

    dynamo.reset()
    with inductor_config.patch({"fx_graph_cache": False}):
        compiled = torch.compile(fn, fullgraph=True)
        _, sources = run_and_get_code(compiled, x)
    return "\n".join(sources)


# ---------------------------------------------------------------------------
# Part 2: a decode-like loop of many small kernels, replayed from a CUDA graph
# ---------------------------------------------------------------------------


def decode_step(h, weights, biases):
    """One decode-like step: a small matmul and a tanh per layer, two kernels each."""
    for w, b in zip(weights, biases):
        h = torch.tanh(torch.addmm(b, h, w))
    return h


class GraphedDecode:
    """Captures decode_step in a CUDA graph and replays it."""

    def __init__(self, weights, biases, example_h):
        self.static_h = example_h.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                decode_step(self.static_h, weights, biases)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = decode_step(self.static_h, weights, biases)

    def __call__(self, h):
        self.static_h.copy_(h)
        self.graph.replay()
        return self.static_out


# ---------------------------------------------------------------------------
# Part 3: the kernel as a custom op the compiler can trace around or through
# ---------------------------------------------------------------------------


@custom_op("learn_gpu17::chain_opaque", mutates_args=())
def chain_opaque(x: torch.Tensor) -> torch.Tensor:
    """The compiler sees a black box with this signature."""
    return fused_chain(x)


@chain_opaque.register_fake
def _chain_opaque_fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@triton_op("learn_gpu17::chain_traceable", mutates_args={})
def chain_traceable(x: torch.Tensor) -> torch.Tensor:
    """The compiler traces through wrap_triton and sees the kernel launch."""
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n > 0:

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK"]),)

        wrap_triton(_chain_kernel)[grid](x, y, n, BLOCK=BLOCK)
    return y
