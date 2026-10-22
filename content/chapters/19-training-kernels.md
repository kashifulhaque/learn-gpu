---
title: Training kernels
slug: 19-training-kernels
part: "Part 4 — Kernels for ML"
summary: Why training is limited by memory rather than arithmetic, how to write a custom backward with torch.autograd.Function, and how a chunked, fused linear cross-entropy removes the largest tensor in a language-model step.
minutes: 150
gpu: true
objectives:
  - "Account for the memory a training step holds: parameters, gradients, optimizer state, and the activations a transformer layer saves for the backward pass."
  - "Explain recomputation (activation checkpointing) as a trade of arithmetic for bytes, and say what it costs."
  - "Write a torch.autograd.Function with save_for_backward and once_differentiable, and verify it with gradcheck in float64."
  - "Size the logits tensor of a large-vocabulary model and explain why it dominates the step's peak memory."
  - "Derive the gradient of cross-entropy through a linear layer, and compute it chunk by chunk with an online logsumexp so the logits never exist at full size."
  - "Place Liger Kernel, Cut Cross-Entropy, and torch.nn.functional.linear_cross_entropy relative to the lab's hand-written version."
  - "Explain what fused optimizers, fp32 gradient accumulation, loss scaling, and bf16 each buy."
lab: 19-fused-xent
---

# Training kernels

> [!TLDR]
> - Training does three times the arithmetic of inference per layer and keeps the
>   forward pass's intermediate tensors alive until the backward pass uses them.
>   The binding limit is usually memory, not FLOP/s.
> - Memory has three parts: parameters, gradients, and optimizer state (16 bytes
>   per parameter for mixed-precision Adam), and activations (544 MiB per layer
>   for a 4096-token, 4096-wide GPT-style layer with fused attention).
> - A custom backward is a `torch.autograd.Function`: you save what the backward
>   needs, return one gradient per input, and check it with `gradcheck` in
>   float64.
> - For a 151,936-token vocabulary, 8192 tokens of logits are 4.64 GiB in
>   float32. A fused, chunked linear cross-entropy computes loss and gradient one
>   chunk at a time, so the full logits never exist.
> - Fused optimizers, fp32 accumulation, and loss scaling are the same lesson
>   again: fewer passes over memory, and numbers wide enough to survive the sum.

Every kernel you have written so far ran on the way forward. Training adds a
second pass in the opposite direction, and that pass changes what a kernel is
for. A backward kernel reads tensors the forward pass wrote minutes of
wall-clock earlier, which means the forward pass has to keep them, which means
the model's activations compete with its weights for the same 80 GB.

This chapter is about that competition. You first count the bytes a training
step holds, then learn the two tools that trade arithmetic for bytes:
recomputation and fusion. The running example is the one place where the count
goes most wrong, the output layer of a language model, where a tensor of
tokens-by-vocabulary logits is bigger than everything else in the step. The lab
builds the kernel that removes it, which is what
`torch.nn.functional.linear_cross_entropy` does in PyTorch 2.14.

## Before you start

**The autograd graph.** PyTorch records each operation on tensors that require
gradients, along with whatever that operation will need later. `loss.backward()`
walks that record in reverse and calls each operation's backward function once.

**Online softmax from chapter 15.** A softmax over a long row can be computed in
one pass by keeping a running maximum and a running sum that you rescale when
the maximum grows. [Chapter 15](/c/15-softmax-and-normalization) derives it, and
this chapter uses it on rows of 151,936 values.

**Number formats from chapter 9.** bfloat16 has an 8-bit exponent like float32
and 7 stored mantissa bits. float16 has a 5-bit exponent and 10 mantissa bits.
[Chapter 9](/c/09-floating-point-on-the-gpu) covers rounding and accumulation
error, and this chapter's last section depends on both.

**The roofline from chapter 4.** An elementwise pass over a large tensor is
limited by bandwidth: the A100 80GB PCIe is rated at 1935 GB/s and measured at
1275 GB/s for a copy. [Chapter 4](/c/04-roofline-and-latency) draws the roofline.

## What changes when you train

Training runs each layer's matrix multiplication three times where inference
runs it once, and keeps more of what it computes. This section answers what the
extra work is and why it shifts the limit from arithmetic to memory.

Take one linear layer, $Y = X W^{\top}$, with $X$ of shape $[M, K]$ and $W$ of
shape $[N, K]$. The forward pass is one GEMM of $2MNK$ FLOP. The backward pass
must produce two gradients from the incoming $\partial L / \partial Y$:

$$
\hla{\frac{\partial L}{\partial X}} = \frac{\partial L}{\partial Y}\, W
\qquad
\hlb{\frac{\partial L}{\partial W}} = \left(\frac{\partial L}{\partial Y}\right)^{\!\top} X
$$

Each is a GEMM of the same $2MNK$ FLOP. So a training step costs three GEMMs
per linear layer, three times the forward cost, and chapters 10 to 14's kernels
run three times as often.

The second change matters more. The weight gradient $\hlb{\partial L / \partial W}$
needs $X$, the layer's input, and the input gradient needs $W$. The forward pass
produced $X$ long before the backward pass wants it, so autograd *saves* it. Every
layer saves its inputs, and every nonlinearity saves something too, so the
saved tensors pile up for the whole forward pass and are released only as the
backward pass walks back through the layers.

> [!KEY] The peak is at the turn
> Memory in a training step rises through the forward pass, peaks where the
> forward pass ends and the backward pass begins, and falls as the backward
> pass frees what it consumes. Everything the backward pass will need is alive
> at the peak.

==The arithmetic fits. The saved tensors don't.== This is why a model that serves comfortably on one GPU can fail to train on the
same GPU with the same batch.

## Where the memory goes

A training step holds four kinds of memory, and the first three are fixed by the
model's size. The section answers how many bytes each parameter costs, and how
many bytes the activations of one layer cost.

### Parameters, gradients, and optimizer state

Mixed-precision training with Adam keeps, for every parameter:

| Item | Bytes | Why |
|---|---|---|
| Weight, bf16 | 2 | Used by the forward and backward GEMMs |
| Gradient, bf16 | 2 | Produced by the backward pass |
| Master weight, fp32 | 4 | The weight the optimizer updates, so small updates aren't rounded away |
| Adam first moment $m$, fp32 | 4 | A running mean of the gradient |
| Adam second moment $v$, fp32 | 4 | A running mean of the squared gradient |
| **Total** | **16** | |

The 16 bytes per parameter is the figure the [ZeRO paper](https://arxiv.org/abs/1910.02054)
uses for this recipe. A 1 billion parameter model holds 16 GB, and a 7 billion
parameter model holds 112 GB, which is already more than one 80 GB card before
a single activation exists. That arithmetic is why chapter 22 shards the state
across GPUs. On one card, it limits you to at most 5 billion parameters
($80 / 16$), fewer once the activations take their share.

### Activations of one transformer layer

The activations are the part you control with kernels. The saved tensors of a
GPT-style layer can be counted exactly, and the count is the one Korthikanti et
al. give in [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198).
Write $s$ for the sequence length, $b$ for the batch size, $h$ for the hidden
size, and $a$ for the number of attention heads. Each tensor below is stored in
a 16-bit format, 2 bytes per element, and a dropout mask costs 1 byte per element.

| Saved tensor | Elements | Bytes |
|---|---|---|
| Input to the first LayerNorm | $sbh$ | $2sbh$ |
| Input to the QKV projection (the LayerNorm output) | $sbh$ | $2sbh$ |
| $Q$ and $K$, for the score matrix | $2sbh$ | $4sbh$ |
| $V$ | $sbh$ | $2sbh$ |
| Attention scores after softmax | $as^{2}b$ | $2as^{2}b$ |
| Dropout mask on the scores | $as^{2}b$ | $as^{2}b$ |
| Dropout output, input to the $PV$ product | $as^{2}b$ | $2as^{2}b$ |
| Input to the attention output projection | $sbh$ | $2sbh$ |
| Dropout mask after attention | $sbh$ | $sbh$ |
| Input to the second LayerNorm | $sbh$ | $2sbh$ |
| Input to the first MLP linear | $sbh$ | $2sbh$ |
| Input to the GeLU, width $4h$ | $4sbh$ | $8sbh$ |
| Input to the second MLP linear, width $4h$ | $4sbh$ | $8sbh$ |
| Dropout mask after the MLP | $sbh$ | $sbh$ |

The terms that scale with $sbh$ add to $34sbh$ bytes, and the three that scale
with the score matrix add to $5as^{2}b$ bytes. So one layer saves:

$$
\boxed{\text{bytes per layer} = \hla{34\,sbh} + \hlb{5\,a\,s^{2}\,b}}
$$

The first term is linear in the sequence length. The second is quadratic, and it
is the score matrix that [chapter 16](/c/16-flash-attention) removes: FlashAttention
never writes the scores, so with it a layer saves only the $34sbh$ term (plus a
per-row logsumexp, which is $s$ floats per head and negligible).

> [!EXAMPLE] One layer of a 4096-wide model at 4096 tokens
> Take $s = 4096$, $b = 1$, $h = 4096$, $a = 32$. Then $sbh = 16{,}777{,}216$.
> The linear term is $34 \times 16{,}777{,}216 = 570{,}425{,}344$ bytes, or
> 544 MiB. The score-matrix term is $5 \times 32 \times 4096^{2} = 2{,}684{,}354{,}560$
> bytes, or 2560 MiB. Without fused attention a layer saves 3104 MiB (3.03 GiB);
> with it, 544 MiB. Across 32 layers that is 97 GiB without FlashAttention,
> which doesn't fit on any card in this course, and 17 GiB with it. A batch of 4
> sequences multiplies the 17 GiB to 68 GiB.

Two cautions apply. The constant 34 is for a GPT-style layer with a 4h MLP,
GeLU, and dropout; a Llama-style layer with SwiGLU and no dropout has a
different constant, and you should count your own layer the same way, tensor by
tensor. And the number is the saved set. Transient buffers inside a kernel come
on top, and the allocator's fragmentation comes on top of those.

## Recomputation: arithmetic for bytes

Recomputation, also called activation checkpointing, trades the activations of a
layer for a second forward pass through it. The section answers what you save
and what you pay.

The idea comes from [Chen et al., Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).
Run the forward pass without saving a layer's internals, keep only the layer's
input, and when the backward pass reaches that layer, run its forward again to
rebuild the internals, then differentiate. In PyTorch it is one wrapper:

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return x
```

With a checkpoint at every layer, the forward pass saves only each layer's input,
which is $2sbh$ bytes, instead of $34sbh$. For the example above that is 32 MiB
per layer instead of 544 MiB, so 32 layers hold 1 GiB instead of 17 GiB. During
the backward pass one layer at a time is rebuilt, so the peak adds one layer's
$34sbh$: 1.53 GiB in total, about one eleventh of 17 GiB.

The price is one extra forward pass. A step is one unit of forward and two of
backward (three GEMMs against one), and recomputation adds one more forward
unit, so the step costs $4/3$ of the original arithmetic, a 33% increase.
Korthikanti et al. show you can pay much less: *selective* recomputation
throws away only the cheap-to-recompute, expensive-to-store tensors (the
attention scores, in their setting) and keeps the rest.

> [!NOTE] FlashAttention is recomputation inside a kernel
> FlashAttention's backward pass rebuilds the score tiles from $Q$, $K$ and the
> saved logsumexp instead of reading a stored $s \times s$ matrix. It's selective
> recomputation done in SRAM, and it's cheaper than the framework-level
> version because the recomputed tiles never touch HBM.

You can now see the shape of the rest of the chapter. Recomputation is one way to
spend FLOPs to save bytes. The fused loss in this chapter is another, and it
needs no extra forward pass at all, because it finishes the backward of the loss
inside the forward.

## Writing a custom backward

A `torch.autograd.Function` is how you give autograd a forward and a backward you
wrote yourself, and the lab's fused loss is one. This section answers what the
two methods receive and return, and how to test them.

### The anatomy of a Function

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # tensors the backward will need
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_out           # one gradient per input of forward

y = Square.apply(x)
```

The rules are short and each one has a failure that is silent or confusing:

- **`forward` runs with gradient recording off.** Operations inside it aren't
  recorded, so the whole body is, to autograd, one opaque operation. You choose
  what to save.
- **Save tensors with `ctx.save_for_backward`, and everything else as
  attributes.** `save_for_backward` lets autograd detect that you modified a
  saved tensor in place after saving it, and raise an error at backward time
  instead of returning a wrong gradient. Non-tensor values, such as a chunk size
  or a dtype, go on `ctx` directly.
- **`backward` returns one value per argument of `forward`**, in order. An
  argument with no gradient, such as an integer target or a chunk size, gets
  `None`. A tensor argument that doesn't need a gradient can also get `None`;
  `ctx.needs_input_grad` tells you which ones do.
- **`backward` receives one gradient per output of `forward`**, and for a scalar
  loss it is a scalar tensor, usually 1. Your backward must scale by it, even
  though it is usually 1, because the loss may be multiplied by something
  before it reaches `backward()`.
- **`@once_differentiable`** on `backward` says the backward is not itself
  differentiable. Autograd then raises an error if someone asks for a second
  derivative, instead of silently returning a wrong one. Use it whenever your
  backward calls operations whose own gradients you haven't thought about.

PyTorch also supports a second style, in which `forward` takes no `ctx` and a
separate `setup_context(ctx, inputs, output)` saves what the backward needs. It
is the style that works with `torch.func` transforms, and it is what you use
when you register an autograd formula for a `torch.library` custom op, as
[chapter 17](/c/17-fusion-and-compilers) does.

### Checking a backward you wrote

A wrong gradient doesn't crash. It trains badly, slowly, and without an error, so
you test every custom backward numerically before you trust it.
`torch.autograd.gradcheck` compares your analytical gradient with finite
differences:

```python
h = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
t = torch.tensor([1, 6, -100, 0, 3])
torch.autograd.gradcheck(lambda a, b: fused_linear_xent(a, b, t, chunk_size=2),
                         (h, w), eps=1e-6, atol=1e-6)
```

The inputs are float64 because a finite difference subtracts two nearly equal
numbers, and in float32 the subtraction itself leaves only a few correct digits.
The shapes are tiny because `gradcheck` runs the function once per input
element. The lab's tests run exactly this check on your Function, with an
ignored token and a chunk that doesn't divide the batch.

## The logits problem

The output layer of a language model multiplies each token's hidden state by a
matrix with one row per vocabulary entry, and the product is the largest tensor
in the step. This section answers how large, and what holds it.

The final hidden states $H$ have shape $[N, d]$ for $N$ tokens. The output weight
$W$ has shape $[V, d]$ for a vocabulary of $V$ entries, and the logits are
$Z = H W^{\top}$, of shape $[N, V]$. The loss is the cross-entropy of
$\mathrm{softmax}(Z)$ against the target token of each row.

Vocabularies have grown. Llama 3 uses 128,256 entries, Qwen2.5 uses 151,936, and
Gemma 2 uses 256,000. Take a step of $N = 8192$ tokens, say two sequences of
4096, and the Qwen2.5 vocabulary:

$$
\hla{N} \times \hlb{V} \times 4\ \text{bytes} = 8192 \times 151{,}936 \times 4 = 4{,}978{,}638{,}848\ \text{bytes} = 4.64\ \text{GiB}
$$

That is the logits in float32, the dtype the loss computation wants. In bfloat16
they are 2.32 GiB. The hidden state they came from is $8192 \times 4096 \times 2 =
64$ MiB at $d = 4096$, so the logits are 74 times larger than the activations
they were computed from.

```viz
19-logits-memory
```

The unfused path then makes copies. A typical PyTorch implementation computes
the logits in bf16, casts them to float32 for the loss, and runs a log-softmax
that saves its output for the backward pass. During the forward pass the three
tensors are alive together. In the lab's baseline they are 2.32, 4.64, and 4.64
GiB at this size, and the backward pass allocates more of the same shape for the
gradient. Autograd keeps one of them, the $N \times V$ softmax, until the
backward pass consumes it.

> [!WARNING] The logits grow with the batch and the vocabulary, not the model
> Doubling the tokens per step doubles the logits and nothing else in the
> output layer's cost. A step that fits at 8192 tokens can fail at 32,768 even
> if every transformer layer is checkpointed, because the logits alone are
> 18.5 GiB in float32. Recomputing layers doesn't touch them.

The tensor is also a poor use of bandwidth. Each of its elements is written
once by the GEMM, read and written by the cast, the softmax, and the gradient,
and finally read by two more GEMMs. Chapter 13 of the companion course
[showed](https://qwen.ifkash.dev/c/13-fusion-in-triton) what a pass over a
large tensor costs: you pay for every trip to HBM. This tensor makes many
trips, and the worked example below counts them: 22.5 GB of traffic for the
unfused path.

## Fusing the head: chunk, then finish the backward

The idea is to never build all $N$ rows of the logits at once. This section
answers how the loss and both gradients can be computed from one chunk of rows
at a time, and what that costs.

### The gradient of the loss

Start with one row. The logit vector is $z = W h$ for one hidden state $h$ and a
target $y$. The loss is

$$
\ell = \log \sum_{j} e^{z_j} - z_y = \operatorname{lse}(z) - z_y
$$

where $\operatorname{lse}$ is the log of the sum of exponentials. Differentiate
with respect to the logit $z_j$. The first term gives the softmax probability
$p_j$, and the second gives $-1$ at $j = y$ and zero elsewhere:

$$
\boxed{\frac{\partial \ell}{\partial z_j} = \hla{p_j} - \hlb{[j = y]}, \qquad p_j = e^{z_j - \operatorname{lse}(z)}}
$$

In words: the gradient on the logits is the predicted distribution minus a
one-hot vector at the target. For a mean over $n$ valid tokens, multiply every
row's gradient by $1/n$. Call the whole $[N, V]$ gradient matrix $G$. The two
gradients you need are then exactly the GEMMs from the start of the chapter:

$$
\frac{\partial L}{\partial H} = G\, W
\qquad
\frac{\partial L}{\partial W} = G^{\top} H
$$

### Why the backward can run inside the forward

The loss is ==a scalar at the end of the graph==. Its upstream gradient is a single
number $g$, normally 1, and $g$ multiplies every element of $G$ equally. So
$G$ depends on the logits and the targets only, and you can compute it, and the
two GEMMs that consume it, before the backward pass starts. Then `backward`
scales the precomputed gradients by $g$ and returns them.

> [!KEY] A scalar loss lets you finish the backward early
> The gradient of the logits depends on nothing downstream except a scalar
> scale. Computing $G$, $G W$, and $G^{\top} H$ in the forward pass means the
> logits can be freed as soon as each chunk is done, and the backward pass needs
> only two saved tensors: the gradient of the hidden states and the gradient of
> the weight.

The trade is that the work happens whether or not a backward pass follows. A
forward-only evaluation of a model with this loss would waste two of the three
GEMMs, so the production versions let you turn the early gradient off.
It also limits the technique to a scalar loss, `mean` or `sum`. With
`reduction="none"` the upstream gradient is a per-token vector you don't have
yet, and the chunked path in PyTorch documents that case separately.

### The chunk loop

Process the rows in chunks of $c$ tokens. For each chunk the loop does five
things, and then lets go of the logits:

```python
for start in range(0, N, c):
    h_c = h[start:start + c]
    logits = h_c @ w.t()                              # [c, V]: the only logits that exist
    loss += row_loss_and_grad_(logits, t_c, scale).sum()   # logits now holds G for this chunk
    grad_h[start:start + c] = logits @ w              # G_c W
    grad_w += logits.t() @ h_c                        # G_c^T H_c, summed over chunks
```

The figure steps through the loop. Only the highlighted block of the logits ever
has storage.

```viz
19-chunked-xent
```

The loss is the same sum in a different association, so it doesn't depend on the
chunk size, and the lab checks that with chunk sizes of 1, 7, 64, and the batch.

### What the chunk size costs

The memory is easy. One chunk of float32 logits is $c \times V \times 4$ bytes: 1024
rows of the Qwen2.5 vocabulary is 593.5 MiB, against 4.64 GiB for all 8192
rows, an 8x reduction because there are 8 chunks. Do the arithmetic in bfloat16
in place and it halves again to 296.75 MiB.

The arithmetic is unchanged. Each chunk does three GEMMs of $2cdV$ FLOP, and the
chunks together do the same $6NdV$ as the unfused path. The fused version does
not recompute anything.

The cost is in the weight gradient. Every chunk adds a $[V, d]$ matrix into the
accumulator, and that matrix is $V \times d$ regardless of $c$. With $K = N / c$
chunks, the accumulator is read and written $K$ times, so the traffic from it
grows as $K \times V \times d \times 4 \times 2$ bytes in float32. Small chunks
save logits memory and pay for it in accumulator traffic, and the optimum sits
where the two are comparable. Liger Kernel and PyTorch both pick the chunk size
with a heuristic of this kind, which the next sections describe.

> [!EXAMPLE] The lab's geometry, and what the bytes predict
> The lab uses $N = 8192$ tokens, $d = 512$, $V = 65{,}536$, bf16, and
> chunks of 512 rows. The bf16 logits alone are $8192 \times 65{,}536 \times 2 =
> 1.07$ GB (1024 MiB). One chunk is 64 MiB.
>
> The three GEMMs are $6 \times 8192 \times 512 \times 65{,}536 = 1.65 \times 10^{12}$
> FLOP, which is 5.29 ms at the 312 TFLOP/s rating. That floor is the same for the
> fused and unfused paths.
>
> Count the elementwise traffic with $u = 1.07$ GB (a bf16 $N \times V$ tensor)
> and $f = 2.15$ GB (a float32 one). The unfused path moves $5u + 8f = 22.5$ GB:
> the GEMM write, the cast, the log-softmax, the gradient of the loss, the
> gradient of the log-softmax, the cast back, and two GEMM reads. At 1275 GB/s
> that is 17.7 ms. The fused path moves the logits write, two row-kernel
> reads, one row-kernel write, and two GEMM reads, which is $6u = 6.4$ GB or
> 5.05 ms, plus the accumulator's $16 \times 2 \times 134$ MB $= 4.3$ GB, or 3.4 ms.
>
> So the byte count predicts about 23 ms for the unfused step and 14 ms for the
> fused one, with memory an order of magnitude lower or better. Treat these as predictions: the lab
> measures both, and chapter 13 of the companion course is a warning that byte
> counts can over-promise.

### Accumulating in float32

The weight gradient is a sum over all $N$ tokens, and the chunk loop adds it up
$K$ times. If the accumulator were bfloat16, each addition would round to 8
significant bits, and once the running sum is 256 times larger than a chunk's
contribution the contribution rounds away entirely. So the accumulator is
float32, as `torch.nn.LinearCrossEntropyOptions` does by default for half-precision
input (its `acc_dtype`). The lab's `FusedLinearXent` keeps `grad_w` in float32 and
converts to the weight's dtype only when `backward` returns it.

The matrix multiply that makes each chunk's contribution reads bf16 operands
and, inside the tensor core, accumulates in float32 already. What you have to
decide is where the float32 sum gets rounded. The lab's solution asks
`torch.addmm` on CUDA to add the product into the float32 accumulator without
rounding it to bf16 first, and falls back to rounding each chunk's product once if
that isn't available.

## The softmax-gradient kernel

The chunk's logits have to become $G$, and a custom kernel does that in two
passes over each row without a float32 copy. This section answers how one
Triton program turns a row of logits into its gradient.

The torch version of this step takes five or six kernels, each a full pass over
the chunk. The Triton version uses one program per row, so a chunk of 512 rows
launches 512 programs, which is more than four per SM on a 108-SM A100 and
enough to fill it.

### Pass 1: an online logsumexp

A row has 65,536 or 151,936 elements and a program can't hold it in registers,
so the program walks the row in blocks of 1024. It keeps a running maximum $m$
and a running sum $s = \sum e^{z_j - m}$ over the elements seen so far. When a
block raises the maximum, the old sum is rescaled to the new maximum, the same
rescaling [chapter 15](/c/15-softmax-and-normalization) uses for online softmax:

$$
m' = \max\!\big(\hla{m},\ \max_{\text{block}} z\big), \qquad
s' = \hlb{s}\, e^{m - m'} + \sum_{j \in \text{block}} e^{z_j - m'}
$$

After the last block, $\operatorname{lse} = m + \log s$. The program also reads the
target's logit $z_y$ and writes the row's loss, $\operatorname{lse} - z_y$. It must
read $z_y$ now, because pass 2 overwrites it.

### Pass 2: overwrite the row with its gradient

A second sweep recomputes $p_j = e^{z_j - \operatorname{lse}}$ for each block,
subtracts 1 at the target column, multiplies by the scale, and stores the result
over the logits, in the logits' own dtype:

```python
@triton.jit
def _xent_grad_kernel(logits_ptr, targets_ptr, loss_ptr, stride, V, scale,
                      ignore_index, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    target = tl.load(targets_ptr + row)
    offs = tl.arange(0, BLOCK)

    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for start in range(0, V, BLOCK):               # pass 1: online logsumexp
        cols = start + offs
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(s)

    valid = target != ignore_index
    safe = tl.where(valid, target, 0)
    target_logit = tl.load(base + safe).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))

    for start in range(0, V, BLOCK):               # pass 2: overwrite with the gradient
        cols = start + offs
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(cols == safe, p - 1.0, p) * scale
        p = tl.where(valid, p, 0.0)
        tl.store(base + cols, p.to(logits_ptr.dtype.element_ty), mask=mask)
```

Three details are easy to get wrong. The masked loads use `other=-inf`, so the
padding lanes of the last block add $e^{-\infty} = 0$ to the sum and cannot win
the max. The row offset is `int64`, because a 32-bit offset overflows at $2^{31} = 2.1 \times 10^{9}$
elements, which is a chunk of 14,134 rows of a 151,936 vocabulary, and a bigger
chunk or vocabulary gets there. And an ignored token
(the lab uses `-100`, PyTorch's default) must write zeros, not a gradient, and
must not count in the mean.

The row kernel is memory-bound, as chapter 4 predicts: it reads the row twice and
writes it once, with a few FLOPs per element. It moves $3 \times c \times V \times 2$
bytes per chunk in bf16, which is the $3u$ in the example above.

## The production versions

Three implementations do this job, and each draws the line in a different place.
This section answers what each one does that the lab's version doesn't.

**Liger Kernel** ([Hsu et al.](https://arxiv.org/abs/2410.10989)) is a set of
Triton kernels for LLM training. Its authors report about 20% higher training
throughput and about 60% lower GPU memory than Hugging Face implementations,
using kernel fusion and input chunking; the abstract doesn't name the hardware.
Its fused linear cross-entropy uses the chunking idea of this chapter: it
processes the tokens in chunks and never builds the full logits tensor.

**Cut Cross-Entropy** ([Wijmans et al., Apple](https://arxiv.org/abs/2411.09009))
goes further. It computes only the target's logit and the logsumexp over the
vocabulary on the fly, inside one kernel that does the matrix multiply and the
reduction together in on-chip memory, so even a chunk of logits is never written to
HBM. The authors report loss-computation memory falling from 24 GB to 1 MB, and
the classifier head's total training memory from 28 GB to 1 GB, for Gemma 2 (2B).
The abstract doesn't name the GPU, so read those as the authors' figures. Their
backward also skips the parts of the gradient whose softmax probability is below
numerical precision, which saves work that the lab's version spends.

**`torch.nn.functional.linear_cross_entropy`** is PyTorch's own. Per the
[PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear_cross_entropy.html),
`linear_cross_entropy(input, linear_weight, target, *, linear_bias=None,
weight=None, reduction='mean', ignore_index=None, label_smoothing=0.0,
options=None)` is documented as equivalent to a linear projection followed by
`cross_entropy`. With `options=None` it runs the reference path and builds the
full logits. Passing a `torch.nn.LinearCrossEntropyOptions` opts into
chunking over rows:

| Option | Default | What it does |
|---|---|---|
| `batch_chunk_size` | `None` | Rows per chunk. `None` means one chunk. |
| `chunking_method` | `'auto'` | A heuristic that picks the chunk size. `'aspect_ratio'` sizes a chunk so its logits buffer matches the input's footprint. |
| `acc_policy` | `'auto'` | `'accurate'` or `'compact'`. Compact accumulates the weight gradient with `addmm_` per chunk and avoids a $V \times d$ scratch buffer. |
| `acc_dtype` | `None` | The accumulation dtype. For fp16 or bf16 input on a GPU that supports it, it resolves to float32. |
| `allow_retain_graph` | `False` | If `False`, a second `backward()` raises an error, because the chunked op consumes its precomputed gradients in place. |

The aspect-ratio heuristic is the balance described earlier. For
$N = 8192$, $d = 4096$, and $V = 151{,}936$, the documented formula
`next_pow2(ceil(N / ceil(V / d)))` gives $\lceil V/d \rceil = 38$,
$\lceil 8192 / 38 \rceil = 216$, and a chunk of 256 rows.

The documentation also lists what the chunked path gives up. It sits outside the
standard autograd contract, because it precomputes gradients in the forward pass
and consumes them in the backward pass. It doesn't support higher-order or
forward-mode derivatives or `torch.func.grad`, and `torch.compile` falls back to
eager at that op. Those are the same restrictions the lab's `once_differentiable`
Function has.

> [!NOTE] The lab builds the mechanism, and PyTorch ships the product
> The lab's `FusedLinearXent` is what the chunked path of
> `linear_cross_entropy` does, written out so that you see each piece: the chunk
> loop, the in-place gradient, the fp32 accumulation, and the two saved tensors.
> The tests run PyTorch's op on the same inputs when your PyTorch has it and skip
> the comparison when it doesn't. In production, use the built-in unless you need
> something it lacks.

## Fused optimizers and gradient accumulation

The optimizer is the other place where a training step spends memory bandwidth on
many small tensors. This section answers why `AdamW(fused=True)` exists and why
gradients accumulate in float32.

### One pass over the parameters

An Adam update reads the parameter $p$, the gradient $g$, and both moments, and
writes back $p$, $m$, and $v$. In float32 that is 16 bytes read and 12 bytes
written, 28 bytes per parameter, and the arithmetic is a handful of FLOPs. It is
about as memory-bound as an operation gets. For a 1 billion parameter model the
update moves 28 GB, which takes at least $28 \times 10^{9} / 1.275 \times 10^{12} =
22$ ms at the measured copy ceiling.

A naive implementation runs a loop over the parameter tensors and, for each one,
issues about eight separate elementwise operations: scale the moments, add the
gradient terms, take the square root, divide, apply the weight decay, and update
the parameter. Each reads and writes the whole tensor, so the traffic is several
times the 28 bytes, and each is a kernel launch. A model with 300 parameter
tensors would launch about $300 \times 8 = 2400$ kernels, and at about 5
microseconds of queue time per launch (measured in learn-inference chapter 12)
that is 12 ms of launching, which small tensors can't hide. The 8 per tensor is
an estimate of the operation count; the principle holds whatever the count is.

PyTorch offers two fixes, selected with arguments to the optimizer:

- **`foreach=True`** uses *multi-tensor apply*. A single kernel launch receives a
  list of tensor pointers and processes chunks of many tensors, so the launch
  count drops from the tensor count times the op count to roughly the op count.
  The traffic per op is unchanged.
- **`fused=True`** goes one step further. One kernel does the entire update for
  all the tensors it receives, reading each element once and writing it once.
  That is the 28 bytes per parameter above, the floor.

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
```

This is chapter 13's lesson applied to an optimizer: fusion saves the trips the
intermediates make to memory, and here every intermediate is as large as the
model. For a launch-bound case with thousands of tiny tensors it also saves the
launches that chapter 17 deals with.

### Accumulating gradients

Gradient accumulation runs several micro-batches and sums their gradients before
one optimizer step, so a large effective batch fits in a small memory budget.
The sum is a long chain of additions into one buffer, and it has the same
precision problem as the weight gradient above.

If the accumulation buffer is bfloat16, then each addition rounds to 8 significant
bits. When the running sum is 256 or more times a micro-batch's gradient,
that gradient is rounded away entirely, and well before then each addition adds
rounding error that a float32 buffer wouldn't. Keep the accumulation buffer in
float32. It costs 4 bytes per parameter instead of 2, a real cost against the 16
bytes of the budget above, and it is the cheapest correctness you can buy.

## Mixed precision: loss scaling and bf16

Mixed precision runs the GEMMs in a 16-bit format and keeps the sums and the
master weights in float32. This section answers why float16 needs a loss scale
and bfloat16 doesn't.

The GEMMs use the tensor cores at 312 TFLOP/s for either 16-bit format, against 19.5
TFLOP/s for float32 on the CUDA cores, so the 16x gap is the reason to do it.
What differs between the two formats is the range. From the format table in
FACTS.md and [chapter 9](/c/09-floating-point-on-the-gpu):

| Format | Exponent bits | Smallest positive normal | Smallest subnormal | Largest finite |
|---|---|---|---|---|
| float16 | 5 | $2^{-14} = 6.1 \times 10^{-5}$ | $2^{-24} = 6.0 \times 10^{-8}$ | 65,504 |
| bfloat16 | 8 | $2^{-126} = 1.2 \times 10^{-38}$ | $2^{-133} = 9.2 \times 10^{-41}$ | $3.4 \times 10^{38}$ |
| float32 | 8 | $2^{-126} = 1.2 \times 10^{-38}$ | $2^{-149} = 1.4 \times 10^{-45}$ | $3.4 \times 10^{38}$ |

Gradients are small numbers. A gradient of $10^{-8}$ is common in a deep network,
and in float16 it is below the smallest subnormal, $6.0 \times 10^{-8}$, so it
becomes zero. A layer whose gradients all become zero stops learning.

*Loss scaling* fixes this by multiplying the loss by a constant $S$ before the
backward pass. By the chain rule every gradient is multiplied by $S$ as well, so
the gradients move up into float16's range. With $S = 2^{16} = 65{,}536$, a gradient of
$10^{-8}$ becomes $6.6 \times 10^{-4}$, which is a normal float16 value. Before the
optimizer step the gradients are divided by $S$ again, in float32.

Too large an $S$ overflows to infinity (above 65,504), so a *dynamic* scaler
adapts. PyTorch's `torch.amp.GradScaler` starts high, halves the scale and skips
the step when it finds an infinity or NaN in the gradients, and doubles the scale
after a run of clean steps (by default, 2000). The check and the unscale are
themselves fused multi-tensor kernels, since they touch every gradient.

bfloat16 has float32's exponent, so its range is the same as float32's and its
smallest subnormal is $9.2 \times 10^{-41}$, far below any gradient. It needs no
loss scale. What it gives up is precision: 7 stored mantissa bits against float16's
10, so its machine epsilon is $2^{-7} = 7.8 \times 10^{-3}$ against $9.8 \times 10^{-4}$.
That is why the master weights, the accumulators, and the loss itself stay in
float32 in either recipe, and why the lab's kernel does its softmax arithmetic in
float32 registers even when the logits are stored in bf16.

> [!TIP] Which format to choose on the A100
> bfloat16 is the default for training on an A100 because it removes the scaler
> and its skipped steps. float16 still has the finer precision, which matters for
> some models, and it needs the scaler. Neither is a speed difference: both run
> at 312 TFLOP/s on the tensor cores.

## What goes wrong

- **A `backward` that ignores `grad_output`.** The Function works until someone
  writes `(2 * loss).backward()`, and then the gradients are half what they
  should be. Always multiply by the incoming gradient, and test with a scale
  other than 1.
- **Saving the wrong thing.** A tensor saved by attribute instead of
  `save_for_backward` isn't checked for in-place modification, and an in-place
  update of an input after the forward pass gives a wrong gradient with no error.
- **Returning the wrong number of gradients.** `backward` returns one value per
  argument of `forward`, including the non-tensor ones. A missing `None` is an
  error, but an extra gradient in the wrong slot is a wrong answer.
- **Testing in float32.** `gradcheck` in float32 fails on correct code because the
  finite difference has too few digits. Test in float64 with tiny shapes, and
  test the awkward cases: one token, an ignored token, a chunk larger than the
  batch.
- **Forgetting the ignored tokens.** The mean is over valid tokens only, and an
  ignored row must have a zero gradient. A fused kernel that skips either one
  agrees with the reference until the first padding token.
- **Reading the target's logit after overwriting it.** In the in-place kernel,
  pass 2 destroys the logits. Read $z_y$ in pass 1.
- **Measuring memory with `memory_allocated`.** `torch.cuda.max_memory_allocated`
  is the peak you want, and you have to reset it with
  `torch.cuda.reset_peak_memory_stats()` and subtract the memory already held by
  the inputs, or the number includes things that aren't the loss's.
- **Chunks too small.** At 16 rows per chunk, the $V \times d$ accumulator
  update runs 512 times per step. Past a point, the weight-gradient traffic
  costs more than the logits memory it saves.
- **Accumulating in bf16.** The loss looks fine for the first steps and drifts
  later, because the accumulator rounds small contributions away. Keep the
  accumulation dtype float32.

> [!RECAP]
> - Training costs three GEMMs per linear layer and keeps every layer's saved
>   inputs until the backward pass, so memory, not FLOP/s, sets the batch size.
> - Mixed-precision Adam holds 16 bytes per parameter, and a GPT-style layer
>   saves $34sbh$ bytes of activations plus $5as^{2}b$ for the score matrix.
> - Recomputation spends one extra forward pass (4/3 of the arithmetic) to save the
>   activations. A custom backward is a `torch.autograd.Function`, checked with
>   `gradcheck` in float64.
> - The logits are $N \times V \times 4$ bytes: 4.64 GiB for 8192 tokens and a
>   151,936 vocabulary. A scalar loss lets you compute the loss and both gradients
>   one chunk at a time, and the full logits never exist.
> - A fused optimizer reads and writes each parameter once, and fp32 accumulation
>   and loss scaling keep small gradients from rounding away.

## Check your understanding

> [!QUESTION] Why does a training step need about three times the arithmetic of the forward pass?
> Each linear layer's backward pass computes two gradients, one for its input and
> one for its weight, and each is a GEMM with the same $2MNK$ FLOP as the forward
> GEMM. One forward GEMM plus two backward GEMMs is three.

> [!QUESTION] A model has 3 billion parameters and trains with mixed-precision Adam. How much memory do the parameters, gradients, and optimizer state take, and does it fit on an 80 GB A100?
> $3 \times 10^{9} \times 16$ bytes is 48 GB, so it fits, leaving about 32 GB for
> activations, the logits, and the allocator's overhead. The 16 bytes are 2
> (bf16 weight), 2 (bf16 gradient), 4 (fp32 master weight), 4 ($m$), and 4 ($v$).

> [!QUESTION] You checkpoint every layer of a 32-layer model, and the step still runs out of memory at a larger batch. What is the most likely culprit?
> The logits. They scale with tokens times vocabulary and aren't part of any
> layer, so layer checkpointing doesn't touch them: at 32,768 tokens and a
> 151,936 vocabulary they are 18.5 GiB in float32. Use a fused, chunked
> cross-entropy.

> [!QUESTION] Why does the fused loss compute the gradient in the forward pass, and when would that be wrong?
> The loss is a scalar, so the gradient of the logits is the softmax minus the
> one-hot target times a scalar upstream gradient. Everything but that scalar is
> known in the forward pass, so the gradients can be computed there and scaled in
> `backward`. It is wasted work if no backward follows, and it doesn't apply to a
> per-token (`reduction="none"`) loss, whose upstream gradient is a vector you don't have yet.

> [!QUESTION] The lab uses chunks of 512 rows. What do you pay for by making them 64 rows, and what do you gain?
> You gain memory: a chunk of logits is 8 times smaller, 8 MiB instead of 64 MiB in
> bf16. You pay in the weight gradient. There are 8 times as many chunks, so the
> $V \times d$ accumulator is read and written 8 times as often, and each chunk's GEMMs are
> smaller and run further from the tensor-core peak.

> [!QUESTION] A gradient of $10^{-8}$ is computed in float16. What happens to it, and how does loss scaling fix it?
> It is below half the smallest subnormal, $2^{-24} = 6.0 \times 10^{-8}$, so it
> rounds to zero and the information is lost. Multiplying the loss by
> $2^{16} = 65{,}536$ scales it to $6.6 \times 10^{-4}$, a normal float16 value; the
> optimizer divides by the same factor afterwards in float32.

> [!QUESTION] Why does the Triton kernel read the target's logit before its second pass?
> The second pass overwrites the logits with the gradient, in place. After it, the
> original $z_y$ is gone, and the loss $\operatorname{lse} - z_y$ would be wrong.

## Lab

> [!TRY] Fused linear cross-entropy
> Fill in `softmax_xent_grad_torch_`, the Triton kernel behind
> `softmax_xent_grad_triton_`, and `FusedLinearXent.forward` and `backward`. You
> pass when the loss and the gradients of the hidden states and the weight match a float64
> reference for shapes that include a single token, a vocabulary smaller than one
> block, ignored targets, and a chunk larger than the batch; when
> `torch.autograd.gradcheck` passes in float64; and when the fused step's peak
> memory is below the size of the bf16 logits alone. Your lab reports
> `ref_peak_mb`, `fused_peak_mb`, and `memory_ratio`, the step times
> `ref_ms` and `fused_ms`, and `fused_tflops` against the 312 TFLOP/s rating. If
> your PyTorch has `linear_cross_entropy`, it reports that op's memory and time
> too, as `torch_lce_peak_mb` and `torch_lce_ms`. Compare them with the
> predictions in the worked example above.

## Further reading

- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009):
  Wijmans et al., on never storing the logits, and on skipping negligible gradient.
- [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989):
  Hsu et al., and its [repository](https://github.com/linkedin/Liger-Kernel) for the reference Triton kernels.
- [`torch.nn.functional.linear_cross_entropy`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear_cross_entropy.html)
  and [`LinearCrossEntropyOptions`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LinearCrossEntropyOptions.html):
  the production equivalent and its limits.
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html):
  custom Functions, `save_for_backward`, `once_differentiable`, and `gradcheck`.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174):
  Chen et al., the origin of recomputation.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198):
  Korthikanti et al., the per-layer accounting and selective recomputation.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740): Micikevicius et al.,
  master weights and loss scaling.
- [Triton layer normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html):
  a backward kernel with partial reductions.
- [NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine): fused
  kernels and FP8 training layers.
