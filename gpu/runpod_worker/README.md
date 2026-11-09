# RunPod worker

RunPod runs every lab in the course. This directory holds the serverless
worker that executes labs: `handler.py` receives a lab id and the reader's code,
runs it through the lab runner in `gpu/lab_runner.py`, and streams events
back.

The image is built on `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel`. The
*devel* variant carries `nvcc`, which labs need to compile CUDA C++ with
`torch.utils.cpp_extension.load_inline`. The runtime variant has none. The
image also installs `ninja` and a pinned `triton`, and it sets
`TORCH_CUDA_ARCH_LIST=8.0`, because the labs target the A100 (sm_80). A Hopper
lab passes `arch="sm_90a"` to `load_cuda` for that compilation. The image
carries the CUTLASS 3.9.2 headers (`CUTLASS_INCLUDE_DIR`), the
`nvidia-cutlass-dsl` 4.8.1 package built for CUDA 13, and the Nsight Compute
2025.3.1 CLI (`ncu`). It does not install the Nsight Compute GUI.

## Build and push

Run the build from the repository root, so the context includes `content/`:

```bash
docker build -f gpu/runpod_worker/Dockerfile -t REGISTRY/learn-gpu-worker:latest .
```

Replace `REGISTRY` with your container registry. Then push it:

```bash
docker push REGISTRY/learn-gpu-worker:latest
```

[The publish workflow](../../.github/workflows/publish-runpod-worker.yml) builds
and pushes the same image to GitHub Container Registry, so you only need to
build by hand when you want to test an image before it lands on `main`.

The worker carries the labs inside the image. A new image is the only way a
code change reaches the GPU.

## Create the endpoint

1. In the RunPod console, create a serverless endpoint from the pushed image.
2. Choose an **A100 80GB** worker. Labs measure bandwidth and throughput
   against that card's figures, and the CUDA code compiles for sm_80.
3. Leave the minimum worker count at zero. An idle endpoint costs nothing, and
   the compute panel can put always-on workers back to zero if one gets set.
4. Copy the endpoint id into `RUNPOD_ENDPOINT_ID` in the app's `.env`, and set
   `RUNPOD_API_KEY`. `RUNPOD_ENDPOINT_ID` is the A100 endpoint when
   `RUNPOD_ENDPOINT_A100` is unset, so an existing deployment keeps serving
   the A100 labs. `H100-80GB` reads `RUNPOD_ENDPOINT_H100`. A lab with
   `"gpus": 2` reads `RUNPOD_ENDPOINT_2XA100`. A missing endpoint does not
   send the lab to a different card. For an unconfigured H100, the lab pane
   shows `needs H100, not configured on this deployment`. The other cards use
   the same shape of sentence. The checks that need no GPU still run.

The worker needs no network volume and no Hugging Face token: no lab downloads
weights.

## What the first run costs you

A cold worker pulls the image and imports PyTorch before it runs anything, which
takes a minute or two. A lab that compiles CUDA then waits another 30 to 60
seconds for `nvcc`; the build is cached in `TORCH_EXTENSIONS_DIR`, so a warm
worker that has seen the same source skips it. The compute panel shows the
endpoint's worker states, so a run that seems stuck is usually a worker still
initializing, or one RunPod has throttled for lack of capacity.
