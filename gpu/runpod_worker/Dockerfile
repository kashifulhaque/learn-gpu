# RunPod serverless worker. Carries the labs and a full CUDA toolkit, so every
# run gets the code it was built with and can compile CUDA C++ with nvcc.
#
# The base image is a *devel* image: the runtime variant has no nvcc, and
# `torch.utils.cpp_extension.load_inline` needs it. Its CUDA toolkit (13.0) is
# the one this torch wheel was built against, which is what load_inline wants.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    # Build for the A100 only. Compiling for every architecture multiplies
    # nvcc's time and the lab's first-run wait.
    TORCH_CUDA_ARCH_LIST=8.0 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    MAX_JOBS=4 \
    LG_LABS_DIR=/repo/content/labs

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# triton 3.8.0 is the release torch 2.14.1 pins; it is named here so a change to
# the base image cannot move it silently. The base image already ships numpy and
# ninja that match its torch. Its Python is the system one (Ubuntu 24.04), which
# refuses pip installs unless told otherwise.
RUN pip install --no-cache-dir --break-system-packages \
        triton==3.8.0 \
        einops==0.8.1 \
        runpod==1.7.7

WORKDIR /repo
COPY content/labs /repo/content/labs
COPY gpu/harness.py /repo/gpu/harness.py
COPY gpu/lab_runner.py /repo/gpu/lab_runner.py
COPY gpu/runpod_worker/handler.py /repo/gpu/handler.py

# Fail the build, not the first lab, if the toolchain is missing.
RUN nvcc --version && ninja --version && python -c "import triton, torch; print(torch.__version__, torch.version.cuda, triton.__version__)"

CMD ["python", "-u", "/repo/gpu/handler.py"]
