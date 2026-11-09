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
        git curl ca-certificates build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# CUTLASS 3.9.2 is the last 3.x release and is header-only. Labs include
# /opt/cutlass/include. https://github.com/NVIDIA/cutlass/releases/tag/v3.9.2
ARG CUTLASS_REF=v3.9.2
RUN git clone --depth 1 --filter=blob:none --sparse --branch ${CUTLASS_REF} \
        https://github.com/NVIDIA/cutlass.git /opt/cutlass \
    && git -C /opt/cutlass sparse-checkout set include \
    && rm -rf /opt/cutlass/.git \
    && test -f /opt/cutlass/include/cutlass/cutlass.h \
    && test -f /opt/cutlass/include/cute/tensor.hpp
ENV CUTLASS_INCLUDE_DIR=/opt/cutlass/include

# triton 3.8.0 is the release torch 2.14.1 pins; it is named here so a change to
# the base image cannot move it silently. The base image already ships numpy and
# ninja that match its torch. Its Python is the system one (Ubuntu 24.04), which
# refuses pip installs unless told otherwise.
#
# nvidia-cutlass-dsl 4.8.1 is the CUDA 13 build of the CuTe DSL. That package
# line starts at 4.0, so the C++ headers above stay on CUTLASS 3.9.2 and the
# Python package is pinned separately. The [cu13] extra matches this image.
# https://pypi.org/project/nvidia-cutlass-dsl/4.8.1/
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html
RUN pip install --no-cache-dir --break-system-packages \
        triton==3.8.0 \
        einops==0.8.1 \
        runpod==1.7.7 \
        "nvidia-cutlass-dsl[cu13]==4.8.1"

# Nsight Compute 2025.3.1 matches CUDA Toolkit 13.0 Update 1, the toolkit in
# this image. The redistributable also contains the GUI (ncu-ui and host/).
# Keep the CLI only.
# https://developer.nvidia.com/nsight-compute-history
# https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-2025.3.1.4-archive.tar.xz
ARG NCU_VERSION=2025.3.1.4
RUN set -eu; \
    curl -fsSL -o /tmp/ncu.tar.xz \
        "https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-${NCU_VERSION}-archive.tar.xz"; \
    mkdir -p /tmp/ncu-src; \
    tar -xJf /tmp/ncu.tar.xz -C /tmp/ncu-src; \
    root=""; \
    for dir in $(find /tmp/ncu-src -type d -name target); do root=$(dirname "$dir"); break; done; \
    test -n "$root"; \
    if [ ! -f "$root/ncu" ]; then \
        inner=$(find "$root" -type f -name ncu | head -n 1); \
        test -n "$inner"; \
        ln -s "$inner" "$root/ncu"; \
    fi; \
    rm -rf "$root/host" "$root/ncu-ui"; \
    find "$root" -name ncu-ui -print0 | xargs -0 -r rm -rf; \
    rm -rf /opt/nsight-compute; \
    mv "$root" /opt/nsight-compute; \
    ln -sfn /opt/nsight-compute/ncu /usr/local/bin/ncu; \
    rm -rf /tmp/ncu-src /tmp/ncu.tar.xz; \
    test ! -e /opt/nsight-compute/ncu-ui; \
    test ! -d /opt/nsight-compute/host; \
    ncu --version

WORKDIR /repo
COPY content/labs /repo/content/labs
COPY gpu/harness.py /repo/gpu/harness.py
COPY gpu/lab_runner.py /repo/gpu/lab_runner.py
COPY gpu/runpod_worker/handler.py /repo/gpu/handler.py

# Fail the build, not the first lab, if the toolchain is missing.
RUN nvcc --version && ninja --version \
    && python -c "import importlib.metadata as m, triton, torch; assert m.version('nvidia-cutlass-dsl') == '4.8.1'; print(torch.__version__, torch.version.cuda, triton.__version__)" \
    && test -f "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h" \
    && test -f "${CUTLASS_INCLUDE_DIR}/cute/tensor.hpp" \
    && ncu --version \
    && test ! -e /opt/nsight-compute/ncu-ui \
    && test ! -d /opt/nsight-compute/host

CMD ["python", "-u", "/repo/gpu/handler.py"]
