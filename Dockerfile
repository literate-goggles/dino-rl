FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=UTC

# mostly development or debugging tools
RUN apt update && apt install -y sudo curl wget git ffmpeg tmux s3fs htop nvtop

RUN apt-get update && apt-get install -y --no-install-recommends \
    # essential RDMA libraries
    rdma-core \
    libibverbs-dev \
    # diagnostic tools
    infiniband-diags \
    iproute2 \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# System libraries required by the Selenium-managed headless Chrome used by the
# browser-image PPO env (dino_rl/browser_env.py). Without these the cached
# Chrome binary aborts on launch with "error while loading shared libraries".
# This is Chrome's declared dependency set (see its deb.deps manifest).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 \
    libatspi2.0-0 libcairo2 libcups2 libcurl4 libdbus-1-3 libexpat1 libgbm1 \
    libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 libudev1 libvulkan1 \
    libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 \
    libxrandr2 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /build
RUN pip install wheel setuptools pip pybind11
RUN pip3 --no-cache-dir install --upgrade awscli

ADD requirements.txt /build/requirements.txt

RUN grep -v "flash-attn" /build/requirements.txt > /build/requirements_no_flash.txt
RUN pip install -r /build/requirements_no_flash.txt
RUN pip install flash-attn==2.7.0.post2 --no-build-isolation

# The base image ships torch built for CUDA 12.4 (kernels up to sm_90), which
# cannot run on Blackwell GPUs (B200 = sm_100): every CUDA op fails with
# "no kernel image is available for execution on the device". Reinstall torch
# from the CUDA 12.8 index, which bundles its own CUDA runtime and includes
# sm_100/sm_120 kernels. Done last so it overrides the base image's torch.
RUN pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

