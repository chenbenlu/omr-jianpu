# syntax=docker/dockerfile:1.6
# OMR-to-Jianpu dev image.
# The team's GPUs (RTX 5060 Laptop / 5070 / 6000) are Blackwell (sm_120).
# Blackwell binary kernels first shipped in PyTorch 2.7.0 (April 2025) against
# CUDA 12.8. PyTorch 2.6 and earlier wheels only target sm_50…sm_90 and fail
# at runtime with "no kernel image is available for execution on the device",
# even when the CUDA *toolkit* is 12.6+. 2.8 is the current stable line.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# System deps. libgl1 / libglib2.0-0 are required by OpenCV; build-essential
# for any custom CUDA op the training team may compile later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv to a system-wide location.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Install Python deps at build time as root, into the conda env's site-packages.
# This means the runtime `omr` user does NOT need write access to /opt/conda —
# which is critical: a `chown -R omr /opt/conda` here would copy ~15 GB of CUDA
# libs into a new layer (Docker copy-on-write) and balloon the image to ~47 GB.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system -r /tmp/requirements.txt

# Non-root user. UID/GID 1000 matches the default WSL2 user so files created inside
# the container do not show up as root-owned on the host bind mount.
# IMPORTANT: only mkdir /workspace and let the bind mount handle ownership at runtime.
# Do NOT `chown -R` system dirs like /opt/conda — see comment above.
ARG USERNAME=omr
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home ${USERNAME} \
    && mkdir -p /workspace \
    && chown ${USERNAME}:${USERNAME} /workspace

WORKDIR /workspace
USER ${USERNAME}

# Keep the container alive so VS Code Dev Containers / `make shell` can attach.
CMD ["sleep", "infinity"]
