# syntax=docker/dockerfile:1.6
# Team GPUs are Blackwell (sm_120). PyTorch <2.7 wheels target sm_50…sm_90
# only and crash on Blackwell with "no kernel image is available for
# execution on the device" even when the CUDA toolkit is 12.6+. Do not
# downgrade this tag.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# libgl1 / libglib2.0-0 are OpenCV runtime deps; build-essential is for any
# custom CUDA op the training team may compile later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Deps are installed at build time as root so the runtime `omr` user does not
# need write access to /opt/conda. Do NOT `chown -R` /opt/conda anywhere: Docker
# copy-on-write would duplicate ~15 GB of CUDA libs into a new layer and the
# image would balloon to ~47 GB.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system -r /tmp/requirements.txt

# UID/GID 1000 matches the default WSL2 user, so bind-mounted files don't
# show up root-owned on the host.
ARG USERNAME=omr
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home ${USERNAME} \
    && mkdir -p /workspace \
    && chown ${USERNAME}:${USERNAME} /workspace

WORKDIR /workspace
USER ${USERNAME}

CMD ["sleep", "infinity"]
