ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.1.1
ARG TORCH_VERSION=2.3.0

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \ 
    apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    curl git python3 python3-dev build-essential pkg-config ninja-build libaio1

ENV VIRTUAL_ENV=/opt/venv
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

RUN /root/.cargo/bin/uv venv --seed ${VIRTUAL_ENV}
ENV PATH="/root/.cargo/bin/:${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build
COPY requirements.txt /build

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==${TORCH_VERSION}+cu121 torchvision torchaudio triton xformers && \
    uv pip install https://github.com/chengzeyi/stable-fast/releases/download/v1.0.5/stable_fast-1.0.5+torch230cu121-cp310-cp310-manylinux2014_x86_64.whl && \
    uv pip install git+https://github.com/huggingface/diffusers.git git+https://github.com/huggingface/transformers && \
    uv pip install -r requirements.txt && \
    uv pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnx onnxruntime-gpu && \
    uv pip install tensorflow bitsandbytes DeepCache git+https://github.com/openai/CLIP.git && \
    uv pip install tensorrt --no-build-isolation && \
    uv pip install --pre -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu121 oneflow && \
    uv pip install --pre git+https://github.com/siliconflow/onediff.git

# RUN DS_BUILD_OPS=1 pip install deepspeed 

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    curl git python3 libgl1 libglib2.0-0 libglfw3-dev libgles2-mesa-dev \
    pkg-config libcairo2 libcairo2-dev libjemalloc-dev libibverbs-dev 
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so"

RUN curl -L https://repo.jellyfin.org/files/ffmpeg/ubuntu/latest-6.x/amd64/jellyfin-ffmpeg6_6.0.1-6-jammy_amd64.deb -o ffmpeg.deb
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt install ./ffmpeg.deb -y && \
    ln -s /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/bin/ffmpeg && \
    ln -s /usr/lib/jellyfin-ffmpeg/ffprobe /usr/bin/ffprobe && \
    ln -s /usr/lib/jellyfin-ffmpeg/vainfo /usr/bin/vainfo && \
    rm /ffmpeg.deb

RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/* /var/log/*

ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app
COPY . /app

ARG DATA_DIR=/config
VOLUME ${DATA_DIR}
RUN mkdir -p data && mv data ${DATA_DIR}
ENV SD_DATADIR=${DATA_DIR}

ARG MODELS_DIR=${DATA_DIR}/models
RUN mkdir -p models && mv models ${MODELS_DIR}
ENV SD_MODELSDIR=${MODELS_DIR}

ENV SD_CONFIG=${DATA_DIR}/config.json

ARG PORT=7860
EXPOSE ${PORT}
ENV PORT=${PORT}

STOPSIGNAL SIGINT

RUN python3 -c "import installer; installer.install_submodules()"

CMD ["python3", "launch.py", "--version", "--insecure", "--allow-code", "--listen", "--cors-origins", "*", "--skip-requirements", "--skip-torch", "--experimental"]
