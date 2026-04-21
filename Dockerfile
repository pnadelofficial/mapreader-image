# =============================================================================
# MapReader + MapTextPipeline (rumsey-finetune.pth)
#
# Base: CUDA 11.7 + cuDNN 8 + Ubuntu 20.04
# Matches the MapTextPipeline upstream spec exactly:
#   Python 3.8 / PyTorch 2.0.1+cu117 / CUDA 11.7
#
# Build:
#   docker build -t mapreader-maptext .
#
# Run (GPU):
#   docker run --gpus all -it \
#     -v $(pwd)/maps:/workspace/maps \
#     -v $(pwd)/patches:/workspace/patches \
#     -v $(pwd)/models:/workspace/models \
#     -p 8888:8888 \
#     mapreader-maptext
#
# Run (CPU only):
#   docker run -it \
#     -v $(pwd)/maps:/workspace/maps \
#     -v $(pwd)/patches:/workspace/patches \
#     -v $(pwd)/models:/workspace/models \
#     -p 8888:8888 \
#     mapreader-maptext
#
# For Singularity (HPC) - build on a machine with Docker, then convert:
#   docker save mapreader-maptext | gzip > mapreader-maptext.tar.gz
#   singularity build mapreader-maptext.sif docker-archive://mapreader-maptext.tar.gz
#
# Or pull directly with Singularity if you push to a registry:
#   singularity pull docker://youruser/mapreader-maptext:latest
# =============================================================================
 
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
 
# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
 
# CUDA environment - pin to 11.7 throughout
ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
 
WORKDIR /workspace
 
# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # GDAL for geospatial support (needed by geopandas/mapreader geo features)
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Make python3.8 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && python -m pip install --upgrade pip setuptools wheel
 
# ---------------------------------------------------------------------------
# 2. PyTorch - pinned to 2.0.1+cu117 to match MapTextPipeline spec exactly
# ---------------------------------------------------------------------------
RUN pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
 
# Sanity check - will print True if CUDA is available at build time
# (only works if building on a GPU host; safe to ignore if building on CPU)
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || true
 
# ---------------------------------------------------------------------------
# 3. detectron2 - build from source with CUDA 11.7
#    No --no-build-isolation needed here because torch is already installed
# ---------------------------------------------------------------------------
RUN pip install ninja==1.11.1

RUN git clone https://github.com/facebookresearch/detectron2.git /opt/detectron2 \
    && cd /opt/detectron2 \
    && pip install -e .
 
# ---------------------------------------------------------------------------
# 4. MapTextPipeline (maps-as-data fork for CPU compatibility)
# ---------------------------------------------------------------------------
RUN git clone https://github.com/maps-as-data/MapTextPipeline.git /opt/MapTextPipeline \
    && cd /opt/MapTextPipeline \
    && pip install -r requirements.txt \
    && python setup.py build develop
 
# Set the ADET_PATH environment variable that MapReader uses to locate the pipeline
ENV ADET_PATH=/opt/MapTextPipeline
 
# ---------------------------------------------------------------------------
# 5. MapReader with text spotting extras
# ---------------------------------------------------------------------------
RUN pip install "mapreader[text]"
 
# ---------------------------------------------------------------------------
# 6. Jupyter + common scientific stack for interactive use
# ---------------------------------------------------------------------------
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    pandas \
    geopandas \
    shapely \
    tqdm
 
# ---------------------------------------------------------------------------
# 7. Expose model weight / config volumes and workspace
#    Mount your maps, patches, and model files here at runtime
# ---------------------------------------------------------------------------
VOLUME ["/workspace/maps", "/workspace/patches", "/workspace/models"]
 
# ---------------------------------------------------------------------------
# 8. Entrypoint - starts Jupyter by default, easy to override
# ---------------------------------------------------------------------------
EXPOSE 8888
 
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/workspace"]