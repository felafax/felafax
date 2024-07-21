# For TPU v3-8, use pt-xla-2.1
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH ./

LABEL project="LLaMa3-tunerX-docker"

# Install system dependencies
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
  cmake \
  curl \
  wget \
  sudo \
  gnupg \
  libsm6 \
  libxext6 \
  libxrender-dev \
  lsb-release \
  ca-certificates \
  build-essential \
  git \
  libgl1 \ 
  && rm -rf /var/lib/apt/lists/*

# Download libtpu.so
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.6.0/libtpu.so -o /lib/libtpu.so

WORKDIR /home/

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip

# Install required libs
# pip install --no-cache-dir -r requirements.txt
RUN pip install transformers==4.38.2 -U
RUN pip install datasets==2.18.0
RUN pip install trl==0.8.1 peft==0.10.0
RUN pip install accelerate==0.28.0
RUN pip install jupyterlab

RUN git clone https://github.com/felafax/tunerX.git tunerX

WORKDIR /home

# Command to run when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
