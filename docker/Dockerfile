# For TPU v3-8, use pt-xla-2.3
FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH ./
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

LABEL project="LLaMa3-tunerX-docker-v5"

# RUN sed -i 's/deb.debian.org/ftp.de.debian.org/g' /etc/apt/sources.list

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
RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu1.9.0/libtpu.so -o /lib/libtpu.so

WORKDIR /home/

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip

# Install required libs
# pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir transformers==4.43.3 -U
RUN pip install --no-cache-dir datasets==2.18.0
RUN pip install --no-cache-dir trl==0.8.1 peft==0.10.0
RUN pip install --no-cache-dir accelerate==0.28.0
RUN pip install --no-cache-dir jupyterlab
RUN pip install --no-cache-dir torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html

WORKDIR /home


# Clone the RoadrunnerX repository
RUN git clone https://github.com/felafax/RoadrunnerX.git

# Command to run when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]