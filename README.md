# Felafax -- tune LLaMa3.1 on Google Cloud TPUs for 30% lower cost and scale seamlessly!

![image](./utils/assets/image.jpg)

RoadRunnerX is a framework for continued-training and fine-tuning open source LLMs using **XLA runtime**. We take care of necessary runtime setup and provide a Jupyter notebook out-of-box to just get started.
- Easy to use.
- Easy to configure all aspects of training (designed for ML researchers and hackers).
- Easy to scale training from a single TPU VM with 8 cores to entire TPU Pod containing 6000 TPU cores (**1000X**)!

## Goal
Our goal at [felafax](https://felafax.ai) is to build infra to make it easier to run AI workloads on non-NVIDIA hardware (TPU, AWS Trainium, AMD GPU, and Intel GPU).

## Currently supported models
- **LLaMa-3.1 JAX Implementation** $${\color{red}New!}$$	 
  - Converted from PyTorch to JAX for improved performance
  - By default, runs 2-way data parallel and 2-way model parallel training (2 data parallel model copies and each model copy is sharded across two TPU chips).
  - Compatible with NVIDIA GPUs and TPUs
  - Full-precision training support

- **LLaMa-3/3.1 PyTorch XLA**
  - LoRA and full-precision training support

- **Gemma2 Models (2B, 9B, 27B)**
  - Optimized for Cloud TPUs
  - Fast full-precision training

## Setup

**For a hosted version with a seamless workflow, please request access [here](https://tally.so/r/mRLeaQ). Here is a demo of our platform ([demo](https://www.loom.com/share/0b4bb303513d44a1bc5d01d2c22f1a85?sid=01c6b9c4-a94e-4959-b135-b5adaf792e68)) 🦊.**

If you prefer a self-hosted training version, follow the instructions below. These steps will guide you through launching a TPU VM on your Google Cloud account and starting a Jupyter notebook. With just 3 simple steps, you'll be up and running in under 10 minutes. 🚀

1. Install gcloud command-line tool and authenticate your account (SKIP this STEP if you already have gcloud installed and have used TPUs before! 😎)

   ```bash
    # Download gcloud CLI
    curl https://sdk.cloud.google.com | bash
    source ~/.bashrc

    # Authenticate gcloud CLI
    gcloud auth login

    # Create a new project for now
    gcloud projects create LLaMa3-tunerX --set-as-default

    # Config SSH and add
    gcloud compute config-ssh --quiet
   
    # Set up default credentials
    gcloud auth application-default login

    # Enable Cloud TPU API access
    gcloud services enable compute.googleapis.com tpu.googleapis.com storage-component.googleapis.com aiplatform.googleapis.com
   ```

2. Spin up a TPU v5-8 VM 🤠.

    ```bash
    sh ./launch_tuner.sh
    ```
    Keep an eye on the terminal -- you might be asked to input SSH key password and need to put in your HuggingFace token. 

3. Clone the repo and install dependencies

    ```bash
    git clone https://github.com/felafax/felafax.git
    cd felafax
    pip install -r requirements.txt
    ```

3. Open the Jupyter notebook at `https://localhost:888` and start fine-tuning!

## Credits:
- Google Deepmind's [Gemma repo](https://github.com/google-deepmind/gemma).
- [EasyLM](https://github.com/young-geng/EasyLM) for great work on llama models in JAX
- PyTorch XLA FSDP and SPMD testing done by [HeegyuKim](https://github.com/HeegyuKim/torch-xla-SPMD).
- Examples from [PyTorch-XLA](https://github.com/pytorch/xla/) repo.

## Contact
If you have any questions, please contact us at founders@felafax.ai.
