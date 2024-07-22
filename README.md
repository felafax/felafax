# TunerX 🦊 -- tune LLaMa3 on Google Cloud TPUs for 30% lower cost and easly scale training from 1X to 1000X!

Felafax TunerX is a framework for continued-training and fine-tuning open source LLMs using **XLA runtime**. We take care of neceessary runtime setup and provide a Jupyter notebook out-of-box to just get started.
- Easy to use
- Easy to configure all aspects of training (designed for ML Developers by ML Developers)"
- Easy to scale training from a single VM with 8 TPU cores to entire TPU Pod containing 6000 TPU cores (**1000X**)!

## Currently supported models

- LLaMa3 8B on Google Cloud TPUs
  - Supports LoRA and full-precision training.
  - Tested on TPU v3-8

## Setup

The instructions below guide you through launching a TPU VM on your Google Cloud account and firing up a Jupyter notebook. With just 3 simple steps, you'll be up and running in under 10 minutes. 🚀

1. Install gcloud command-line tool and authenticate your account (SKIP this STEP if you already have gcloud installed and have used TPUs before! 😎)

   ```bash
    # Download gcloud CLI
    curl https://sdk.cloud.google.com | bash
    source ~/.bashrc

    # Authenticate gcloud CLI
    gcloud auth login

    # Create a new project for now
    gcloud projects create LLaMa3-tunerX --set-as-default

    # Set up default credentials
    gcloud auth application-default login

    # Enable Cloud TPU API access
    gcloud services enable compute.googleapis.com tpu.googleapis.com storage-component.googleapis.com aiplatform.googleapis.com
   ```

2. Spin up a TPU v3-8 VM 🤠.

    ```bash
    sh ./launch_tuner.sh
    ```
    Keep an eye on the terminal -- you might be asked to input SSH key password and need to put in your HuggingFace token. 

3. Open the Jupyter notebook at `https://localhost:888` and start fine-tuning!

## Credits:
- PyTorch XLA FSDP and SPMD testing done by [HeegyuKim](https://github.com/HeegyuKim/torch-xla-SPMD).
- Examples from [PyTorch-XLA](https://github.com/pytorch/xla/) repo.
