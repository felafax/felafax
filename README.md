# Felafax -- tune LLaMa3.1 on Google Cloud TPUs for 30% lower cost and scale seamlessly!
<div align="center">
    <div>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/felafax/felafax?logo=github" />
        <img alt="GitHub License" src="https://img.shields.io/github/license/felafax/felafax"></img>
    </div>
    <br />
</div>
          
![image](./misc/assets/roadrunner.jpg)

Felafax is a framework for continued-training and fine-tuning open source LLMs using **XLA runtime**. We take care of necessary runtime setup and provide a Jupyter notebook out-of-box to just get started.
- Easy to use.
- Easy to configure all aspects of training (designed for ML researchers and hackers).
- Easy to scale training from a single TPU VM with 8 cores to entire TPU Pod containing 6000 TPU cores (**1000X**)!

## ✨ Finetune for Free

Add your dataset, click "Run All", and you'll run on free TPU resource on Google Colab!

| Felafax supports | Free Notebooks |
|-------------------|-----------------|
| **Llama 3.1 (8B)** | [▶️ Start for free on Google Colab TPU](https://colab.research.google.com/github/felafax/felafax/blob/main/~archive/Llama3_1_8b_on_Free_Colab_TPU.ipynb) |

## Goal
Our goal at [felafax](https://felafax.ai) is to build infra to make it easier to run AI workloads on non-NVIDIA hardware (TPU, AWS Trainium, AMD GPUs, and Intel GPUs).

## Currently supported models
- **LLaMa-3.1 JAX Implementation** $${\color{red}New!}$$	 
  - Converted from PyTorch to JAX for improved performance
  - On TPU v4, v5, runs 2-way data parallel and 2-way model parallel training (2 data parallel model copies and each model copy is sharded across two TPU chips).
  - On TPU v2, v3, runs 1 model copy sharded across 8 cores.
  - Full-precision and LoRA training support

- **LLaMa-3/3.1 PyTorch XLA**
  - LoRA and full-precision training support
  - [codepointer](https://github.com/felafax/felafax/tree/main/~archive/llama3_pytorch_xla)

## Setup

**For a hosted version with a seamless workflow, please request access [here](https://tally.so/r/mRLeaQ). 🦊.**

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

## AMD 405B fine-tuning run:
We recently fine-tuned the llama3.1 405B model on 8xAMD MI300x GPUs using JAX instead of PyTorch. JAX's advanced sharding APIs allowed us to achieve great performance. Check out our [blog post](https://dub.sh/felafax-amd-blog) to learn about the setup and the sharding tricks we used.

We did LoRA fine-tuning with all model weights and lora parameters in bfloat16 precision, and with LoRA rank of 8 and LoRA alpha of 16:
- **Model Size:** The LLaMA model weights occupy around 800GB of VRAM.
- **LoRA Weights + Optimizer State:** Approximately 400GB of VRAM.
- **Total VRAM Usage:** 77% of the total VRAM, around 1200GB.
- **Constraints:** Due to the large size of the 405B model, there was limited space for batch size and sequence length. The batch size used was 16 and the sequence length was 64.
- **Training Speed:** ~35 tokens/second
- **Memory Efficiency:** Consistently around 70%
- **Scaling:** With JAX, scaling was near-linear across 8 GPUs.

The GPU utilization and VRAM utilization graphs can be found below. However, we still need to calculate the Model FLOPs Utilization (MFU).
Note: We couldn't run the JIT-compiled version of the 405B model due to infrastructure and VRAM constraints (we need to investigate this further). The entire training run was executed in JAX eager mode, so there is significant potential for performance improvements.

- GPU utilization:
  ![image](./misc/assets/amd_405B_run_sep22/gpu_utilization.png)
- VRAM  utilization:
  ![image](./misc/assets/amd_405B_run_sep22/vram_utilization.png)
- rocm-smi data can be found [here](./misc/assets/amd_405b_run_sep22/rocm_smi_cleaned_405b_batchsize16_seqlen64.csv).

## Credits:
- Google Deepmind's [Gemma repo](https://github.com/google-deepmind/gemma).
- [EasyLM](https://github.com/young-geng/EasyLM) and EleutherAI for great work on llama models in JAX
- PyTorch XLA FSDP and SPMD testing done by [HeegyuKim](https://github.com/HeegyuKim/torch-xla-SPMD).
- Examples from [PyTorch-XLA](https://github.com/pytorch/xla/) repo.

## Contact
If you have any questions, please contact us at founders@felafax.ai.
