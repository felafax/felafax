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

Our goal at [felafax](https://felafax.ai) is to build infra to make it easier to run AI workloads on non-NVIDIA hardware (TPU, AWS Trainium, AMD GPUs, and Intel GPUs).

## ✨ Finetune for Free

Add your dataset, click "Run All", and you'll run on free TPU resource on Google Colab!

| Felafax supports | Free Notebooks |
|-------------------|-----------------|
| **Llama 3.1 (1B, 3B)** | [▶️ Start for free on Google Colab TPU](https://dub.sh/felafax-colab) |

## Currently supported models
- **LLaMa-3.1 JAX Implementation** $${\color{red}New!}$$	 
  - Converted from PyTorch to JAX for improved performance
  - Full-precision and LoRA training support for **1B, 3B, 8B, 70B, 405B**.
  - Run efficiently across diverse hardware (TPUs, AWS Trainium, NVIDIA, AMD) through JAX's hardware-optimized XLA backend
  - Scale seamlessly to handle larger context lengths and datasets by sharding across multiple accelerators

- **LLaMa-3/3.1 PyTorch XLA**
  - LoRA and full-precision training support
  - [codepointer](https://github.com/felafax/felafax/tree/main/~archive/llama3_pytorch_xla)

## Running fine-tuning via Felafax CLI $${\color{red}New!}$$	

Get started with fine-tuning your models using the Felafax CLI in a few simple steps.

### Step 1. Install the CLI and authenticate

Start off by installing the CLI.

```bash
pip install pipx
pipx install felafax-cli
```

Then, generate an Auth Token:

- Visit [preview.felafax.ai](https://preview.felafax.ai) and create/sign in to your account.
- Navigate to [Tokens](https://preview.felafax.ai/tokens) page and create a new token.

Finally, authenticate your CLI session using your token:

```bash
felafax-cli auth login --token <your_token>
```

### Step 2. Set up the fine-tuning config

First, generate a default configuration file for fine-tuning. This command generates a `config.yml` file in the current directory with default hyperparameter values.

```bash
felafax-cli tune init-config
```

Second, update the config file with your hyperparameters:

- **HuggingFace knobs:**

  - Provide your HuggingFace token and repository ID to upload the fine-tuned model.

- **Dataset pipeline and training params:**
  - Adjust `batch_size`, `max_seq_length` to use for fine-tuning dataset.
  - Set num_steps to `null` if you want trainig to run through entire dataset. If num_steps is set to a number, training will stop after the specified number of steps.
  - Set `learning_rate` and `lora_rank` to use for fine-tuning.
  - `eval_interval` is the number of steps between evaluations.

### Step 3. Start the fine-tuning run

Run the follow command to see the list of base models you can fine-tune, we support all variants of LLaMA-3.1 as of now.

```bash
felafax-cli tune start --help
```

Now, you can start the fine-tuning process with your selected model from above list and dataset name from HuggingFace (like `yahma/alpaca-cleaned`):

```bash
felafax-cli tune start --model <your_selected_model> --config ./config.yml --hf-dataset-id <your_hf_dataset_name>
```

Example command to get you started:

```bash
felafax-cli tune start --model llama3-2-1b --config ./config.yml --hf-dataset-id yahma/alpaca-cleaned
```

After you start the fine-tuning job, **Felafax CLI takes care of spinning up the TPUs, running the training, and it uploads the fine-tuned model to the HuggingFace Hub.**

### Other handy commands

##### Monitor the fine-tuning job
You can stream realtime logs to monitor the progress of your fine-tuning job:

```bash
# Use `<job_name>` with the job namethat you get after starting the fine-tuning.
felafax-cli tune logs --job-id <job_name> -f
```

##### List your fine-tuned models
After fine-tuning is complete, you can list all your fine-tuned models:

```bash
felafax-cli model list
```

##### Chat with your fine-tuned model (runs on TPU again!):

You can start an interactive terminal session to chat with your fine-tuned model:

```bash
# Replace `<model_id>` with model id from `model list` command you ran above.
felafax-cli model chat --model-id <model_id>
```

##### Use help to explore more commands!

The CLI is broken into three main command groups:
- **`tune`**: To start/stop fine-tuning jobs.
- **`model`**: To manage and interact with your fine-tuned models.
- **`files`**: To upload/view yourdataset files.

Use the `--help` flag to discover more about any command group:
```bash
felafax-cli tune --help
```

## AMD 405B fine-tuning run
<details>

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
</details>

## Credits:
- Google Deepmind's [Gemma repo](https://github.com/google-deepmind/gemma).
- [EasyLM](https://github.com/young-geng/EasyLM) and EleutherAI for great work on llama models in JAX
- PyTorch XLA FSDP and SPMD testing done by [HeegyuKim](https://github.com/HeegyuKim/torch-xla-SPMD).
- Examples from [PyTorch-XLA](https://github.com/pytorch/xla/) repo.

## Contact
If you have any questions, please contact us at founders@felafax.ai.
