# Scripts for LLaMa 3/3.1 fine-tuning (LoRA or full-precision).

This repository contains scripts and notebooks for fine-tuning LLaMa 3 and LLaMa 3.1 models.

## Contents

- `trainer_lib/`: Directory containing utility files for training
  - `dataset.py`: Configure the dataset for training
  - `model.py`: Advanced configuration for LoRA
  - `model_partitioning.py`: Configure model partitioning
- `llama3_70b_train.ipynb`: Notebook for training 70B model
- `llama3_8b_train.ipynb`: Notebook for training 8B model
- `multihost_trainer.py`: Script for multi-host training

## Supported Models

- LLaMa 3 and LLaMa 3.1 (8B, 70B variants)
  - Can be trained on a single TPU VM host (preferably use v5p-8)
- LLaMa 3.1 405B
  - Requires multi-host training, will be available through our cloud platform at [felafax.ai](https://felafax.ai).
  - Sign-up for the [waitlist](https://tally.so/r/mRLeaQ)!

## Quick Start

To begin training, use one of the following Jupyter notebooks:

- For 8B model: `llama3_8b_train.ipynb`
- For 70B model: `llama3_70b_train.ipynb`

Open the appropriate notebook in JupyterLab and follow the instructions within to start your training process.
