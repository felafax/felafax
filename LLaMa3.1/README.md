# Supports LLaMa3 and LLaMa 3.1 fine-tuning

LLaMa 3 and 3.1 8B, 70B variants can be trained on a single TPU VM host (preferably use v5p-8). LLaMa 3.1 405B requires multi-host training and will be available through our platform at [felafax.ai](https://felafax.ai), sign-up for the waitlist!


To just start the trainer with out-of-box setup, just run `python trainer.py` on the Jupyterlab terminal.


Once you verify that out-of-box setup works, you can customize the different aspects of your pipeline:
- Configure the dataset you want to use in `dataset.py`
- Under `trainer.py`:
    - Configure the `MODEL_NAME` and insert your `HUGGINGFACE_TOKEN`.
    - Configure hyperparams and LoRA config via trainer.
- For more advanced configuration of LoRA and model_partitioning, check otu `model.py` and `model_partitioning.py` file respectively.
