#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Gemma2 2B model on Roadrunner with JAX, Flax.
# 
# We have adopted the Gemma notebook from Google Deepmind to use HuggingFace's libraries, added support for doing **model parallel training** and simplified the setup.

# ## Setup 

# In[1]:


import os
import sys
import importlib
def import_local_module(module_path: str):
    sys.path.append('')
    module = importlib.import_module(module_path)
    return importlib.reload(module)

# Imports felafax trainer_engine
setup = import_local_module("trainer_engine.setup")
setup.setup_environment()


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kagglehub -q\n!pip install ipywidgets -q\n!pip install torch --index-url https://download.pytorch.org/whl/cpu -q\n!pip install git+https://github.com/felafax/gemma.git -q\n!pip install qax -q\n!pip install jax-lorax -q\n!pip install ipywidgets -q\n!pip install torch --index-url https://download.pytorch.org/whl/cpu -q\n!pip install --upgrade jax -q \n!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q\n!pip install "flax[all]" -q\n!pip install --upgrade optax==0.2.2\n!pip install --upgrade einops\n!pip install --no-cache-dir transformers==4.43.3\n!pip install --no-cache-dir datasets==2.18.0\n!pip install --upgrade tqdm\n!pip install --upgrade requests\n!pip install --upgrade typing-extensions\n!pip install --upgrade mlxu>=0.1.13\n!pip install --upgrade sentencepiece\n!pip install --upgrade pydantic\n!pip install --upgrade fastapi\n!pip install --upgrade uvicorn\n!pip install --upgrade gradio\n')


# In[17]:


globals().update(setup.setup_imports())

utils = import_local_module("trainer_engine.utils")
jax_utils = import_local_module("trainer_engine.jax_utils")
llama_model = import_local_module("trainer_engine.llama_model")
checkpoint_lib = import_local_module("trainer_engine.checkpoint_lib")
training_pipeline = import_local_module("trainer_engine.training_pipeline")


# ## Step 0: Input your HF username, token and download model weights

# ### Select the base model you want to fine-tune 👇

# In[5]:


# Select a supported model from above list to use!
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
JAX_MODEL_NAME = "felafax/llama-3.1-8B-JAX"
model_path = "/mnt/persistent-disk/fax/llama3.1_8b_serialized.flax"


# ### Input your HuggingFace🤗 username and token below

# In[6]:


hf_model_name = MODEL_NAME
HUGGINGFACE_USERNAME = input("INPUT: Please provide your HUGGINGFACE_USERNAME: ")
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")


# In[7]:


config = AutoConfig.from_pretrained(
    MODEL_NAME, 
    token=HUGGINGFACE_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    token=HUGGINGFACE_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token


# In[8]:


from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id=JAX_MODEL_NAME, token=HUGGINGFACE_TOKEN)


# ## Step 1: prepare the dataset
# 
# For this project, we're utilizing the refined **Alpaca dataset**, curated by yahma. This dataset is a carefully filtered selection of 52,000 entries from the original Alpaca collection. Feel free to substitute this section with your own data preparation code if you prefer.
# 
# It's crucial to include the EOS_TOKEN (End of Sequence Token) in your tokenized output. Failing to do so may result in endless generation loops.

# In[9]:


def get_dataset(*, tokenizer, batch_size=1, max_length=32, max_examples=None):
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""
    
    EOS_TOKEN = tokenizer.eos_token
    
    # Defines formatting function.
    def _format_prompts(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _tokenize(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length+1)
        return {
            'input_tokens': [input_id[:-1] for input_id in tokenized['input_ids']],
            'target_tokens': [input_id[1:] for input_id in tokenized['input_ids']],
            'loss_masks': [input_id[1:] for input_id in tokenized['attention_mask']]
        }

    def _custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """
        Collates batch items and converts PyTorch tensors to JAX arrays.
        Applies default_data_collator, then converts tensors to JAX format.
        """
        collated = default_data_collator(batch)
        jax_batch = {}
        for key, value in collated.items():
            jax_batch[key] = jnp.array(value.numpy()) if isinstance(value, torch.Tensor) else value
        
        return jax_batch

    # Load and preprocess the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if max_examples:
        dataset = dataset.select(range(max_examples))
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    for split in ['train', 'test']:
        ds[split] = ds[split].map(_tokenize, batched=True, remove_columns=dataset.column_names)

    # Create DataLoaders
    dataloader_args = dict(shuffle=True, batch_size=batch_size, collate_fn=_custom_collate_fn)
    train_dataloader = torch.utils.data.DataLoader(ds['train'], **dataloader_args)
    test_dataloader = torch.utils.data.DataLoader(ds['test'], **dataloader_args)

    return train_dataloader, test_dataloader


# **Uncomment below code ⬇️ if you'd like to run and test 💯 your dataset pipeline.**

# In[10]:


def test_dataset_pipeline(tokenizer):
    """Print shapes of first batch to verify dataset pipeline."""
    train_loader, _ = get_dataset(tokenizer=tokenizer, batch_size=1, max_length=32, max_examples=512)
    batch = next(iter(train_loader))
    print("Input tokens shape:", batch['input_tokens'].shape)
    print("Target mask shape:", batch['target_tokens'].shape)
test_dataset_pipeline(tokenizer)


# ## Step 2: Train the model by configuring the hyperparameters below.

# In[11]:


@chex.dataclass(frozen=True)
class TrainingConfig:
  learning_rate: float = 1e-4
  num_epochs: int = 1
  max_steps: int | None = 5  # max number of training steps (**set to None** to train for full num_epochs)

  # Dataset config
  batch_size: int = 32
  max_length: int = 64  # max seq lenght of tokens in input batch
  dataset_size_limit: int | None = 512    # limit on number of dataset examples for testing (**set to None** to use full dataset)

  # Misc config
  print_every_n_steps: int = 1

training_cfg = TrainingConfig()


# **NOTE**: The **time-to-first step of training will be slow** because XLA takes time initially to compile the computational graph. However, once the compilation is complete, subsequent steps will run much faster using the compiled and cached graph, leveraging the full power of all TPU cores for accelerated training.

# In[12]:


devices = jax.devices()
device_count = len(devices)
device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
mesh = Mesh(devices=device_mesh, axis_names=('dp', 'fsdp', 'mp'))


# In[13]:


llama_config = llama_model.LlamaConfig.get_standard_llama_config("llama3_8b")
llama_config = llama_model.LlamaConfig.finalize_config(llama_config)
model = llama_model.CausalLlamaModule(
    llama_config, # Merges with PretrainedConfig from huggingface.
    dtype=jnp.float32,
    param_dtype=jnp.float32,
)
optimizer = optax.sgd(training_cfg.learning_rate)


# In[14]:


train_dataloader, val_dataloader = get_dataset(tokenizer=tokenizer, max_length=training_cfg.max_length, max_examples=training_cfg.dataset_size_limit)


# In[15]:


model_path = os.path.join(model_path, "llama3.1_8b_serialized.flax")


# In[ ]:


state = training_pipeline.train_loop(
    model=model,
    model_path=model_path,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    training_cfg=training_cfg,
    mesh=mesh
)

