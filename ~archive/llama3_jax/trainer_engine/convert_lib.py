"""TODO: fix this file soon!"""
import gc
import json
import math
import os
import shutil

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.traverse_util import flatten_dict
from transformers import LlamaConfig, LlamaForCausalLM

from . import checkpoint_lib


def save_hf_compatible_checkpoint(load_path, out_dir, model_configurator):

    def match_keywords(string, positives, negatives):
        """
        Helper function to match keywords in a string.
            
        Returns: True if all positive keywords are present and no negative keywords are present
        """
        for positive in positives:
            if positive not in string:
                return False
        for negative in negatives:
            if negative in string:
                return False
        return True

    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def write_json(text, path):
        with open(path, "w") as f:
            json.dump(text, f)

    def permute(w, n_heads, input_dim, output_dim):
        """
        Helper function to permute weight matrices for sliced rotary embedding.
        
        Args:
        w: Input weight matrix
        n_heads: Number of attention heads
        input_dim: Input dimension
        output_dim: Output dimension
        
        Returns: Permuted weight matrix
        """
        return w.view(n_heads, output_dim // n_heads // 2, 2,
                      input_dim).transpose(1,
                                           2).reshape(output_dim, input_dim)

    def load_and_convert_checkpoint(path):
        """
        Load a Flax checkpoint and convert it to PyTorch format.
        
        1. Load the Flax checkpoint
        2. Flatten the parameter dictionary
        3. Convert Flax tensors to PyTorch tensors
        4. Transpose weight matrices for certain layers
        """
        _, flax_params = checkpoint_lib.Checkpointer.load_trainstate_checkpoint(
            path)
        flax_params = flatten_dict(flax_params['params'], sep='.')
        torch_params = {}
        for key, tensor in flax_params.items():
            # Transpose linear layer weights to match PyTorch's dimension order
            #
            # Flax/JAX and PyTorch handle linear layer weights differently:
            # - Flax/JAX: weights shape is (in_features, out_features)
            #   Matrix multiplication: output = input @ weight
            # - PyTorch: weights shape is (out_features, in_features)
            #   Matrix multiplication: output = weight @ input
            #
            # Example:
            # For a linear layer with 768 input features and 3072 output features:
            # - Flax weight shape: (768, 3072)
            # - PyTorch weight shape: (3072, 768)
            #
            # By transposing the weights during conversion, we ensure that
            # the converted PyTorch model produces the same outputs as the original Flax model.
            if match_keywords(key, ["kernel"], ["norm", 'ln_f']):
                tensor = tensor.T  # Transpose weight matrices for linear layers
            torch_params[key] = torch.tensor(
                checkpoint_lib.float_tensor_to_dtype(tensor, 'fp32'),
                dtype=torch.float16)
        return torch_params

    def write_model(loaded, model_path, llama_pretrained_config):
        """
        Write the converted model to disk in HuggingFace format.
        
        1. Create necessary directories
        2. Set up model configuration
        3. Iterate through layers, converting and saving parameters
        4. Save model configuration and metadata
        5. Load the saved model to verify and finalize the conversion
        """
        os.makedirs(model_path, exist_ok=True)
        tmp_model_path = os.path.join(model_path, "tmp")
        os.makedirs(tmp_model_path, exist_ok=True)

        llama_config = llama_pretrained_config
        n_layers = llama_config.num_hidden_layers
        n_heads = llama_config.num_attention_heads
        n_kv_heads = llama_config.num_key_value_heads
        dim = llama_config.hidden_size
        dims_per_head = dim // n_heads
        base = llama_config.rope_theta
        inv_freq = 1.0 / (base**(torch.arange(0, dims_per_head, 2).float() /
                                 dims_per_head))

        param_count = 0
        index_dict = {"weight_map": {}}
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            state_dict = {
                # Convert attention weights
                f"model.layers.{layer_i}.self_attn.q_proj.weight":
                permute(
                    loaded[f"transformer.h.{layer_i}.attention.wq.kernel"],
                    llama_config.num_attention_heads,
                    llama_config.hidden_size,
                    llama_config.hidden_size,
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight":
                permute(
                    loaded[f"transformer.h.{layer_i}.attention.wk.kernel"],
                    llama_config.num_key_value_heads,
                    llama_config.hidden_size,
                    llama_config.hidden_size //
                    (llama_config.num_attention_heads //
                     llama_config.num_key_value_heads),
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight":
                loaded[f"transformer.h.{layer_i}.attention.wv.kernel"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight":
                loaded[f"transformer.h.{layer_i}.attention.wo.kernel"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight":
                loaded[f"transformer.h.{layer_i}.feed_forward.w1.kernel"],
                f"model.layers.{layer_i}.mlp.down_proj.weight":
                loaded[f"transformer.h.{layer_i}.feed_forward.w2.kernel"],
                f"model.layers.{layer_i}.mlp.up_proj.weight":
                loaded[f"transformer.h.{layer_i}.feed_forward.w3.kernel"],
                f"model.layers.{layer_i}.input_layernorm.weight":
                loaded[f"transformer.h.{layer_i}.attention_norm.kernel"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight":
                loaded[f"transformer.h.{layer_i}.ffn_norm.kernel"],
            }

            # Save rotary embedding parameters
            state_dict[
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # Save non-layer-specific parameters (embeddings, final norm, and LM head)
        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["transformer.wte.embedding"],
            "model.norm.weight": loaded["transformer.ln_f.kernel"],
            "lm_head.weight": loaded["lm_head.kernel"],
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # Write configs
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(
            index_dict,
            os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

        config = LlamaConfig(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=llama_config.num_key_value_heads,
            initializer_range=llama_config.initializer_range,
            rms_norm_eps=llama_config.rms_norm_eps,
            max_position_embeddings=llama_config.max_position_embeddings,
            rope_theta=llama_config.rope_theta,
        )
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        # Load the saved model to verify and finalize the conversion
        print("Loading the checkpoint in a Llama model.")
        model = LlamaForCausalLM.from_pretrained(tmp_model_path,
                                                 torch_dtype=torch.float16)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

        print("Saving in the Transformers format.")
        model.save_pretrained(model_path)
        shutil.rmtree(tmp_model_path)

    loaded_params = load_and_convert_checkpoint(load_path)
    write_model(
        loaded_params,
        model_path=out_dir,
        llama_pretrained_config=model_configurator.get_hf_pretrained_config(
            model_configurator.get_model_config()),
    )


def upload_checkpoint_to_hf(checkpoint_path, hf_path, hf_token):
    from huggingface_hub import HfApi
    api = HfApi()

    print(f"Uploading checkpoint to {hf_path}...")
    api.upload_folder(folder_path=checkpoint_path,
                      repo_id=hf_path,
                      repo_type="model",
                      ignore_patterns=[".*"],
                      token=hf_token)
    print(f"Uploaded checkpoint to {hf_path}.")
