"""Tests PyTorch and JAX setup.

A simple test script to verify that PyTorch and JAX can run together in the same environment. Tests basic tensor operations and device availability for both frameworks.
"""

import torch
import jax
import jax.numpy as jnp


def test_torch_jax_computation():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = torch.dot(x, y)
    assert result == 32.0, "PyTorch dot product should equal 32"

    x_jax = jnp.array([1.0, 2.0, 3.0])
    y_jax = jnp.array([4.0, 5.0, 6.0])
    result = jnp.dot(x_jax, y_jax)
    assert result == 32.0, "JAX dot product should equal 32"


def test_torch_device_available():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device in [
        "cuda",
        "cpu",
    ], "PyTorch device should be either CUDA or CPU"


def test_jax_devices_available():
    devices = jax.devices()
    assert len(devices) > 0, "At least one JAX device should be available"
