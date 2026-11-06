"""Tests for flashmd.config."""

import dataclasses

import pytest
import torch

from flashmd import config as config_lib


def test_default_is_strict():
    config = config_lib.ExecutionConfig()
    assert config.backend is config_lib.Backend.FLASH
    assert config.dtype is None
    assert not config.allow_tf32
    assert not config.reduced_precision


def test_reference_disables_every_kernel():
    config = config_lib.ExecutionConfig.reference()
    assert config.is_reference
    assert not config.cuda_graphs
    assert not config.compile
    assert not config.kernel_enabled("schnet.fused_cfconv")


def test_kernel_overrides():
    config = config_lib.ExecutionConfig(kernels={"schnet.fused_rbf": False})
    assert not config.kernel_enabled("schnet.fused_rbf")
    assert config.kernel_enabled("schnet.fused_cfconv")


def test_kernel_overrides_are_frozen():
    overrides = {"schnet.fused_rbf": False}
    config = config_lib.ExecutionConfig(kernels=overrides)
    overrides["schnet.fused_rbf"] = True
    assert not config.kernel_enabled("schnet.fused_rbf")
    with pytest.raises(TypeError):
        config.kernels["schnet.fused_rbf"] = True  # type: ignore[index]


def test_backend_accepts_strings():
    config = config_lib.ExecutionConfig(backend="reference")
    assert config.is_reference


def test_config_is_frozen():
    config = config_lib.ExecutionConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.compile = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [{"dtype": torch.int32}, {"max_batch": 0}, {"backend": "fast"}],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        config_lib.ExecutionConfig(**kwargs)


def test_replace_returns_new_config():
    config = config_lib.ExecutionConfig()
    tf32 = config.replace(allow_tf32=True, dtype=torch.float32)
    assert tf32.allow_tf32 and tf32.dtype is torch.float32
    assert not config.allow_tf32
