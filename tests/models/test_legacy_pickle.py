"""Tests for flashmd.models.legacy_pickle."""

import io
import os
import sys
import types

import pytest
import torch

from flashmd.models import legacy_pickle
from flashmd.models.radial_basis import ExpNormalBasis

_FAKE_PATH = "mlcg.nn.radial_basis.exp_normal"


@pytest.fixture
def fake_mlcg(monkeypatch):
    """Installs a stand-in for mlcg's ExpNormalBasis for the test only."""
    parts = _FAKE_PATH.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fake_cls = type(
        "ExpNormalBasis",
        (ExpNormalBasis,),
        {"__module__": _FAKE_PATH, "__qualname__": "ExpNormalBasis"},
    )
    setattr(sys.modules[_FAKE_PATH], "ExpNormalBasis", fake_cls)
    return fake_cls


def _save(obj) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("mlcg.nn.schnet", "flashmd.models.schnet"),
        (
            "mlcg.nn.radial_basis.exp_normal",
            "flashmd.models.radial_basis.exp_normal",
        ),
        ("mlcg.nn.prior", "flashmd.prior"),
        ("mlcg.data.atomic_data", "flashmd.data.atomic_data"),
        ("mlcg.nn.schnetx", "mlcg.nn.schnetx"),
        ("torch.nn", "torch.nn"),
    ],
)
def test_remap_module(module, expected):
    assert legacy_pickle.remap_module(module) == expected


def test_loads_mlcg_pickle_as_flashmd_class(fake_mlcg, monkeypatch):
    original = fake_mlcg(cutoff=5.0, num_rbf=8)
    buffer = _save(original)
    assert _FAKE_PATH.encode() in buffer.getvalue()
    for name in [
        n for n in sys.modules if n == "mlcg" or n.startswith("mlcg.")
    ]:
        monkeypatch.delitem(sys.modules, name)

    restored = legacy_pickle.load(buffer)

    assert type(restored) is ExpNormalBasis
    dist = torch.linspace(0.0, 6.0, 13)
    torch.testing.assert_close(restored(dist), original(dist))


def test_protocol_two_builtins_load():
    restored = legacy_pickle.load(_save({"s": {1, 2}, "t": int, "n": 3}))
    assert restored == {"s": {1, 2}, "t": int, "n": 3}


@pytest.mark.parametrize("obj", [os.system, eval])
def test_refuses_unexpected_globals(obj):
    with pytest.raises(legacy_pickle.UnexpectedGlobalError):
        legacy_pickle.load(_save([obj]))


@pytest.mark.heavy
def test_published_checkpoint_loads_without_mlcg():
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.hf_hub_download("pingzhili/cg-schnet", "model_and_prior.pt")
    except Exception as error:  # Network is optional for this test.
        pytest.skip(f"Hugging Face Hub unavailable: {error}")
    from flashmd.models import load_and_adapt_old_checkpoint

    model = load_and_adapt_old_checkpoint(path)
    modules = {type(m).__module__ for m in model.modules()}
    assert not any(m.startswith("mlcg") for m in modules)
    assert any(isinstance(m, ExpNormalBasis) for m in model.modules())
