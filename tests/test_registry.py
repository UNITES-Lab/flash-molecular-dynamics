"""Tests for flashmd.registry."""

from collections.abc import Iterator

import pytest
import torch

from flashmd import config as config_lib
from flashmd import registry
from flashmd.models import base


class _DummyGenerator(base.Generator):
    """Minimal generator used to exercise plugin resolution."""

    name = "dummy"
    units = base.Units(energy="kcal/mol")

    @classmethod
    def load(cls, source, *, device="cpu", dtype=None):
        del source, device, dtype
        return cls()

    def optimize(self, config: config_lib.ExecutionConfig) -> None:
        del config

    @property
    def reference_module(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def sample(self, request, *, generator=None) -> Iterator[base.SampleBatch]:
        del generator
        yield base.SampleBatch(
            start=0,
            positions=torch.zeros(
                request.num_samples, len(request.sequence), 3
            ),
            orientations=torch.eye(3).expand(
                request.num_samples, len(request.sequence), 3, 3
            ),
        )


NotAPlugin = object


@pytest.fixture
def spec():
    spec = registry.ModelSpec(
        alias="test-dummy",
        plugin=f"{__name__}:_DummyGenerator",
        source="unused",
        description="Test entry.",
    )
    registry.register(spec)
    yield spec
    registry.unregister(spec.alias)


def test_register_and_get(spec):
    assert registry.get("test-dummy") is spec
    assert spec in registry.list_models()


def test_duplicate_alias_raises(spec):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)
    registry.register(spec, overwrite=True)


def test_unknown_alias_lists_known_models(spec):
    with pytest.raises(KeyError, match="test-dummy"):
        registry.get("no-such-model")


def test_resolve_plugin_imports_class(spec):
    plugin_cls = registry.resolve_plugin(spec)
    assert plugin_cls is _DummyGenerator
    plugin = plugin_cls.load(spec.source)
    batch = next(plugin.sample(base.SampleRequest("ACD", num_samples=2)))
    assert batch.positions.shape == (2, 3, 3)


def test_resolve_rejects_non_plugins():
    bad = registry.ModelSpec("bad", f"{__name__}:NotAPlugin", "unused")
    with pytest.raises(TypeError):
        registry.resolve_plugin(bad)


def test_missing_extra_gives_install_hint():
    missing = registry.ModelSpec(
        "missing", "flashmd.models.mace.no_such_module:Plugin", "unused"
    )
    with pytest.raises(ImportError, match=r"flash-molecular-dynamics\[mace\]"):
        registry.resolve_plugin(missing)


@pytest.mark.parametrize("path", ["no_colon", ":Class", "module:"])
def test_spec_validates_import_path(path):
    with pytest.raises(ValueError):
        registry.ModelSpec("x", path, "unused")


def test_sample_request_validation():
    with pytest.raises(ValueError):
        base.SampleRequest("", num_samples=1)
    with pytest.raises(ValueError):
        base.SampleRequest("ACD", num_samples=0)
