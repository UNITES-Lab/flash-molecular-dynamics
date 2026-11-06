"""Registry of published models that FlashMD can run.

Each entry maps a short alias such as `"bioemu-v1.1"` to a plugin class and
the source its weights come from. Plugin classes are named by import path and
imported only when an entry is loaded, so `import flashmd` stays light and the
MACE and BioEmu extras are needed only by the models that use them.

Third-party packages can add entries through the `flashmd.models` entry-point
group. Each entry point must resolve to a callable returning an iterable of
`ModelSpec`.

Typical usage:

    spec = registry.get("bioemu-v1.1")
    plugin_cls = registry.resolve_plugin(spec)
"""

from collections.abc import Iterable
import dataclasses
import importlib
from importlib import metadata
import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from flashmd.models import base

ENTRY_POINT_GROUP = "flashmd.models"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One loadable model.

    Attributes:
        alias: Short name users pass to `flashmd.load`.
        plugin: Import path of the plugin class, as `"package.module:Class"`.
        source: Argument passed to the plugin's `load`, such as a Hugging
            Face repository id or an upstream model name.
        description: One line shown by `flashmd.registry.list_models`.
    """

    alias: str
    plugin: str
    source: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validates the import path format."""
        module, sep, attr = self.plugin.partition(":")
        if not (module and sep and attr):
            raise ValueError(
                f"plugin must look like 'package.module:Class', got "
                f"{self.plugin!r}."
            )


_lock = threading.Lock()
_specs: dict[str, ModelSpec] = {}
_entry_points_loaded = False


def register(spec: ModelSpec, *, overwrite: bool = False) -> None:
    """Adds a model to the registry.

    Args:
        spec: The entry to add.
        overwrite: Whether an existing entry with the same alias may be
            replaced.

    Raises:
        ValueError: If the alias is taken and `overwrite` is `False`.
    """
    with _lock:
        if spec.alias in _specs and not overwrite:
            raise ValueError(
                f"Model alias {spec.alias!r} is already registered."
            )
        _specs[spec.alias] = spec


def unregister(alias: str) -> None:
    """Removes a model from the registry if present.

    Args:
        alias: The alias to remove.
    """
    with _lock:
        _specs.pop(alias, None)


def _load_entry_points() -> None:
    """Registers entries advertised by installed packages, once."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            provider = entry_point.load()
            specs: Iterable[ModelSpec] = provider()
        except Exception as error:  # A broken plugin must not break FlashMD.
            logger.warning(
                "Skipping FlashMD plugin entry point {}: {}",
                entry_point.name,
                error,
            )
            continue
        for spec in specs:
            register(spec, overwrite=True)


def get(alias: str) -> ModelSpec:
    """Looks up a registered model.

    Args:
        alias: The alias to look up.

    Returns:
        The matching entry.

    Raises:
        KeyError: If no model is registered under `alias`. The message lists
            the known aliases.
    """
    _load_entry_points()
    with _lock:
        if alias in _specs:
            return _specs[alias]
        known = ", ".join(sorted(_specs)) or "none"
    raise KeyError(f"Unknown model {alias!r}. Known models: {known}.")


def list_models() -> list[ModelSpec]:
    """Returns all registered models sorted by alias."""
    _load_entry_points()
    with _lock:
        return [_specs[alias] for alias in sorted(_specs)]


def resolve_plugin(spec: ModelSpec) -> "type[base.Potential | base.Generator]":
    """Imports and returns the plugin class of an entry.

    Args:
        spec: The entry whose plugin to import.

    Returns:
        The plugin class.

    Raises:
        ImportError: If the plugin's module or its optional dependencies are
            missing. The message names the extra to install when known.
        TypeError: If the import path does not name a FlashMD plugin class.
    """
    from flashmd.models import base  # Deferred to keep `import flashmd` light.

    module_name, _, attr = spec.plugin.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        hint = _install_hint(module_name)
        raise ImportError(
            f"Cannot load the plugin for {spec.alias!r}: {error}.{hint}"
        ) from error
    plugin_cls = getattr(module, attr, None)
    if not (
        isinstance(plugin_cls, type)
        and issubclass(plugin_cls, (base.Potential, base.Generator))
    ):
        raise TypeError(
            f"{spec.plugin} is not a FlashMD Potential or Generator."
        )
    return plugin_cls


def _install_hint(module_name: str) -> str:
    """Returns an install hint for a plugin module, or an empty string."""
    for extra in ("mace", "bioemu"):
        if f".models.{extra}" in module_name:
            return (
                f" Install it with: pip install "
                f"'flash-molecular-dynamics[{extra}]'."
            )
    return ""
