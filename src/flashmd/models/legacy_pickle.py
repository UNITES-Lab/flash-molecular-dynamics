"""Loads pickled mlcg checkpoints and data without mlcg installed.

The published CGSchNet checkpoint and its starting configurations are pickles
of `mlcg` objects. `load` resolves each pickled class against FlashMD's ports
of those modules and refuses globals from modules a CGSchNet file never uses.

The allowlist catches files that were not written by mlcg or FlashMD. It is
not a security boundary: like `torch.load(weights_only=False)`, this executes
pickle code, so only load files you trust.

Typical usage:

    model = legacy_pickle.load("model_and_prior.pt")
    configurations = legacy_pickle.load("1enh_configurations.pt")
"""

import _compat_pickle
from collections.abc import Callable
import importlib
import os
import pickle
import types
from typing import Any, BinaryIO

import torch

# Longest prefix wins; submodules keep their relative path.
_MODULE_REMAP: dict[str, str] = {
    "mlcg.nn.schnet": "flashmd.models.schnet",
    "mlcg.nn.gradients": "flashmd.models.gradients",
    "mlcg.nn.mlp": "flashmd.models.mlp",
    "mlcg.nn.cutoff": "flashmd.models.cutoff",
    "mlcg.nn.radial_basis": "flashmd.models.radial_basis",
    "mlcg.nn.prior": "flashmd.prior",
    "mlcg.data": "flashmd.data",
    "mlcg.neighbor_list": "flashmd.neighbor_list",
    "mlcg.geometry": "flashmd.geometry",
}

_ALLOWED_PACKAGES: frozenset[str] = frozenset(
    {
        "_codecs",
        "_operator",
        "collections",
        "copyreg",
        "flashmd",
        "inspect",
        "numpy",
        "operator",
        "torch",
        "torch_geometric",
        "typing",
    }
)

_ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
        "type",
    }
)


class UnexpectedGlobalError(pickle.UnpicklingError):
    """Raised when a pickle references a global outside the allowlist."""


def remap_module(module: str) -> str:
    """Maps an mlcg module path to its FlashMD port.

    Args:
        module: Module path recorded in the pickle.

    Returns:
        The FlashMD module path, or `module` unchanged if no mapping applies.
    """
    for prefix in sorted(_MODULE_REMAP, key=len, reverse=True):
        if module == prefix or module.startswith(prefix + "."):
            return _MODULE_REMAP[prefix] + module[len(prefix) :]
    return module


def resolve(module: str, name: str) -> Any:
    """Returns the global `module.name` after remapping and allowlisting.

    Args:
        module: Module path recorded in the pickle, after Python 2 name
            fixing.
        name: Attribute name recorded in the pickle.

    Returns:
        The resolved object.

    Raises:
        UnexpectedGlobalError: If the global is not allowed.
    """
    target = remap_module(module)
    if target == "builtins":
        if name not in _ALLOWED_BUILTINS:
            raise UnexpectedGlobalError(f"Refusing builtins.{name}.")
    elif target.split(".", 1)[0] not in _ALLOWED_PACKAGES:
        raise UnexpectedGlobalError(f"Refusing {module}.{name}.")
    return getattr(importlib.import_module(target), name)


class _Unpickler(pickle.Unpickler):
    """Unpickler that remaps mlcg classes and enforces the allowlist."""

    def find_class(self, module: str, name: str) -> Any:
        """Applies Python 2 name fixes, then resolves through `resolve`.

        Overriding `find_class` skips the fixes the default implementation
        applies to protocol 2 pickles (torch's default), so they are applied
        here. Python 3 never writes the old names, so this is a no-op for
        newer protocols.
        """
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
        return resolve(module, name)


def _make_pickle_module() -> types.ModuleType:
    """Returns a module object that `torch.load` accepts as `pickle_module`."""
    module = types.ModuleType("flashmd_legacy_pickle")
    module.Unpickler = _Unpickler  # type: ignore[attr-defined]
    module.load = pickle.load  # type: ignore[attr-defined]
    return module


PICKLE_MODULE: types.ModuleType = _make_pickle_module()


def load(
    f: str | os.PathLike[str] | BinaryIO,
    *,
    map_location: str | torch.device | Callable[..., Any] | None = None,
) -> Any:
    """Loads a pickled mlcg or FlashMD object.

    Args:
        f: Path or binary file object.
        map_location: Passed to `torch.load`.

    Returns:
        The unpickled object, with mlcg classes replaced by FlashMD classes.

    Raises:
        UnexpectedGlobalError: If the pickle references a disallowed global.
    """
    return torch.load(
        f,
        map_location=map_location,
        pickle_module=PICKLE_MODULE,
        weights_only=False,
    )
