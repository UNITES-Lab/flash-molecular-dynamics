"""Execution settings shared by every model plugin.

`ExecutionConfig` is the single switchboard for how a model runs. It replaces
the import-time environment variables of FlashSchNet (`MLCG_USE_*`), the
`MACE_DISABLE_TRITON_GEOMETRY` variable of FlashMACE and the sampling flags of
FlashBioEmu.

The default is strict: the model runs in the dtype of its checkpoint with TF32
and reduced-precision submodules off, so a FlashMD result can be compared with
the upstream reference at matched precision. Faster numerics are opt-in.

Typical usage:

    config = config_lib.ExecutionConfig()
    reference = config_lib.ExecutionConfig.reference()
    fast = config_lib.ExecutionConfig(dtype=torch.float32, allow_tf32=True)
"""

from collections.abc import Mapping
import dataclasses
import enum
import types

import torch


class Backend(enum.StrEnum):
    """Which implementation of a model runs."""

    REFERENCE = "reference"
    """The unmodified upstream module, eager PyTorch."""

    FLASH = "flash"
    """FlashMD kernels and execution paths."""


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    """How a model executes.

    Attributes:
        backend: Reference or flash execution. With `Backend.REFERENCE` every
            FlashMD kernel, graph capture and compilation is off, whatever the
            other fields say.
        dtype: Floating-point dtype for model parameters and activations.
            `None` keeps the dtype stored in the checkpoint.
        allow_tf32: Whether float32 matrix multiplications may use TF32 tensor
            cores. Applied only inside FlashMD calls, never globally.
        reduced_precision: Whether plugins may run selected submodules in
            float16 or bfloat16 (for example the CGSchNet filter networks).
        cuda_graphs: Whether repeated steps are captured once and replayed.
        compile: Whether plugins may apply `torch.compile` to submodules.
        max_batch: Upper bound on replicas or samples per device batch.
            `None` lets the memory planner choose.
        kernels: Per-kernel overrides keyed by kernel name. A `False` entry
            disables that kernel; kernels not listed are enabled.
    """

    backend: Backend = Backend.FLASH
    dtype: torch.dtype | None = None
    allow_tf32: bool = False
    reduced_precision: bool = False
    cuda_graphs: bool = True
    compile: bool = False
    max_batch: int | None = None
    kernels: Mapping[str, bool] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates fields and freezes the kernel overrides."""
        if not isinstance(self.backend, Backend):
            object.__setattr__(self, "backend", Backend(self.backend))
        if self.dtype is not None and not self.dtype.is_floating_point:
            raise ValueError(f"dtype must be floating point, got {self.dtype}.")
        if self.max_batch is not None and self.max_batch < 1:
            raise ValueError(
                f"max_batch must be positive, got {self.max_batch}."
            )
        object.__setattr__(
            self, "kernels", types.MappingProxyType(dict(self.kernels))
        )

    @classmethod
    def reference(cls, dtype: torch.dtype | None = None) -> "ExecutionConfig":
        """Returns a config that runs the unmodified upstream module.

        Args:
            dtype: Optional dtype override; `None` keeps the checkpoint dtype.

        Returns:
            A config with the reference backend and every optimization off.
        """
        return cls(
            backend=Backend.REFERENCE,
            dtype=dtype,
            cuda_graphs=False,
            compile=False,
        )

    @property
    def is_reference(self) -> bool:
        """Whether this config selects the reference backend."""
        return self.backend is Backend.REFERENCE

    def kernel_enabled(self, name: str) -> bool:
        """Returns whether the named FlashMD kernel may run.

        Args:
            name: Kernel name, for example `"schnet.fused_cfconv"`.

        Returns:
            `False` under the reference backend or when `kernels[name]` is
            `False`, otherwise `True`.
        """
        if self.is_reference:
            return False
        return self.kernels.get(name, True)

    def replace(self, **changes: object) -> "ExecutionConfig":
        """Returns a copy with the given fields changed.

        Args:
            **changes: Field names and their new values.

        Returns:
            A new `ExecutionConfig`.
        """
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]
