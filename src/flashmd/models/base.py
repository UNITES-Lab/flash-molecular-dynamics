"""Contracts between the FlashMD engine and model plugins.

A plugin wraps one published model family. It loads upstream weights without
changing them, exposes the unmodified upstream module as its reference path,
and swaps in FlashMD execution when `optimize` is called with a flash config.

Two kinds of plugins exist:

* `Potential` maps positions to energies and forces. The engine integrates
  the dynamics (CGSchNet, MACE).
* `Generator` draws independent structures for a sequence. There is no time
  axis (BioEmu).
"""

import abc
from collections.abc import Iterator, Sequence
import dataclasses
import os
from typing import Any, ClassVar, NamedTuple, Self

import torch

from flashmd import config as config_lib


@dataclasses.dataclass(frozen=True)
class Units:
    """Physical units a plugin uses for its inputs and outputs.

    Attributes:
        energy: Energy unit, for example `"kcal/mol"` or `"eV"`.
        length: Length unit, for example `"angstrom"`.
    """

    energy: str
    length: str = "angstrom"


class PotentialOutput(NamedTuple):
    """Energies and forces for a batch of replicas.

    Attributes:
        energy: Float tensor of shape [R], one total energy per replica.
        forces: Float tensor of shape [R, N, 3], the negative gradient of
            `energy` with respect to positions.
    """

    energy: torch.Tensor
    forces: torch.Tensor


class PreparedBatch(abc.ABC):
    """Replica-invariant state that a potential builds once per batch.

    Subclasses hold whatever does not change between steps: topology,
    element-dependent tables, precomputed embeddings, captured graphs. The
    engine treats the object as opaque apart from these two properties.
    """

    @property
    @abc.abstractmethod
    def num_replicas(self) -> int:
        """Number of replicas R in the batch."""

    @property
    @abc.abstractmethod
    def num_atoms(self) -> int:
        """Number of particles N per replica (atoms or beads)."""


class _Plugin(abc.ABC):
    """Behaviour shared by potentials and generators.

    Attributes:
        name: Registry name of the model family, for example `"mace"`.
        units: Units of the plugin's inputs and outputs.
    """

    name: ClassVar[str]
    units: ClassVar[Units]

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        source: str | os.PathLike[str],
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Loads published weights without modifying them.

        Args:
            source: Registry alias, Hugging Face repository id or local path.
            device: Device to place the model on.
            dtype: Optional dtype; `None` keeps the checkpoint dtype.

        Returns:
            A plugin running the reference path until `optimize` is called.
        """

    @abc.abstractmethod
    def optimize(self, config: config_lib.ExecutionConfig) -> None:
        """Switches execution according to `config`.

        Calling this with a reference config restores the upstream path.

        Args:
            config: The execution settings to apply.
        """

    @property
    @abc.abstractmethod
    def reference_module(self) -> torch.nn.Module:
        """The unmodified upstream module, used for parity checks."""


class Potential(_Plugin):
    """A model that maps positions to energies and forces."""

    @abc.abstractmethod
    def prepare(self, systems: Sequence[Any]) -> PreparedBatch:
        """Builds the replica-invariant state for a batch of systems.

        Args:
            systems: One entry per replica, in the plugin's native system
                type. All entries must share composition and topology.

        Returns:
            State reused by every subsequent `compute` call.
        """

    @abc.abstractmethod
    def compute(
        self, batch: PreparedBatch, positions: torch.Tensor
    ) -> PotentialOutput:
        """Evaluates energies and forces.

        Args:
            batch: State returned by `prepare`.
            positions: Float tensor of shape [R, N, 3] in `units.length`.

        Returns:
            Energies of shape [R] and forces of shape [R, N, 3].
        """


@dataclasses.dataclass(frozen=True)
class SampleRequest:
    """What a generator should draw.

    Attributes:
        sequence: Amino-acid sequence in one-letter code.
        num_samples: Total number of independent samples.
        seed: Base random seed; batch k uses `seed + first_index_of_batch`.
        batch_size: Samples per device batch; `None` lets the planner choose.
    """

    sequence: str
    num_samples: int
    seed: int = 0
    batch_size: int | None = None

    def __post_init__(self) -> None:
        """Validates the request."""
        if not self.sequence:
            raise ValueError("sequence must not be empty.")
        if self.num_samples < 1:
            raise ValueError(
                f"num_samples must be positive, got {self.num_samples}."
            )


class SampleBatch(NamedTuple):
    """One batch of generated structures.

    Attributes:
        start: Index of the first sample of the batch within the request.
        positions: Float tensor of shape [B, L, 3], residue positions.
        orientations: Float tensor of shape [B, L, 3, 3], residue frames.
    """

    start: int
    positions: torch.Tensor
    orientations: torch.Tensor


class Generator(_Plugin):
    """A model that draws independent structures for a sequence."""

    @abc.abstractmethod
    def sample(
        self,
        request: SampleRequest,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[SampleBatch]:
        """Draws samples batch by batch.

        Args:
            request: What to draw.
            generator: Optional random generator; seeds from `request` are
                used when `None`.

        Yields:
            Batches in order of `start`.
        """
