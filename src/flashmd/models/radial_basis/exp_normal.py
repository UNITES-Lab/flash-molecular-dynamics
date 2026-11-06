"""Exponential-normal radial basis used by the published CGSchNet model.

Ported from `mlcg.nn.radial_basis.exp_normal` (MIT, Clementi Group; see
THIRD_PARTY_NOTICES.md). Attribute names match the original class exactly,
because published checkpoints pickle this module and restore its state
without calling `__init__`.
"""

import torch
from torch import nn

from flashmd.models import cutoff as cutoff_lib
from flashmd.models.radial_basis import base


class ExpNormalBasis(base._RadialBasis):
    r"""Exponential-normal radial basis functions (PhysNet).

    Each function has the form

    .. math::

        f_n(r) = f_{cut}(r) \exp\left[-\beta_n
        \left(e^{\alpha (r_{low} - r)} - \mu_n\right)^2\right],
        \qquad \alpha = \frac{5}{r_{high} - r_{low}}.

    Attributes:
        cutoff: Smooth cutoff envelope applied to every function.
        num_rbf: Number of basis functions.
        trainable: Whether `means` and `betas` are parameters or buffers.
        alpha: Distance rescaling factor.
        means: Float tensor of shape [num_rbf], centers in rescaled space.
        betas: Float tensor of shape [num_rbf], inverse widths.
    """

    cutoff: cutoff_lib._Cutoff
    num_rbf: int
    trainable: bool
    alpha: float
    means: torch.Tensor
    betas: torch.Tensor

    def __init__(
        self,
        cutoff: float | cutoff_lib._Cutoff,
        num_rbf: int = 50,
        trainable: bool = True,
    ) -> None:
        """Initializes the basis.

        Args:
            cutoff: Upper cutoff in angstrom, which selects a cosine cutoff
                from 0, or an explicit cutoff module.
            num_rbf: Number of basis functions.
            trainable: Whether centers and widths are optimized.

        Raises:
            TypeError: If `cutoff` is neither a number nor a cutoff module.
        """
        super().__init__()
        if isinstance(cutoff, (float, int)):
            self.cutoff = cutoff_lib.CosineCutoff(0, cutoff)
        elif isinstance(cutoff, cutoff_lib._Cutoff):
            self.cutoff = cutoff
        else:
            raise TypeError(
                f"Supplied cutoff {cutoff} is neither a number nor a _Cutoff."
            )
        self.check_cutoff()

        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 5.0 / (self.cutoff.cutoff_upper - self.cutoff.cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the default centers and widths (Unke and Meuwly, 2019)."""
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff.cutoff_upper + self.cutoff.cutoff_lower
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        """Resets centers and widths to their initial values."""
        means, betas = self._initial_params()
        with torch.no_grad():
            self.means.copy_(means)
            self.betas.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Expands distances in the basis.

        Args:
            dist: Float tensor of shape [E], pairwise distances.

        Returns:
            Float tensor of shape [E, num_rbf].
        """
        dist = dist.unsqueeze(-1)
        return self.cutoff(dist) * torch.exp(
            -self.betas
            * (
                torch.exp(self.alpha * (-dist + self.cutoff.cutoff_lower))
                - self.means
            )
            ** 2
        )
