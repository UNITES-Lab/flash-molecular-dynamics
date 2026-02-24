import torch
import nvtx
from typing import Sequence, Any, List
from ..data.atomic_data import AtomicData
from ..data._keys import *


def _log_shape(logger, label: str, tensor, indent: int = 0):
    """Helper to log tensor shape if logger is available."""
    if logger is None:
        return
    prefix = "  " * indent
    if isinstance(tensor, torch.Tensor):
        logger.write(f"{prefix}{label}: shape={list(tensor.shape)}, dtype={tensor.dtype}\n")
    else:
        logger.write(f"{prefix}{label}: {type(tensor).__name__}\n")


class SumOut(torch.nn.Module):
    r"""Property pooling wrapper for models

    Parameters
    ----------
    models:
        Dictionary of predictors models keyed by their name attribute
    targets:
        List of prediction targets that will be pooled

    Example
    -------
    To combine SchNet force predictions with prior interactions:

    .. code-block:: python

        import torch
        from flashmd.models import (StandardSchNet, GradientsOut, SumOut,
                                    CosineCutoff, GaussianBasis)
        from flashmd.prior import HarmonicBonds, HarmonicAngles
        from flashmd.data._keys import FORCE_KEY, ENERGY_KEY

        bond_terms = GradientsOut(HarmonicBonds(bond_stats), FORCE_KEY)
        angle_terms = GradientsOut(HarmonicAngles(angle_stats), FORCE_KEY)
        cutoff = CosineCutoff()
        rbf = GaussianBasis(cutoff)
        energy_network = StandardSchNet(cutoff, rbf, [128])
        force_network = GradientsOut(energy_model, FORCE_KEY)

        models = torch.nn.ModuleDict{
                     "bonds": bond_terms,
                     "angles": angle_terms,
                     "SchNet": force_network
                 }
        full_model = SumOut(models, targets=[ENERGY_KEY, FORCE_KEY])


    """

    name: str = "SumOut"

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        targets: List[str] = None,
    ):
        super(SumOut, self).__init__()
        if targets is None:
            targets = [ENERGY_KEY, FORCE_KEY]
        self.targets = targets
        self.models = models

    @nvtx.annotate("SumOut.forward", color="blue")
    def forward(self, data: AtomicData) -> AtomicData:
        r"""Sums output properties from individual models into global
        property predictions

        Parameters
        ----------
        data:
            AtomicData instance whose 'out' field has been populated
            for each predictor in the model. For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },

            ...
            )

        Returns
        -------
        data:
            AtomicData instance with updated 'out' field that now contains
            prediction target keys that map to tensors that have summed
            up the respective contributions from each predictor in the model.
            For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    ENERGY_KEY: ...,
                    FORCE_KEY: ...,
            ...
            )

        """
        logger = getattr(data, '_shape_logger', None)
        if logger:
            logger.write("    [SumOut.forward] Aggregating model outputs:\n")
            logger.write(f"      Models: {list(self.models.keys())}\n")
            logger.write(f"      Targets: {self.targets}\n")

        for target in self.targets:
            data.out[target] = 0.00
        for name in self.models.keys():
            with nvtx.annotate(f"SumOut_model_{name}", color="purple"):
                data = self.models[name](data)
            with nvtx.annotate(f"SumOut_accumulate_{name}", color="gray"):
                for target in self.targets:
                    data.out[target] += data.out[name][target]

            # Log each model's output shapes
            if logger:
                logger.write(f"      [{name}] output:\n")
                for target in self.targets:
                    if name in data.out and target in data.out[name]:
                        _log_shape(logger, f"        {target}", data.out[name][target])

        # Log final aggregated output
        if logger:
            logger.write("      [Aggregated] output:\n")
            for target in self.targets:
                _log_shape(logger, f"        {target}", data.out[target])

        return data

    def neighbor_list(self, **kwargs):
        nl = {}
        for _, model in self.models.items():
            nl.update(**model.neighbor_list(**kwargs))
        return nl


class EnergyOut(torch.nn.Module):
    r"""Extractor for energy computed via a flashmd compatible model.

    Parameters
    ----------
    model:
        model whose target should be extyracted
    targets:
        List of prediction targets that will be extracted

    """

    name: str = "EnergyOut"

    def __init__(
        self,
        model: torch.nn.Module,
        targets: List[str] = None,
    ):
        super().__init__()
        if targets is None:
            targets = ["enegy"]
        self.targets = targets
        self.model = model
        self.name = self.model.name

    def forward(self, data: AtomicData) -> AtomicData:
        data = self.model(data)
        for target in self.targets:
            data.out[target] = data.out[self.name][target]
        return data


class GradientsOut(torch.nn.Module):
    r"""Gradient wrapper for models.

    Parameters
    ----------
    targets:
        The gradient targets to produce from a model output. These can be any
        of the gradient properties referenced in `flashmd.data._keys`.
        At the moment only forces are implemented.

    Example
    -------
        To predict forces from an energy model, one would supply a model that
        predicts a scalar atom property (an energy) and specify the `FORCE_KEY`
        in the targets.
    """

    _targets = {FORCE_KEY: ENERGY_KEY}

    def __init__(self, model: torch.nn.Module, targets: str = FORCE_KEY):
        super(GradientsOut, self).__init__()
        self.model = model
        self.name = self.model.name
        self.targets = []
        if isinstance(targets, str):
            self.targets = [targets]
        elif isinstance(targets, Sequence):
            self.targets = targets
        assert any(
            [k in GradientsOut._targets for k in self.targets]
        ), f"targets={self.targets} should be any of {GradientsOut._targets}"

    @nvtx.annotate("GradientsOut.forward", color="red")
    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the gradient layer.

        Parameters
        ----------
        data:
            AtomicData instance

        Returns
        -------
        data:
            Updated AtomicData instance, where the "out" field has
            been populated with the base predictions of the model (eg,
            the energy as well as the target predictions produced through
            gradient operations.
        """
        logger = getattr(data, '_shape_logger', None)
        if logger:
            logger.write(f"    [GradientsOut.forward] Computing gradients for {self.name}:\n")
            _log_shape(logger, "      data.pos (input)", data.pos)

        with nvtx.annotate("GradientsOut_enable_grad", color="gray"):
            data.pos.requires_grad_(True)

        with nvtx.annotate(f"GradientsOut_model_{self.name}", color="lime"):
            data = self.model(data)

        if FORCE_KEY in self.targets:
            with nvtx.annotate("GradientsOut_extract_energy", color="yellow"):
                if self.name == "SumOut":
                    y = data.out[ENERGY_KEY]
                else:
                    y = data.out[self.name][ENERGY_KEY]

            if logger:
                _log_shape(logger, "      energy (for autograd)", y)

            with nvtx.annotate("GradientsOut_autograd_backward", color="orange"):
                dy_dr = torch.autograd.grad(
                    y.sum(),
                    data.pos,
                    # grad_outputs=torch.ones_like(y),
                    # retain_graph=self.training,
                    create_graph=self.training,
                )[0]

            if logger:
                _log_shape(logger, "      dy_dr (gradient)", dy_dr)

            with nvtx.annotate("GradientsOut_assign_forces", color="gray"):
                if self.name == "SumOut":
                    data.out[FORCE_KEY] = -dy_dr
                else:
                    data.out[self.name][FORCE_KEY] = -dy_dr

            if logger:
                _log_shape(logger, "      forces (output)", -dy_dr)

            # assert not torch.any(torch.isnan(dy_dr)), f"nan in {self.name}"

        with nvtx.annotate("GradientsOut_detach_pos", color="gray"):
            data.pos = data.pos.detach()

        return data

    def neighbor_list(self, **kwargs: Any):
        return self.model.neighbor_list(**kwargs)
