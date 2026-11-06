"""Tests for flashmd.models.radial_basis.exp_normal."""

import math

import pytest
import torch

from flashmd.models import cutoff as cutoff_lib
from flashmd.models.radial_basis import ExpNormalBasis


def _reference(dist, cutoff_upper, num_rbf):
    """Evaluates the PhysNet basis directly from its definition."""
    start = math.exp(-cutoff_upper)
    means = torch.linspace(start, 1.0, num_rbf, dtype=torch.float64)
    beta = (2.0 / num_rbf * (1.0 - start)) ** -2
    alpha = 5.0 / cutoff_upper
    d = dist.to(torch.float64).unsqueeze(-1)
    envelope = 0.5 * (torch.cos(d * math.pi / cutoff_upper) + 1.0)
    envelope = envelope * (d < cutoff_upper)
    return envelope * torch.exp(-beta * (torch.exp(-alpha * d) - means) ** 2)


@pytest.mark.parametrize("trainable", [True, False])
def test_matches_definition(trainable):
    basis = ExpNormalBasis(cutoff=5.0, num_rbf=16, trainable=trainable)
    dist = torch.linspace(0.0, 6.0, 31)
    torch.testing.assert_close(
        basis(dist).to(torch.float64),
        _reference(dist, 5.0, 16),
        rtol=1e-5,
        atol=1e-6,
    )


def test_attribute_names_match_mlcg():
    basis = ExpNormalBasis(cutoff=5.0, num_rbf=8, trainable=True)
    assert isinstance(basis.cutoff, cutoff_lib.CosineCutoff)
    assert basis.num_rbf == 8
    assert basis.trainable is True
    assert basis.alpha == pytest.approx(1.0)
    assert {name for name, _ in basis.named_parameters()} == {"means", "betas"}

    frozen = ExpNormalBasis(cutoff=5.0, num_rbf=8, trainable=False)
    assert {name for name, _ in frozen.named_buffers()} >= {"means", "betas"}
    assert not list(frozen.parameters())


def test_reset_parameters_restores_defaults():
    basis = ExpNormalBasis(cutoff=5.0, num_rbf=8)
    expected = basis.means.detach().clone()
    with torch.no_grad():
        basis.means.add_(1.0)
    basis.reset_parameters()
    torch.testing.assert_close(basis.means.detach(), expected)


def test_rejects_bad_cutoff():
    with pytest.raises(TypeError):
        ExpNormalBasis(cutoff="5")
    with pytest.raises(ValueError):
        ExpNormalBasis(cutoff=cutoff_lib.CosineCutoff(10.0, 0.0))
