import torch
import pytest
from flashmd.models.radial_basis import GaussianBasis
from flashmd.models.cutoff import IdentityCutoff


def test_cutoff_error_raise():
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        GaussianBasis(cutoff=IdentityCutoff(cutoff_lower=10, cutoff_upper=0))


def test_cutoff_defaults():
    cutoff_upper = 10
    basis = GaussianBasis(cutoff=cutoff_upper)
    assert isinstance(basis.cutoff, IdentityCutoff)
    assert basis.cutoff.cutoff_lower == 0
    assert basis.cutoff.cutoff_upper == cutoff_upper
