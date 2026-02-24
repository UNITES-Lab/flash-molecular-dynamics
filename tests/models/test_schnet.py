import torch
from torch_geometric.data.collate import collate
import pytest
from typing import List
import warnings

from flashmd.models.schnet import StandardSchNet
from flashmd.models.radial_basis import GaussianBasis
from flashmd.models.gradients import GradientsOut
from flashmd.models.cutoff import IdentityCutoff, CosineCutoff
from flashmd.data.atomic_data import AtomicData
from flashmd.data._keys import ENERGY_KEY, FORCE_KEY
from flashmd.neighbor_list.neighbor_list import make_neighbor_list


def _make_test_data(n_atoms=10, n_mols=3, cutoff=5.0):
    """Create test data without depending on ASE/Topology."""
    data_list = []
    for _ in range(n_mols):
        pos = torch.randn(n_atoms, 3).float() * 2.0
        atom_types = torch.randint(0, 5, (n_atoms,))

        # Build fully-connected neighbor list
        src = []
        dst = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        index_mapping = torch.tensor([src, dst], dtype=torch.long)
        nl = {
            "SchNet": make_neighbor_list(
                tag="SchNet", order=2, index_mapping=index_mapping
            )
        }

        data = AtomicData.from_points(
            pos=pos,
            atom_types=atom_types,
            neighbor_list=nl,
        )
        data_list.append(data)

    collated_data, _, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )
    return data_list, collated_data


standard_basis = GaussianBasis(cutoff=5)
standard_cutoff = IdentityCutoff(cutoff_lower=0, cutoff_upper=5)

# prepare test data
data_list, collated_data = _make_test_data()
force_shape = collated_data.pos.shape
energy_shape = torch.Size([len(data_list)])


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (
            GaussianBasis(IdentityCutoff(1, 5)),
            CosineCutoff(0, 5),
            UserWarning,
        ),
    ],
)
def test_cutoff_warning(basis, cutoff, expected_warning):
    with pytest.warns(expected_warning):
        StandardSchNet(basis, cutoff, [128, 128])


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (GaussianBasis(IdentityCutoff(0, 5)), CosineCutoff(0, 5), None),
    ],
)
def test_cutoff_warning_None(basis, cutoff, expected_warning):
    with warnings.catch_warnings(record=True):
        StandardSchNet(basis, cutoff, [128, 128])


def test_minimum_interaction_block():
    with pytest.raises(ValueError):
        StandardSchNet(
            standard_basis,
            standard_cutoff,
            [128, 128],
            num_interactions=-1,
        )


@pytest.mark.parametrize(
    "out_keys, expected_shapes",
    [
        (
            [ENERGY_KEY, FORCE_KEY],
            [energy_shape, force_shape],
        )
    ],
)
def test_prediction(out_keys, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the corresponding shapes of the outputs are correct given the
    requested gradient targets.
    """
    test_schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    model = GradientsOut(test_schnet, targets=FORCE_KEY).double()
    result = model(collated_data)
    assert len(result.out) != 0
    assert "SchNet" in result.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in result.out[model.name].keys()
        assert result.out[model.name][key].shape == shape
