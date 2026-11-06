"""Regression test: --disable_optim must turn off torch.compile."""

from flashmd.simulation import base


def test_compile_model_attribute_controls_attach_flag():
    simulation = base._Simulation.__new__(base._Simulation)
    simulation._compile_model_flag = True
    simulation.compile_model = False
    assert simulation._compile_model_flag is False
    assert simulation.compile_model is False
