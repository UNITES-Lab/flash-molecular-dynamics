from typing import List, Tuple, Union
import torch
import numpy as np
import nvtx

from ..data.atomic_data import AtomicData
from ..data._keys import MASS_KEY, VELOCITY_KEY, POSITIONS_KEY
from .base import _Simulation
from .langevin import LangevinSimulation


class NVESimulation(_Simulation):
    r"""NVE (microcanonical) simulation using Velocity Verlet integrator.

    The Velocity Verlet algorithm:

    .. math::
        v_{t+1/2} &= v_t + \frac{\Delta t}{2m} F(x_t) \\
        x_{t+1} &= x_t + \Delta t \cdot v_{t+1/2} \\
        F_{t+1} &= -\nabla U(x_{t+1}) \\
        v_{t+1} &= v_{t+1/2} + \frac{\Delta t}{2m} F_{t+1}

    This is a symplectic, time-reversible integrator that conserves energy
    in the absence of numerical errors.

    Initial velocities are sampled from Maxwell-Boltzmann distribution.
    """

    @nvtx.annotate("NVESimulation.timestep", color="cyan")
    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        v = data[VELOCITY_KEY]
        x = data[POSITIONS_KEY]
        m = data[MASS_KEY][:, None]

        # Velocity Verlet
        v_half = v + 0.5 * self.dt * forces / m
        x_new = x + self.dt * v_half
        data[POSITIONS_KEY] = x_new

        potential, forces_new = self.calculate_potential_and_forces(data)

        v_new = v_half + 0.5 * self.dt * forces_new / m
        data[VELOCITY_KEY] = v_new

        return data, potential, forces_new

    def _attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        super()._attach_configurations(configurations, beta)

        if VELOCITY_KEY not in self.initial_data:
            self.initial_data[VELOCITY_KEY] = (
                LangevinSimulation.sample_maxwell_boltzmann(
                    self.beta.repeat_interleave(self.n_atoms),
                    self.initial_data[MASS_KEY],
                ).to(self.dtype)
            )
        assert self.initial_data[VELOCITY_KEY].shape == self.initial_data[POSITIONS_KEY].shape

    def _set_up_simulation(self, overwrite: bool = False):
        super()._set_up_simulation(overwrite)
        if self.save_energies:
            self.simulated_kinetic_energies = torch.zeros(self._save_size, self.n_sims)
        else:
            self.simulated_kinetic_energies = None

    def save(self, data: AtomicData, forces: torch.Tensor, potential: torch.Tensor, t: int):
        super().save(data, forces, potential, t)
        if self.save_energies:
            v = data[VELOCITY_KEY].view(-1, self.n_atoms, self.n_dims)
            m = data[MASS_KEY].view(self.n_sims, self.n_atoms)
            ke = 0.5 * torch.sum(m[:, :, None] * v**2, dim=(1, 2))
            save_ind = (t // self.save_interval) - self._npy_file_index * self._save_size
            self.simulated_kinetic_energies[save_ind, :] = ke

    def write(self):
        if self.save_energies:
            key = self._get_numpy_count()
            ke_export = self._swap_and_export(self.simulated_kinetic_energies)
            np.save(f"{self.filename}_kineticenergy_{key}.npy", ke_export)
            self.simulated_kinetic_energies = torch.zeros(self._save_size, self.n_sims)
        super().write()

    def reshape_output(self):
        super().reshape_output()
        if self.save_energies:
            self.simulated_kinetic_energies = self._swap_and_export(
                self.simulated_kinetic_energies
            )


NVESimulation.__doc__ += _Simulation.__doc__
