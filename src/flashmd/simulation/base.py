# This code is adapted from https://github.com/coarse-graining/cgnet
# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm, Andreas Kraemer, Clark Templeton, Iryna Zaporozhets

import logging
import warnings
from typing import List, Optional, Tuple, Union, Callable
import torch
import nvtx
import numpy as np
from torch_geometric.data.collate import collate
import os
import time
from copy import deepcopy
from jsonargparse.typing import Path_fr

from .logging import logger
from ..utils import tqdm

from ..data.atomic_data import AtomicData
from ..data._keys import (
    ENERGY_KEY,
    FORCE_KEY,
    MASS_KEY,
    VELOCITY_KEY,
    POSITIONS_KEY,
)
from .specialize_prior import condense_all_priors_for_simulation
from .torch_compile_warning import (
    torch_compile_waring,
    force_torch_compile_warning,
)


# Physical Constants
KBOLTZMANN = 1.38064852e-23  # Boltzmann's constant in Joules/Kelvin
AVOGADRO = 6.022140857e23  # Dimensionaless Avogadro's number
JPERKCAL = 4184  # Ratio of Joules/kilocalorie


class _Simulation(object):
    r"""
    Parameters
    ----------
    dt : float, default=5e-4
        The integration time step for the dynamics. The simulation code is unit agnostic
        and will use the same distance and energy units that the network and initial configurations.
        However, for the two other parameters in a simulation, time and mass, only one of them is unit-agnostic:
        **if you provide the time thinking in a particular unit, the unit of mass is defined, and
        viceversa. Be aware that only either mass or time can be provided in custom units**.

        The following relation must be satisfied by the units to ensure consistency:

        .. math::

            [\text{Energy}] = [\text{Mass}]\frac{[\text{Length}]^2}{[\text{Time}]^2}

        For example, if you are using kcal/mol for the energy, Angstrom from the distance and time
        in picoseconds, the masses will need to be provided in units of AMU/418.39.
    save_forces : bool, default=False
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_energies : bool, default=False
        Whether to save potential at the same saved interval as the simulation
        coordinates
    save_force_components: bool, default=False
            Whether to save model energy components at the same saved interval as the simulation
        coordinates.
    save_energy_components: bool, default=False
        Whether to save model energy components at the same saved interval as the simulation
        coordinates.
    force_components: list[str] | None, default= None - list of keys to be saved. Only used when
        save_force_components is True.
    energy_components: list[str] | None, default= None - list of keys to be saved. Only used when
        save_energy_components is True.
    create_checkpoints: bool, default=False
        Save the atomic data object so it can be reloaded in. Overwrites previous object.
    read_checkpoint_file: [str,bool], default=None
        Whether to read in checkpoint file from. Can specify explicit file path or try to infer from self.filename
    n_timesteps : int, default=100
        The length of the simulation in simulation timesteps
    save_interval : int, default=10
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int, default=233
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : str, default='cpu'
        Device upon which simulation computation will be carried out
    dtype : str, default='single'
        precision to run the simulation with (single or double)
    export_interval : int, default=None
        Interval at which .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_energies
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
        If None, export_interval will be set to n_timesteps to output one file
        for the entire simulation
    log_interval : int, default=None
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : str, default='write'
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : str, default=None
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.
    add_timestamp : bool, default=False
        If True, appends a timestamp (YYYYMMDD_HHMMSS) to the filename to
        avoid conflicts when running repeated simulations.
    output_dir : str, default="./outputs"
        Directory where output files (logs, checkpoints, coordinates) are saved.
        The directory will be created if it doesn't exist.
    specialize_priors: bool, default=False
        use optimized version of the priors for a particular molecule
    dtype : str, default='single'
        precision to run the simulation with (single or double)
    sim_subroutine :
        Optional subroutine to run at at the interval specified by
        subroutine_interval after simulation updates. The subroutine should
        take only the internal collated `AtomicData` instance as an argument.
    sim_subroutine_interval :
        Specifies the interval, in simulation steps, between successive calls to
        the subroutine, if specified.
    save_subroutine :
        Specifies additional saving procedures for extra information at the
        same interval as export_interval. The subroutine should take only the
        internal collated `AtomicData` and the current timestep // save_interval as
        arguments.
    compile:
        If set to True, model will be compiled using torch.compile during simulation
        setup. Uses dynamic=True mode.
    compile_mode:
        Specifies the compilation mode for torch.compile.
        See https://docs.pytorch.org/docs/stable/generated/torch.compile.html
        for more information.
    force_compile:
        If set to True, compilation will be forced even for single structure
        simulations. This should be used with extreme caution since it may
        expose known issues with scatter operations and may produce invalid
        outputs or NaN gradients.
    compile_model:
        If set to True, applies torch.compile to the model during model attachment
        (in _attach_model) with mode="reduce-overhead" and dynamic=False. This is
        applied AFTER GPTQ quantization and AFTER setting requires_grad=False.
        This is different from the `compile` option which is applied later during
        simulation setup with dynamic=True.
    profile_start_step: int, optional
        Step at which to start CUDA profiling. Use with nsys --capture-range=cudaProfilerApi.
        If None, profiling captures from the beginning (default nsys behavior).
    profile_end_step: int, optional
        Step at which to stop CUDA profiling. Use with nsys --capture-range=cudaProfilerApi.
        If None, profiling captures until the end.
    print_shape: bool, default=False
        Whether to print tensor shapes at key computation steps. Shapes are logged to
        a file named "{filename}_print_shape.log". Useful for debugging and understanding
        the computational flow.
    print_shape_steps: int, default=3
        Number of timesteps to log shapes for. After this many steps, shape logging stops
        to avoid large log files.
    dump_neighbor_list: bool, default=False
        Whether to save neighbor lists (edge_index) at every timestep to a .pt file.
        The file is saved as "{filename}_neighbor_list.pt" and contains a list of
        edge_index tensors (one per timestep), stored on CPU.
    dump_neighbor_list_last_n: int, optional
        If specified, only keep the last N timesteps of neighbor lists (uses a
        circular buffer). Useful for dumping only the last N steps of a long
        simulation without excessive memory usage.
    gptq: str, optional
        GPTQ quantization mode for filter networks. Supported values:
        - None: No quantization (default)
        - "w16a16": FP16 weights and activations using Triton kernels
        IMPORTANT: When enabled, requires CUDA and Triton. No fallback provided.
    """

    def __init__(
        self,
        dt: float = 5e-4,
        save_forces: bool = False,
        save_energies: bool = False,
        save_force_components: bool = False,
        save_energy_components: bool = False,
        force_components: list[str] | str | None = None,
        energy_components: list[str] | str | None = None,
        n_timesteps: int = 100,
        save_interval: int = 10,
        create_checkpoints: bool = False,
        read_checkpoint_file: Union[Path_fr, bool] = None,
        random_seed: Optional[int] = 233,
        device: str = "cpu",
        dtype: str = "single",
        export_interval: Optional[int] = None,
        log_interval: Optional[int] = None,
        log_type: str = "write",
        filename: Optional[str] = None,
        add_timestamp: bool = False,
        output_dir: str = "./outputs",
        specialize_priors: bool = False,
        tqdm_refresh: float = 10,
        sim_subroutine: Optional[Callable] = None,
        sim_subroutine_interval: Optional[int] = None,
        save_subroutine: Optional[Callable] = None,
        compile: bool = False,
        compile_mode: str = "default",
        force_compile: bool = False,
        compile_model: bool = True,
        profile_start_step: Optional[int] = None,
        profile_end_step: Optional[int] = None,
        print_shape: bool = False,
        print_shape_steps: int = 3,
        dump_neighbor_list: bool = False,
        dump_neighbor_list_last_n: Optional[int] = None,
        gptq: Optional[str] = "w16a16",
    ):
        self.model = None
        self.gptq = gptq
        # Timing attributes for throughput measurement (second half of simulation)
        self._warmup_end_time = None
        self._simulation_end_time = None
        self._post_warmup_steps = 0
        if gptq is not None and gptq not in ["w16a16"]:
            raise ValueError(
                f"Unsupported GPTQ mode: {gptq}. Supported: 'w16a16'"
            )
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.print_shape = print_shape
        self.print_shape_steps = print_shape_steps
        self._shape_log_file = None
        self._shape_log_step_count = 0
        self.dump_neighbor_list = dump_neighbor_list
        self.dump_neighbor_list_last_n = dump_neighbor_list_last_n
        self._neighbor_list_dump = []  # List to store edge_index at each timestep
        self.initial_data = None
        self.specialize_priors = specialize_priors
        self.save_forces = save_forces
        self.save_energies = save_energies
        self.save_force_components = save_force_components
        self.save_energy_components = save_energy_components

        if force_components is None:
            self.force_components = None
        elif isinstance(force_components, str):
            self.force_components = {force_components: None}
        else:
            self.force_components = {key: None for key in force_components}

        if energy_components is None:
            self.energy_components = None
        elif isinstance(energy_components, str):
            self.energy_components = {energy_components: None}
        else:
            self.energy_components = {key: None for key in energy_components}

        self.n_timesteps = n_timesteps
        self.save_interval = save_interval
        self.dt = dt
        if dtype == "single":
            self.dtype = torch.float32
        elif dtype == "double":
            self.dtype = torch.float64

        self.device = torch.device(device)
        if export_interval is None:
            self.export_interval = self.n_timesteps
        else:
            self.export_interval = export_interval
        self.log_interval = log_interval
        self.create_checkpoints = create_checkpoints
        self.read_checkpoint_file = read_checkpoint_file
        if self.read_checkpoint_file == False:
            self.read_checkpoint_file = None

        if log_type not in ["print", "write"]:
            raise ValueError("log_type can be either 'print' or 'write'")
        self.log_type = log_type
        self.output_dir = output_dir
        if filename is not None:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            # Build filename with optional timestamp
            if add_timestamp:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"
            # Prepend output directory to filename
            self.filename = os.path.join(output_dir, filename)
        else:
            self.filename = filename
        self.sim_subroutine = sim_subroutine
        self.sim_subroutine_interval = sim_subroutine_interval
        self.save_subroutine = save_subroutine
        self.tqdm_refresh = tqdm_refresh

        # check to make sure input options for the simulation
        self.input_option_checks()

        if random_seed is None:
            self.rng = None
        else:
            self.rng = torch.Generator(device=self.device).manual_seed(
                random_seed
            )
        self.random_seed = random_seed
        self._simulated = False
        self.compile = compile
        self.compile_mode = compile_mode
        self.force_compile = force_compile
        self._compile_model_flag = compile_model

    def attach_model_and_configurations(
        self,
        model: torch.nn.Module,
        configurations: List[AtomicData],
        beta: Union[float, List[float]],
    ):
        if self.specialize_priors:
            model, configurations = condense_all_priors_for_simulation(
                model, configurations
            )
            logger.info("Prior models have been specialized for the simulation.")
        if self.filename is not None:
            torch.save(
                (deepcopy(model), deepcopy(configurations)),
                f"{self.filename}_specialized_model_and_config.pt",
            )
        self._attach_model(model)
        self._attach_configurations(configurations, beta=beta)

    def _attach_model(self, model: torch.nn.Module):
        """setup the model to use in the simulation

        Parameters
        ----------
        model : torch.nn.Module
            Trained model used to generate simulation data
        """
        self.model = model.eval().to(device=self.device, dtype=self.dtype)

        # Patch legacy models: add missing use_fused_tanh_linear attribute
        self._patch_legacy_model()

        # Apply GPTQ quantization if requested (before setting requires_grad=False)
        if self.gptq is not None:
            self._apply_gptq_quantization()

        # Set requires_grad=False AFTER GPTQ quantization
        # (GPTQ creates new parameters that need this setting)
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply torch.compile if compile_model is enabled
        # This happens AFTER GPTQ quantization and AFTER requires_grad=False
        if self._compile_model_flag:
            # Use dynamic=True because edge count varies during simulation
            # Use mode="default" since "reduce-overhead" (CUDA graphs) requires static shapes
            logger.info("Applying torch.compile to model (mode='default', dynamic=True)...")
            self.model = torch.compile(
                self.model, mode="default", dynamic=True
            )
            logger.info("torch.compile applied successfully")


    def _patch_legacy_model(self):
        """Patch legacy models that are missing new attributes."""
        try:
            from ..models.schnet import InteractionBlock
            for module in self.model.modules():
                if isinstance(module, InteractionBlock):
                    if not hasattr(module, 'use_fused_tanh_linear'):
                        module.use_fused_tanh_linear = False
        except ImportError:
            pass  # SchNet not available, skip

    def _apply_gptq_quantization(self):
        """Apply GPTQ quantization to the model's filter networks."""
        if self.gptq == "w16a16":
            from ..models.gptq import apply_gptq_w16a16_to_model, validate_gptq_w16a16

            logger.info(f"[GPTQ] Applying W16A16 quantization to filter networks...")
            self.model = apply_gptq_w16a16_to_model(self.model)

            # Validate that quantization was applied correctly (no fallback!)
            validate_gptq_w16a16(self.model)
            logger.info(f"[GPTQ] W16A16 quantization applied successfully")

            # Warm up Triton kernels (autotuning takes ~10s on first call)
            self._warmup_gptq_kernels()

    def _warmup_gptq_kernels(self):
        """Warm up Triton kernels to trigger autotuning before simulation."""
        import torch
        logger.info(f"[GPTQ] Warming up Triton kernels (this may take ~10 seconds)...")

        # Find a GPTQ filter network to warm up
        from ..models.gptq import GPTQW16A16FilterNetwork
        gptq_filter = None
        for module in self.model.modules():
            if isinstance(module, GPTQW16A16FilterNetwork):
                gptq_filter = module
                break

        if gptq_filter is None:
            logger.warning("[GPTQ] No GPTQ filter network found for warmup")
            return

        # Create dummy input with typical simulation size
        # Use a range of M values to trigger autotuning for all configurations
        K = gptq_filter.in_features
        device = gptq_filter.weight0.device

        # Warmup with a few different M values to cover autotuning
        for M in [100_000, 500_000, 900_000]:
            x = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
            try:
                y = gptq_filter(x)
                # Also trigger backward (for force computation)
                torch.autograd.grad(y.sum(), x)[0]
            except Exception as e:
                logger.warning(f"[GPTQ] Warmup failed for M={M}: {e}")

        torch.cuda.synchronize()
        logger.info(f"[GPTQ] Triton kernel warmup complete")

    def _attach_configurations(
        self,
        configurations: List[AtomicData],
        beta: Union[float, List[float]],
        overdamped: bool = False,
    ):
        """Setup the starting atomic configurations.
        Parameters
        ----------
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures for
            parallel simulations.
        beta:
            Desired temperature(s) of the simulation
        """
        self.validate_data_list(configurations)
        self.initial_data = self.collate(configurations).to(device=self.device)
        if not overdamped:
            self.initial_data[MASS_KEY] = self.initial_data[MASS_KEY].to(
                self.dtype
            )
        self.initial_data[POSITIONS_KEY] = self.initial_data[POSITIONS_KEY].to(
            self.dtype
        )
        self.n_sims = len(configurations)
        self.n_atoms = len(configurations[0].atom_types)
        self.n_dims = configurations[0].pos.shape[1]
        # Keep initial_pos_spread on GPU for faster comparison in save()
        self.initial_pos_spread = (
            torch.cat([data.pos.std(dim=1) for data in configurations])
            .max()
            .detach()
            .to(self.device)
        )

        # Load in checkpointed data values and then wipe to conserve space
        if self.checkpointed_data is not None:
            self.initial_data[VELOCITY_KEY] = self.checkpointed_data[
                VELOCITY_KEY
            ]
            self.initial_data[POSITIONS_KEY] = self.checkpointed_data[
                POSITIONS_KEY
            ]
            self.checkpointed_data = None

        if isinstance(beta, float):
            if beta <= 0:
                raise ValueError(
                    "Beta must be positive, but {} was supplied".format(beta)
                )
            self.beta = (
                torch.tensor(self.n_sims * [beta])
                .to(self.device)
                .to(self.dtype)
            )
        else:
            self.beta = torch.tensor(beta).to(self.device).to(self.dtype)
            if not all([b >= 0 for b in self.beta]):
                raise ValueError(
                    "All betas must be positive, but {} contains an illegal value.".format(
                        self.beta
                    )
                )
        self.beta = self.beta.to(self.dtype)
        assert all([torch.isfinite(b) for b in self.beta])
        assert len(self.beta) == len(configurations)

    @nvtx.annotate("Simulation.simulate", color="blue")
    def simulate(self, overwrite: bool = False, prof=None) -> np.ndarray:
        """Generates independent simulations.

        Parameters
        ----------
        overwrite :
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords :
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval
        """

        with nvtx.annotate("setup_simulation", color="yellow"):
            self._set_up_simulation(overwrite)
            data = deepcopy(self.initial_data)
            data = data.to(self.device)

        with nvtx.annotate("compile_model", color="orange"):
            self._apply_dynamic_compile(data)

        with nvtx.annotate("initial_force_calculation", color="green"):
            _, forces = self.calculate_potential_and_forces(data)

        if self.export_interval is not None:
            t_init = self.current_timestep * self.export_interval
        else:
            t_init = 0
        if t_init >= self.n_timesteps:
            raise ValueError(
                f"Simulation has already been running for {t_init} steps, which is larger than the target number of steps {self.n_timesteps}"
            )

        # Save initial checkpoint (step 0) if starting fresh
        if self.create_checkpoints and t_init == 0:
            initial_checkpoint = {
                POSITIONS_KEY: deepcopy(data[POSITIONS_KEY].detach()),
                VELOCITY_KEY: deepcopy(data[VELOCITY_KEY].detach()),
                "current_timestep": 0,
                "export_interval": self.export_interval,
                "save_interval": self.save_interval,
                "log_interval": self.log_interval,
            }
            torch.save(
                initial_checkpoint,
                "{}_checkpoint_init.pt".format(self.filename),
            )

        with nvtx.annotate("simulation_loop", color="purple"):
            for t in tqdm(
                range(t_init, self.n_timesteps),
                desc="Simulation timestep",
                mininterval=self.tqdm_refresh,
                initial=t_init,
                total=self.n_timesteps,
            ):
                # Start CUDA profiler at specified step
                if self.profile_start_step is not None and t == self.profile_start_step:
                    torch.cuda.cudart().cudaProfilerStart()

                # Record time at halfway point for throughput measurement (second half only)
                halfway_step = self.n_timesteps // 2
                if t == halfway_step and self._warmup_end_time is None:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    self._warmup_end_time = time.perf_counter()
                    # Reset peak memory stats for second-half measurement
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                with nvtx.annotate(f"timestep_{t}", color="cyan"):
                    # Log timestep header for shape logging
                    if self.print_shape and self._shape_log_file and self._shape_log_step_count <= self.print_shape_steps:
                        self._shape_log_file.write(f"\n{'='*80}\n")
                        self._shape_log_file.write(f"TIMESTEP {t}\n")
                        self._shape_log_file.write(f"{'='*80}\n")

                    # step forward in time
                    with nvtx.annotate("integrator_timestep", color="green"):
                        data, potential, forces = self.timestep(data, forces)

                    # save to arrays if relevant
                    if (t + 1) % self.save_interval == 0:
                        with nvtx.annotate("save_arrays", color="red"):
                            # save arrays
                            self.save(
                                data=data,
                                forces=forces,
                                potential=potential,
                                t=t,
                            )

                        # write numpys to file if relevant; this can be indented here because
                        # it only happens when time points are also recorded
                        if self.export_interval is not None:
                            if (t + 1) % self.export_interval == 0:
                                with nvtx.annotate("write_to_disk", color="magenta"):
                                    self.write()
                                if self.save_subroutine is not None:
                                    self.save_subroutine(
                                        data, (t + 1) // self.save_interval
                                    )

                        # log if relevant; this can be indented here because
                        # it only happens when time when time points are also recorded
                        if self.log_interval is not None:
                            if int((t + 1) % self.log_interval) == 0:
                                self.log((t + 1) // self.save_interval)

                    if self.sim_subroutine != None:
                        if (t + 1) % self.sim_subroutine_interval == 0:
                            with nvtx.annotate("sim_subroutine", color="yellow"):
                                data = self.sim_subroutine(data)

                    # Store final potential for summary (keep on GPU to avoid sync every step)
                    self._final_potential = potential.detach()

                    # reset data outputs to collect the new forces/energies
                    data.out = {}
                    if prof:
                        prof.step()

                    # Increment shape log step counter
                    if self.print_shape and self._shape_log_file:
                        self._shape_log_step_count += 1
                        if self._shape_log_step_count == self.print_shape_steps:
                            self._shape_log_file.write(f"\n{'='*80}\n")
                            self._shape_log_file.write(f"Shape logging stopped after {self.print_shape_steps} timesteps.\n")
                            self._shape_log_file.write(f"{'='*80}\n")
                            self._shape_log_file.flush()

                # Stop CUDA profiler at specified step
                if self.profile_end_step is not None and t == self.profile_end_step:
                    torch.cuda.cudart().cudaProfilerStop()

        # Record end time and peak memory for throughput measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._simulation_end_time = time.perf_counter()
        halfway_step = self.n_timesteps // 2
        self._post_warmup_steps = self.n_timesteps - halfway_step
        # Record peak memory for second half (after reset at halfway)
        if torch.cuda.is_available():
            self._second_half_peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            self._second_half_peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GB
        else:
            self._second_half_peak_memory_allocated = 0
            self._second_half_peak_memory_reserved = 0

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t + 1) % self.export_interval > 0:
                with nvtx.annotate("final_write", color="magenta"):
                    self.write()

        # Always print summary at the end
        self.summary()

        self.reshape_output()

        self._simulated = True

        # Close shape log file
        if self.print_shape and self._shape_log_file:
            self._shape_log_file.write(f"\nSimulation completed. Total timesteps: {self.n_timesteps}\n")
            self._shape_log_file.close()
            self._shape_log_file = None

        # Save neighbor list dump to .pt file
        if self.dump_neighbor_list and self._neighbor_list_dump and self.filename is not None:
            nl_dump_path = f"{self.filename}_neighbor_list.pt"
            torch.save(self._neighbor_list_dump, nl_dump_path)
            logger.info(f"Neighbor list saved to: {nl_dump_path}")
            logger.info(f"  Total timesteps saved: {len(self._neighbor_list_dump)}")

        return

    def _run_and_check(self, data):
        """
        Runs the model, computes potential and forces, and validates numerical correctness.
        """
        potential, forces = self.calculate_potential_and_forces(data)

        if torch.isnan(forces).any() or torch.isinf(forces).any():
            raise RuntimeError("Invalid values detected in computed forces.")
        if torch.isnan(potential).any() or torch.isinf(potential).any():
            raise RuntimeError("Invalid values detected in computed potential.")

        return

    def _apply_dynamic_compile(self, data):
        """Compiles the model using torch.compile with dynamic=True for simulation.

        This is separate from the compile_model parameter which applies torch.compile
        with mode='reduce-overhead' and dynamic=False during model attachment.
        """
        if self.compile:
            if (data.batch.max() == 0) and not self.force_compile:
                warnings.warn(
                    "Compilation is allowed only when more than one structure is provided to avoid "
                    "issues with scatter operations. The simulation will run in non-compiled mode. "
                    "A `force_compile=True` flag exists, but it should be used with extreme "
                    "caution. Forcing compilation can expose known issues with scatter operations "
                    "and may produce invalid outputs or NaN gradients. "
                )
            else:
                if data.batch.max() == 0:
                    warnings.warn(force_torch_compile_warning())
                torch._logging.set_logs(dynamo=logging.ERROR)
                try:
                    self.model = torch.compile(
                        self.model, dynamic=True, mode=self.compile_mode
                    )
                    self._run_and_check(data)
                    return
                except Exception as e:
                    # if compilation fails in some parts of the model,by enabling
                    # torch._dynamo.config.suppress_errors = True
                    # errors are suppressed and problematic parts are runned in eager mode
                    # !! it is reccomended to do so only for debugging purposes,
                    # best practice is to manually add @torch.compiler.disable decorator
                    # to problematic parts of the code !!

                    warnings.warn(torch_compile_waring(e))
                    # Also emit a Python warning for testing/logging integration
                    warnings.warn(
                        "torch.compile fallback active! See printed message above.",
                        stacklevel=2,
                    )

                    # import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(
                        self.model, dynamic=True, mode=self.compile_mode
                    )
        self._run_and_check(data)

        return

    def log(self, iter_: int):
        """Utility to log simulation progress"""
        logstring = "{}/{} time points saved ({})".format(
            iter_, self.n_timesteps // self.save_interval, time.asctime()
        )
        logger.info(logstring)

    def get_throughput_metrics(self) -> dict:
        """
        Get throughput metrics from the second half of simulation.

        Returns a dict with:
        - second_half_elapsed_time: Time for the second half of simulation
        - second_half_steps: Number of steps in second half
        - throughput: (steps * n_sims) / time
        - ms_per_timestep: Milliseconds per timestep
        - first_half_steps: Number of steps skipped (first half)
        """
        if self._warmup_end_time is None or self._simulation_end_time is None:
            return None

        second_half_time = self._simulation_end_time - self._warmup_end_time
        first_half_steps = self.n_timesteps // 2
        second_half_steps = self._post_warmup_steps

        if second_half_time > 0 and second_half_steps > 0:
            throughput = (second_half_steps * self.n_sims) / second_half_time
            ms_per_step = (second_half_time / second_half_steps) * 1000
        else:
            throughput = 0
            ms_per_step = 0

        # Get second-half memory stats
        peak_memory_allocated = getattr(self, '_second_half_peak_memory_allocated', 0)
        peak_memory_reserved = getattr(self, '_second_half_peak_memory_reserved', 0)

        return {
            "second_half_elapsed_time": second_half_time,
            "second_half_steps": second_half_steps,
            "throughput": throughput,
            "ms_per_timestep": ms_per_step,
            "first_half_steps": first_half_steps,
            "n_sims": self.n_sims,
            "n_atoms": self.n_atoms,
            "peak_memory_allocated_gb": peak_memory_allocated,
            "peak_memory_reserved_gb": peak_memory_reserved,
        }

    def summary(self):
        """Logs summary information after finishing the simulation"""
        logger.info("=" * 50)
        logger.info("Simulation Complete ({})".format(time.asctime()))
        logger.info("-" * 50)
        logger.info(f"Total timesteps: {self.n_timesteps}")
        logger.info(f"dt: {self.dt}")

        # Log final potential(s) - transfer to CPU here (deferred from loop for perf)
        if hasattr(self, '_final_potential') and self._final_potential is not None:
            potential = self._final_potential.cpu().flatten()
            if potential.numel() == 1:
                logger.info(f"Final potential: {potential.item():.6f}")
            else:
                # Multiple simulations - show individual values and statistics
                for i, p in enumerate(potential):
                    logger.info(f"Final potential (sim {i}): {p.item():.6f}")
                logger.info("-" * 50)
                mean_pot = potential.mean().item()
                std_pot = potential.std().item()
                min_pot = potential.min().item()
                max_pot = potential.max().item()
                logger.info(f"Mean potential: {mean_pot:.6f} ± {std_pot:.6f}")
                logger.info(f"Min: {min_pot:.6f}, Max: {max_pot:.6f}")

        # Log output location
        if self.filename is not None:
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Output prefix: {os.path.basename(self.filename)}")

        logger.info("=" * 50)

    @nvtx.annotate("calculate_potential_and_forces", color="green")
    def calculate_potential_and_forces(
        self, data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to calculate predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        data :
            collated AtomicData instance from the previous timestep

        Returns
        -------
        potential :
            scalar potential predicted by the model
        forces :
            vector forces predicted by the model
        """
        # Log input shapes before model forward
        if self.print_shape and self._shape_log_step_count <= self.print_shape_steps:
            self._shape_log_file.write("  [calculate_potential_and_forces] INPUT:\n")
            self._log_shape("    data.pos", data.pos)
            self._log_shape("    data.atom_types", data.atom_types)
            self._log_shape("    data.batch", data.batch)
            if hasattr(data, 'ptr'):
                self._log_shape("    data.ptr", data.ptr)
            # Log neighbor list info
            if hasattr(data, 'neighbor_list') and data.neighbor_list:
                self._shape_log_file.write("    neighbor_list:\n")
                for nl_name, nl_data in data.neighbor_list.items():
                    if isinstance(nl_data, dict) and "index_mapping" in nl_data:
                        self._log_shape(f"      {nl_name}.index_mapping", nl_data["index_mapping"], indent=3)
                        if "cell_shifts" in nl_data and nl_data["cell_shifts"] is not None:
                            self._log_shape(f"      {nl_name}.cell_shifts", nl_data["cell_shifts"], indent=3)
            # Pass shape logger to model via data object
            data._shape_logger = self._shape_log_file

        # Pass dump_neighbor_list flag to model via data object
        if self.dump_neighbor_list:
            data._dump_neighbor_list = True

        with nvtx.annotate("model_forward", color="lime"):
            data = self.model(data)

        # Remove shape logger from data
        if hasattr(data, '_shape_logger'):
            delattr(data, '_shape_logger')

        # Collect neighbor list (edge_index) if dumping is enabled
        if self.dump_neighbor_list:
            if hasattr(data, '_dump_neighbor_list'):
                delattr(data, '_dump_neighbor_list')
            # Collect edge_index from model outputs (stored by models like SchNet)
            timestep_nl = {}
            for key, val in data.out.items():
                if isinstance(val, dict) and "edge_index" in val:
                    timestep_nl[key] = val["edge_index"].detach().cpu().clone()
            if timestep_nl:
                self._neighbor_list_dump.append(timestep_nl)
                # Keep only last N if specified (circular buffer behavior)
                if (
                    self.dump_neighbor_list_last_n is not None
                    and len(self._neighbor_list_dump)
                    > self.dump_neighbor_list_last_n
                ):
                    self._neighbor_list_dump.pop(0)

        with nvtx.annotate("extract_energy_forces", color="olive"):
            potential = data.out[ENERGY_KEY].detach()
            forces = data.out[FORCE_KEY].detach()

        # Log output shapes after model forward
        if self.print_shape and self._shape_log_step_count <= self.print_shape_steps:
            self._shape_log_file.write("  [calculate_potential_and_forces] OUTPUT:\n")
            self._log_shape("    potential (energy)", potential)
            self._log_shape("    forces", forces)
            # Log model output dict structure
            self._shape_log_file.write("    data.out keys: {}\n".format(list(data.out.keys())))
            for key, val in data.out.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        if isinstance(subval, torch.Tensor):
                            self._log_shape(f"      data.out[{key}][{subkey}]", subval)
                elif isinstance(val, torch.Tensor):
                    self._log_shape(f"      data.out[{key}]", val)
            self._shape_log_file.write("\n")

        return potential, forces

    def timestep(self):
        raise NotImplementedError

    @staticmethod
    def validate_data_list(data_list: List[AtomicData]):
        """Helper method to check and collate the initial data list"""

        pos_shape = data_list[0].pos.shape
        atom_types = data_list[0].atom_types
        nls = data_list[0].neighbor_list
        if MASS_KEY not in data_list[0]:
            initial_masses = False
        else:
            initial_masses = True

        # check to make sure every structure has the same number of atoms
        # and the proper neighbor_list structure
        for frame, data in enumerate(data_list):
            current_nls = data.neighbor_list
            if data.pos.shape != pos_shape:
                raise ValueError(
                    "Postions shape {} at frame {} differes from shape {} "
                    "in previous frames.".format(
                        data.pos.shape, frame, pos_shape
                    )
                )
            if (
                np.testing.assert_array_equal(
                    data.atom_types.numpy(), atom_types.numpy()
                )
                == False
            ):
                raise ValueError(
                    "Atom types {} at frame {} are not equal to atom types in "
                    "previous frames.".format(data.atom_types, frame)
                )
            if set(current_nls.keys()) != set(nls.keys()):
                raise ValueError(
                    "Neighbor list keyset {} at frame {} does not match keysets "
                    "of previous frames.".format(
                        set(data.neighbor_list.keys()), frame
                    )
                )
            for key in current_nls.keys():
                mapping = current_nls[key]["index_mapping"]
                if (
                    np.testing.assert_array_equal(
                        mapping.numpy(), nls[key]["index_mapping"]
                    )
                    == False
                ):
                    raise ValueError(
                        "Index mapping {} for key {} at frame {} does not match "
                        "those of previous frames.".format(mapping, key, frame)
                    )
            if MASS_KEY in data and initial_masses == False:
                raise ValueError(
                    "Masses {} supplied for frame {}, but previous frames "
                    "have no masses.".format(data.masses, frame)
                )
            if initial_masses == None and MASS_KEY not in data:
                raise ValueError(
                    "Masses are none for frame {}, but previous frames "
                    "have masses {}.".format(frame, data.masses)
                )
            if MASS_KEY in data:
                if data.masses.shape != atom_types.shape:
                    raise ValueError(
                        "Number of masses {} at frame {} do not match number of atoms "
                        "in previous frames.".format(
                            data.masses.shape[0], atom_types.shape[0]
                        )
                    )

    @staticmethod
    def collate(data_list: List[AtomicData]) -> AtomicData:
        """Method for collating a list of individual AtomicData instances into a
        single AtomicData instance, with proper incrementing of AtomicData properties.
        """

        collated_data, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return collated_data

    def input_option_checks(self):
        """Method to catch any problems before starting a simulation:
        - Make sure the save_interval evenly divides the simulation length
        - Checks compatibility of arguments to save and log
        - Sets up saving parameters for numpy and log files, if relevant
        """

        # make sure save interval is a factor of total n_timesteps
        if self.n_timesteps % self.save_interval != 0:
            raise ValueError(
                "The save_interval must be a factor of the simulation n_timesteps"
            )

        # check whether a directory is specified if any saving is done
        if self.export_interval is not None and self.filename is None:
            raise RuntimeError(
                "Must specify filename if export_interval isn't None"
            )
        if self.log_interval is not None:
            if self.log_type == "write" and self.filename is None:
                raise RuntimeError(
                    "Must specify filename if log_interval isn't None and log_type=='write'"
                )

        # checkpoint loading
        if self.read_checkpoint_file is not None:
            if isinstance(self.read_checkpoint_file, Path_fr):
                checkpointed_data = torch.load(
                    self.read_checkpoint_file(), weights_only=False
                )
            elif self.read_checkpoint_file:
                # Find the latest checkpoint file (with highest index)
                import glob
                pattern = "{}_checkpoint_*.pt".format(self.filename)
                checkpoint_files = sorted(glob.glob(pattern))
                if checkpoint_files:
                    fn = checkpoint_files[-1]  # Get the latest one
                else:
                    # Fallback to old format for backwards compatibility
                    fn = "{}_checkpoint.pt".format(self.filename)
                assert os.path.exists(fn), f"No checkpoint file found matching {pattern} or {fn}"
                checkpointed_data = torch.load(fn, weights_only=False)
            self.checkpointed_data = checkpointed_data
            self.current_timestep = self.checkpointed_data["current_timestep"]
            if "export_interval" in self.checkpointed_data.keys():
                if (
                    self.export_interval
                    != self.checkpointed_data["export_interval"]
                ):
                    warnings.warn(
                        "specified export_interval doesn't match the export interval in the checkpoint, using checkpointed export interval instead",
                        UserWarning,
                    )
                    self.export_interval = self.checkpointed_data[
                        "export_interval"
                    ]
            if "save_interval" in self.checkpointed_data.keys():
                if (
                    self.save_interval
                    != self.checkpointed_data["save_interval"]
                ):
                    warnings.warn(
                        "specified save_interval doesn't match the save interval in the checkpoint, using checkpointed save interval instead",
                        UserWarning,
                    )
                    self.save_interval = self.checkpointed_data["save_interval"]
            if "log_interval" in self.checkpointed_data.keys():
                if self.log_interval != self.checkpointed_data["log_interval"]:
                    warnings.warn(
                        "specified log_interval doesn't match the log interval in the checkpoint, using checkpointed log interval instead",
                        UserWarning,
                    )
                    self.log_interval = self.checkpointed_data["log_interval"]
        else:
            self.checkpointed_data = None
            self.current_timestep = 0

        # saving numpys
        if self.export_interval is not None:
            if self.n_timesteps // self.export_interval >= 10000:
                raise ValueError(
                    "Simulation saving is not implemented if more than 10000 files will be generated"
                )

            if os.path.isfile(
                "{}_coords_{}.npy".format(self.filename, self.current_timestep)
            ):
                raise ValueError(
                    "{} already exists; choose a different filename.".format(
                        "{}_coords_{}.npy".format(
                            self.filename, self.current_timestep
                        )
                    )
                )

            if self.export_interval is not None:
                if self.export_interval % self.save_interval != 0:
                    raise ValueError(
                        "Numpy saving must occur at a multiple of save_interval"
                    )
                if self.read_checkpoint_file is not None:
                    # the `current_timestep` in the checkpoint is actually the number of the last
                    # numpy filed saved. We need to use it to reset the _npy_file_index
                    self._npy_file_index = self.current_timestep
                else:
                    self._npy_file_index = 0

        # logging
        if self.log_interval is not None:
            if self.log_interval % self.save_interval != 0:
                raise ValueError(
                    "Logging must occur at a multiple of save_interval"
                )

            if self.log_type == "write":
                self._log_file = self.filename + "_log.txt"

                if os.path.isfile(self._log_file):
                    if self.checkpointed_data is None:
                        raise ValueError(
                            "{} already exists; choose a different filename.".format(
                                self._log_file
                            )
                        )
        # simulation subroutine
        if self.sim_subroutine != None and self.sim_subroutine_interval == None:
            raise ValueError(
                "subroutine {} specified, but subroutine_interval is ambiguous.".format(
                    self.sim_subroutine
                )
            )
        if self.sim_subroutine_interval != None and self.sim_subroutine == None:
            raise ValueError(
                "subroutine interval specified, but subroutine is ambiguous."
            )

        # Saving extra force components
        if self.save_force_components and (self.force_components is None):
            raise ValueError(
                f"save_energy_components is requested,but no energy_components provided"
            )
        # Saving extra energy components
        if self.save_energy_components and (self.energy_components is None):
            raise ValueError(
                f"save_energy_components is requested,but no energy_components provided"
            )

        # Dump neighbor list validation
        if self.dump_neighbor_list and self.filename is None:
            raise ValueError(
                "dump_neighbor_list is enabled but filename is None. "
                "Please provide a filename for saving the neighbor list."
            )

    def _get_numpy_count(self):
        """Returns a string 0000-9999 for appending to numpy file outputs"""
        return f"{self._npy_file_index:04d}"

    def _swap_and_export(
        self, input_tensor: torch.Tensor, axis1: int = 0, axis2: int = 1
    ) -> np.ndarray:
        """Helper method to exchange the zeroth and first axes of tensors that
        will be output or exported as numpy arrays

        Parameters
        ----------
        input_tensor:
            Tensor of shape (n_save_steps, n_sims, n_atoms, n_dims)
        axis1:
            The axis that will be occupied by data from axis2 after the swap
        axis2:
            The axis that will be occupied by data from axis1 after the swap

        Returns
        -------
        swapped_data:
            Numpy array of the input data with swapped axes
        """

        axes = list(range(len(input_tensor.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = input_tensor.permute(*axes)
        return swapped_data.cpu().detach().numpy()

    def _log_shape(self, label: str, tensor_or_dict, indent: int = 0):
        """Helper method to log tensor shapes to the shape log file.

        Parameters
        ----------
        label : str
            Description of what this tensor represents
        tensor_or_dict : torch.Tensor or dict
            The tensor whose shape should be logged, or a dict of tensors
        indent : int
            Indentation level for nested structures
        """
        if not self.print_shape or self._shape_log_file is None:
            return
        if self._shape_log_step_count > self.print_shape_steps:
            return

        prefix = "  " * indent
        if isinstance(tensor_or_dict, torch.Tensor):
            self._shape_log_file.write(
                f"{prefix}{label}: shape={list(tensor_or_dict.shape)}, "
                f"dtype={tensor_or_dict.dtype}, device={tensor_or_dict.device}\n"
            )
        elif isinstance(tensor_or_dict, dict):
            self._shape_log_file.write(f"{prefix}{label}:\n")
            for k, v in tensor_or_dict.items():
                if isinstance(v, torch.Tensor):
                    self._log_shape(str(k), v, indent + 1)
                elif isinstance(v, dict):
                    self._log_shape(str(k), v, indent + 1)
        else:
            self._shape_log_file.write(f"{prefix}{label}: {type(tensor_or_dict).__name__}\n")

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        if self._simulated and not overwrite:
            raise RuntimeError(
                "Simulation results are already populated. "
                "To rerun, set overwrite=True."
            )

        # Reset neighbor list dump for new simulation
        if self.dump_neighbor_list:
            self._neighbor_list_dump = []

        self._save_size = int(self.export_interval / self.save_interval)

        self.simulated_coords = torch.zeros(
            (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
        )
        if self.save_forces:
            self.simulated_forces = torch.zeros(
                (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
            )
        else:
            self.simulated_forces = None

        if self.save_energies:
            self.simulated_potential = torch.zeros(self._save_size, self.n_sims)
        else:
            self.simulated_potential = None

        if self.save_force_components:
            self.force_components = {
                key: torch.zeros(
                    self._save_size, self.n_sims, self.n_atoms, self.n_dims
                )
                for key in self.force_components
            }
        else:
            self.force_components = None

        if self.save_energy_components:
            self.energy_components = {
                key: torch.zeros(self._save_size, self.n_sims)
                for key in self.energy_components
            }
        else:
            self.energy_components = None

        if self.log_interval is not None:
            logstring = "Generating {} simulations of n_timesteps {} saved at {}-step intervals ({})".format(
                self.n_sims,
                self.n_timesteps,
                self.save_interval,
                time.asctime(),
            )
            logger.info(logstring)

        # Setup shape logging file
        if self.print_shape and self.filename is not None:
            shape_log_path = f"{self.filename}_print_shape.log"
            self._shape_log_file = open(shape_log_path, "w")
            self._shape_log_file.write(f"Shape Log - {time.asctime()}\n")
            self._shape_log_file.write("=" * 80 + "\n")
            self._shape_log_file.write(f"Simulation Configuration:\n")
            self._shape_log_file.write(f"  n_sims: {self.n_sims}\n")
            self._shape_log_file.write(f"  n_atoms: {self.n_atoms}\n")
            self._shape_log_file.write(f"  n_dims: {self.n_dims}\n")
            self._shape_log_file.write(f"  n_timesteps: {self.n_timesteps}\n")
            self._shape_log_file.write(f"  dt: {self.dt}\n")
            self._shape_log_file.write(f"  device: {self.device}\n")
            self._shape_log_file.write(f"  dtype: {self.dtype}\n")
            self._shape_log_file.write("=" * 80 + "\n\n")
            self._shape_log_step_count = 0
        elif self.print_shape and self.filename is None:
            warnings.warn(
                "print_shape=True but filename is None. Shape logging disabled."
            )

    def save(
        self,
        data: AtomicData,
        forces: torch.Tensor,
        potential: torch.Tensor,
        t: int,
    ):
        """Utility to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy
        Parameters
        ----------
        x_new :
            current coordinates
        forces:
            current forces
        potential :
            current potential
        t :
            current timestep
        """
        x_new = data.pos.view(-1, self.n_atoms, self.n_dims)
        forces = forces.view(-1, self.n_atoms, self.n_dims)

        pos_spread = x_new.std(dim=(1, 2)).detach()

        # Combine conditions on GPU for single sync instead of multiple
        diverged = (pos_spread.max() > 1e3 * self.initial_pos_spread) | torch.any(
            torch.isnan(pos_spread)
        )
        if diverged.item():
            raise RuntimeError(
                f"Simulation of trajectory blew up at #timestep={t}"
            )

        save_ind = (
            t // self.save_interval
        ) - self._npy_file_index * self._save_size

        self.simulated_coords[save_ind, :, :] = x_new

        if self.save_forces:
            self.simulated_forces[save_ind, :, :] = forces

        if self.save_energies:
            if self.simulated_potential is None:
                assert potential.shape[0] == self.n_sims
                potential_dims = [self._save_size, self.n_sims] + [
                    potential.shape[j] for j in range(1, len(potential.shape))
                ]
                self.simulated_potential = torch.zeros((potential_dims))

            self.simulated_potential[save_ind] = potential

        if self.save_force_components:
            for key, tensor in self.force_components.items():  # type: ignore , check for None in input validation
                tensor[save_ind, :, :] = deepcopy(
                    data.out[key][FORCE_KEY].detach()
                ).view(-1, self.n_atoms, self.n_dims)

        if self.save_energy_components:
            for key, tensor in self.energy_components.items():  # type: ignore , check for None in input validation
                tensor[save_ind] = deepcopy(data.out[key][ENERGY_KEY].detach())

        if self.create_checkpoints:
            self.checkpoint = {}
            self.checkpoint[POSITIONS_KEY] = deepcopy(
                data[POSITIONS_KEY].detach()
            )
            self.checkpoint[VELOCITY_KEY] = deepcopy(
                data[VELOCITY_KEY].detach()
            )

    def write(self):
        """Utility to write numpy arrays to disk"""
        key = self._get_numpy_count()

        coords_to_export = self.simulated_coords
        coords_to_export = self._swap_and_export(coords_to_export)
        np.save("{}_coords_{}.npy".format(self.filename, key), coords_to_export)

        if self.save_forces:
            forces_to_export = self.simulated_forces
            forces_to_export = self._swap_and_export(forces_to_export)
            np.save(
                "{}_forces_{}.npy".format(self.filename, key), forces_to_export
            )

        if self.save_energies:
            potentials_to_export = self.simulated_potential
            potentials_to_export = self._swap_and_export(potentials_to_export)
            np.save(
                "{}_potential_{}.npy".format(self.filename, key),
                potentials_to_export,
            )

        if self.save_force_components:
            components_to_export = {
                name: self._swap_and_export(i)
                for name, i in self.force_components.items()
            }
            np.savez(
                "{}_force_components_{}.npz".format(self.filename, key),
                **components_to_export,
            )

        if self.save_energy_components:
            components_to_export = {
                name: self._swap_and_export(i)
                for name, i in self.energy_components.items()
            }
            np.savez(
                "{}_energy_components_{}.npz".format(self.filename, key),
                **components_to_export,
            )

        if self.create_checkpoints:
            self.checkpoint["current_timestep"] = self._npy_file_index + 1
            self.checkpoint["export_interval"] = self.export_interval
            self.checkpoint["save_interval"] = self.save_interval
            self.checkpoint["log_interval"] = self.log_interval
            torch.save(
                self.checkpoint,
                "{}_checkpoint_{}.pt".format(self.filename, key),
            )

        # Reset simulated coords, forces and potential
        self.simulated_coords = torch.zeros(
            (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
        )
        if self.save_forces:
            self.simulated_forces = torch.zeros(
                (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
            )
        else:
            self.simulated_forces = None

        if self.save_energies:
            self.simulated_potential = torch.zeros(self._save_size, self.n_sims)
        else:
            self.simulated_potential = None

        if self.save_force_components:
            self.force_components = {
                key: torch.zeros(
                    self._save_size, self.n_sims, self.n_atoms, self.n_dims
                )
                for key in self.force_components
            }
        else:
            self.force_components = None

        if self.save_energy_components:
            self.energy_components = {
                key: torch.zeros(self._save_size, self.n_sims)
                for key in self.energy_components
            }
        else:
            self.energy_components = None

        self._npy_file_index += 1

    def reshape_output(self):
        # reshape output attributes
        self.simulated_coords = self._swap_and_export(self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(self.simulated_forces)

        if self.save_energies:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential
            )

        if self.save_force_components:
            self.force_components = {
                key: self._swap_and_export(tensor)
                for key, tensor in self.force_components.items()
            }

        if self.save_energy_components:
            self.energy_components = {
                key: self._swap_and_export(tensor)
                for key, tensor in self.energy_components.items()
            }

    def attach_model(self, model: torch.nn.Module):
        warnings.warn(
            "using 'attach_model' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_model(model)

    def attach_configurations(
        self, configurations: List[AtomicData], beta: Union[float, List[float]]
    ):
        warnings.warn(
            "using 'attach_configurations' is deprecated, use 'attach_model_and_configurations' instead.",
            DeprecationWarning,
        )
        self._attach_configurations(configurations, beta)
