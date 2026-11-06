# CGSchNet CPU parity, 2026-11-05

`tools/parity/cgschnet_mlcg_parity.py` loaded the published checkpoint
(`pingzhili/cg-schnet`, `model_and_prior.pt`) through FlashMD's class remap
(commit `6d2a87c` on `unify`) and through the original mlcg classes, then
compared energies and forces on 8 replicas of the two 1ENH configurations
(2152 beads) with seeded Gaussian noise of 0.05 angstrom.

| Reference | dtype | Max relative difference | Total forces |
|---|---|---|---|
| FlashSchNet fork of mlcg, `ebe2d35` | float64 | 0 | 0 |
| FlashSchNet fork of mlcg, `ebe2d35` | float32 | 2.8e-8 (SchNet forces) | 0 |
| Upstream mlcg, `1bab998` (fork point) | float64 | 9.6e-16 | 2.1e-16 |
| Upstream mlcg, `1bab998` (fork point) | float32 | 3.3e-7 | 1.2e-7 |
| Upstream mlcg, `f80cf71` (HEAD) | float64 | 9.6e-16 | 2.1e-16 |
| Upstream mlcg, `f80cf71` (HEAD) | float32 | 3.3e-7 | 1.2e-7 |

Upstream runs use `--drop-empty-terms`: upstream priors scatter without
`dim_size`, so the five dihedral types absent from 1ENH return empty energies
and `SumOut` cannot add them. The fork and FlashMD return zeros for them.

CPU tensors take the PyTorch path in every package, so these records cover
the port and the remap, not the Triton kernels.

Environment: Python 3.12.12, torch 2.9.1+cu130 on CPU, torch_geometric
2.7.0, numpy 2.3.5. Upstream mlcg imports MACE at import time, so those runs
added mace-torch 0.3.13 and e3nn 0.5.6 on `PYTHONPATH`.
