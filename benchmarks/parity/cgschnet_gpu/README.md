# CGSchNet GPU gates

`tools/gates/cgschnet_gpu_gates.sbatch` on one NVIDIA RTX PRO 6000 Blackwell
(driver 610.43.02), environment from `tools/cluster/build_env.sh`
(torch 2.10.0+cu128, triton 3.6.0). Inputs: 8 replicas of the two 1ENH
configurations with seeded 0.05 angstrom noise. Values are max relative
differences of the SchNet forces against the float32 reference with every
kernel off.

| Model | Run 1, `5db62a1` | Run 2, `e22876b` |
|---|---|---|
| public checkpoint | 2.0e-2 | 6.7e-7 |
| random SchNet, `GaussianBasis` (fused RBF) | 3.5e-2 | 6.3e-4 |
| random SchNet, 256 filters | 3.3e-1 | 7.8e-4 |

Run 1 found two force bugs in the fused CFConv kernels:

- The autograd functions returned no gradient for the edge distances, so
  the cutoff envelope derivative was missing. Flash forces matched a
  reference with the envelope detached to 6e-7.
- The source-CSR backward covered only the first 128 filters.

Run 2 (both fixed) isolates the remaining error on the random models to the
fused tanh-linear kernel alone: every `only:` arm except that switch, and
every `without:` arm that removes it, is within 2e-6. That kernel ran
`tl.dot` in TF32; `135f012` makes it follow PyTorch's float32 matmul
precision. The public checkpoint never runs it.

The float32 reference itself differs from float64 by 2e-5. W16A16 filter
networks add 6.4e-4 on the public checkpoint.

`run1.summary.json` and `run2.summary.json` hold every pair.
