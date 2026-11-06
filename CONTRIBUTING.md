# Contributing to FlashMD

## Branches

`main` holds the released FlashSchNet package. Unification work happens on
`unify` and merges into `main` at release. The plan is in
[`docs/design/unification.md`](docs/design/unification.md).

## Code style

FlashMD follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Ruff and mypy enforce most of it; the rest is reviewed by hand.

- Docstrings use the Google layout: a one-line summary, a blank line, then
  `Args:`, `Returns:`, `Raises:`, `Yields:` or `Attributes:` sections.
- Tensor arguments state shape and dtype, for example
  `positions: Float tensor of shape [R, N, 3], in angstrom.`
- Public functions and classes carry full type annotations. Use PEP 604 unions
  (`int | None`).
- Lines are at most 80 characters; indentation is 4 spaces.
- Prefer importing modules over names (`from flashmd import config`), except
  for `typing` and `collections.abc`.
- No work at import time: no environment reads, prints or monkeypatches.
  Kernel availability is checked when a kernel is first used.
- No mutable default arguments.

Modules that predate the unification are listed under
`[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`. Remove a module's
entry when you port it, and add the module to `[tool.mypy] files`.

Install the hooks once:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Tests

| Marker | Meaning | Where it runs |
|---|---|---|
| none | CPU unit test | GitHub Actions on every push and pull request |
| `heavy` | slow, may download published weights | GitHub Actions weekly and on manual dispatch |
| `gpu` | needs a CUDA GPU | the UNITES cluster or SkyPilot, see below |

Every fast path ships with a parity test against the unmodified reference
path on identical inputs, and every custom `torch.autograd.Function` has a
float64 `gradcheck`.

## Continuous integration

`.github/workflows/ci.yml` installs `requirements/ci-cpu.lock.txt`, a pinned
CPU environment (torch CPU build, the matching `torch-cluster` wheel from the
PyG index, and the dev tools), then runs ruff, mypy and the CPU tests. After
changing dependencies in `pyproject.toml`, regenerate the lock:

```bash
tools/ci/compile_lock.sh
```

`requirements/ci-cpu.constraints.txt` pins the torch version; it must match
the PyG wheel page the script reads `torch-cluster` from.

## GPU checks

GitHub-hosted runners have no GPU. The correctness gates and golden captures
run as Slurm jobs on the UNITES cluster, in the environment built by
`tools/cluster/build_env.sh`:

```bash
PYTHON=$ENV/bin/python REPO=$PWD DATA=$DATA OUT=$OUT \
    sbatch --export=ALL tools/gates/cgschnet_gpu_gates.sbatch
sbatch tools/goldens/mace_goldens.sbatch
sbatch tools/goldens/bioemu_goldens.sbatch
```

## Third-party code

FlashMD adapts code from several MIT and BSD-licensed projects. When you copy
or adapt code, keep its copyright notice and add the project to
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
