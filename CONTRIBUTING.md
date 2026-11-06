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
| none | CPU unit test | GitHub Actions on every push |
| `gpu` | needs a CUDA GPU | the UNITES cluster or SkyPilot, see below |
| `heavy` | slow; skipped with `--light` | nightly |

Every fast path ships with a parity test against the unmodified reference
path on identical inputs, and every custom `torch.autograd.Function` has a
float64 `gradcheck`.

Run the GPU tests on the UNITES cluster:

```bash
sbatch tools/cluster/gpu_tests.sbatch
```

## Third-party code

FlashMD adapts code from several MIT and BSD-licensed projects. When you copy
or adapt code, keep its copyright notice and add the project to
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
