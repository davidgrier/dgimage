# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`dgimage` is the one shared implementation of image transformations
that originate in the Grier group and are used by more than one project:
[QVideo](https://github.com/davidgrier/QVideo) (the `CircleTransformFilter`)
and [pylorenzmie](https://github.com/davidgrier/pylorenzmie) (`lib/CircleTransform`).
Neither of those should carry its own copy.  The library therefore stays
small and light:

- Dependencies are `numpy` and `scipy` only.  No Qt, pandas or camera code.
  OpenCV is the one optional accelerator (extra `fast`): `circletransform`
  imports it lazily and uses it for the derivatives if it is there.  Any
  accelerator must be optional and must agree, to rounding, with the plain
  scipy path, and both paths are tested against `scipy.signal.savgol_filter`.
- A transformation takes and returns arrays and does not change its input.
  The scale and the dtype of the pixels must not matter: callers pass
  `uint8`, `uint16` counts or normalized floats.
- Results must stay what pylorenzmie computed before it switched to this
  package (the circle transform agrees to 3e-15).  If a change
  alters output, say so in `CHANGELOG.md` and tell the other two projects.

The first release holds the circle transform and the azimuthal statistics
(moved from `pylorenzmie.lib.Azimuthal`, results identical).  The old,
unpackaged `~/python/dgmath` is the ancestor of this package and can retire.

**Scope rule (the owner's): dgimage holds only algorithms for which no
existing library has a better solution.**  Do not add, propose or document an
algorithm that scipy, scikit-image, OpenCV, trackpy or similar already does as
well or better (a local-maximum finder was considered and declined, 2026-11-12:
the dilation is a few lines of scipy and trackpy refines and filters).  Check
the existing libraries first; the circle transform (the group's own) and the
azimuthal statistics (no library offers `med` and `mad` of radii together with
the cached geometry) pass that test.

## Commands

```bash
pip install -e ".[dev]"                       # tests, ruff, pyright
python -m pytest                              # tests/ and the doctests in the package
python -m pytest tests/test_circletransform.py::TestKernel::test_it_is_kept_for_a_shape
python -m pytest --cov --cov-report=term-missing   # fails under 95%
ruff check .
pyright
pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html
```

`pytest` runs doctests (`--doctest-modules`, set in `pyproject.toml`), so the
`Examples` section of every public function is a test.  Keep the examples
small and make them run.

## Layout

One module per transformation in `dgimage/`, each re-exported from
`dgimage/__init__.py` (`__all__` lists the public names; `azimuthal`
is exported as a module, used as `azimuthal.avg(image, center)`).  A
transformation has a plain function for one call and, where it keeps state
that depends on the input shape (the circle transform's Fourier kernel), a
class that caches it; the function reuses a module-level instance for the
default parameters.  `py.typed` marks the package as typed.

Documentation is Sphinx with the PyData theme and the NYU colors
(`docs/_static/nyu.css`), the same as QVideo.  A new transformation needs a
page entry in `docs/api/index.rst`, a section in `docs/quickstart.rst`, a row
in the README table, and a `CHANGELOG.md` entry.

## Style

Single quotes (docstrings too), lines of at most 79 characters, NumPy-style
docstrings, American English.  `ruff` and `pyright` (pinned in the `dev` extra, as in QVideo) run in CI
(the `Lint and types` job); the scipy stubs are loose, so wrap scipy results in `np.asarray`
and use `np.real` / `np.imag` rather than `.real` / `.imag` on them.

## Releasing

`pyproject.toml` is the only place for the version number; `__version__` is
read from the installed metadata.  **Never edit `CITATION.cff` by hand**: CI
(`update-citation.yml`) rewrites its version and date when a tag is pushed.

1. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, commit, push.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.

The tag runs `publish.yml` (build, PyPI through Trusted Publishing, and the
GitHub Release; do not run `gh release create` yourself, it will conflict)
and `update-citation.yml`.  The GitHub Release fires Zenodo's webhook, which
mints the DOI.  A change to documentation only does not need a release.
