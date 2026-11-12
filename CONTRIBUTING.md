# Contributing to imagetransforms

## What belongs here

A transformation of an image (or of a stack of them) that more than one
project of the group uses, or that is general enough to be reused: the
orientation alignment transform, a local-maximum finder, an azimuthal average.
The library depends on `numpy` and `scipy` only; an accelerator that needs
another package (OpenCV, for example) must be optional and must give the same
answer.  Code for a camera, a GUI or a particular analysis belongs in
[QVideo](https://github.com/davidgrier/QVideo) or
[pylorenzmie](https://github.com/davidgrier/pylorenzmie).

If you are not sure whether an idea fits, open an issue first.

## Development environment

```bash
git clone https://github.com/davidgrier/imagetransforms.git
cd imagetransforms
pip install -e ".[dev]"
python -m pytest          # tests and doctests
ruff check .
pyright
```

## Style

- Single quotes, including docstrings; lines of at most 79 characters.
- NumPy-style docstrings with an `Examples` section that runs as a doctest.
- American English spelling.
- A transformation takes and returns arrays, never a GUI object; it does not
  change its input.
- Tests use `unittest`.  Compare an accelerated version with the plain one.

## Releasing

`pyproject.toml` holds the version; `CITATION.cff` is updated by CI when a tag
is pushed.  Bump the version, update `CHANGELOG.md`, commit and push, then
`git tag vX.Y.Z && git push origin vX.Y.Z`.  The tag publishes to PyPI and
creates the GitHub Release, which Zenodo archives.
