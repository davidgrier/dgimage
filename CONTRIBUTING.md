# Contributing to dgimage

## What belongs here

An algorithm of the group that more than one project uses, **for which no
existing library has a better solution**: the orientation alignment transform,
the azimuthal statistics.  If scipy, scikit-image, OpenCV, trackpy or another
package already does it as well or better, use that package and do not
re-implement it here, however short the code.  Check before you write.

The library depends on `numpy` and `scipy` only; an accelerator that needs
another package (OpenCV, for example) must be optional and must give the same
answer.  Code for a camera, a GUI or a particular analysis belongs in
[QVideo](https://github.com/davidgrier/QVideo) or
[pylorenzmie](https://github.com/davidgrier/pylorenzmie).

If you are not sure whether an idea fits, open an issue first.

## Development environment

```bash
git clone https://github.com/davidgrier/dgimage.git
cd dgimage
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
