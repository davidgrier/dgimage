# imagetransforms: image transformations for holography and microscopy

[![PyPI version](https://img.shields.io/pypi/v/imagetransforms)](https://pypi.org/project/imagetransforms/)
[![Python](https://img.shields.io/pypi/pyversions/imagetransforms)](https://pypi.org/project/imagetransforms/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://github.com/davidgrier/imagetransforms/actions/workflows/test.yml/badge.svg)](https://github.com/davidgrier/imagetransforms/actions/workflows/test.yml)
[![Documentation](https://app.readthedocs.org/projects/imagetransforms/badge/?version=latest)](https://imagetransforms.readthedocs.io/en/latest/)

**imagetransforms** is a small library of the image transformations that the
[Grier group](https://github.com/davidgrier) uses across projects.  It needs
only `numpy` and `scipy`: no camera, no GUI and no analysis framework, so that
[QVideo](https://github.com/davidgrier/QVideo),
[pylorenzmie](https://github.com/davidgrier/pylorenzmie) and other programs
can share one implementation of each algorithm.

## Install

```bash
pip install imagetransforms
```

## Use

```python
import numpy as np
from imagetransforms import circletransform

heat = circletransform(image)      # peaks are the centers of rings
y, x = np.unravel_index(heat.argmax(), heat.shape)
```

The transform of Krishnatreya and Grier locates circularly symmetric features,
such as the rings of the hologram of a sphere, without knowing their radius.
A program that transforms many frames of one shape builds a
`CircleTransform`, which keeps its kernel.

## Contents

| Transformation | What it does |
|----------------|--------------|
| `circletransform` | orientation alignment transform: a map of the centers of rings |

More will join it as projects of the group need them.

## Documentation

<https://imagetransforms.readthedocs.io>

## Citing

See `CITATION.cff`.  The orientation alignment transform is described in
B. J. Krishnatreya and D. G. Grier, "Fast feature identification for
holographic tracking: the orientation alignment transform," *Optics Express*
**22**, 12773-12778 (2014).

## License

GPL-3.0-or-later, like QVideo and pylorenzmie.  See [LICENSE](LICENSE).
