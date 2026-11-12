'''Make the figures of the documentation from holograms of pylorenzmie.

The figures are made by hand and committed (``docs/_static/*.png``), so that
building the documentation needs neither pylorenzmie nor matplotlib's data:

    pip install pylorenzmie matplotlib
    python docs/make_figures.py

The holograms are the example images that pylorenzmie distributes: a real
frame of two silica spheres, and a crop of one sphere, each divided by the
background.
'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pylorenzmie.utilities import example_hologram
from scipy import ndimage

from dgimage import azimuthal, circletransform

OUT = Path(__file__).parent / '_static'
STYLE = dict(cmap='gray', vmin=0.4, vmax=1.6, interpolation='none')


def peaks(heat: np.ndarray, count: int, size: int = 41):
    '''The *count* strongest local maxima of *heat*, as (x, y).'''
    local = heat == ndimage.maximum_filter(heat, size=size)
    ys, xs = np.nonzero(local & (heat > 0.05))
    order = np.argsort(heat[ys, xs])[::-1][:count]
    return xs[order], ys[order]


def transform_figure() -> None:
    '''Two spheres: the hologram, and the map of the centers of rings.'''
    image = example_hologram('image0010.png').data
    heat = circletransform(image)
    x, y = peaks(heat, 2)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4.2),
                             constrained_layout=True)
    axes[0].imshow(image, **STYLE)
    axes[0].set_title('hologram of two spheres')
    axes[1].imshow(heat, cmap='magma', interpolation='none')
    axes[1].set_title('circletransform(hologram)')
    axes[1].scatter(x, y, s=140, facecolors='none', edgecolors='c', lw=1.5)
    for ax in axes:
        ax.axis('off')
    fig.savefig(OUT / 'circletransform.png', dpi=110)
    plt.close(fig)
    centers = list(zip(x.tolist(), y.tolist(), strict=True))
    print('circletransform.png: centers', centers)


def profile_figure() -> None:
    '''One sphere: the hologram, and its azimuthal median about its center.'''
    image = example_hologram('crop.png').data
    x, y = peaks(circletransform(image), 1)
    center = (float(x[0]), float(y[0]))
    median = azimuthal.med(image, center)
    mean = azimuthal.avg(image, center)
    radius = np.arange(len(median))
    fig, axes = plt.subplots(ncols=2, figsize=(10, 3.8),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.4]))
    axes[0].imshow(image, **STYLE)
    axes[0].plot(*center, 'c+', ms=14, mew=2)
    axes[0].set_title('hologram of one sphere')
    axes[0].axis('off')
    axes[1].plot(radius, mean, color='0.7', lw=1, label='avg')
    axes[1].plot(radius, median, color='C0', lw=1.5, label='med')
    axes[1].axhline(1., color='0.5', lw=0.6, ls=':')
    axes[1].set_xlim(0, min(center) + 5)    # beyond it only the corners
    axes[1].set_xlabel('distance from the center [pixels]')
    axes[1].set_ylabel('normalized intensity')
    axes[1].set_title('azimuthal.med(hologram, center)')
    axes[1].legend(frameon=False)
    fig.savefig(OUT / 'azimuthal.png', dpi=110)
    plt.close(fig)
    print('azimuthal.png: center', center)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    transform_figure()
    profile_figure()
