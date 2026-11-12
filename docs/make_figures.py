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
from scipy import ndimage, optimize

from dgimage import azimuthal, circletransform

OUT = Path(__file__).parent / '_static'
STYLE = dict(cmap='gray', vmin=0.4, vmax=1.6, interpolation='none')

# The part of the frame of two spheres that the figure shows (x0, x1, y0, y1),
# with more room left and below the spheres than above and right of them.
# The transform is computed on the whole frame, so the crop adds no edges.
WINDOW = (0, 1280, 390, 1024)


def peaks(heat: np.ndarray, count: int, size: int = 41):
    '''The *count* strongest local maxima of *heat*, as (x, y).'''
    local = heat == ndimage.maximum_filter(heat, size=size)
    ys, xs = np.nonzero(local & (heat > 0.05))
    order = np.argsort(heat[ys, xs])[::-1][:count]
    return xs[order], ys[order]


def refine(image: np.ndarray, guess, radius: int = 150):
    '''Move *guess* (x, y) to where *image* is most nearly symmetric: the
    center that minimizes the azimuthal standard deviation.'''
    def spread(center):
        _, deviation = azimuthal.std(image, tuple(center))
        return np.nansum(deviation[2:radius])

    return optimize.minimize(spread, guess, method='Nelder-Mead',
                             options=dict(xatol=1e-3, fatol=1e-9)).x


def transform_figure() -> None:
    '''Two spheres: the hologram, and the map of the centers of rings.'''
    image = example_hologram('image0010.png').data
    heat = circletransform(image)
    x0, x1, y0, y1 = WINDOW
    fig, axes = plt.subplots(ncols=2, figsize=(10, 3.4),
                             constrained_layout=True)
    axes[0].imshow(image, **STYLE)
    axes[0].set_title('hologram of two spheres')
    # The peaks are narrow against a dark map: the square root stretches them.
    axes[1].imshow(np.sqrt(heat), cmap='gray', interpolation='none')
    axes[1].set_title('circletransform(hologram), square root')
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis('off')
    fig.savefig(OUT / 'circletransform.png', dpi=110, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print('circletransform.png written')


def profile_figure() -> None:
    '''One sphere: the hologram, and its azimuthal median about its center.

    The circle transform finds the sphere to a few pixels; the center that
    makes the hologram most symmetric sharpens the profile.
    '''
    image = example_hologram('crop.png').data
    x, y = peaks(circletransform(image), 1)
    coarse = (float(x[0]), float(y[0]))
    center = tuple(float(c) for c in refine(image, coarse))
    median = azimuthal.med(image, center)
    rough = azimuthal.med(image, coarse)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 3.8),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1.4]))
    axes[0].imshow(image, **STYLE)
    axes[0].plot(*coarse, 'x', color='tab:orange', ms=9, mew=2,
                 label='circle transform')
    axes[0].plot(*center, '+', color='c', ms=14, mew=2,
                 label='refined')
    axes[0].set_title('hologram of one sphere')
    axes[0].legend(loc='lower right', fontsize=8, framealpha=0.8)
    axes[0].axis('off')
    axes[1].plot(np.arange(len(rough)), rough, color='tab:orange', lw=1,
                 alpha=0.8, label='about the circle transform peak')
    axes[1].plot(np.arange(len(median)), median, color='C0', lw=1.5,
                 label='about the refined center')
    axes[1].axhline(1., color='0.5', lw=0.6, ls=':')
    axes[1].set_xlim(0, min(center) + 5)    # beyond it only the corners
    axes[1].set_xlabel('distance from the center [pixels]')
    axes[1].set_ylabel('normalized intensity')
    axes[1].set_title('azimuthal.med(hologram, center)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / 'azimuthal.png', dpi=110)
    plt.close(fig)
    print('azimuthal.png: coarse', coarse, 'refined',
          tuple(round(c, 2) for c in center))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    transform_figure()
    profile_figure()
