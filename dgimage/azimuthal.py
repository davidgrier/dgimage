'''Azimuthal statistics of an image: a profile as a function of radius.

The pixels of a gray image are binned by their distance from a center, in
whole pixels, and a statistic of each bin is returned as a function of that
radius: the average, the standard deviation, the median or the median
absolute deviation.  The profile of the rings of the hologram of a sphere is
the usual use, but any image with a center will do.

The geometry of the binning (the radius of every pixel, and their order)
depends only on the shape of the image and the center, so it is cached: a
series of frames of one shape, about one center, pays for it once.

Examples
--------
>>> import numpy as np
>>> from dgimage import azimuthal
>>> y, x = np.indices((9, 9))
>>> image = np.hypot(x - 4, y - 4)
>>> profile = azimuthal.avg(image, center=(4, 4))     # (x, y)
>>> [round(float(v), 1) for v in profile[:4]]
[0.0, 1.2, 2.3, 3.3]
'''
from __future__ import annotations

from collections.abc import Callable
from functools import cached_property, lru_cache, wraps
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ['azimuthaloperator', 'avg', 'std', 'med', 'mad']

_Data = NDArray[np.floating]
_Center = tuple[float, float] | NDArray[np.floating]
_Profile = NDArray[np.floating]


class _Context:

    '''The radial geometry of an image shape and a center.

    The radius of every pixel (in whole pixels, raveled) is made at once; the
    order of the pixels by radius, which only the median needs, and the
    counts and boundaries of the radii are made the first time they are
    asked for.  A statistic that does not need them (the average, the
    standard deviation) never pays for the sort, which is most of the time
    for a center that has not been seen before.
    '''

    def __init__(self, ny: int, nx: int, x_p: float, y_p: float) -> None:
        x = np.arange(nx) - x_p
        y = np.arange(ny) - y_p
        self.r: NDArray[np.int32] = np.hypot.outer(y, x).astype(
            np.int32).ravel()
        self.nmax = int(self.r.max()) + 1       # number of radii

    @cached_property
    def sort_idx(self) -> NDArray[np.intp]:
        '''The pixels by radius (stable).  Radii that fit 16 bits sort by
        radix sort, ten times faster than the generic stable sort and in the
        same order.'''
        r = self.r.astype(np.uint16) if self.nmax <= 0xFFFF else self.r
        return np.argsort(r, kind='stable')

    @cached_property
    def counts(self) -> NDArray[np.intp]:
        '''The number of pixels at each radius.'''
        return np.bincount(self.r, minlength=self.nmax)

    @cached_property
    def boundaries(self) -> NDArray[np.intp]:
        '''Where each radius starts in the order by radius; one more
        entry than radii, the last being the number of pixels.'''
        return np.concatenate(([0], np.cumsum(self.counts)))


@lru_cache(maxsize=16)
def _build_context(ny: int, nx: int, x_p: float, y_p: float) -> _Context:
    '''Return the radial geometry of (ny, nx) about (x_p, y_p), kept.'''
    return _Context(ny, nx, x_p, y_p)


def azimuthaloperator(func: Callable[[_Data, _Context], Any]
                      ) -> Callable[..., Any]:
    '''Decorate a function of radius-binned data to make a function of an
    image.

    A function decorated with ``@azimuthaloperator`` takes the raveled data
    and the radial geometry (``r``, ``sort_idx``, ``boundaries`` and
    ``nmax``) and returns a profile.  The decorated function takes
    ``f(image, center=None)``, where *center* is an ``(x, y)`` pair in
    pixels.  The default is ``(nx / 2, ny / 2)``, which for an axis of 9
    pixels is 4.5, not the index of the middle pixel, 4: give the center
    when it matters.  The geometry is cached by shape and center.

    Raises
    ------
    ValueError
        (when the decorated function is called) If the image is not two
        dimensional.
    '''
    @wraps(func)
    def operator(image: ArrayLike, center: _Center | None = None) -> Any:
        data = np.asarray(image)
        if data.ndim != 2:
            raise ValueError('the image must be two dimensional, not '
                             f'{data.ndim} dimensional')
        ny, nx = data.shape
        x_p, y_p = (nx / 2., ny / 2.) if center is None else center
        context = _build_context(ny, nx, float(x_p), float(y_p))
        return func(data.ravel(), context)

    return operator


@azimuthaloperator
def avg(d: _Data, ctx: _Context) -> _Profile:
    '''Azimuthal average of a 2-D image.

    Parameters
    ----------
    image : array_like
        Two-dimensional data.
    center : tuple of float, optional
        ``(x, y)`` center of the average.  Default: ``(nx / 2, ny / 2)``.

    Returns
    -------
    numpy.ndarray
        The average value as a function of the distance from the center,
        in whole pixels.
    '''
    return np.bincount(ctx.r, d) / ctx.counts


@azimuthaloperator
def std(d: _Data, ctx: _Context) -> tuple[_Profile, _Profile]:
    '''Azimuthal standard deviation of a 2-D image.

    Parameters
    ----------
    image : array_like
        Two-dimensional data.
    center : tuple of float, optional
        ``(x, y)`` center.  Default: ``(nx / 2, ny / 2)``.

    Returns
    -------
    avg : numpy.ndarray
        The azimuthal average as a function of the distance from the center.
    std : numpy.ndarray
        The azimuthal standard deviation.
    '''
    nr = ctx.counts
    a = np.bincount(ctx.r, d) / nr
    s = np.sqrt(np.bincount(ctx.r, (d - a[ctx.r]) ** 2) / nr)
    return a, s


@azimuthaloperator
def med(d: _Data, ctx: _Context) -> _Profile:
    '''Azimuthal median of a 2-D image.

    Parameters
    ----------
    image : array_like
        Two-dimensional data.
    center : tuple of float, optional
        ``(x, y)`` center.  Default: ``(nx / 2, ny / 2)``.

    Returns
    -------
    numpy.ndarray
        The median as a function of the distance from the center.  A radius
        with no pixels (a center outside the image) is NaN.
    '''
    ordered = d[ctx.sort_idx]
    result = np.empty(ctx.nmax)
    for n in range(ctx.nmax):
        start, stop = ctx.boundaries[n], ctx.boundaries[n + 1]
        result[n] = np.median(ordered[start:stop]) if start < stop else np.nan
    return result


@azimuthaloperator
def mad(d: _Data, ctx: _Context) -> tuple[_Profile, _Profile]:
    '''Azimuthal median and median absolute deviation of a 2-D image.

    Parameters
    ----------
    image : array_like
        Two-dimensional data.
    center : tuple of float, optional
        ``(x, y)`` center.  Default: ``(nx / 2, ny / 2)``.

    Returns
    -------
    med : numpy.ndarray
        The azimuthal median.  A radius with no pixels is NaN.
    mad : numpy.ndarray
        The azimuthal median absolute deviation, which is robust against
        outliers such as dust.
    '''
    ordered = d[ctx.sort_idx]
    m = np.empty(ctx.nmax)
    dev = np.empty(ctx.nmax)
    for n in range(ctx.nmax):
        start, stop = ctx.boundaries[n], ctx.boundaries[n + 1]
        if start < stop:
            ring = ordered[start:stop]
            m[n] = np.median(ring)
            dev[n] = np.median(np.abs(ring - m[n]))
        else:
            m[n] = dev[n] = np.nan
    return m, dev
