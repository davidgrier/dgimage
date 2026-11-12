'''The orientation alignment transform: a map of the centers of rings.

The transform of Krishnatreya and Grier locates circularly symmetric
features, such as the rings of the hologram of a sphere, without knowing
their radius.  Every pixel carries the direction of the gradient of the
image; each pixel votes, by that direction, for the center of a ring that
could have produced it, and the votes of all radii add in Fourier space.
The peaks of the result are the centers of the rings.

The defaults reproduce the published algorithm in double precision.  Two
options trade accuracy for speed, and are off unless asked for: ``dtype`` for
single precision (1.7 times faster on a frame of 1280 by 1024, with a
difference of 3e-7 of the peak) and ``binning`` for a map on a coarser grid
(2.3 times faster for 2, 6 times for 4, with a peak that moves by about the
width of the peak, 10 to 20 pixels in a hologram of a sphere).

References
----------
B. J. Krishnatreya and D. G. Grier, 'Fast feature identification for
holographic tracking: the orientation alignment transform,' *Optics Express*
**22**, 12773-12778 (2014).
'''
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft, ndimage, signal

__all__ = ['CircleTransform', 'circletransform']

#: The width [pixels] and order of the Savitzky-Golay filter that
#: differentiates the image; the values of the published algorithm.
WINDOW = 13
POLYORDER = 3


_OPENCV: list[Any] = []     # [module or None], filled on first use

_DTYPES = {np.dtype(np.float64): np.dtype(np.complex128),
           np.dtype(np.float32): np.dtype(np.complex64)}


def _opencv() -> Any:
    '''Return OpenCV, which is optional, or ``None`` without it.'''
    if not _OPENCV:
        try:
            import cv2
        except ImportError:
            cv2 = None
        _OPENCV.append(cv2)
    return _OPENCV[0]


@lru_cache(maxsize=16)
def _operator(window: int, polyorder: int
              ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    '''The taps of the derivative filter and its edge operator.

    ``scipy.signal.savgol_filter`` correlates the interior of a signal with
    the taps and, within half a window of an end, evaluates the derivative of
    the polynomial that fits the first (or last) *window* samples.  That fit
    is linear, so it is a matrix; scipy's own filter applied to the identity
    gives it exactly.
    '''
    taps = signal.savgol_coeffs(window, polyorder, deriv=1, use='dot')
    edge = np.asarray(signal.savgol_filter(np.eye(window), window, polyorder,
                                           deriv=1, axis=0))
    return taps, edge


def _derivative(gray: NDArray[np.floating], window: int, polyorder: int,
                axis: int) -> NDArray[np.floating]:
    '''The Savitzky-Golay derivative of *gray* along *axis*, in its dtype.

    The same numbers as ``scipy.signal.savgol_filter(gray, window,
    polyorder, deriv=1, axis=axis)``, to rounding (1e-12 of the scale of the
    image in double precision), three to ten times faster with OpenCV and 1.5
    to 2 times without.
    '''
    if min(gray.shape) < window:            # scipy says what is wrong
        return np.asarray(signal.savgol_filter(gray, window, polyorder, 1,
                                               axis=axis))
    taps, edge = _operator(window, polyorder)
    taps = taps.astype(gray.dtype)
    edge = edge.astype(gray.dtype)
    half = window // 2
    cv2 = _opencv()
    if cv2 is not None:
        kernel = taps[None, :] if axis == 1 else taps[:, None]
        out = cv2.filter2D(np.ascontiguousarray(gray), -1, kernel,
                           borderType=cv2.BORDER_CONSTANT)
    else:
        out = np.moveaxis(ndimage.correlate1d(
            np.moveaxis(gray, axis, -1), taps, axis=-1, mode='constant'),
            -1, axis)
        out = np.ascontiguousarray(out)
    if axis == 1:
        out[:, :half] = gray[:, :window] @ edge[:half].T
        out[:, -half:] = gray[:, -window:] @ edge[-half:].T
    else:
        out[:half] = edge[:half] @ gray[:window]
        out[-half:] = edge[-half:] @ gray[-window:]
    return out


def _bin(gray: NDArray[np.floating], binning: int) -> NDArray[np.floating]:
    '''The mean of blocks of *binning* by *binning* pixels; rows and columns
    that do not fill a block are dropped.'''
    ny, nx = (n // binning for n in gray.shape)
    kept = gray[:ny * binning, :nx * binning]
    mean = np.zeros((ny, nx), dtype=gray.dtype)
    for i in range(binning):                # adding slices is the fastest
        for j in range(binning):            # way to average blocks
            mean += kept[i::binning, j::binning]
    mean /= binning ** 2
    return mean


class CircleTransform:

    '''The orientation alignment transform, with its kernel kept.

    The kernel depends only on the shape of the image and the precision, so
    an instance that transforms many frames of one shape computes it once.

    Parameters
    ----------
    window : int
        Width [pixels] of the Savitzky-Golay filter that takes the
        derivatives.  An even value is rounded up to the next odd number.
        Larger values smooth noise but blur the centers.  Default: 13.
    polyorder : int
        Order of the polynomial of that filter, less than *window*.
        Default: 3.
    dtype : {numpy.float64, numpy.float32}
        Precision of the calculation and of the map.  ``numpy.float32`` is
        about 1.7 times faster; the map differs from the double precision map
        by 3e-7 of its peak, and the peak is in the same place.
        Default: ``numpy.float64``, the reference.
    binning : int
        Average blocks of *binning* by *binning* pixels before transforming.
        The map is on the coarser grid, of shape ``(ny // binning,
        nx // binning)``: a position in it is *binning* times smaller than
        in the image.  A frame of 1280 by 1024 takes 2.3 times less time with
        2 and 6 times less with 4.  The cost is position: the peak of a
        sphere moved by 12 to 20 pixels of the image, about its own width, so
        use the map to find where to look and refine the center in the full
        image (for example with :mod:`~dgimage.azimuthal`).  The rings must
        be coarser than the blocks, or averaging washes them out, and
        *window* and *polyorder* apply to the binned image, so rings of a
        given size are smoothed *binning* times more.  Default: 1, the image
        as it is.
    '''

    def __init__(self, window: int = WINDOW, polyorder: int = POLYORDER, *,
                 dtype: Any = np.float64, binning: int = 1) -> None:
        self._kernels: dict[tuple, NDArray[np.complexfloating]] = {}
        self.window = window
        self.polyorder = polyorder
        self.dtype = dtype
        self.binning = binning

    @property
    def window(self) -> int:
        '''Width of the derivative filter [pixels]; odd, at least 3.'''
        return self._window

    @window.setter
    def window(self, window: int) -> None:
        window = max(3, int(window))
        self._window = window + (1 - window % 2)

    @property
    def polyorder(self) -> int:
        '''Order of the polynomial of the derivative filter, at least 1.'''
        return self._polyorder

    @polyorder.setter
    def polyorder(self, polyorder: int) -> None:
        self._polyorder = max(1, int(polyorder))

    @property
    def dtype(self) -> np.dtype:
        '''Precision of the calculation: ``float64`` or ``float32``.'''
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Any) -> None:
        dtype = np.dtype(dtype)
        if dtype not in _DTYPES:
            raise ValueError('dtype must be float64 or float32, not '
                             f'{dtype}')
        self._dtype = dtype

    @property
    def binning(self) -> int:
        '''Size of the blocks that are averaged first; at least 1.'''
        return self._binning

    @binning.setter
    def binning(self, binning: int) -> None:
        self._binning = max(1, int(binning))

    def kernel(self, shape: tuple[int, int]) -> NDArray[np.complexfloating]:
        '''Return the Fourier space kernel for images of *shape*.

        The kernel is :math:`e^{-2i\\theta_k}/|\\mathbf{k}|`, arranged
        for the order of the FFT, in the precision of the transform
        (``complex128`` for ``float64``, ``complex64`` for ``float32``).
        '''
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, self._dtype)
        kernel = self._kernels.get(key)
        if kernel is not None:
            return kernel
        ny, nx = shape
        kx = fft.fftshift(np.linspace(-1., 1., nx, endpoint=False))
        ky = fft.fftshift(np.linspace(-1., 1., ny, endpoint=False))
        k = np.hypot.outer(ky, kx) + 0.001      # keeps the DC term finite
        kernel = np.subtract.outer(1.j * ky, kx) / k
        kernel *= kernel / k
        kernel = kernel.astype(_DTYPES[self._dtype])
        self._kernels.clear()                   # one shape at a time
        self._kernels[key] = kernel
        return kernel

    def transform(self, image: ArrayLike) -> NDArray[np.floating]:
        '''Return the transform of *image*, scaled so that its peak is 1.

        Parameters
        ----------
        image : array_like
            A gray image of shape ``(ny, nx)``, of any real type.  The
            scale of the pixels does not matter.

        Returns
        -------
        numpy.ndarray
            Array of the :attr:`dtype` of the transform, between 0 and 1,
            of the shape of *image* (of ``(ny // binning, nx // binning)``
            if :attr:`binning` is more than 1).  A uniform image gives
            zeros.

        Raises
        ------
        ValueError
            If *image* is not two dimensional, is smaller than one block of
            the binning, or the filter does not fit (``polyorder`` must be
            less than ``window``).
        '''
        gray = np.asarray(image, dtype=self._dtype)
        if gray.ndim != 2:
            raise ValueError('the image must be two dimensional, not '
                             f'{gray.ndim} dimensional')
        if self._polyorder >= self._window:
            raise ValueError('polyorder must be less than window')
        if self._binning > 1:
            if min(gray.shape) < self._binning:
                raise ValueError('the image is smaller than one block of '
                                 f'{self._binning} pixels')
            gray = _bin(gray, self._binning)
        psi = np.empty(gray.shape, dtype=_DTYPES[self._dtype])
        psi.real = _derivative(gray, self._window, self._polyorder, 1)
        psi.imag = _derivative(gray, self._window, self._polyorder, 0)
        psi *= psi
        psi = np.asarray(fft.fft2(psi, workers=-1, overwrite_x=True))
        psi *= self.kernel(gray.shape)
        psi = np.asarray(fft.ifft2(psi, workers=-1, overwrite_x=True))
        c = np.square(np.real(psi)) + np.square(np.imag(psi))
        peak = c.max()
        if peak > np.finfo(self._dtype).eps:
            c /= peak
        else:
            c[:] = 0.
        return c

    __call__ = transform


_default = CircleTransform()


def circletransform(image: ArrayLike, window: int = WINDOW,
                    polyorder: int = POLYORDER, *, dtype: Any = np.float64,
                    binning: int = 1) -> NDArray[np.floating]:
    '''Return the orientation alignment transform of *image*.

    A function for one call; a program that transforms many frames builds a
    :class:`CircleTransform`, which keeps its kernel.

    Parameters
    ----------
    image : array_like
        A gray image of shape ``(ny, nx)``.
    window : int
        Width of the derivative filter [pixels].  Default: 13.
    polyorder : int
        Order of the derivative filter.  Default: 3.
    dtype : {numpy.float64, numpy.float32}
        Precision.  ``numpy.float32`` is 1.7 times faster and differs from
        the double precision map by 3e-7 of its peak.
        Default: ``numpy.float64``.
    binning : int
        Average blocks of this many pixels first, for a map on a coarser grid
        that takes 2.3 times less time for 2 and 6 times less for 4, with a
        peak that is only as good as the width of the peak (see
        :class:`CircleTransform`).  Default: 1.

    Returns
    -------
    numpy.ndarray
        The map of the centers of rings, with its peak scaled to 1.

    Examples
    --------
    >>> import numpy as np
    >>> y, x = np.indices((96, 128))
    >>> rings = np.cos(0.8 * np.hypot(x - 70, y - 40))
    >>> heat = circletransform(rings)
    >>> tuple(int(i) for i in np.unravel_index(heat.argmax(), heat.shape))
    (40, 70)
    '''
    if (window == WINDOW and polyorder == POLYORDER
            and np.dtype(dtype) == np.float64 and binning == 1):
        return _default.transform(image)
    return CircleTransform(window, polyorder, dtype=dtype,
                           binning=binning).transform(image)
