'''The orientation alignment transform: a map of the centers of rings.

The transform of Krishnatreya and Grier locates circularly symmetric
features, such as the rings of the hologram of a sphere, without knowing
their radius.  Every pixel carries the direction of the gradient of the
image; each pixel votes, by that direction, for the center of a ring that
could have produced it, and the votes of all radii add in Fourier space.
The peaks of the result are the centers of the rings.

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
    '''The Savitzky-Golay derivative of *gray* along *axis*.

    The same numbers as ``scipy.signal.savgol_filter(gray, window,
    polyorder, deriv=1, axis=axis)``, to rounding (1e-12 of the scale of the
    image), three to ten times faster with OpenCV and 1.5 to 2 times without.
    '''
    if min(gray.shape) < window:            # scipy says what is wrong
        return np.asarray(signal.savgol_filter(gray, window, polyorder, 1,
                                               axis=axis))
    taps, edge = _operator(window, polyorder)
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


class CircleTransform:

    '''The orientation alignment transform, with its kernel kept.

    The kernel depends only on the shape of the image, so an instance that
    transforms many frames of one shape computes it once.

    Parameters
    ----------
    window : int
        Width [pixels] of the Savitzky-Golay filter that takes the
        derivatives.  An even value is rounded up to the next odd number.
        Larger values smooth noise but blur the centers.  Default: 13.
    polyorder : int
        Order of the polynomial of that filter, less than *window*.
        Default: 3.
    '''

    def __init__(self, window: int = WINDOW,
                 polyorder: int = POLYORDER) -> None:
        self._kernel = np.ones((1, 1), dtype=complex)
        self.window = window
        self.polyorder = polyorder

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

    def kernel(self, shape: tuple[int, int]) -> NDArray[np.complexfloating]:
        '''Return the Fourier space kernel for images of *shape*.

        The kernel is :math:`e^{-2i\\theta_k}/|\\mathbf{k}|`, arranged
        for the order of the FFT.
        '''
        shape = (int(shape[0]), int(shape[1]))
        if shape == self._kernel.shape:
            return self._kernel
        ny, nx = shape
        kx = fft.fftshift(np.linspace(-1., 1., nx, endpoint=False))
        ky = fft.fftshift(np.linspace(-1., 1., ny, endpoint=False))
        k = np.hypot.outer(ky, kx) + 0.001      # keeps the DC term finite
        kernel = np.subtract.outer(1.j * ky, kx) / k
        kernel *= kernel / k
        self._kernel = kernel
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
            ``float`` array of the shape of *image*, between 0 and 1.
            A uniform image gives zeros.

        Raises
        ------
        ValueError
            If *image* is not two dimensional, or the filter does not fit
            (``polyorder`` must be less than ``window``).
        '''
        gray = np.asarray(image, dtype=float)
        if gray.ndim != 2:
            raise ValueError('the image must be two dimensional, not '
                             f'{gray.ndim} dimensional')
        if self._polyorder >= self._window:
            raise ValueError('polyorder must be less than window')
        psi = np.empty(gray.shape, dtype=complex)
        psi.real = _derivative(gray, self._window, self._polyorder, 1)
        psi.imag = _derivative(gray, self._window, self._polyorder, 0)
        psi *= psi
        psi = np.asarray(fft.fft2(psi, workers=-1, overwrite_x=True))
        psi *= self.kernel(gray.shape)
        psi = np.asarray(fft.ifft2(psi, workers=-1, overwrite_x=True))
        c = np.square(np.real(psi)) + np.square(np.imag(psi))
        peak = c.max()
        if peak > np.finfo(float).eps:
            c /= peak
        else:
            c[:] = 0.
        return c

    __call__ = transform


_default = CircleTransform()


def circletransform(image: ArrayLike, window: int = WINDOW,
                    polyorder: int = POLYORDER) -> NDArray[np.floating]:
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
    if window == WINDOW and polyorder == POLYORDER:
        return _default.transform(image)
    return CircleTransform(window, polyorder).transform(image)
