Quickstart
==========

The orientation alignment transform
-----------------------------------

The rings of the hologram of a sphere are circularly symmetric about the
position of the sphere, whatever its size.  The orientation alignment
transform [Krishnatreya2014]_ turns an image into a map whose peaks are the
centers of such rings::

    import numpy as np
    from dgimage import circletransform

    heat = circletransform(image)
    y, x = np.unravel_index(heat.argmax(), heat.shape)

For example, a real hologram of two silica spheres (from
`pylorenzmie <https://github.com/davidgrier/pylorenzmie>`_) becomes two
sharp peaks, one at the center of each set of rings:

.. image:: _static/circletransform.png
   :alt: A hologram of two spheres and its orientation alignment transform,
         which has one peak at the center of each sphere.

``image`` is a gray image of any real type, ``uint8`` or ``uint16`` counts
included, and the scale of its pixels does not matter.  The result is a
``float`` array of the same shape, scaled so that its peak is 1; a uniform
image gives zeros.  To find several particles, look for the local maxima of
the map, for example with ``scipy.ndimage.maximum_filter`` or
``skimage.feature.peak_local_max``.

Every pixel contributes the direction of its gradient, computed with a
Savitzky-Golay filter of width ``window`` (13 pixels) and order ``polyorder``
(3), the values of the paper.  A wider window smooths noise and blurs the
centers::

    heat = circletransform(image, window=7, polyorder=2)

Many frames
-----------

The Fourier space kernel depends on the shape of the image only.  A program
that transforms a series of frames builds a :class:`~dgimage.CircleTransform`
once and calls it::

    from dgimage import CircleTransform

    transform = CircleTransform()
    for frame in frames:
        heat = transform(frame)

Radial profiles
---------------

The profile of a ring pattern about its center, the average of the pixels at
each distance in whole pixels, comes from the :mod:`~dgimage.azimuthal`
module::

    from dgimage import azimuthal

    profile = azimuthal.avg(image, center=(x, y))
    mean, spread = azimuthal.std(image, center=(x, y))
    median = azimuthal.med(image, center=(x, y))
    median, deviation = azimuthal.mad(image, center=(x, y))

The profile of the hologram of one sphere, about the center that the circle
transform found, shows the fringes as a function of radius.  The median
follows the average closely here, but ignores a speck of dust that the
average would feel:

.. image:: _static/azimuthal.png
   :alt: A hologram of one sphere and its azimuthal median and average as
         functions of the distance from the center of the sphere.

``center`` is ``(x, y)`` in pixels.  Without it the center is
``(nx / 2, ny / 2)``, which is half a pixel beyond the middle pixel of an
axis of odd length, so give the center when it matters.  The median and its
absolute deviation are robust against outliers such as dust; a radius that
contains no pixels is ``NaN``.  The radial geometry depends only on the shape
and the center and is cached, so a series of frames about one center pays for
it once.

.. [Krishnatreya2014] B. J. Krishnatreya and D. G. Grier, "Fast feature
   identification for holographic tracking: the orientation alignment
   transform," *Optics Express* **22**, 12773-12778 (2014).
