Quickstart
==========

The orientation alignment transform
-----------------------------------

The rings of the hologram of a sphere are circularly symmetric about the
position of the sphere, whatever its size.  The orientation alignment
transform [Krishnatreya2014]_ turns an image into a map whose peaks are the
centers of such rings::

    import numpy as np
    from imagetransforms import circletransform

    heat = circletransform(image)
    y, x = np.unravel_index(heat.argmax(), heat.shape)

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
that transforms a series of frames builds a :class:`~imagetransforms.CircleTransform`
once and calls it::

    from imagetransforms import CircleTransform

    transform = CircleTransform()
    for frame in frames:
        heat = transform(frame)

.. [Krishnatreya2014] B. J. Krishnatreya and D. G. Grier, "Fast feature
   identification for holographic tracking: the orientation alignment
   transform," *Optics Express* **22**, 12773-12778 (2014).
