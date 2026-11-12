# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Two options for a faster circle transform that give up some accuracy**,
  keyword-only and off by default, on `circletransform` and `CircleTransform`:
  `dtype=np.float32` (single precision: 1.7 times faster; the map differs by
  3e-7 of its peak, and the peak is where it was) and `binning=n` (average
  blocks of `n` by `n` pixels first; the map is on the coarser grid, of shape
  `(ny // n, nx // n)`; 2.3 times faster for 2 and 6 times for 4; the peak of
  a sphere is one blob whose centroid agrees with the full map's to about a
  pixel, so locate by the centroid and refine the center in the full image).
  The times are for a real 1280 x 1024 hologram with OpenCV.
- **A faster `azimuthal` for a new center.**  The order of the pixels by
  radius, which only the median needs, is made when it is first asked for, and
  with a 16 bit radix sort (ten times faster than the generic stable sort, in
  the same order).  The average of a 1280 x 1024 image about a center that
  has not been used takes 9 ms instead of 88 ms; the numbers are identical.
  A search that moves the center (a refinement) benefits most.

### Changed

- **A faster circle transform with the same numbers.**  The Savitzky-Golay
  derivatives, which took half of the time, are now a correlation (OpenCV's
  `filter2D` if it is installed, `scipy.ndimage.correlate1d` if not) with the
  ends of the image evaluated by the matrix of scipy's own polynomial fit, so
  the result is that of `scipy.signal.savgol_filter` to rounding (3e-15 of the
  peak, tested against it and against pylorenzmie's transform).  A frame of
  1280 x 1024 takes 15 ms with OpenCV and 25 ms without, against 29 ms; one of
  2048 x 2048 takes 43 ms and 95 ms, against 110 ms.  OpenCV stays optional:
  `pip install dgimage[fast]`.

## [0.1.0] - unreleased

### Added

- `circletransform(image, window=13, polyorder=3)` and `CircleTransform`, the
  orientation alignment transform of Krishnatreya and Grier (2014), for
  locating the centers of rings.  The map is `float`, scaled so that its peak
  is 1; the pixels of the image may be of any real type.  Written to be the
  one implementation that QVideo and pylorenzmie share.
- `azimuthal`: `avg`, `std`, `med` and `mad`, the azimuthal average, standard
  deviation, median and median absolute deviation of an image as a function of
  the distance from a center, and `azimuthaloperator`, the decorator that
  makes a statistic of radius-binned data a function of an image.  The radial
  geometry is cached by shape and center.  The results are identical to those
  of `pylorenzmie.lib.Azimuthal`, from which the module comes.
