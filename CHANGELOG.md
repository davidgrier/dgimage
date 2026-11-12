# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
