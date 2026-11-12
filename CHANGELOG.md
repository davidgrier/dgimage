# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
