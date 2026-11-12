'''Image transformations for holography and microscopy.

A small library of the transformations that the group of David Grier uses
across projects: lightweight (numpy and scipy only) and independent of any
camera, GUI or analysis package, so that :mod:`QVideo`, :mod:`pylorenzmie`
and others share one implementation of each.
'''
from importlib.metadata import PackageNotFoundError, version

from imagetransforms import azimuthal
from imagetransforms.circletransform import CircleTransform, circletransform

try:
    __version__ = version('imagetransforms')
except PackageNotFoundError:        # not installed
    __version__ = '0+unknown'

__all__ = ['azimuthal', 'CircleTransform', 'circletransform']
