'''Tests of the orientation alignment transform.'''
import sys
import unittest
from unittest import mock

import numpy as np
from scipy import fft, signal

from dgimage import CircleTransform, circletransform
from dgimage.circletransform import _OPENCV, _derivative, _opencv


def rings(shape=(96, 128), center=(40, 70), k=0.8):
    '''Concentric rings about *center* (y, x).'''
    y, x = np.indices(shape)
    return np.cos(k * np.hypot(x - center[1], y - center[0]))


def peak(image):
    return tuple(int(i) for i in np.unravel_index(image.argmax(),
                                                  image.shape))


class TestTransform(unittest.TestCase):

    def test_the_peak_is_the_center_of_the_rings(self):
        self.assertEqual(peak(circletransform(rings())), (40, 70))

    def test_another_center(self):
        self.assertEqual(peak(circletransform(rings(center=(60, 33)))),
                         (60, 33))

    def test_the_map_is_float_between_zero_and_one(self):
        heat = circletransform(rings())
        self.assertEqual(heat.shape, (96, 128))
        self.assertEqual(heat.dtype, np.float64)
        self.assertAlmostEqual(float(heat.max()), 1.)
        self.assertGreaterEqual(float(heat.min()), 0.)

    def test_a_uniform_image_gives_zeros(self):
        heat = circletransform(np.full((32, 40), 7.))
        self.assertTrue(np.all(heat == 0.))

    def test_the_scale_of_the_pixels_does_not_matter(self):
        image = rings()
        np.testing.assert_allclose(circletransform(image),
                                   circletransform(1000. * image + 5.),
                                   atol=1e-9)

    def test_integer_images_are_accepted(self):
        counts = np.round(2047.5 * (1. + rings())).astype(np.uint16)
        self.assertEqual(peak(circletransform(counts)), (40, 70))
        eight = (counts >> 4).astype(np.uint8)
        self.assertEqual(peak(circletransform(eight)), (40, 70))

    def test_a_list_is_accepted(self):
        self.assertEqual(circletransform(rings().tolist()).shape, (96, 128))

    def test_a_color_image_is_refused(self):
        with self.assertRaises(ValueError):
            circletransform(np.zeros((8, 8, 3)))

    def test_the_input_is_not_changed(self):
        image = rings()
        before = image.copy()
        circletransform(image)
        np.testing.assert_array_equal(image, before)


class TestParameters(unittest.TestCase):

    def test_defaults_are_the_published_ones(self):
        transform = CircleTransform()
        self.assertEqual((transform.window, transform.polyorder), (13, 3))

    def test_an_even_window_is_made_odd(self):
        self.assertEqual(CircleTransform(window=12).window, 13)

    def test_the_window_is_at_least_three(self):
        self.assertEqual(CircleTransform(window=0).window, 3)

    def test_the_order_is_at_least_one(self):
        self.assertEqual(CircleTransform(polyorder=0).polyorder, 1)

    def test_an_order_that_does_not_fit_is_refused(self):
        with self.assertRaises(ValueError):
            CircleTransform(window=5, polyorder=5).transform(rings())

    def test_the_function_passes_the_parameters(self):
        image = rings()
        np.testing.assert_allclose(
            circletransform(image, window=7, polyorder=2),
            CircleTransform(7, 2).transform(image))

    def test_a_different_window_gives_a_different_map(self):
        image = rings() + 0.05 * np.random.default_rng(0).normal(
            size=(96, 128))
        self.assertFalse(np.allclose(circletransform(image),
                                     circletransform(image, window=7)))


class TestKernel(unittest.TestCase):

    def test_it_is_kept_for_a_shape(self):
        transform = CircleTransform()
        first = transform.kernel((32, 48))
        self.assertIs(transform.kernel((32, 48)), first)

    def test_it_follows_a_change_of_shape(self):
        transform = CircleTransform()
        transform.transform(rings((64, 64), (30, 30)))
        small = transform.transform(rings((48, 80), (20, 50)))
        self.assertEqual(small.shape, (48, 80))
        self.assertEqual(peak(small), (20, 50))

    def test_calling_the_object_transforms(self):
        transform = CircleTransform()
        np.testing.assert_array_equal(transform(rings()),
                                      transform.transform(rings()))


def reference(image, window=13, polyorder=3):
    '''The transform as first written, with scipy's savgol_filter.'''
    gray = np.asarray(image, dtype=float)
    psi = np.empty(gray.shape, dtype=complex)
    psi.real = signal.savgol_filter(gray, window, polyorder, 1, axis=1)
    psi.imag = signal.savgol_filter(gray, window, polyorder, 1, axis=0)
    psi *= psi
    psi = fft.fft2(psi)
    psi *= CircleTransform(window, polyorder).kernel(gray.shape)
    psi = fft.ifft2(psi)
    c = psi.real ** 2 + psi.imag ** 2
    return c / c.max()


def backends():
    '''The ways to take a derivative: with OpenCV if it is there, and
    without.'''
    yield 'scipy', mock.patch('dgimage.circletransform._opencv',
                              return_value=None)
    if _opencv() is not None:
        yield 'opencv', mock.patch('dgimage.circletransform._opencv',
                                   wraps=_opencv)


class TestDerivative(unittest.TestCase):

    '''The derivative is scipy's, to rounding, with or without OpenCV.'''

    def setUp(self):
        rng = np.random.default_rng(3)
        self.image = 100. * rng.random((60, 75)) + np.arange(75)

    def test_it_equals_the_filter_of_scipy(self):
        for name, patch in backends():
            for window, polyorder in ((13, 3), (7, 2), (5, 1), (21, 4)):
                for axis in (0, 1):
                    with self.subTest(backend=name, window=window,
                                      polyorder=polyorder, axis=axis), patch:
                        ours = _derivative(self.image, window, polyorder, axis)
                        theirs = signal.savgol_filter(
                            self.image, window, polyorder, 1, axis=axis)
                        np.testing.assert_allclose(
                            ours, theirs, atol=1e-9 * np.abs(theirs).max())

    def test_the_edges_are_scipys_polynomial_fits(self):
        # a cubic is differentiated exactly all the way to the ends
        x = np.arange(40.)
        image = np.tile(0.01 * x ** 3 - x, (30, 1))
        for name, patch in backends():
            with self.subTest(backend=name), patch:
                out = _derivative(image, 13, 3, 1)
                np.testing.assert_allclose(
                    out[0], 0.03 * x ** 2 - 1., atol=1e-8)

    def test_an_image_as_small_as_the_window(self):
        image = self.image[:13, :13]
        for name, patch in backends():
            for axis in (0, 1):
                with self.subTest(backend=name, axis=axis), patch:
                    np.testing.assert_allclose(
                        _derivative(image, 13, 3, axis),
                        signal.savgol_filter(image, 13, 3, 1, axis=axis),
                        atol=1e-9)

    def test_an_image_smaller_than_the_window_is_refused_as_before(self):
        for name, patch in backends():
            with self.subTest(backend=name), patch:
                with self.assertRaises(ValueError):
                    circletransform(np.zeros((8, 8)))

    def test_a_transposed_image(self):
        image = self.image.T
        for name, patch in backends():
            with self.subTest(backend=name), patch:
                np.testing.assert_allclose(
                    _derivative(image, 13, 3, 0),
                    signal.savgol_filter(image, 13, 3, 1, axis=0),
                    atol=1e-8)

    def test_the_whole_transform_equals_the_original_algorithm(self):
        image = rings() + 0.05 * np.random.default_rng(0).normal(
            size=(96, 128))
        for name, patch in backends():
            for window, polyorder in ((13, 3), (9, 2)):
                with self.subTest(backend=name, window=window), patch:
                    np.testing.assert_allclose(
                        CircleTransform(window, polyorder).transform(image),
                        reference(image, window, polyorder), atol=1e-9)


class TestSinglePrecision(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(5)
        self.image = rings((96, 128), (40, 70)) + 0.1 * rng.normal(
            size=(96, 128))

    def test_the_map_is_float32(self):
        heat = circletransform(self.image, dtype=np.float32)
        self.assertEqual(heat.dtype, np.float32)
        self.assertEqual(heat.shape, (96, 128))

    def test_the_default_is_double_precision(self):
        self.assertEqual(CircleTransform().dtype, np.float64)
        self.assertEqual(circletransform(self.image).dtype, np.float64)

    def test_it_agrees_with_double_precision(self):
        for name, patch in backends():
            with self.subTest(backend=name), patch:
                single = circletransform(self.image, dtype=np.float32)
                double = circletransform(self.image)
                self.assertLess(float(np.abs(single - double).max()), 1e-5)
                self.assertEqual(peak(single), peak(double))

    def test_the_dtype_may_be_a_string_or_a_type(self):
        for dtype in ('float32', np.float32, np.dtype('float32')):
            with self.subTest(dtype=dtype):
                self.assertEqual(CircleTransform(dtype=dtype).dtype,
                                 np.float32)

    def test_other_precisions_are_refused(self):
        for dtype in (np.float16, np.int32, 'complex64'):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    CircleTransform(dtype=dtype)

    def test_the_kernel_has_the_precision(self):
        self.assertEqual(CircleTransform(dtype=np.float32).kernel(
            (32, 32)).dtype, np.complex64)
        self.assertEqual(CircleTransform().kernel((32, 32)).dtype,
                         np.complex128)

    def test_changing_the_precision_changes_the_kernel(self):
        transform = CircleTransform()
        double = transform.kernel((32, 32))
        transform.dtype = np.float32
        self.assertEqual(transform.kernel((32, 32)).dtype, np.complex64)
        transform.dtype = np.float64
        np.testing.assert_array_equal(transform.kernel((32, 32)), double)

    def test_a_uniform_image_gives_zeros(self):
        heat = circletransform(np.full((32, 40), 7.), dtype=np.float32)
        self.assertTrue(np.all(heat == 0.))

    def test_the_derivative_in_single_precision(self):
        image = self.image.astype(np.float32)
        for name, patch in backends():
            for axis in (0, 1):
                with self.subTest(backend=name, axis=axis), patch:
                    out = _derivative(image, 13, 3, axis)
                    self.assertEqual(out.dtype, np.float32)
                    np.testing.assert_allclose(
                        out, signal.savgol_filter(image, 13, 3, 1, axis=axis),
                        atol=1e-4 * np.abs(out).max())


class TestBinning(unittest.TestCase):

    def setUp(self):
        # rings 20 pixels apart: binning averages away finer ones
        self.image = rings((192, 256), (80, 140), k=0.3)

    def test_the_default_is_no_binning(self):
        self.assertEqual(CircleTransform().binning, 1)
        np.testing.assert_array_equal(
            circletransform(self.image, binning=1), circletransform(self.image))

    def test_the_map_is_on_the_coarser_grid(self):
        for n in (2, 3, 4):
            with self.subTest(binning=n):
                heat = circletransform(self.image, binning=n)
                self.assertEqual(heat.shape, (192 // n, 256 // n))
                self.assertAlmostEqual(float(heat.max()), 1.)

    def test_the_peak_is_where_it_was_to_within_the_binning(self):
        y, x = 80, 140
        for n in (2, 4):
            with self.subTest(binning=n):
                j, i = peak(circletransform(self.image, window=5,
                                            polyorder=2, binning=n))
                self.assertLessEqual(abs(n * j - y), 2 * n)
                self.assertLessEqual(abs(n * i - x), 2 * n)

    def test_rows_and_columns_that_do_not_fill_a_block_are_dropped(self):
        heat = circletransform(self.image[:191, :255], binning=4)
        self.assertEqual(heat.shape, (47, 63))

    def test_a_block_is_the_mean_of_its_pixels(self):
        from dgimage.circletransform import _bin
        image = np.arange(16.).reshape(4, 4)
        np.testing.assert_array_equal(
            _bin(image, 2), [[2.5, 4.5], [10.5, 12.5]])

    def test_it_is_not_below_one(self):
        self.assertEqual(CircleTransform(binning=0).binning, 1)

    def test_an_image_smaller_than_a_block_is_refused(self):
        with self.assertRaises(ValueError):
            circletransform(np.zeros((3, 3)), binning=4)

    def test_it_combines_with_single_precision(self):
        heat = circletransform(self.image, binning=2, dtype=np.float32)
        self.assertEqual(heat.dtype, np.float32)
        self.assertEqual(heat.shape, (96, 128))


class TestOpenCV(unittest.TestCase):

    def setUp(self):
        self.saved = list(_OPENCV)
        _OPENCV.clear()
        self.addCleanup(self.restore)

    def restore(self):
        _OPENCV[:] = self.saved

    def test_without_opencv_it_is_none(self):
        with mock.patch.dict(sys.modules, {'cv2': None}):
            self.assertIsNone(_opencv())

    def test_the_answer_is_kept(self):
        with mock.patch.dict(sys.modules, {'cv2': None}):
            _opencv()
        # the import is not tried again
        self.assertEqual(len(_OPENCV), 1)
        self.assertIsNone(_opencv())

    def test_with_opencv_it_is_the_module(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV is not installed')
        self.assertIs(_opencv(), cv2)


if __name__ == '__main__':      # pragma: no cover
    unittest.main()
