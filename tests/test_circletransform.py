'''Tests of the orientation alignment transform.'''
import unittest

import numpy as np

from dgimage import CircleTransform, circletransform


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


if __name__ == '__main__':      # pragma: no cover
    unittest.main()
