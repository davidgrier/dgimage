'''Tests of the azimuthal statistics.'''
import unittest

import numpy as np

from dgimage import azimuthal
from dgimage.azimuthal import _build_context


def distance(shape=(41, 51), center=(25., 20.)):
    '''The distance of every pixel from *center* (x, y).'''
    y, x = np.indices(shape)
    return np.hypot(x - center[0], y - center[1])


class TestAverage(unittest.TestCase):

    def test_a_uniform_image_has_a_flat_profile(self):
        profile = azimuthal.avg(np.full((30, 30), 3.))
        np.testing.assert_allclose(profile, 3.)

    def test_the_profile_of_a_distance_map_is_the_distance(self):
        profile = azimuthal.avg(distance(), center=(25., 20.))
        radii = np.arange(len(profile))
        # the mean distance of the pixels in a one pixel bin is within it
        self.assertTrue(np.all(profile >= radii - 1e-9))
        self.assertTrue(np.all(profile < radii + 1.))

    def test_the_default_center_is_the_middle(self):
        image = distance((40, 60), center=(30., 20.))
        np.testing.assert_array_equal(azimuthal.avg(image),
                                      azimuthal.avg(image, center=(30., 20.)))

    def test_a_center_may_be_an_array(self):
        image = distance()
        np.testing.assert_array_equal(
            azimuthal.avg(image, center=np.array([25., 20.])),
            azimuthal.avg(image, center=(25., 20.)))

    def test_integer_images_are_accepted(self):
        counts = np.round(distance()).astype(np.uint16)
        self.assertEqual(azimuthal.avg(counts, (25., 20.)).dtype, np.float64)

    def test_a_color_image_is_refused(self):
        with self.assertRaises(ValueError):
            azimuthal.avg(np.zeros((4, 4, 3)))

    def test_the_input_is_not_changed(self):
        image = distance()
        before = image.copy()
        azimuthal.avg(image)
        np.testing.assert_array_equal(image, before)


class TestSpread(unittest.TestCase):

    def test_std_returns_the_average_too(self):
        image = distance()
        mean, spread = azimuthal.std(image, (25., 20.))
        np.testing.assert_array_equal(mean, azimuthal.avg(image, (25., 20.)))
        self.assertEqual(spread.shape, mean.shape)

    def test_a_uniform_image_has_no_spread(self):
        _, spread = azimuthal.std(np.full((30, 30), 3.))
        np.testing.assert_allclose(spread, 0., atol=1e-12)

    def test_the_spread_of_known_values(self):
        # radius 0 is one pixel; its spread is zero
        image = np.arange(25.).reshape(5, 5)
        mean, spread = azimuthal.std(image, (2., 2.))
        self.assertEqual(mean[0], image[2, 2])
        self.assertEqual(spread[0], 0.)
        self.assertGreater(spread[1], 0.)


class TestMedian(unittest.TestCase):

    def test_the_median_ignores_an_outlier_that_the_average_feels(self):
        image = np.ones((41, 41))
        image[20 + 5, 20] = 1e6                  # one pixel at radius 5
        self.assertGreater(azimuthal.avg(image, (20., 20.))[5], 100.)
        self.assertEqual(azimuthal.med(image, (20., 20.))[5], 1.)

    def test_mad_of_a_uniform_image_is_zero(self):
        m, dev = azimuthal.mad(np.full((20, 20), 2.))
        np.testing.assert_allclose(m, 2.)
        np.testing.assert_allclose(dev, 0.)

    def test_mad_is_robust_against_an_outlier(self):
        image = np.ones((41, 41))
        image[25, 20] = 1e6
        _, dev = azimuthal.mad(image, (20., 20.))
        self.assertEqual(dev[5], 0.)

    def test_a_radius_with_no_pixels_is_nan(self):
        # a center far outside the image leaves the small radii empty
        image = np.ones((10, 10))
        m = azimuthal.med(image, center=(-20., 5.))
        self.assertTrue(np.isnan(m[0]))
        self.assertEqual(m[-1], 1.)
        medians, dev = azimuthal.mad(image, center=(-20., 5.))
        self.assertTrue(np.isnan(medians[0]) and np.isnan(dev[0]))


class TestCache(unittest.TestCase):

    def test_the_geometry_is_kept_for_a_shape_and_center(self):
        a = _build_context(30, 40, 20., 15.)
        self.assertIs(_build_context(30, 40, 20., 15.), a)
        self.assertIsNot(_build_context(30, 40, 21., 15.), a)

    def test_the_name_and_docstring_of_an_operator_survive(self):
        self.assertEqual(azimuthal.avg.__name__, 'avg')
        self.assertIn('Azimuthal average', azimuthal.avg.__doc__)


if __name__ == '__main__':      # pragma: no cover
    unittest.main()
