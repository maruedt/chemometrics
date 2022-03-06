# Copyright 2021 Matthias Rüdt
#
# This file is part of chemometrics.
#
# chemometrics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# chemometrics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with chemometrics.  If not, see <https://www.gnu.org/licenses/>.

import chemometrics as cm  # accounts for relativ path
import numpy as np
import unittest
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid


class TestGenerate_spectra(unittest.TestCase):
    r"""
    Test the `generate_spectra` function.
    """

    def test_shape(self):
        """
        Test if correct shape is generated
        """
        n_wl = 200
        expected_shape = (n_wl,)
        output_shape = cm.generate_spectra(n_wl, 2, 50).shape
        self.assertTrue(np.all(expected_shape == output_shape))

    def test_no_bands(self):
        """
        Test if ``n_band = 0`` returns zero vector.
        """
        n_wl = 10
        n_bands = 0
        bandwidth = 100
        spectra = cm.generate_spectra(n_wl, n_bands, bandwidth)
        isZero = np.all(np.isclose(np.zeros(n_wl), spectra))
        self.assertTrue(isZero)

class TestPsaudoVoigt(unittest.TestCase):
    r"""
    Test Pseudo Voigt profiles
    """

    def test_shape(self):
        """
        Test if _pseudo_voigt returns correct shape.
        """

        wl = np.arange(1000)[:, None]
        parameters = np.array([
            [200, 0.5, 50, 500],
            [100, 0.8, 20, 200]
        ]).T

        spec = cm.pseudo_voigt_spectra(wl, parameters)

        self.assertTrue(spec.shape == wl.shape)

    def test_statistics_pdf(self):
        """
        Test area, mode and full width at half max of a single Voigt peak
        in lorentzian limit
        """
        wl = np.arange(1000)[:, None]
        par_list = [
                np.array([[1, 0, 50, 500],]).T,
                np.array([[1, 0.3, 50, 500],]).T,
                np.array([[1, 1, 40, 280],]).T
        ]


        for parameters in par_list:
            spec = cm.pseudo_voigt_spectra(wl, parameters)

            # calculate mode (wl_max)
            wl_max = wl[np.argmax(spec)][0]
            assert_allclose(parameters[3], wl_max)
            signal_max = np.max(spec)
            assert_allclose(parameters[0], signal_max)

            # calculate fwhm by inverting half (>wl_max) of the
            # propability  density function and analyzing width

            invPDF = interp1d(spec[wl_max:,0], wl[wl_max:,0])
            fwhm = 2*(invPDF(signal_max/2)-wl_max)

            assert_allclose(fwhm, 2*parameters[2])
