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
