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

from .context import chemometrics as cm  # accounts for relativ path
import numpy as np
import unittest


class TestPLSRegression(unittest.TestCase):
    """
    Test adjustments done to PLSRegression child class
    """
    def setUp(self):
        self.n_wl = 100
        self.n_samples = 100
        self.n_conc = 2

        Y = np.random.uniform(size=[self.n_samples, self.n_conc])
        noise = 0.1
        spectra = np.zeros(shape=[self.n_wl, self.n_conc])

        for i in range(self.n_conc):
            spectra[:, i] = cm.generate_spectra(self.n_wl, self.n_wl//20, 1)

        X = Y @ spectra.T + np.random.normal(scale=noise,
                                             size=[self.n_samples, self.n_wl])
        self.pls = cm.PLSRegression()
        self.pls = self.pls.fit(X, Y)

    def test_vip_shape(self):
        """
        Test the shape of the vip variable
        """
        self.assertTrue(self.pls.vip_.shape == (self.n_wl, ))

    def test_vip_value(self):
        """
        Test that sum of squared VIPs == number of X variables (definition!)
        """
        self.assertTrue(np.isclose(np.sum(self.pls.vip_**2), self.n_wl))
