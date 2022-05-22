# Copyright 2022 Matthias Rüdt
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

import chemometrics as cm
import numpy as np
import unittest
from chemometrics.tests.test_base import TestLVmixin


class TestPCA(unittest.TestCase, TestLVmixin):
    """
    Test PCA class
    """
    def setUp(self):
        self.n_wl = 50
        self.n_samples = 500
        self.n_tests = 100
        self.n_conc = 2

        self.Y = np.random.normal(size=[self.n_samples, self.n_conc])
        noise = 0.1
        spectra = np.zeros(shape=[self.n_wl, self.n_conc])

        for i in range(self.n_conc):
            spectra[:, i] = cm.generate_spectra(self.n_wl, self.n_wl//20, 1)

        self.X = self.Y @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_samples, self.n_wl]
        )
        self.model = cm.PCA()
        self.model = self.model.fit(self.X)

        self.Y_test = np.random.normal(size=[self.n_tests, self.n_conc])
        self.X_test = self.Y_test @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_tests, self.n_wl]
        )
