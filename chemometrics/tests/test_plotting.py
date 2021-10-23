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
import matplotlib


class TestPlot_colored_spectra(unittest.TestCase):
    r"""
    Test the `plot_colored_spectra` function.
    """

    def test_return_arguments(self):
        """
        Test if the correct shape and object type is returend.
        """
        n_series, n_variables = (10, 50)
        # generate dummy data
        x = np.arange(0, n_variables)
        Y = np.ones([n_series, n_variables]).T * np.linspace(0, 10,
                                                             num=n_series)
        lines = cm.plot_colored_series(Y, x=x)
        self.assertTrue(type(lines) == list)
        self.assertEqual(n_series, len(lines))
        for line in lines:
            self.assertTrue(type(line) == matplotlib.lines.Line2D)


class TestPlot_svd(unittest.TestCase):
    r"""
    Test the `plot_svd` function.
    """

    def test_return_arguments(self):
        n_series, n_variables = (50, 100)
        n_comp = 3
        n_eigenvalues = 12
        mean = np.zeros(n_variables)
        x = np.arange(n_variables)[:, None]
        dist = x.T - x
        cov = np.exp(-(dist / n_variables * 5)**2)

        # draw a few samples from gaussian process as data
        D = np.random.multivariate_normal(mean, cov, size=n_series)

        # execute function with artificial data
        cm.plot_svd(D, n_comp=n_comp, n_eigenvalues=n_eigenvalues)
        fig = matplotlib.pyplot.gcf()
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].lines), n_comp)
        self.assertEqual(len(fig.axes[1].lines), n_eigenvalues)
        self.assertEqual(len(fig.axes[2].lines), n_comp)
