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
#
# Additional information on the code history, original copyright holder and
# original license is available in ./license.md


"""
Testing chemometrics.mcr.regressors

"""

import numpy as np

from numpy.testing import (assert_equal, assert_array_equal,
                           assert_allclose)

from chemometrics.mcr.regressor import OLS, NNLS

import unittest


class TestRegressors(unittest.TestCase):
    """
    """

    def test_generic_parts(self):
        methods = [OLS, NNLS]

        for method in methods:
            self._test_instantiation(method())
            self._test_dimensionality(method())
            self._test_basic_positive_least_squares(method())

    def _test_instantiation(self, regr):
        """ Test correct instantiation of classes """
        self.assertTrue(regr.coef_ is None)
        self.assertTrue(regr.X_ is None)
        self.assertTrue(regr.residual_ is None)
        self.assertTrue(hasattr(regr, 'fit'))

    def _test_dimensionality(self, regr):
        """ Test correct dimension of regressor methods """

        A = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
        x = np.array([1, 2, 3])
        X = x[:, None]
        B = np.dot(A, X)
        b = np.dot(A, x)

        # b = 1D
        regr.fit(A, b)
        assert_equal(regr.X_.ndim, 1)
        assert_equal(regr.X_.shape[0], regr.X_.size)
        assert_equal(regr.X_.shape[0], A.shape[-1])
        assert_array_equal(x.shape, regr.X_.shape)
        assert_array_equal(x.T.shape, regr.coef_.shape)

        # B = 2D
        regr.fit(A, B)
        assert_equal(regr.X_.ndim, 2)
        assert_array_equal(regr.X_.shape, [A.shape[-1], B.shape[-1]])

        assert_array_equal(X.shape, regr.X_.shape)
        assert_array_equal(X.T.shape, regr.coef_.shape)

    def _test_basic_positive_least_squares(self, regr):
        """

        """
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
        x = np.array([1, 2, 3])
        X = x[:, None]
        B = np.dot(A, X)
        b = np.dot(A, x)

        # b is 1D
        regr.fit(A, b)
        assert_allclose(x, regr.X_)

        # b is 1D->2D
        regr.fit(A, b[:, None])
        assert_allclose(X, regr.X_)

        # b = 2D
        regr.fit(A, B)
        assert_allclose(X, regr.X_)

    def test_nnls_negative_x(self):
        """ Test nnls """
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
        x = np.array([1, -2, 3])
        b = np.dot(A, x)

        regr = NNLS()

        # b is 1D
        regr.fit(A, b)
        assert_allclose(True, regr.X_ >= 0)  # Must be non-negative
        assert_allclose(False, regr.X_ == x)  # Must be non-negative
        # assert_allclose(x, regr.X_)

    def test_ols_negative_x(self):
        """ Test nnls """
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
        x = np.array([1, -2, 3])
        b = np.dot(A, x)

        regr = OLS()

        # b is 1D
        regr.fit(A, b)
        assert_allclose(x, regr.X_)


if __name__ == '__main__':
    unittest.main()
