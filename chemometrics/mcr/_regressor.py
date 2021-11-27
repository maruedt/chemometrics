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
Built-in least squares / regression methods.

All models will follow the formalism, AX = B, solve for X.

"""

from abc import ABC, abstractmethod

import numpy as np

from scipy.linalg import lstsq
from scipy.optimize import nnls


class LinearRegression(ABC):
    """ Abstract class for linear regression methods """

    def __init__(self):
        self.X_ = None
        self.residual_ = None

    @property
    def coef_(self):
        if self.X_ is None:
            return None
        else:
            return self.X_.T

    @abstractmethod
    def fit(self, A, B):
        """ AX = B, solve for X """


class OLS(LinearRegression):
    """
    Ordinary least squares regression

    AX = B, solve for X (coefficients.T)

    Attributes
    ----------
    coef_ : ndarray
        Regression coefficients (X.T)

    residual_ : ndarray
        Residual (sum-of-squares)

    rank_ : int
        Effective rank of matrix A

    svs_ : ndarray
        Singular values of matrix A

    Notes
    -----
    This is simply a wrapped version of Ordinary Least Squares
    (scipy.linalg.lstsq).

    ``coef_`` is X.T, which is the formalism of scikit-learn

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rank_ = None
        self.svs_ = None

    def fit(self, A, B):
        """ Solve for X: AX = B"""
        self.X_, self.residual_, self.rank_, self.svs_ = lstsq(A, B)


class NNLS(LinearRegression):
    """
    Non-negative constrained least squares regression

    AX = B, solve for X (coeffients.T)

    Attributes
    ----------
    coef_ : ndarray
        Regression coefficients

    residual_ : ndarray
        Residual (sum-of-squares)

    Notes
    -----
    This is simply a wrapped version of NNLS
    (scipy.optimize.nnls).

    ``coef_`` is ``X.T``, which is the formalism of scikit-learn
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit(self, A, B):
        """ Solve for X: AX = B"""

        if B.ndim == 2:
            N = B.shape[-1]
        else:
            N = 0

        self.X_ = np.zeros((A.shape[-1], N))
        self.residual_ = np.zeros((N))

        # nnls is Ax = b; thus, need to iterate along
        # columns of B
        if N == 0:
            self.X_, self.residual_ = nnls(A, B)
        else:
            for num in range(N):
                self.X_[:, num], self.residual_[num] = nnls(A, B[:, num])
