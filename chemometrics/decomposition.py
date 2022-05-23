# Governed by two licenses
#
# Parts of the documentation:
# Adapted from sklearn, released under BSD-3 clause license
#
# Everything not adapted from sklearn:
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

from .base import LVmixin
from sklearn.decomposition import PCA as _PCA


class PCA(_PCA, LVmixin):
    """
    Principal component analysis with added chemometric functionality

    Linear factorization of the data matrix X into scores and loadings
    (=components) similar to a truncated singular value decomposition.
    Next to the transformer capabilities, PCA provides additionally different
    metrics on the fitted latent variable model.
    """

    def __init__(
        self,
        n_components=2,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.x_scores_ = super().fit_transform(X)
        self.x_residual_std_ = self._calculate_x_residual_std_(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """
        self.x_scores_ = super().fit_transform(X)
        self.x_residual_std_ = self._calculate_x_residual_std_(X)
        return self.x_scores.copy()

    @property
    def x_loadings_(self):
        """
        x_loadings_ : ndarray of shape (n_features, n_components)
            The loadings of `X`.
        """
        return self.components_.T
