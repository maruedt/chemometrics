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
import numpy as np
import sklearn
from sklearn.decomposition import PCA as _PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt


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

    def score(self, X, y=None, scoring='r2'):
        """
        """
        X_hat = self.inverse_transform(self.transform(X))
        X_bar = np.mean(X, axis=0)
        predictor = np.sum((X-X_hat)**2)
        denominator = np.sum((X-X_bar)**2)
        score = 1-predictor/denominator

        return score

    @property
    def x_loadings_(self):
        """
        The loadings of `X` with shape (n_features, n_components).
        """
        return self.components_.T


def fit_pca(X, pipeline=None, cv_object=None, max_lv=10):
    r"""
    Auto-calibrate PCA model and generate analytical plots

    A PCA model is calibrated based on the maximization of the coefficient of
    determination during cross-validation (Q2). The function provides
    multiple plots for assessing the model quality. The first figure addresses
    the model performance during cross validation and the estimation of optimal
    number of latent variables by showing R2/Q2 values (R^2
    as bars, Q^2 as boxplots based on the individual rotations). The
    second figure shows four subplots with analytical information for the
    optimal model. The plotted figures are: a) observed versus predicted 2)
    predicted versus residuals 3) leverage versus residuals 4) Variable
    importance in projection (VIP) scores.

    Parameters
    ----------
    X : (n, m) ndarray
        Matrix of predictors. n samples x m predictors
    Y : (n, o) ndarray
        Matrix of responses. n samples x o responses
    pipeline : {None, sklearn.pipeline.Pipeline}
        A pipeline object providing a workflow of preprocessing and a
        PLSRegression model. The last entry must be a
        chemometrics.PCA instance.
    cv_object : {None, cv_object}
        An object providing guidance for cross-validation. Typically, it will
        be an instance of an sklearn.model_selection.BaseCrossValidator object.
    max_lv : int
        Number of latent variables up to which the cross-validation score will
        be screened.

    Returns
    -------
    pipeline : Pipeline
        The calibrated model pipeline
    summary : dict
        Summary of the model calibration.
    """
    if not pipeline:
        pipeline = make_pipeline(PCA())
    elif not (isinstance(pipeline, sklearn.pipeline.Pipeline)):
        raise TypeError(
            "pipeline argument provided is of type "
            + "{0} and not of type Pipeline.".format(type(pipeline))
        )
    elif not isinstance(pipeline[-1],
                        PCA):
        # check if pipeline ends with PCA
        raise TypeError(
            "Type of last object provided in pipline is "
            + "{0} but should be a PLSRegression".format(type(pipeline[-1]))
        )

    if not cv_object:
        cv_object = KFold(n_splits=5)

    # perform CV of model up to max_lv
    r2 = []
    q2 = []
    for n_lv in range(1, max_lv+1):
        pipeline[-1].n_components = n_lv
        q2.append(cross_val_score(pipeline, X, cv=cv_object))
        r2.append(pipeline.fit(X)[-1].explained_variance_ratio_.sum())

    q2 = np.stack(q2).T
    r2 = np.array(r2)

    # plot CV results
    fig_cv = plt.figure(figsize=(6, 6))
    plt.bar(np.arange(1, max_lv+1), r2, alpha=0.5)
    plt.boxplot(q2, positions=np.arange(1, max_lv+1))
    plt.xlabel('Latent variables')
    plt.ylabel('R2, Q2')

    # recover best q2 and adjust model accordingly
    pipeline[-1].n_components = np.argmax(np.median(q2, axis=0)) + 1
    pipeline = pipeline.fit(X)

    # plot PLS performance after preprocessing
    if len(pipeline) > 1:
        X_preprocessed = pipeline[:-1].transform(X)
    else:
        X_preprocessed = X
    pipeline[-1].distance_plot(X_preprocessed)
    fig_model = plt.gcf()

    # arrange return arguments
    analysis = {
        'r2': r2,
        'q2': q2,
        'q2_mean': np.mean(q2, axis=0),
        'q2_median': np.median(q2, axis=0),
        'optimal_lv': pipeline[-1].n_components,
        'figure_cv': fig_cv,
        'figure_model': fig_model
    }

    return pipeline, analysis
