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

import sklearn
from sklearn.cross_decomposition import PLSRegression as _PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt


class PLSRegression(_PLSRegression):
    """
    PLS regression with added chemometric functionality
    """

    def __init__(self, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, Y):
        super().fit(X, Y)
        self.vip_ = self._calculate_vip()

        # calculate variation orthogonal to model plane
        X_hat = self.inverse_transform(self.transform(X))
        sse_cal = np.sum((X - X_hat)**2)
        scaled = 1 if self.scale else 0
        norm_factor = (self.x_scores_.shape[0] - self.n_components
                       - scaled) * (X.shape[1] - self.n_components)
        self.x_residual_std_ = np.sqrt(sse_cal / norm_factor)
        return self

    def _calculate_vip(self):
        """ Calculate variable importance in projection (VIP)

        The VIP scores provide a concise summary of how important the different
        predictors are for the model. It summarizes and normalizes the
        information of the loading vector(s). All predictors with VIP-scores
        greater than 1 have a greater influence on the model than if all
        predictors were equally informative. Variable/predictor selection can
        easily be performed by dropping all variables with a score smaller than
        a certain threshold. Typically, 1.0 or 1.5 is used as cut-off. Method
        adapted from [1].

        References
        ----------
        .. [1] Mehmood et al. Chemometrics and
        Intelligent Laboratory Systems 118 (2012) 62–69.
        """

        ss = np.sum(self.y_loadings_ ** 2, axis=0) *\
            np.sum(self.x_scores_ ** 2, axis=0)
        counter = np.sum(ss * self.x_weights_**2, axis=1)
        denominator = np.sum(ss)
        return np.sqrt(self.n_features_in_ * counter / denominator)

    def hat(self, X):
        """
        Calculate the hat (projection) matrix

        Calculate the hat matrix in the X/Y score space. The hat matrix $H$
        projects the observed $Y$ onto the predicted $\hat Y$. For obtaining
        the standard hat matrix, the provided X matrix should correspond to the
        matrix used during the calibration (call to `fit`) [1].

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors

        Returns
        -------
        hat : (n, n) ndarray
            Hat matrix, symmetric matrix, n x n samples

        References
        ----------
        Calculation according to [1]
        .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
        Wikström, and S. Wold. Multi- and Megavariate Data Analysis, Part I
        Basic Principles and Applications. Second Edition.
        """
        S = self.transform(X)
        return S @ np.linalg.inv(S.T @ S) @ S.T

    def leverage(self, X):
        """
        Calculate the statistical leverage

        Calculate the leverage (self-influence of Y) in the X/Y score space.
        For obtaining the standard leverage, the provided X matrix should
        correspond to the matrix used during calibration (call to `fit`).

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors

        Returns
        -------
        leverage : (n, ) ndarray
            leverage for n samples
        """
        return np.diag(self.hat(X))

    def dmodx(self, X, normalize=True, absolute=False):
        """
        Calculate distance to model hyperplane in X (DModX)

        DModX provides the distance to the model hyperplane spanned by the
        loading vectors. Any information in the predictors that is not captured
        by the PLS model contributes to DModX. If the DModX is normalized,
        DModX is devided by the mean residual variance of X observed during
        model calibration.

        Parameters
        ----------
        X : (n, m) ndarray
            matrix of predictors. n samples x m predictors

        normalize : {True (default); False}
            normalization of DModX by error in X during calibration

        absolute : {True; False (default)}
            return the absolute distance to the model plane (not normalized by
            degrees of freedom)

        Returns
        -------
        dmodx : (n, ) ndarray
            distance of n samples to model hyperplane
        """

        sse = np.sum((X - self.inverse_transform(self.transform(X)))**2,
                     axis=1)
        dmodx = np.sqrt(sse)

        if not absolute:
            dmodx /= np.sqrt(X.shape[1] - self.n_components)
            if normalize:
                dmodx /= self.x_residual_std_

        return dmodx

    def plot(self, X, Y):
        """
        Displays a figure with 4 common analytical plots for PLS models

        Generates a figure with four subplots providing analytical insights
        into the PLS model. Typically, the calibration data is used for
        the method call. following four subplots are generated:
        1) observed -> predicted. Provides insights into the linearity of the
        data and shows how well the model performes over the model range.
        2) predicted -> residuals. Similar to 1). Useful for evaluating the
        error structure (e.g. homoscedasticity).
        3) leverage -> residuals. Provides insights into any data
        points/outliers which strongly affect the model. Optimally, the points
        should be scattered in the center left.
        4) predictors -> VIP. Provides insights into the predictor importance
        for the model.

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors
        Y : (n, o) ndarray
            Matrix of responses. n samples x o responses

        Returns
        -------
        axes : list(axis, ...)
            List of axis for subplots
        """
        fig = plt.figure(figsize=(15, 15))
        Y_hat = self.predict(X)
        residuals = Y - Y_hat
        leverage = self.leverage(X)

        # make plots
        # 1) observed vs predicted
        plt.subplot(221)
        plt.scatter(Y, Y_hat)
        plt.axline((np.min(Y), np.min(Y)), (np.max(Y), np.max(Y)),
                   color='k', alpha=0.2)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')

        # 2) predicted vs residuals
        plt.subplot(222)
        plt.scatter(Y_hat, residuals)
        plt.axhline(0, color='k', alpha=0.2)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')

        # 3) leverage mapped to residuals (lmr) plot
        plt.subplot(223)
        for i in range(residuals.shape[1]):
            plt.scatter(leverage, residuals[:, i], alpha=0.5)
        plt.xlabel('Leverage')
        plt.ylabel('Residuals')

        # 4) VIPs
        plt.subplot(224)
        plt.plot(self.vip_)
        plt.axhline(1, color=np.zeros([3]), alpha=0.2)
        plt.axhline(1.5, color=np.zeros([3]), alpha=0.4)
        plt.xlabel('Predictor')
        plt.ylabel('VIP')

        return fig.axes


def fit_pls(X, Y, pipeline=None, cv_object=None, max_lv=10):
    """
    Auto-calibrate PLS model and generate analytical plots

    A PLS model is calibrated based on the maximization of the coefficient of
    determination during cross-validation ($Q^2$). The function provides
    multiple plots for assessing the model quality. The first figure addresses
    the model performance during cross validation and the estimation of optimal
    number of latent variables by showing $R^2$/$Q^2$ values ($R^2 as bars,
    $Q^2$ as boxplots based on the individual rotations). The second figure
    shows four subplots with analytical information for the optimal model. The
    plotted figures are: a) observed versus predicted 2) predicted versus
    residuals 3) leverage versus residuals 4) Variable importance in projection
    (VIP) scores.

    Parameters
    ----------
    X : (n, m) ndarray
        Matrix of predictors. n samples x m predictors
    Y : (n, o) ndarray
        Matrix of responses. n samples x o responses
    pipeline : {None, sklearn.pipeline.Pipeline}
        A pipeline object providing a workflow of preprocessing and a
        PLSRegression model. The last entry must be a
        chemometrics.PLSRegression instance.
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
        pipeline = make_pipeline(PLSRegression())
    elif not (isinstance(pipeline, sklearn.pipeline.Pipeline)):
        raise TypeError(
            "pipeline argument provided is of type "
            + "{0} and not of type Pipeline.".format(type(pipeline))
        )
    elif not isinstance(pipeline[-1],
                        PLSRegression):
        # check if pipeline ends with PLSRegression
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
        q2.append(cross_val_score(pipeline, X, Y, cv=cv_object))
        r2.append(pipeline.fit(X, Y).score(X, Y))

    q2 = np.stack(q2).T
    r2 = np.array(r2)

    # plot CV results
    fig_cv = plt.figure(figsize=(6, 6))
    plt.bar(np.arange(1, max_lv+1), r2, alpha=0.5)
    plt.boxplot(q2, positions=np.arange(1, max_lv+1))
    plt.xlabel('Latent variables')
    plt.ylabel('R2 / Q2')

    # recover best q2 and adjust model accordingly
    pipeline[-1].n_components = np.argmax(np.median(q2, axis=0)) + 1
    pipeline = pipeline.fit(X, Y)

    # plot PLS performance after preprocessing
    if len(pipeline) > 1:
        X_preprocessed = pipeline[:-1].predict(X)
    else:
        X_preprocessed = X
    pipeline[-1].plot(X_preprocessed, Y)
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
