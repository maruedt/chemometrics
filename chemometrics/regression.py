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
        return self

    def _calculate_vip(self):
        """
        Calculate variable importance in projection (VIP)

        Method adapted from Mehmood et al. Chemometrics and Intelligent
        Laboratory Systems 118 (2012) 62–69.
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
        matrix used during the calibration (call to `fit`).
        """
        S = self.transform(X)
        return S @ np.linalg.inv(S.T @ S) @ S.T

    def leverage(self, X):
        """
        Calculate the statistical leverage

        Calculate the leverage (self-influence of Y) in the X/Y score space.
        For obtaining the standard leverage, the provided X matrix should
        correspond to the matrix used during calibration (call to `fit`).
        """
        return np.diag(self.hat(X))

    def plot(self, X, Y):
        """
        Displays a figure with 4 common analytical plots for PLS models
        """
        fig = plt.figure(figsize=(15, 15))
        Y_hat = self.predict(X)
        residuals = Y - Y_hat
        leverage = self.leverage(X)

        # make lmr plots
        # 1) observed vs predicted
        plt.subplot(221)
        plt.scatter(Y, Y_hat)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')

        # 2) predicted vs residuals
        plt.subplot(222)
        plt.scatter(Y_hat, residuals)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')

        # 3) leverage vs residuals
        plt.subplot(223)
        for i in range(residuals.shape[1]):
            plt.scatter(leverage, residuals[:, i], alpha=0.5)
        plt.xlabel('Leverage')
        plt.ylabel('Residuals')

        # 4) VIPs
        plt.subplot(224)
        plt.plot(self.vip_)
        plt.xlabel('Predictor')
        plt.ylabel('VIP')

        return fig.axes


def fit_pls(X, Y, pipeline=None, cv_object=None, max_lv=10):
    """
    Calibrate PLS model and generate analytical plots
    """
    if not pipeline:
        pipeline = make_pipeline(PLSRegression())
    elif not (isinstance(pipeline, sklearn.pipeline.Pipeline)):
        raise TypeError(
            "pipeline argument provided is of type "
            + "{0} and not of type Pipeline.".format(type(pipeline))
        )
    elif not isinstance(pipeline[-1],
                        sklearn.cross_decomposition.PLSRegression):
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
    pipeline[-1].n_components = np.argmax(np.mean(q2, axis=0)) + 1
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
        'optimal_lv': pipeline[-1].n_components,
        'figure_cv': fig_cv,
        'figure_model': fig_model
    }

    return pipeline, analysis
