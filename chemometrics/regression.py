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
from scipy import stats
import matplotlib.pyplot as plt


class PLSRegression(_PLSRegression):
    r"""
    PLS regression with added chemometric functionality


    References
    ----------
    Calculations according to

    .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
           Wikström, and S. Wold. Multi- and Megavariate Data Analysis,
           Part I Basic Principles and Applications. Second Edition.


    """

    def __init__(self, n_components=2, *, scale=False,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=scale,
                         max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, Y):
        super().fit(X, Y)
        self.vip_ = self._calculate_vip()
        self.x_residual_std_ = self._calculate_x_residual_std_(X)
        return self

    def _calculate_vip(self):
        r""" Calculate variable importance in projection (VIP)

        The VIP scores provide a concise summary of how important the different
        predictors are for the model. It summarizes and normalizes the
        information of the loading vector(s). All predictors with VIP-scores
        greater than 1 have a greater influence on the model than if all
        predictors were equally informative. Variable/predictor selection can
        easily be performed by dropping all variables with a score smaller than
        a certain threshold. Typically, 1.0 or 1.5 is used as cut-off. Method
        adapted from [2]_.

        References
        ----------

        .. [2] Mehmood et al. Chemometrics and Intelligent Laboratory Systems
               118 (2012) 62–69.
        """
        ss = np.sum(self.y_loadings_ ** 2, axis=0) *\
            np.sum(self.x_scores_ ** 2, axis=0)
        counter = np.sum(ss * self.x_weights_**2, axis=1)
        denominator = np.sum(ss)
        return np.sqrt(self.n_features_in_ * counter / denominator)

    def _calculate_x_residual_std_(self, X):
        r"""
        Calculate the standard deviation of the X residuals

        The x residual standard deviation is calculated according to [1]_ not
        including the correction factor v (since it is not exactly defined in
        [1]_).
        """
        X_hat = self.inverse_transform(self.transform(X))
        sse_cal = np.sum((X - X_hat)**2)
        norm_factor = (self.x_scores_.shape[0] - self.n_components
                       - 1) * (X.shape[1] - self.n_components)
        return np.sqrt(sse_cal / norm_factor)

    def hat(self, X):
        r"""
        Calculate the hat (projection) matrix

        Calculate the hat matrix in the X/Y score space. The hat matrix
        :math:`H` projects the observed :math:`Y` onto the predicted
        :math:`\hat Y`. For  obtaining the standard hat matrix, the provided X
        matrix should  correspond to the matrix used during the calibration
        (call to `fit`)  [1]_.

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors

        Returns
        -------
        hat : (n, n) ndarray
            Hat matrix, symmetric matrix, n x n samples

        """
        S = self.transform(X)
        return S @ np.linalg.inv(S.T @ S) @ S.T

    def leverage(self, X):
        r"""
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

    def residuals(self, X, Y, scaling='studentize'):
        r"""
        Calculate (normalized) residuals

        Calculate the (normalized) residuals. The scaling scheme may be
        defined between 'none', 'standardize' and 'studentize'. The normalized
        residuals should only be calculated with the current training set.

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors
        Y : (n, o) ndarray
            Matrix of responses. n samples x o responses
        scaling : {'none', 'standardize', 'studentize' (default)}
            Define scaling of returned residuals

        Returns
        -------
        residuals : (n, o)
            Matrix of unscaled, standardized or studentized residuals

        Notes
        -----
        The response-wise standard deviation :math:`\sigma_j` is calculated
        according to

        .. math:: \sigma_j = \sqrt{\frac{\sum_i=1^n r_{i,j}^2}{n - p}}.

        Residuals are studentized according to

        .. math:: \hat{r}_i = \frac{r_i}{\sigma\sqrt{(1-h_{ii})}},

        with :math:`\hat{r}_i` being the studentized residuals,
        :math:`r_i` the original residuals and :math:`h_{ii}` the
        leverage.
        """
        Y_pred = self.predict(X)
        residuals = Y_pred - Y
        # internal standard deviation
        std = np.sqrt(np.sum(residuals**2, axis=0)
                      / (X.shape[0] - self.n_components))[:, None].T

        if scaling == 'none':
            scaling_factor = 1
        elif scaling == 'standardize':
            scaling_factor = std
        elif scaling == 'studentize':
            scaling_factor = std * np.sqrt(1 - self.leverage(X)[:, None])
        else:
            raise(TypeError(f'unsupported scaling: {scaling}'))

        return residuals / scaling_factor

    def dmodx(self, X, normalize=True, absolute=False):
        r"""
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

    def crit_dmodx(self, confidence=0.95):
        r"""
        Critical distance to hyperplane based on an F2 test

        The critical distance to the model hyperplane is estimated based on
        an F2 distribution. Values above crit_dmodx may be considered outliers.
        dmodx is only approximately F2 distributed [1]_. It is thus worth
        noting that the estimated critcal distance is biased. It however gives
        a reasonable indication of points worth investigating.

        """
        degf_cali = self.n_features_in_ - self.n_components - 1
        degf_test = self.n_features_in_ - self.n_components
        f_crit = stats.f.ppf(confidence, degf_test,
                             degf_cali)
        return np.sqrt(f_crit)

    def dhypx(self, X):
        r"""
        Normalized distance on hyperplane

        Provides a distance on the hyperplane, normalized by the distance
        observed during calibration. It can be a useful measure to see whether
        new data is comparable to the calibration data. The normalized dhypx
        is slightly biased towards larger values since the estimated
        `x_residual_std_` is slightly underestimated during model calibration
        [1]_.

        """
        var_cal = np.var(self.x_scores_, axis=0)
        x_scores2_norm = self.transform(X)**2 / var_cal
        return np.sum(x_scores2_norm, axis=1)

    def crit_dhypx(self, confidence=0.95):
        r"""
        Calculate critical dhypx according to Hotelling's T2

        """
        comp = self.n_components
        samples = self.x_scores_.shape[0]
        f_crit = stats.f.ppf(confidence, self.n_components,
                             self.x_scores_.shape[0]-self.n_components)
        factor = comp * (samples**2 - 1) / (samples * (samples - comp))
        return f_crit * factor

    def cooks_distance(self, X, Y):
        r"""
        Calculate Cook's distance from the calibration data

        Parameters
        ----------
        X : (n, m) ndarray
            Matrix of predictors. n samples x m predictors
        Y : (n, o) ndarray
            Matrix of responses. n samples x o responses

        Returns
        -------
        distances : (n, o) ndarray
            List of axis for subplots

        Notes
        -----
        Cooks distance is calculated according to

        .. math::

            D_i = \frac{r_i^2}{p\hat\sigma} \frac{h_{ii}}{(1-h_{ii})^2}

        """
        h = self.leverage(X)
        residuals = self.residuals(X, Y, scaling='none')
        mse = (np.sum(residuals**2, axis=0)
               / (X.shape[0] - self.n_components))[:, None].T
        coefficient1 = h / (1-h)**2
        coefficient2 = residuals**2 / (self.n_components * mse)
        return coefficient1[:, None] * coefficient2

    def plot(self, X, Y):
        r"""
        Displays a figure with 4 common analytical plots for PLS models

        Generates a figure with four subplots providing analytical insights
        into the PLS model. Typically, the calibration data is used for
        the method call. following four subplots are generated:
        1) observed -> predicted. Provides insights into the linearity of the
        data and shows how well the model performes over the model range.
        2) predicted -> studentized residuals. Similar to 1). Useful for
        evaluating the error structure (e.g. homoscedasticity) and detecting
        outliers (studentized residuals > 3)
        3) leverage -> studentized residuals. Provides insights into any data
        points/outliers which strongly affect the model. Optimally, the points
        should be scattered in the center left. The plot includes a limit on
        the Cook's distance of 0.5 and 1 as dashed and solid bordeaux
        lines, respectively.
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

        Notes
        -----
        The residuals are studentized according to

        .. math:: \hat{r}_i = \frac{r_i}{\sqrt{MSE (1-h_{ii)}}}

        The Cook's distance limit is calculated according to

        .. math:: \hat{r}_i = \pm \sqrt{D_{crit} p \frac{(1-h_{ii})}{h_{ii}}}

        with :math:`\hat{r}_i` being the studentized residuals,
        :math:`r_i` the original
        residuals, MSE the mean squared error, :math:`h_{ii}` the leverage,
        :math:`D_{crit}` the critical distance, :math:`p` the number of
        latent variables.
        """
        fig = plt.figure(figsize=(15, 15))
        Y_pred = self.predict(X)
        residuals = self.residuals(X, Y)
        leverage = self.leverage(X)

        # make plots
        # 1) observed vs predicted
        plt.subplot(221)
        plt.scatter(Y, Y_pred)
        plt.axline((np.min(Y), np.min(Y)), (np.max(Y), np.max(Y)),
                   color='k', alpha=0.2)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')

        # 2) predicted vs residuals
        plt.subplot(222)
        plt.scatter(Y_pred, residuals)
        plt.axhline(0, color='k', alpha=0.2)
        plt.xlabel('Predicted')
        plt.ylabel('Studentized residuals')

        # 3) leverage mapped to residuals (lmr) plot
        plt.subplot(223)
        for i in range(residuals.shape[1]):
            plt.scatter(leverage, residuals[:, i], alpha=0.5)

        plt.xlabel('Leverage')
        plt.ylabel('Studentized residuals')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # add cook's distance limit
        def cook(distance, linestyle):
            # prepare data from close to zero to bigger than max leverage
            x = np.linspace(1e-8, np.max(leverage)*1.2)
            lim = np.sqrt(distance*self.n_components * (1-x)/x)
            plt.plot(x, lim, color=[0.4, 0, 0], linestyle=linestyle)
            plt.plot(x, -lim, color=[0.4, 0, 0], linestyle=linestyle)

        cook(0.5, 'dotted')
        cook(1, 'solid')
        # reset axis limits
        plt.xlim(xlim)
        plt.ylim(ylim)

        # 4) VIPs
        plt.subplot(224)
        plt.plot(self.vip_)
        plt.axhline(1, color=np.zeros([3]), alpha=0.2)
        plt.axhline(1.5, color=np.zeros([3]), alpha=0.4)
        plt.xlabel('Predictor')
        plt.ylabel('VIP')

        return fig.axes

    def distance_plot(self, X, sample_id=None, confidence=0.95):
        r"""
        Plot distances colinear and orthogonal to model predictor hyperplane

        Generates a figure with two subplots. The subplots provide information
        on how `X` behaves compared to the calibration data. Subplots:
        1) Distance in model hyperplane of predictors. Provides insight into
        the magnitude of variation within the hyperplane compared to the
        calibration data. Large values indicate samples which are outside of
        the calibration space but may be described by linearly scaled latent
        variables.
        2) Distance orthogonal to model hyperplane. Provides insight into the
        magnitude of variation orthogonal to the model hyperplane compared to
        the calibration data. Large values indicate samples which show a
        significant trend not observed in the calibration data.

        """
        plt.figure(figsize=(15, 15))

        if not sample_id:
            sample_id = np.arange(X.shape[0])
        # make plots
        # 1) dhypx
        plt.subplot(211)
        plt.plot(sample_id, self.dhypx(X))
        plt.ylabel('X distance on hyperplane')
        plt.axhline(y=self.crit_dhypx(confidence=confidence))

        plt.subplot(212)
        plt.plot(sample_id, self.dmodx(X))
        plt.axhline(y=self.crit_dmodx(confidence=confidence))
        plt.xlabel('Sample ID')
        plt.ylabel('Distance to X-hyperplane')

        return plt.gcf().axes


def fit_pls(X, Y, pipeline=None, cv_object=None, max_lv=10):
    r"""
    Auto-calibrate PLS model and generate analytical plots

    A PLS model is calibrated based on the maximization of the coefficient of
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
    plt.ylabel('R2, Q2')

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
