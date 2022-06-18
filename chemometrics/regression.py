# Copyright 2021, 2022 Matthias Rüdt
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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin
)
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
import matplotlib.pyplot as plt
from .utils import pseudo_voigt_spectra
from .base import LVmixin


class PLSRegression(_PLSRegression, LVmixin):
    r"""
    PLS regression with added chemometric functionality


    References
    ----------
    Calculations according to

    .. [Eriksson] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
           Wikström, and S. Wold. Multi- and Megavariate Data Analysis,
           Part I Basic Principles and Applications. Second Edition.


    """

    def __init__(self, n_components=2, *,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(n_components=n_components, scale=False,
                         max_iter=max_iter, tol=tol, copy=copy)

    def fit(self, X, Y):
        super().fit(X, Y)
        self.n_samples_ = self.x_scores_.shape[0]
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

    def hat(self, X):
        r"""
        Calculate the hat (projection) matrix

        Calculate the hat matrix in the X/Y score space. The hat matrix
        :math:`H` projects the observed :math:`Y` onto the predicted
        :math:`\hat Y`. For  obtaining the standard hat matrix, the provided X
        matrix should  correspond to the matrix used during the calibration
        (call to `fit`)  [Eriksson]_.

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


def fit_pls(X, Y, pipeline: Pipeline = None, cv_object=None, max_lv=10):
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
        X_preprocessed = pipeline[:-1].transform(X)
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


class IHM(TransformerMixin, MultiOutputMixin,
          BaseEstimator):
    """
    Indirect Hard Modeling (IHM) without linear regression

    IHM models spectra based on a mechanistic model of multiple
    pure component spectra each consisting of flexible peaks. The spectra are
    described by the peak parameters. For new spectra, the mechanistic spectral
    model is adjusted by a parameter optimization. This allows to correct for
    a variety of effects such as instrument specific shifts or sensor
    variability. IHM returns the parameters of the fitted model.

    Parameters
    ----------
    features : ndarray of shape (n_features, 1), default=None
        feature-related x variable
    peak_parameters : list of ndarrays
        List of peak parameter arrays
    bl_order : int (default: 2)
        Order of background polynome
    spectra_generator : function, default=pseudo_voigt_spectra
        Reference to spectra-generating function
    method : {'LG'}
        Algorithm for spectral fit:
            - 'LG' (default): largest gradient method as descirbed in
                [EKriesten]_.
    gradient_truncation : int (default: 20)
        For peak_parameter fitting, only step along the most important gradient
        directions up to the number of directions given by
        `gradient_truncation`.

    Attributes
    ----------
    n_components_ : int
        Number of components in model
    linearized_breakpoints_ : ndarray
        Vector which indicates at what point different sections of the
        linarized parameter vector end. Structure: (backkground parameters,
        component weights, component shifts, spectra parameters)

    Notes
    -----
    The current optimization strategy follows the largest gradient approach
    described in [EKriesten]_ . To reduce the complexity of the optimization
    problem, first global parameters are optimized (background, spectral shift,
    spectral weights). Peak parameters are optimized one by one depending
    on the gradient size up to a certain number of parameters.

    Nomenclature
    ------------
    features:
        vector of feature or spectral dimension (e.g. wavelength, wavenumber)
    peak:
        a feature-associated effect described by a scaled
        probablity function
    component:
        a chemical species described by a linear combination of peaks
    baseline:
        slowely varying effect not associated to a specific component
    spectra:
        a linear combination of multiple components and baseline effects

    References
    ----------
    Implemented according to
    .. [EKriesten] Kriesten et al. Chemometrics and Intelligent Laboratory
        Systems 91 (2008) 181-193.
    """

    def __init__(self, features, peak_parameters, bl_order=2,
                 spectra_generator=pseudo_voigt_spectra,
                 method='LG',
                 gradient_truncation=20):

        self.features = features
        self.n_components_ = len(peak_parameters)
        self.peak_parameters = np.concatenate(peak_parameters, axis=1)
        self.spectra_generator = spectra_generator
        self.bl_order = bl_order
        self.gradient_truncation = gradient_truncation
        self.method = method

        n_peakparameters = self.peak_parameters.size

        # summarize breaks in peak_parameters matrix
        self._component_breaks = np.array(
            [component.shape[1] for component in peak_parameters]
        ).cumsum()

        # summarize information on total parameters
        self.linearized_breakpoints_ = np.array((
                self.bl_order+1,  # baseline
                self.n_components_,  # component weights
                self.n_components_,  # component shifts
                n_peakparameters  # number of peakparameter
            )).cumsum()

        # prepare polynomial baseline matrix for later use
        self._baseline = np.zeros((self.bl_order+1, self.features.shape[0]))
        multiplier = np.linspace(-1, 1, num=self.features.shape[0])
        for i in range(0, self.bl_order+1):
            self._baseline[i, :] = multiplier ** i

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        """
        Transform spectra in IHM parameter set
        """
        X_transformed = np.apply_along_axis(self._adjust2spectrum, 1, X)
        return X_transformed

    def _adjust2spectrum(self, spectrum):
        self._current_spectrum = spectrum
        # intialize parameter estimates
        self._weights = np.ones(self.n_components_)
        self._bl = np.zeros(self.bl_order+1)
        self._shifts = np.zeros(self.n_components_)
        self._peak_parameters = self.peak_parameters.copy()

        # fit global parameters
        ini_par = np.concatenate([self._bl, self._weights, self._shifts])
        result = least_squares(self._obj_fun_global_par, ini_par)
        breaks = self.linearized_breakpoints_
        self._bl = result.x[:breaks[0]]
        self._weights = result.x[breaks[0]:breaks[1]]
        self._shifts = result.x[breaks[1]:breaks[2]]

        if self.method == 'LG':
            self._largest_gradient()
        else:
            raise(KeyError(f'Unknown method {self.method}'))

        linearized_parameters = np.concatenate([
            self._bl,
            self._weights,
            self._shifts,
            self._peak_parameters.ravel()
        ])
        return linearized_parameters

    def _largest_gradient(self):
        breaks = self.linearized_breakpoints_
        # select parameters for optimization
        jac = approx_derivative(
            lambda x: np.sum(self._obj_fun_all_peak_parameters(x)**2),
            self.peak_parameters.ravel()
        )
        parameter_sequence = np.argsort(jac)[::-1]
        mask = np.zeros(self.peak_parameters.shape)

        if self.gradient_truncation < parameter_sequence.size:
            mask[parameter_sequence[:self.gradient_truncation]] = 1
        else:
            mask[:] = 1

        # adjust peak positions
        sparsity = np.ones(self.features.shape[0])[:, None]*mask[0, :, None].T
        estimate_pp = least_squares(
            self._obj_fun_peak_position,
            self._peak_parameters[0, :],
            jac_sparsity=sparsity
        )
        self._peak_parameters[0, :] = estimate_pp.x

        # fit baseline, weights
        ini_par = np.concatenate([self._bl, self._weights])
        result = least_squares(self._obj_fun_baseline_weights, ini_par)
        self._bl = result.x[:breaks[0]]
        self._weights = result.x[breaks[0]:breaks[1]]

        # optimize all peak parameters
        sparsity = np.ones(
            self.features.shape[0]
        )[:, None]*mask.ravel()[:, None].T
        estimate_pp = least_squares(
            self._obj_fun_all_peak_parameters,
            self.peak_parameters.ravel(),
            jac_sparsity=sparsity
        )
        self._peak_parameters = np.reshape(estimate_pp.x,
                                           self.peak_parameters.shape)

        # fit baseline, weights
        ini_par = np.concatenate([self._bl, self._weights])
        result = least_squares(self._obj_fun_baseline_weights, ini_par)
        self._bl = result.x[:breaks[0]]
        self._weights = result.x[breaks[0]:breaks[1]]

    def _obj_fun_global_par(self, global_param):
        breaks = self.linearized_breakpoints_
        bl = global_param[:breaks[0]]
        weights = global_param[breaks[0]:breaks[1]]
        shifts = global_param[breaks[1]:breaks[2]]
        spectrum = self._compile_spectrum(bl, weights, shifts,
                                          self._peak_parameters)
        return self._current_spectrum - spectrum

    def _obj_fun_peak_position(self, peak_positions):
        # generate adjusted peak parameter set
        peak_parameters = self._peak_parameters.copy()
        peak_parameters[0, :] = peak_positions
        spectrum = self._compile_spectrum(self._bl, self._weights,
                                          self._shifts, peak_parameters)
        return self._current_spectrum - spectrum

    def _obj_fun_baseline_weights(self, bl_weights):
        breaks = self.linearized_breakpoints_
        bl = bl_weights[:breaks[0]]
        weights = bl_weights[breaks[0]:breaks[1]]
        spectrum = self._compile_spectrum(bl, weights, self._shifts,
                                          self._peak_parameters)
        return self._current_spectrum - spectrum

    def _obj_fun_all_peak_parameters(self, peak_parameters):
        peak_parameters = peak_parameters.reshape(self.peak_parameters.shape)
        spectrum = self._compile_spectrum(self._bl, self._weights,
                                          self._shifts, peak_parameters)
        return self._current_spectrum - spectrum

    def _compile_spectrum(self, bl, weights, shifts, peak_parameters):
        """
        Generate a spectrum based on provided parameter set

        Returns
        -------
        spectra : ndarray (n_components x n_features)
        """

        # generate pure component spectra
        pure_spectra = np.zeros([self.n_components_, self.features.shape[0]])
        start_ind = 0
        for i, end_ind in enumerate(self._component_breaks):
            pure_spectra[i, :] = self.spectra_generator(
                self.features - shifts[i],
                peak_parameters[:, start_ind:end_ind]
            ).T
            start_ind = end_ind
        # generate spectra
        spectra = weights @ pure_spectra
        baseline = bl @ self._baseline
        return spectra + baseline


class IHMRegression(IHM):
    """
    Indirect Hard Modeling (IHM) of spectra with OLS prediction

    IHM models spectra based on a mechanistic description of multiple
    pure component spectra each consisting of a set of peaks. The spectra are
    described by the peak parameters. For new spectra, the mechanistic spectral
    model is adjusted by a parameter optimization. This allows to correct for
    a variety of effects such as instrument specific shifts or sensor
    variability. The weights of the optimized spectral parameter set are used
    for concentration predictions. Concentrations should be given as
    molalities. At low concentrations, the solute concentrations may be
    approximated as mass concentration, molarity, molar fraction, weight
    fraction etc (see Notes).

    Note: The first component spectra is always assumed to be the
    solvent. The predicted molalities for the solvent are always constant and
    should, by definition, be equal to the inverse molar weight 1/M_W [mol/kg].

    Parameters
    ----------
    features : ndarray of shape (n_features, 1), default=None
        feature-related x variable
    peak_parameters : list of ndarrays
        List of peak parameter arrays
    bl_order : int (default: 2)
        Order of background polynome
    spectra_generator : function, default=pseudo_voigt_spectra
        Reference to spectra-generating function
    method : {'LG'}
        Algorithm for spectral fit:
            - 'LG' (default): largest gradient method as descirbed in
                [EKriesten]_.
    gradient_truncation : int (default: 20)
        For peak_parameter fitting, only step along the most important gradient
        directions up to the number of directions given by
        `gradient_truncation`.

    Attributes
    ----------
    n_components_ : int
        Number of components in model
    linearized_breakpoints_ : ndarray
        Vector which indicates at what point different sections of the
        linarized parameter vector end. Structure: (background parameters,
        component weights, component shifts, spectra parameters)
    regressor_ : LinearRegression
        Estimator which converts weights to concentrations

    Notes
    -----
    The current optimization strategy follows the largest gradient approach
    described in [EKriesten]_ . To reduce the complexity of the optimization
    problem, first global parameters are optimized (background, spectral shift,
    spectral weights). Peak parameters are optimized one by one depending
    on the gradient size up to a certain number of parameters.

    Concentration: The optical spectroscopic sensor probes an unknown  volume
    $V$. Each chemical species in the probed volume contributes to the Raman
    signal with a weight $w_i$ proportional to the number of moles in the
    probed volume.
    .. math:: w_i \propto N_i

    We may furthermore expand:
    .. math::
        w_i K_i= N_i
        w_i K_i = N_i

    The solvent is denoted by a subscript $S$. Following the same argument as
    above, we may define the solvent mass in the probed volume
    .. math:: m_S = w_S K'_S = w_S \frac{K_S}{M_{W, S}}

    with :math: `M_{W, S}`, the molar mass of the solvent.

    Molality is given by:
    .. math::
        b_i = \frac{N_i}{m_S}=\frac{w_i K_i}{w_S K'_S}=k_i \frac{w_i}{w_S}

    Nomenclature
    ------------
    features:
        feature or spectral dimension (e.g. wavelength, wavenumber)
    peak:
        a feature-associated effect described by a scaled
        probablity function
    component:
        a chemical species described by a linear combination of peaks
    baseline:
        slowely varying effect not associated to a specific component
    spectra:
        a linear combination of multiple components and baseline effects
    """
    def __init__(self, features, peak_parameters, bl_order=2,
                 spectra_generator=pseudo_voigt_spectra,
                 method='LG',
                 gradient_truncation=20):
        self.regressor_ = LinearRegression(fit_intercept=False)

        super().__init__(features, peak_parameters, bl_order,
                         spectra_generator, method, gradient_truncation)

    def fit(self, X, y):
        """
        Calibrate regression model of IHM
        """
        parameters = self.transform(X)

        weight_range = self.linearized_breakpoints_[0:2]
        weights = parameters[:, weight_range[0]:weight_range[1]]
        normalized_weights = weights / weights[:, 0, None]

        self.regressor_ = self.regressor_.fit(normalized_weights, y)
        return self

    def predict(self, X):
        """
        Predict concentrations from given X
        """
        parameters = self.transform(X)
        weight_range = self.linearized_breakpoints_[0:2]
        weights = parameters[:, weight_range[0]:weight_range[1]]
        normalized_weights = weights / weights[:, 0, None]

        y = self.regressor_.predict(normalized_weights)
        return y
