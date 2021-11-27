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

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (FLOAT_DTYPES)
from scipy.optimize import minimize_scalar


def asym_ls(X, y, asym_factor=0.1):
    r"""
    Perform an asymmetric least squares regression.

    A linear regression is performed where either negative (default) or
    positive residuals are more strongly penalized.
    `asym_factor` defines the asymmetry of the regression.

    Parameters
    ----------
    X : (n, m) ndarray
        coefficient matrix
    y : {(n,) ndarray, (n, o) ndarray}
        dependent variables

    Returns
    -------
    beta : (m, o) ndarray
        regression coefficients

    Notes
    -----
    Following equation is solved:

    .. math:: \hat{\beta} = \argmin_\beta (w(y-Xb))^2

    with :math:`w` being a diagonal matrix of weights.
    If :math:`(y-Xb)>0: w_{ii}=asym_factor`
    otherwise :math:`w_{ii} = 1 - asym_factor`

    The problem is solved by iteratively adjusting :math:`w` and using a normal
    least-squares regression [1]_. The alogrithm stops as soon as the weights
    are not adjusted from the previous cycle or the maximum number of cycles
    are exceeded.

    References
    ----------
    .. [1] Hans F.M. Boelens, Reyer J. Dijkstra, Paul H.C. Eilers, Fiona
       Fitzpatrick, Johan A. Westerhuis, New background correction method for
       liquid chromatography with diode array detection, infrared spectroscopic
       detection and Raman spectroscopic detection, J. Chromatogr. A,
       vol. 1057, pp. 21-30, 2004.

    Examples
    --------
    X = np.random.normal(size=[10,3])
    y = np.random.normal(size=[10,1])
    beta = chem.asym_als(X, y)
    """
    max_cycles = 10
    n, m = X.shape
    if y.ndim == 1:
        y = y[:, None]
    o = y.shape[1]
    beta = np.zeros(shape=[m, o])

    # generate solver function
    def solver1d(y1d):
        fun = _asym_ls_y1d(X, y1d, asym_factor=asym_factor,
                           max_cycles=max_cycles)
        return fun
    # iterate over each regression
    beta = np.apply_along_axis(solver1d, 0, y)
    return beta


def _asym_ls_y1d(X, y, asym_factor=0.1, max_cycles=10):
    """
    Asymmetric least-squares on 1d y data.
    """
    if y.ndim == 1:
        y = y[:, None]
    # initialize variables for iterative regression
    w_new = np.ones([X.shape[0], 1])
    w = None
    cycle = 0
    # iterate linear regression until weights converge
    while not np.all(w == w_new) and cycle < max_cycles:
        # update weights
        w = w_new.copy()
        # update variables for weighted regression
        X_scaled = w * X
        y_scaled = w * y
        # solve weighted least squares problem
        beta = np.linalg.lstsq(X_scaled, y_scaled, rcond=-1)[0]
        # calculate new weights
        residuals = y - np.dot(X, beta)
        w_new[residuals > 0] = asym_factor
        w_new[residuals <= 0] = 1 - asym_factor
        # increase counter
        cycle += 1
    return beta[:, 0]


class Emsc(TransformerMixin, BaseEstimator):
    r"""
    Performs extended multiplicative scatter correction (EMSC).

    `Emsc` is a spectral pretreatment which is based on a linear decomposition
    of data into baseline contributions and chemical information. Baseline
    contributions are modelled by polynomial terms up to order `p_order`. The
    chemical information is summarized by the mean spectrum orthogonalized to
    the baseline terms. `Emsc` additionally provides a the functionality for
    orthogonalizing spectra with respect to background information and for
    normalizing the returned spectra.

    Parameters
    ----------
    p_order : int
        Polynoms up to order `p_order` are included for baseline subtraction.
    background : {None (default), (m, o) ndarray}
        Perform additional orthogonalization with respect to background. If
        ``None`` omitted. Otherwise, ``m`` variables x ``o`` background spectra
    normalize : {False (default), True}
        Perform normalization of results
    algorithm : {'als' (default)}
        choice of algorithms for regression. Currently, only asymmetric least
        squares is supported

    Attributes
    ----------
    regressor_ : array of floats
        Matrix of regressor variables for background subtraction.
    coefficients_ : array of floats
        Coefficients of last transform.

    Notes
    -----
    An introduction to EMSC is given in [1]_. Asymmetric least squares
    regression may be looked up at [2]_.

    References
    ----------
    .. [1] Nils Kristian Afseth, Achim Kohler, Extended multiplicative signal
       correction in vibrational spectroscopy, a tutorial, Chemometrics and
       Intelligent Laboratory Systems, vol. 117, pp. 92-99, 2012.
    .. [2] Hans F.M. Boelens, Reyer J. Dijkstra, Paul H.C. Eilers, Fiona
       Fitzpatrick, Johan A. Westerhuis, New background correction method for
       liquid chromatography with diode array detection, infrared spectroscopic
       detection and Raman spectroscopic detection, J. Chromatogr. A,
       vol. 1057, pp. 21-30, 2004.
    """

    def __init__(self, p_order=2, background=None, normalize=False,
                 algorithm='als', asym_factor=0.1):
        self.p_order = p_order
        # convert background to two dimensional array if exists
        if background is not None:
            if background.ndim == 1:
                background = background[:, None]
        self.background = background
        self.normalize = normalize
        self.algorithm = algorithm
        self.asym_factor = asym_factor

    def fit(self, X, y=None):
        r"""
        Calculate regression matrix for later use.

        Parameters
        ----------
        X : (n, m) ndarray
            Data to be pretreated. ``n`` samples x ``m`` variables (typically
            wavelengths)
        y
            Ignored
        """
        X = self._validate_data(X, estimator=self, dtype=FLOAT_DTYPES)

        n_series, n_variables = X.shape
        # generate matrix of baseline polynomials
        baseline = np.zeros([n_variables, self.p_order+1])
        multiplier = np.linspace(-1, 1, num=n_variables)
        for i in range(0, self.p_order+1):
            baseline[:, i] = multiplier ** i
        # matrix for summarizing all factors
        regressor = baseline.copy()

        # if included: prepare background data
        if self.background is not None:
            # orthogonalize background to baseline information
            beta_background = asym_ls(baseline, self.background,
                                      asym_factor=self.asym_factor)
            background_pretreated = self.background - np.dot(baseline,
                                                             beta_background)
            regressor = np.concatenate([regressor, background_pretreated],
                                       axis=1)

        # prepare estimate of chemical information
        X_bar = np.mean(X, axis=0)[:, None]  # mean spectra
        beta_X_bar = asym_ls(regressor, X_bar, asym_factor=self.asym_factor)
        X_bar_pretreated = X_bar - np.dot(regressor, beta_X_bar)
        self.regressor_ = np.concatenate((regressor, X_bar_pretreated), axis=1)

        return self

    def transform(self, X, copy=True):
        r"""
        Perform baseline correction by subtracting fit of regresor on `X`

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        """

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES, copy=copy)
        # perform EMSC on data
        coefficients = asym_ls(self.regressor_, X.T,
                               asym_factor=self.asym_factor)
        X = X.T - np.dot(self.regressor_[:, :-1], coefficients[:-1, :])
        if self.normalize:
            X = X @ np.diag(1/coefficients[-1, :])

        self.coefficients_ = coefficients.T
        return X.T


class Whittaker(TransformerMixin, BaseEstimator):
    r"""
    Smooth `X` with a whittaker smoother

    `Whittaker` smooths `X` with a whittaker smoother. The smoother smooths
    the data with a non-parametric line constraint by its derivative
    smoothness. `penalty` defines the penalty on non-smoothness.
    The whittaker smoother is very efficient and a useful drop-in replacement
    for Savitzky-Golay smoothing.

    Parameters
    ----------
    penalty : float or 'auto' (default)
        Scaling factor of the penalty term for non-smoothness. If 'auto' is
        given, a penalty is estimated based on an algorithmically optimized
        leave-one-out cross validation

    constraint_order : int
        Defines on which order of derivative the constraint acts on.

    deriv : int
        Derivative of the data. Default: 0 - no derivative. Note: deriv should
        always be <= constraint_order.

    Attributes
    ----------
    estimate_penalty : boolean
        `True` if penalty is estimated.

    penalty_ : float
        The applied penalty for smoothing.

    solve1d_ : function
        Solver for smoothing of 1D vector

    Notes
    -----
    `Whittaker` uses sparse matrices for efficiency reasons. `X` may
    however be a full matrix.
    In contrast to the proposed algorithm by Eilers [1]_, no Cholesky
    decomposition is used. The reason is twofold. The Cholesky decomposition
    is not implemented for sparse matrices in Numpy/Scipy. Eilers uses the
    Cholesky decomposition to prevent Matlab from "reordering the sparse
    equation systems for minimal bandwidth". Matlab seems to rely on UMFPACK
    for sparse matrix devision [2]_ which implements column reordering for
    sparsity preservation. As sparse matrix we are working with is square and
    positive-definite, we can rely on the builtin `factorize` method, which
    solves with UMFPACK if installed, otherwise with SuperLU.

    Derivatives are implemented by multiplying the smoothed matrix with a
    (local) difference matrix. This is not explicitly described in [1]_.
    However, the approach is consistent with the underlying idea of the
    Whittaker smoother as the (local) differences are used in the derivation of
    the filter. Note: the derivative should always be smaller equal to the
    constraint order. This is, since the Whittaker filter won't explizitly
    penaltize higher derivative fluctuations than the constraint order.

    References
    ----------
    .. [1] Paul H. Eilers, A perfect smoother, Anal. Chem., vol 75, 14, pp.
       3631-3636, 2003.

    .. [2] UMFPAC, https://en.wikipedia.org/wiki/UMFPACK, accessed
       03.May.2020.
    """

    def __init__(self, penalty='auto', constraint_order=2, deriv=0):
        if penalty == 'auto':
            self.isPenaltyEstimated = True
            self.penalty_ = None
        elif type(penalty) in (int, float):
            self.isPenaltyEstimated = False
            self.penalty_ = penalty
        else:
            raise TypeError('penalty type not correct.')

        self.constraint_order = constraint_order
        self.deriv = deriv

    def fit(self, X, y=None):
        r"""
        Calculate regression matrix for later use.

        Parameters
        ----------
        X : (n, m) ndarray
            Data to be pretreated. ``n`` samples x ``m`` variables (typically
            wavelengths)

        y :
            Ignored
        """
        X = self._validate_data(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.isPenaltyEstimated:
            self._estimate_penalty(X)

        self._fit(X)
        return self

    def _fit(self, X):
        "Fit without argument checks or parameter estimation."
        n_series, n_var = X.shape
        C = _get_whittaker_lhs(n_var, self.penalty_, self.constraint_order)
        self.solve1d_ = splinalg.factorized(C.tocsc())

    def transform(self, X, copy=True):
        r"""
        Do Whittaker smoothing.

        Parameters
        ----------
        X : (n, m) ndarray
            Data to be pretreated. ``n`` samples x ``m`` variables (typically
            wavelengths)

        copy : bool (`True` default)
            Whether to genrate a copy of the input file or calculate in place.
        """
        # call to differentation free transform
        X = self._transform(X, copy=copy)
        # differentiate
        if self.deriv > 0:
            X = X @ _sp_diff_matrix(X.shape[1], diff_order=self.deriv).T

        return X

    def _transform(self, X, copy=True):
        """
        Perform the filtering without differentiation.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES, copy=copy)
        X = np.apply_along_axis(self.solve1d_, 1, X)

        return X

    def score(self, X, y=None):
        r"""
        Calculate cross-validation error of whittaker smoother.

        Computes the cross-validation error of a whittaker smoother by a
        leave-one-out cross-validation scheme. The algorithm uses an
        approximation scheme and does not perform the explicit leave-one-out
        cross-validation. Users need should be careful when applying this
        cross-validation scheme to data with autocorrelated noise.
        The algorithm then tends to undersmooth the data.

        Parameters
        ----------
        X : (n, m) ndarray
            Data. ``n`` samples x ``m`` variables (typically
            wavelengths)

        y :
            Ignored

        """
        n_var = X.shape[0]
        z = self._transform(X)
        residuals = z - X
        h_bar = _calc_whittaker_h_bar(n_var, self.penalty_,
                                      self.constraint_order)
        # cross-validation error approximation based on formula proposed by
        # Eiler.
        cv_residuals = residuals / (1 - h_bar)
        error = np.sum(cv_residuals ** 2) / n_var
        return error

    def plot(self, X, logpenalty=[-4, 4]):
        r"""
        Plot CV performance over given range

        Provides an analytical plot of the Whittaker filter score depending on
        the penalty. Each score is the estimated based on a leave-one-out
        approach (see also `score`).
        """
        n_points = 100

        penalties = 10 ** np.linspace(logpenalty[0], logpenalty[1], n_points)
        scores = np.zeros(penalties.shape)
        # store original penalty to restore state
        original_penalty = self.penalty_

        for i in range(n_points):
            self.penalty_ = penalties[i]
            self._fit(X)
            scores[i] = self.score(X)

        plt.plot(penalties, scores)
        plt.xlabel('Penalty')
        plt.ylabel('Score')
        ax = plt.gca()
        ax.semilogx()

        # reset original penalty_
        self.penalty_ = original_penalty
        return ax

    def _obj_fun(self, X, penalty):
        r"Objective funtion for penalty estimation"
        self.penalty_ = 10**penalty
        self._fit(X)
        return self.score(X)

    def _estimate_penalty(self, X):
        r"""
        Estimate optimal penalty based on score.

        The penalty of the whittaker filter is adjusted until the leave-one-out
        error is minimized. The function uses Brent's algorithm and varies
        `penalty_` on a logarithmic scale.
        """

        def obj_fun(log_penalty):
            return self._obj_fun(X, log_penalty)
        bracket = [0, 4]
        res = minimize_scalar(obj_fun, bracket=bracket)

        self.penalty_ = 10**res.x


class AsymWhittaker(TransformerMixin, BaseEstimator):
    r"""
    Background correction `X` with an asymmetric Whittaker filter

    `AsymWhittaker` smooths `X` with an asymmetric Whittaker filter. The filter
    estimates the background by a non-parametric line constraint by its
    derivative smoothness. `penalty` defines the penalty on non-smoothness.

    Parameters
    ----------
    penalty : float
        Scaling factor of the penalty term for non-smoothness.

    constraint_order : int
        Defines on which order of derivative the smoothness constraint acts on.

    asym_factor : float
        Relative weight of negative residuals. Positive residuals obtain
        a weight of `1-asym_factor`. The default value is `0.99`.

    Attributes
    ----------

    background_ : (n, m) ndarray
        Estimated background from last call to `transform`.

    Notes
    -----
    `AsymWhittaker` uses sparse matrices for efficiency reasons. `X` may
    however be a full matrix.
    In contrast to the proposed algorithm by Eilers [1]_, no Cholesky
    decomposition is used. The reason is twofold. The Cholesky decomposition
    is not implemented for sparse matrices in Numpy/Scipy. Eilers uses the
    Cholesky decomposition to prevent Matlab from "reordering the sparse
    equation systems for minimal bandwidth". Matlab seems to rely on UMFPACK
    for sparse matrix devision [2]_ which implements column reordering for
    sparsity preservation. As the sparse matrix we are working with is square
    and positive-definite, we can rely on the builtin `factorize` method, which
    solves with UMFPACK if installed, otherwise with SuperLU.

    References
    ----------
    Application of whittaker smoother to spectroscopic data [1]_.

    .. [1] Paul H. Eilers, A perfect smoother, Anal. Chem., vol 75, 14, pp.
       3631-3636, 2003.
    .. [2] UMFPAC, https://en.wikipedia.org/wiki/UMFPACK, accessed
       03.May.2020.
    """

    def __init__(self, penalty, constraint_order=2, asym_factor=0.99):
        self.penalty = penalty
        self.constraint_order = constraint_order
        self.asym_factor = asym_factor

    def fit(self, X, y=None):
        r"""
        Calculate regression matrix for later use.

        Parameters
        ----------
        X : (n, m) ndarray
            Data to be pretreated. ``n`` samples x ``m`` variables (typically
            wavelengths)

        y :
            Ignored
        """
        X = self._validate_data(X, estimator=self, dtype=FLOAT_DTYPES)

        return self

    def transform(self, X, copy=True):
        r"""
        Do asymmetric Whittaker background subtraction.

        Parameters
        ----------
        X : (n, m) ndarray
            Data to be pretreated. ``n`` samples x ``m`` variables (typically
            wavelengths)

        copy : bool (`True` default)
            Whether to genrate a copy of the input file or calculate in place.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES, copy=copy)
        self.background_ = np.apply_along_axis(self._solve1d, 1, X)
        return X - self.background_

    def _solve1d(self, x, max_iterations=10):
        # prepare loop for asymmetric least squares
        weights = np.ones(self.n_features_in_)
        weights_old = np.zeros(self.n_features_in_)
        counter = 0
        # loop until optimum or max iterations
        while (np.any(weights != weights_old) & (counter < max_iterations)):
            scaled_x = x * weights
            C = _get_whittaker_lhs(self.n_features_in_, self.penalty,
                                   self.constraint_order, weights=weights)
            solve1d = splinalg.factorized(C.tocsc())
            x_bg = solve1d(scaled_x)
            # prepare for next loop: weights and counter
            residuals = x - x_bg
            weights_old = weights.copy()
            weights[residuals < 0] = self.asym_factor
            weights[residuals >= 0] = 1 - self.asym_factor
            counter += 1
        return x_bg


def _get_whittaker_lhs(n_var, penalty, constraint_order, weights=None):
    r"""
    Return the left matrix for whittaker smoothing

    Warning: if weights are used, also right hand side (i.e. unsmoothed data)
    needs to be multiplied by weights)
    """
    D = _sp_diff_matrix(n_var, constraint_order)
    if weights is None:
        lhs = sparse.eye(n_var, format='csc') + penalty * D.transpose().dot(D)
    else:
        lhs = sparse.diags(weights, format='csc') + penalty *\
            D.transpose().dot(D)
    return lhs


def _calc_whittaker_h_bar(n_var, penalty, constraint_order,
                          size_estimator=100):
    r"""
    Calculate estimate of the mean diagonal of the whittaker smoother matrix.
    """
    # reduce size of estimator if necessary
    if n_var < size_estimator:
        size_estimator = n_var

    # the penalty needs to be adjusted for an unbiased estimator
    size_ratio = size_estimator / n_var
    adjusted_penalty = size_ratio ** (2 * constraint_order) * penalty
    # calculate H in the approximation size
    C = _get_whittaker_lhs(size_estimator, adjusted_penalty,
                           constraint_order).toarray()
    rhs = np.eye(size_estimator)
    H = np.linalg.lstsq(C, rhs, rcond=-1)[0]
    return np.mean(np.diag(H))


def _sp_diff_matrix(m, diff_order=1):
    r"""
    Generate a sparse difference matrix used for ``whittaker``
    """
    E = sparse.eye(m, format='csc')
    for i in range(diff_order):
        E = E[1:, :] - E[:-1, :]
    return E
