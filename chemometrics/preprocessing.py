# Copyright 2020 Matthias Rüdt
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
import warnings


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
    least-squares regression [1]. The alogrithm stops as soon as the weights
    are not adjusted from the previous cycle.

    References
    ----------
    .. [1] Hans F.M. Boelens, Reyer J. Dijkstra, Paul H.C. Eilers, Fiona
    Fitzpatrick, Johan A. Westerhuis, New background correction method for
    liquid chromatography with diode array detection, infrared spectroscopic
    detection and Raman spectroscopic detection, J. Chromatogr. A, vol. 1057,
    pp. 21-30, 2004.

    Examples
    --------
    X = np.random.normal(size=[10,3])
    y = np.random.normal(size=[10,1])
    beta = chem.asym_als(X, y)
    """
    n, m = X.shape
    if y.ndim == 1:
        y = y[:, None]
    o = y.shape[1]
    beta = np.zeros(shape=[m, o])
    # iterate over each regression
    for i in range(o):
        # initialize variables for iterative regression
        w = np.zeros([n, 1])
        w_new = np.ones([n, 1])
        max_cycles = 10
        cycle = 0
        # iterate linear regression until weights converge
        while not np.all(w == w_new) and cycle < max_cycles:
            # update weights
            w = w_new.copy()
            # update variables for weighted regression
            X_scaled = w * X
            y_scaled = w * y[:, i][:, None]
            # solve weighted least squares problem
            beta[:, i] = np.linalg.lstsq(X_scaled, y_scaled, rcond=-1)[0].T
            # calculate new weights
            residuals = y[:, i] - np.dot(X, beta[:, i])
            w_new[residuals > 0] = asym_factor
            w_new[residuals <= 0] = 1 - asym_factor
            # increase counter
            cycle += 1
    return beta


def emsc(D, p_order=2, background=None, normalize=False, algorithm='als',
         **kwargs):
    r"""
    Perform extended multiplicative scatter correction (EMSC) on `D`

    `emsc` is a spectral pretreatment which is based on a linear decomposition
    of data into baseline contributions and chemical information. Baseline
    contributions are modelled by polynomial terms up to order `p_order`. The
    chemical information is summarized by the mean spectrum orthogonalized to
    the baseline terms. `emsc` additionally provides a the functionality for
    orthogonalizing spectra with respect to background information and for
    normalizing the returned spectra.

    Parameters
    ----------
    D : (n, m) ndarray
        Data to be pretreated. ``n`` samples x ``m`` variables (typically
        wavelengths)
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

    Returns
    -------
    D_pretreated : (n, m) ndarray
        Pretreated data.
    coefficients : (n, k) ndarray
        Coefficient matrix summarizing regression coefficients of EMSC.

    References
    ----------
    An introduction to EMSC is given in [1]. Asymmetric least squares
    regression may be looked up at [2].
    .. [1] Nils Kristian Afseth, Achim Kohler, Extended multiplicative signal
    correction in vibrational spectroscopy, a tutorial, Chemometrics and
    Intelligent Laboratory Systems, vol. 117, pp. 92-99, 2012.
    .. [2] Hans F.M. Boelens, Reyer J. Dijkstra, Paul H.C. Eilers, Fiona
    Fitzpatrick, Johan A. Westerhuis, New background correction method for
    liquid chromatography with diode array detection, infrared spectroscopic
    detection and Raman spectroscopic detection, J. Chromatogr. A, vol. 1057,
    pp. 21-30, 2004.
    """
    # prepare asymmetry factor for als
    if 'asym_factor' in kwargs:
        asym_factor = kwargs['asym_factor']
    else:
        asym_factor = 0.1

    n_series, n_variables = D.shape
    # generate matrix of baseline polynomials
    baseline = np.zeros([n_variables, p_order+1])
    multiplier = np.linspace(-1, 1, num=n_variables)
    for i in range(0, p_order+1):
        baseline[:, i] = multiplier ** i
    # matrix for summarizing all factors
    regressor = baseline.copy()

    # if included: prepare background data
    if background is not None:
        # convert background to two dimensional array as otherwise hard to
        # detect/interprete errors may occure
        if background.ndim == 1:
            background = background[:, None]
        # orthogonalize background to baseline information
        beta_background = asym_ls(baseline, background,
                                  asym_factor=asym_factor)
        background_pretreated = background - np.dot(baseline, beta_background)
        regressor = np.concatenate([regressor, background_pretreated], axis=1)

    # prepare estimate of chemical information
    D_bar = np.mean(D, axis=0)[:, None]  # mean spectra
    beta_D_bar = asym_ls(regressor, D_bar, asym_factor=asym_factor)
    D_bar_pretreated = D_bar - np.dot(regressor, beta_D_bar)
    regressor = np.concatenate((regressor, D_bar_pretreated), axis=1)

    # perform EMSC on data
    coefficients = asym_ls(regressor, D.T, asym_factor=asym_factor)
    D_pretreated = D.T - np.dot(regressor[:, :-1], coefficients[:-1, :])
    if normalize:
        D_pretreated = D_pretreated @ np.diag(1/coefficients[-1, :])

    return D_pretreated.T, coefficients.T


def whittaker(X, penalty, constraint_order=2):
    r"""
    Smooth `X` with a whittaker smoother

    `whittaker` smooths `X` with a whittaker smoother. The smoother smooths
    the data with a non-parametric line constraint by its second derivative
    smoothness. `penalty` defines the penalty on non-smoothness.
    The whittaker smoother is very efficient and a useful drop-in replacement
    for Savitzky-Golay smoothing.

    Parameters
    ----------
    X : (n, m) ndarray
        Matrix containing data series to smooth. The function expects. ``n``
        datapoints in ``m``series.
    penalty : float
        scaling factor of the penality term for non-smoothness
    constraint_order : int
        defines on which order of derivative the constraint acts on.

    Returns
    -------
    smoothed : (n, m) ndarray
        Smoothed `X`

    Notes
    -----
    `whittaker` uses a sparse matrices for efficiency reasons. `X` may
    however be a full matrix.
    In contrast to the proposed algorithm by Eilers [1], no Cholesky
    decomposition is used. The reason is twofold. The Cholesky decomposition
    is not implemented for sparse matrices in Numpy/Scipy. Eilers uses the
    Cholesky decomposition to prevent Matlab from "reordering the sparse
    equation systems for minimal bandwidth". Matlab seems to rely on UMFPACK
    for sparse matrix devision [2] which implements column reordering for
    sparsity preservation. As sparse matrix we are working with is square and
    positive-definite, we can rely on the builtin `factorize` method, which
    solves with UMFPACK if installed, otherwise with SuperLU.

    References
    ----------
    Application of whittaker smoother to spectroscopic data [1].

    .. [1] Paul H. Eilers, A perfect smoother, Anal. Chem., vol 75, 14, pp.
    3631-3636, 2003.
    .. [2] UMFPAC, https://en.wikipedia.org/wiki/UMFPACK, accessed
    03.May.2020.
    """
    n_var, n_series = X.shape
    D = _sp_diff_matrix(n_var, constraint_order)
    C = sparse.eye(n_var) + penalty * D.transpose().dot(D)
    X_smoothed = np.zeros([n_var, n_series])
    lin_solve = splinalg.factorized(C)

    for i in range(n_series):
        X_smoothed[:, i] = lin_solve(X[:, i])

    return X_smoothed


def _sp_diff_matrix(m, diff_order=1):
    r"""
    generates a sparse difference matrix used for ``whittaker``
    """
    E = sparse.eye(m).tocsc()
    for i in range(diff_order):
        E = E[1:, :] - E[:-1, :]
    return E
