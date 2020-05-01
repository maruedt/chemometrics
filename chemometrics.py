"""A (spectroscopic) chemometric library

Provides a number of functions for data pretreatment, plotting and regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


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


def whittacker(X, penalty):
    r"""
    Smooth `X` with a whittacker smoother

    `whittacker` smooths `X` with a Whittacker smoother. The smoother smooths
    the data with a non-parametric line constraint by its second derivative
    smoothness. `penalty` defines the penalty on non-smoothness.
    The Whittacker smoother is very efficient and a useful drop-in replacement
    for Savitzky-Golay smoothing.

    Parameters
    ----------
    X : (n, m) ndarray
        Matrix containing data series to smooth. The function expects. ``n``
        datapoints in ``m``series.
    lambda : float
        scaling factor of the penality term for non-smoothness

    Returns
    -------
    smoothed : (n, m) ndarray
        Smoothed `X`

    Notes
    -----
    `whittacker` uses a sparse matrices for efficiency reasons. `X` may
    however be a full matrix.

    References
    ----------
    Application of Whittacker smoother to spectroscopic data [1].

    .. [1] Paul H. Eilers, A perfect smoother, Anal. Chem., vol 75, 14, pp.
    3631-3636, 2003.
    """
    pass


def plot_colored_series(Y, x=None, reference=None):
    r"""
    Plot matrix colored by position or `reference`

    Generate a line plot with `x` on x-axis and one or multiple dataseries
    `Y`. The lines are either colored by position in the matrix `Y` or by
    value in the `reference` matrix.

    Parameters
    ----------
    Y : (n, m) ndarray
        Matrix containing data series to plot. The function expects. ``n``
        datapoints in ``m``series.
    x : {None, (n,) ndarray}
        Location on x-axis
    reference : {None (default), (m,) ndarray}
        Reference values to color data series by. If None, the series are
        colored by the position in the second dimension of matrix ``Y``.

    Returns
    -------
    lines : list
        A list of line objects generated by plotting the spectra.
    """
    # define number of input series for line plot
    if (Y.ndim > 1):
        n_series = Y.shape[1]
    else:
        n_series = 1
    if x is None:
        x = np.arange(Y.shape[0])
    # if no reference is given a dummy reference is needed (sequential
    # coloring)
    if reference is None:
        reference = np.arange(n_series)
    myMapper = matplotlib.cm.ScalarMappable(cmap='viridis')
    colors = myMapper.to_rgba(reference)
    lines = []
    for i in range(n_series):
        line_i = plt.plot(x, Y[:, i], color=colors[i, :])
        lines.append(line_i[0])
    return lines


def plot_svd(D, n_comp=5, n_eigenvalues=20):
    r"""
    Plot SVD-matrices in three subplots.

    Perform a Singular Value Decomposition (SVD) and plot the three matrices
    in three subplots. The number of singular vectors shown is ``n_comp``. The
    left subplot contains the left singular vectors, the middle subplot the
    singular values, the right subplot the right singular vectors. The
    function is a useful tool to get first insights into a data set. It helps
    to evaluate which components contain information and which mainly noise.
    Compared to Principal Component Analysis (PCA), the singular vectors are
    normalized and scaling results from the eigenvalues.

    Parameters
    ----------
    D : (n, m) ndarray
        Matrix containing data to plot and analyze. The function expects. ``n``
        samples with ``m`` signals (e.g. wavelengths, measurements).

    n_comp : int
        Number of singular vectors to plot.

    n_eigenvalues : int

    Returns
    -------
    fig : figure
        A list of line objects generated by plotting the spectra.
    """
    u, s, vh = linalg.svd(D)

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(u[:, :n_comp])
    plt.subplot(132)
    for i in range(n_eigenvalues):
        if i < n_comp:
            plt.plot(i, s[i], 'o')
        else:
            plt.plot(i, s[i], 'ok')
    plt.gca().set_yscale('log')
    plt.subplot(133)
    plt.plot(vh.T[:, :n_comp])

    return fig


def generate_spectra(n_wl, n_band, bandwidth):
    r"""
    Generate a dummy spectra with n_band
    """
    wl = np.arange(n_wl)
    spectra = np.zeros(n_wl)
    for i in range(n_band):
        center_wl = np.random.choice(wl)
        bandwidth_i = np.random.gamma(shape=bandwidth, scale=bandwidth)
        intensity = np.random.poisson(lam=5)*np.random.normal(loc=1, scale=0.2)
        current_spectra = intensity * _gaussian_fun(wl, center_wl, bandwidth_i)
        spectra += current_spectra
    return spectra


def generate_background(n_wl, rel_lengthscale=0.5, size=1):
    r"""
    Generate dummy background.

    Generate dummy background by drawing samples from a gaussian process.
    """
    mean = np.zeros(n_wl)
    x = np.arange(n_wl)[:, None]
    dist = x.T - x
    cov = np.exp(-(dist / (n_wl * rel_lengthscale))**2)

    # draw a few samples from gaussian process as data
    background = np.random.multivariate_normal(mean, cov, size=size)
    return background


def _gaussian_fun(x, mu, sigma):
    r"""
    Generates Gaussian profile
    """
    return np.exp(-((x - mu) / sigma) ** 2 / 2)
