"""A (spectroscopic) chemometric library

Provides a number of functions for data pretreatment, plotting and regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import pdb


def asym_ls(X, y, asym_factor=0.1):
    r"""
    Perform an asymmetric least squares regression.

    A linear regression is performed where negative (default) or positive
    residuals are more strongly penalized.
    `asym_factor` defines the asymmetry of the regression.

    Parameters
    ----------
    X : (n, m) ndarray

    y : (n, 1) ndarray
        response

    Returns
    -------
    beta : (m, 1) ndarray
        regression coefficients

    Notes
    -----

    Following equation is solved:

    .. math:: \hat{\beta} = \argmin_\beta (w(y-Xb))^2

    with :math:`w` being a diagonal matrix of weights.
    If :math:`(y-Xb)>0: w_ii=asym_factor`
    otherwise :math:`w_ii = 1 - asym_factor`

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
    m = np.shape(y)[0]
    w = np.zeros([m, 1])
    w_new = np.ones([m, 1])
    X_scaled = X
    y_scaled = y

    # iterate weighted ls, until it converges to a solution
    while not np.all(w == w_new):
        # update weights
        w = w_new.copy()
        # update variables for weighted regression
        X_scaled = w * X
        y_scaled = w * y
        # solve weighted ls
        beta = np.linalg.lstsq(X_scaled, y_scaled, rcond=None)[0]

        # calculate new asymmetry weights
        residuals = y - np.dot(X, beta)
        w_new[residuals > 0] = asym_factor
        w_new[residuals <= 0] = 1 - asym_factor
        # small_res = np.abs(residuals) < eps
        # w_new[small_res] = 1 - asym_factor + (residuals[small_res]/eps + 1) *
        #    (asym_factor - 0.5)
    return beta


def plot_colored_series(x, y, reference=None):
    """
    Plot spectra colored by position or intensity
    """
    # define number of input series for line plot
    if (y.ndim > 1):
        n_series = y.shape[1]
    else:
        n_series = 1
    # if no reference is given a dummy reference is needed
    if reference is None:
        reference = np.arange(n_series)
    myMapper = matplotlib.cm.ScalarMappable(cmap='viridis')
    colors = myMapper.to_rgba(reference)
    lines = []
    for i in range(n_series):
        line_i = plt.plot(x, y[:, i], color=colors[i, :])
        lines.append(line_i)
    return lines


def generate_spectra(n_wl, n_band, bandwidth):
    """
    Generate a dummy spectra with n_band
    """
    wl = np.arange(n_wl)
    spectra = np.zeros(n_wl)
    for i in range(n_band):
        center_wl = np.random.choice(wl)
        bandwidth_i = np.random.gamma(shape=bandwidth*1.2, scale=bandwidth)
        intensity = np.random.poisson(lam=5)*np.random.normal(loc=1,
                                                              scale=0.2)

        current_spectra = intensity * _gaussian_fun(wl, center_wl, bandwidth_i)
        spectra += current_spectra
    return spectra


def _gaussian_fun(x, mu, sigma):
    """
    Generates Gaussian profile
    """
    return np.exp(-((x - mu) / sigma) ** 2 / 2)
