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


def generate_spectra(n_wl, n_band, bandwidth):
    r"""
    Generate a dummy spectra with n_band

    Parameters
    ----------
    n_wl : int
        number of wavelengths/signals to generate
    rel_lengthscale : float
        lengths scale of the gaussian process kernel
    n_band : int
        number of background spectra to generate

    Returns
    -------
    background : (n_wl, n_band)
        artificial background spectra
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

    Parameters
    ----------
    n_wl : int
        number of wavelengths/signals to generate
    rel_lengthscale : float
        lengths scale of the gaussian process kernel
    size : int
        number of background spectra to generate

    Returns
    -------
    background : (n_wl, size)
        artificial background spectra
    """
    mean = np.zeros(n_wl)
    # use a gaussian kernel based weighting for the covariance matrix
    x = np.arange(n_wl)[:, None]
    dist = x.T - x
    cov = np.exp(-(dist / (n_wl * rel_lengthscale))**2)

    # draw 'size' samples from gaussian process as data
    background = np.random.multivariate_normal(mean, cov, size=size)
    return background.T


def generate_data(n_wl=100, n_samples=100, n_conc=2, noise=0.1):
    """
    Generate artificial spectroscopic XY data without background
    """
    Y = np.random.uniform(size=[n_samples, n_conc])
    spectra = np.zeros(shape=[n_wl, n_conc])

    for i in range(n_conc):
        spectra[:, i] = generate_spectra(n_wl, n_wl//20, 1)

    X = Y @ spectra.T + np.random.normal(scale=noise,
                                         size=[n_samples, n_wl])
    return X, Y


def _gaussian_fun(x, mu, sigma):
    r"""
    Generates Gaussian profile
    """
    return np.exp(-((x - mu) / sigma) ** 2 / 2)
