"""A (spectroscopic) chemometric library

Provides a number of functions for data pretreatment, plotting and regression.
"""

import numpy as np


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
