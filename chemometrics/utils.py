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


import numpy as np


def generate_spectra(n_wl, n_band, bandwidth):
    r"""
    Generate a dummy spectra with n_band

    A dummy spectra is generated based on a given number of peaks. The
    bandwidth, location and height is selected randomly. The center of each
    peak follows a uniform distribution over the range of the spectra.
    Each peaks bandwidth is drawn from a gamma distribution. The gamma
    distribution has a heavy tail resulting in the generation of (also) broad
    peaks. The peak height is Poisson distributed with a mean height of 5. Each
    peak in the artificial spectrum follows a Gaussian shape.

    Parameters
    ----------
    n_wl : int
        number of wavelengths/signals to generate
    n_band : int
        number of bands to generate
    bandwidth : float
        impacts the bandwidth of each band.

    Returns
    -------
    spectra : ndarray (n_wl, )
        artificial spectra
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

    An artificial spectroscopic dataset is generated, which resembles a Raman,
    FTIR acquisition. The spectra are disturbed by white noise. Concentrations
    are uniformly distributed between 0 and 1.

    Parameters
    ----------
    n_wl : int
        number of wavelengths/signals to generate (default: 100)
    n_samples : int
        number of samples generated (default: 100)
    n_conc : int
        number of background spectra to generate
    noise : float
        white noise level in the artificial spectra

    Returns
    -------
    X : (n_samples, n_wl)
        artificial spectra
    Y : (n_samples, n_conc)
        artificial reference data
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


def pseudo_voigt_spectra(x, parameter):
    r"""
    Generates vector based on pseudo-Voigt profiles

    The function takes a wavenumber vector (x) and a set of parameters with the
    shape 4xn. It generates a mechanistic approximation a spectrum based on
    pseudo Voigt peaks.


    Parameters
    ----------
    x : (n_wl, 1)
        evaluated wavelengths
    parameters : (4, n_peaks)
        parameter definition for voigt profiles. parameters[0, :] define the
        peak positions. parameters[1, :] is proportional to the peak area.
        parameters[2, :] define the percentage of Lorentzian to
        Gauss profile.
        parameters[3, :] define the half width at half height of the peak.

    Returns
    -------
    spectra : (n_wl, 1)
        spectra with n_peaks evaluated voigt profiles

    Notes
    -----
    The pseudo-Voigt peaks are implemented similarly as described in [3]_.
    The parameters are chosen such that they do only affect one
    statistical property of the profile: The definition of the peak width is
    selected such that the Full Width at Half Maximum (FWHM) does not change
    when changing the relative contributions of the Gaussian/Lorentzian
    profile. Similarly, the peak height is not affected by the other parameters
    (peaks normalized by height not by area!). This approach to the peak height
    was chosen as the absolute peak height is a more intuitive parameter than
    the peak area.

    """
    if x.ndim == 1:
        x = x[:, None]

    alpha = parameter[1, :, None]  # n_peaks x 1
    beta = parameter[2, :, None]  # n_peaks x 1
    gamma2 = (parameter[3, :, None].T**2)  # 1 x n_peaks
    omega = parameter[0, :, None].T  # 1 x n_peaks

    # shift x coordinates
    delta_x2 = (x-omega)**2  # n_wl x n_peaks

    # calculate matrix of Gaussian peaks (n_wl x n_peaks)
    gaussian_matrix = np.exp(-np.log(2) * delta_x2/gamma2)

    # calculate matrix of Lorentzian peaks (n_wl x n_peaks)
    lorentzian_matrix = gamma2/(gamma2+delta_x2)

    # convert to vector by matrix multiplication w scaling factor
    gaussian_vector = gaussian_matrix @ (alpha * beta)
    lorentzian_vector = lorentzian_matrix @ (alpha * (1-beta))

    return gaussian_vector + lorentzian_vector
