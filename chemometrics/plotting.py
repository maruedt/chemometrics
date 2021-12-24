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


import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from sklearn.utils.extmath import randomized_svd


def plot_colored_series(Y, x=None, reference=None):
    r"""
    Plot lines colored by position or `reference`

    Generate a line plot with `x` on x-axis and one or multiple dataseries
    `Y`. The lines are either colored by position in the matrix `Y` or by
    value in the `reference` matrix.

    Parameters
    ----------
    Y : (n, m) ndarray
        Matrix containing data series to plot. The function expects. `n`
        datapoints in `m` series.
    x : {None, (n,) ndarray}
        Location on x-axis
    reference : {None (default), (m,) ndarray}
        Reference values to color data series by. If None, the series are
        colored by the position in the second dimension of matrix `Y`.

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

    References
    ----------
    Plot style adapted from personal communication with Matthias Sawall as
    in Figure 5 [1]_.

    .. [1] M. Sawall, A. Börner, C. Kubis, D. Selent, R. Ludwig, and K.
           Neymeyr. Model-free multivariate curve resolution combined with
           model-based kinetics: Algorithm and applications. J. Chemom.,
           26:538–548, 2012.
    """
    u, s, vh = randomized_svd(D, n_components=n_eigenvalues, random_state=None)

    _ = plt.figure(figsize=(15, 5))
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
