# Copyright 2022 Matthias Rüdt
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
from scipy import stats
import matplotlib.pyplot as plt


class LVmixin():
    """
    Mixin for latent variable models

    Latent variable models are projection models which transform the X data
    into a latent structure.
    """

    @property
    def _n_components(self):
        """
        Get number of components independent of naming scheme
        """
        try:
            n_comps = self.n_components_
        except AttributeError:
            n_comps = self.n_components
        return n_comps

    def _calculate_x_residual_std_(self, X):
        r"""
        Calculate the standard deviation of the X residuals

        The x residual standard deviation is calculated according to [1]_ not
        including the correction factor v (since it is not exactly defined in
        [1]_).
        """
        X_hat = self.inverse_transform(self.transform(X))
        sse_cal = np.sum((X - X_hat)**2)
        norm_factor = (self.n_samples_ - self.n_components
                       - 1) * (X.shape[1] - self.n_components)
        return np.sqrt(sse_cal / norm_factor)

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
        degf_cali = self.n_features_in_ - self._n_components - 1
        degf_test = self.n_features_in_ - self._n_components
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
        comp = self._n_components
        samples = self.n_samples_
        f_crit = stats.f.ppf(confidence, self._n_components,
                             samples-self._n_components)
        factor = comp * (samples**2 - 1) / (samples * (samples - comp))
        return f_crit * factor

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
