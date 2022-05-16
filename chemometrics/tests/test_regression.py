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

import chemometrics as cm  # accounts for relativ path
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import unittest
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib
from scipy.stats import binom


class TestPLSRegression(unittest.TestCase):
    """
    Test adjustments done to PLSRegression child class
    """

    def setUp(self):
        self.n_wl = 50
        self.n_samples = 500
        self.n_tests = 100
        self.n_conc = 2

        self.Y = np.random.normal(size=[self.n_samples, self.n_conc])
        noise = 0.1
        spectra = np.zeros(shape=[self.n_wl, self.n_conc])

        for i in range(self.n_conc):
            spectra[:, i] = cm.generate_spectra(self.n_wl, self.n_wl//20, 1)

        self.X = self.Y @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_samples, self.n_wl]
        )
        self.pls = cm.PLSRegression()
        self.pls = self.pls.fit(self.X, self.Y)

        self.Y_test = np.random.normal(size=[self.n_tests, self.n_conc])
        self.X_test = self.Y_test @ spectra.T + np.random.normal(
            scale=noise,
            size=[self.n_tests, self.n_wl]
        )

    def test_vip_shape(self):
        """
        Test the shape of the vip variable
        """
        self.assertTrue(self.pls.vip_.shape == (self.n_wl, ))

    def test_vip_value(self):
        """
        Test that sum of squared VIPs == number of X variables (definition!)
        """
        self.assertTrue(np.isclose(np.sum(self.pls.vip_**2), self.n_wl))

    def test_hat_shape(self):
        """
        Test the shape of the vip variable
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(hat.shape == (self.n_samples, self.n_samples))

    def test_hat_works(self):
        """
        Test that hat works according to its definition i.e. H * Y = Ŷ
        """
        hat = self.pls.hat(self.X)
        # mean centered prediction
        Y_hat = self.pls.predict(self.X) - np.mean(self.Y, axis=0)
        self.assertTrue(np.allclose(hat @ self.Y, Y_hat))

    def test_hat_symmetric(self):
        """
        Test that hat matrix is symmetric
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(hat, hat.T))

    def test_hat_indempotent(self):
        """
        Test that hat matrix is indempotent (hat = hat squared)
        """
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(hat, hat @ hat))

    def test_leverage_shape(self):
        """
        Test that leverage provides correct matrix shape
        """
        leverage = self.pls.leverage(self.X)
        self.assertTrue(leverage.shape == (self.n_samples, ))

    def test_leverage_definition(self):
        """
        Test that leverage is diag of the hat matrix
        """
        leverage = self.pls.leverage(self.X)
        hat = self.pls.hat(self.X)
        self.assertTrue(np.allclose(leverage, np.diag(hat)))

    def test_residuals_shape(self):
        """
        Test residual function
        """
        residuals = self.pls.residuals(self.X, self.Y)
        self.assertTrue(residuals.shape == (self.n_samples, self.n_conc))

    def test_residuals_raise_TypeError(self):

        with self.assertRaises(TypeError):
            self.pls.residuals(self.X, self.Y, scaling='bla')

    def test_residuals_scaling(self):
        """
        Test different scaling options of residuals
        """
        residuals_unscld = self.pls.residuals(self.X, self.Y, scaling='none')
        residuals_std = self.pls.residuals(self.X, self.Y,
                                           scaling='standardize')
        residuals_stud = self.pls.residuals(self.X, self.Y,
                                            scaling='studentize')
        std_estimated = residuals_unscld / residuals_std

        std = np.sqrt(np.sum(residuals_unscld**2, axis=0)
                      / (self.X.shape[0] - self.pls.n_components))[:, None].T

        stud_scaling = residuals_std / residuals_stud
        true_scaling = np.sqrt(1 - self.pls.leverage(self.X))
        self.assertTrue(np.allclose(std_estimated, std))
        self.assertTrue(np.allclose(stud_scaling, true_scaling[:, None]))

    def test_plot(self):
        """
        Test that plot generates 4 subplot in figure
        """
        axes = self.pls.plot(self.X, self.Y)
        for ax in axes:
            self.assertIsInstance(ax, matplotlib.axes.Axes)

        lines = axes[2].get_lines()
        p = self.pls.n_components

        for line in lines:
            h = line.get_xdata()
            weighting = h / (1 - h)
            stud_res = line.get_ydata()
            D_rev = stud_res**2 / p * weighting
            close_to05 = np.allclose(D_rev, 0.5)
            close_to1 = np.allclose(D_rev, 1)
            self.assertTrue(close_to05 or close_to1)

    def test_dmodx_shape(self):
        """
        Test that dmodx provides a vector of the correct shape
        """
        dmodx = self.pls.dmodx(self.X)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

        dmodx = self.pls.dmodx(self.X, normalize=False)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

        dmodx = self.pls.dmodx(self.X, absolute=True)
        self.assertTrue(dmodx.shape == (self.n_samples, ))

    def test_dmodx_length(self):
        """
        Test that dmodx = 0 if data from hyperplan is taken
        """
        X_hat = self.pls.inverse_transform(self.pls.transform(self.X))
        dmodx = self.pls.dmodx(X_hat, normalize=False)
        self.assertTrue(np.allclose(dmodx, 0))

    def test_score_plus_dmodx_length(self):
        """
        Test that the variation on the model plane and the variation
        orthogonal to the model plane yield the total variation.
        """
        dmodx = self.pls.dmodx(self.X, normalize=False)
        absolut_dmodx = dmodx**2 * (self.X.shape[1] - self.pls.n_components)
        X_hat_bar = self.pls.x_scores_ @ self.pls.x_loadings_.T
        ss_X_hat = np.sum(X_hat_bar**2, axis=1)
        X_bar = self.X - np.mean(self.X, axis=0)
        ss_X = np.sum(X_bar**2, axis=1)
        self.assertTrue(np.allclose(ss_X_hat + absolut_dmodx, ss_X,
                                    atol=1e-2))

    def test_distance_plot_return_arg(self):
        """
        Test that distance_plot returns an axis cv_object
        """
        ax = self.pls.distance_plot(self.X)
        self.assertIsInstance(ax, list)
        for inst in ax:
            self.assertIsInstance(inst, matplotlib.axes.Subplot)

    def test_crit_dmodx(self):
        """
        Test that number of outliers corresponds

        Performs a binomial statistical test. This test fails more often then
        the confidence level would predict.

        Potential explanation:
        Since dmodx
        is only approximately f2 distributed [1], crit_dmodx is not completely
        accurate. Especially when considering outliers at high confidence, the
        test seems to be too conservative. As a work-around, the binomial test
        is performed at a confidence level of 0.5 for the crit_dmodx (50%
        outliers).

        References
        ----------
        .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
        Wikström, and S. Wold. Multi- and Megavariate Data Analysis, Part I
        Basic Principles and Applications. Second Edition.
        """
        f_confidence = 0.50
        crit_dmodx = self.pls.crit_dmodx(confidence=f_confidence)
        dmodx = self.pls.dmodx(self.X_test)
        count = np.sum(dmodx > crit_dmodx)

        self._test_binom(1-f_confidence, self.n_tests, count)

    def test_cooks_distance_return_shape(self):
        "Test that cooks_distance return args has correct shape"

        cook = self.pls.cooks_distance(self.X, self.Y)
        self.assertTrue(cook.shape == (self.n_samples, self.n_conc))

    def test_cooks_distance_values(self):
        """
        Test reverse function of cooks distance yields residuals
        """
        cook = self.pls.cooks_distance(self.X, self.Y)
        residuals = self.pls.residuals(self.X, self.Y, scaling='none')

        h = self.pls.leverage(self.X)
        weighting = (1-h)**2/h
        mse = (np.sum(residuals**2, axis=0)
               / (self.X.shape[0] - self.pls.n_components))[:, None].T
        inverted_res = np.sqrt(cook * self.pls.n_components * mse
                               * weighting[:, None])

        self.assertTrue(np.allclose(np.abs(residuals), inverted_res))

    def test_crit_dhypx(self):
        """
        Test that number of outliers corresponds for crit_dhypx

        Performs a biomial statistical test. This test fails more often
        then the confidence level would predict. Potential explanation:
        crit_dhypx is not corrected for the bias in estimating the variance of
        the scores based on the training set [1]. This leads to a biased
        normalized dhypx calculations.

        References
        ----------
        .. [1] L. Eriksson, E. Johansson, N. Kettaneh-Wold, J. Trygg, C.
        Wikström, and S. Wold. Multi- and Megavariate Data Analysis, Part I
        Basic Principles and Applications. Second Edition.
        """
        f_confidence = 0.95
        crit_dhypx = self.pls.crit_dhypx(confidence=f_confidence)
        dhypx = self.pls.dhypx(self.X_test)
        count = np.sum(dhypx > crit_dhypx)
        self._test_binom(1-f_confidence, self.n_tests, count)

    def _test_binom(self, p, n_samples, n_positiv, false_positiv=0.005):
        """
        Perform a binomial test
        """
        pdist = binom(n_samples, p)
        limit_low = pdist.ppf(false_positiv/2)
        limit_high = pdist.ppf(1 - false_positiv/2)
        self.assertTrue(n_positiv >= limit_low)
        self.assertTrue(n_positiv <= limit_high)


class TestFit_pls(unittest.TestCase):
    """
    Test fit_pls function.
    """

    def test_raise_TypeError_pipeline(self):
        """
        Test if function raises a TypeError if wrong pipeline is provided
        """
        X, Y = cm.generate_data()
        with self.assertRaises(TypeError):
            cm.fit_pls(X, Y, pipeline="string as an example wrong object")

    def test_pipeline(self):
        """
        Test if pipeline is accepted as input argument
        """
        X, Y = cm.generate_data()
        pipeline = make_pipeline(StandardScaler(), cm.PLSRegression())
        model, analysis = cm.fit_pls(X, Y, pipeline=pipeline)

    def test_accept_cv_object(self):
        """
        Test if function raises a TypeError of wrong CV object is provided
        """
        X, Y = cm.generate_data()

        n_splits = 7
        model, analysis = cm.fit_pls(X, Y, cv_object=KFold(n_splits=n_splits))

        self.assertTrue(analysis['q2'].shape[0] == n_splits)

    def test_return_type(self):
        """
        Test if functions returns an object of type Pipeline
        """
        max_lv = 11
        X, Y = cm.generate_data()
        # basic example
        self._assert_return_args(cm.fit_pls(X, Y, max_lv=max_lv))

        # with pipeline provided
        pipe = make_pipeline(cm.PLSRegression())
        self._assert_return_args(cm.fit_pls(X, Y, pipeline=pipe))

        # with cross validation
        cv = KFold()
        self._assert_return_args(cm.fit_pls(X, Y, cv_object=cv))

    def _assert_return_args(self, returned_args):
        """
        Assert that the returned arguments of fit_pls are correct
        """
        # first return argument must be a pipeline
        pipeline = returned_args[0]
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline[-1], PLSRegression)

        # second return argument must capture info on calibration/cv in dict
        calibration_info = returned_args[1]
        self.assertIsInstance(calibration_info, dict)
        keys = ['q2', 'r2', 'figure_cv', 'figure_model']
        for key in keys:
            self.assertIn(key, calibration_info)


class Test_IHM_LG(unittest.TestCase):
    """
    Test IHM class
    """
    def setUp(self):
        """
        Initialize a dummy instance of ihm
        """
        self.n_features = 200
        self.feature_vector = np.arange(self.n_features)
        self.ini_parameters = [
            np.array([[50, 1, 0.5, 10], [150, 0.5, 0.2, 5]]).T,
            np.array([[100, 1, 0.5, 15]]).T
        ]
        self.ihm = cm.regression.IHM(self.feature_vector, self.ini_parameters)

        # generate set of target parameters different from starting point
        self.true_param = self.ihm.peak_parameters.copy()
        self.true_param *=1.1

        self.bl = np.array([1, 1, 0.5])
        self.weights = np.array([0.7, 0.3])
        self.shift = np.zeros(2)

        # generate spectrum & fit ihm model
        self.test_spectrum = self.ihm._compile_spectrum(
            self.bl, self.weights, self.shift, self.true_param
        )
        self.ihm._adjust2spectrum(self.test_spectrum)

    def test_init(self):
        """
        Assert that initalization of IHM runs as expected
        """
        ihm = self.ihm
        n_components = len(self.ini_parameters)
        self.assertTrue(
            self.ihm.peak_parameters.shape == (4, 3))
        assert_equal(self.ihm.features, self.feature_vector)
        self.assertTrue(ihm.bl_order == 2)
        self.assertTrue(ihm.n_components_ == len(self.ini_parameters))

        assert_allclose(ihm._component_breaks, np.array([2, 3]))
        linearized_breakpoints = np.array([3, 2, 2, 12]).cumsum()
        assert_allclose(ihm.linearized_breakpoints_, linearized_breakpoints)

        assert_allclose(ihm._baseline.shape, np.array([3, self.n_features]))

    def test_compile_spectrum_shape(self):
        """
        Test _compile spectrum
        """
        bl = np.array([1, 1, 0.001])
        weights = np.ones([2])
        shift = np.zeros([2])

        spectrum = self.ihm._compile_spectrum(
            bl, weights, shift, self.ihm.peak_parameters
        )

        self.assertTrue(spectrum.shape==(self.n_features, ))

    def test_adjust2spectrum_baseline(self):
        """
        Assert that _adjust2spectrum baselin is close in dummy case
        """
        # check baseline parameters
        assert_allclose(self.ihm._bl, self.bl, rtol=1e-1)

    def test_adjust2spectrum_peak_parameters(self):
        # check peak positions
        estimated_param = self.ihm._peak_parameters.copy()
        start = 0
        for i, end in enumerate(self.ihm._component_breaks):
            estimated_param[0, start:end] += self.ihm._shifts[i]
            start = end
        assert_allclose(estimated_param[0, :], self.true_param[0, :],
                        rtol=1e-2)

    def test_adjust2spectrum_spectrum(self):
        # check accuracy of estimated spectrum
        estimated_spectrum = self.ihm._compile_spectrum(
            self.ihm._bl, self.ihm._weights, self.ihm._shifts,
            self.ihm._peak_parameters
        )
        assert_allclose(estimated_spectrum, self.test_spectrum, rtol=1e-2)

    def test_transform_shape(self):
        """
        Assert that a matrix of parameters is returned by transform
        """
        n_spectra = 3
        spectra = np.zeros([n_spectra, self.n_features])
        pparam = self.ihm.peak_parameters.copy()
        rng = np.random.default_rng(0)

        for i in range(n_spectra):
            bl = rng.uniform(size=[3])
            weights = rng.uniform(size=[2])
            shifts = rng.uniform(size=[2])
            p_scaler = rng.uniform(low=0.5, high=1.5)
            spectra[i, :] = self.ihm._compile_spectrum(
                bl, weights, shifts, p_scaler*pparam
            )

        transformed = self.ihm.transform(spectra)
        length = self.ihm.linearized_breakpoints_[-1]
        self.assertTrue(transformed.shape == (n_spectra, length))

    def test_raises_unkown_method(self):
        """
        Assert that error is raised if an unkown method is provided to ihm
        """
        with self.assertRaises(KeyError):
            ihm = cm.regression.IHM(self.feature_vector, self.ini_parameters,
                                    method='unkown method')
            ihm.transform(self.test_spectrum[:, None])


if __name__ == "__main__":
    unittest.main()
