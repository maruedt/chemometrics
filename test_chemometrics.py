import chemometrics as cm
import numpy as np
import unittest
import matplotlib


class TestAsym_ls(unittest.TestCase):
    """
    Test cases for `asym_ls`
    """

    def test_shape(self):
        """
        Test that output shape is as expected
        """
        shape_x = [10, 3]
        shape_y = [10, 2]
        expected_shape = (shape_x[1], shape_y[1])

        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)
        output_shape = cm.asym_ls(X, y).shape
        self.assertEqual(expected_shape, output_shape)

    def test_w_ju7symmetric_ls(self):
        """
        Test if symmetric weights results in least squares solution
        """
        shape_x = [10, 3]
        shape_y = [10, 1]
        X = np.random.normal(size=shape_x)
        y = np.random.normal(size=shape_y)

        beta_als = cm.asym_ls(X, y, asym_factor=0.5)
        beta_ls = np.linalg.lstsq(X, y, rcond=-1)[0]
        self.assertTrue(np.all(np.isclose(beta_als, beta_ls)))

    def test_location(self):
        """
        Test if different asym_factors produce qualitatively correct effect
        """
        shape = [10, 1]
        asym_factors = [0.001, 0.0011, 0.5, 0.9, 0.99]
        x = np.ones(shape)
        y = np.arange(shape[0])[:, None]
        last_beta = 0

        for factor_i in asym_factors:
            current_beta = cm.asym_ls(x, y, asym_factor=factor_i)
            self.assertTrue(current_beta > last_beta)


class TestEmsc(unittest.TestCase):
    r"""
    Test the `emsc` function
    """

    def test_shape(self):
        r"""
        Check the shape of the return matrix
        """
        n_series, n_variables = (10, 50)
        # generate dummy data and background
        scaler = np.linspace(0, 10, num=n_variables)
        D = np.ones([n_series, n_variables]) * scaler[:, None].T
        background = 0.5 * D[0, :]
        background = background[:, None]
        background_list = [None, background]
        # iterate over different inputs
        for bg in background_list:
            D_pretreated, coefficients = cm.emsc(D, p_order=0, background=bg)
            self.assertTrue(D_pretreated.shape == (n_series, n_variables))
            self.assertTrue(coefficients.shape[0] == n_series)

    def test_background_subtraction(self):
        r"""
        Test wether background subtraction works
        """
        n_series, n_variables = (10, 50)
        # generate dummy data and background
        scaler = np.arange(n_variables)
        D = np.ones([n_series, n_variables]) * scaler[:, None].T
        background = 0.5 * D[0, :]
        background = background[:, None]
        D_pretreated, coefficients = cm.emsc(
            D,
            p_order=0,
            background=background
        )
        self.assertTrue(np.all(np.isclose(np.zeros([n_series, n_variables]),
                        D_pretreated)))


class TestGenerate_spectra(unittest.TestCase):
    r"""
    Test the `generate_spectra` function.
    """

    def test_shape(self):
        """
        Test if correct shape is generated
        """
        n_wl = 200
        expected_shape = (n_wl,)
        output_shape = cm.generate_spectra(n_wl, 2, 50).shape
        self.assertTrue(np.all(expected_shape == output_shape))

    def test_no_bands(self):
        """
        Test if ``n_band = 0`` returns zero vector.
        """
        n_wl = 10
        n_bands = 0
        bandwidth = 100
        spectra = cm.generate_spectra(n_wl, n_bands, bandwidth)
        isZero = np.all(np.isclose(np.zeros(n_wl), spectra))
        self.assertTrue(isZero)


class TestPlot_colored_spectra(unittest.TestCase):
    r"""
    Test the `plot_colored_spectra` function.
    """

    def test_return_arguments(self):
        """
        Test if the correct shape and object type is returend.
        """
        n_series, n_variables = (10, 50)
        # generate dummy data
        x = np.arange(0, n_variables)
        Y = np.ones([n_series, n_variables]).T * np.linspace(0, 10,
                                                             num=n_series)
        lines = cm.plot_colored_series(x, Y)
        self.assertTrue(type(lines) == list)
        self.assertEqual(n_series, len(lines))
        for line in lines:
            self.assertTrue(type(line) == matplotlib.lines.Line2D)


class TestPlot_svd(unittest.TestCase):
    r"""
    Test the `plot_svd` function.
    """

    def test_return_arguments(self):
        n_series, n_variables = (50, 100)
        n_comp = 3
        n_eigenvalues = 12
        mean = np.zeros(n_variables)
        x = np.arange(n_variables)[:, None]
        dist = x.T - x
        cov = np.exp(-(dist / n_variables * 5)**2)

        # draw a few samples from gaussian process as data
        D = np.random.multivariate_normal(mean, cov, size=n_series)

        # execute function with artificial data
        fig = cm.plot_svd(D, n_comp=n_comp, n_eigenvalues=n_eigenvalues)

        self.assertTrue(type(fig) is matplotlib.figure.Figure)
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].lines), n_comp)
        self.assertEqual(len(fig.axes[1].lines), n_eigenvalues)
        self.assertEqual(len(fig.axes[2].lines), n_comp)


if __name__ == '__main__':
    unittest.main()
