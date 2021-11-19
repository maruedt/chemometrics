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
#
# Additional information on the code history, original copyright holder and
# original license is available in ./license.md


"""
Test McrAR functionality
"""

import numpy as np

from numpy.testing import assert_equal

import unittest
from chemometrics.mcr import McrAR
import chemometrics.mcr.constraint as constraint
from chemometrics.mcr.regressor import OLS, NNLS
from sklearn.linear_model import Ridge


def dataset2():
    """ Setups dataset 2 components"""

    M = 21
    N = 21
    P = 101
    n_components = 2

    C_img = np.zeros((M, N, n_components))
    C_img[..., 0] = np.dot(np.ones((M, 1)), np.linspace(0, 1, N)[None, :])
    C_img[..., 1] = 1 - C_img[..., 0]

    St_known = np.zeros((n_components, P))
    St_known[0, 40:60] = 1
    St_known[1, 60:80] = 2

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    return C_known, D_known, St_known


def dataset3():
    """ Setups dataset 3 components"""
    M = 21
    N = 21
    P = 101
    n_components = 3

    C_img = np.zeros((M, N, n_components))
    C_img[..., 0] = np.dot(np.ones((M, 1)), np.linspace(0, 1, N)[None, :])
    C_img[..., 1] = np.dot(np.linspace(0, 1, M)[:, None], np.ones((1, N)))
    C_img[..., 2] = 1 - C_img[..., 0] - C_img[..., 1]
    C_img = C_img / C_img.sum(axis=-1)[:, :, None]

    St_known = np.zeros((n_components, P))
    St_known[0, 30:50] = 1
    St_known[1, 50:70] = 2
    St_known[2, 70:90] = 3
    St_known += 1

    C_known = C_img.reshape((-1, n_components))

    D_known = np.dot(C_known, St_known)

    return C_known, D_known, St_known


class TestMcrAR(unittest.TestCase):
    """
    Test McrAR
    """

    def setUp(self):
        self.dataset2 = dataset2()
        self.dataset3 = dataset3()

    def test_mcr_ideal_default(self):
        """ Provides C/St_known so optimal should be 1 iteration """

        C_known, D_known, St_known = self.dataset2

        mcrar = McrAR()
        mcrar.fit(D_known, ST=St_known)
        self.assertLess(mcrar.n_iter_opt, 3)
        self.assertTrue(((mcrar.D_ - D_known)**2).mean() < 1e-10)
        self.assertTrue(((mcrar.D_opt_ - D_known)**2).mean() < 1e-10)

        mcrar.fit(D_known, C=C_known)
        self.assertLess(mcrar.n_iter_opt, 3)
        self.assertTrue(((mcrar.D_ - D_known)**2).mean() < 1e-10)
        self.assertTrue(((mcrar.D_opt_ - D_known)**2).mean() < 1e-10)

    def test_mcr_ideal_str_regressors(self):
        """ Test MCR with string-provded regressors"""

        C_known, D_known, St_known = self.dataset2

        mcrar = McrAR(c_regr='OLS', st_regr='OLS')
        mcrar.fit(D_known, ST=St_known, verbose=True)
        self.assertEqual(1, mcrar.n_iter_opt)
        assert isinstance(mcrar.c_regressor, OLS)
        assert isinstance(mcrar.st_regressor, OLS)

        mcrar = McrAR(c_regr='NNLS', st_regr='NNLS')
        mcrar.fit(D_known, ST=St_known)
        self.assertEqual(1, mcrar.n_iter_opt)
        assert isinstance(mcrar.c_regressor, NNLS)
        assert isinstance(mcrar.st_regressor, NNLS)
        assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
        assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

        # Provided C_known this time
        mcrar = McrAR(c_regr='OLS', st_regr='OLS')
        mcrar.fit(D_known, C=C_known)

        # Turns out some systems get it in 1 iteration, some in 2
        # assert_equal(1, mcrar.n_iter_opt)
        self.assertEqual(True, mcrar.n_iter_opt <= 2)

        assert ((mcrar.D_ - D_known)**2).mean() < 1e-10
        assert ((mcrar.D_opt_ - D_known)**2).mean() < 1e-10

    def test_mcr_max_iterations(self):
        """ Test MCR exits at max_iter"""

        C_known, D_known, St_known = self.dataset2

        # Seeding with a constant of 0.1 for C, actually leads to a bad local
        # minimum; thus, the err_change gets really small with a relatively bad
        # error. The tol_err_change is set to None, so it makes it to max_iter.
        mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_increase=None, tol_n_increase=None,
                      tol_err_change=None, tol_n_above_min=None)
        mcrar.fit(D_known, C=C_known*0 + 0.1)
        assert mcrar.exit_max_iter_reached

    def test_mcr_tol_increase(self):
        """ Test MCR exits due error increasing above a tolerance fraction"""

        C_known, D_known, St_known = self.dataset2

        # Seeding with a constant of 0.1 for C, actually leads to a bad local
        # minimum; thus, the err_change gets really small with a relatively bad
        # error.
        mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_increase=0, tol_n_increase=None,
                      tol_err_change=None, tol_n_above_min=None)
        mcrar.fit(D_known, C=C_known*0 + 0.1)
        assert mcrar.exit_tol_increase

    def test_mcr_tol_n_increase(self):
        """
        Test MCR exits due iterating n times with an increase in error

        Note: On some CI systems, the minimum err bottoms out; thus,
        tol_n_above_min needed to be set to 0 to trigger a break.
        """

        C_known, D_known, St_known = self.dataset2

        mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_increase=None, tol_n_increase=0,
                      tol_err_change=None, tol_n_above_min=None)
        mcrar.fit(D_known, C=C_known*0 + 0.01)
        assert mcrar.exit_tol_n_increase

    def test_mcr_tol_err_change(self):
        """ Test MCR exits due error increasing by a value """

        C_known, D_known, St_known = self.dataset2

        mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_increase=None, tol_n_increase=None,
                      tol_err_change=1e-20, tol_n_above_min=None)
        mcrar.fit(D_known, C=C_known)
        assert mcrar.exit_tol_err_change

    def test_mcr_tol_n_above_min(self):
        """
        Test MCR exits due to half-terating n times with error above the
        minimum error.

        Note: On some CI systems, the minimum err bottoms out; thus,
        tol_n_above_min needed to be set to 0 to trigger a break.
        """

        C_known, D_known, St_known = self.dataset2

        mcrar = McrAR(max_iter=50, c_regr='OLS', st_regr='OLS',
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_increase=None, tol_n_increase=None,
                      tol_err_change=None, tol_n_above_min=0)
        mcrar.fit(D_known, C=C_known*0 + 0.1)
        assert mcrar.exit_tol_n_above_min

    def test_mcr_st_semilearned(self):
        """ Test when St items are fixed, i.e., enforced to be the same as the
        input, always """

        C_known, D_known, St_known = self.dataset3

        ST_guess = St_known.copy()
        ST_guess[2, :] = np.random.randn(St_known.shape[1])

        mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_err_change=1e-10)

        mcrar.fit(D_known, ST=ST_guess, st_fix=[0, 1])
        self.assertTrue(np.all(mcrar.ST_[0, :] == St_known[0, :]))
        self.assertTrue(np.all(mcrar.ST_[1, :] == St_known[1, :]))

    def test_mcr_c_semilearned(self):
        """ Test when C items are fixed, i.e., enforced to be the same as the
        input, always """

        C_known, D_known, St_known = self.dataset3

        # initialize concentration guess with true values
        C_guess = C_known.copy()
        # randomize third column
        C_guess[:, 2] = np.abs(np.random.randn(C_guess.shape[0])+0.1)

        mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_err_change=1e-10)

        mcrar.fit(D_known, C=C_guess, c_fix=[0, 1])
        # assert that column 1 and 2 of concentration remain unchanged
        assert_equal(mcrar.C_[:, 0], C_known[:, 0])
        assert_equal(mcrar.C_[:, 1], C_known[:, 1])

    def test_mcr_semilearned_both_c_st(self):
        """
        Test the special case when C & ST are provided, requiring C-fix ST-fix
        to be provided
        """
        C_known, D_known, St_known = self.dataset3

        C_guess = C_known.copy()
        # randomize third column
        C_guess[:, 2] = np.abs(np.random.randn(C_guess.shape[0])+0.1)

        mcrar = McrAR(max_iter=50, tol_increase=100, tol_n_increase=10,
                      st_constraints=[constraint.Nonneg()],
                      c_constraints=[constraint.Nonneg(),
                                     constraint.Normalizer()],
                      tol_err_change=1e-10)

        mcrar.fit(D_known, C=C_guess, ST=St_known,
                  c_fix=[0, 1], st_fix=[0], c_first=True)
        assert_equal(mcrar.C_[:, 0], C_known[:, 0])
        assert_equal(mcrar.C_[:, 1], C_known[:, 1])
        assert_equal(mcrar.ST_[0, :], St_known[0, :])

        # ST-solve first
        mcrar.fit(D_known, C=C_guess, ST=St_known, c_fix=[0, 1], st_fix=[0],
                  c_first=False)
        assert_equal(mcrar.C_[:, 0], C_known[:, 0])
        assert_equal(mcrar.C_[:, 1], C_known[:, 1])
        assert_equal(mcrar.ST_[0, :], St_known[0, :])

    def test_mcr_errors(self):

        # Providing both C and S^T estimates without C_fix and St_fix
        with self.assertRaises(TypeError):
            mcrar = McrAR()
            mcrar.fit(np.random.randn(10, 5), C=np.random.randn(10, 3),
                      ST=np.random.randn(3, 5))

        # Providing both C and S^T estimates without both C_fix and St_fix
        with self.assertRaises(TypeError):
            mcrar = McrAR()
            # Only c_fix
            mcrar.fit(np.random.randn(10, 5), C=np.random.randn(10, 3),
                      ST=np.random.randn(3, 5), c_fix=[0])

        with self.assertRaises(TypeError):
            mcrar = McrAR()
            # Only st_fix
            mcrar.fit(np.random.randn(10, 5), C=np.random.randn(10, 3),
                      ST=np.random.randn(3, 5), st_fix=[0])

        # Providing no estimates
        with self.assertRaises(TypeError):
            mcrar = McrAR()
            mcrar.fit(np.random.randn(10, 5))

        # Unknown regression method
        with self.assertRaises(ValueError):
            mcrar = McrAR(c_regr='NOTREAL')

        # regression object with no fit method
        with self.assertRaises(ValueError):
            mcrar = McrAR(c_regr=print)

    def test_props_features_samples_targets(self):
        """ Test mcrar properties for features, targets, samples """
        C_known, D_known, St_known = dataset2()

        mcrar = McrAR()
        mcrar.fit(D_known, ST=St_known)

        self.assertTrue(mcrar.n_targets == C_known.shape[-1])  # n_components
        self.assertTrue(mcrar.n_samples == D_known.shape[0])
        self.assertTrue(mcrar.n_features == D_known.shape[-1])

    def test_sklearn_linear_regression(self):
        """ Test MCR-AR works with sklearn-regressor """
        C_known, D_known, St_known = dataset2()

        mcrar = McrAR(c_regr=Ridge(), st_regr=Ridge())
        mcrar.fit(D_known, ST=St_known)

        # check that variables are set correctly w external regressor
        self.assertTrue(mcrar.n_targets == C_known.shape[-1])  # n_components
        self.assertTrue(mcrar.n_samples == D_known.shape[0])
        self.assertTrue(mcrar.n_features == D_known.shape[-1])

        # check that Ridge is used
        self.assertIsInstance(mcrar.c_regressor, Ridge)
        self.assertIsInstance(mcrar.st_regressor, Ridge)
        self.assertTrue(mcrar.c_regressor.n_features_in_ == C_known.shape[1])
        self.assertTrue(mcrar.st_regressor.n_features_in_ == C_known.shape[1])


if __name__ == '__main__':
    unittest.main()
