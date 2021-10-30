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
Testing mcr.constraints

"""

import numpy as np

from numpy.testing import assert_allclose

import chemometrics.mcr.constraint as constraint
import unittest


class TestConstraint(unittest.TestCase):
    def test_nonneg(self):
        A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]])
        A_nn = np.array([[1, 2, 3], [0, 0, 0], [1, 2, 3]])

        constr_nn = constraint.Nonneg(copy=True)
        out = constr_nn.transform(A)
        assert_allclose(A_nn, out)

        constr_nn = constraint.Nonneg(copy=False)
        out = constr_nn.transform(A)
        assert_allclose(A_nn, A)

    def test_cumsumnonneg(self):
        """ Cum-Sum Nonnegativity Constraint """
        A = np.array([[2, -2, 3, -2], [-1, -2, -3, 7], [1, -2, -3, 7]])
        A_nn_ax1 = np.array([[2, 0, 3, -2], [0, 0, 0, 7], [1, 0, 0, 7]])
        A_nn_ax0 = np.array([[2, 0, 3, 0], [-1, 0, 0, 7], [1, 0, 0, 7]])

        # Axis -1
        constr_nn = constraint.CumsumNonneg(copy=True, axis=-1)
        out = constr_nn.transform(A)
        assert_allclose(A_nn_ax1, out)

        # Axis 0
        constr_nn = constraint.CumsumNonneg(copy=False, axis=0)
        out = constr_nn.transform(A)
        assert_allclose(A_nn_ax0, A)

    def test_zeroendpoints(self):
        """ 0-Endpoints Constraint """
        A = np.array([[1, 2, 3, 4],
                      [3, 6, 9, 12],
                      [4, 8, 12, 16]]).astype(float)
        A_ax1 = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]).astype(float)
        A_ax0 = np.array([[0, 0, 0, 0],
                          [0.5, 1, 1.5, 2],
                          [0, 0, 0, 0]]).astype(float)

        # Axis 0
        constr_ax0 = constraint.ZeroEndPoints(copy=True, axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(A_ax0, out)

        # Axis -1
        constr_ax1 = constraint.ZeroEndPoints(copy=True, axis=-1)
        out = constr_ax1.transform(A)
        assert_allclose(A_ax1, out)

        with self.assertRaises(TypeError):
            constr_ax1 = constraint.ZeroEndPoints(copy=True, axis=3)

        # Axis 0 -- NOT copies
        constr_ax0 = constraint.ZeroEndPoints(copy=False, axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(A_ax0, A)

    def test_zeroendpoints_span(self):
        """ 0-Endpoints Constraint """
        A = np.array([[1, 2, 3, 4],
                      [3, 6, 9, 12],
                      [4, 8, 12, 16]]).astype(float)

        # Axis 1
        constr_ax1 = constraint.ZeroEndPoints(copy=True, axis=1, span=2)
        out = constr_ax1.transform(A)
        assert_allclose(out[:, [0, 1]].mean(axis=1), 0)
        assert_allclose(out[:, [1, 2]].mean(axis=1), 0)

        # Axis 0
        constr_ax0 = constraint.ZeroEndPoints(copy=True, axis=0, span=2)
        out = constr_ax0.transform(A)
        assert_allclose(out[[0, 1], :].mean(axis=0), 0)
        assert_allclose(out[[1, 2], :].mean(axis=0), 0)

        # effective an assert_not_equal
        assert_allclose([q != 0 for q in out[:, 0]], True)
        assert_allclose([q != 0 for q in out[:, -1]], True)

        # Axis 1 -- no copy
        constr_ax1 = constraint.ZeroEndPoints(copy=False, axis=1, span=2)
        out = constr_ax1.transform(A)
        assert_allclose(A[:, [0, 1]].mean(axis=1), 0)
        assert_allclose(A[:, [1, 2]].mean(axis=1), 0)

    def test_zerocumsumendpoints(self):
        """ Cum-Sum 0-Endpoints Constraint """
        A_diff1 = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]]
                           ).astype(float)
        A_diff0 = np.array([[3, 3, 3, 3],
                            [3, 3, 3, 3],
                            [3, 3, 3, 3]]).astype(float)

        # A_ax1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # A_ax0 = np.array([[0, 0, 0], [0.5, 1, 1.5], [0, 0, 0]])

        # Axis 0
        constr_ax0 = constraint.ZeroCumSumEndPoints(copy=True, axis=0)
        out = constr_ax0.transform(A_diff0)
        assert_allclose(out, 0)
        assert_allclose(np.cumsum(out, axis=0), 0)

        # Axis -1
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=True, axis=-1)
        out = constr_ax1.transform(A_diff1)
        assert_allclose(out, 0)
        assert_allclose(np.cumsum(out, axis=1), 0)

        # Axis = -1 -- NOT copy
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=False, axis=-1)
        out = constr_ax1.transform(A_diff1)
        assert_allclose(A_diff1, 0)
        assert_allclose(np.cumsum(A_diff1, axis=1), 0)

    def test_zerocumsumendpoints_nodes(self):
        """ Cum-Sum 0-Endpoints Constraint with defined nodes"""

        A = np.array([[1, 2, 1, 12],
                      [1, 2, 2, 3],
                      [1, 2, 3, 6]]).astype(float)
        A_transform_ax0 = np.array([[0, 0, -1, 5],
                                    [0, 0, 0, -4],
                                    [0, 0, 1, -1]])
        A_transform_ax1 = np.array([[-3, -2, -3, 8],
                                    [-1, 0, 0, 1],
                                    [-2, -1, 0, 3]])

        # COPY Axis 0, node=0 (same as endpoints)
        constr_ax0 = constraint.ZeroCumSumEndPoints(copy=True, nodes=[0],
                                                    axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(out, A_transform_ax0)
        # assert_allclose(np.cumsum(out, axis=0), 0)

        # COPY Axis -1, node=0 (same as endpoints)
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=True, nodes=[0],
                                                    axis=-1)
        out = constr_ax1.transform(A)
        assert_allclose(out, A_transform_ax1)
        # assert_allclose(np.cumsum(out, axis=1), 0)

        # OVERWRITE, Axis = 0, node=0 (same as endpoints)
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax0 = constraint.ZeroCumSumEndPoints(copy=False, nodes=[0],
                                                    axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(A, A_transform_ax0)

        # OVERWRITE, Axis = 1, node=0 (same as endpoints)
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=False, nodes=[0],
                                                    axis=-1)
        out = constr_ax1.transform(A)
        assert_allclose(A, A_transform_ax1)

        # COPY, Axis = 0, Nodes = all
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax0 = constraint.ZeroCumSumEndPoints(copy=True, nodes=[0, 1, 2],
                                                    axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(out, 0)

        # COPY, Axis = 1, Nodes = all
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=True,
                                                    nodes=[0, 1, 2, 3], axis=1)
        out = constr_ax1.transform(A)
        assert_allclose(out, 0)

        # OVERWRITE, Axis = 0, Nodes = all
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax0 = constraint.ZeroCumSumEndPoints(copy=False,
                                                    nodes=[0, 1, 2], axis=0)
        out = constr_ax0.transform(A)
        assert_allclose(A, 0)

        # OVERWRITE, Axis = 1, Nodes = all
        A = np.array([[1, 2, 1, 12], [1, 2, 2, 3], [1, 2, 3, 6]]).astype(float)
        constr_ax1 = constraint.ZeroCumSumEndPoints(copy=False,
                                                    nodes=[0, 1, 2, 3], axis=1)
        out = constr_ax1.transform(A)
        assert_allclose(A, 0)

    def test_norm(self):
        """ Test normalization """
        # A must be dtype.float for in-place math (copy=False)
        constr_norm = constraint.Norm(axis=0, copy=False)
        A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]])  # dtype: int32
        with self.assertRaises(TypeError):
            out = constr_norm.transform(A)

        # Axis must be 0,1, or -1
        with self.assertRaises(ValueError):
            constr_norm = constraint.Norm(axis=2, copy=False)

        A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]], dtype=float)
        A_norm0 = A / A.sum(axis=0)[None, :]
        A_norm1 = A / A.sum(axis=1)[:, None]

        constr_norm = constraint.Norm(axis=0, copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_norm0, out)

        constr_norm = constraint.Norm(axis=1, copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_norm1, out)

        constr_norm = constraint.Norm(axis=-1, copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_norm1, out)

        constr_norm = constraint.Norm(axis=0, copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_norm0, A)

        A = np.array([[1, 2, 3], [-1, -2, -3], [1, 2, 3]], dtype=float)
        constr_norm = constraint.Norm(axis=1, copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_norm1, A)

    def test_norm_fixed_axes(self):
        # AXIS = 1
        A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0]], dtype=float)
        A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                               [0.1, 0.3, 0.6, 0.0]], dtype=float)

        # Fixed axes must be integers
        with self.assertRaises(TypeError):
            constr_norm = constraint.Norm(axis=1, fix=2.2, copy=True)

        # Dtype must be integer related
        with self.assertRaises(TypeError):
            constr_norm = constraint.Norm(axis=1, fix=np.array([2.2]),
                                          copy=True)

        # COPY: True
        # Fix of type int
        constr_norm = constraint.Norm(axis=1, fix=2, copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type list
        constr_norm = constraint.Norm(axis=1, fix=[2, 3], copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type tuple
        constr_norm = constraint.Norm(axis=1, fix=(2), copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type ndarray
        constr_norm = constraint.Norm(axis=1, fix=np.array([2]), copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # COPY: False
        A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                               [0.1, 0.3, 0.6, 0.0]], dtype=float)

        # Fix of type int
        A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0]], dtype=float)
        constr_norm = constraint.Norm(axis=1, fix=2, copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type list
        A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0]], dtype=float)
        constr_norm = constraint.Norm(axis=1, fix=[2, 3], copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type tuple
        A = np.array([[0.0, 0.2, 1.0, 0.0], [0.25, 0.25, 0.0, 0.0],
                     [0.3, 0.9, 0.6, 0.0]], dtype=float)
        constr_norm = constraint.Norm(axis=1, fix=(2), copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type ndarray
        A = np.array([[0.0, 0.2, 1.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0]], dtype=float)
        constr_norm = constraint.Norm(axis=1, fix=np.array([2]), copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # AXIS = 0
        # Lazy, so just transposed
        A = np.array([[0.0, 0.2, 1.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0]], dtype=float).T
        A_fix2_ax1 = np.array([[0.0, 0.0, 1.0, 0.0],
                               [0.5, 0.5, 0.0, 0.0],
                               [0.1, 0.3, 0.6, 0.0]], dtype=float).T
        # COPY: True
        # Fix of type int
        constr_norm = constraint.Norm(axis=0, fix=2, copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type list
        constr_norm = constraint.Norm(axis=0, fix=[2, 3], copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type tuple
        constr_norm = constraint.Norm(axis=0, fix=(2), copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # Fix of type ndarray
        constr_norm = constraint.Norm(axis=0, fix=np.array([2]), copy=True)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, out)

        # COPY: False
        A_fix2_ax1 = np.array([[0.0, 0.0, 1.0],
                               [0.5, 0.5, 0.0],
                               [0.1, 0.3, 0.6]], dtype=float).T

        # Fix of type int
        A = np.array([[0.0, 0.2, 1.0],
                      [0.25, 0.25, 0.0],
                      [0.3, 0.9, 0.6]], dtype=float).T
        constr_norm = constraint.Norm(axis=0, fix=2, copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type list
        A = np.array([[0.0, 0.2, 1.0],
                      [0.25, 0.25, 0.0],
                      [0.3, 0.9, 0.6]], dtype=float).T
        constr_norm = constraint.Norm(axis=0, fix=[2], copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type tuple
        A = np.array([[0.0, 0.2, 1.0],
                      [0.25, 0.25, 0.0],
                      [0.3, 0.9, 0.6]], dtype=float).T
        constr_norm = constraint.Norm(axis=0, fix=(2), copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

        # Fix of type ndarray
        A = np.array([[0.0, 0.2, 1.0],
                      [0.25, 0.25, 0.0],
                      [0.3, 0.9, 0.6]], dtype=float).T
        constr_norm = constraint.Norm(axis=0, fix=np.array([2]), copy=False)
        out = constr_norm.transform(A)
        assert_allclose(A_fix2_ax1, A)

    def test_cut_below(self):
        """ Test cutting below (and not equal to) a value """
        A = np.array([[1, 2, 3, 4],
                      [4, 5, 6, 7],
                      [7, 8, 9, 10]]).astype(float)
        A_transform = np.array([[0, 0, 0, 4],
                                [4, 5, 6, 7],
                                [7, 8, 9, 10]]).astype(float)

        constr = constraint.CutBelow(copy=True, value=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform)

        # No Copy
        constr = constraint.CutBelow(copy=False, value=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform)

    def test_cut_below_exclude(self):
        """ Test cutting below (and not equal to) a value """
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        A_transform_excl_0_ax1 = np.array([[1, 0, 0, 4],
                                           [4, 5, 6, 7],
                                           [7, 8, 9, 10]]).astype(float)
        A_transform_excl_0_ax0 = np.array([[1, 2, 3, 4],
                                           [4, 5, 6, 7],
                                           [7, 8, 9, 10]]).astype(float)

        # COPY
        constr = constraint.CutBelow(copy=True, value=4, exclude=0,
                                     exclude_axis=1)
        out = constr.transform(A)
        assert_allclose(out, A_transform_excl_0_ax1)

        constr = constraint.CutBelow(copy=True, value=4, exclude=0,
                                     exclude_axis=0)
        out = constr.transform(A)
        assert_allclose(out, A_transform_excl_0_ax0)

        # OVERWRITE
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        constr = constraint.CutBelow(copy=False, value=4, exclude=0,
                                     exclude_axis=1)
        out = constr.transform(A)
        assert_allclose(A, A_transform_excl_0_ax1)

        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        constr = constraint.CutBelow(copy=False, value=4, exclude=0,
                                     exclude_axis=0)
        out = constr.transform(A)
        assert_allclose(A, A_transform_excl_0_ax0)

    def test_cut_below_nonzerosum(self):
        """
        Test cutting below (and not equal to) a value with the constrain that
            no columns (axis=-1) will all become 0
        """
        A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        cutoff = 0.7
        A_correct = np.array([[0.0, 0.7], [0.4, 0.6], [0.5, 0.5]])

        constr = constraint.CutBelow(copy=True, value=cutoff, axis_sumnz=-1)
        out = constr.transform(A)
        assert_allclose(out, A_correct)

        constr = constraint.CutBelow(copy=False, value=cutoff, axis_sumnz=-1)
        constr.transform(A)
        assert_allclose(A, A_correct)

    def test_cut_below_nonzerosum_exclude(self):
        """
        Test cutting below (and not equal to) a value with the constrain that
            no columns (axis=-1) will all become 0
        """
        A = np.array([[0.3, 0.7], [0.3, 0.7], [0.7, 0.3]])
        cutoff = 0.7
        A_correct = np.array([[0.3, 0.7], [0.3, 0.7], [0.7, 0.0]])

        # COPY
        constr = constraint.CutBelow(copy=True, value=cutoff, axis_sumnz=-1,
                                     exclude=0, exclude_axis=-1)
        out = constr.transform(A)
        assert_allclose(out, A_correct)

        # OVERWRITE
        constr = constraint.CutBelow(copy=False, value=cutoff, axis_sumnz=-1,
                                     exclude=0, exclude_axis=-1)
        _ = constr.transform(A)
        assert_allclose(A, A_correct)

        # constr = constraint.CutBelow(copy=False, value=cutoff, axis_sumnz=-1)
        # constr.transform(A)
        # assert_allclose(A, A_correct)

    def test_cut_above_nonzerosum(self):
        """
        Test cutting above (and not equal to) a value with the constrain that
            no columns (axis=-1) will all become 0
        """
        A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        cutoff = 0.4
        A_correct = np.array([[0.3, 0.0], [0.4, 0.0], [0.5, 0.5]])

        constr = constraint.CutAbove(copy=True, value=cutoff, axis_sumnz=-1)
        out = constr.transform(A)
        assert_allclose(out, A_correct)

        # NO Copy
        A = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        cutoff = 0.4
        A_correct = np.array([[0.3, 0.0], [0.4, 0.0], [0.5, 0.5]])

        constr = constraint.CutAbove(copy=False, value=cutoff, axis_sumnz=-1)
        constr.transform(A)
        assert_allclose(A, A_correct)

    def test_compress_below(self):
        """ Test compressing below (and not equal to) a value """
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        A_transform = np.array([[4, 4, 4, 4],
                                [4, 5, 6, 7],
                                [7, 8, 9, 10]]).astype(float)

        constr = constraint.CompressBelow(copy=True, value=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform)

        # No Copy
        constr = constraint.CompressBelow(copy=False, value=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform)

    def test_cut_above(test_mcr_ideal_default):
        """ Test cutting above (and not equal to) a value """
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        A_transform = np.array([[1, 2, 3, 4],
                                [4, 0, 0, 0],
                                [0, 0, 0, 0]]).astype(float)

        constr = constraint.CutAbove(copy=True, value=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform)

        # No Copy
        constr = constraint.CutAbove(copy=False, value=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform)

    def test_cut_above_exclude(self):
        """ Test cutting above (and not equal to) a value """
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        A_transform_excl_0_ax1 = np.array([[1, 2, 3, 4],
                                           [4, 0, 0, 0],
                                           [7, 0, 0, 0]]).astype(float)
        A_transform_excl_2_ax0 = np.array([[1, 2, 3, 4],
                                           [4, 0, 0, 0],
                                           [7, 8, 9, 10]]).astype(float)

        constr = constraint.CutAbove(copy=True, value=4, exclude=0,
                                     exclude_axis=-1)
        out = constr.transform(A)
        assert_allclose(out, A_transform_excl_0_ax1)

        constr = constraint.CutAbove(copy=True, value=4, exclude=2,
                                     exclude_axis=0)
        out = constr.transform(A)
        assert_allclose(out, A_transform_excl_2_ax0)

        # No Copy
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        constr = constraint.CutAbove(copy=False, value=4, exclude=0,
                                     exclude_axis=-1)
        _ = constr.transform(A)
        assert_allclose(A, A_transform_excl_0_ax1)

        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        constr = constraint.CutAbove(copy=False, value=4, exclude=2,
                                     exclude_axis=0)
        _ = constr.transform(A)
        assert_allclose(A, A_transform_excl_2_ax0)

    def test_compress_above(self):
        """ Test compressing above (and not equal to) a value """
        A = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]).astype(float)
        A_transform = np.array([[1, 2, 3, 4],
                                [4, 4, 4, 4],
                                [4, 4, 4, 4]]).astype(float)

        constr = constraint.CompressAbove(copy=True, value=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform)

        # No Copy
        constr = constraint.CompressAbove(copy=False, value=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform)

    def test_replace_zeros(self):
        """ Test replace zeros using a single feature """
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=0, feature = 0
        A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 1.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=1, feature = 0
        A_transform_ax1 = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [1.0, 0.0, 0.0, 0.0]], dtype=float)

        # Axis 0, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=0, feature=0)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax0)

        # Axis 1, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=1, feature=0)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax1)

        # Axis 0, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=0, feature=0)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax0)

        # Axis 1, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=1, feature=0)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax1)

    def test_replace_zeros_multifeature(self):
        """ Replace zeros using multiple features """

        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=0, feature = [0,1]
        A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 0.5],
                                    [0.25, 0.25, 0.0, 0.5],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=1, feature = [0,1]
        A_transform_ax1 = np.array([[0.5, 0.5, 0.0, 0.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [0.5, 0.5, 0.0, 0.0]], dtype=float)

        # Axis 0, copy=True, feature=[0,1]
        constr = constraint.ReplaceZeros(copy=True, axis=0, feature=[0, 1])
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax0)

        # Axis 1, copy=True, feature=[0,1]
        constr = constraint.ReplaceZeros(copy=True, axis=1, feature=[0, 1])
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax1)

        # Axis 1, copy=True, feature=np.array([0,1])
        constr = constraint.ReplaceZeros(copy=True, axis=1,
                                         feature=np.array([0, 1]))
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax1)

        # Axis 1, copy=True, feature=None
        constr = constraint.ReplaceZeros(copy=True, axis=1, feature=None)
        out = constr.transform(A)
        assert_allclose(out, A)

        # Axis 0, copy=FALSE, feature=[0,1]
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=0, feature=[0, 1])
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax0)

        # Axis 1, copy=FALSE, feature=[0,1]
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=1, feature=[0, 1])
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax1)

    def test_replace_zeros_non1fval(self):
        """ Test replace zeros using a single feature with fval != 1 """
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=0, feature = 0
        A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 4.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=1, feature = 0
        A_transform_ax1 = np.array([[4.0, 0.0, 0.0, 0.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [4.0, 0.0, 0.0, 0.0]], dtype=float)

        # Axis 0, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=0, feature=0, fval=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax0)

        # Axis 1, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=1, feature=0, fval=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax1)

        # Axis 0, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=0, feature=0, fval=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax0)

        # Axis 1, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=1, feature=0, fval=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax1)

    def test_replace_zeros_non1fval_multifeature(self):
        """ Test replace zeros using a 2 features with fval != 1 """
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=0, feature = 0
        A_transform_ax0 = np.array([[0.0, 0.0, 0.0, 2.0],
                                    [0.25, 0.25, 0.0, 2.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]], dtype=float)
        # Axis=1, feature = 0
        A_transform_ax1 = np.array([[2.0, 2.0, 0.0, 0.0],
                                    [0.25, 0.25, 0.0, 0.0],
                                    [0.3, 0.9, 0.6, 0.0],
                                    [2.0, 2.0, 0.0, 0.0]], dtype=float)

        # Axis 0, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=0, feature=[0, 1],
                                         fval=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax0)

        # Axis 1, copy=True
        constr = constraint.ReplaceZeros(copy=True, axis=1, feature=[0, 1],
                                         fval=4)
        out = constr.transform(A)
        assert_allclose(out, A_transform_ax1)

        # Axis 0, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=0, feature=[0, 1],
                                         fval=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax0)

        # Axis 1, copy=FALSE
        A = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.25, 0.25, 0.0, 0.0],
                      [0.3, 0.9, 0.6, 0.0],
                      [0.0, 0.0, 0.0, 0.0]], dtype=float)

        constr = constraint.ReplaceZeros(copy=False, axis=1, feature=[0, 1],
                                         fval=4)
        out = constr.transform(A)
        assert_allclose(A, A_transform_ax1)

    def test_planarize_no_noise(self):
        """ Test constraint.Planarize with no noise """
        C_img = np.zeros((10, 20, 2))  # Y, X, Target
        x = np.arange(C_img.shape[1])
        y = np.arange(C_img.shape[0])
        n_targets = C_img.shape[-1]

        X, Y = np.meshgrid(x, y)

        C_0_ideal = 0.1*X + 0.3*Y + 3
        C_img[:, :, 0] = C_0_ideal
        C_img[:, :, 1] = 0.2*X + 0.4*Y + 3

        C_ravel = C_img.reshape((-1, n_targets))

        # COPY
        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=True)
        out = constr.transform(C_ravel)
        out_img = out.reshape(C_img.shape)

        assert_allclose(out_img[..., 1], C_img[..., 1])
        # Data is a plane. Should return the same plane to within numerical
        # error
        assert_allclose(out_img[..., 0], C_0_ideal)

        # OVERWRITE
        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=False)
        _ = constr.transform(C_ravel)
        assert_allclose(C_ravel.reshape(C_img.shape)[..., 0], C_img[..., 0])
        assert_allclose(C_ravel.reshape(C_img.shape)[..., 1], C_img[..., 1])

    def test_planarize_noisy(self):
        """ Test constraint.Planarize """
        C_img = np.zeros((10, 20, 2))  # Y, X, Target
        x = np.arange(C_img.shape[1])
        y = np.arange(C_img.shape[0])
        n_targets = C_img.shape[-1]

        X, Y = np.meshgrid(x, y)

        C_img[:, :, 0] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10, 20)
        C_img[:, :, 1] = 0.1*X + 0.3*Y + 3

        C_ravel = C_img.reshape((-1, n_targets))

        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=True)
        out = constr.transform(C_ravel)
        out_img = out.reshape(C_img.shape)

        assert_allclose(out_img[..., 1], C_img[..., 1])

        self.assertTrue(
            (np.sum((out_img[..., 0] - C_img[..., 1])**2))
            < (np.sum((C_img[..., 0] - C_img[..., 1])**2))
        )

    def test_planarize_noisy_list_target(self):
        """ Test constraint.Planarize """
        C_img = np.zeros((10, 20, 2))  # Y, X, Target
        x = np.arange(C_img.shape[1])
        y = np.arange(C_img.shape[0])
        n_targets = C_img.shape[-1]

        X, Y = np.meshgrid(x, y)

        C_img[:, :, 0] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10, 20)
        C_img[:, :, 1] = 0.1*X + 0.3*Y + 3 + 0.1*np.random.randn(10, 20)
        C_ideal_01 = 0.1*X + 0.3*Y + 3

        C_ravel = C_img.reshape((-1, n_targets))

        constr = constraint.Planarize([0, 1], (10, 20), scaler=None, copy=True)
        out = constr.transform(C_ravel)
        out_img = out.reshape(C_img.shape)

        # Test that RSS is better after planarize than before
        self.assertTrue(
            np.sum((C_img[..., 0] - C_ideal_01)**2)
            > np.sum((out_img[..., 0] - C_ideal_01)**2)
        )
        self.assertTrue(
            np.sum((C_img[..., 1] - C_ideal_01)**2)
            > np.sum((out_img[..., 1] - C_ideal_01)**2)
        )

        # Two output targets are not identical
        self.assertTrue(np.sum((out_img[..., 0] - out_img[..., 1])**2) > 0)

    def test_planarize_err_type_input(self):
        """ Inputting a target that is not a list, tuple, or ndarray results in
        Type Error """
        with self.assertRaises(TypeError):
            _ = constraint.Planarize({'a': 0}, (10, 20), scaler=None,
                                     copy=True)

    def test_planarize_set_scaler(self):
        """ Test setting or not setting scaler """

        C_img = np.zeros((10, 20, 2))  # Y, X, Target
        x = np.arange(C_img.shape[1])
        y = np.arange(C_img.shape[0])
        n_targets = C_img.shape[-1]

        X, Y = np.meshgrid(x, y)

        C_img[:, :, 0] = 0.1*X + 0.3*Y - 2.5
        C_img[:, :, 1] = 0.1*X + 0.3*Y - 2.5

        C_ravel = C_img.reshape((-1, n_targets))

        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=True)
        _ = constr.transform(C_ravel)

        self.assertTrue(constr.scaler is not None)
        self.assertTrue(constr.scaler > 100)

        constr = constraint.Planarize(0, (10, 20), scaler=1.0, copy=True)
        _ = constr.transform(C_ravel)

        self.assertTrue(constr.scaler is not None)
        self.assertTrue(constr.scaler == 1.0)

    def test_planarize_use_above_and_below_on_plane(self):
        """ Test constraint.Planarize """
        C_img = np.zeros((10, 20, 2))  # Y, X, Target
        x = np.arange(C_img.shape[1])
        y = np.arange(C_img.shape[0])
        n_targets = C_img.shape[-1]

        X, Y = np.meshgrid(x, y)

        C_img[:, :, 0] = 0.1*X + 0.3*Y - 2.5
        C_img[:, :, 1] = 0.1*X + 0.3*Y - 2.5

        C_ravel = C_img.reshape((-1, n_targets))

        # COPY -- DO NOT Apply limits to plane
        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=True,
                                      use_vals_above=0,
                                      use_vals_below=1, lims_to_plane=False)
        out = constr.transform(C_ravel)

        # reality check on constructed data
        self.assertTrue(C_ravel.min() < 0)
        self.assertTrue(C_ravel.max() > 1)

        # check scaling of output data
        self.assertTrue(out[:, 0].min() < 0)
        self.assertTrue(out[:, 0].max() > 1)
        self.assertTrue(out[:, 1].min() < 0)
        self.assertTrue(out[:, 1].max() > 1)

        # COPY -- Apply limits to plane
        constr = constraint.Planarize(0, (10, 20), scaler=None, copy=True,
                                      use_vals_above=0,
                                      use_vals_below=1, lims_to_plane=True)
        out = constr.transform(C_ravel)

        # check scaling if limits to plane apply
        self.assertTrue(out[:, 0].min() >= 0)
        self.assertTrue(out[:, 0].max() <= 1)
        self.assertTrue(out[:, 1].min() < 0)
        self.assertTrue(out[:, 1].max() > 1)


if __name__ == '__main__':
    unittest.main()
