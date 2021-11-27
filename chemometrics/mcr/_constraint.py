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
McrAR constraints

All classes need a transform method. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""


from abc import (ABC, abstractmethod)

import numpy as np

from sklearn.utils import check_array

__all__ = ['Constraint', 'Nonneg', 'CumsumNonneg', 'ZeroEndPoints',
           'ZeroCumSumEndPoints', 'Normalizer', 'CutBelow', 'CutAbove',
           'CompressBelow', 'CompressAbove', 'ReplaceZeros']


class Constraint(ABC):
    """ Abstract class for constraints

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, copy=True):
        self.copy = copy

    @abstractmethod
    def transform(self, A):
        """ Transform A input based on constraint """


class Nonneg(Constraint):
    """
    Non-negativity constraint. All negative entries made 0.

    The non-negativity constraint is typically used if a negative result would
    be unphysical or at least highly unlikely. Typical examples: absolute
    concentrations, absorption spectra.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, copy=False):
        """ A must be non-negative"""
        super().__init__(copy)

    def transform(self, A):
        """ Apply nonnegative constraint"""
        A = check_array(A, dtype="numeric", copy=self.copy)
        A *= (A > 0)
        return A


class CumsumNonneg(Constraint):
    """
    Cumulative-Summation non-negativity constraint

    The `CumSumNonneg` constraint enforces the cumulative sum over the data
    to always be positive. This is useful for processing first derivative
    data if for the non-derived data negative entries would be unphysical or at
    least highly unlikely. Typical example: first derivative of an absorption
    spectra

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, axis=1, copy=False):
        """ A must be non-negative"""
        super().__init__(copy)
        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        A = check_array(A, dtype="numeric", copy=self.copy)

        A *= (np.cumsum(A, self.axis) > 0)
        return A


class ZeroEndPoints(Constraint):
    """
    Enforce the endpoints to be zero

    The ZeroEndPoints constraint fits a linear baseline to the data and
    subtracts the baseline from the data. Each component is handled
    independently. If a span is given, first `span` points at each end of the
    data are averaged and only thereafter, the baseline is calculated and
    subtracted.
    The ZeroEndPoints constraint may be useful if we want to prevent MCR from
    modeling background contributions.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    axis : int
        Axis to operate on

    span : int
        Number of pixels along the ends to average.
    """

    def __init__(self, axis=-1, span=1, copy=False):
        """ A must be non-negative"""
        super().__init__(copy)
        if [0, 1, -1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis
        self.span = span

    def transform(self, A):
        """ Apply zero endpoints constraint"""

        A = check_array(A, dtype="numeric", copy=self.copy)

        # generate view with relevant axis in first dimension
        if (self.axis != 0):
            A = A.T

        pix_vec = np.arange(A.shape[0])
        delta_x = (A[-self.span:, :].mean(axis=0)
                   - A[:self.span, :].mean(axis=0))
        delta_y = (pix_vec[-self.span:].mean()
                   - pix_vec[:self.span].mean())
        slope = delta_x / delta_y
        intercept = (A[:self.span, :]
                     - np.dot(pix_vec[:self.span, None],
                              slope[None, :])).mean(axis=0)

        A -= (np.dot(pix_vec[:, None],
              slope[None, :]) + intercept[None, :])

        # restore original array format
        if (self.axis != 0):
            A = A.T
        return A


class ZeroCumSumEndPoints(Constraint):
    """
    Enforce certain points in cumulative sum to be near-zero.

    Reduces the cumsum baseline of data by piecewise subtraction of
    constant values between given nodes and the end points of the data. This
    approach is useful for first derivative data with localized peaks and
    no expected offset in the pure spectra. Each peak is delimited by a
    node.

    Parameters
    ----------
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    nodes : list of int
        In addition to end-points, other points to ensure are approximately 0
    axis : int
        Axis to operate on
    """

    def __init__(self, nodes=None, axis=-1, copy=False):
        """ A must be non-negative"""
        super().__init__(copy)

        self.nodes = nodes
        if [0, 1, -1].count(axis) != 1:
            raise TypeError('Axis must be 0, 1, or -1')

        self.axis = axis

    def transform(self, A):
        """ Apply cumsum nonnegative constraint"""
        A = check_array(A, dtype="numeric", copy=self.copy)

        # generate view with relevant axis in first dimension
        if (self.axis != 0):
            A = A.T

        if self.nodes:
            self.nodes = set(self.nodes)
        else:
            self.nodes = set()

        # generate a set at least including the endpoints of the data
        self.nodes.update({0, A.shape[0]})
        self.nodes.discard(-1)
        self.nodes = list(self.nodes)

        # subtract constant between nodes
        for num in range(len(self.nodes) - 1):
            n0 = self.nodes[num]
            n1 = self.nodes[num+1]
            A[n0:n1, :] -= A[n0:n1, :]\
                .mean(0)[None, :]

        # restore original array format
        if (self.axis != 0):
            A = A.T
        return A


class Normalizer(Constraint):
    """
    Normalization constraint.

    Normalizes the data along the selected axis. This means, that the
    transformed vector subsequently sums to one. If the magnitude of the
    factorized scores and loadings is not set by another constraint,
    Normalizer is necessary for a well-defined MCR problem.

    Parameters
    ----------
    axis : int
        Which axis of input matrix A to apply normalization across.
    fix : list
        Keep fix-axes as-is and normalize the remaining axes based on the
        residual of the fixed axes.
    set_zeros_to_feature : int
        Set all samples which sum-to-zero across axis to 1 for a particular
         feature (See Notes)
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    Notes
    -----

    -   For set_zeros_to_feature, assuming the data represents concentration
         with a matrix [n_samples, n_features] and the axis is across the
         features, for every sample that sums to 0 across axis, would be
         replaced with a vector [n_features] of zeros except at
         set_zeros_to_feature, which would equal 1. I.e., this pixel is
         now pure substance of index value set_zeros_to_feature.


    """

    def __init__(self, axis=-1, fix=None, copy=False):
        """Normalize along axis"""
        super().__init__(copy)
        if fix is None:
            self.fix = fix
        elif isinstance(fix, int):
            self.fix = [fix]
        elif isinstance(fix, (list, tuple)):
            self.fix = fix
        elif isinstance(fix, np.ndarray):
            if np.issubdtype(fix.dtype, np.integer):
                self.fix = fix.tolist()
            else:
                raise TypeError('fix ndarray must be of dtype int')
        else:
            raise TypeError('Parameter fix must be of type None, int, list,',
                            'tuple, ndarray')

        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply normalization constraint """
        A = check_array(A, dtype=float, copy=self.copy)

        # generate view with relevant axis in first dimension
        if (self.axis != 0):
            A = A.T

        if not self.fix:  # No fixed axes
            A /= A.sum(axis=0)[None, :]
        else:  # Fixed axes
            not_fix_locs = [v for v in np.arange(A.shape[0]).tolist()
                            if self.fix.count(v) == 0]
            scaler = np.ones(A.shape)
            div = A[not_fix_locs, :].sum(axis=0)[None, :]
            div[div == 0] = 1
            scaler[not_fix_locs, :] = (
                (1 - A[self.fix, :].sum(axis=0)[None, :]) / div
            )
            A *= scaler

        # restore original array format
        if (self.axis != 0):
            A = A.T

        return A


class ReplaceZeros(Constraint):
    """
    Samples that sum-to-zero across axis are replaced with a vector of 0's
    except for a 1 at feature if a single value. In a concentration context,
    e.g., samples with no concentration are replaced with 100% concentration of
    a set feature. If multiple features given, equal amounts of each feature
    (summing to 1) are used.

    Parameters
    ----------
    axis : int
        Which axis of input matrix A to apply normalization acorss.
    feature : int, list, tuple
        Set all samples which sum-to-zero across axis to fval for a particular
         feature (or fractional) for multiple features.
    fval : float
        Value of summation across axis of replacement vector.
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)

    """

    def __init__(self, axis=-1, feature=None, fval=1, copy=False):
        """Replace sum-to-zero samples with new feature vector along axis"""
        super().__init__(copy)
        self.fval = fval
        if feature is None:
            self.feature = feature
        elif isinstance(feature, int):
            self.feature = [feature]
        elif isinstance(feature, (list, tuple)):
            self.feature = feature
        elif isinstance(feature, np.ndarray):
            if np.issubdtype(feature.dtype, np.integer):
                self.feature = feature.tolist()
            else:
                raise TypeError('ndarrays must be of dtype int')
        else:
            raise TypeError('Parameter feature must be of type None, int,'
                            + 'list, tuple, ndarray')

        if not ((axis == 0) | (axis == 1) | (axis == -1)):
            raise ValueError('Axis must be 0,1, or -1')
        self.axis = axis

    def transform(self, A):
        """ Apply constraint """
        A = check_array(A, dtype=float, copy=self.copy)

        # generate view with relevant axis in first dimension
        if (self.axis != 0):
            A = A.T

        if self.feature:
            replacement = np.zeros(A.shape[self.axis])
            replacement[self.feature] = self.fval
            replacement /= replacement.sum()
            replacement *= self.fval

            A[:, A.sum(axis=0) == 0] = replacement[:, None]

        # restore original array format
        if (self.axis != 0):
            A = A.T
        return A


class _CutExclude(Constraint):
    """
    Parent class for methods that cut and can exclude

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : {None, int}
        If not None, cut below value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    exclude : int, list, tuple, ndarray
        Exclude rows/columns along axis defined by `exclude_axis`. If an index
        is listed in `exclude`, all entries along given `exclude_axis` are not
        transformed.
    exclude_axis : int
        Along which axis entries are excluded. See also `exclude`
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, axis_sumnz=None, exclude=None,
                 exclude_axis=-1, copy=False):
        """ """
        super().__init__(copy)
        self.value = value
        self.axis = axis_sumnz
        self.exclude = exclude
        self.exclude_axis = exclude_axis

        self._excl_mat = None

    def _make_excl_mat(self, A_shape):
        """generate mask which excludes certain entries"""
        X, Y = np.meshgrid(np.arange(A_shape[1]), np.arange(A_shape[0]))
        if self.exclude is None:
            # no exclusions
            self._excl_mat = np.zeros(X.shape, dtype=bool)
        else:
            # select axis along which to exclude
            if self.exclude_axis == 0:
                # exclude all entries along dimension based on index
                self._excl_mat = np.isin(Y, self.exclude)
            else:
                self._excl_mat = np.isin(X, self.exclude)


class CutBelow(_CutExclude):
    """
    Cut values below (and not-equal to) a certain threshold.

    The values of an array are set to zero, if they are below a certain
    threshold. This implies that the values may afterwards be further from the
    threshold than before. A typical example where this functionality may be
    interesting is, if peaks should be localized and the interaction between
    different components minimized.
    Additionally, this constraint provides functionalities to exclude sections
    of the data from being processed.

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : {None, int}
        If not None, cut below value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    exclude : int, list, tuple, ndarray
        Exclude rows/columns along axis defined by `exclude_axis`. If an index
        is listed in `exclude`, all entries along given `exclude_axis` are not
        transformed.
    exclude_axis : int
        Along which axis entries are excluded. See also `exclude`
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, axis_sumnz=None, exclude=None, exclude_axis=-1,
                 copy=False):
        """ Initialize """
        super().__init__(value=value, axis_sumnz=axis_sumnz, exclude=exclude,
                         exclude_axis=exclude_axis, copy=copy)

    def transform(self, A):
        """ Apply cut-below value constraint"""
        A = check_array(A, dtype=float, copy=self.copy)

        if self._excl_mat is None:
            self._make_excl_mat(A.shape)

        # change behavior depending if axis_sumnz was defined or not
        if self.axis is None:
            A *= ((A >= self.value) | self._excl_mat)
        else:
            A *= (np.alltrue(A < self.value, axis=self.axis, keepdims=True)
                  + (A >= self.value) + self._excl_mat)
        return A


class CutAbove(_CutExclude):
    """
    Cut values above (and not-equal to) a certain threshold

    The values of an array are set to zero, if they are above a certain
    threshold. This implies that the values may afterwards be further from the
    threshold than before. This constraint provides functionalities to exclude
    sections of the data from being processed.

    Parameters
    ----------

    value : float
        Cutoff value
    axis_sumnz : {None, int}
        If not None, cut below value only applied where sum across specified
        axis does not go to 0, i.e. all values cut.
    exclude : int, list, tuple, ndarray
        Exclude rows/columns along axis defined by `exclude_axis`. If an index
        is listed in `exclude`, all entries along given `exclude_axis` are not
        transformed.
    exclude_axis : int
        Along which axis entries are excluded. See also `exclude`
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, axis_sumnz=None, exclude=None, exclude_axis=-1,
                 copy=False):
        """ """
        super().__init__(value=value, axis_sumnz=axis_sumnz, exclude=exclude,
                         exclude_axis=exclude_axis, copy=copy)

    def transform(self, A):
        """ Apply cut-above value constraint"""
        A = check_array(A, dtype=float, copy=self.copy)

        if self._excl_mat is None:
            self._make_excl_mat(A.shape)

        # change behavior depending if axis_sumnz was defined or not
        if self.axis is None:
            A *= ((A <= self.value) | self._excl_mat)
        else:
            A *= (np.alltrue(A > self.value, axis=self.axis, keepdims=True)
                  + (A <= self.value) + self._excl_mat)
        return A


class CompressBelow(Constraint):
    """
    Compress values below (and not-equal to) a certain threshold

    The constraint sets all values which are smaller than the threshold value
    to the threshold value.

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        super().__init__(copy)
        self.value = value

    def transform(self, A):
        """ Apply compress-below value constraint"""
        A = check_array(A, dtype=float, copy=self.copy)

        temp = self.value*(A < self.value)
        A *= (A >= self.value)
        A += temp
        return A


class CompressAbove(Constraint):
    """
    Compress values above (and not-equal to) a certain threshold (set to value)

    The constraint sets all values which are larger than the threshold value
    to the threshold value.

    Parameters
    ----------

    value : float
        Cutoff value
    copy : bool
        Make copy of input data, A; otherwise, overwrite (if mutable)
    """

    def __init__(self, value=0, copy=False):
        """  """
        super().__init__(copy)
        self.value = value

    def transform(self, A):
        """ Apply compress-above value constraint"""
        A = check_array(A, dtype=float, copy=self.copy)

        temp = self.value*(A > self.value)
        A *= (A <= self.value)
        A += temp
        return A
