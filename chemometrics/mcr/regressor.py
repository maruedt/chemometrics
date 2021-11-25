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

"""
Built-in least squares / regression methods.

All models will follow the formalism, AX = B, solve for X.

NOTE: coef_ will be X.T, which is the formalism that scikit-learn follows

"""

from ._regressor import (
    LinearRegression,
    OLS,
    NNLS
)

__all__ = ['LinearRegression', 'OLS', 'NNLS']
