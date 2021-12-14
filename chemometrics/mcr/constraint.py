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
McrAR constraints

All classes need a transform method. Note, unlike sklearn, transform can copy
or overwrite input depending on copy attribute.
"""

from ._constraint import (
    Constraint,
    Nonneg,
    CumsumNonneg,
    ZeroEndPoints,
    ZeroCumSumEndPoints,
    Normalizer,
    CutBelow,
    CutAbove,
    CompressBelow,
    CompressAbove,
    ReplaceZeros,
    Unimodal,
)

__all__ = ['Constraint', 'Nonneg', 'CumsumNonneg', 'ZeroEndPoints',
           'ZeroCumSumEndPoints', 'Normalizer', 'CutBelow', 'CutAbove',
           'CompressBelow', 'CompressAbove', 'ReplaceZeros', 'Unimodal']
