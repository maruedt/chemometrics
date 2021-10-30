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
Testing chemometrics.mcr.regressors

"""

import numpy as np

from chemometrics.mcr.metric import mse
import unittest


class TestMSE(unittest.TestCase):
    """
    Test mse calculation
    """

    def test_mse(self):
        A = np.ones((3, 3))
        B = np.eye(3)
        error = mse(None, None, A, B)
        self.assertEqual(error, 6/9)


if __name__ == '__main__':
    unittest.main()
