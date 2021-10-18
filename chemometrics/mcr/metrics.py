"""
Metrics used in :mod:`chemometrics.mcr`

All functions must take C, ST, D_actual, D_calculated
"""


def mse(C, ST, D_actual, D_calculated):
    """ Mean square error """
    return ((D_actual - D_calculated)**2).sum()/D_actual.size
