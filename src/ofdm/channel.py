"""Channel models."""

import numpy as np


def add_awgn(symbols, ebn0_db, bits_per_symbol, rng):
    """Add complex white Gaussian noise at a given Eb/N0.

    Assumes unit average symbol energy, Es = 1, so Eb = 1 / bits_per_symbol.
    The real and imaginary parts of the noise are independent N(0, N0/2).

    Parameters
    ----------
    symbols : array_like of complex transmitted symbols.
    ebn0_db : float, Eb/N0 in decibels.
    bits_per_symbol : int, for example 2 for QPSK.
    rng : numpy.random.Generator, the source of randomness.

    Returns
    -------
    r : complex ndarray, symbols plus noise, same shape as symbols.
    """
    gamma = 10 ** (ebn0_db / 10)
    eb = 1 / bits_per_symbol
    n0 = eb / gamma
    sigma = np.sqrt(n0 / 2)
    symbols = np.asarray(symbols)
    noise = sigma * (rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape))
    return symbols + noise