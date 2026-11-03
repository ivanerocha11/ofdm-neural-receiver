"""Channel models."""

import numpy as np


def add_awgn(symbols, ebn0_db, bits_per_symbol, rng):
    """Add complex white Gaussian noise at a given Eb/N0.

    Assumes the transmitted samples have unit average energy, so
    Eb = 1 / bits_per_symbol. The real and imaginary parts of the noise
    are independent N(0, N0/2).

    Parameters
    ----------
    symbols : array_like of complex transmitted samples, either QPSK
        symbols or an OFDM time-domain stream.
    ebn0_db : float, Eb/N0 in decibels.
    bits_per_symbol : float, bits carried per transmitted sample, for
        example 2 for plain QPSK, or 2N / (N + P) for OFDM with a prefix.
    rng : numpy.random.Generator, the source of randomness.

    Returns
    -------
    r : complex ndarray, symbols plus noise, same shape as symbols.
    """
    symbols = np.asarray(symbols)

    gamma = 10 ** (ebn0_db / 10)
    eb = 1 / bits_per_symbol
    n0 = eb / gamma
    sigma = np.sqrt(n0 / 2)

    noise = sigma * (
        rng.standard_normal(symbols.shape)
        + 1j * rng.standard_normal(symbols.shape)
    )
    return symbols + noise