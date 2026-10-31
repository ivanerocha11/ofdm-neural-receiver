"""Performance metrics and theoretical reference curves."""

import numpy as np
from scipy.special import erfc


def q_function(x):
    """Gaussian tail probability Q(x) = P(Z > x) for Z ~ N(0, 1)."""
    return 0.5 * erfc(x / np.sqrt(2))


def bit_error_rate(bits_tx, bits_rx):
    """Fraction of positions where bits_tx and bits_rx differ.

    Parameters
    ----------
    bits_tx, bits_rx : array_like of 0s and 1s, with the same length.

    Returns
    -------
    ber : float in [0, 1].
    """
    bits_tx = np.asarray(bits_tx)
    bits_rx = np.asarray(bits_rx)
    if bits_tx.shape != bits_rx.shape:
        raise ValueError("bits_tx and bits_rx must have the same shape")
    return float(np.mean(bits_tx != bits_rx))


def qpsk_ber_theory(ebn0_db):
    """Theoretical QPSK bit error rate in AWGN, Q(sqrt(2 Eb/N0)).

    Parameters
    ----------
    ebn0_db : float or array_like, Eb/N0 in decibels.

    Returns
    -------
    ber : float or ndarray, same shape as ebn0_db.
    """
    gamma = 10 ** (np.asarray(ebn0_db) / 10)
    return q_function(np.sqrt(2 * gamma))