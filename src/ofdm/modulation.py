"""QPSK modulation and demodulation with Gray labeling."""

import numpy as np


def qpsk_modulate(bits):
    """Map pairs of bits to unit-energy QPSK symbols.

    The first bit of each pair sets the sign of the real part 
    and the second sets the sign of the imaginary part, with 0 -> + and 1 -> -.
    So 00 -> (1 + 1j)/sqrt(2), 01 -> (1 - 1j)/sqrt(2),
    10 -> (-1 + 1j)/sqrt(2), 11 -> (-1 - 1j)/sqrt(2).

    Parameters
    ----------
    bits : array_like of 0s and 1s, with even length.

    Returns
    -------
    symbols : complex ndarray of length len(bits) // 2.
    """
    bits = np.asarray(bits)
    if bits.size % 2 != 0:
        raise ValueError("number of bits must be even for QPSK")
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("bits must be 0 or 1")

    pairs = bits.astype(int).reshape(-1, 2)
    i = 1 - 2 * pairs[:, 0]
    q = 1 - 2 * pairs[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demodulate(r):
    """Recover bits from received QPSK values by quadrant.

    Each received value is decided independently: the first bit is 1 if
    the real part is negative, and the second bit is 1 if the imaginary
    part is negative.

    Parameters
    ----------
    r : array_like of complex received values.

    Returns
    -------
    bits : integer ndarray of 0s and 1s, of length 2 * len(r), in the
        same order that qpsk_modulate expects.
    """
    r = np.asarray(r)
    bits = np.empty(2 * r.size, dtype=int)
    bits[0::2] = r.real < 0
    bits[1::2] = r.imag < 0
    return bits