import numpy as np
import pytest

from ofdm.modulation import qpsk_demodulate, qpsk_modulate


def test_roundtrip():
    """Noiseless modulate then demodulate returns the original bits."""
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=1000)

    recovered = qpsk_demodulate(qpsk_modulate(bits))

    np.testing.assert_array_equal(recovered, bits)


def test_known_symbols():
    """Each bit pair maps to the symbol in the docstring."""
    bits = [0, 0, 0, 1, 1, 0, 1, 1]
    expected = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    np.testing.assert_allclose(qpsk_modulate(bits), expected)


def test_unit_energy():
    """Every symbol has |s|^2 = 1."""
    symbols = qpsk_modulate([0, 0, 0, 1, 1, 0, 1, 1])

    np.testing.assert_allclose(np.abs(symbols) ** 2, 1.0)


def test_odd_length_raises():
    """An odd number of bits is rejected."""
    with pytest.raises(ValueError):
        qpsk_modulate([0, 1, 0])


def test_non_binary_raises():
    """Values other than 0 and 1 are rejected."""
    with pytest.raises(ValueError):
        qpsk_modulate([0, 2])


def test_uint8_bits():
    """Unsigned integer bits do not wrap around."""
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    expected = qpsk_modulate([0, 1, 1, 0])

    np.testing.assert_allclose(qpsk_modulate(bits), expected)