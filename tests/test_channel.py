import numpy as np
import pytest

from ofdm.channel import add_awgn


@pytest.mark.parametrize("ebn0_db", [0, 5, 10])
def test_noise_power(ebn0_db):
    """Average noise energy matches N0 = Eb / (Eb/N0)."""
    rng = np.random.default_rng(0)
    n0 = 0.5 / 10 ** (ebn0_db / 10)

    r = add_awgn(np.zeros(100_000), ebn0_db, bits_per_symbol=2, rng=rng)

    np.testing.assert_allclose(np.mean(np.abs(r) ** 2), n0, rtol=0.02)


def test_real_and_imaginary_split_evenly():
    """Real and imaginary parts each carry variance N0 / 2."""
    rng = np.random.default_rng(1)
    n0 = 0.5 / 10 ** (10 / 10)

    r = add_awgn(np.zeros(100_000), 10, bits_per_symbol=2, rng=rng)

    np.testing.assert_allclose(np.var(r.real), n0 / 2, rtol=0.02)
    np.testing.assert_allclose(np.var(r.imag), n0 / 2, rtol=0.02)


def test_shape_preserved():
    """Output has the same shape as the input."""
    rng = np.random.default_rng(2)
    symbols = np.ones((4, 8), dtype=complex)

    r = add_awgn(symbols, 10, bits_per_symbol=2, rng=rng)

    assert r.shape == symbols.shape


def test_reproducible_with_same_seed():
    """The same seed gives the same noise."""
    symbols = np.zeros(10)

    r1 = add_awgn(symbols, 10, 2, np.random.default_rng(3))
    r2 = add_awgn(symbols, 10, 2, np.random.default_rng(3))

    np.testing.assert_array_equal(r1, r2)