import numpy as np
import pytest

from ofdm.modulation import qpsk_modulate
from ofdm.ofdm import (
    add_cyclic_prefix,
    ofdm_demodulate,
    ofdm_modulate,
    remove_cyclic_prefix,
)

N = 64
P = 16


def random_symbols(num_blocks, rng):
    """Random QPSK symbols filling num_blocks blocks of N."""
    bits = rng.integers(0, 2, size=2 * N * num_blocks)
    return qpsk_modulate(bits)


def test_roundtrip_noiseless():
    """Demodulating the transmitted stream returns the original symbols."""
    rng = np.random.default_rng(0)
    symbols = random_symbols(10, rng)

    recovered = ofdm_demodulate(ofdm_modulate(symbols, N, P), N, P)

    np.testing.assert_allclose(recovered, symbols)


def test_output_length():
    """Each block of N symbols becomes N + P samples."""
    rng = np.random.default_rng(1)

    x = ofdm_modulate(random_symbols(3, rng), N, P)

    assert x.size == 3 * (N + P)


def test_prefix_is_tail():
    """The first P samples of each block equal its last P samples."""
    rng = np.random.default_rng(2)

    x = ofdm_modulate(random_symbols(5, rng), N, P).reshape(-1, N + P)

    np.testing.assert_allclose(x[:, :P], x[:, -P:])


def test_energy_preserved():
    """Without the prefix, each block has the same energy as its symbols."""
    rng = np.random.default_rng(3)
    symbols = random_symbols(5, rng)

    x = ofdm_modulate(symbols, N, P).reshape(-1, N + P)
    body = remove_cyclic_prefix(x, P)

    np.testing.assert_allclose(
        np.sum(np.abs(body) ** 2, axis=-1),
        np.sum(np.abs(symbols.reshape(-1, N)) ** 2, axis=-1),
    )


@pytest.mark.parametrize("prefix_len", [0, 1, 16, 64])
def test_remove_undoes_add(prefix_len):
    """Removing the prefix recovers the original blocks, including P = 0."""
    blocks = np.arange(2 * N).reshape(2, N)

    restored = remove_cyclic_prefix(add_cyclic_prefix(blocks, prefix_len), prefix_len)

    np.testing.assert_array_equal(restored, blocks)


def test_bad_lengths_raise():
    """Streams that do not divide into whole blocks are rejected."""
    with pytest.raises(ValueError):
        ofdm_modulate(np.ones(N + 1), N, P)
    with pytest.raises(ValueError):
        ofdm_demodulate(np.ones(N + P + 1), N, P)


def test_bad_prefix_raises():
    """Prefix lengths outside 0 <= P <= N are rejected."""
    with pytest.raises(ValueError):
        add_cyclic_prefix(np.ones(N), N + 1)
    with pytest.raises(ValueError):
        add_cyclic_prefix(np.ones(N), -1)