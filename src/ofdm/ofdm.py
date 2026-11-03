"""OFDM modulation and demodulation with a cyclic prefix."""

import numpy as np


def add_cyclic_prefix(blocks, P):
    """Prepend the last P samples of each block to that block.

    Parameters
    ----------
    blocks : array_like, shape (..., N). Each row along the last axis is a block.
    P : int, prefix length, with 0 <= P <= N.

    Returns
    -------
    ndarray, shape (..., N + P).
    """
    blocks = np.asarray(blocks)
    N = blocks.shape[-1]
    if not 0 <= P <= N:
        raise ValueError("P must satisfy 0 <= P <= N")
    return np.concatenate([blocks[..., N - P:], blocks], axis=-1)


def remove_cyclic_prefix(blocks, P):
    """Discard the first P samples of each block.

    Parameters
    ----------
    blocks : array_like, shape (..., N + P).
    P : int, prefix length.

    Returns
    -------
    ndarray, shape (..., N).
    """
    blocks = np.asarray(blocks)
    if not 0 <= P <= blocks.shape[-1]:
        raise ValueError("P must satisfy 0 <= P <= block length")
    return blocks[..., P:]


def ofdm_modulate(symbols, N, P):
    """Map symbols onto N frequency bins per block and produce a time signal.

    The symbols are split into consecutive blocks of N. Each block is
    transformed with a unitary inverse DFT, given a cyclic prefix of length
    P, and the blocks are concatenated into one stream.

    Parameters
    ----------
    symbols : array_like of complex symbols, length a multiple of N.
    N : int, number of frequency bins per block.
    P : int, cyclic prefix length.

    Returns
    -------
    x : complex ndarray, length (len(symbols) // N) * (N + P).
    """
    symbols = np.asarray(symbols)
    if symbols.size % N != 0:
        raise ValueError("number of symbols must be a multiple of N")

    blocks = symbols.reshape(-1, N)
    time_blocks = np.fft.ifft(blocks, axis=-1, norm="ortho")
    return add_cyclic_prefix(time_blocks, P).ravel()


def ofdm_demodulate(r, N, P):
    """Recover the frequency-bin values from a received OFDM stream.

    The stream is split into blocks of N + P samples. Each block has its
    prefix removed and is transformed with a unitary DFT.

    Parameters
    ----------
    r : array_like of complex received samples, length a multiple of N + P.
    N : int, number of frequency bins per block.
    P : int, cyclic prefix length.

    Returns
    -------
    Y : complex ndarray, length (len(r) // (N + P)) * N, one value per bin
        per block, in the same order that ofdm_modulate takes symbols.
    """
    r = np.asarray(r)
    if r.size % (N + P) != 0:
        raise ValueError("number of samples must be a multiple of N + P")

    time_blocks = remove_cyclic_prefix(r.reshape(-1, N + P), P)
    return np.fft.fft(time_blocks, axis=-1, norm="ortho").ravel()