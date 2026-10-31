import numpy as np
import pytest

from ofdm.metrics import bit_error_rate, q_function, qpsk_ber_theory


def test_ber_identical_is_zero():
    """Identical bit arrays have no errors."""
    bits = [0, 1, 1, 0]

    assert bit_error_rate(bits, bits) == 0.0


def test_ber_all_flipped_is_one():
    """Fully flipped bits are all errors."""
    assert bit_error_rate([0, 1, 1, 0], [1, 0, 0, 1]) == 1.0


def test_ber_one_of_four():
    """One wrong bit out of four gives 0.25."""
    assert bit_error_rate([0, 1, 1, 0], [0, 1, 1, 1]) == 0.25


def test_ber_mismatched_shapes_raise():
    """Arrays of different lengths are rejected."""
    with pytest.raises(ValueError):
        bit_error_rate([0, 1], [0, 1, 1])


def test_q_function_at_zero():
    """Half of a standard normal lies above zero."""
    assert q_function(0) == pytest.approx(0.5)


def test_qpsk_theory_at_0_db():
    """At 0 dB, the QPSK bit error rate is Q(sqrt(2))."""
    assert qpsk_ber_theory(0) == pytest.approx(0.0786, abs=1e-4)


def test_qpsk_theory_decreases():
    """More signal means fewer errors, and lists are accepted."""
    ber = qpsk_ber_theory([0, 5, 10])

    assert ber.shape == (3,)
    assert np.all(np.diff(ber) < 0)