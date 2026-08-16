import numpy as np

from local_spherical_doa.signals import (
    add_awgn,
    fractional_delay,
    speech_like_signal,
)


def test_speech_like_signal_length():
    rng = np.random.default_rng(1)

    signal = speech_like_signal(
        duration_seconds=0.01,
        fs=48000,
        rng=rng,
    )

    assert signal.shape == (480,)


def test_fractional_delay_shape():
    signal = np.zeros(1024)
    signal[100] = 1.0

    delayed = fractional_delay(
        signal,
        delay_samples=2.5,
    )

    assert delayed.shape == signal.shape
    assert np.all(np.isfinite(delayed))


def test_awgn_without_noise():
    rng = np.random.default_rng(1)
    signal = np.ones(100)

    output = add_awgn(
        signal,
        snr_db=None,
        rng=rng,
    )

    assert np.array_equal(
        output,
        signal,
    )