from __future__ import annotations

import numpy as np


def speech_like_signal(
    duration_seconds: float,
    fs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a stochastic broadband speech-like signal.
    """
    if duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be positive"
        )

    if fs <= 0:
        raise ValueError(
            "fs must be positive"
        )

    length = int(
        round(duration_seconds * fs)
    )

    output = np.zeros(
        length,
        dtype=np.float64,
    )

    position = 0

    while position < length:
        remaining = length - position

        requested_length = int(
            rng.uniform(
                0.02,
                0.10,
            ) * fs
        )

        segment_length = min(
            max(
                64,
                requested_length,
            ),
            remaining,
        )

        t = np.arange(
            segment_length,
            dtype=np.float64,
        ) / fs

        f0 = rng.uniform(
            90.0,
            240.0,
        )

        segment = np.zeros(
            segment_length,
            dtype=np.float64,
        )

        for harmonic in range(1, 12):
            segment += (
                rng.uniform(
                    0.03,
                    1.0,
                )
                * np.sin(
                    2.0
                    * np.pi
                    * harmonic
                    * f0
                    * t
                    + rng.uniform(
                        0.0,
                        2.0 * np.pi,
                    )
                )
            )

        segment += 0.04 * rng.normal(
            size=segment_length
        )

        fade_length = min(
            segment_length // 2,
            max(
                1,
                int(0.01 * fs),
            ),
        )

        envelope = np.ones(
            segment_length,
            dtype=np.float64,
        )

        if fade_length > 0:
            envelope[:fade_length] = (
                np.linspace(
                    0.0,
                    1.0,
                    fade_length,
                )
            )

            envelope[-fade_length:] = (
                np.linspace(
                    1.0,
                    0.0,
                    fade_length,
                )
            )

        segment *= envelope

        output[
            position:position + segment_length
        ] = segment

        position += segment_length

    output -= np.mean(output)

    peak = float(
        np.max(
            np.abs(output)
        )
    )

    if peak > 1e-12:
        output /= peak

    return output


def fractional_delay(
    signal: np.ndarray,
    delay_samples: float,
) -> np.ndarray:
    """
    Apply a fractional delay using FFT phase rotation.
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    if signal.size == 0:
        return signal.copy()

    n = len(signal)

    spectrum = np.fft.rfft(
        signal
    )

    frequencies = np.fft.rfftfreq(
        n
    )

    shifted = (
        spectrum
        * np.exp(
            -2j
            * np.pi
            * frequencies
            * delay_samples
        )
    )

    return np.fft.irfft(
        shifted,
        n=n,
    )


def add_awgn(
    signal: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add white Gaussian noise at a given SNR.
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if snr_db is None:
        return signal.copy()

    signal_power = float(
        np.mean(
            signal ** 2
        )
    )

    if signal_power <= 1e-20:
        return signal.copy()

    snr_linear = 10.0 ** (
        snr_db / 10.0
    )

    noise_power = (
        signal_power
        / snr_linear
    )

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=signal.shape,
    )

    return signal + noise