from __future__ import annotations

import numpy as np

from .signals import (
    add_awgn,
    fractional_delay,
)


def simulate_spherical_array(
    source_signal: np.ndarray,
    delays_seconds: np.ndarray,
    fs: int,
    rng: np.random.Generator,
    snr_db: float | None = None,
) -> np.ndarray:
    """
    Simulate channels for one spherical microphone array.

    Returns:
        shape (samples, microphones)
    """
    source_signal = np.asarray(
        source_signal,
        dtype=np.float64,
    ).reshape(-1)

    delays_seconds = np.asarray(
        delays_seconds,
        dtype=np.float64,
    ).reshape(-1)

    if source_signal.size == 0:
        raise ValueError(
            "source_signal must not be empty"
        )

    if delays_seconds.size == 0:
        raise ValueError(
            "delays_seconds must not be empty"
        )

    channels = []

    for delay_seconds in delays_seconds:
        channel = fractional_delay(
            source_signal,
            delay_seconds * fs,
        )

        channel = add_awgn(
            channel,
            snr_db,
            rng,
        )

        channels.append(channel)

    return np.column_stack(
        channels
    )