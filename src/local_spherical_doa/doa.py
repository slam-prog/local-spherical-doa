from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def direction_from_angles(
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    Convert azimuth/elevation to a unit vector.
    """
    azimuth = np.radians(
        azimuth_deg
    )

    elevation = np.radians(
        elevation_deg
    )

    direction = np.array(
        [
            np.cos(elevation)
            * np.cos(azimuth),
            np.cos(elevation)
            * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )

    return direction / (
        np.linalg.norm(direction)
        + 1e-12
    )


def parabolic_peak_offset(
    left: float,
    center: float,
    right: float,
) -> float:
    """
    Estimate fractional peak offset from three samples.
    """
    denominator = (
        left
        - 2.0 * center
        + right
    )

    if abs(denominator) < 1e-12:
        return 0.0

    offset = 0.5 * (
        (left - right)
        / denominator
    )

    return float(
        np.clip(
            offset,
            -1.0,
            1.0,
        )
    )


def _normalize(
    signal: np.ndarray,
) -> np.ndarray:
    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    signal = signal - np.mean(
        signal
    )

    norm = np.linalg.norm(
        signal
    )

    if norm <= 1e-12:
        return np.zeros_like(
            signal
        )

    return signal / norm


def gcc_phat_delay_samples(
    first: np.ndarray,
    second: np.ndarray,
    max_delay_samples: int,
) -> float:
    """
    Estimate delay of second relative to first.

    The sign correction matches fractional_delay().
    """
    first = _normalize(
        first
    )

    second = _normalize(
        second
    )

    n = len(first) + len(second) - 1

    fft_size = 1

    while fft_size < n:
        fft_size *= 2

    first_spectrum = np.fft.rfft(
        first,
        n=fft_size,
    )

    second_spectrum = np.fft.rfft(
        second,
        n=fft_size,
    )

    cross_spectrum = (
        first_spectrum
        * np.conj(
            second_spectrum
        )
    )

    cross_spectrum /= (
        np.abs(
            cross_spectrum
        )
        + 1e-12
    )

    correlation = np.fft.irfft(
        cross_spectrum,
        n=fft_size,
    )

    correlation = np.concatenate(
        [
            correlation[
                -(fft_size // 2):
            ],
            correlation[
                :fft_size // 2
            ],
        ]
    )

    lags = np.arange(
        -fft_size // 2,
        fft_size // 2,
        dtype=np.float64,
    )

    valid = (
        np.abs(lags)
        <= max_delay_samples
    )

    valid_indices = np.flatnonzero(
        valid
    )

    if valid_indices.size == 0:
        raise RuntimeError(
            "No valid GCC-PHAT lag"
        )

    peak_index = int(
        valid_indices[
            np.argmax(
                np.abs(
                    correlation[valid]
                )
            )
        ]
    )

    fractional_offset = 0.0

    if (
        peak_index > 0
        and peak_index
        < len(correlation) - 1
    ):
        fractional_offset = (
            parabolic_peak_offset(
                abs(
                    correlation[
                        peak_index - 1
                    ]
                ),
                abs(
                    correlation[
                        peak_index
                    ]
                ),
                abs(
                    correlation[
                        peak_index + 1
                    ]
                ),
            )
        )

    integer_lag = lags[
        peak_index
    ]

    return float(
        -(
            integer_lag
            + fractional_offset
        )
    )


def _maximum_delay_samples(
    microphone_positions_m: np.ndarray,
    fs: int,
    speed_of_sound_mps: float,
) -> int:
    microphone_positions_m = np.asarray(
        microphone_positions_m,
        dtype=np.float64,
    )

    differences = (
        microphone_positions_m[:, None, :]
        - microphone_positions_m[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    maximum_distance = float(
        np.max(distances)
    )

    return int(
        np.ceil(
            maximum_distance
            / speed_of_sound_mps
            * fs
        )
    ) + 2


def _delay_cost(
    angles_deg: np.ndarray,
    measured_delays_seconds: np.ndarray,
    microphone_positions_m: np.ndarray,
    speed_of_sound_mps: float,
) -> float:
    azimuth_deg = float(
        angles_deg[0] % 360.0
    )

    elevation_deg = float(
        np.clip(
            angles_deg[1],
            -90.0,
            90.0,
        )
    )

    direction = direction_from_angles(
        azimuth_deg,
        elevation_deg,
    )

    theoretical = -(
        microphone_positions_m @ direction
    ) / speed_of_sound_mps

    predicted = (
        theoretical[1:]
        - theoretical[0]
    )

    residual = (
        measured_delays_seconds
        - predicted
    )

    return float(
        np.mean(
            residual ** 2
        )
    )


def _grid_search(
    measured_delays_seconds: np.ndarray,
    microphone_positions_m: np.ndarray,
    azimuth_grid_deg: np.ndarray,
    elevation_grid_deg: np.ndarray,
    speed_of_sound_mps: float,
) -> tuple[float, float]:
    best_cost = np.inf
    best_azimuth = 0.0
    best_elevation = 0.0

    for elevation_deg in (
        elevation_grid_deg
    ):
        for azimuth_deg in (
            azimuth_grid_deg
        ):
            cost = _delay_cost(
                np.array(
                    [
                        azimuth_deg,
                        elevation_deg,
                    ],
                    dtype=np.float64,
                ),
                measured_delays_seconds,
                microphone_positions_m,
                speed_of_sound_mps,
            )

            if cost < best_cost:
                best_cost = cost
                best_azimuth = float(
                    azimuth_deg
                )
                best_elevation = float(
                    elevation_deg
                )

    return (
        best_azimuth,
        best_elevation,
    )


def estimate_local_doa(
    channels: np.ndarray,
    microphone_positions_m: np.ndarray,
    fs: int,
    azimuth_center_deg: float,
    azimuth_span_deg: float = 45.0,
    elevation_center_deg: float = 25.0,
    elevation_span_deg: float = 45.0,
    speed_of_sound_mps: float = 343.0,
    coarse_step_deg: float = 4.0,
    fine_step_deg: float = 0.5,
    fine_window_deg: float = 4.0,
    refine: bool = True,
) -> tuple[float, float]:
    """
    Estimate DOA inside a bounded local field of view.

    The default field of view is:
        azimuth span   = 45 degrees
        elevation span = 45 degrees
    """
    channels = np.asarray(
        channels,
        dtype=np.float64,
    )

    microphone_positions_m = np.asarray(
        microphone_positions_m,
        dtype=np.float64,
    )

    if channels.ndim != 2:
        raise ValueError(
            "channels must have shape "
            "(samples, microphones)"
        )

    if channels.shape[1] != (
        microphone_positions_m.shape[0]
    ):
        raise ValueError(
            "Channel count and microphone count "
            "must match"
        )

    max_delay_samples = (
        _maximum_delay_samples(
            microphone_positions_m,
            fs,
            speed_of_sound_mps,
        )
    )

    reference = channels[:, 0]

    measured_delays_seconds = []

    for microphone_index in range(
        1,
        channels.shape[1],
    ):
        delay_samples = (
            gcc_phat_delay_samples(
                reference,
                channels[:, microphone_index],
                max_delay_samples,
            )
        )

        measured_delays_seconds.append(
            delay_samples / fs
        )

    measured_delays_seconds = np.asarray(
        measured_delays_seconds,
        dtype=np.float64,
    )

    azimuth_start = (
        azimuth_center_deg
        - azimuth_span_deg / 2.0
    )

    azimuth_stop = (
        azimuth_center_deg
        + azimuth_span_deg / 2.0
    )

    elevation_start = max(
        -90.0,
        elevation_center_deg
        - elevation_span_deg / 2.0,
    )

    elevation_stop = min(
        90.0,
        elevation_center_deg
        + elevation_span_deg / 2.0,
    )

    coarse_azimuth_grid = (
        np.arange(
            azimuth_start,
            azimuth_stop
            + coarse_step_deg,
            coarse_step_deg,
        )
        % 360.0
    )

    coarse_elevation_grid = np.arange(
        elevation_start,
        elevation_stop
        + coarse_step_deg,
        coarse_step_deg,
    )

    coarse_azimuth, coarse_elevation = (
        _grid_search(
            measured_delays_seconds,
            microphone_positions_m,
            coarse_azimuth_grid,
            coarse_elevation_grid,
            speed_of_sound_mps,
        )
    )

    fine_azimuth_grid = (
        np.arange(
            coarse_azimuth
            - fine_window_deg,
            coarse_azimuth
            + fine_window_deg
            + fine_step_deg,
            fine_step_deg,
        )
        % 360.0
    )

    fine_elevation_grid = np.arange(
        max(
            -90.0,
            coarse_elevation
            - fine_window_deg,
        ),
        min(
            90.0,
            coarse_elevation
            + fine_window_deg,
        )
        + fine_step_deg,
        fine_step_deg,
    )

    fine_azimuth, fine_elevation = (
        _grid_search(
            measured_delays_seconds,
            microphone_positions_m,
            fine_azimuth_grid,
            fine_elevation_grid,
            speed_of_sound_mps,
        )
    )

    final_azimuth = fine_azimuth
    final_elevation = fine_elevation

    if refine:
        result = minimize(
            _delay_cost,
            x0=np.array(
                [
                    final_azimuth,
                    final_elevation,
                ],
                dtype=np.float64,
            ),
            args=(
                measured_delays_seconds,
                microphone_positions_m,
                speed_of_sound_mps,
            ),
            method="Nelder-Mead",
            options={
                "xatol": 1e-5,
                "fatol": 1e-20,
                "maxiter": 1000,
            },
        )

        refined_azimuth = float(
            result.x[0] % 360.0
        )

        refined_elevation = float(
            np.clip(
                result.x[1],
                elevation_start,
                elevation_stop,
            )
        )

        initial_cost = _delay_cost(
            np.array(
                [
                    final_azimuth,
                    final_elevation,
                ],
                dtype=np.float64,
            ),
            measured_delays_seconds,
            microphone_positions_m,
            speed_of_sound_mps,
        )

        refined_cost = _delay_cost(
            np.array(
                [
                    refined_azimuth,
                    refined_elevation,
                ],
                dtype=np.float64,
            ),
            measured_delays_seconds,
            microphone_positions_m,
            speed_of_sound_mps,
        )

        if refined_cost <= initial_cost:
            final_azimuth = refined_azimuth
            final_elevation = refined_elevation

    return (
        float(final_azimuth),
        float(final_elevation),
    )