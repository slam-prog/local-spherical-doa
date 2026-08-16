from __future__ import annotations

import sys

import numpy as np


sys.path.insert(
    0,
    "src",
)


from local_spherical_doa.arrays import (
    simulate_spherical_array,
)
from local_spherical_doa.doa import (
    estimate_local_doa,
)
from local_spherical_doa.geometry import (
    direction_from_angles,
    far_field_delays_seconds,
    fibonacci_sphere,
)
from local_spherical_doa.signals import (
    speech_like_signal,
)


def angular_error_deg(
    first_direction: np.ndarray,
    second_direction: np.ndarray,
) -> float:
    first_direction = np.asarray(
        first_direction,
        dtype=np.float64,
    )

    second_direction = np.asarray(
        second_direction,
        dtype=np.float64,
    )

    first_direction /= (
        np.linalg.norm(
            first_direction
        )
        + 1e-12
    )

    second_direction /= (
        np.linalg.norm(
            second_direction
        )
        + 1e-12
    )

    cosine = np.clip(
        np.dot(
            first_direction,
            second_direction,
        ),
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def main():
    rng = np.random.default_rng(
        20260816
    )

    fs = 192000
    duration_seconds = 0.10
    microphone_count = 8
    radius_m = 0.025
    snr_db = 30.0
    speed_of_sound_mps = 343.0

    azimuth_center_deg = 125.0
    azimuth_span_deg = 45.0

    elevation_center_deg = 25.0
    elevation_span_deg = 45.0

    true_azimuth_deg = 125.0
    true_elevation_deg = 25.0

    microphone_positions = (
        fibonacci_sphere(
            microphone_count,
            radius_m,
        )
    )

    true_direction = direction_from_angles(
        true_azimuth_deg,
        true_elevation_deg,
    )

    source = speech_like_signal(
        duration_seconds,
        fs,
        rng,
    )

    delays_seconds = (
        far_field_delays_seconds(
            microphone_positions,
            true_direction,
            speed_of_sound_mps,
        )
    )

    channels = simulate_spherical_array(
        source_signal=source,
        delays_seconds=delays_seconds,
        fs=fs,
        rng=rng,
        snr_db=snr_db,
    )

    estimated_azimuth_deg, estimated_elevation_deg = (
        estimate_local_doa(
            channels=channels,
            microphone_positions_m=(
                microphone_positions
            ),
            fs=fs,
            azimuth_center_deg=(
                azimuth_center_deg
            ),
            azimuth_span_deg=(
                azimuth_span_deg
            ),
            elevation_center_deg=(
                elevation_center_deg
            ),
            elevation_span_deg=(
                elevation_span_deg
            ),
            speed_of_sound_mps=(
                speed_of_sound_mps
            ),
            coarse_step_deg=4.0,
            fine_step_deg=0.5,
            fine_window_deg=4.0,
            refine=True,
        )
    )

    estimated_direction = direction_from_angles(
        estimated_azimuth_deg,
        estimated_elevation_deg,
    )

    error = angular_error_deg(
        true_direction,
        estimated_direction,
    )

    print(
        {
            "true_azimuth_deg": (
                true_azimuth_deg
            ),
            "true_elevation_deg": (
                true_elevation_deg
            ),
            "estimated_azimuth_deg": (
                estimated_azimuth_deg
            ),
            "estimated_elevation_deg": (
                estimated_elevation_deg
            ),
            "angular_error_deg": error,
            "field_of_view": {
                "azimuth_center_deg": (
                    azimuth_center_deg
                ),
                "azimuth_span_deg": (
                    azimuth_span_deg
                ),
                "elevation_center_deg": (
                    elevation_center_deg
                ),
                "elevation_span_deg": (
                    elevation_span_deg
                ),
            },
            "microphone_count": (
                microphone_count
            ),
            "radius_m": radius_m,
            "sample_rate_hz": fs,
            "snr_db": snr_db,
            "warning": (
                "Simulation results only; "
                "not hardware measurements."
            ),
        }
    )


if __name__ == "__main__":
    main()