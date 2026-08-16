from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local 45-degree "
            "spherical-array DOA estimation."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260816,
    )

    parser.add_argument(
        "--fs",
        type=int,
        default=192000,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--microphones",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--radius-mm",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--snr-db",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--azimuth-center",
        type=float,
        default=125.0,
    )

    parser.add_argument(
        "--azimuth-span",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--elevation-center",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--elevation-span",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "benchmark_local_spherical_doa.json"
        ),
    )

    return parser.parse_args()


def angular_error_deg(
    true_direction: np.ndarray,
    estimated_direction: np.ndarray,
) -> float:
    true_direction = np.asarray(
        true_direction,
        dtype=np.float64,
    )

    estimated_direction = np.asarray(
        estimated_direction,
        dtype=np.float64,
    )

    true_direction /= (
        np.linalg.norm(
            true_direction
        )
        + 1e-12
    )

    estimated_direction /= (
        np.linalg.norm(
            estimated_direction
        )
        + 1e-12
    )

    cosine = np.clip(
        np.dot(
            true_direction,
            estimated_direction,
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


def random_direction_in_sector(
    rng: np.random.Generator,
    azimuth_center_deg: float,
    azimuth_span_deg: float,
    elevation_center_deg: float,
    elevation_span_deg: float,
):
    """
    Generate a random direction inside the
    configured local field of view.
    """
    azimuth_deg = float(
        rng.uniform(
            azimuth_center_deg
            - azimuth_span_deg / 2.0,
            azimuth_center_deg
            + azimuth_span_deg / 2.0,
        )
    )

    elevation_low = max(
        -90.0,
        elevation_center_deg
        - elevation_span_deg / 2.0,
    )

    elevation_high = min(
        90.0,
        elevation_center_deg
        + elevation_span_deg / 2.0,
    )

    elevation_deg = float(
        rng.uniform(
            elevation_low,
            elevation_high,
        )
    )

    direction = direction_from_angles(
        azimuth_deg,
        elevation_deg,
    )

    return (
        azimuth_deg,
        elevation_deg,
        direction,
    )


def summarize(
    values: list[float],
) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return {
            "count": 0,
            "mean_error_deg": None,
            "median_error_deg": None,
            "p90_error_deg": None,
            "p95_error_deg": None,
            "max_error_deg": None,
            "within_0_5deg": None,
            "within_1deg": None,
            "within_3deg": None,
            "within_5deg": None,
        }

    return {
        "count": int(
            values.size
        ),
        "mean_error_deg": float(
            np.mean(values)
        ),
        "median_error_deg": float(
            np.median(values)
        ),
        "p90_error_deg": float(
            np.percentile(
                values,
                90,
            )
        ),
        "p95_error_deg": float(
            np.percentile(
                values,
                95,
            )
        ),
        "max_error_deg": float(
            np.max(values)
        ),
        "within_0_5deg": float(
            np.mean(
                values <= 0.5
            )
        ),
        "within_1deg": float(
            np.mean(
                values <= 1.0
            )
        ),
        "within_3deg": float(
            np.mean(
                values <= 3.0
            )
        ),
        "within_5deg": float(
            np.mean(
                values <= 5.0
            )
        ),
    }


def main():
    args = parse_args()

    if args.trials <= 0:
        raise ValueError(
            "--trials must be positive"
        )

    if args.microphones < 4:
        raise ValueError(
            "--microphones must be at least 4"
        )

    rng = np.random.default_rng(
        args.seed
    )

    radius_m = (
        args.radius_mm / 1000.0
    )

    microphone_positions = (
        fibonacci_sphere(
            args.microphones,
            radius_m,
        )
    )

    speed_of_sound_mps = 343.0

    errors_deg = []
    azimuth_errors_deg = []
    elevation_errors_deg = []

    failed_trials = []
    worst_cases = []

    for trial_index in range(
        args.trials
    ):
        try:
            (
                true_azimuth_deg,
                true_elevation_deg,
                true_direction,
            ) = random_direction_in_sector(
                rng,
                args.azimuth_center,
                args.azimuth_span,
                args.elevation_center,
                args.elevation_span,
            )

            source = speech_like_signal(
                args.duration,
                args.fs,
                rng,
            )

            delays_seconds = (
                far_field_delays_seconds(
                    microphone_positions,
                    true_direction,
                    speed_of_sound_mps,
                )
            )

            channels = (
                simulate_spherical_array(
                    source_signal=source,
                    delays_seconds=delays_seconds,
                    fs=args.fs,
                    rng=rng,
                    snr_db=args.snr_db,
                )
            )

            (
                estimated_azimuth_deg,
                estimated_elevation_deg,
            ) = estimate_local_doa(
                channels=channels,
                microphone_positions_m=(
                    microphone_positions
                ),
                fs=args.fs,
                azimuth_center_deg=(
                    args.azimuth_center
                ),
                azimuth_span_deg=(
                    args.azimuth_span
                ),
                elevation_center_deg=(
                    args.elevation_center
                ),
                elevation_span_deg=(
                    args.elevation_span
                ),
                speed_of_sound_mps=(
                    speed_of_sound_mps
                ),
                coarse_step_deg=4.0,
                fine_step_deg=0.5,
                fine_window_deg=4.0,
                refine=True,
            )

            estimated_direction = (
                direction_from_angles(
                    estimated_azimuth_deg,
                    estimated_elevation_deg,
                )
            )

            error = angular_error_deg(
                true_direction,
                estimated_direction,
            )

            azimuth_error = abs(
                (
                    estimated_azimuth_deg
                    - true_azimuth_deg
                    + 180.0
                )
                % 360.0
                - 180.0
            )

            elevation_error = abs(
                estimated_elevation_deg
                - true_elevation_deg
            )

            errors_deg.append(
                float(error)
            )

            azimuth_errors_deg.append(
                float(azimuth_error)
            )

            elevation_errors_deg.append(
                float(elevation_error)
            )

            worst_cases.append(
                {
                    "trial": int(
                        trial_index
                    ),
                    "true_azimuth_deg": float(
                        true_azimuth_deg
                    ),
                    "true_elevation_deg": float(
                        true_elevation_deg
                    ),
                    "estimated_azimuth_deg": float(
                        estimated_azimuth_deg
                    ),
                    "estimated_elevation_deg": float(
                        estimated_elevation_deg
                    ),
                    "angular_error_deg": float(
                        error
                    ),
                    "azimuth_error_deg": float(
                        azimuth_error
                    ),
                    "elevation_error_deg": float(
                        elevation_error
                    ),
                }
            )

        except Exception as error:
            failed_trials.append(
                {
                    "trial": int(
                        trial_index
                    ),
                    "error": repr(
                        error
                    ),
                }
            )

            print(
                f"[FAIL] trial={trial_index}: {error}"
            )

    worst_cases = sorted(
        worst_cases,
        key=lambda item: item[
            "angular_error_deg"
        ],
        reverse=True,
    )[:20]

    result = {
        "project": (
            "45-degree local spherical DOA"
        ),
        "trials": int(
            args.trials
        ),
        "successful_trials": int(
            len(errors_deg)
        ),
        "failures": int(
            len(failed_trials)
        ),
        "seed": int(
            args.seed
        ),
        "sample_rate_hz": int(
            args.fs
        ),
        "duration_seconds": float(
            args.duration
        ),
        "microphone_count": int(
            args.microphones
        ),
        "radius_m": float(
            radius_m
        ),
        "snr_db": float(
            args.snr_db
        ),
        "field_of_view": {
            "azimuth_center_deg": float(
                args.azimuth_center
            ),
            "azimuth_span_deg": float(
                args.azimuth_span
            ),
            "elevation_center_deg": float(
                args.elevation_center
            ),
            "elevation_span_deg": float(
                args.elevation_span
            ),
        },
        "estimator": {
            "method": (
                "GCC-PHAT"
            ),
            "subsample_interpolation": True,
            "coarse_step_deg": 4.0,
            "fine_step_deg": 0.5,
            "continuous_refinement": True,
        },
        "angular_error": summarize(
            errors_deg
        ),
        "azimuth_absolute_error": summarize(
            azimuth_errors_deg
        ),
        "elevation_absolute_error": summarize(
            elevation_errors_deg
        ),
        "worst_cases": worst_cases,
        "failed_trials": failed_trials[:20],
        "warning": (
            "Simulation results only; "
            "not hardware measurements "
            "or anatomical localization."
        ),
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()