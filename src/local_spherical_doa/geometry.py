from __future__ import annotations

import numpy as np


def fibonacci_sphere(
    microphone_count: int,
    radius_m: float,
) -> np.ndarray:
    """
    Generate approximately uniform microphone positions
    on a sphere centered at the origin.

    Returns:
        ndarray with shape (microphone_count, 3)
    """
    if microphone_count < 4:
        raise ValueError(
            "microphone_count must be at least 4"
        )

    if radius_m <= 0:
        raise ValueError(
            "radius_m must be positive"
        )

    indices = np.arange(
        microphone_count,
        dtype=np.float64,
    )

    golden_angle = np.pi * (
        3.0 - np.sqrt(5.0)
    )

    y = 1.0 - (
        2.0 * indices + 1.0
    ) / microphone_count

    ring_radius = np.sqrt(
        np.maximum(
            0.0,
            1.0 - y * y,
        )
    )

    theta = golden_angle * indices

    x = ring_radius * np.cos(theta)
    z = ring_radius * np.sin(theta)

    return radius_m * np.column_stack(
        [x, y, z]
    )


def direction_from_angles(
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    Convert azimuth/elevation to a unit direction vector.
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


def angles_from_direction(
    direction: np.ndarray,
) -> tuple[float, float]:
    """
    Convert a direction vector into azimuth/elevation.
    """
    direction = np.asarray(
        direction,
        dtype=np.float64,
    )

    direction = direction / (
        np.linalg.norm(direction)
        + 1e-12
    )

    azimuth_deg = np.degrees(
        np.arctan2(
            direction[1],
            direction[0],
        )
    ) % 360.0

    elevation_deg = np.degrees(
        np.arcsin(
            np.clip(
                direction[2],
                -1.0,
                1.0,
            )
        )
    )

    return (
        float(azimuth_deg),
        float(elevation_deg),
    )


def far_field_delays_seconds(
    microphone_positions_m: np.ndarray,
    direction: np.ndarray,
    speed_of_sound_mps: float = 343.0,
) -> np.ndarray:
    """
    Compute plane-wave relative delays for a spherical array.

    The output is relative to the array center.
    """
    microphone_positions_m = np.asarray(
        microphone_positions_m,
        dtype=np.float64,
    )

    direction = np.asarray(
        direction,
        dtype=np.float64,
    )

    direction = direction / (
        np.linalg.norm(direction)
        + 1e-12
    )

    return -(
        microphone_positions_m @ direction
    ) / speed_of_sound_mps