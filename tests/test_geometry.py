import numpy as np

from local_spherical_doa.geometry import (
    angles_from_direction,
    direction_from_angles,
    far_field_delays_seconds,
    fibonacci_sphere,
)


def test_fibonacci_sphere_shape():
    positions = fibonacci_sphere(
        microphone_count=8,
        radius_m=0.025,
    )

    assert positions.shape == (8, 3)


def test_fibonacci_sphere_radius():
    positions = fibonacci_sphere(
        microphone_count=8,
        radius_m=0.025,
    )

    radii = np.linalg.norm(
        positions,
        axis=1,
    )

    assert np.allclose(
        radii,
        0.025,
        atol=1e-12,
    )


def test_direction_angle_round_trip():
    direction = direction_from_angles(
        125.0,
        25.0,
    )

    azimuth, elevation = (
        angles_from_direction(
            direction
        )
    )

    assert np.isclose(
        azimuth,
        125.0,
        atol=1e-10,
    )

    assert np.isclose(
        elevation,
        25.0,
        atol=1e-10,
    )


def test_zero_center_delay():
    positions = fibonacci_sphere(
        microphone_count=8,
        radius_m=0.025,
    )

    direction = direction_from_angles(
        125.0,
        25.0,
    )

    delays = far_field_delays_seconds(
        positions,
        direction,
    )

    assert delays.shape == (8,)
    assert np.all(np.isfinite(delays))