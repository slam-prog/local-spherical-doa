import numpy as np

from local_spherical_doa.doa import (
    direction_from_angles,
    parabolic_peak_offset,
)


def test_direction_is_unit_norm():
    direction = direction_from_angles(
        125.0,
        25.0,
    )

    assert np.isclose(
        np.linalg.norm(direction),
        1.0,
        atol=1e-12,
    )


def test_parabolic_peak_offset_zero():
    offset = parabolic_peak_offset(
        left=1.0,
        center=2.0,
        right=1.0,
    )

    assert np.isclose(
        offset,
        0.0,
    )