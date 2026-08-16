# Method

## Coordinate convention

Azimuth is measured in degrees in the XY plane from +X
toward +Y.

Elevation is measured from the XY plane toward +Z.

A unit direction vector is:

u = [
    cos(elevation) cos(azimuth),
    cos(elevation) sin(azimuth),
    sin(elevation)
]

## Array model

The microphones are located on a spherical surface centered
at the origin.

The default array has eight microphones and a radius of 25 mm.

## Far-field delay model

For microphone position m_i and unit source direction u:

tau_i = -(m_i dot u) / c

where c is the speed of sound.

The estimator compares measured pairwise delays with theoretical
delays over a bounded angular grid.

## Error metric

The angular error is the angle between the true and estimated
unit direction vectors:

error = acos(clip(dot(u_true, u_estimated), -1, 1))

The result is reported in degrees.

## Estimator

1. Estimate delays with GCC-PHAT.
2. Perform a coarse local grid search.
3. Perform a fine local grid search.
4. Optionally refine continuously.
5. Convert the final angles to a unit vector.
6. Compute spherical angular error.