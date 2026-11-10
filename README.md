# Quaternion dynamics

This repository contains documentation and code relative to a formulation that allows implementing linear dynamical system (like Linear Quadratic Regulator, online minimum-jerk tracking, spring-mass-damper, etc.) using quaternion, and a custom definition of the error vector.

See `docs/linear_systems_on_SO3.md` for the mathematical details.

See `docs/attitude_filter_in_SO3_coordinates.md` for an attitude filtering approach that reproduces the Invariant Extended Kalman Filter (but derived from the "linear dynamics" model).
