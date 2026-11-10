""" Quaternion operations

This file implements many useful quaternion operation.
A special class is not implemented for quaternions, and they are just considered as 1D numpy arrays with 4 elements.

NOTE: Quaternion Notation: the scalar part of the quaternion is assumed to be the FIRST element of the array (Hamilton notation).
"""
import numpy as np


# functions has been tested with jax.jit, uncomment proper lines to use Jax instead of NumPy (and import jit and jax)


# @ jit
def vec_to_zquat(v) -> np.ndarray:
    """Prepend a leading 0 to the input vector, useful to prepare R3 vector for quaternion multiplication

    Args:
        v (numpy.ndarray): 1D array

    Returns:
        (numpy.ndarray): 1D array
    """
    return np.array((0.0, *v))


# @ jit
def is_unitary(v, rtol=1e-05, atol=1e-08) -> np.ndarray:
    """Check whether the input vector has a unitary norm, whithin a tolerance

    Args:
        v (numpy.ndarray): 1D array
        rtol (float): The relative tolerance parameter (see numpy.isclose()).
        atol (float): The absolute tolerance parameter (see numpy.isclose()).


    Returns:
        (bool): whether the input has ~unitary norm or not
    """
    return np.isclose(np.linalg.norm(v), 1.0, rtol=rtol, atol=atol).item()


# @ jit
def normalize(v) -> np.ndarray:
    """Normalize the input vector

    Args:
        v (numpy.ndarray): 1D array

    Returns:
        (numpy.ndarray): 1D array
    """
    return np.nan_to_num(v / np.linalg.norm(v))


# @ jit
def conjugate(q) -> np.ndarray:
    """Conjugate the input quaternion.
    The 1D input array is treated as a quaternion q=(w,xi,yj,zk)

    Args:
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (4,)
    """
    return np.array((q[0], -q[1], -q[2], -q[3]))


# @ jit
def vec(q) -> np.ndarray:
    """Vector part of the input quaternion
    The 1D input array is treated as a quaternion q=(w,xi,yj,zk)

    Args:
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (3,)
    """
    return np.array((q[1], q[2], q[3]))


# @ jit
def nq(q) -> np.ndarray:
    """Normalized vector part of the input quaternion
    The 1D input array is treated as a quaternion q=(w,xi,yj,zk)

    Args:
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (3,)
    """
    return normalize(vec(q))


# @ jit
def mulq(p, q) -> np.ndarray:
    """Quaternion multiplication
    The 1D input arrays are treated as a quaternions q=(w,xi,yj,zk)

    Args:
        p (numpy.ndarray): 1D array of size (4,)
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (4,)
    """
    return np.array(
        (
            p[0] * q[0] - vec(p).dot(vec(q)),
            *(p[0] * vec(q) + q[0] * vec(p) + np.cross(vec(p), vec(q))),
        )
    )


# @ jit
def diq(p, q) -> np.ndarray:
    """Angular distance between the two input quaternions
    The 1D input arrays are treated as a quaternions q=(w,xi,yj,zk).
    The input is supposed to be UNITARY quaternions.

    Args:
        p (numpy.ndarray): 1D array of size (4,)
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (1,)
    """
    angle = 2 * np.arctan2(
        np.linalg.norm(vec(mulq(conjugate(p), q))), (mulq(conjugate(p), q))[0]
    )
    return angle


# @ jit
def expq(q) -> np.ndarray:
    """Exponentiate the input quaternion
    Can be used for integration by passing in the angular velocity as a quaternion with null scalar part (use vec_to_zquat()).
    The 1D input array is treated as a quaternions q=(w,xi,yj,zk).

    Args:
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (4,), unitary
    """
    return np.exp(q[0]) * np.array(
        (np.cos(np.linalg.norm(vec(q))), *(nq(q) * np.sin(np.linalg.norm(vec(q)))))
    )


# @ jit
def logq(q, p=None) -> np.ndarray:
    """Logarithmic map of q to R3, considered centered in (aka, relative to) p.
    The 1D input arrays are treated as a quaternions q=(w,xi,yj,zk).

    Args:
        q (numpy.ndarray): 1D array of size (4,)

    Returns:
        (numpy.ndarray): 1D array of size (3,)
    """
    if p is None:
        return diq(np.array((1, 0, 0, 0)), q) * nq(q)
    else:
        return diq(p, q) * nq(mulq(q, conjugate(p)))


# @ jit
def expv(v, p=None) -> np.ndarray:
    """Inverse of the logarithmic map, considered centered in (i.e., logarithmic map relative to) p.
    Args:
        v (numpy.ndarray): 1D array of size (3,)

    Returns:
        (numpy.ndarray): 1D array of size (4,), unitary
    """
    th = np.linalg.norm(v)
    n = normalize(v)
    if p is None:
        return np.array((np.cos(th / 2), *(n * np.sin(th / 2))))
    else:
        return mulq(p, np.array((np.cos(th / 2), *(n * np.sin(th / 2)))))


# @jit
def q_to_R(q) -> np.ndarray:
    """Convert a quaternion to a rotation matrix
    Args:
        q (numpy.ndarray): 1D array of size (4,), unitary
    """
    if not is_unitary(q):
        print(q)
        raise ValueError("Input quaternion is not unitary")
    return np.array(
        (
            (
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ),
            (
                2 * (q[1] * q[2] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ),
            (
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ),
        )
    )
