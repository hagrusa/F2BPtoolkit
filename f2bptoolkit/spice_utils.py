"""SPICE kernel utilities for setting up initial conditions."""

import numpy as np
from typing import Union, List, Tuple


def load_kernels(kernel_files: Union[str, List[str]]):
    """
    Load one or more SPICE kernel files.

    Parameters
    ----------
    kernel_files : str or list of str
        Path(s) to SPICE kernel files (.bsp, .tls, .tpc, .tf, etc.).
    """
    import spiceypy as spice
    if isinstance(kernel_files, str):
        kernel_files = [kernel_files]
    for kf in kernel_files:
        spice.furnsh(kf)


def unload_kernels(kernel_files: Union[str, List[str]]):
    """Unload SPICE kernel files."""
    import spiceypy as spice
    if isinstance(kernel_files, str):
        kernel_files = [kernel_files]
    for kf in kernel_files:
        spice.unload(kf)


def str_to_et(epoch: Union[str, float]) -> float:
    """
    Convert an epoch string or float to SPICE ephemeris time (ET).

    Parameters
    ----------
    epoch : str or float
        If str, parsed by SPICE (e.g. '2022-10-01T00:00:00').
        If float, assumed to already be ET seconds past J2000.

    Returns
    -------
    float
        Ephemeris time in seconds past J2000.
    """
    import spiceypy as spice
    if isinstance(epoch, (int, float)):
        return float(epoch)
    return spice.str2et(str(epoch))


def state_from_spice(
    kernel_files: Union[str, List[str]],
    primary_name: str,
    secondary_name: str,
    epoch: Union[str, float],
    frame: str = "J2000",
    observer: str = "SSB",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the relative position and velocity of the secondary w.r.t. the primary
    at a given epoch using SPICE kernels.

    The returned state is expressed in the given reference frame and is in SI
    units (meters, m/s).  Note that SPICE gives states in the inertial frame,
    so the result is in the *inertial* (N) frame.  To use this as the F2BP
    initial state (which is in the primary body frame A), rotate by N→A:

        N_to_A = rotation_matrix_from_spice(...)   # returns N→A
        r_A = N_to_A @ r_N
        v_A = N_to_A @ v_N

    Parameters
    ----------
    kernel_files : str or list of str
        SPICE kernel files to furnish.
    primary_name : str
        SPICE name/ID of the primary body (e.g., "DIDYMOS").
    secondary_name : str
        SPICE name/ID of the secondary body (e.g., "DIMORPHOS").
    epoch : str or float
        Epoch at which to evaluate the state.
    frame : str
        Reference frame.  Default: "J2000".
    observer : str
        Observer for the state.  Default: "SSB" (Solar System Barycenter).

    Returns
    -------
    position : ndarray, shape (3,)
        Relative position r_secondary - r_primary in meters.
    velocity : ndarray, shape (3,)
        Relative velocity v_secondary - v_primary in m/s.
    """
    import spiceypy as spice

    load_kernels(kernel_files)
    et = str_to_et(epoch)

    # State of primary w.r.t. observer
    state_primary, _ = spice.spkez(
        spice.bodn2c(primary_name) if isinstance(primary_name, str) else primary_name,
        et, frame, "NONE", 0
    )

    # State of secondary w.r.t. observer
    state_secondary, _ = spice.spkez(
        spice.bodn2c(secondary_name) if isinstance(secondary_name, str) else secondary_name,
        et, frame, "NONE", 0
    )

    # Relative state (secondary w.r.t. primary)
    rel_state = np.array(state_secondary) - np.array(state_primary)

    # Convert km → m, km/s → m/s
    position = rel_state[0:3] * 1000.0
    velocity = rel_state[3:6] * 1000.0

    return position, velocity


def rotation_matrix_from_spice(
    body_name: str,
    epoch: Union[str, float],
    body_frame: str,
    inertial_frame: str = "J2000",
) -> np.ndarray:
    """
    Get the rotation matrix from the inertial frame to the body-fixed frame
    at a given epoch.

    Returns the N→A matrix.  To get the A→N matrix needed by ``InitialState.A_to_N``,
    transpose the result:  ``A_to_N = rotation_matrix_from_spice(...).T``.

    Parameters
    ----------
    body_name : str
        SPICE name of the body (needed only if body_frame is derived from it).
    epoch : str or float
        Epoch at which to evaluate the rotation.
    body_frame : str
        SPICE name of the body-fixed frame (e.g., "DIDYMOS_FIXED").
    inertial_frame : str
        Inertial reference frame.  Default: "J2000".

    Returns
    -------
    N_to_A : ndarray, shape (3,3)
        Rotation matrix N→A (inertial to primary body frame).
        Transpose to get A→N for use as ``InitialState.A_to_N``.
    """
    import spiceypy as spice

    et = str_to_et(epoch)
    # pxform returns the matrix that rotates vectors FROM inertial_frame TO body_frame,
    # i.e., N→A.  Transpose to get A→N for use as InitialState.A_to_N.
    N_to_A = np.array(spice.pxform(inertial_frame, body_frame, et))
    return N_to_A


def angular_velocity_from_spice(
    body_name: str,
    epoch: Union[str, float],
    body_frame: str,
    inertial_frame: str = "J2000",
) -> np.ndarray:
    """
    Get the angular velocity of a body (in its body-fixed frame) at an epoch.

    Parameters
    ----------
    body_name : str
        SPICE name of the body.
    epoch : str or float
        Epoch.
    body_frame : str
        SPICE name of the body-fixed frame.
    inertial_frame : str
        Inertial reference frame.  Default: "J2000".

    Returns
    -------
    omega : ndarray, shape (3,)
        Angular velocity in rad/s, expressed in the body-fixed frame.
    """
    import spiceypy as spice

    et = str_to_et(epoch)
    # xf2rav decomposes a state transformation matrix into rotation + angular velocity
    xform = spice.sxform(inertial_frame, body_frame, et)   # 6x6 state matrix
    rot, omega_inertial = spice.xf2rav(xform)              # rot (3x3), omega in inertial frame

    # Rotate angular velocity to body frame
    omega_body = np.array(rot) @ np.array(omega_inertial)
    return omega_body
