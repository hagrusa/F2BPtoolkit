"""Initial state for the Full Two-Body Problem simulation."""

import numpy as np
from typing import Optional


class InitialState:
    """
    State vector for the F2BP.

    The 30-element state vector is:
      [r(3), v(3), omega_c(3), omega_s(3), A_to_N(9), B_to_A(9)]

    where
      r       : relative position of secondary in primary body frame (A), km
      v       : relative velocity in A frame, km/s
      omega_c : primary angular velocity in A frame, rad/s
      omega_s : secondary angular velocity in secondary body frame (B), rad/s
      A_to_N  : rotation matrix A→N  (Cc in Hou notation), row-major (9 elements)
      B_to_A  : rotation matrix B→A  (C  in Hou notation), row-major (9 elements)

    The Python API works in SI (meters, m/s).  Conversion to C++ internal
    units (km, km/s) is done by ``to_vector_km()``.
    """

    def __init__(self):
        self._position: Optional[np.ndarray] = None    # m
        self._velocity: Optional[np.ndarray] = None    # m/s
        self._omega_primary: Optional[np.ndarray] = None   # rad/s
        self._omega_secondary: Optional[np.ndarray] = None # rad/s
        self._A_to_N: np.ndarray = np.eye(3)   # A→N
        self._B_to_A: np.ndarray = np.eye(3)  # B→A

    # ── position ──────────────────────────────────────────────────────────────

    @property
    def position(self) -> Optional[np.ndarray]:
        """Relative position (secondary w.r.t. primary) in primary body frame, meters."""
        return self._position

    @position.setter
    def position(self, r):
        self._position = np.asarray(r, dtype=float).reshape(3)

    # ── velocity ──────────────────────────────────────────────────────────────

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Relative velocity in primary body frame, m/s."""
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        self._velocity = np.asarray(v, dtype=float).reshape(3)

    # ── angular velocities ────────────────────────────────────────────────────

    @property
    def omega_primary(self) -> Optional[np.ndarray]:
        """Primary angular velocity in primary body frame (A), rad/s."""
        return self._omega_primary

    @omega_primary.setter
    def omega_primary(self, w):
        self._omega_primary = np.asarray(w, dtype=float).reshape(3)

    @property
    def omega_secondary(self) -> Optional[np.ndarray]:
        """Secondary angular velocity in secondary body frame (B), rad/s."""
        return self._omega_secondary

    @omega_secondary.setter
    def omega_secondary(self, w):
        self._omega_secondary = np.asarray(w, dtype=float).reshape(3)

    # ── attitude matrices ─────────────────────────────────────────────────────

    @property
    def A_to_N(self) -> np.ndarray:
        """
        Rotation matrix from primary body frame (A) to inertial frame (N), shape (3,3).
        Satisfies v_N = A_to_N @ v_A.  Default: identity (frames aligned).
        """
        return self._A_to_N

    @A_to_N.setter
    def A_to_N(self, C):
        self._A_to_N = np.asarray(C, dtype=float).reshape(3, 3)

    @property
    def B_to_A(self) -> np.ndarray:
        """
        Rotation matrix from secondary body frame (B) to primary body frame (A), shape (3,3).
        Satisfies v_A = B_to_A @ v_B.  Default: identity (frames aligned).
        """
        return self._B_to_A

    @B_to_A.setter
    def B_to_A(self, C):
        self._B_to_A = np.asarray(C, dtype=float).reshape(3, 3)

    # ── helpers ───────────────────────────────────────────────────────────────

    def to_vector_km(self) -> np.ndarray:
        """
        Return the 30-element state vector in C++ internal units
        (km, km/s, rad/s), shape (30,).
        """
        x = np.zeros(30)
        x[0:3]  = self._position / 1000.0        # m → km
        x[3:6]  = self._velocity / 1000.0        # m/s → km/s
        x[6:9]  = self._omega_primary            # rad/s unchanged
        x[9:12] = self._omega_secondary          # rad/s unchanged
        x[12:21] = self._A_to_N.flatten(order='C')   # A→N, row-major
        x[21:30] = self._B_to_A.flatten(order='C')   # B→A, row-major
        return x

    def validate(self):
        """Raise ValueError if any required component is missing."""
        missing = []
        if self._position is None:
            missing.append("position")
        if self._velocity is None:
            missing.append("velocity")
        if self._omega_primary is None:
            missing.append("omega_primary")
        if self._omega_secondary is None:
            missing.append("omega_secondary")
        if missing:
            raise ValueError(f"Initial state is missing: {', '.join(missing)}")

    def __repr__(self):
        r = self._position
        return (f"InitialState(r={r} m, "
                f"|r|={np.linalg.norm(r):.1f} m)" if r is not None else "InitialState(<unset>)")
