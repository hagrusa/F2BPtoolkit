"""Simulation results container."""

import os
import numpy as np
from typing import Optional

# Bump when the on-disk layout changes in a backward-incompatible way.
_FORMAT_VERSION = "1"
_MAGIC          = "f2bptoolkit-results"


class SimulationResults:
    """
    Container for F2BP simulation output.

    All quantities are in SI units unless noted.

    Attributes
    ----------
    times : ndarray, shape (N,)
        Integration times in seconds.
    position : ndarray, shape (N, 3)
        Relative position of secondary w.r.t. primary in primary body frame, meters.
    velocity : ndarray, shape (N, 3)
        Relative velocity in primary body frame, m/s.
    omega_primary : ndarray, shape (N, 3)
        Primary angular velocity in primary body frame (A), rad/s.
    omega_secondary : ndarray, shape (N, 3)
        Secondary angular velocity in secondary body frame (B), rad/s.
    A_to_N : ndarray, shape (N, 3, 3)
        Rotation matrices A→N at each output time (columns = primary body-frame
        axes expressed in the inertial frame).  The Hou ODE integrates
        dCc/dt = Cc @ tilde(ωc), which produces the active/A→N form.
        To get N→A, take the transpose: ``A_to_N.swapaxes(1, 2)``.
    B_to_A : ndarray, shape (N, 3, 3)
        Rotation matrices B→A at each output time (columns = secondary body-frame
        axes expressed in the primary body frame).  The Hou ODE for C integrates
        to the B→A active form.  To get A→B, take the transpose:
        ``B_to_A.swapaxes(1, 2)``.
    separation : ndarray, shape (N,)
        Scalar separation |r| in meters.
    n_steps : int
        Number of output timesteps.
    """

    def __init__(self, times, states_km, G,
                 mass_primary=None, mass_secondary=None,
                 inertia_primary=None, inertia_secondary=None,
                 hyp_states_km=None, solar_states_km=None,
                 potential_km=None):
        """
        Build from C++ output (km units).

        Parameters
        ----------
        times : array-like, shape (N,)
        states_km : array-like, shape (N, 30) or (N*30,)
            States in km/km/s/rad/s units.
        G : float
            Gravitational constant in SI.
        mass_primary, mass_secondary : float, optional
            Masses in kg (returned by C++ from inertia integrals).
        inertia_primary, inertia_secondary : array-like, shape (3,), optional
            Principal moments of inertia in kg·km².
        hyp_states_km : array-like, shape (N, 6) or (N*6,), optional
            Flyby perturber states in km/km/s.
        solar_states_km : array-like, shape (N, 6) or (N*6,), optional
            Heliocentric body states in km/km/s.
        potential_km : array-like, shape (N,), optional
            Mutual gravitational potential in kg·km²·s⁻² (C++ internal units).
        """
        self._times = np.asarray(times, dtype=float)
        N = len(self._times)

        raw = np.asarray(states_km, dtype=float).reshape(N, 30)

        # Convert km → m, km/s → m/s
        self._position  = raw[:, 0:3]  * 1000.0
        self._velocity  = raw[:, 3:6]  * 1000.0
        self._omega_c   = raw[:, 6:9]           # rad/s unchanged
        self._omega_s   = raw[:, 9:12]
        # Rotation matrices stored row-major in state vector.
        # Hou ODE integrates dCc/dt = Cc @ tilde(ωc) → Cc = A→N (v_N = Cc @ v_A)
        # Hou ODE integrates dC/dt = C @ tilde(ωs_B) - tilde(ωc_A) @ C → C = B→A (v_A = C @ v_B)
        self._A_to_N = raw[:, 12:21].reshape(N, 3, 3)   # A→N
        self._B_to_A = raw[:, 21:30].reshape(N, 3, 3)   # B→A

        # Perturbation states (optional)
        if hyp_states_km is not None and len(hyp_states_km) > 0:
            h = np.asarray(hyp_states_km, dtype=float).reshape(-1, 6)
            self._flyby_position = h[:, 0:3] * 1000.0
            self._flyby_velocity = h[:, 3:6] * 1000.0
        else:
            self._flyby_position = None
            self._flyby_velocity = None

        if solar_states_km is not None and len(solar_states_km) > 0:
            s = np.asarray(solar_states_km, dtype=float).reshape(-1, 6)
            self._solar_position = s[:, 0:3] * 1000.0
            self._solar_velocity = s[:, 3:6] * 1000.0
        else:
            self._solar_position = None
            self._solar_velocity = None

        self._G  = G
        self._Mc = mass_primary
        self._Ms = mass_secondary
        # Store inertia in SI (kg·m²) by converting from kg·km²
        self._IA = np.asarray(inertia_primary)   * 1e6 if inertia_primary   is not None else None
        self._IB = np.asarray(inertia_secondary) * 1e6 if inertia_secondary is not None else None

        # Mutual gravitational potential (kg·km²·s⁻² → J via ×1e6)
        if potential_km is not None and len(potential_km) > 0:
            self._potential = np.asarray(potential_km, dtype=float) * 1e6
        else:
            self._potential = None

        # Lazily computed derived quantities
        self._energy = None
        self._angular_momentum = None

    # ── core state properties ─────────────────────────────────────────────────

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def omega_primary(self) -> np.ndarray:
        return self._omega_c

    @property
    def omega_secondary(self) -> np.ndarray:
        return self._omega_s

    @property
    def A_to_N(self) -> np.ndarray:
        """Rotation matrices A→N, shape (N, 3, 3).  Transpose for N→A."""
        return self._A_to_N

    @property
    def B_to_A(self) -> np.ndarray:
        """Rotation matrices B→A, shape (N, 3, 3).  Transpose for A→B."""
        return self._B_to_A

    @property
    def separation(self) -> np.ndarray:
        """Scalar separation |r| in meters, shape (N,)."""
        return np.linalg.norm(self._position, axis=1)

    @property
    def potential_energy(self) -> Optional[np.ndarray]:
        """Mutual gravitational potential energy in J, shape (N,)."""
        return self._potential

    @property
    def n_steps(self) -> int:
        return len(self._times)

    # ── orientation utilities ──────────────────────────────────────────────────

    def dcm_secondary(self, frame: str = 'orbital') -> np.ndarray:
        """
        Rotation matrices mapping the requested reference frame to the secondary
        body frame B, shape (N, 3, 3).

        Hou ODE rotation-matrix conventions (4 frames: A, B, N, O):
            A_to_N  (Cc in Hou notation)  A→N  (v_N = A_to_N @ v_A)
            B_to_A  (C  in Hou notation)  B→A  (v_A = B_to_A @ v_B)

        Parameters
        ----------
        frame : {'inertial', 'orbital', 'primary'}
            'inertial'  – N→B  =  A_to_B @ N_to_A  =  B_to_A.T @ A_to_N.T
            'orbital'   – O→B  =  N_to_B @ O_to_N  (Hill/LVLH frame where
                          x̂ = radial outward, ẑ = orbit angular momentum, ŷ = ẑ×x̂)
            'primary'   – A→B  =  B_to_A.T

        Returns
        -------
        ndarray, shape (N, 3, 3)
        """
        A_to_N = self._A_to_N                    # (N,3,3)  A→N
        B_to_A = self._B_to_A                    # (N,3,3)  B→A
        N_to_A = A_to_N.swapaxes(1, 2)          # N→A
        A_to_B = B_to_A.swapaxes(1, 2)          # A→B

        if frame == 'primary':
            return A_to_B                        # A→B

        # N→B = (A→B) @ (N→A)
        N_to_B = np.einsum('nij,njk->nik', A_to_B, N_to_A)

        if frame == 'inertial':
            return N_to_B

        if frame == 'orbital':
            # Build inertial-frame r and v from A-frame quantities
            r_A = self._position               # (N,3) in A frame
            v_A = self._velocity               # (N,3) inertial velocity in A frame

            r_N = np.einsum('nij,nj->ni', A_to_N, r_A)   # (A→N) @ r_A = r_N
            v_N = np.einsum('nij,nj->ni', A_to_N, v_A)   # (A→N) @ v_A = v_N

            x_hat = r_N / np.linalg.norm(r_N, axis=1, keepdims=True)
            h_vec = np.cross(r_N, v_N)
            z_hat = h_vec / np.linalg.norm(h_vec, axis=1, keepdims=True)
            y_hat = np.cross(z_hat, x_hat)

            # N_to_O: rows = O basis vectors expressed in N  (v_O = N_to_O @ v_N)
            N_to_O = np.stack([x_hat, y_hat, z_hat], axis=1)

            # O→B = (N→B) @ (O→N) = N_to_B @ N_to_O.T
            return np.einsum('nij,nkj->nik', N_to_B, N_to_O)   # N_to_B @ N_to_O.T

        raise ValueError(f"frame must be 'inertial', 'orbital', or 'primary'; got {frame!r}")

    def secondary_euler_angles(self, frame: str = 'orbital',
                               convention: str = 'ZYX') -> tuple:
        """
        Euler angles of the secondary body (Dimorphos) relative to *frame*.

        Parameters
        ----------
        frame : {'inertial', 'orbital', 'primary'}
            Reference frame — see :meth:`dcm_secondary`.
        convention : str
            Only ``'ZYX'`` is currently supported (yaw-pitch-roll / aerospace).

        Returns
        -------
        roll, pitch, yaw : ndarray, shape (N,), degrees
            Angles such that  M = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        """
        if convention != 'ZYX':
            raise ValueError("Only ZYX convention is currently supported.")

        M = self.dcm_secondary(frame)          # (N,3,3)  frame→B

        roll  = np.degrees(np.arctan2( M[:, 2, 1],  M[:, 2, 2]))
        pitch = np.degrees(np.arcsin(-M[:, 2, 0]))
        yaw   = np.degrees(np.arctan2( M[:, 1, 0],  M[:, 0, 0]))

        return roll, pitch, yaw

    # ── derived quantities ─────────────────────────────────────────────────────

    @property
    def angular_momentum(self) -> Optional[np.ndarray]:
        """
        Total angular momentum vector in inertial frame, shape (N, 3), kg·m²/s.

        Requires inertia tensors (available after Simulation.integrate()).
        """
        if self._IA is None or self._IB is None or self._Mc is None:
            return None
        if self._angular_momentum is not None:
            return self._angular_momentum

        N = self.n_steps
        m = self._Mc * self._Ms / (self._Mc + self._Ms)
        H = np.zeros((N, 3))

        for i in range(N):
            A_to_N = self._A_to_N[i]   # A→N  (v_N = A_to_N @ v_A)
            B_to_A = self._B_to_A[i]   # B→A  (v_A = B_to_A @ v_B)
            r = self._position[i]       # m, in A frame
            v = self._velocity[i]       # m/s, in A frame

            # Transform to inertial frame:  r_N = (A→N) @ r_A
            r_N = A_to_N @ r
            v_N = A_to_N @ v

            # Orbital angular momentum (inertial)
            L_orb = m * np.cross(r_N, v_N)

            # Primary spin angular momentum in N:  (A→N) @ diag(IA) @ ωc_A
            wc_A = self._omega_c[i]
            L_c = A_to_N @ (self._IA * wc_A)

            # Secondary spin angular momentum.
            # omega_s in state is in A frame (despite docstring).
            # Convert A→B via B_to_A.T, then to N via A_to_N @ B_to_A (B→A→N).
            ws_A = self._omega_s[i]
            ws_B = B_to_A.T @ ws_A            # A_to_B = B_to_A.T; A→B applied to ωs_A
            L_s_B = self._IB * ws_B           # angular momentum in B frame
            L_s = A_to_N @ B_to_A @ L_s_B    # B→A→N

            H[i] = L_orb + L_c + L_s

        self._angular_momentum = H
        return H

    @property
    def angular_momentum_magnitude(self) -> Optional[np.ndarray]:
        """Magnitude of total angular momentum, shape (N,), kg·m²/s."""
        H = self.angular_momentum
        return None if H is None else np.linalg.norm(H, axis=1)

    @property
    def kinetic_energy_orbital(self) -> Optional[np.ndarray]:
        """Translational kinetic energy in J, shape (N,)."""
        if self._Mc is None:
            return None
        m = self._Mc * self._Ms / (self._Mc + self._Ms)
        return 0.5 * m * np.sum(self._velocity**2, axis=1)

    @property
    def kinetic_energy_rotation_primary(self) -> Optional[np.ndarray]:
        """Primary rotational kinetic energy in J, shape (N,)."""
        if self._IA is None:
            return None
        return 0.5 * np.sum(self._IA * self._omega_c**2, axis=1)

    @property
    def kinetic_energy_rotation_secondary(self) -> Optional[np.ndarray]:
        """Secondary rotational kinetic energy in J, shape (N,)."""
        if self._IB is None:
            return None
        return 0.5 * np.sum(self._IB * self._omega_s**2, axis=1)

    # ── perturber accessors ───────────────────────────────────────────────────

    @property
    def flyby_position(self) -> Optional[np.ndarray]:
        """Flyby perturber position relative to binary barycenter, m, shape (N, 3)."""
        return self._flyby_position

    @property
    def flyby_velocity(self) -> Optional[np.ndarray]:
        """Flyby perturber velocity, m/s, shape (N, 3)."""
        return self._flyby_velocity

    @property
    def solar_position(self) -> Optional[np.ndarray]:
        """Heliocentric solar position relative to binary barycenter, m, shape (N, 3)."""
        return self._solar_position

    @property
    def solar_velocity(self) -> Optional[np.ndarray]:
        """Heliocentric solar velocity, m/s, shape (N, 3)."""
        return self._solar_velocity

    # ── file I/O ──────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save results to a compressed NumPy archive (``.npz``).

        All quantities are stored in SI units (m, m/s, rad/s, kg, kg·m²).
        The file can be reloaded with :meth:`SimulationResults.load`.

        Parameters
        ----------
        path : str
            Output path.  The ``.npz`` extension is required.

        Raises
        ------
        ValueError
            If *path* does not end in ``.npz``.
        """
        path = str(path)
        if not path.endswith('.npz'):
            raise ValueError(
                f"path must end with '.npz', got: {path!r}"
            )

        arrays: dict = {
            '_magic':          np.array(_MAGIC),
            '_format_version': np.array(_FORMAT_VERSION),
            'G':               np.array(self._G),
            'times':           self._times,
            'position':        self._position,
            'velocity':        self._velocity,
            'omega_primary':   self._omega_c,
            'omega_secondary': self._omega_s,
            'A_to_N':          self._A_to_N,
            'B_to_A':          self._B_to_A,
        }

        if self._Mc is not None:
            arrays['mass_primary']   = np.array(self._Mc)
        if self._Ms is not None:
            arrays['mass_secondary'] = np.array(self._Ms)
        if self._IA is not None:
            arrays['inertia_primary']   = self._IA
        if self._IB is not None:
            arrays['inertia_secondary'] = self._IB
        if self._flyby_position is not None:
            arrays['flyby_position'] = self._flyby_position
            arrays['flyby_velocity'] = self._flyby_velocity
        if self._solar_position is not None:
            arrays['solar_position'] = self._solar_position
            arrays['solar_velocity'] = self._solar_velocity
        if self._potential is not None:
            arrays['potential_energy'] = self._potential

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'SimulationResults':
        """
        Load results from a ``.npz`` file written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        SimulationResults

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file is not a valid f2bptoolkit results archive.
        """
        path = str(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Results file not found: {path!r}")

        data = np.load(path, allow_pickle=False)

        magic = str(data['_magic'])
        if magic != _MAGIC:
            raise ValueError(
                f"{path!r} is not a valid f2bptoolkit results file "
                f"(expected magic {_MAGIC!r}, got {magic!r})"
            )

        version = str(data['_format_version'])
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path!r} was written with format version {version!r}; "
                f"this installation expects version {_FORMAT_VERSION!r}"
            )

        def _opt(key):
            return data[key] if key in data else None

        return cls._from_arrays(
            times           = data['times'],
            position        = data['position'],
            velocity        = data['velocity'],
            omega_primary   = data['omega_primary'],
            omega_secondary = data['omega_secondary'],
            A_to_N          = data['A_to_N'],
            B_to_A          = data['B_to_A'],
            G               = float(data['G']),
            mass_primary    = float(data['mass_primary'])   if 'mass_primary'   in data else None,
            mass_secondary  = float(data['mass_secondary']) if 'mass_secondary' in data else None,
            inertia_primary   = _opt('inertia_primary'),
            inertia_secondary = _opt('inertia_secondary'),
            flyby_position  = _opt('flyby_position'),
            flyby_velocity  = _opt('flyby_velocity'),
            solar_position  = _opt('solar_position'),
            solar_velocity  = _opt('solar_velocity'),
            potential_energy = _opt('potential_energy'),
        )

    @classmethod
    def _from_arrays(cls, *, times, position, velocity, omega_primary,
                     omega_secondary, A_to_N, B_to_A, G,
                     mass_primary=None, mass_secondary=None,
                     inertia_primary=None, inertia_secondary=None,
                     flyby_position=None, flyby_velocity=None,
                     solar_position=None, solar_velocity=None,
                     potential_energy=None) -> 'SimulationResults':
        """
        Construct directly from SI arrays (bypasses the km-unit ``__init__``).

        All array quantities must already be in SI units (m, m/s, rad/s,
        kg, kg·m²).  Used internally by :meth:`load`.
        """
        obj = object.__new__(cls)
        obj._times   = np.asarray(times,           dtype=float)
        obj._position = np.asarray(position,        dtype=float)
        obj._velocity = np.asarray(velocity,        dtype=float)
        obj._omega_c  = np.asarray(omega_primary,   dtype=float)
        obj._omega_s  = np.asarray(omega_secondary, dtype=float)
        obj._A_to_N   = np.asarray(A_to_N,          dtype=float)
        obj._B_to_A   = np.asarray(B_to_A,          dtype=float)
        obj._G  = float(G)
        obj._Mc = float(mass_primary)   if mass_primary   is not None else None
        obj._Ms = float(mass_secondary) if mass_secondary is not None else None
        obj._IA = np.asarray(inertia_primary,   dtype=float) if inertia_primary   is not None else None
        obj._IB = np.asarray(inertia_secondary, dtype=float) if inertia_secondary is not None else None
        obj._flyby_position = np.asarray(flyby_position, dtype=float) if flyby_position is not None else None
        obj._flyby_velocity = np.asarray(flyby_velocity, dtype=float) if flyby_velocity is not None else None
        obj._solar_position = np.asarray(solar_position, dtype=float) if solar_position is not None else None
        obj._solar_velocity = np.asarray(solar_velocity, dtype=float) if solar_velocity is not None else None
        # potential_energy from _from_arrays is already in SI (J) — no conversion needed
        obj._potential        = np.asarray(potential_energy, dtype=float) if potential_energy is not None else None
        obj._energy           = None
        obj._angular_momentum = None
        return obj

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        t_span = (self._times[-1] - self._times[0]) / 86400.0 if len(self._times) > 1 else 0
        return (f"SimulationResults(n_steps={self.n_steps}, "
                f"duration={t_span:.2f} days, "
                f"|r|_mean={self.separation.mean():.0f} m)")
