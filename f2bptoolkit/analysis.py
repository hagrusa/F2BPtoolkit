"""Analysis tools for F2BP simulation results."""

import numpy as np
from typing import Optional, Tuple
from .results import SimulationResults


class AnalysisInterface:
    """
    Analysis methods attached to simulation results.

    Accessed as ``sim.analysis`` after running ``sim.integrate()``.
    """

    def __init__(self, results: SimulationResults):
        self._r = results

    def energy_conservation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalised total energy conservation error (dE/E0).

        Total energy = orbital kinetic + rotational kinetic (both bodies)
        + mutual gravitational potential.

        Returns
        -------
        times : ndarray
        dE_over_E0 : ndarray
            (E(t) - E(t0)) / |E(t0)|
        """
        r = self._r
        Kt = r.kinetic_energy_orbital
        Kr1 = r.kinetic_energy_rotation_primary
        Kr2 = r.kinetic_energy_rotation_secondary
        U = r.potential_energy
        if Kt is None:
            raise ValueError("Masses not available; run integrate() first.")
        if U is None:
            raise ValueError("Potential energy not available; run integrate() first.")

        E = Kt + Kr1 + Kr2 + U
        E0 = E[0]
        return r.times, (E - E0) / abs(E0)

    def angular_momentum_conservation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalised angular momentum conservation error (d|H|/|H0|).

        Returns
        -------
        times : ndarray
        dH_over_H0 : ndarray
        """
        r = self._r
        H_mag = r.angular_momentum_magnitude
        if H_mag is None:
            raise ValueError("Inertia tensors not available; run integrate() first.")
        H0 = H_mag[0]
        return r.times, (H_mag - H0) / H0

    def mean_separation(self) -> float:
        """Time-averaged separation |r| in meters."""
        return float(np.mean(self._r.separation))

    def orbital_period_estimate(self) -> Optional[float]:
        """
        Estimate the orbital period in seconds from the separation time series
        using a simple auto-correlation peak.

        Returns None if estimation fails.
        """
        sep = self._r.separation
        dt = float(np.median(np.diff(self._r.times)))
        N = len(sep)
        if N < 10:
            return None

        # Detrend and compute autocorrelation
        s = sep - np.mean(sep)
        ac = np.correlate(s, s, mode='full')
        ac = ac[N - 1:]   # keep lags >= 0
        ac /= ac[0]

        # Find first minimum then the next maximum (= period)
        from scipy.signal import argrelmax
        try:
            peaks, _ = argrelmax(ac, order=5)
            if len(peaks) > 0:
                return float(peaks[0]) * dt
        except Exception:
            pass
        return None

    def spin_rates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the magnitudes of primary and secondary spin rates over time.

        Returns
        -------
        omega_primary_mag : ndarray, shape (N,), rad/s
        omega_secondary_mag : ndarray, shape (N,), rad/s
        """
        wp = np.linalg.norm(self._r.omega_primary, axis=1)
        ws = np.linalg.norm(self._r.omega_secondary, axis=1)
        return wp, ws

    def mutual_inclination(self) -> np.ndarray:
        """
        Angle between the primary spin vector and the orbital angular momentum
        vector (in inertial frame), in radians, shape (N,).
        """
        r = self._r
        # Transform position and velocity from A frame to inertial frame N:
        # v_N = A_to_N @ v_A
        r_N = np.einsum('nij,nj->ni', r.A_to_N, r.position)
        v_N = np.einsum('nij,nj->ni', r.A_to_N, r.velocity)
        h_vec = np.cross(r_N, v_N, axis=1)

        # Primary spin direction in N: v_N = A_to_N @ v_A
        wc_N = np.einsum('nij,nj->ni', r.A_to_N, r.omega_primary)

        h_norm = np.linalg.norm(h_vec, axis=1, keepdims=True)
        w_norm = np.linalg.norm(wc_N, axis=1, keepdims=True)
        cos_theta = np.sum(h_vec * wc_N, axis=1) / (h_norm[:, 0] * w_norm[:, 0] + 1e-30)
        cos_theta = np.clip(cos_theta, -1, 1)
        return np.arccos(cos_theta)

    def eccentricity_vector(self) -> np.ndarray:
        """
        Laplace-Runge-Lenz (eccentricity) vector in inertial frame, shape (N, 3).

        Computed assuming a point-mass central body (valid when higher-order
        perturbations are small).
        """
        r = self._r
        if r._Mc is None:
            raise ValueError("Masses not available")
        mu = self._r._G * (r._Mc + r._Ms)

        # Transform to inertial frame:  v_N = A_to_N @ v_A
        r_I = np.einsum('nij,nj->ni', r.A_to_N, r.position)
        v_I = np.einsum('nij,nj->ni', r.A_to_N, r.velocity)

        sep = np.linalg.norm(r_I, axis=1, keepdims=True)
        h_I = np.cross(r_I, v_I, axis=1)
        e_vec = np.cross(v_I, h_I, axis=1) / mu - r_I / sep
        return e_vec

    def eccentricity(self) -> np.ndarray:
        """Instantaneous eccentricity magnitude, shape (N,)."""
        return np.linalg.norm(self.eccentricity_vector(), axis=1)
