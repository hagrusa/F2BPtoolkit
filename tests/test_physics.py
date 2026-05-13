"""
Physics-invariant tests for F2BPtoolkit.

All tests require the compiled C++ extension and are marked
@pytest.mark.integration.  They verify physical correctness rather than
API behaviour:

  - Basic integration smoke tests (output shapes, masses, integrator agreement)
  - Rotation matrix orthogonality throughout integration
  - Angular momentum conservation (magnitude and direction)
  - Kepler limit (gravity_order=0, spheres → exact two-body dynamics)
  - Energy conservation comparisons across integrators
  - nOut output-step count and time-spacing correctness
  - Synchronous rotation boundedness (orbital-frame yaw libration)
"""

import numpy as np
import pytest

import f2bptoolkit as f2bp
from f2bptoolkit.integrators import RK4, LGVI, ABM


# ── shared parameters (Didymos–Dimorphos, approximate) ────────────────────────

_G    = 6.674e-11
_rhoA = 2170.0;  _aA, _bA, _cA = 400.0, 395.0, 340.0
_rhoB = 2400.0;  _aB, _bB, _cB =  85.0,  73.0,  63.0
_MA   = (4/3) * np.pi * _aA * _bA * _cA * _rhoA
_MB   = (4/3) * np.pi * _aB * _bB * _cB * _rhoB
_r0   = 1195.0
_v_c  = np.sqrt(_G * (_MA + _MB) / _r0)
_n    = np.sqrt(_G * (_MA + _MB) / _r0**3)
_T    = 2 * np.pi / _n   # approximate orbital period in seconds


def _ellipsoid_sim(gravity_order=2, dt=10.0, nOut=1, n_periods=1, eccentricity=0.0):
    """
    Didymos–Dimorphos with ellipsoidal shapes.

    *r0* is the periapsis distance.  For eccentricity > 0 the initial velocity
    is the periapsis velocity so that the orbit has the requested eccentricity.
    """
    GM    = _G * (_MA + _MB)
    a_orb = _r0 / (1.0 - eccentricity)          # semi-major axis
    v_peri = np.sqrt(GM * (1.0 + eccentricity) / _r0)   # periapsis velocity
    T_orb = 2 * np.pi * np.sqrt(a_orb**3 / GM)

    sim = f2bp.Simulation()
    sim.gravity_order = gravity_order
    p = f2bp.Body("Didymos")
    p.shape   = f2bp.EllipsoidShape(_aA, _bA, _cA)
    p.density = _rhoA
    sim.add(p)
    s = f2bp.Body("Dimorphos")
    s.shape   = f2bp.EllipsoidShape(_aB, _bB, _cB)
    s.density = _rhoB
    sim.add(s)
    sim.set_state(
        position=[_r0, 0.0, 0.0],
        velocity=[0.0, v_peri, 0.0],
        omega_primary=[0.0, 0.0, _n],
        omega_secondary=[0.0, 0.0, _n],
    )
    return sim.integrate(
        t_final=n_periods * T_orb,
        integrator=RK4(dt=dt),
        nOut=nOut,
    )


def _sphere_sim(gravity_order=0, dt=10.0, nOut=1, n_periods=1, eccentricity=0.0):
    """
    Point-mass (sphere) two-body sim — admits analytical Keplerian solution.

    *r0* is the periapsis distance.  The initial velocity is set to the
    periapsis velocity for the requested eccentricity.

    Returns (results, T, r0) where T is the true orbital period.
    """
    rhoA = 2170.0; rA = 400.0
    rhoB = 2400.0; rB =  85.0
    MA   = (4/3) * np.pi * rA**3 * rhoA
    MB   = (4/3) * np.pi * rB**3 * rhoB
    r0   = 1195.0
    GM   = _G * (MA + MB)
    a    = r0 / (1.0 - eccentricity)            # semi-major axis
    v_peri = np.sqrt(GM * (1.0 + eccentricity) / r0)   # periapsis velocity
    T    = 2 * np.pi * np.sqrt(a**3 / GM)       # true orbital period
    n_spin = np.sqrt(GM / r0**3)                # approximate spin rate

    sim = f2bp.Simulation()
    sim.gravity_order = gravity_order
    p = f2bp.Body("P"); p.shape = f2bp.SphereShape(rA); p.density = rhoA
    s = f2bp.Body("S"); s.shape = f2bp.SphereShape(rB); s.density = rhoB
    sim.add(p); sim.add(s)
    sim.set_state(
        position=[r0, 0.0, 0.0],
        velocity=[0.0, v_peri, 0.0],
        omega_primary=[0.0, 0.0, n_spin],
        omega_secondary=[0.0, 0.0, n_spin],
    )
    # Snap t_final to an exact integer number of dt steps so the integration
    # endpoint lands precisely at N periods.
    t_final = n_periods * T
    n_steps = int(round(t_final / dt))
    t_final_exact = n_steps * dt

    return sim.integrate(
        t_final=t_final_exact,
        integrator=RK4(dt=dt),
        nOut=nOut,
    ), T, r0


# ══════════════════════════════════════════════════════════════════════════════
# Rotation matrix orthogonality
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRotationMatrixOrthogonality:
    """A_to_N and B_to_A must remain proper rotation matrices throughout."""

    def _check_orthogonal(self, R_series, atol=1e-12, label="R"):
        """Assert R @ R.T ≈ I for every timestep."""
        I_batch = np.einsum('nij,nkj->nik', R_series, R_series)
        eye     = np.broadcast_to(np.eye(3), I_batch.shape)
        err     = np.max(np.abs(I_batch - eye))
        assert err < atol, (
            f"{label}: max |R@R.T - I| = {err:.2e} > {atol:.2e}"
        )

    def test_A_to_N_orthogonal_rk4(self):
        r = _ellipsoid_sim(dt=10.0, nOut=5, n_periods=2)
        self._check_orthogonal(r.A_to_N, atol=1e-12, label="A_to_N (RK4)")

    def test_B_to_A_orthogonal_rk4(self):
        r = _ellipsoid_sim(dt=10.0, nOut=5, n_periods=2)
        self._check_orthogonal(r.B_to_A, atol=1e-12, label="B_to_A (RK4)")

    def test_A_to_N_orthogonal_lgvi(self):
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(2*_T, integrator=LGVI(dt=10.0), nOut=5)
        self._check_orthogonal(r.A_to_N, atol=1e-12, label="A_to_N (LGVI)")

    def test_B_to_A_orthogonal_lgvi(self):
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(2*_T, integrator=LGVI(dt=10.0), nOut=5)
        self._check_orthogonal(r.B_to_A, atol=1e-12, label="B_to_A (LGVI)")

    def test_A_to_N_det_is_plus_one(self):
        """Determinant must be +1 (proper rotation, not a reflection)."""
        r = _ellipsoid_sim(dt=10.0, nOut=10, n_periods=1)
        dets = np.linalg.det(r.A_to_N)
        np.testing.assert_allclose(dets, 1.0, atol=1e-12)

    def test_B_to_A_det_is_plus_one(self):
        r = _ellipsoid_sim(dt=10.0, nOut=10, n_periods=1)
        dets = np.linalg.det(r.B_to_A)
        np.testing.assert_allclose(dets, 1.0, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Angular momentum conservation
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestAngularMomentumConservation:
    """Total angular momentum magnitude and direction should be constant."""

    def test_magnitude_conserved_rk4(self):
        """RK4: |H| drift < 1e-12 over 5 orbits."""
        r = _ellipsoid_sim(dt=10.0, nOut=10, n_periods=5)
        H = r.angular_momentum_magnitude
        drift = np.max(np.abs(H - H[0])) / H[0]
        assert drift < 1e-12, f"|H| drift = {drift:.2e}"

    def test_magnitude_conserved_lgvi(self):
        """LGVI: |H| drift < 1e-9 over 5 orbits."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(5*_T, integrator=LGVI(dt=10.0), nOut=10)
        H = r.angular_momentum_magnitude
        drift = np.max(np.abs(H - H[0])) / H[0]
        assert drift < 1e-9, f"LGVI |H| drift = {drift:.2e}"

    def test_magnitude_conserved_abm(self):
        """ABM: |H| drift < 1e-12 over 5 orbits."""
        r = _ellipsoid_sim(dt=10.0, nOut=10, n_periods=5)
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(5*_T, integrator=ABM(dt=10.0), nOut=10)
        H = r.angular_momentum_magnitude
        drift = np.max(np.abs(H - H[0])) / H[0]
        assert drift < 1e-12, f"ABM |H| drift = {drift:.2e}"

    def test_direction_conserved_rk4(self):
        """H direction should not deviate more than 1e-6 degrees over 5 orbits."""
        r = _ellipsoid_sim(dt=10.0, nOut=10, n_periods=5)
        H = r.angular_momentum      # (N, 3)
        H_hat = H / np.linalg.norm(H, axis=1, keepdims=True)
        cos_angles = np.clip(H_hat @ H_hat[0], -1, 1)
        max_angle_deg = np.degrees(np.arccos(np.min(cos_angles)))
        assert max_angle_deg < 1e-6, f"H direction drift = {max_angle_deg:.2e} deg"

    def test_analysis_angular_momentum_conservation_rk4(self):
        """analysis.angular_momentum_conservation() relative error < 1e-9."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        sim.integrate(3*_T, integrator=RK4(dt=10.0), nOut=10)
        _, dH = sim.analysis.angular_momentum_conservation()
        assert np.max(np.abs(dH)) < 1e-9

    def test_analysis_angular_momentum_conservation_lgvi(self):
        """LGVI: analysis.angular_momentum_conservation() relative error < 1e-9."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        sim.integrate(3*_T, integrator=LGVI(dt=10.0), nOut=10)
        _, dH = sim.analysis.angular_momentum_conservation()
        assert np.max(np.abs(dH)) < 1e-9

    def test_analysis_angular_momentum_conservation_abm(self):
        """ABM: analysis.angular_momentum_conservation() relative error < 1e-9."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        sim.integrate(3*_T, integrator=ABM(dt=10.0), nOut=10)
        _, dH = sim.analysis.angular_momentum_conservation()
        assert np.max(np.abs(dH)) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# Energy conservation
# ══════════════════════════════════════════════════════════════════════════════

def _total_energy(results, MA, MB):
    """
    Total mechanical energy: KE_orbital + KE_rot_primary + KE_rot_secondary
    + PE, where PE = -G*MA*MB/|r| (point-mass mutual potential).

    Valid for gravity_order=0 with spherical bodies.  For extended bodies the
    true PE includes higher-order inertia-integral terms not available from
    Python, so this helper should only be used with the Kepler (sphere) case.
    """
    PE = -_G * MA * MB / results.separation
    return (results.kinetic_energy_orbital
            + results.kinetic_energy_rotation_primary
            + results.kinetic_energy_rotation_secondary
            + PE)


@pytest.mark.integration
class TestEnergyConservation:
    """
    Total energy (KE + PE) must be conserved.  KE alone is NOT conserved
    because it exchanges with the mutual gravitational potential.

    These tests use spherical bodies with gravity_order=0 (Kepler limit)
    so that PE = -G*M1*M2/r is exact.
    """

    # Sphere masses matching _sphere_sim()
    _rhoA_s = 2170.0; _rA = 400.0
    _rhoB_s = 2400.0; _rB =  85.0
    _MA_s = (4/3) * np.pi * _rA**3 * _rhoA_s
    _MB_s = (4/3) * np.pi * _rB**3 * _rhoB_s

    def test_total_energy_conserved_rk4(self):
        """RK4: total energy (KE + PE) drift < 0.01% over 5 orbits."""
        r, T, r0 = _sphere_sim(gravity_order=0, dt=5.0, nOut=5, n_periods=5)
        E = _total_energy(r, self._MA_s, self._MB_s)
        drift = abs(E[-1] - E[0]) / abs(E[0])
        assert drift < 1e-9, f"RK4 total energy drift = {drift:.2e}"

    def test_total_energy_conserved_abm(self):
        """ABM: total energy (KE + PE) drift < 0.01% over 5 orbits."""
        rhoA = self._rhoA_s; rA = self._rA
        rhoB = self._rhoB_s; rB = self._rB
        MA = (4/3)*np.pi*rA**3*rhoA
        MB = (4/3)*np.pi*rB**3*rhoB
        r0 = 1195.0
        v_c = np.sqrt(_G*(MA+MB)/r0)
        n   = np.sqrt(_G*(MA+MB)/r0**3)
        T   = 2*np.pi/n

        sim = f2bp.Simulation()
        sim.gravity_order = 0
        p = f2bp.Body("P"); p.shape = f2bp.SphereShape(rA); p.density = rhoA
        s = f2bp.Body("S"); s.shape = f2bp.SphereShape(rB); s.density = rhoB
        sim.add(p); sim.add(s)
        sim.set_state([r0,0,0],[0,v_c,0],[0,0,n],[0,0,n])
        r = sim.integrate(5*T, integrator=ABM(dt=5.0), nOut=5)
        E = _total_energy(r, MA, MB)
        drift = abs(E[-1] - E[0]) / abs(E[0])
        assert drift < 1e-9, f"ABM total energy drift = {drift:.2e}"

    def test_total_energy_conserved_lgvi(self):
        """
        LGVI: max total energy (KE + PE) deviation < 1e-6 over 5 orbits.

        LGVI is a symplectic integrator — it exactly conserves a modified
        Hamiltonian, so the true energy oscillates but does not drift secularly.
        The threshold is looser than RK4/ABM to account for this bounded
        oscillation; the key property checked is absence of secular growth.
        """
        rhoA = self._rhoA_s; rA = self._rA
        rhoB = self._rhoB_s; rB = self._rB
        MA = (4/3)*np.pi*rA**3*rhoA
        MB = (4/3)*np.pi*rB**3*rhoB
        r0 = 1195.0
        v_c = np.sqrt(_G*(MA+MB)/r0)
        n   = np.sqrt(_G*(MA+MB)/r0**3)
        T   = 2*np.pi/n

        sim = f2bp.Simulation()
        sim.gravity_order = 0
        p = f2bp.Body("P"); p.shape = f2bp.SphereShape(rA); p.density = rhoA
        s = f2bp.Body("S"); s.shape = f2bp.SphereShape(rB); s.density = rhoB
        sim.add(p); sim.add(s)
        sim.set_state([r0,0,0],[0,v_c,0],[0,0,n],[0,0,n])
        r = sim.integrate(5*T, integrator=LGVI(dt=5.0), nOut=5)
        E = _total_energy(r, MA, MB)
        drift = np.max(np.abs(E - E[0])) / abs(E[0])
        assert drift < 1e-6, f"LGVI total energy max deviation = {drift:.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# Kepler limit
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestKeplerLimit:
    """
    With gravity_order=0 and spherical bodies the dynamics reduce to the
    classical two-body problem.  Analytical Keplerian predictions must hold.
    """

    def test_circular_orbit_separation_constant(self):
        """Separation should vary by < 0.1% for a near-circular orbit."""
        r, T, r0 = _sphere_sim(gravity_order=0, dt=5.0, nOut=5, n_periods=1)
        sep = r.separation
        variation = (sep.max() - sep.min()) / sep.mean()
        assert variation < 1e-9, f"Separation variation = {variation:.2e}"

    def test_orbital_period_matches_kepler(self):
        """
        After exactly one Kepler period the separation should return to r0
        within 0.1%.  Use e=0.1 for a general elliptic orbit.
        """
        r, T, r0 = _sphere_sim(gravity_order=0, dt=5.0, nOut=1, n_periods=1, eccentricity=0.1)
        sep_final   = r.separation[-1]
        sep_initial = r.separation[0]
        err = abs(sep_final - sep_initial) / sep_initial
        assert err < 1e-9, (
            f"Separation at t=T: {sep_final:.2f} m vs initial {sep_initial:.2f} m "
            f"(err={err:.2e})"
        )

    def test_semi_major_axis_conserved(self):
        """
        Vis-viva: a = -GM/(v²  - 2GM/r).  For a Keplerian orbit the
        semi-major axis should be constant throughout.  Use e=0.1 so that
        a ≠ r0 and the variation is non-trivially tested.
        """
        r, T, r0 = _sphere_sim(gravity_order=0, dt=5.0, nOut=5, n_periods=2, eccentricity=0.1)
        rhoA = 2170.0; rA = 400.0
        rhoB = 2400.0; rB =  85.0
        MA   = (4/3) * np.pi * rA**3 * rhoA
        MB   = (4/3) * np.pi * rB**3 * rhoB
        GM   = _G * (MA + MB)

        sep  = r.separation                                    # (N,) m
        v_sq = np.sum(r.velocity**2, axis=1)                  # (N,) m²/s²
        a    = 1.0 / (2.0/sep - v_sq/GM)                      # vis-viva: a in m

        # Semi-major axis should be constant to within 0.01%
        variation = (a.max() - a.min()) / a.mean()
        assert variation < 1e-9, f"Semi-major axis variation = {variation:.2e}"

    def test_eccentricity_conserved(self):
        """Eccentricity vector magnitude should be constant for Keplerian orbit.
        Use e=0.1 so the eccentricity magnitude is well away from zero and the
        relative variation is well-defined (avoids divide-by-zero at e≈0)."""
        r, T, r0 = _sphere_sim(gravity_order=0, dt=5.0, nOut=5, n_periods=2, eccentricity=0.1)
        # Build eccentricity vector from first principles in A frame
        # (at t=0, A=N so the frame doesn't matter)
        rhoA = 2170.0; rA = 400.0
        rhoB = 2400.0; rB =  85.0
        MA   = (4/3) * np.pi * rA**3 * rhoA
        MB   = (4/3) * np.pi * rB**3 * rhoB
        GM   = _G * (MA + MB)

        pos  = r.position    # (N, 3) m, in A frame
        vel  = r.velocity    # (N, 3) m/s

        sep  = np.linalg.norm(pos, axis=1, keepdims=True)
        v_sq = np.sum(vel**2, axis=1, keepdims=True)
        e_vec = (v_sq/GM - 1.0/sep) * pos - np.sum(pos*vel, axis=1, keepdims=True)/GM * vel
        ecc = np.linalg.norm(e_vec, axis=1)

        variation = (ecc.max() - ecc.min()) / (ecc.mean() + 1e-12)
        assert variation < 1e-9, f"Eccentricity variation = {variation:.2e}"

    def test_point_mass_vs_extended_body_differ(self):
        """
        gravity_order=0 (point mass) and gravity_order=2 (extended) should
        give measurably different separations over 5 orbits — confirms
        gravity_order is actually used.
        """
        r0_pm, T, _ = _sphere_sim(gravity_order=0, dt=10.0, nOut=1, n_periods=5)
        r0_ext      = _ellipsoid_sim(gravity_order=2, dt=10.0, nOut=1, n_periods=5)
        # The final separations should differ by more than numerical noise
        diff = abs(r0_pm.separation[-1] - r0_ext.separation[-1])
        assert diff > 1.0, f"Point-mass and extended-body orbits too similar (diff={diff:.3f} m)"


# ══════════════════════════════════════════════════════════════════════════════
# nOut output-step count and time spacing
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestNOutBehavior:

    def test_nOut_1_gives_all_steps(self):
        """nOut=1 should yield ⌊t_final/dt⌋ + 1 steps (every step recorded)."""
        dt = 10.0
        t_final = 1000.0
        r = _ellipsoid_sim(dt=dt, nOut=1, n_periods=0)
        # Rerun with known t_final
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(t_final, integrator=RK4(dt=dt), nOut=1)
        expected = int(t_final / dt) + 1
        assert r.n_steps == expected

    def test_nOut_10_gives_every_10th_step(self):
        """nOut=10 should yield roughly 1/10 the steps of nOut=1."""
        t_final = 1000.0
        dt = 10.0

        def _run(nOut):
            sim = f2bp.Simulation()
            sim.gravity_order = 2
            p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
            s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
            sim.add(p); sim.add(s)
            sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
            return sim.integrate(t_final, integrator=RK4(dt=dt), nOut=nOut)

        r1  = _run(nOut=1)
        r10 = _run(nOut=10)
        expected = int(t_final / dt / 10) + 1
        assert r10.n_steps == expected

    def test_nOut_output_times_spaced_by_nOut_dt(self):
        """Time gaps between outputs should equal nOut * dt."""
        t_final = 500.0
        dt      = 10.0
        nOut    = 5

        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(t_final, integrator=RK4(dt=dt), nOut=nOut)

        diffs = np.diff(r.times)
        np.testing.assert_allclose(diffs, nOut * dt, rtol=1e-9)

    def test_nOut_times_start_at_zero(self):
        """First output time should be 0.0."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(500.0, integrator=RK4(dt=10.0), nOut=5)
        assert r.times[0] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Synchronous rotation — orbital-frame yaw must remain bounded
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestSynchronousRotation:
    """
    A body started in the 1:1 spin-orbit resonance (synchronous rotation)
    should librate, not drift, in the orbital frame.  The orbital-frame yaw
    of the secondary must remain bounded over many orbital periods.
    """

    def test_orbital_frame_yaw_bounded(self):
        """
        Orbital-frame yaw of Dimorphos should stay within ±90° over 10 orbits
        when started approximately synchronous.
        """
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("Didymos")
        p.shape   = f2bp.EllipsoidShape(_aA, _bA, _cA)
        p.density = _rhoA
        sim.add(p)
        s = f2bp.Body("Dimorphos")
        s.shape   = f2bp.EllipsoidShape(_aB, _bB, _cB)
        s.density = _rhoB
        sim.add(s)
        sim.set_state(
            position=[_r0, 0.0, 0.0],
            velocity=[0.0, _v_c, 0.0],
            omega_primary=[0.0, 0.0, _n],
            omega_secondary=[0.0, 0.0, _n],
        )
        results = sim.integrate(10 * _T, integrator=RK4(dt=10.0), nOut=20)

        _, _, yaw = results.secondary_euler_angles(frame='orbital')
        max_yaw = np.max(np.abs(yaw))
        assert max_yaw < 90.0, (
            f"Orbital-frame yaw exceeded 90°: max={max_yaw:.2f}°. "
            "Spin-orbit coupling may be broken."
        )

    def test_inertial_frame_yaw_drifts_secularly(self):
        """
        Inertial-frame yaw should accumulate secular drift (Dimorphos orbits
        the primary) — confirms the two frames are genuinely different.
        """
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("Didymos")
        p.shape   = f2bp.EllipsoidShape(_aA, _bA, _cA)
        p.density = _rhoA
        sim.add(p)
        s = f2bp.Body("Dimorphos")
        s.shape   = f2bp.EllipsoidShape(_aB, _bB, _cB)
        s.density = _rhoB
        sim.add(s)
        sim.set_state(
            position=[_r0, 0.0, 0.0],
            velocity=[0.0, _v_c, 0.0],
            omega_primary=[0.0, 0.0, _n],
            omega_secondary=[0.0, 0.0, _n],
        )
        results = sim.integrate(10 * _T, integrator=RK4(dt=10.0), nOut=20)

        _, _, yaw_N = results.secondary_euler_angles(frame='inertial')
        total_drift = abs(yaw_N[-1] - yaw_N[0])
        # Should drift by approximately 10 × 360° = 3600° over 10 orbits
        assert total_drift > 100.0, (
            f"Inertial yaw drift only {total_drift:.1f}°; expected secular accumulation."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Basic integration smoke tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """
    Sanity checks: do integrators run, produce correct output shapes, agree
    with each other, and extract sensible physical quantities?
    """

    def test_rk4_orbit_is_circular(self):
        """Separation ratio max/min < 1.1 for a near-circular orbit."""
        r = _ellipsoid_sim(dt=10.0, nOut=50, n_periods=1)
        sep = r.separation
        assert sep.max() / sep.min() < 1.1

    def test_result_shapes(self):
        """All output arrays have consistent shape (N, ...)."""
        r = _ellipsoid_sim(dt=60.0, nOut=10, n_periods=1)
        N = r.n_steps
        assert r.times.shape            == (N,)
        assert r.position.shape         == (N, 3)
        assert r.velocity.shape         == (N, 3)
        assert r.omega_primary.shape    == (N, 3)
        assert r.omega_secondary.shape  == (N, 3)
        assert r.A_to_N.shape           == (N, 3, 3)
        assert r.B_to_A.shape           == (N, 3, 3)

    def test_masses_positive(self):
        """Masses computed from inertia integrals must be positive."""
        r = _ellipsoid_sim(dt=10.0, nOut=1, n_periods=0)
        # Rerun with tiny t_final just to get masses
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(100.0, integrator=RK4(dt=10.0))
        assert r._Mc > 0
        assert r._Ms > 0

    def test_lgvi_runs(self):
        """LGVI integrator completes without error and returns results."""
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
        sim.add(p); sim.add(s)
        sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
        r = sim.integrate(500.0, integrator=LGVI(dt=10.0), nOut=5)
        assert r.n_steps >= 2

    def test_abm_agrees_with_rk4(self):
        """ABM and RK4 final separation agree to within 1%."""
        def _run(integrator):
            sim = f2bp.Simulation()
            sim.gravity_order = 2
            p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(_aA,_bA,_cA); p.density = _rhoA
            s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(_aB,_bB,_cB); s.density = _rhoB
            sim.add(p); sim.add(s)
            sim.set_state([_r0,0,0],[0,_v_c,0],[0,0,_n],[0,0,_n])
            return sim.integrate(_T, integrator=integrator)

        r_rk4 = _run(RK4(dt=10.0))
        r_abm  = _run(ABM(dt=10.0))
        diff = abs(r_rk4.separation[-1] - r_abm.separation[-1]) / r_rk4.separation[-1]
        assert diff < 0.01, f"RK4 vs ABM separation diff = {diff:.2e}"
