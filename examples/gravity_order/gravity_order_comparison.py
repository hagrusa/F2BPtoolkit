"""
Gravity order comparison: 2nd vs 4th order mutual potential
============================================================

Runs two identical Didymos–Dimorphos simulations differing only in the
truncation order of the mutual gravitational potential expansion:

  * Order 2 — quadrupole (J₂ / C₂₂) terms
  * Order 4 — additionally includes hexadecapole terms

Note on higher orders: for uniform ellipsoidal bodies the inertia integral
tensors (Hou 2016, Eq. 9) are fully specified by the 2nd and 4th-order shape
moments.  Orders 6 and 8 are analytically zero for this geometry and are
therefore not included.  For polyhedral shape models the full tensor would be
populated and higher orders would be meaningful.

Two figures are produced:

  Figure 1 — Conservation diagnostics
    Normalised total energy and angular momentum errors for both runs.

  Figure 2 — Orbital and rotational dynamics (order-4 vs order-2)
    * Orbital phase difference  (true anomaly, geometric)
    * 3-D position difference   |r₄ − r₂|  in inertial frame
    * Dimorphos spin rate
    * Libration angle  (yaw in orbital / Hill frame)
"""

import numpy as np
import matplotlib.pyplot as plt

import f2bptoolkit as f2bp

# ── System parameters ──────────────────────────────────────────────────────────

G = 6.674e-11   # m³ kg⁻¹ s⁻²

rhoA       = 2170.0
aA, bA, cA = 400.0, 395.0, 340.0   # m

rhoB       = 2400.0
aB, bB, cB = 85.0, 73.0, 63.0   # m

MA = (4./3.) * np.pi * aA * bA * cA * rhoA
MB = (4./3.) * np.pi * aB * bB * cB * rhoB

r0     = 1195.0
v_circ = np.sqrt(G * (MA + MB) / r0)
n_orb  = np.sqrt(G * (MA + MB) / r0**3)
T_orb  = 2 * np.pi / n_orb

T_didymos  = 2.26 * 3600.0
w_primary  = 2 * np.pi / T_didymos
w_sync     = n_orb

print(f"Primary mass:    {MA:.4e} kg")
print(f"Secondary mass:  {MB:.4e} kg")
print(f"Orbital period:  {T_orb/3600:.3f} h")

# ── Integration settings ───────────────────────────────────────────────────────

N_PERIODS = 10
t_final   = N_PERIODS * T_orb
dt        = 60.0
nOut      = 10

print(f"\nIntegrating {N_PERIODS} orbits ({t_final/86400:.2f} days), "
      f"dt={dt:.0f} s, nOut={nOut}")


def _build_and_run(order: int):
    sim = f2bp.Simulation()
    sim.gravity_order = order

    primary = f2bp.Body("Didymos")
    primary.shape   = f2bp.EllipsoidShape(a=aA, b=bA, c=cA)
    primary.density = rhoA
    sim.add(primary)

    secondary = f2bp.Body("Dimorphos")
    secondary.shape   = f2bp.EllipsoidShape(a=aB, b=bB, c=cB)
    secondary.density = rhoB
    sim.add(secondary)

    sim.set_state(
        position        = [r0, 0.0, 0.0],
        velocity        = [0.0, v_circ, 0.0],
        omega_primary   = [0.0, 0.0, w_primary],
        omega_secondary = [0.0, 0.0, w_sync],
    )

    results = sim.integrate(
        t_final    = t_final,
        integrator = f2bp.LGVI(dt=dt),
        nOut       = nOut,
    )
    print(f"  order={order}: {results.n_steps} output steps")
    return results, sim


print("\nRunning order-2 simulation ...")
res2, sim2 = _build_and_run(2)

print("Running order-4 simulation ...")
res4, sim4 = _build_and_run(4)

t_days = res2.times / 86400.0   # identical grid for both

# ── Conservation diagnostics ───────────────────────────────────────────────────

_, dE2 = sim2.analysis.energy_conservation()
_, dH2 = sim2.analysis.angular_momentum_conservation()
_, dE4 = sim4.analysis.energy_conservation()
_, dH4 = sim4.analysis.angular_momentum_conservation()

# ── Orbital phase (true anomaly, geometric) ───────────────────────────────────
# Position is stored in the A (primary body) frame.  Transform to inertial
# frame N, then project onto the initial orbital plane and compute the
# azimuthal angle from the initial position direction.

def _orbital_phase(res) -> np.ndarray:
    """Return unwrapped orbital phase [rad] in the inertial frame."""
    r_N = np.einsum('nij,nj->ni', res.A_to_N, res.position)
    v_N = np.einsum('nij,nj->ni', res.A_to_N, res.velocity)

    h0   = np.cross(r_N[0], v_N[0])
    h0  /= np.linalg.norm(h0)
    e_r  = r_N[0] / np.linalg.norm(r_N[0])
    e_th = np.cross(h0, e_r)

    return np.unwrap(np.arctan2(r_N @ e_th, r_N @ e_r))


phi2 = _orbital_phase(res2)
phi4 = _orbital_phase(res4)
dphi = np.degrees(phi4 - phi2)   # order-4 minus order-2

# ── 3-D position difference (inertial frame) ──────────────────────────────────

def _pos_inertial(res) -> np.ndarray:
    return np.einsum('nij,nj->ni', res.A_to_N, res.position)

dr = np.linalg.norm(_pos_inertial(res4) - _pos_inertial(res2), axis=1)   # m

# ── Dimorphos spin rate ───────────────────────────────────────────────────────

ws2 = np.degrees(np.linalg.norm(res2.omega_secondary, axis=1))   # deg/s
ws4 = np.degrees(np.linalg.norm(res4.omega_secondary, axis=1))

# ── Libration angle (yaw in orbital / Hill frame) ────────────────────────────

_, _, yaw2 = res2.secondary_euler_angles(frame='orbital')
_, _, yaw4 = res4.secondary_euler_angles(frame='orbital')

# ── Summary printout ──────────────────────────────────────────────────────────

print(f"\n{'Quantity':<38}  {'Order 2':>10}  {'Order 4':>10}")
print("─" * 62)
print(f"{'Max |dE/E₀|':<38}  {np.max(np.abs(dE2)):>10.2e}  {np.max(np.abs(dE4)):>10.2e}")
print(f"{'Max |d|H|/|H₀||':<38}  {np.max(np.abs(dH2)):>10.2e}  {np.max(np.abs(dH4)):>10.2e}")
print(f"{'Max |Δr| [m]':<38}  {'—':>10}  {dr.max():>10.4f}")
print(f"{'Max |Δφ| [deg]':<38}  {'—':>10}  {np.abs(dphi).max():>10.6f}")
print(f"{'Yaw amplitude [deg]':<38}  {np.abs(yaw2).max():>10.4f}  {np.abs(yaw4).max():>10.4f}")
print(f"{'Mean |ωs| [deg/s]':<38}  {ws2.mean():>10.6f}  {ws4.mean():>10.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Conservation diagnostics
# ══════════════════════════════════════════════════════════════════════════════

COL2 = "steelblue"
COL4 = "coral"
LW   = 0.9

print("\nPlotting Fig 1: conservation diagnostics ...")
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4))

ax = axes1[0]
ax.plot(t_days, dE2, lw=LW, color=COL2, label="Order 2 (quadrupole)")
ax.plot(t_days, dE4, lw=LW, color=COL4, ls="--", label="Order 4 (hexadecapole)")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]")
ax.set_ylabel("(E − E₀) / |E₀|")
ax.set_title("Total Energy Conservation Error")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=9)

ax = axes1[1]
ax.plot(t_days, dH2, lw=LW, color=COL2, label="Order 2 (quadrupole)")
ax.plot(t_days, dH4, lw=LW, color=COL4, ls="--", label="Order 4 (hexadecapole)")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]")
ax.set_ylabel("(|H| − |H₀|) / |H₀|")
ax.set_title("Angular Momentum Conservation Error")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=9)

fig1.suptitle("Conservation Diagnostics: Order 2 vs Order 4", fontsize=13, fontweight="bold")
fig1.tight_layout()
fig1.savefig("cmp_fig1_conservation.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Orbital and rotational dynamics
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 2: orbital and rotational dynamics ...")
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))

# ── Orbital phase difference ──────────────────────────────────────────────────
ax = axes2[0, 0]
ax.plot(t_days, dphi, lw=LW, color=COL4)
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Δφ [deg]")
ax.set_title("Orbital Phase Difference  (Order 4 − Order 2)")

# ── 3-D position difference ───────────────────────────────────────────────────
ax = axes2[0, 1]
ax.plot(t_days, dr, lw=LW, color=COL4)
ax.set_xlabel("Time [days]")
ax.set_ylabel("|Δr| [m]")
ax.set_title("3-D Position Difference  (Order 4 − Order 2, inertial frame)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# ── Dimorphos spin rate ───────────────────────────────────────────────────────
ax = axes2[1, 0]
ax.plot(t_days, ws2, lw=LW, color=COL2, label="Order 2 (quadrupole)")
ax.plot(t_days, ws4, lw=LW, color=COL4, ls="--", label="Order 4 (hexadecapole)")
ax.set_xlabel("Time [days]")
ax.set_ylabel("|ωs| [deg/s]")
ax.set_title("Dimorphos Spin Rate")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=9)

# ── Libration angle ───────────────────────────────────────────────────────────
ax = axes2[1, 1]
ax.plot(t_days, yaw2, lw=LW, color=COL2, label="Order 2 (quadrupole)")
ax.plot(t_days, yaw4, lw=LW, color=COL4, ls="--", label="Order 4 (hexadecapole)")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Yaw [deg]")
ax.set_title("Libration Angle  (yaw in Hill frame)")
ax.legend(fontsize=9)

fig2.suptitle("Orbital and Rotational Dynamics: Order 2 vs Order 4", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("cmp_fig2_dynamics.png", dpi=150, bbox_inches="tight")

print("\nFigures saved:")
print("  cmp_fig1_conservation.png")
print("  cmp_fig2_dynamics.png")
