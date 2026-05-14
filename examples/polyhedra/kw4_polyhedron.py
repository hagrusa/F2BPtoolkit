"""
1999 KW4 binary asteroid — polyhedral shape model example
==========================================================

Demonstrates using polyhedral (OBJ) shape models instead of ellipsoids.
The 1999 KW4 system consists of:

  Alpha (primary):   ~1.53 × 1.49 × 1.35 km,  rho ~ 1970 kg/m³
  Beta  (secondary): ~0.57 × 0.46 × 0.35 km,  rho ~ 2960 kg/m³
  Orbital separation: ~2.55 km,  period ~ 17.4 h

Shape models from Ostro et al. (2006), vertices in km.  The ``scale=1000``
argument to ``PolyhedronShape.from_obj`` converts them to metres before
passing to the integrator.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import f2bptoolkit as f2bp

# ── System parameters ──────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))

G    = 6.674e-11   # m³ kg⁻¹ s⁻²
rhoA = 1970.0      # kg/m³  (Ostro et al. 2006)
rhoB = 2960.0      # kg/m³

# Approximate masses 
# NOTE: mass is later computed internally based on shape model volumes and density
MA = 2.3548e12   # kg
MB = 1.4218e11   # kg

# Circular orbit initial conditions
r0     = 2550.0                           # m
v_circ = np.sqrt(G * (MA + MB) / r0)
n_orb  = np.sqrt(G * (MA + MB) / r0**3)
T_orb  = 2 * np.pi / n_orb

# Spin: Alpha rotates ~2.76 h, Beta is near-synchronous
T_alpha = 2.7645 * 3600.0
w_alpha = 2 * np.pi / T_alpha
w_sync  = n_orb

print(f"Primary mass:    {MA:.4e} kg")
print(f"Secondary mass:  {MB:.4e} kg")
print(f"Orbital period:  {T_orb/3600:.3f} h")
print(f"Alpha spin:      {T_alpha/3600:.4f} h  → ω = {w_alpha:.6f} rad/s")
print(f"Beta sync spin:  ω = {w_sync:.6f} rad/s")

# ── Build simulation ───────────────────────────────────────────────────────────

print("\nLoading polyhedral shape models ...")
shape_alpha = f2bp.PolyhedronShape.from_obj(
    os.path.join(HERE, "a66391_1999kw4_primary.obj"), scale=1000.0)
shape_beta  = f2bp.PolyhedronShape.from_obj(
    os.path.join(HERE, "a66391_1999kw4_secondary.obj"), scale=1000.0)

sim = f2bp.Simulation()
sim.gravity_order = 2

alpha = f2bp.Body("Alpha")
alpha.shape   = shape_alpha
alpha.density = rhoA
sim.add(alpha)

beta = f2bp.Body("Beta")
beta.shape   = shape_beta
beta.density = rhoB
sim.add(beta)

sim.set_state(
    position        = [r0, 0.0, 0.0],
    velocity        = [0.0, v_circ, 0.0],
    omega_primary   = [0.0, 0.0, w_alpha],
    omega_secondary = [0.0, 0.0, w_sync],
)

# ── Integrate ─────────────────────────────────────────────────────────────────

N_PERIODS = 10
t_final   = N_PERIODS * T_orb
dt        = 60.0
nOut      = 10

print(f"\nIntegrating {N_PERIODS} orbits ({t_final/3600:.1f} h) "
      f"with LGVI  dt={dt:.0f} s, nOut={nOut} ...")

results = sim.integrate(
    t_final    = t_final,
    integrator = f2bp.LGVI(dt=dt),
    nOut       = nOut,
)

print(f"Done.  Output steps: {results.n_steps}")
print(results)

# ── Analysis ───────────────────────────────────────────────────────────────────

ana    = sim.analysis
t_days = results.times / 86400.0

_, dE = ana.energy_conservation()
_, dH = ana.angular_momentum_conservation()
wp, ws = ana.spin_rates()
_, _, yaw = results.secondary_euler_angles(frame='orbital')

print(f"\nMax |dE/E₀|:    {np.max(np.abs(dE)):.2e}")
print(f"Max |d|H|/|H₀||: {np.max(np.abs(dH)):.2e}")
print(f"Mean separation: {ana.mean_separation()/1000:.4f} km")

# ── Plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Separation vs time
ax = axes[0, 0]
ax.plot(t_days, results.separation * 1e-3, lw=0.9, color="steelblue")
ax.set_xlabel("Time [days]"); ax.set_ylabel("|r| [km]")
ax.set_title("Body Separation")

# Conservation
ax = axes[0, 1]
ax.plot(t_days, dE, lw=0.9, color="steelblue", label="dE/E₀")
ax.plot(t_days, dH, lw=0.9, color="coral",     label="d|H|/|H₀|", ls="--")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]"); ax.set_ylabel("Relative error")
ax.set_title("Conservation Errors")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=8)

# Spin rates
ax = axes[1, 0]
ax.plot(t_days, np.degrees(wp), lw=0.9, color="steelblue", label="Alpha |ω|")
ax.plot(t_days, np.degrees(ws), lw=0.9, color="coral",     label="Beta |ω|")
ax.set_xlabel("Time [days]"); ax.set_ylabel("|ω| [deg/s]")
ax.set_title("Spin Rates")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=8)

# Libration angle
ax = axes[1, 1]
ax.plot(t_days, yaw, lw=0.9, color="seagreen")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Time [days]"); ax.set_ylabel("Yaw [deg]")
ax.set_title("Beta Libration Angle (yaw in Hill frame)")

fig.suptitle("1999 KW4 — Polyhedral Shape Models", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("kw4.png", dpi=150, bbox_inches="tight")
print("\nSaved kw4.png")

# ── Animation ──────────────────────────────────────────────────────────────────

# print("\nRendering animation ...")
# sim.animate.matplotlib(
#     frame          = 'orbital',
#     fps            = 20.0,
#     primary_color  = 'royalblue',
#     secondary_color= 'tomato',
#     save_path      = 'orbit.gif',
#     dpi            = 100,
# )
# print("Saved orbit.gif")
