"""
Didymos-Dimorphos binary asteroid simulation
=============================================

Reproduces a basic F2BP integration for the Didymos system using
ellipsoidal shape models and a 2nd-order mutual potential expansion.

Reference body parameters (approximate pre-DART values):
  Didymos:   a=400 m, b=395 m, c=340 m, rho=2170 kg/m^3
  Dimorphos: a=85 m,  b=73 m,  c=63 m,  rho=2400 kg/m^3
  Orbit:     a~1195 m, P~11.92 h; Didymos spin P~2.26 h, Dimorphos ~synchronous
"""

import numpy as np
import matplotlib.pyplot as plt
import f2bptoolkit as f2bp

# ── System parameters ──────────────────────────────────────────────────────────

G = 6.674e-11   # m^3 kg^-1 s^-2

# Didymos (primary)
rhoA   = 2400.0       # kg/m^3
aA, bA, cA = 400.0, 395.0, 340.0   # m

# Dimorphos (secondary)
rhoB   = 2400.0       # kg/m^3
aB, bB, cB = 90.0, 73.0, 63.0   # m

# Approximate masses (4/3 pi a b c rho)
MA = (4./3.) * np.pi * aA * bA * cA * rhoA
MB = (4./3.) * np.pi * aB * bB * cB * rhoB
print(f"Primary mass:   {MA:.4e} kg")
print(f"Secondary mass: {MB:.4e} kg")

# Circular orbit radius and velocity (point-mass approximation)
r0    = 1195.0      # m, approximate semi-major axis
v_circ = np.sqrt(G * (MA + MB) / r0)
print(f"Circular velocity: {v_circ:.4f} m/s")

# Mean motion (rad/s) → period
n   = np.sqrt(G * (MA + MB) / r0**3)
T   = 2 * np.pi / n
print(f"Orbital period: {T/3600:.3f} h  ({T:.1f} s)")

# Spin rates
T_didymos  = 2.26 * 3600.0   # s — Didymos fast rotation period
w_primary  = 2 * np.pi / T_didymos   # rad/s
w_sync     = n                        # rad/s — Dimorphos ≈ tidally locked
print(f"Didymos spin period: {T_didymos/3600:.3f} h  → ω = {w_primary:.6f} rad/s")
print(f"Dimorphos (sync) spin: ω = {w_sync:.6f} rad/s  ({T/3600:.3f} h)")

# ── Build simulation ───────────────────────────────────────────────────────────

sim = f2bp.Simulation()
sim.gravity_order = 2

primary = f2bp.Body("Didymos")
primary.shape   = f2bp.EllipsoidShape(a=aA, b=bA, c=cA)
primary.density = rhoA
sim.add(primary)

secondary = f2bp.Body("Dimorphos")
secondary.shape   = f2bp.EllipsoidShape(a=aB, b=bB, c=cB)
secondary.density = rhoB
sim.add(secondary)

# Initial state: Didymos spins at its fast rotation (2.26 h); Dimorphos tidally locked.
# Circular orbit, long axis of Dimorphos pointing toward Didymos (1:1 spin-orbit resonance).
# Relative position and velocity expressed in the primary body frame (A).
# At t=0 we assume A=N (Cc=I), so body frame = inertial frame.
sim.set_state(
    position        = [r0, 0.0, 0.0],      # m
    velocity        = [0.0, v_circ*0.985, 0.0],  # m/s
    omega_primary   = [0.0, 0.0, w_primary],  # rad/s — Didymos fast spin
    omega_secondary = [0.001*w_sync, 0.001*w_sync, w_sync],     # rad/s — Dimorphos synchronous
    # attitude matrices default to identity — both frames aligned with inertial
)

# ── Integrate ─────────────────────────────────────────────────────────────────

t_final = 2.0 * T          # 2 orbital periods
dt      = 60.0              # s, fixed step
nOut    = 10               # record every 10 steps (~500 output points per orbit)

print(f"\nIntegrating for {t_final/86400:.2f} days "
      f"with LGVI dt={dt:.0f} s ...")

results = sim.integrate(
    t_final   = t_final,
    integrator = f2bp.LGVI(dt=dt),
    nOut       = nOut,
)

print(f"Done. Output steps: {results.n_steps}")
print(results)

# ── Quick diagnostics ─────────────────────────────────────────────────────────

t_days = results.times / 86400.0
sep_km = results.separation / 1000.0

print(f"\nSeparation: min={sep_km.min():.4f} km, "
      f"max={sep_km.max():.4f} km, "
      f"mean={sep_km.mean():.4f} km")

# Spin rates
wp = np.linalg.norm(results.omega_primary,   axis=1)
ws = np.linalg.norm(results.omega_secondary, axis=1)
print(f"Primary  spin drift: {(wp[-1]-wp[0])/wp[0]*100:.4f}%")
print(f"Secondary spin drift: {(ws[-1]-ws[0])/ws[0]*100:.4f}%")

# ── Dimorphos Euler angles (ZYX convention) ───────────────────────────────────
# 'inertial' : angles of B relative to the inertial frame N  (secular drift)
# 'orbital'  : angles of B relative to the Hill/LVLH frame O (libration only)
#              O: x̂ = radial outward, ẑ = orbit angular momentum, ŷ = ẑ×x̂

roll_N, pitch_N, yaw_N = results.secondary_euler_angles(frame='inertial')
roll,   pitch,   yaw   = results.secondary_euler_angles(frame='orbital')

print(f"\nInertial-frame yaw range: {yaw_N.min():.1f}° to {yaw_N.max():.1f}°  "
      f"(expected secular drift)")
print(f"Orbital-frame  yaw range: {yaw.min():.4f}° to {yaw.max():.4f}°  "
      f"(libration only)")

# ── Plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Separation vs time
ax = axes[0]
ax.plot(t_days, sep_km, lw=0.8, color="steelblue")
ax.set_xlabel("Time [days]")
ax.set_ylabel("|r| [km]")
ax.set_title("Separation vs. Time")

# Dimorphos inertial-frame Euler angles (shows secular yaw drift)
ax = axes[1]
ax.plot(t_days, roll_N,  lw=0.8, label="Roll",  color="steelblue")
ax.plot(t_days, pitch_N, lw=0.8, label="Pitch", color="coral")
ax.plot(t_days, yaw_N,   lw=0.8, label="Yaw",   color="seagreen")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Dimorphos Euler Angles — Inertial Frame")
ax.legend(fontsize=8)

# Dimorphos orbital-frame Euler angles (secular drift removed)
ax = axes[2]
ax.plot(t_days, roll,  lw=0.8, label="Roll",  color="steelblue")
ax.plot(t_days, pitch, lw=0.8, label="Pitch", color="coral")
ax.plot(t_days, yaw,   lw=0.8, label="Yaw",   color="seagreen")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Dimorphos Euler Angles — Orbital Frame (Hill/LVLH)")
ax.legend(fontsize=8)

fig.suptitle("Didymos–Dimorphos F2BP Simulation", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("didymos_dimorphos.png", dpi=150, bbox_inches="tight")
print("\nSaved didymos_dimorphos.png")
plt.show()

# ── Animation ─────────────────────────────────────────────────────────────────

# Matplotlib animation — orbital frame, one frame per orbit (~500 steps/orbit → stride 25)
print("\nRendering matplotlib animation (orbital frame) ...")
sim.animate.matplotlib(
    frame     = "orbital",
    stride    = 25,           # ~20 frames per orbit
    fps       = 20.0,
    save_path = "didymos_dimorphos_orbital.gif",
)
print("Saved didymos_dimorphos_orbital.gif")

# ParaView MP4 — inertial frame, every 10th step
print("\nRendering ParaView MP4 (inertial frame) ...")
sim.animate.paraview(
    save_path  = "didymos_dimorphos_inertial.mp4",
    frame      = "inertial",
    stride     = 10,
    fps        = 20.0,
)
print("Saved didymos_dimorphos_inertial.mp4")
