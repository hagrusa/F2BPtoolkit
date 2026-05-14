"""
Didymos–Dimorphos binary asteroid simulation
============================================

Full F2BP integration for the Didymos–Dimorphos system using ellipsoidal
shape models and a 2nd-order mutual potential expansion (J2/C22 terms).

Demonstrates the full analysis and visualization pipeline:
  - Orbital geometry (2-D and 3-D orbit, separation)
  - Conservation diagnostics (total energy, angular momentum)
  - Spin dynamics (rates, components, inclination)
  - Orbital elements (eccentricity, period estimate)
  - Attitude evolution (rotation matrix elements, Euler angles)
  - Animation of the binary system

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import f2bptoolkit as f2bp

# ── System parameters ──────────────────────────────────────────────────────────

G = 6.674e-11   # m³ kg⁻¹ s⁻²

# Didymos (primary)
rhoA       = 2400.0
aA, bA, cA = 400.0, 395.0, 340.0   # m

# Dimorphos (secondary)
rhoB       = 2400.0
aB, bB, cB = 85.0, 73.0, 63.0   # m

# Approximate masses (uniform-density ellipsoid: 4/3 π a b c ρ)
MA = (4./3.) * np.pi * aA * bA * cA * rhoA
MB = (4./3.) * np.pi * aB * bB * cB * rhoB
print(f"Primary mass:    {MA:.4e} kg")
print(f"Secondary mass:  {MB:.4e} kg")

# Circular orbit initial conditions
r0     = 1195.0                           # m, approximate semi-major axis
v_circ = np.sqrt(G * (MA + MB) / r0)     # m/s, circular velocity
n_orb  = np.sqrt(G * (MA + MB) / r0**3)  # rad/s, mean motion
T_orb  = 2 * np.pi / n_orb               # s, orbital period
print(f"Circular velocity (approx):   {v_circ:.4f} m/s")
print(f"Orbital period    (approx):      {T_orb/3600:.3f} h")

# Spin rates — Didymos fast rotator (~2.26 h), Dimorphos tidally locked
T_didymos = 2.26 * 3600.0              # s
w_primary  = 2 * np.pi / T_didymos    # rad/s
w_sync     = n_orb                     # rad/s, synchronous with orbit
print(f"Didymos spin period: {T_didymos/3600:.3f} h  → ω = {w_primary:.6f} rad/s")
print(f"Dimorphos sync spin: ω = {w_sync:.6f} rad/s  ({T_orb/3600:.3f} h)")

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

# Start in approximate 1:1 spin-orbit resonance (Dimorphos tidally locked,
# long axis pointing toward Didymos).  At t=0 we assume A=N (identity
# attitude matrices), so the body frame coincides with the inertial frame.
sim.set_state(
    position        = [r0, 0.0, 0.0],          # m, secondary w.r.t. primary in A frame
    velocity        = [0.0, v_circ, 0.0],       # m/s
    omega_primary   = [0.0, 0.0, w_primary],    # rad/s — Didymos fast spin
    omega_secondary = [0.0, 0.0, w_sync],       # rad/s — Dimorphos synchronous
)

# ── Integrate ─────────────────────────────────────────────────────────────────

N_PERIODS = 10
t_final   = N_PERIODS * T_orb
dt        = 60.0     # s, fixed RK4 step
nOut      = 20       # record every 20 steps

print(f"\nIntegrating {N_PERIODS} orbits ({t_final/86400:.2f} days) "
      f"with LGVI  dt={dt:.0f} s, nOut={nOut} ...")

results = sim.integrate(
    t_final    = t_final,
    integrator = f2bp.LGVI(dt=dt),
    nOut       = nOut,
)

print(f"Done.  Output steps: {results.n_steps}")
print(results)

# ── Convenience aliases ────────────────────────────────────────────────────────

t_days = results.times / 86400.0
ana    = sim.analysis
p      = sim.plot

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Summary dashboard (orbit, separation, spins, energy)
# ══════════════════════════════════════════════════════════════════════════════

print("\nPlotting Fig 1: summary dashboard ...")
fig1, _ = p.summary()
fig1.savefig("fig1_summary.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Orbital geometry (2-D + 3-D)
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 2: orbital geometry ...")
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# 2-D projections in A frame (km)
for ax, plane, idx in zip(axes2[:2],
                          ["xy", "xz"],
                          [(0, 1), (0, 2)]):
    pos_km = results.position * 1e-3
    i, j   = idx
    ax.plot(pos_km[:, i], pos_km[:, j], lw=0.7, color="steelblue", alpha=0.8)
    ax.scatter([0], [0], color="k", s=80, marker="*", zorder=5, label="Didymos")
    ax.scatter([pos_km[0, i]], [pos_km[0, j]], color="green", s=40, zorder=5, label="t=0")
    ax.scatter([pos_km[-1, i]], [pos_km[-1, j]], color="red", s=40, zorder=5, label="t=end")
    ax.set_xlabel(f"{'xyz'[i]} [km]")
    ax.set_ylabel(f"{'xyz'[j]} [km]")
    ax.set_aspect("equal")
    ax.set_title(f"Orbit ({plane.upper()} plane)")
    ax.legend(fontsize=8)

# 3-D orbit
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
ax3d = fig2.add_subplot(133, projection="3d")
pos_km = results.position * 1e-3
ax3d.plot(pos_km[:, 0], pos_km[:, 1], pos_km[:, 2],
          lw=0.6, color="steelblue", alpha=0.8)
ax3d.scatter(0, 0, 0, color="k", s=80, marker="*")
ax3d.set_xlabel("x [km]"); ax3d.set_ylabel("y [km]"); ax3d.set_zlabel("z [km]")
ax3d.set_title("Orbit (3D)")

fig2.suptitle("Dimorphos Relative Orbit", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("fig2_orbit.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Conservation diagnostics
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 3: conservation diagnostics ...")
t_con, dE = ana.energy_conservation()
t_con, dH = ana.angular_momentum_conservation()
t_con_days = t_con / 86400.0

# Angular momentum vector components (inertial frame)
H_vec = results.angular_momentum   # (N, 3) kg m²/s
H_mag = results.angular_momentum_magnitude

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 7))

ax = axes3[0, 0]
ax.plot(t_con_days, dE, lw=0.9, color="steelblue")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Time [days]")
ax.set_ylabel("(E − E₀) / |E₀|")
ax.set_title("Total Energy Conservation")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes3[0, 1]
ax.plot(t_con_days, dH, lw=0.9, color="coral")
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlabel("Time [days]")
ax.set_ylabel("(|H| − |H₀|) / |H₀|")
ax.set_title("Angular Momentum Magnitude Conservation")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes3[1, 0]
labels_H = ["Hx", "Hy", "Hz"]
colors_H = ["steelblue", "coral", "seagreen"]
for k in range(3):
    ax.plot(t_days, H_vec[:, k], lw=0.8, label=labels_H[k], color=colors_H[k])
ax.set_xlabel("Time [days]")
ax.set_ylabel("H [kg m² / s]")
ax.set_title("Angular Momentum Components (Inertial)")
ax.legend(fontsize=8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes3[1, 1]
# Total energy breakdown: KE orbital, KE rotational, PE
KE_orb  = results.kinetic_energy_orbital
KE_rot1 = results.kinetic_energy_rotation_primary
KE_rot2 = results.kinetic_energy_rotation_secondary
PE      = results.potential_energy
E_total = KE_orb + KE_rot1 + KE_rot2 + PE
ax.plot(t_days, KE_orb  * 1e-9, lw=0.8, label="KE orbital",   color="steelblue")
ax.plot(t_days, KE_rot1 * 1e-9, lw=0.8, label="KE primary",   color="seagreen")
ax.plot(t_days, KE_rot2 * 1e-9, lw=0.8, label="KE secondary", color="gold")
ax.plot(t_days, PE      * 1e-9, lw=0.8, label="PE mutual",     color="coral")
ax.plot(t_days, E_total * 1e-9, lw=1.2, label="E total",       color="k", ls="--")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Energy [GJ]")
ax.set_title("Energy Budget")
ax.legend(fontsize=7)

fig3.suptitle("Conservation Diagnostics", fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.savefig("fig3_conservation.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Spin dynamics
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 4: spin dynamics ...")
wp_rad, ws_rad = ana.spin_rates()   # rad/s magnitudes, shape (N,)
inc = ana.mutual_inclination()    # rad, shape (N,)
colors3 = ["steelblue", "coral", "seagreen"]

# Secondary ω in B frame: ω_B = B_to_A.T @ ω_A  (A_to_B applied to ω_A)
omega_c   = results.omega_primary    # (N, 3) already in A (primary body) frame
omega_s_A = results.omega_secondary  # (N, 3) in A frame
omega_s_B = np.einsum('nij,nj->ni',
                      results.B_to_A.swapaxes(1, 2),   # A_to_B = B_to_A.T
                      omega_s_A)                         # → (N, 3) in B frame

# Dimorphos Euler angles in the orbital frame
roll_O, pitch_O, yaw_O = results.secondary_euler_angles(frame='orbital')

fig4, axes4 = plt.subplots(2, 2, figsize=(12, 7))

ax = axes4[0, 0]
ax.plot(t_days, np.degrees(wp_rad), lw=0.9, color="steelblue", label="Didymos")
ax.plot(t_days, np.degrees(ws_rad), lw=0.9, color="coral",     label="Dimorphos")
ax.set_xlabel("Time [days]")
ax.set_ylabel("|ω| [deg/s]")
ax.set_title("Spin Rate Magnitudes")
ax.legend(fontsize=8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes4[0, 1]
for ang, lbl, col in zip([roll_O, pitch_O, yaw_O],
                          ["Roll", "Pitch", "Yaw"], colors3):
    ax.plot(t_days, ang, lw=0.8, label=lbl, color=col)
ax.set_xlabel("Time [days]")
ax.set_ylabel("Angle [deg]")
ax.set_title("Dimorphos Euler Angles — Orbital Frame")
ax.legend(fontsize=8)

ax = axes4[1, 0]
for k, lbl in enumerate(["ωx", "ωy", "ωz"]):
    ax.plot(t_days, np.degrees(omega_c[:, k]), lw=0.8, label=lbl, color=colors3[k])
ax.set_xlabel("Time [days]")
ax.set_ylabel("ω [deg/s]")
ax.set_title("Didymos Angular Velocity — Body Frame (A)")
ax.legend(fontsize=8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes4[1, 1]
for k, lbl in enumerate(["ωx", "ωy", "ωz"]):
    ax.plot(t_days, np.degrees(omega_s_B[:, k]), lw=0.8, label=lbl, color=colors3[k])
ax.set_xlabel("Time [days]")
ax.set_ylabel("ω [deg/s]")
ax.set_title("Dimorphos Angular Velocity — Body Frame (B)")
ax.legend(fontsize=8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig4.suptitle("Spin Dynamics", fontsize=13, fontweight="bold")
fig4.tight_layout()
fig4.savefig("fig4_spin.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Orbital elements
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 5: orbital elements ...")
ecc = ana.eccentricity()    # instantaneous eccentricity magnitude (N,)
sep = results.separation    # m

T_est = ana.orbital_period_estimate()
print(f"Orbital period estimate (autocorrelation): "
      + (f"{T_est/3600:.3f} h" if T_est is not None else "N/A"))
print(f"Mean separation: {ana.mean_separation():.1f} m  "
      f"({ana.mean_separation()/1000:.3f} km)")

# Vis-viva semi-major axis: a = 1 / (2/r - v²/GM)
v_sq = np.sum(results.velocity**2, axis=1)
GM   = G * (MA + MB)
a_vv = 1.0 / (2.0 / sep - v_sq / GM)    # m

fig5, axes5 = plt.subplots(2, 2, figsize=(12, 7))

ax = axes5[0, 0]
ax.plot(t_days, sep * 1e-3, lw=0.9, color="steelblue")
ax.set_xlabel("Time [days]")
ax.set_ylabel("|r| [km]")
ax.set_title(f"Separation  (mean = {sep.mean()*1e-3:.3f} km)")

ax = axes5[0, 1]
ax.plot(t_days, ecc, lw=0.9, color="coral")
ax.set_xlabel("Time [days]")
ax.set_ylabel("e")
ax.set_title("Instantaneous Eccentricity (LRL vector magnitude)")

ax = axes5[1, 0]
ax.plot(t_days, np.degrees(inc), lw=0.9, color="seagreen")
ax.set_xlabel("Time [days]")
ax.set_ylabel("θ [deg]")
ax.set_title("Mutual Inclination w.r.t. Primary Spin Pole")

ax = axes5[1, 1]
ax.plot(t_days, a_vv * 1e-3, lw=0.9, color="seagreen")
ax.set_xlabel("Time [days]")
ax.set_ylabel("a [km]")
ax.set_title("Vis-viva Semi-major Axis")

fig5.suptitle("Orbital Elements", fontsize=13, fontweight="bold")
fig5.tight_layout()
fig5.savefig("fig5_orbital_elements.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Attitude evolution (Euler angles, all three frames)
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Fig 6: attitude / Euler angles ...")
roll_N, pitch_N, yaw_N = results.secondary_euler_angles(frame='inertial')
roll_A, pitch_A, yaw_A = results.secondary_euler_angles(frame='primary')
ang_labels = ["Roll", "Pitch", "Yaw"]
ang_colors = ["steelblue", "coral", "seagreen"]

fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4))

for ax, title, angs in zip(
        axes6,
        ["Inertial frame (N)", "Orbital/Hill frame (O)", "Primary body frame (A)"],
        [(roll_N, pitch_N, yaw_N),
         (roll_O, pitch_O, yaw_O),
         (roll_A, pitch_A, yaw_A)]):
    for lbl, col, ang in zip(ang_labels, ang_colors, angs):
        ax.plot(t_days, ang, lw=0.8, label=lbl, color=col)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title(f"Dimorphos Euler Angles\n{title}")
    ax.legend(fontsize=8)

fig6.suptitle("Dimorphos Attitude Evolution (ZYX convention)", fontsize=12, fontweight="bold")
fig6.tight_layout()
fig6.savefig("fig6_attitude.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary diagnostics
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("DIAGNOSTICS")
print("═" * 60)

max_dE = np.max(np.abs(dE))
max_dH = np.max(np.abs(dH))
print(f"Max |dE/E₀|:           {max_dE:.2e}")
print(f"Max |d|H|/|H₀||:       {max_dH:.2e}")
print(f"Mean separation:       {ana.mean_separation()/1000:.4f} km")
print(f"Mean eccentricity:     {ecc.mean():.4f}  (max {ecc.max():.4f})")
print(f"Mean inclination:      {np.degrees(inc).mean():.3f} deg")
print(f"Didymos spin drift:    {(wp_rad[-1]-wp_rad[0])/wp_rad[0]*100:.4f}%")
print(f"Dimorphos spin drift:  {(ws_rad[-1]-ws_rad[0])/ws_rad[0]*100:.4f}%")
print(f"Orbital-frame yaw max: {np.abs(yaw_O).max():.2f} deg  "
      f"({'librating' if np.abs(yaw_O).max() < 90 else 'TUMBLING'})")

# ══════════════════════════════════════════════════════════════════════════════
# Animation — orbital frame, with Dimorphos principal-axis arrows
# ══════════════════════════════════════════════════════════════════════════════
# We build this manually (rather than calling sim.animate.matplotlib) so we
# can:
#   1. Zoom the view to ~2× the orbital separation (secondary fills ~10% of
#      the frame instead of ~3%).
#   2. Draw three quiver arrows on Dimorphos showing its principal axes (body
#      frame B columns expressed in the orbital display frame).

print("\nRendering matplotlib animation (orbital frame, zoomed + attitude arrows) ...")

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from f2bptoolkit.animation import (_EllipsoidMesh, _body_rotations)
import matplotlib.patches as mpatches

STRIDE    = max(1, results.n_steps // 200)   # ~200 frames
FPS       = 20.0
ALPHA     = 0.85
COL_PRIM  = "royalblue"
COL_SEC   = "tomato"
# Arrow colours: principal axes x̂_B (red), ŷ_B (green), ẑ_B (blue)
ARROW_COLORS = ["red", "limegreen", "dodgerblue"]
ARROW_LABELS = ["x̂_B (long)", "ŷ_B (mid)", "ẑ_B (short)"]
# Arrow length = 2 × corresponding semi-axis so they extend well outside body
ARROW_LENGTHS = [2 * aB, 2 * bB, 2 * cB]

prim_mesh = _EllipsoidMesh(aA, bA, cA)
sec_mesh  = _EllipsoidMesh(aB, bB, cB)

# Axis limits: tight around the orbit (no padding beyond max_sep)
max_sep = results.separation.max()
lim     = max_sep * 1.15   # just enough to keep the orbit visible

indices = list(range(0, results.n_steps, STRIDE))

fig_ani = plt.figure(figsize=(8, 8))
ax_ani  = fig_ani.add_subplot(111, projection='3d')
ax_ani.set_title("Didymos–Dimorphos — Orbital Frame")
ax_ani.set_xlabel("x [m]"); ax_ani.set_ylabel("y [m]"); ax_ani.set_zlabel("z [m]")
ax_ani.set_xlim3d(-lim, lim)
ax_ani.set_ylim3d(-lim, lim)
ax_ani.set_zlim3d(-lim, lim)

time_text = ax_ani.text2D(0.02, 0.95, '', transform=ax_ani.transAxes,
                           fontsize=9, family='monospace')
ax_ani.legend(handles=[
    mpatches.Patch(color=COL_PRIM,  label="Didymos"),
    mpatches.Patch(color=COL_SEC,   label="Dimorphos"),
    *[mpatches.Patch(color=c, label=l)
      for c, l in zip(ARROW_COLORS, ARROW_LABELS)],
], loc='upper right', fontsize=7)

_objects = []   # surfaces + quivers removed each frame

def _update_ani(idx):
    for obj in _objects:
        try:
            obj.remove()
        except Exception:
            pass
    _objects.clear()

    A_to_N = results.A_to_N[idx]
    B_to_A = results.B_to_A[idx]
    r_A    = results.position[idx]
    v_A    = results.velocity[idx]

    prim_R, sec_R, sec_off = _body_rotations(A_to_N, B_to_A, r_A, v_A, 'orbital')

    # Primary surface
    X, Y, Z = prim_mesh.surface_grid(prim_R)
    _objects.append(ax_ani.plot_surface(
        X, Y, Z, color=COL_PRIM, alpha=ALPHA, linewidth=0, shade=True))

    # Secondary surface
    X, Y, Z = sec_mesh.surface_grid(sec_R)
    X += sec_off[0]; Y += sec_off[1]; Z += sec_off[2]
    _objects.append(ax_ani.plot_surface(
        X, Y, Z, color=COL_SEC, alpha=ALPHA, linewidth=0, shade=True))

    # Principal-axis arrows on Dimorphos.
    # sec_R rotates body-frame column vectors into the display (orbital) frame.
    # Column k of sec_R is the k-th body axis expressed in the display frame.
    ox, oy, oz = sec_off
    for k, (col, length) in enumerate(zip(ARROW_COLORS, ARROW_LENGTHS)):
        axis_disp = sec_R[:, k] * length   # direction in display frame
        q = ax_ani.quiver(ox, oy, oz,
                          axis_disp[0], axis_disp[1], axis_disp[2],
                          color=col, linewidth=2, arrow_length_ratio=0.25)
        _objects.append(q)

    time_text.set_text(f't = {results.times[idx] / 86400.0:.4f} days')
    return _objects + [time_text]

ani = FuncAnimation(fig_ani, _update_ani, frames=indices,
                    blit=False, interval=1000.0 / FPS)

print("Saving orbit.gif ...")
ani.save("orbit.gif", fps=FPS, dpi=100)
plt.close(fig_ani)
print("Saved  orbit.gif")

print("\nAll figures saved:")
for i, name in enumerate([
    "fig1_summary.png",
    "fig2_orbit.png",
    "fig3_conservation.png",
    "fig4_spin.png",
    "fig5_orbital_elements.png",
    "fig6_attitude.png",
    "didymos_dimorphos_orbital.gif",
], start=1):
    print(f"  {i}. {name}")
