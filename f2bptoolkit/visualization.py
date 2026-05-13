"""Visualization tools for F2BP simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple
from .results import SimulationResults


def _days(times: np.ndarray) -> np.ndarray:
    """Convert seconds to days."""
    return times / 86400.0


class PlotInterface:
    """
    Plotting methods attached to simulation results.

    Accessed as ``sim.plot`` after running ``sim.integrate()``.

    All plot methods return ``(fig, ax)`` so the caller can further
    customise or save the figure.
    """

    def __init__(self, results: SimulationResults):
        self._r = results

    # ── orbit ─────────────────────────────────────────────────────────────────

    def orbit(self, plane: str = "xy", units: str = "km",
              figsize: Tuple = (6, 6)) -> Tuple:
        """
        Plot the relative orbit (secondary w.r.t. primary) projected onto a plane.

        Parameters
        ----------
        plane : str
            One of "xy", "xz", "yz".  Default: "xy".
        units : str
            "m" or "km".  Default: "km".
        figsize : tuple
            Figure size.
        """
        r = self._r.position
        scale = 1.0 if units == "m" else 1e-3
        labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
        idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        i, j = idx[plane]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(r[:, i] * scale, r[:, j] * scale, lw=0.8, color="steelblue")
        ax.scatter([r[0, i] * scale], [r[0, j] * scale], color="green",
                   zorder=5, label="start", s=40)
        ax.scatter([r[-1, i] * scale], [r[-1, j] * scale], color="red",
                   zorder=5, label="end", s=40)
        ax.scatter([0], [0], color="black", zorder=5, s=60, marker="*", label="primary")
        xi, xj = labels[plane]
        ax.set_xlabel(f"{xi} [{units}]")
        ax.set_ylabel(f"{xj} [{units}]")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title("Relative Orbit")
        fig.tight_layout()
        return fig, ax

    def orbit_3d(self, units: str = "km", figsize: Tuple = (7, 6)) -> Tuple:
        """
        3-D relative orbit plot.

        Parameters
        ----------
        units : str
            "m" or "km".  Default: "km".
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        r = self._r.position
        scale = 1.0 if units == "m" else 1e-3

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(r[:, 0] * scale, r[:, 1] * scale, r[:, 2] * scale,
                lw=0.8, color="steelblue")
        ax.scatter(*r[0] * scale, color="green", s=40, zorder=5, label="start")
        ax.scatter(*r[-1] * scale, color="red",   s=40, zorder=5, label="end")
        ax.scatter(0, 0, 0, color="black", s=60, marker="*", label="primary")
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")
        ax.set_zlabel(f"z [{units}]")
        ax.legend(fontsize=8)
        ax.set_title("Relative Orbit (3D)")
        fig.tight_layout()
        return fig, ax

    # ── separation ────────────────────────────────────────────────────────────

    def separation(self, units: str = "km",
                   figsize: Tuple = (8, 3)) -> Tuple:
        """
        Plot scalar separation |r| vs time.

        Parameters
        ----------
        units : str
            "m" or "km".  Default: "km".
        """
        t = _days(self._r.times)
        sep = self._r.separation * (1.0 if units == "m" else 1e-3)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, sep, lw=0.9, color="steelblue")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(f"|r| [{units}]")
        ax.set_title("Separation vs Time")
        fig.tight_layout()
        return fig, ax

    # ── spin rates ────────────────────────────────────────────────────────────

    def spin_rates(self, figsize: Tuple = (8, 4)) -> Tuple:
        """
        Plot primary and secondary spin rate magnitudes vs time.
        """
        t = _days(self._r.times)
        wp = np.linalg.norm(self._r.omega_primary,   axis=1)
        ws = np.linalg.norm(self._r.omega_secondary, axis=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, np.degrees(wp), lw=0.9, label="Primary |ω|", color="steelblue")
        ax.plot(t, np.degrees(ws), lw=0.9, label="Secondary |ω|", color="coral")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Spin rate [deg/s]")
        ax.set_title("Spin Rates vs Time")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def spin_components(self, body: str = "primary",
                        figsize: Tuple = (8, 5)) -> Tuple:
        """
        Plot x, y, z components of spin angular velocity vs time.

        Parameters
        ----------
        body : str
            "primary" or "secondary".
        """
        t = _days(self._r.times)
        omega = (self._r.omega_primary if body == "primary"
                 else self._r.omega_secondary)
        labels = ["ωx", "ωy", "ωz"]
        colors = ["steelblue", "coral", "seagreen"]

        fig, ax = plt.subplots(figsize=figsize)
        for k in range(3):
            ax.plot(t, np.degrees(omega[:, k]), lw=0.9,
                    label=labels[k], color=colors[k])
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Angular velocity [deg/s]")
        ax.set_title(f"{body.capitalize()} Angular Velocity Components")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    # ── energy & angular momentum ─────────────────────────────────────────────

    def energy_conservation(self, potential: Optional[np.ndarray] = None,
                             figsize: Tuple = (8, 3)) -> Tuple:
        """
        Plot normalised energy conservation error dE/E0 vs time.

        Parameters
        ----------
        potential : ndarray, shape (N,), optional
            Gravitational potential energy in Joules.  If not provided, only
            kinetic energy is shown (still useful for checking the integrator).
        """
        from .analysis import AnalysisInterface
        t, dE = AnalysisInterface(self._r).energy_conservation(potential)
        t = _days(t)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, dE, lw=0.9, color="steelblue")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("dE / E₀")
        ax.set_title("Energy Conservation Error")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        return fig, ax

    def angular_momentum_conservation(self, figsize: Tuple = (8, 3)) -> Tuple:
        """
        Plot normalised angular momentum conservation error d|H|/|H0| vs time.
        """
        from .analysis import AnalysisInterface
        t, dH = AnalysisInterface(self._r).angular_momentum_conservation()
        t = _days(t)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, dH, lw=0.9, color="coral")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("d|H| / |H₀|")
        ax.set_title("Angular Momentum Conservation Error")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        return fig, ax

    # ── attitude ──────────────────────────────────────────────────────────────

    def attitude_evolution(self, body: str = "primary",
                           figsize: Tuple = (8, 6)) -> Tuple:
        """
        Plot all 9 elements of the rotation matrix vs time.

        Parameters
        ----------
        body : str
            "primary" (A→N) or "relative" (B→A).
        """
        t = _days(self._r.times)
        C = (self._r.A_to_N if body == "primary" else self._r.B_to_A)

        fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True)
        for ii in range(3):
            for jj in range(3):
                axes[ii, jj].plot(t, C[:, ii, jj], lw=0.8, color="steelblue")
                axes[ii, jj].set_ylim(-1.1, 1.1)
                axes[ii, jj].set_ylabel(f"C[{ii},{jj}]", fontsize=7)
                if ii == 2:
                    axes[ii, jj].set_xlabel("Time [days]", fontsize=7)
        title = "Primary (A→N)" if body == "primary" else "Relative (B→A)"
        fig.suptitle(f"Attitude Matrix: {title}")
        fig.tight_layout()
        return fig, axes

    # ── summary dashboard ─────────────────────────────────────────────────────

    def summary(self, figsize: Tuple = (12, 9)) -> Tuple:
        """
        4-panel summary figure: orbit, separation, spin rates, energy error.
        """
        r = self._r
        t = _days(r.times)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

        # ── orbit (xy) ────────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        pos_km = r.position * 1e-3
        ax0.plot(pos_km[:, 0], pos_km[:, 1], lw=0.8, color="steelblue")
        ax0.scatter([0], [0], color="black", s=60, marker="*", zorder=5)
        ax0.set_xlabel("x [km]"); ax0.set_ylabel("y [km]")
        ax0.set_aspect("equal"); ax0.set_title("Orbit (xy)")

        # ── separation ────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot(t, r.separation * 1e-3, lw=0.9, color="steelblue")
        ax1.set_xlabel("Time [days]"); ax1.set_ylabel("|r| [km]")
        ax1.set_title("Separation")

        # ── spin rates ────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        wp = np.degrees(np.linalg.norm(r.omega_primary,   axis=1))
        ws = np.degrees(np.linalg.norm(r.omega_secondary, axis=1))
        ax2.plot(t, wp, lw=0.9, label="Primary",   color="steelblue")
        ax2.plot(t, ws, lw=0.9, label="Secondary", color="coral")
        ax2.set_xlabel("Time [days]"); ax2.set_ylabel("|ω| [deg/s]")
        ax2.set_title("Spin Rates"); ax2.legend(fontsize=8)

        # ── energy conservation ───────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        Kt  = r.kinetic_energy_orbital
        Kr1 = r.kinetic_energy_rotation_primary
        Kr2 = r.kinetic_energy_rotation_secondary
        if Kt is not None:
            KE = Kt + (Kr1 if Kr1 is not None else 0) + (Kr2 if Kr2 is not None else 0)
            dE = (KE - KE[0]) / abs(KE[0])
            ax3.plot(t, dE, lw=0.9, color="seagreen")
            ax3.axhline(0, color="gray", lw=0.5, ls="--")
            ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax3.set_xlabel("Time [days]"); ax3.set_ylabel("dE / E₀")
        ax3.set_title("Kinetic Energy Conservation")

        fig.suptitle("F2BP Simulation Summary", fontsize=13, fontweight="bold")
        return fig, (ax0, ax1, ax2, ax3)
