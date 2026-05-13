"""Main Simulation class for F2BPtoolkit."""

import numpy as np
from typing import List, Optional, Union

from .body import Body, PolyhedronShape
from .state import InitialState
from .integrators import Integrator, RK4, LGVI, RK87, ABM
from .perturbations import (
    Perturbation, FlybyPerturbation, HeliocentricPerturbation,
    SolarGravityPerturbation, TidalTorquePerturbation,
)
from .results import SimulationResults

# SI constants
_G_SI  = 6.674e-11     # m³ kg⁻¹ s⁻²
_G_KM  = 6.674e-20     # km³ kg⁻¹ s⁻²  (= _G_SI / 1e9)
_AU_M  = 1.496e11      # m
_M_SUN = 1.989e30      # kg


class Simulation:
    """
    Full Two-Body Problem (F2BP) simulation.

    The typical workflow mirrors the Rebound N-body package:

    1. Create a ``Simulation``.
    2. Add two ``Body`` objects (primary first, then secondary).
    3. Set the initial state with ``set_state()`` or ``set_state_from_spice()``.
    4. Optionally add perturbations with ``add_perturbation()``.
    5. Call ``integrate(t_final)`` to run.
    6. Inspect results via ``sim.results``, ``sim.plot``, or ``sim.analysis``.

    Example
    -------
    >>> import f2bptoolkit as f2bp
    >>>
    >>> sim = f2bp.Simulation()
    >>>
    >>> primary = f2bp.Body("Didymos")
    >>> primary.shape   = f2bp.EllipsoidShape(a=400, b=395, c=340)  # m
    >>> primary.density = 2170.0   # kg/m³
    >>> sim.add(primary)
    >>>
    >>> secondary = f2bp.Body("Dimorphos")
    >>> secondary.shape   = f2bp.EllipsoidShape(a=85, b=73, c=63)   # m
    >>> secondary.density = 2400.0
    >>> sim.add(secondary)
    >>>
    >>> sim.set_state(
    ...     position      = [1195.0, 0.0, 0.0],   # m
    ...     velocity      = [0.0, 0.1735, 0.0],    # m/s
    ...     omega_primary = [0.0, 0.0, 7.26e-4],   # rad/s
    ...     omega_secondary = [0.0, 0.0, 7.26e-4], # rad/s (doubly synchronous)
    ... )
    >>>
    >>> sim.gravity_order = 2
    >>> results = sim.integrate(t_final=86400 * 50,
    ...                         integrator=f2bp.RK4(dt=1.0),
    ...                         nOut=30)
    >>> sim.plot.summary()
    """

    def __init__(self, G: float = _G_SI):
        """
        Parameters
        ----------
        G : float
            Gravitational constant in SI (m³ kg⁻¹ s⁻²).  Default: 6.674e-11.
        """
        self._G    = float(G)
        self._G_km = self._G / 1e9      # km³ kg⁻¹ s⁻²
        self._bodies: List[Body] = []
        self._state: Optional[InitialState] = None
        self._gravity_order: int = 2
        self._perturbations: List[Perturbation] = []
        self._results: Optional[SimulationResults] = None

    # ── bodies ────────────────────────────────────────────────────────────────

    def add(self, body: Body):
        """
        Add a body to the simulation.

        The first body added is the primary; the second is the secondary.
        Raises ``ValueError`` if more than two bodies are added or if the
        body is not fully configured.
        """
        if len(self._bodies) >= 2:
            raise ValueError("A binary system has exactly 2 bodies.")
        body.validate()
        self._bodies.append(body)

    @property
    def primary(self) -> Optional[Body]:
        """The primary body, or None if not yet added."""
        return self._bodies[0] if len(self._bodies) > 0 else None

    @property
    def secondary(self) -> Optional[Body]:
        """The secondary body, or None if not yet added."""
        return self._bodies[1] if len(self._bodies) > 1 else None

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> InitialState:
        """
        The ``InitialState`` object.  Created lazily on first access so that
        individual components can be set via ``sim.state.omega_primary = …``
        """
        if self._state is None:
            self._state = InitialState()
        return self._state

    def set_state(
        self,
        position,
        velocity,
        omega_primary,
        omega_secondary,
        A_to_N=None,
        B_to_A=None,
    ):
        """
        Set the initial state.

        Parameters
        ----------
        position : array-like, shape (3,)
            Relative position of secondary w.r.t. primary *in the primary body
            frame (A)*, in meters.
        velocity : array-like, shape (3,)
            Relative velocity in the A frame, in m/s.
        omega_primary : array-like, shape (3,)
            Primary angular velocity in the A frame, in rad/s.
        omega_secondary : array-like, shape (3,)
            Secondary angular velocity in the secondary body frame (B), in rad/s.
        A_to_N : array-like, shape (3,3), optional
            Rotation matrix A→N (v_N = A_to_N @ v_A).  Default: identity.
        B_to_A : array-like, shape (3,3), optional
            Rotation matrix B→A (v_A = B_to_A @ v_B).  Default: identity.
        """
        s = self.state
        s.position        = position
        s.velocity        = velocity
        s.omega_primary   = omega_primary
        s.omega_secondary = omega_secondary
        if A_to_N is not None:
            s.A_to_N = A_to_N
        if B_to_A is not None:
            s.B_to_A = B_to_A

    def set_state_from_spice(
        self,
        kernel_files,
        primary_name: str,
        secondary_name: str,
        epoch,
        frame: str = "J2000",
        observer: str = "SSB",
    ):
        """
        Set the relative position and velocity from SPICE kernels.

        The angular velocities and attitude matrices must still be set
        separately (SPICE ephemerides only contain translational state).
        Use ``sim.state.omega_primary = …`` after calling this method, or
        ``spice_utils.angular_velocity_from_spice()`` if a body-fixed PCK
        kernel is available.

        Parameters
        ----------
        kernel_files : str or list of str
            Path(s) to SPICE kernel files to furnish.
        primary_name : str
            SPICE name of the primary (e.g., ``"DIDYMOS"``).
        secondary_name : str
            SPICE name of the secondary (e.g., ``"DIMORPHOS"``).
        epoch : str or float
            Epoch string (e.g., ``"2022-10-01T00:00:00"``) or ET seconds.
        frame : str
            Reference frame.  Default: ``"J2000"``.
        observer : str
            Observer.  Default: ``"SSB"``.

        Notes
        -----
        SPICE returns state in the inertial frame.  If the simulation state
        is expected in the primary body frame (the default for F2BP), you
        should also set ``A_to_N`` and rotate.  Note that
        ``rotation_matrix_from_spice`` returns N→A (from SPICE ``pxform``),
        which must be transposed to get A→N:

        .. code-block:: python

            N_to_A = spice_utils.rotation_matrix_from_spice(
                "DIDYMOS", epoch, "DIDYMOS_FIXED")
            A_to_N = N_to_A.T
            sim.state.A_to_N    = A_to_N
            sim.state.position  = N_to_A @ pos_inertial   # N→A applied to r_N = r_A
            sim.state.velocity  = N_to_A @ vel_inertial   # N→A applied to v_N = v_A
        """
        from .spice_utils import state_from_spice
        pos, vel = state_from_spice(
            kernel_files, primary_name, secondary_name, epoch, frame, observer
        )
        s = self.state
        s.position = pos
        s.velocity = vel

    # ── gravity order ─────────────────────────────────────────────────────────

    @property
    def gravity_order(self) -> int:
        """
        Mutual potential truncation order (must be a non-negative even integer).

        - 0 : point-mass approximation (fast, inaccurate for extended bodies)
        - 2 : includes J₂ and C₂₂ terms  (good balance of accuracy/speed)
        - 4 : captures 4th-order harmonics (slower)
        """
        return self._gravity_order

    @gravity_order.setter
    def gravity_order(self, n: int):
        n = int(n)
        if n < 0 or n % 2 != 0:
            raise ValueError("gravity_order must be a non-negative even integer (0, 2, 4, …)")
        self._gravity_order = n

    # ── perturbations ─────────────────────────────────────────────────────────

    def add_perturbation(self, perturbation: Perturbation):
        """
        Add a perturbation force/torque.

        Multiple perturbations can be added.  Note: ``LGVI`` does not support
        any perturbations.

        Parameters
        ----------
        perturbation : Perturbation
            A ``FlybyPerturbation``, ``HeliocentricPerturbation``,
            ``SolarGravityPerturbation``, or ``TidalTorquePerturbation``.
        """
        self._perturbations.append(perturbation)

    # ── integrate ─────────────────────────────────────────────────────────────

    def integrate(
        self,
        t_final: float,
        t_start: float = 0.0,
        integrator: Optional[Integrator] = None,
        nOut: Optional[int] = None,
    ) -> SimulationResults:
        """
        Run the simulation.

        Parameters
        ----------
        t_final : float
            Final integration time in seconds.
        t_start : float
            Start time in seconds.  Default: 0.
        integrator : Integrator, optional
            Which integrator to use.  Default: ``RK4(dt=1.0)``.
        nOut : int, optional
            Record a snapshot every *nOut* integration timesteps.
            Default: 1 (every step).  Not supported for ``RK87`` (adaptive
            step size); raises ``ValueError`` if supplied with that integrator.

        Returns
        -------
        SimulationResults
        """
        # ── import C++ extension ──────────────────────────────────────────────
        try:
            from . import _core
        except ImportError as exc:
            raise ImportError(
                "The C++ extension '_core' is not built.  "
                "Run: pip install -e . (requires CMake and Armadillo)"
            ) from exc

        # ── validate ──────────────────────────────────────────────────────────
        if len(self._bodies) != 2:
            raise RuntimeError(
                f"Need exactly 2 bodies; have {len(self._bodies)}."
            )
        self.state.validate()

        integrator = integrator or RK4(dt=1.0)

        # LGVI cannot have perturbations
        if isinstance(integrator, LGVI) and self._perturbations:
            raise ValueError(
                "The LGVI integrator does not support perturbations.  "
                "Use RK4 or ABM when perturbations are present."
            )

        # ── build SimConfig ───────────────────────────────────────────────────
        cfg = _core.SimConfig()

        cfg.G             = self._G_km
        cfg.gravity_order = self._gravity_order

        primary, secondary = self._bodies

        # Inertia orders — for polyhedra enforce order >= gravity_order
        order_a = primary.inertia_order
        order_b = secondary.inertia_order
        if isinstance(primary.shape,   PolyhedronShape):
            order_a = max(order_a, self._gravity_order)
        if isinstance(secondary.shape, PolyhedronShape):
            order_b = max(order_b, self._gravity_order)
        cfg.order_a = order_a
        cfg.order_b = order_b

        # Primary
        cfg.a_shape = primary._shape_flag()
        aA, bA, cA = primary._semi_axes_km()
        cfg.aA, cfg.bA, cfg.cA = aA, bA, cA
        cfg.rhoA = primary._density_kg_km3()
        if isinstance(primary.shape, PolyhedronShape):
            cfg.tet_fileA  = primary.shape.facet_file
            cfg.vert_fileA = primary.shape.vertex_file
        else:
            cfg.tet_fileA  = ""
            cfg.vert_fileA = ""

        # Secondary
        cfg.b_shape = secondary._shape_flag()
        aB, bB, cB = secondary._semi_axes_km()
        cfg.aB, cfg.bB, cfg.cB = aB, bB, cB
        cfg.rhoB = secondary._density_kg_km3()
        if isinstance(secondary.shape, PolyhedronShape):
            cfg.tet_fileB  = secondary.shape.facet_file
            cfg.vert_fileB = secondary.shape.vertex_file
        else:
            cfg.tet_fileB  = ""
            cfg.vert_fileB = ""

        # Initial state
        cfg.x0 = self.state.to_vector_km().tolist()

        # Integration settings
        cfg.t0        = float(t_start)
        cfg.tf        = float(t_final)
        cfg.integ_flag = integrator.flag

        if isinstance(integrator, (RK4, ABM, LGVI)):
            cfg.dt  = integrator.dt
            cfg.tol = 1e-10
            n = int(nOut) if nOut is not None else 1
            if n < 1:
                raise ValueError(f"nOut must be >= 1, got {n}")
            cfg.out_freq = n * integrator.dt
        else:   # RK87
            if nOut is not None:
                raise ValueError(
                    "nOut is not supported for the RK87 adaptive integrator "
                    "(no fixed timestep). Remove nOut or use a fixed-step integrator."
                )
            cfg.dt       = 1.0   # unused by adaptive integrator
            cfg.tol      = integrator.tol
            cfg.out_freq = 0.0   # 0 = every accepted step

        # Perturbation defaults (all off)
        cfg.flyby_toggle = 0
        cfg.helio_toggle = 0
        cfg.sg_toggle    = 0
        cfg.tt_toggle    = 0

        cfg.Mplanet   = 0.0
        cfg.a_hyp     = -1.0e6   # km (dummy hyperbolic orbit)
        cfg.e_hyp     = 1.5
        cfg.i_hyp     = 0.0
        cfg.RAAN_hyp  = 0.0
        cfg.om_hyp    = 0.0
        cfg.tau_hyp   = 0.0

        cfg.Msolar    = _M_SUN
        cfg.a_helio   = _AU_M / 1000.0   # m → km
        cfg.e_helio   = 0.0
        cfg.i_helio   = 0.0
        cfg.RAAN_helio = 0.0
        cfg.om_helio  = 0.0
        cfg.tau_helio = 0.0

        cfg.sol_rad   = 1.0
        cfg.au_def    = _AU_M / 1000.0

        cfg.love1    = 0.0
        cfg.love2    = 0.0
        cfg.refrad1  = 1.0
        cfg.refrad2  = 1.0
        cfg.eps1     = 0.0
        cfg.eps2     = 0.0
        cfg.Msun     = _M_SUN

        # Apply perturbations
        for p in self._perturbations:
            if isinstance(p, FlybyPerturbation):
                cfg.flyby_toggle = 1
                cfg.Mplanet  = p.mass
                cfg.a_hyp    = p.semi_major_axis / 1000.0   # m → km
                cfg.e_hyp    = p.eccentricity
                cfg.i_hyp    = p.inclination
                cfg.RAAN_hyp = p.raan
                cfg.om_hyp   = p.arg_periapsis
                cfg.tau_hyp  = p.tau

            elif isinstance(p, HeliocentricPerturbation):
                cfg.helio_toggle = 1
                cfg.Msolar    = p.mass_sun
                cfg.Msun      = p.mass_sun
                cfg.a_helio   = p.semi_major_axis / 1000.0
                cfg.e_helio   = p.eccentricity
                cfg.i_helio   = p.inclination
                cfg.RAAN_helio = p.raan
                cfg.om_helio  = p.arg_periapsis
                cfg.tau_helio = p.tau

            elif isinstance(p, SolarGravityPerturbation):
                cfg.sg_toggle = 1
                cfg.sol_rad   = p.solar_radius
                cfg.au_def    = p.au / 1000.0

            elif isinstance(p, TidalTorquePerturbation):
                cfg.tt_toggle = 1
                cfg.love1    = p.love_number_primary
                cfg.love2    = p.love_number_secondary
                cfg.refrad1  = p.ref_radius_primary  / 1000.0
                cfg.refrad2  = p.ref_radius_secondary / 1000.0
                cfg.eps1     = p.lag_angle_primary
                cfg.eps2     = p.lag_angle_secondary

        # ── run ───────────────────────────────────────────────────────────────
        result = _core.run_simulation(cfg)

        if result.status != "success":
            raise RuntimeError(f"Simulation failed: {result.status}")

        N = len(result.times)

        self._results = SimulationResults(
            times       = np.array(result.times),
            states_km   = np.array(result.states).reshape(N, 30),
            G           = self._G,
            mass_primary   = result.mass_primary,
            mass_secondary = result.mass_secondary,
            inertia_primary   = np.array(result.inertia_primary),
            inertia_secondary = np.array(result.inertia_secondary),
            hyp_states_km   = (np.array(result.hyp_states)
                               if cfg.flyby_toggle else None),
            solar_states_km = (np.array(result.solar_states)
                               if cfg.helio_toggle else None),
        )
        return self._results

    # ── convenience accessors ─────────────────────────────────────────────────

    @property
    def results(self) -> Optional[SimulationResults]:
        """The most recent ``SimulationResults``, or None before integration."""
        return self._results

    @property
    def plot(self):
        """Plotting interface (``sim.plot.orbit()``, etc.)."""
        from .visualization import PlotInterface
        if self._results is None:
            raise RuntimeError("No results yet — call integrate() first.")
        return PlotInterface(self._results)

    @property
    def analysis(self):
        """Analysis interface (``sim.analysis.energy_conservation()``, etc.)."""
        from .analysis import AnalysisInterface
        if self._results is None:
            raise RuntimeError("No results yet — call integrate() first.")
        return AnalysisInterface(self._results)

    @property
    def animate(self):
        """Animation interface (``sim.animate.matplotlib()``, ``sim.animate.paraview()``)."""
        from .animation import AnimationInterface
        if self._results is None:
            raise RuntimeError("No results yet — call integrate() first.")
        return AnimationInterface(self._results, self._bodies)

    def __repr__(self):
        bodies = [b.name for b in self._bodies] or ["(none)"]
        return (f"Simulation(bodies={bodies}, "
                f"gravity_order={self._gravity_order}, "
                f"perturbations={len(self._perturbations)})")
