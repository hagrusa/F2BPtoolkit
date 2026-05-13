"""Integrator configuration classes."""


class Integrator:
    """Base class for integrators."""
    flag: int = 0


class RK4(Integrator):
    """
    Classical 4th-order Runge-Kutta, fixed time step.

    Supports all perturbations. Non-symplectic.

    Parameters
    ----------
    dt : float
        Time step in seconds.  Default: 1.0.
    """

    flag = 1

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def __repr__(self):
        return f"RK4(dt={self.dt} s)"


class LGVI(Integrator):
    """
    Lie Group Variational Integrator, fixed time step.

    Symplectic and SO(3)-preserving. Does **not** support perturbations
    (flyby, heliocentric, tidal torque).

    Parameters
    ----------
    dt : float
        Time step in seconds.  Default: 1.0.
    """

    flag = 2

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def __repr__(self):
        return f"LGVI(dt={self.dt} s)"


class RK87(Integrator):
    """
    Adaptive Runge-Kutta 7(8) (Dormand-Prince).

    Step size is chosen automatically to meet the tolerance.  The output
    cadence is controlled by ``nOut`` in ``Simulation.integrate()``.

    Parameters
    ----------
    tol : float
        Relative error tolerance.  Default: 1e-10.
    """

    flag = 3

    def __init__(self, tol: float = 1e-10):
        self.tol = float(tol)

    def __repr__(self):
        return f"RK87(tol={self.tol})"


class ABM(Integrator):
    """
    Adams-Bashforth-Moulton 4th-order predictor-corrector, fixed time step.

    Supports all perturbations. Non-symplectic. Generally faster than RK4
    for the same step size.

    Parameters
    ----------
    dt : float
        Time step in seconds.  Default: 1.0.
    """

    flag = 4

    def __init__(self, dt: float = 1.0):
        self.dt = float(dt)

    def __repr__(self):
        return f"ABM(dt={self.dt} s)"
