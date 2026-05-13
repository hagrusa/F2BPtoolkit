"""
F2BPtoolkit — Full Two-Body Problem toolkit for binary asteroid dynamics.

Quick start
-----------
>>> import f2bptoolkit as f2bp
>>>
>>> sim = f2bp.Simulation()
>>>
>>> primary = f2bp.Body("Didymos")
>>> primary.shape   = f2bp.EllipsoidShape(a=400, b=395, c=340)  # meters
>>> primary.density = 2170.0   # kg/m³
>>> sim.add(primary)
>>>
>>> secondary = f2bp.Body("Dimorphos")
>>> secondary.shape   = f2bp.EllipsoidShape(a=85, b=73, c=63)
>>> secondary.density = 2400.0
>>> sim.add(secondary)
>>>
>>> sim.set_state(
...     position        = [1195e3, 0, 0],       # m
...     velocity        = [0, 0.1735, 0],        # m/s
...     omega_primary   = [0, 0, 7.26e-4],       # rad/s
...     omega_secondary = [0, 0, 7.26e-4],
... )
>>> sim.gravity_order = 2
>>> results = sim.integrate(86400 * 50, integrator=f2bp.RK4(dt=1.0),
...                         nOut=30)
>>> sim.plot.summary()
"""

from .simulation   import Simulation
from .body         import Body, SphereShape, EllipsoidShape, PolyhedronShape
from .integrators  import RK4, LGVI, RK87, ABM
from .perturbations import (
    FlybyPerturbation,
    HeliocentricPerturbation,
    SolarGravityPerturbation,
    TidalTorquePerturbation,
)
from .results      import SimulationResults
from . import spice_utils
from . import analysis
from . import visualization

__version__ = "0.1.0"

__all__ = [
    "Simulation",
    "Body",
    "SphereShape",
    "EllipsoidShape",
    "PolyhedronShape",
    "RK4", "LGVI", "RK87", "ABM",
    "FlybyPerturbation",
    "HeliocentricPerturbation",
    "SolarGravityPerturbation",
    "TidalTorquePerturbation",
    "SimulationResults",
    "spice_utils",
    "analysis",
    "visualization",
]
