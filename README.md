# F2BPtoolkit

A Python/C++ toolkit for simulating the **Full Two-Body Problem (F2BP)** — the
coupled rotational and translational dynamics of two extended, self-gravitating
bodies.  The primary application is binary asteroid systems.

The physics are based on the Hou (2016) formulation and adapted from
[GUBAS](https://github.com/the-gubas-group/GUBAS) (General Use Binary Asteroid
Simulator).  The C++ core is exposed to Python via pybind11, giving a clean,
Rebound-style API while keeping the computation fast.

---

## Features

- **Multiple shape models** — uniform-density sphere, tri-axial ellipsoid, or
  arbitrary polyhedron (vertex/facet files)
- **Mutual gravitational potential** expanded to arbitrary even order via inertia
  integrals (order 0 = point mass, order 2 = J₂/C₂₂, order 4, …)
- **Four integrators**
  - `RK4` — classical 4th-order Runge-Kutta (fixed step)
  - `LGVI` — Lie Group Variational Integrator (symplectic, SO(3)-preserving)
  - `RK87` — adaptive Dormand-Prince 7(8) (variable step)
  - `ABM` — Adams-Bashforth-Moulton 4th-order predictor-corrector (fixed step)
- **Perturbations** (RK4/RK87/ABM only)
  - `FlybyPerturbation` — point-mass third body on a Keplerian orbit
  - `HeliocentricPerturbation` — differential solar gravity along a heliocentric orbit
  - `SolarGravityPerturbation` — Hill's problem approximation for solar gravity
  - `TidalTorquePerturbation` — Murray-Dermott tidal dissipation
- **SPICE integration** via `spiceypy` for loading ephemeris initial conditions
- **Analysis tools** — energy/angular momentum conservation, eccentricity,
  mutual inclination, orbital period estimate
- **Visualization** — matplotlib-based orbit, spin, attitude, and conservation plots

---

## Installation

### Requirements

- Python ≥ 3.9
- [CMake](https://cmake.org/) ≥ 3.15
- [Armadillo](http://arma.sourceforge.net/) C++ linear algebra library
- A C++14 compiler (AppleClang, GCC, MSVC)

**macOS (Homebrew):**
```bash
brew install armadillo cmake
```

**Linux (apt):**
```bash
sudo apt install libarmadillo-dev cmake
```

### Install

```bash
git clone https://github.com/your-username/F2BPtoolkit.git
cd F2BPtoolkit
pip install -e .
```

The build system (scikit-build-core + pybind11) compiles the C++ extension
automatically during `pip install`.

---

## Quick start

```python
import f2bptoolkit as f2bp

sim = f2bp.Simulation()
sim.gravity_order = 2   # include J2/C22 terms

# Primary body
primary = f2bp.Body("Didymos")
primary.shape   = f2bp.EllipsoidShape(a=400, b=395, c=340)   # meters
primary.density = 2170.0   # kg/m³
sim.add(primary)

# Secondary body
secondary = f2bp.Body("Dimorphos")
secondary.shape   = f2bp.EllipsoidShape(a=85, b=73, c=63)
secondary.density = 2400.0
sim.add(secondary)

# Initial state (SI units throughout)
sim.set_state(
    position        = [1195.0, 0.0, 0.0],       # m, in primary body frame
    velocity        = [0.0, 0.1735, 0.0],         # m/s
    omega_primary   = [0.0, 0.0, 7.26e-4],        # rad/s
    omega_secondary = [0.0, 0.0, 7.26e-4],
)

# Run for 50 days with RK4, output every 5 minutes
results = sim.integrate(
    t_final   = 86400 * 50,
    integrator = f2bp.RK4(dt=10.0),
    nOut       = 30,
)

# Built-in plots
sim.plot.summary()
sim.plot.orbit()
sim.plot.spin_rates()
sim.plot.energy_conservation()
```

---

## API reference

### `Simulation`

The main class.  Workflow: add two bodies → set state → (add perturbations) →
integrate → inspect results.

```python
sim = f2bp.Simulation(G=6.674e-11)
sim.gravity_order = 2        # 0, 2, 4, … (default 2)

sim.add(body)                # add primary then secondary
sim.set_state(...)           # see below
sim.add_perturbation(...)    # optional

results = sim.integrate(t_final, integrator=f2bp.RK4(dt=1.0), nOut=30)

sim.results    # SimulationResults object
sim.plot       # PlotInterface
sim.analysis   # AnalysisInterface
```

---

### `Body`

```python
body = f2bp.Body("name")
body.shape         = f2bp.EllipsoidShape(a, b, c)   # meters
body.density       = 2170.0                          # kg/m³
body.inertia_order = 2                               # expansion order (even)
```

**Shape models:**

| Class | Parameters |
|---|---|
| `SphereShape(radius)` | radius in meters |
| `EllipsoidShape(a, b, c)` | tri-axial semi-axes in meters |
| `PolyhedronShape(vertex_file, facet_file)` | CSV files (see below) |

**Polyhedron file format:**

*vertex_file* — columns: `[id, x_m, y_m, z_m]` (coordinates in meters)

*facet_file* — columns: `[v1_idx, v2_idx, v3_idx, ...]` (0-indexed vertex indices per tetrahedron)

---

### `set_state`

All quantities in SI (meters, m/s, rad/s).  Positions and velocities are in
the **primary body frame (A)**.  Attitude matrices default to identity
(body frame aligned with the inertial frame at t=0).

```python
sim.set_state(
    position        = [rx, ry, rz],        # m, secondary w.r.t. primary in A frame
    velocity        = [vx, vy, vz],         # m/s, in A frame
    omega_primary   = [wx, wy, wz],         # rad/s, in A frame
    omega_secondary = [wx, wy, wz],         # rad/s, in secondary body frame (B)
    A_to_N = np.eye(3),   # A→N rotation matrix (optional, default: identity)
    B_to_A = np.eye(3),   # B→A rotation matrix (optional, default: identity)
)
```

---

### Integrators

| Class | Type | Notes |
|---|---|---|
| `RK4(dt=1.0)` | fixed-step | general purpose, all perturbations |
| `LGVI(dt=1.0)` | fixed-step, symplectic | no perturbations; best energy conservation |
| `RK87(tol=1e-10)` | adaptive | automatic step size control |
| `ABM(dt=1.0)` | fixed-step | faster than RK4 for same step size |

---

### Perturbations

```python
# Planetary flyby (hyperbolic trajectory)
sim.add_perturbation(f2bp.FlybyPerturbation(
    mass            = 5.972e24,   # kg
    semi_major_axis = -1e9,       # m (negative for hyperbolic)
    eccentricity    = 1.5,
    inclination     = 0.0,        # rad
    raan            = 0.0,
    arg_periapsis   = 0.0,
    tau             = 0.0,        # time of periapsis in seconds
))

# Heliocentric orbit (differential solar gravity)
sim.add_perturbation(f2bp.HeliocentricPerturbation(
    semi_major_axis = 1.5 * 1.496e11,   # 1.5 AU in meters
    eccentricity    = 0.0,
))

# Tidal dissipation
sim.add_perturbation(f2bp.TidalTorquePerturbation(
    love_number_primary    = 1e-3,
    love_number_secondary  = 1e-3,
    ref_radius_primary     = 400.0,   # m
    ref_radius_secondary   = 85.0,
    lag_angle_primary      = 1e-4,    # rad
    lag_angle_secondary    = 1e-4,
))
```

> **Note:** `LGVI` does not support perturbations.  Use `RK4` or `ABM` when
> perturbations are active.

---

### `SimulationResults`

All output quantities are in SI units.

```python
results.times              # (N,)     seconds
results.position           # (N, 3)   m, in primary body frame
results.velocity           # (N, 3)   m/s
results.omega_primary      # (N, 3)   rad/s, in primary body frame
results.omega_secondary    # (N, 3)   rad/s, in secondary body frame
results.A_to_N   # (N,3,3)  rotation matrices A→N  (v_N = A_to_N @ v_A)
results.B_to_A   # (N,3,3)  rotation matrices B→A  (v_A = B_to_A @ v_B)
results.separation         # (N,)     m, scalar |r|

# Derived quantities (require masses, computed automatically)
results.kinetic_energy_orbital             # (N,) J
results.kinetic_energy_rotation_primary    # (N,) J
results.kinetic_energy_rotation_secondary  # (N,) J
results.angular_momentum                   # (N,3) kg·m²/s
results.angular_momentum_magnitude         # (N,)  kg·m²/s

# Perturber states (populated when a perturbation was active)
results.flyby_position     # (N, 3) m
results.solar_position     # (N, 3) m
```

---

### Analysis interface

```python
a = sim.analysis   # or f2bp.analysis.AnalysisInterface(results)

t, dE  = a.energy_conservation()           # (times, dE/E0)
t, dH  = a.angular_momentum_conservation() # (times, d|H|/|H0|)
e_vec  = a.eccentricity_vector()           # (N, 3) Laplace-Runge-Lenz vector
e      = a.eccentricity()                  # (N,)   instantaneous eccentricity
theta  = a.mutual_inclination()            # (N,)   rad, spin–orbit angle
T_est  = a.orbital_period_estimate()       # float, seconds (None if estimation fails)
wp, ws = a.spin_rates()                    # (N,), (N,) rad/s
```

---

### Visualization interface

```python
p = sim.plot   # or f2bp.visualization.PlotInterface(results)

p.summary()                     # 4-panel dashboard
p.orbit(plane="xy")             # projected 2D orbit
p.orbit_3d()                    # 3D orbit
p.separation()                  # |r| vs time
p.spin_rates()                  # |ω| for both bodies vs time
p.spin_components(body="primary")
p.energy_conservation()
p.angular_momentum_conservation()
p.attitude_evolution(body="primary")
```

All methods return `(fig, ax)` for further customisation:

```python
fig, ax = p.orbit()
ax.set_title("My orbit")
fig.savefig("orbit.png", dpi=150)
```

---

### SPICE utilities

```python
import f2bptoolkit.spice_utils as su

# Get relative state from SPICE kernels
pos, vel = su.state_from_spice(
    kernel_files   = ["didymos.bsp", "naif0012.tls"],
    primary_name   = "DIDYMOS",
    secondary_name = "DIMORPHOS",
    epoch          = "2022-10-01T00:00:00",
)

# Get the N→A rotation matrix from a PCK kernel, then derive A→N
N_to_A = su.rotation_matrix_from_spice(
    body_name  = "DIDYMOS",
    epoch      = "2022-10-01T00:00:00",
    body_frame = "DIDYMOS_FIXED",
)
A_to_N = N_to_A.T

# Get angular velocity in the body frame
omega = su.angular_velocity_from_spice(
    body_name  = "DIDYMOS",
    epoch      = "2022-10-01T00:00:00",
    body_frame = "DIDYMOS_FIXED",
)

# Wire everything into the simulation
sim.set_state(
    position        = N_to_A @ pos,   # rotate inertial → A frame
    velocity        = N_to_A @ vel,
    omega_primary   = omega,
    omega_secondary = [0.0, 0.0, 7.26e-4],
    A_to_N          = A_to_N,
)
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Pure-Python API tests run without the compiled extension.  Integration tests
(marked `@pytest.mark.integration`) are automatically skipped if `_core` is not
built.

---

## State vector convention

The 30-element state vector stored internally is:

```
[r(3), v(3), ω_c(3), ω_s(3), C_c(9), C(9)]
```

| Slice | Symbol | Description | Units |
|---|---|---|---|
| 0:3   | **r**   | Relative position (secondary w.r.t. primary), in A frame | km (internal) / m (Python API) |
| 3:6   | **v**   | Relative velocity, in A frame | km/s / m/s |
| 6:9   | **ω_c** | Primary angular velocity, in A frame | rad/s |
| 9:12  | **ω_s** | Secondary angular velocity, in B frame | rad/s |
| 12:21 | **C_c** | Rotation matrix N→A, row-major | — |
| 21:30 | **C**   | Rotation matrix A→B, row-major | — |

The C++ layer works in **km, kg, s**.  The Python API converts all inputs from
SI (m, m/s) and all outputs back to SI before returning them.

---

## Physics background

The mutual gravitational potential between two extended bodies is expanded as
a double series in inertia integrals following Hou (2016):

```
U = G ∑_{p,q} t_k · a_{pq} · b_{pq} · T_A(p) · T_B(q) / R^(p+q+1)
```

where **T_A** and **T_B** are the inertia integral tensors of the two bodies,
**R** is the separation distance, and the expansion is truncated at order `n`
(set via `sim.gravity_order`).  Order 2 includes the J₂ and C₂₂ gravity
harmonics; higher orders add smaller corrections.

The equations of motion are the full 6-DOF coupled translational/rotational
equations in the primary body frame, including the gravity-gradient torque that
links the orbital and rotational degrees of freedom.

**Reference:**
Hou, X., Scheeres, D.J., & Xin, X. (2016). "Mutual potential between two rigid
bodies with arbitrary shapes and mass distributions." *Celestial Mechanics and
Dynamical Astronomy*, 124(1), 67–82.

---

## License

MIT
