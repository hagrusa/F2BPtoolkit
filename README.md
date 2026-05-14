# F2BPtoolkit

A Python/C++ toolkit for simulating the **Full Two-Body Problem (F2BP)** — the coupled rotational and translational dynamics of two extended, rigid bodies. The primary application is binary asteroid systems.

The physics follow the [Hou et al. (2016)](#references) formulation of the F2BP and are adapted from [GUBAS](https://github.com/meyeralexj/gubas) (General Use Binary Asteroid Simulator). The C++ core is exposed to Python via pybind11.

> **Note:** This package is experimental. Significant portions of the code remain untested (effectively all of it). Use at your own risk. Please let me know if you find mistakes.

If you use this package, please cite the references listed [below](#references).

---

## Features

- **Shape models** — uniform-density sphere, tri-axial ellipsoid, or arbitrary polyhedron (vertex/facet files)
- **Mutual gravitational potential** expanded to arbitrary order via inertia integrals
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
- **Analysis tools** — energy/angular momentum conservation, eccentricity, mutual inclination, orbital period estimate
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

The build system (scikit-build-core + pybind11) compiles the C++ extension automatically during `pip install`.

---

## Quick start

```python
import f2bptoolkit as f2bp

sim = f2bp.Simulation()
sim.gravity_order = 2   # include J2/C22 terms

# Primary body
primary = f2bp.Body("Didymos")
primary.shape   = f2bp.EllipsoidShape(a=400, b=395, c=340)   # meters
primary.density = 2170.0                                       # kg/m³
sim.add(primary)

# Secondary body
secondary = f2bp.Body("Dimorphos")
secondary.shape   = f2bp.EllipsoidShape(a=85, b=73, c=63)
secondary.density = 2400.0
sim.add(secondary)

# Initial state (SI units throughout)
sim.set_state(
    position        = [1195.0, 0.0, 0.0],   # m, in primary body frame
    velocity        = [0.0, 0.1735, 0.0],    # m/s
    omega_primary   = [0.0, 0.0, 7.26e-4],  # rad/s
    omega_secondary = [0.0, 0.0, 7.26e-4],   # rad/s, in primary body frame (A)
)

# Run for 50 days with RK4, recording every 30 steps
results = sim.integrate(
    t_final    = 86400 * 50,
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

```python
sim = f2bp.Simulation(G=6.674e-11)
sim.gravity_order = 2        # 0, 2, 4, … (default 2)

sim.add(body)                # add primary then secondary
sim.set_state(...)
sim.add_perturbation(...)    # optional

results = sim.integrate(t_final, integrator=f2bp.RK4(dt=1.0), nOut=30)

sim.results    # SimulationResults object
sim.plot       # PlotInterface
sim.analysis   # AnalysisInterface
sim.animate    # AnimationInterface (after integrate())
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

*vertex_file* — columns: `[id, x_m, y_m, z_m]`

*facet_file* — columns: `[v1_idx, v2_idx, v3_idx, ...]` (0-indexed vertex indices per tetrahedron)

---

### `set_state`

All quantities in SI (meters, m/s, rad/s). Positions and velocities are in the **primary body frame (A)**. Attitude matrices default to identity (body frames aligned with the inertial frame at t=0).

```python
sim.set_state(
    position        = [rx, ry, rz],   # m, secondary w.r.t. primary in A frame
    velocity        = [vx, vy, vz],   # m/s, in A frame
    omega_primary   = [wx, wy, wz],   # rad/s, in A frame
    omega_secondary = [wx, wy, wz],   # rad/s, in primary body frame (A)
    A_to_N = np.eye(3),               # A→N rotation matrix (optional)
    B_to_A = np.eye(3),               # B→A rotation matrix (optional)
)
```

---

### Integrators

| Class | Type | Notes |
|---|---|---|
| `RK4(dt=1.0)` | fixed-step | general purpose; supports all perturbations |
| `LGVI(dt=1.0)` | fixed-step, symplectic | no perturbations; best long-term energy behavior |
| `RK87(tol=1e-10)` | adaptive | automatic step-size control |
| `ABM(dt=1.0)` | fixed-step | faster than RK4 at equivalent step size |

> `LGVI` does not support perturbations. Use `RK4` or `ABM` when perturbations are active.

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
    tau             = 0.0,        # time of periapsis (s)
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
    lag_angle_primary      = 1e-4,   # rad
    lag_angle_secondary    = 1e-4,
))
```

---

### `SimulationResults`

All output quantities are in SI units.

```python
results.times              # (N,)     seconds
results.position           # (N, 3)   m, in primary body frame
results.velocity           # (N, 3)   m/s
results.omega_primary      # (N, 3)   rad/s, in primary body frame
results.omega_secondary    # (N, 3)   rad/s, in secondary body frame
results.A_to_N             # (N,3,3)  rotation matrices A→N
results.B_to_A             # (N,3,3)  rotation matrices B→A
results.separation         # (N,)     m, scalar |r|

# Derived quantities (computed automatically from masses)
results.kinetic_energy_orbital             # (N,) J
results.kinetic_energy_rotation_primary    # (N,) J
results.kinetic_energy_rotation_secondary  # (N,) J
results.angular_momentum                   # (N,3) kg·m²/s
results.angular_momentum_magnitude         # (N,)  kg·m²/s

# Perturber states (populated when a perturbation is active)
results.flyby_position     # (N, 3) m
results.solar_position     # (N, 3) m
```

---

### Analysis interface

```python
a = sim.analysis

t, dE  = a.energy_conservation()           # (times, dE/E0)
t, dH  = a.angular_momentum_conservation() # (times, d|H|/|H0|)
e_vec  = a.eccentricity_vector()           # (N, 3) Laplace-Runge-Lenz vector
e      = a.eccentricity()                  # (N,)   instantaneous eccentricity
theta  = a.mutual_inclination()            # (N,)   rad, spin-orbit angle
T_est  = a.orbital_period_estimate()       # float, seconds
wp, ws = a.spin_rates()                    # (N,), (N,) rad/s
```

---

### Visualization interface

```python
p = sim.plot

p.summary()                       # 4-panel dashboard
p.orbit(plane="xy")               # projected 2D orbit
p.orbit_3d()                      # 3D orbit
p.separation()                    # |r| vs time
p.spin_rates()                    # |ω| for both bodies vs time
p.spin_components(body="primary")
p.energy_conservation()
p.angular_momentum_conservation()
p.attitude_evolution(body="primary")
```

All methods return `(fig, ax)` for further customization:

```python
fig, ax = p.orbit()
ax.set_title("My orbit")
fig.savefig("orbit.png", dpi=150)
```

---

### SPICE utilities

```python
import f2bptoolkit.spice_utils as su

pos, vel = su.state_from_spice(
    kernel_files   = ["didymos.bsp", "naif0012.tls"],
    primary_name   = "DIDYMOS",
    secondary_name = "DIMORPHOS",
    epoch          = "2022-10-01T00:00:00",
)

N_to_A = su.rotation_matrix_from_spice(
    body_name  = "DIDYMOS",
    epoch      = "2022-10-01T00:00:00",
    body_frame = "DIDYMOS_FIXED",
)

omega = su.angular_velocity_from_spice(
    body_name  = "DIDYMOS",
    epoch      = "2022-10-01T00:00:00",
    body_frame = "DIDYMOS_FIXED",
)

sim.set_state(
    position        = N_to_A @ pos,
    velocity        = N_to_A @ vel,
    omega_primary   = omega,
    omega_secondary = [0.0, 0.0, 7.26e-4],
    A_to_N          = N_to_A.T,
)
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Pure-Python API tests run without the compiled extension. Integration tests (marked `@pytest.mark.integration`) are skipped automatically if the C++ extension is not built.

---

## State vector convention

The 30-element state vector stored internally is `[r(3), v(3), ω_c(3), ω_s(3), C_c(9), C(9)]`:

| Slice | Symbol | Description | Units |
|---|---|---|---|
| 0:3   | **r**   | Relative position (secondary w.r.t. primary), in A frame | km (internal) / m (API) |
| 3:6   | **v**   | Relative velocity, in A frame | km/s / m/s |
| 6:9   | **ω_c** | Primary angular velocity, in A frame | rad/s |
| 9:12  | **ω_s** | Secondary angular velocity, in A frame | rad/s |
| 12:21 | **C_c** | Rotation matrix N→A, row-major (`results.A_to_N = C_c.T`) | — |
| 21:30 | **C**   | Rotation matrix A→B, row-major (`results.B_to_A = C.T`) | — |

The C++ layer works in **km, kg, s**. The Python API converts all inputs from SI and all outputs back to SI before returning them. The internal matrices `C_c` (N→A) and `C` (A→B) are exposed as their transposes — `A_to_N` and `B_to_A` — so that the transform direction matches the attribute name: `v_N = A_to_N @ v_A`, `v_A = B_to_A @ v_B`.

---

## References

- Hou, X., Scheeres, D.J., & Xin, X. (2016). Mutual potential between two rigid bodies with arbitrary shapes and mass distributions. *Celestial Mechanics and Dynamical Astronomy*, 124(1), 67–82. https://doi.org/10.1007/s10569-015-9646-0
- Davis, A.B. & Scheeres, D.J. (2020). Doubly synchronous binary asteroid mass parameter observability. *Icarus*, 341, 113439. https://doi.org/10.1016/j.icarus.2019.113439

And if you use the tidal torque, flyby, or solar tide perturbations:
- Meyer, A.J. & Scheeres, D.J. (2021). The effect of planetary flybys on singly synchronous binary asteroids. *Icarus*, 367, 114554. https://doi.org/10.1016/j.icarus.2021.114554 
- Meyer, A.J., Scheeres, D.J., Agrusa, H.F., Noiset, G., McMahon, J., Karatekin, O., Hirabayashi, M., & Nakano, R. (2023). Energy dissipation in synchronous binary asteroids. *Icarus*, 391, 115323. https://doi.org/10.1016/j.icarus.2022.115323 

---

## License

MIT
