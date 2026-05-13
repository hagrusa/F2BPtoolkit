"""Perturbation force/torque classes."""


class Perturbation:
    """Base class for perturbations."""
    pass


class FlybyPerturbation(Perturbation):
    """
    Gravitational perturbation from a third body (e.g., a planetary flyby or
    an asteroid on a hyperbolic trajectory past the binary).

    The perturber is modeled as a point mass on a Keplerian orbit specified by
    classical orbital elements.  Use a negative semi-major axis for hyperbolic
    trajectories (e > 1).

    Parameters
    ----------
    mass : float
        Mass of the perturbing body in kg.
    semi_major_axis : float
        Semi-major axis of the perturbing body's orbit in meters.
        Use negative values for hyperbolic trajectories.
    eccentricity : float
        Eccentricity of the orbit (> 1 for hyperbolic).
    inclination : float
        Inclination in radians.
    raan : float
        Right ascension of ascending node in radians.
    arg_periapsis : float
        Argument of periapsis in radians.
    tau : float
        Time of periapsis passage in seconds.
    """

    def __init__(self, mass: float, semi_major_axis: float, eccentricity: float,
                 inclination: float, raan: float, arg_periapsis: float, tau: float):
        self.mass = float(mass)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.inclination = float(inclination)
        self.raan = float(raan)
        self.arg_periapsis = float(arg_periapsis)
        self.tau = float(tau)

    def __repr__(self):
        return (f"FlybyPerturbation(mass={self.mass:.3e} kg, "
                f"a={self.semi_major_axis:.3e} m, e={self.eccentricity})")


class HeliocentricPerturbation(Perturbation):
    """
    Heliocentric orbit perturbation.

    Models the differential solar gravity acting on the binary as it orbits
    the Sun.  The binary barycenter follows a Keplerian heliocentric orbit.

    Parameters
    ----------
    mass_sun : float
        Solar mass in kg.  Default: 1.989e30.
    semi_major_axis : float
        Heliocentric semi-major axis of the binary in meters.
    eccentricity : float
        Eccentricity of the heliocentric orbit.
    inclination : float
        Inclination in radians.
    raan : float
        Right ascension of ascending node in radians.
    arg_periapsis : float
        Argument of periapsis in radians.
    tau : float
        Time of periapsis passage in seconds.
    """

    def __init__(self, semi_major_axis: float, eccentricity: float = 0.0,
                 inclination: float = 0.0, raan: float = 0.0,
                 arg_periapsis: float = 0.0, tau: float = 0.0,
                 mass_sun: float = 1.989e30):
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.inclination = float(inclination)
        self.raan = float(raan)
        self.arg_periapsis = float(arg_periapsis)
        self.tau = float(tau)
        self.mass_sun = float(mass_sun)

    def __repr__(self):
        return (f"HeliocentricPerturbation(a={self.semi_major_axis:.3e} m, "
                f"e={self.eccentricity})")


class SolarGravityPerturbation(Perturbation):
    """
    Legacy Hill's problem approximation for solar gravity.

    Assumes a planar, circular heliocentric orbit.  Prefer
    ``HeliocentricPerturbation`` for more accurate results.

    Parameters
    ----------
    solar_radius : float
        Heliocentric distance of the binary in AU.
    au : float
        Length of 1 AU in meters.  Default: 1.496e11.
    """

    def __init__(self, solar_radius: float, au: float = 1.496e11):
        self.solar_radius = float(solar_radius)
        self.au = float(au)

    def __repr__(self):
        return f"SolarGravityPerturbation(r_sun={self.solar_radius} AU)"


class TidalTorquePerturbation(Perturbation):
    """
    Tidal torque dissipation (Murray & Dermott model, modified for binaries).

    Applies tidal torques to both bodies and the orbit.  Drives the system
    toward the doubly synchronous equilibrium.

    Parameters
    ----------
    love_number_primary : float
        Tidal Love number k_2 of the primary.
    love_number_secondary : float
        Tidal Love number k_2 of the secondary.
    ref_radius_primary : float
        Reference radius of the primary in meters.
    ref_radius_secondary : float
        Reference radius of the secondary in meters.
    lag_angle_primary : float
        Tidal lag angle of the primary in radians (related to quality factor Q).
    lag_angle_secondary : float
        Tidal lag angle of the secondary in radians.
    """

    def __init__(self, love_number_primary: float, love_number_secondary: float,
                 ref_radius_primary: float, ref_radius_secondary: float,
                 lag_angle_primary: float, lag_angle_secondary: float):
        self.love_number_primary = float(love_number_primary)
        self.love_number_secondary = float(love_number_secondary)
        self.ref_radius_primary = float(ref_radius_primary)
        self.ref_radius_secondary = float(ref_radius_secondary)
        self.lag_angle_primary = float(lag_angle_primary)
        self.lag_angle_secondary = float(lag_angle_secondary)

    def __repr__(self):
        return (f"TidalTorquePerturbation(k2_prim={self.love_number_primary}, "
                f"k2_sec={self.love_number_secondary})")
