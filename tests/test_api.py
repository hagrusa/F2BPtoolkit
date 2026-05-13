"""
Exhaustive API accessor tests for F2BPtoolkit.

Covers every property, setter, validator, and repr on every public class
without requiring the compiled C++ extension (_core).  Tests that do need
_core are marked @pytest.mark.integration.
"""

import numpy as np
import pytest

import f2bptoolkit as f2bp
from f2bptoolkit.body import (
    Body, ShapeModel, SphereShape, EllipsoidShape, PolyhedronShape,
)
from f2bptoolkit.state import InitialState
from f2bptoolkit.integrators import RK4, LGVI, RK87, ABM
from f2bptoolkit.perturbations import (
    FlybyPerturbation, HeliocentricPerturbation,
    SolarGravityPerturbation, TidalTorquePerturbation,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_TETRA_VERTS = np.array([
    [1.0,  0.0, -1.0 / np.sqrt(2)],
    [-1.0, 0.0, -1.0 / np.sqrt(2)],
    [0.0,  1.0,  1.0 / np.sqrt(2)],
    [0.0, -1.0,  1.0 / np.sqrt(2)],
])
_TETRA_FACES = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])


def _write_tetra_csv(vpath, fpath):
    ids = np.arange(1, 5, dtype=float).reshape(-1, 1)
    np.savetxt(vpath, np.hstack([ids, _TETRA_VERTS]), delimiter=',', fmt='%.10g')
    np.savetxt(fpath, _TETRA_FACES + 1, delimiter=',', fmt='%d')


def _make_body(name="Test", shape=None, density=2000.0):
    b = Body(name)
    b.shape   = shape or EllipsoidShape(100, 90, 80)
    b.density = density
    return b


def _make_sim():
    """Fully configured two-body sim (no C++ needed until integrate())."""
    sim = f2bp.Simulation()
    sim.add(_make_body("Primary"))
    sim.add(_make_body("Secondary", EllipsoidShape(50, 45, 40)))
    sim.set_state(
        position=[1000.0, 0.0, 0.0],
        velocity=[0.0, 0.1, 0.0],
        omega_primary=[0.0, 0.0, 1e-4],
        omega_secondary=[0.0, 0.0, 1e-4],
    )
    return sim


# ══════════════════════════════════════════════════════════════════════════════
# SphereShape
# ══════════════════════════════════════════════════════════════════════════════

class TestSphereShape:
    def test_radius_stored_as_float(self):
        s = SphereShape(500)
        assert isinstance(s.radius, float)
        assert s.radius == 500.0

    def test_radius_value(self):
        assert SphereShape(123.4).radius == pytest.approx(123.4)

    def test_semi_axes_all_equal_to_radius(self):
        s = SphereShape(300.0)
        assert s.semi_axes() == (300.0, 300.0, 300.0)

    def test_semi_axes_type(self):
        a, b, c = SphereShape(1.0).semi_axes()
        assert a == b == c == 1.0

    def test_repr_contains_class_name(self):
        assert "SphereShape" in repr(SphereShape(100.0))

    def test_repr_contains_radius(self):
        assert "200.0" in repr(SphereShape(200.0))

    def test_is_shape_model(self):
        assert isinstance(SphereShape(1.0), ShapeModel)


# ══════════════════════════════════════════════════════════════════════════════
# EllipsoidShape
# ══════════════════════════════════════════════════════════════════════════════

class TestEllipsoidShape:
    def test_axes_stored_as_float(self):
        s = EllipsoidShape(400, 300, 200)
        assert isinstance(s.a, float)
        assert isinstance(s.b, float)
        assert isinstance(s.c, float)

    def test_axes_values(self):
        s = EllipsoidShape(400.0, 300.0, 200.0)
        assert s.a == 400.0
        assert s.b == 300.0
        assert s.c == 200.0

    def test_semi_axes_returns_abc(self):
        s = EllipsoidShape(1.0, 2.0, 3.0)
        assert s.semi_axes() == (1.0, 2.0, 3.0)

    def test_repr_contains_class_name(self):
        assert "EllipsoidShape" in repr(EllipsoidShape(1, 2, 3))

    def test_repr_contains_all_axes(self):
        r = repr(EllipsoidShape(100.0, 200.0, 300.0))
        assert "100.0" in r
        assert "200.0" in r
        assert "300.0" in r

    def test_is_shape_model(self):
        assert isinstance(EllipsoidShape(1, 2, 3), ShapeModel)


# ══════════════════════════════════════════════════════════════════════════════
# Body
# ══════════════════════════════════════════════════════════════════════════════

class TestBody:

    # ── construction ──────────────────────────────────────────────────────────

    def test_default_name_is_body(self):
        assert Body().name == "Body"

    def test_custom_name_stored(self):
        assert Body("Didymos").name == "Didymos"

    def test_shape_none_before_set(self):
        assert Body().shape is None

    def test_density_none_before_set(self):
        assert Body().density is None

    def test_inertia_order_default_is_2(self):
        assert Body().inertia_order == 2

    # ── shape setter ──────────────────────────────────────────────────────────

    def test_shape_setter_accepts_sphere(self):
        b = Body()
        b.shape = SphereShape(100.0)
        assert isinstance(b.shape, SphereShape)

    def test_shape_setter_accepts_ellipsoid(self):
        b = Body()
        b.shape = EllipsoidShape(100, 90, 80)
        assert isinstance(b.shape, EllipsoidShape)

    def test_shape_setter_accepts_polyhedron(self, tmp_path):
        vp, fp = str(tmp_path/"v.csv"), str(tmp_path/"f.csv")
        _write_tetra_csv(vp, fp)
        b = Body()
        b.shape = PolyhedronShape(vp, fp)
        assert isinstance(b.shape, PolyhedronShape)

    def test_shape_setter_rejects_non_shape_model(self):
        b = Body()
        with pytest.raises(TypeError):
            b.shape = "sphere"

    def test_shape_setter_rejects_int(self):
        b = Body()
        with pytest.raises(TypeError):
            b.shape = 42

    def test_shape_getter_returns_set_value(self):
        b = Body()
        s = SphereShape(50.0)
        b.shape = s
        assert b.shape is s

    # ── density setter ────────────────────────────────────────────────────────

    def test_density_stored_as_float(self):
        b = Body()
        b.density = 2000
        assert isinstance(b.density, float)

    def test_density_positive_value_stored(self):
        b = Body()
        b.density = 2170.0
        assert b.density == pytest.approx(2170.0)

    def test_density_zero_raises(self):
        b = Body()
        with pytest.raises(ValueError):
            b.density = 0.0

    def test_density_negative_raises(self):
        b = Body()
        with pytest.raises(ValueError):
            b.density = -100.0

    def test_density_getter_returns_set_value(self):
        b = Body()
        b.density = 1500.5
        assert b.density == pytest.approx(1500.5)

    # ── inertia_order setter ──────────────────────────────────────────────────

    def test_inertia_order_0_is_valid(self):
        b = Body()
        b.inertia_order = 0
        assert b.inertia_order == 0

    def test_inertia_order_2_is_valid(self):
        b = Body()
        b.inertia_order = 2
        assert b.inertia_order == 2

    def test_inertia_order_4_is_valid(self):
        b = Body()
        b.inertia_order = 4
        assert b.inertia_order == 4

    def test_inertia_order_float_coerced(self):
        b = Body()
        b.inertia_order = 4.0
        assert b.inertia_order == 4
        assert isinstance(b.inertia_order, int)

    def test_inertia_order_odd_raises(self):
        b = Body()
        with pytest.raises(ValueError, match="even"):
            b.inertia_order = 3

    def test_inertia_order_1_raises(self):
        b = Body()
        with pytest.raises(ValueError):
            b.inertia_order = 1

    def test_inertia_order_negative_raises(self):
        b = Body()
        with pytest.raises(ValueError):
            b.inertia_order = -2

    # ── validate ──────────────────────────────────────────────────────────────

    def test_validate_missing_shape_raises(self):
        b = Body("X")
        b.density = 2000.0
        with pytest.raises(ValueError, match="shape"):
            b.validate()

    def test_validate_missing_density_raises(self):
        b = Body("X")
        b.shape = SphereShape(100.0)
        with pytest.raises(ValueError, match="density"):
            b.validate()

    def test_validate_fully_configured_passes(self):
        b = _make_body()
        b.validate()   # must not raise

    # ── internal helpers ──────────────────────────────────────────────────────

    def test_shape_flag_sphere(self):
        b = Body()
        b.shape = SphereShape(100.0)
        b.density = 2000.0
        assert b._shape_flag() == 0

    def test_shape_flag_ellipsoid(self):
        b = Body()
        b.shape = EllipsoidShape(100, 90, 80)
        b.density = 2000.0
        assert b._shape_flag() == 1

    def test_shape_flag_polyhedron(self, tmp_path):
        vp, fp = str(tmp_path/"v.csv"), str(tmp_path/"f.csv")
        _write_tetra_csv(vp, fp)
        b = Body()
        b.shape = PolyhedronShape(vp, fp)
        b.density = 2000.0
        assert b._shape_flag() == 2

    def test_shape_flag_no_shape_raises(self):
        b = Body()
        b.density = 2000.0
        with pytest.raises(ValueError):
            b._shape_flag()

    def test_semi_axes_km_sphere(self):
        b = Body()
        b.shape = SphereShape(2000.0)   # 2000 m = 2 km
        b.density = 2000.0
        assert b._semi_axes_km() == pytest.approx((2.0, 2.0, 2.0))

    def test_semi_axes_km_ellipsoid(self):
        b = Body()
        b.shape = EllipsoidShape(4000.0, 3000.0, 2000.0)
        b.density = 2000.0
        assert b._semi_axes_km() == pytest.approx((4.0, 3.0, 2.0))

    def test_semi_axes_km_polyhedron_returns_zeros(self, tmp_path):
        vp, fp = str(tmp_path/"v.csv"), str(tmp_path/"f.csv")
        _write_tetra_csv(vp, fp)
        b = Body()
        b.shape = PolyhedronShape(vp, fp)
        b.density = 2000.0
        assert b._semi_axes_km() == (0.0, 0.0, 0.0)

    def test_density_kg_km3_conversion(self):
        b = Body()
        b.shape = SphereShape(100.0)
        b.density = 1.0   # 1 kg/m³ → 1e9 kg/km³
        assert b._density_kg_km3() == pytest.approx(1.0e9)

    def test_density_kg_km3_didymos(self):
        b = Body()
        b.shape = SphereShape(100.0)
        b.density = 2170.0
        assert b._density_kg_km3() == pytest.approx(2170.0e9)

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr_contains_name(self):
        b = _make_body("Didymos")
        assert "Didymos" in repr(b)

    def test_repr_contains_density(self):
        b = _make_body()
        assert "2000" in repr(b)

    def test_repr_contains_inertia_order(self):
        b = _make_body()
        assert "inertia_order=2" in repr(b)


# ══════════════════════════════════════════════════════════════════════════════
# InitialState
# ══════════════════════════════════════════════════════════════════════════════

class TestInitialState:

    # ── position ──────────────────────────────────────────────────────────────

    def test_position_none_before_set(self):
        assert InitialState().position is None

    def test_position_setter_from_list(self):
        s = InitialState()
        s.position = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(s.position, [1.0, 2.0, 3.0])

    def test_position_stored_as_float_array(self):
        s = InitialState()
        s.position = [1, 2, 3]
        assert s.position.dtype == float

    def test_position_shape_is_3(self):
        s = InitialState()
        s.position = [1.0, 0.0, 0.0]
        assert s.position.shape == (3,)

    # ── velocity ──────────────────────────────────────────────────────────────

    def test_velocity_none_before_set(self):
        assert InitialState().velocity is None

    def test_velocity_setter_from_list(self):
        s = InitialState()
        s.velocity = [0.0, 0.1, 0.0]
        np.testing.assert_array_equal(s.velocity, [0.0, 0.1, 0.0])

    def test_velocity_stored_as_float_array(self):
        s = InitialState()
        s.velocity = [0, 1, 0]
        assert s.velocity.dtype == float

    def test_velocity_shape_is_3(self):
        s = InitialState()
        s.velocity = [0.0, 0.1, 0.0]
        assert s.velocity.shape == (3,)

    # ── omega_primary ─────────────────────────────────────────────────────────

    def test_omega_primary_none_before_set(self):
        assert InitialState().omega_primary is None

    def test_omega_primary_setter(self):
        s = InitialState()
        s.omega_primary = [0.0, 0.0, 7.26e-4]
        np.testing.assert_array_equal(s.omega_primary, [0.0, 0.0, 7.26e-4])

    def test_omega_primary_shape_is_3(self):
        s = InitialState()
        s.omega_primary = [0.0, 0.0, 1e-4]
        assert s.omega_primary.shape == (3,)

    def test_omega_primary_stored_as_float_array(self):
        s = InitialState()
        s.omega_primary = [0, 0, 1]
        assert s.omega_primary.dtype == float

    # ── omega_secondary ───────────────────────────────────────────────────────

    def test_omega_secondary_none_before_set(self):
        assert InitialState().omega_secondary is None

    def test_omega_secondary_setter(self):
        s = InitialState()
        s.omega_secondary = [0.0, 0.0, 5e-4]
        np.testing.assert_array_equal(s.omega_secondary, [0.0, 0.0, 5e-4])

    def test_omega_secondary_shape_is_3(self):
        s = InitialState()
        s.omega_secondary = [0.0, 0.0, 1e-4]
        assert s.omega_secondary.shape == (3,)

    # ── A_to_N ────────────────────────────────────────────────────────────────

    def test_A_to_N_default_is_identity(self):
        s = InitialState()
        np.testing.assert_array_equal(s.A_to_N, np.eye(3))

    def test_A_to_N_setter_stores_matrix(self):
        s = InitialState()
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
        s.A_to_N = R
        np.testing.assert_array_equal(s.A_to_N, R)

    def test_A_to_N_shape_is_3x3(self):
        assert InitialState().A_to_N.shape == (3, 3)

    def test_A_to_N_stored_as_float_array(self):
        s = InitialState()
        s.A_to_N = np.eye(3, dtype=int)
        assert s.A_to_N.dtype == float

    # ── B_to_A ────────────────────────────────────────────────────────────────

    def test_B_to_A_default_is_identity(self):
        s = InitialState()
        np.testing.assert_array_equal(s.B_to_A, np.eye(3))

    def test_B_to_A_setter_stores_matrix(self):
        s = InitialState()
        R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
        s.B_to_A = R
        np.testing.assert_array_equal(s.B_to_A, R)

    def test_B_to_A_shape_is_3x3(self):
        assert InitialState().B_to_A.shape == (3, 3)

    # ── validate ──────────────────────────────────────────────────────────────

    def _full_state(self):
        s = InitialState()
        s.position        = [1000.0, 0.0, 0.0]
        s.velocity        = [0.0, 0.1, 0.0]
        s.omega_primary   = [0.0, 0.0, 1e-4]
        s.omega_secondary = [0.0, 0.0, 1e-4]
        return s

    def test_validate_missing_position_raises(self):
        s = self._full_state()
        s._position = None
        with pytest.raises(ValueError, match="position"):
            s.validate()

    def test_validate_missing_velocity_raises(self):
        s = self._full_state()
        s._velocity = None
        with pytest.raises(ValueError, match="velocity"):
            s.validate()

    def test_validate_missing_omega_primary_raises(self):
        s = self._full_state()
        s._omega_primary = None
        with pytest.raises(ValueError, match="omega_primary"):
            s.validate()

    def test_validate_missing_omega_secondary_raises(self):
        s = self._full_state()
        s._omega_secondary = None
        with pytest.raises(ValueError, match="omega_secondary"):
            s.validate()

    def test_validate_fully_set_passes(self):
        self._full_state().validate()  # must not raise

    # ── to_vector_km ─────────────────────────────────────────────────────────

    def _state_vector(self):
        s = self._full_state()
        s.position        = [2000.0, 0.0, 0.0]    # m
        s.velocity        = [0.0, 0.5, 0.0]        # m/s
        s.omega_primary   = [0.0, 0.0, 2e-4]       # rad/s
        s.omega_secondary = [0.0, 0.0, 3e-4]       # rad/s
        return s.to_vector_km()

    def test_to_vector_km_shape(self):
        assert self._state_vector().shape == (30,)

    def test_to_vector_km_position_converted_to_km(self):
        v = self._state_vector()
        assert v[0] == pytest.approx(2.0)    # 2000 m → 2 km
        assert v[1] == pytest.approx(0.0)
        assert v[2] == pytest.approx(0.0)

    def test_to_vector_km_velocity_converted_to_km_s(self):
        v = self._state_vector()
        assert v[3] == pytest.approx(0.0)
        assert v[4] == pytest.approx(5e-4)   # 0.5 m/s → 5e-4 km/s
        assert v[5] == pytest.approx(0.0)

    def test_to_vector_km_omega_primary_unchanged(self):
        v = self._state_vector()
        assert v[6]  == pytest.approx(0.0)
        assert v[7]  == pytest.approx(0.0)
        assert v[8]  == pytest.approx(2e-4)

    def test_to_vector_km_omega_secondary_unchanged(self):
        v = self._state_vector()
        assert v[9]  == pytest.approx(0.0)
        assert v[10] == pytest.approx(0.0)
        assert v[11] == pytest.approx(3e-4)

    def test_to_vector_km_A_to_N_default_identity_row_major(self):
        v = self._state_vector()
        np.testing.assert_allclose(v[12:21], np.eye(3).flatten())

    def test_to_vector_km_B_to_A_default_identity_row_major(self):
        v = self._state_vector()
        np.testing.assert_allclose(v[21:30], np.eye(3).flatten())

    def test_to_vector_km_A_to_N_custom_row_major(self):
        s = self._full_state()
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
        s.A_to_N = R
        v = s.to_vector_km()
        np.testing.assert_allclose(v[12:21], R.flatten())

    def test_to_vector_km_B_to_A_custom_row_major(self):
        s = self._full_state()
        R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
        s.B_to_A = R
        v = s.to_vector_km()
        np.testing.assert_allclose(v[21:30], R.flatten())

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr_with_position_set(self):
        s = self._full_state()
        r = repr(s)
        assert "InitialState" in r

    def test_repr_without_position_shows_unset(self):
        s = InitialState()
        assert "unset" in repr(s)


# ══════════════════════════════════════════════════════════════════════════════
# Integrators
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrators:

    # ── RK4 ──────────────────────────────────────────────────────────────────

    def test_rk4_default_dt(self):
        assert RK4().dt == 1.0

    def test_rk4_custom_dt(self):
        assert RK4(dt=30.0).dt == 30.0

    def test_rk4_dt_coerced_to_float(self):
        assert isinstance(RK4(dt=10).dt, float)

    def test_rk4_flag(self):
        assert RK4().flag == 1

    def test_rk4_repr_contains_name(self):
        assert "RK4" in repr(RK4())

    def test_rk4_repr_contains_dt(self):
        assert "30.0" in repr(RK4(dt=30.0))

    # ── LGVI ─────────────────────────────────────────────────────────────────

    def test_lgvi_default_dt(self):
        assert LGVI().dt == 1.0

    def test_lgvi_custom_dt(self):
        assert LGVI(dt=60.0).dt == 60.0

    def test_lgvi_dt_coerced_to_float(self):
        assert isinstance(LGVI(dt=5).dt, float)

    def test_lgvi_flag(self):
        assert LGVI().flag == 2

    def test_lgvi_repr_contains_name(self):
        assert "LGVI" in repr(LGVI())

    def test_lgvi_repr_contains_dt(self):
        assert "60.0" in repr(LGVI(dt=60.0))

    # ── RK87 ─────────────────────────────────────────────────────────────────

    def test_rk87_default_tol(self):
        assert RK87().tol == pytest.approx(1e-10)

    def test_rk87_custom_tol(self):
        assert RK87(tol=1e-12).tol == pytest.approx(1e-12)

    def test_rk87_tol_coerced_to_float(self):
        assert isinstance(RK87(tol=1e-8).tol, float)

    def test_rk87_flag(self):
        assert RK87().flag == 3

    def test_rk87_repr_contains_name(self):
        assert "RK87" in repr(RK87())

    def test_rk87_repr_contains_tol(self):
        assert "1e-12" in repr(RK87(tol=1e-12))

    # ── ABM ──────────────────────────────────────────────────────────────────

    def test_abm_default_dt(self):
        assert ABM().dt == 1.0

    def test_abm_custom_dt(self):
        assert ABM(dt=15.0).dt == 15.0

    def test_abm_dt_coerced_to_float(self):
        assert isinstance(ABM(dt=20).dt, float)

    def test_abm_flag(self):
        assert ABM().flag == 4

    def test_abm_repr_contains_name(self):
        assert "ABM" in repr(ABM())

    def test_abm_repr_contains_dt(self):
        assert "15.0" in repr(ABM(dt=15.0))


# ══════════════════════════════════════════════════════════════════════════════
# Perturbations
# ══════════════════════════════════════════════════════════════════════════════

class TestPerturbations:

    # ── FlybyPerturbation ─────────────────────────────────────────────────────

    def test_flyby_stores_mass(self):
        p = FlybyPerturbation(5.972e24, -1e9, 1.5, 0.0, 0.0, 0.0, 0.0)
        assert p.mass == pytest.approx(5.972e24)

    def test_flyby_stores_semi_major_axis(self):
        p = FlybyPerturbation(1e20, -2e9, 1.5, 0.0, 0.0, 0.0, 0.0)
        assert p.semi_major_axis == pytest.approx(-2e9)

    def test_flyby_stores_eccentricity(self):
        p = FlybyPerturbation(1e20, -1e9, 2.3, 0.0, 0.0, 0.0, 0.0)
        assert p.eccentricity == pytest.approx(2.3)

    def test_flyby_stores_inclination(self):
        p = FlybyPerturbation(1e20, -1e9, 1.5, 0.5, 0.0, 0.0, 0.0)
        assert p.inclination == pytest.approx(0.5)

    def test_flyby_stores_raan(self):
        p = FlybyPerturbation(1e20, -1e9, 1.5, 0.0, 1.2, 0.0, 0.0)
        assert p.raan == pytest.approx(1.2)

    def test_flyby_stores_arg_periapsis(self):
        p = FlybyPerturbation(1e20, -1e9, 1.5, 0.0, 0.0, 0.7, 0.0)
        assert p.arg_periapsis == pytest.approx(0.7)

    def test_flyby_stores_tau(self):
        p = FlybyPerturbation(1e20, -1e9, 1.5, 0.0, 0.0, 0.0, 3600.0)
        assert p.tau == pytest.approx(3600.0)

    def test_flyby_params_stored_as_float(self):
        p = FlybyPerturbation(int(1e20), -int(1e9), 2, 0, 0, 0, 0)
        assert isinstance(p.mass, float)
        assert isinstance(p.eccentricity, float)

    def test_flyby_repr_contains_mass(self):
        p = FlybyPerturbation(5.972e24, -1e9, 1.5, 0.0, 0.0, 0.0, 0.0)
        assert "5.972e+24" in repr(p) or "5.972" in repr(p)

    def test_flyby_repr_contains_class_name(self):
        p = FlybyPerturbation(1e20, -1e9, 1.5, 0.0, 0.0, 0.0, 0.0)
        assert "FlybyPerturbation" in repr(p)

    # ── HeliocentricPerturbation ──────────────────────────────────────────────

    def test_helio_stores_semi_major_axis(self):
        p = HeliocentricPerturbation(1.5 * 1.496e11)
        assert p.semi_major_axis == pytest.approx(1.5 * 1.496e11)

    def test_helio_default_eccentricity_zero(self):
        p = HeliocentricPerturbation(1e11)
        assert p.eccentricity == 0.0

    def test_helio_default_inclination_zero(self):
        p = HeliocentricPerturbation(1e11)
        assert p.inclination == 0.0

    def test_helio_default_raan_zero(self):
        p = HeliocentricPerturbation(1e11)
        assert p.raan == 0.0

    def test_helio_default_arg_periapsis_zero(self):
        p = HeliocentricPerturbation(1e11)
        assert p.arg_periapsis == 0.0

    def test_helio_default_tau_zero(self):
        p = HeliocentricPerturbation(1e11)
        assert p.tau == 0.0

    def test_helio_default_mass_sun(self):
        p = HeliocentricPerturbation(1e11)
        assert p.mass_sun == pytest.approx(1.989e30)

    def test_helio_custom_mass_sun(self):
        p = HeliocentricPerturbation(1e11, mass_sun=2e30)
        assert p.mass_sun == pytest.approx(2e30)

    def test_helio_custom_eccentricity(self):
        p = HeliocentricPerturbation(1e11, eccentricity=0.3)
        assert p.eccentricity == pytest.approx(0.3)

    def test_helio_params_stored_as_float(self):
        p = HeliocentricPerturbation(int(1e11))
        assert isinstance(p.semi_major_axis, float)

    def test_helio_repr_contains_class_name(self):
        assert "HeliocentricPerturbation" in repr(HeliocentricPerturbation(1e11))

    # ── SolarGravityPerturbation ──────────────────────────────────────────────

    def test_solar_stores_solar_radius(self):
        p = SolarGravityPerturbation(1.5)
        assert p.solar_radius == pytest.approx(1.5)

    def test_solar_default_au(self):
        p = SolarGravityPerturbation(1.5)
        assert p.au == pytest.approx(1.496e11)

    def test_solar_custom_au(self):
        p = SolarGravityPerturbation(1.5, au=1.5e11)
        assert p.au == pytest.approx(1.5e11)

    def test_solar_params_stored_as_float(self):
        p = SolarGravityPerturbation(2)
        assert isinstance(p.solar_radius, float)

    def test_solar_repr_contains_class_name(self):
        assert "SolarGravityPerturbation" in repr(SolarGravityPerturbation(1.5))

    def test_solar_repr_contains_radius(self):
        assert "1.5" in repr(SolarGravityPerturbation(1.5))

    # ── TidalTorquePerturbation ───────────────────────────────────────────────

    def test_tidal_stores_love_number_primary(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert p.love_number_primary == pytest.approx(1e-3)

    def test_tidal_stores_love_number_secondary(self):
        p = TidalTorquePerturbation(1e-3, 2e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert p.love_number_secondary == pytest.approx(2e-3)

    def test_tidal_stores_ref_radius_primary(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert p.ref_radius_primary == pytest.approx(400.0)

    def test_tidal_stores_ref_radius_secondary(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert p.ref_radius_secondary == pytest.approx(85.0)

    def test_tidal_stores_lag_angle_primary(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert p.lag_angle_primary == pytest.approx(1e-4)

    def test_tidal_stores_lag_angle_secondary(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 2e-4)
        assert p.lag_angle_secondary == pytest.approx(2e-4)

    def test_tidal_params_stored_as_float(self):
        p = TidalTorquePerturbation(1, 1, 400, 85, 0, 0)
        assert isinstance(p.love_number_primary, float)
        assert isinstance(p.ref_radius_primary, float)

    def test_tidal_repr_contains_class_name(self):
        p = TidalTorquePerturbation(1e-3, 1e-3, 400.0, 85.0, 1e-4, 1e-4)
        assert "TidalTorquePerturbation" in repr(p)

    def test_tidal_repr_contains_love_numbers(self):
        p = TidalTorquePerturbation(1e-3, 2e-3, 400.0, 85.0, 1e-4, 1e-4)
        r = repr(p)
        assert "1e-03" in r or "0.001" in r


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulation:

    # ── construction ──────────────────────────────────────────────────────────

    def test_default_G(self):
        sim = f2bp.Simulation()
        assert sim._G == pytest.approx(6.674e-11)

    def test_custom_G_stored(self):
        sim = f2bp.Simulation(G=6.67430e-11)
        assert sim._G == pytest.approx(6.67430e-11)

    def test_G_km_derived_from_G(self):
        sim = f2bp.Simulation(G=6.674e-11)
        assert sim._G_km == pytest.approx(6.674e-11 / 1e9)

    # ── primary / secondary accessors ─────────────────────────────────────────

    def test_primary_none_before_add(self):
        assert f2bp.Simulation().primary is None

    def test_secondary_none_before_add(self):
        assert f2bp.Simulation().secondary is None

    def test_secondary_none_after_one_body(self):
        sim = f2bp.Simulation()
        sim.add(_make_body())
        assert sim.secondary is None

    def test_primary_set_after_first_add(self):
        sim = f2bp.Simulation()
        b = _make_body("P")
        sim.add(b)
        assert sim.primary is b

    def test_secondary_set_after_second_add(self):
        sim = f2bp.Simulation()
        b1 = _make_body("P")
        b2 = _make_body("S")
        sim.add(b1)
        sim.add(b2)
        assert sim.secondary is b2

    def test_add_third_body_raises(self):
        sim = f2bp.Simulation()
        sim.add(_make_body("P"))
        sim.add(_make_body("S"))
        with pytest.raises(ValueError, match="exactly 2"):
            sim.add(_make_body("X"))

    def test_add_body_missing_shape_raises(self):
        sim = f2bp.Simulation()
        b = Body("bad")
        b.density = 2000.0
        with pytest.raises(ValueError, match="shape"):
            sim.add(b)

    def test_add_body_missing_density_raises(self):
        sim = f2bp.Simulation()
        b = Body("bad")
        b.shape = SphereShape(100.0)
        with pytest.raises(ValueError, match="density"):
            sim.add(b)

    # ── gravity_order ─────────────────────────────────────────────────────────

    def test_gravity_order_default_is_2(self):
        assert f2bp.Simulation().gravity_order == 2

    def test_gravity_order_0_valid(self):
        sim = f2bp.Simulation()
        sim.gravity_order = 0
        assert sim.gravity_order == 0

    def test_gravity_order_2_valid(self):
        sim = f2bp.Simulation()
        sim.gravity_order = 2
        assert sim.gravity_order == 2

    def test_gravity_order_4_valid(self):
        sim = f2bp.Simulation()
        sim.gravity_order = 4
        assert sim.gravity_order == 4

    def test_gravity_order_odd_raises(self):
        sim = f2bp.Simulation()
        with pytest.raises(ValueError):
            sim.gravity_order = 3

    def test_gravity_order_1_raises(self):
        sim = f2bp.Simulation()
        with pytest.raises(ValueError):
            sim.gravity_order = 1

    def test_gravity_order_negative_raises(self):
        sim = f2bp.Simulation()
        with pytest.raises(ValueError):
            sim.gravity_order = -2

    # ── state / set_state ─────────────────────────────────────────────────────

    def test_state_property_returns_initial_state(self):
        sim = f2bp.Simulation()
        assert isinstance(sim.state, InitialState)

    def test_state_created_lazily(self):
        sim = f2bp.Simulation()
        assert sim._state is None
        _ = sim.state
        assert sim._state is not None

    def test_set_state_stores_position(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.position, [1000.0, 0.0, 0.0])

    def test_set_state_stores_velocity(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.velocity, [0.0, 0.1, 0.0])

    def test_set_state_stores_omega_primary(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.omega_primary, [0.0, 0.0, 1e-4])

    def test_set_state_stores_omega_secondary(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.omega_secondary, [0.0, 0.0, 1e-4])

    def test_set_state_A_to_N_defaults_to_identity(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.A_to_N, np.eye(3))

    def test_set_state_B_to_A_defaults_to_identity(self):
        sim = _make_sim()
        np.testing.assert_array_equal(sim.state.B_to_A, np.eye(3))

    def test_set_state_custom_A_to_N(self):
        sim = f2bp.Simulation()
        sim.add(_make_body("P"))
        sim.add(_make_body("S"))
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
        sim.set_state([1000,0,0],[0,0.1,0],[0,0,1e-4],[0,0,1e-4], A_to_N=R)
        np.testing.assert_array_equal(sim.state.A_to_N, R)

    def test_set_state_custom_B_to_A(self):
        sim = f2bp.Simulation()
        sim.add(_make_body("P"))
        sim.add(_make_body("S"))
        R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
        sim.set_state([1000,0,0],[0,0.1,0],[0,0,1e-4],[0,0,1e-4], B_to_A=R)
        np.testing.assert_array_equal(sim.state.B_to_A, R)

    # ── add_perturbation ──────────────────────────────────────────────────────

    def test_add_perturbation_appended(self):
        sim = _make_sim()
        p = FlybyPerturbation(1e22, -1e9, 1.5, 0, 0, 0, 0)
        sim.add_perturbation(p)
        assert p in sim._perturbations

    def test_add_multiple_perturbations(self):
        sim = _make_sim()
        p1 = FlybyPerturbation(1e22, -1e9, 1.5, 0, 0, 0, 0)
        p2 = SolarGravityPerturbation(1.5)
        sim.add_perturbation(p1)
        sim.add_perturbation(p2)
        assert len(sim._perturbations) == 2

    # ── results / plot / analysis / animate before integrate ─────────────────

    def test_results_none_before_integrate(self):
        assert _make_sim().results is None

    def test_plot_before_integrate_raises(self):
        with pytest.raises(RuntimeError, match="No results"):
            _ = _make_sim().plot

    def test_analysis_before_integrate_raises(self):
        with pytest.raises(RuntimeError, match="No results"):
            _ = _make_sim().analysis

    def test_animate_before_integrate_raises(self):
        with pytest.raises(RuntimeError, match="No results"):
            _ = _make_sim().animate

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr_contains_body_names(self):
        sim = _make_sim()
        r = repr(sim)
        assert "Primary" in r
        assert "Secondary" in r

    def test_repr_contains_gravity_order(self):
        sim = _make_sim()
        assert "gravity_order=2" in repr(sim)


# ══════════════════════════════════════════════════════════════════════════════
# nOut validation (requires _core, so integration-marked)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestNOutValidation:
    """nOut validation inside integrate() — requires compiled _core."""

    def _sim(self):
        G = 6.674e-11
        rhoA, aA, bA, cA = 2170.0, 400.0, 395.0, 340.0
        rhoB, aB, bB, cB = 2400.0,  85.0,  73.0,  63.0
        MA = (4/3) * np.pi * aA * bA * cA * rhoA
        MB = (4/3) * np.pi * aB * bB * cB * rhoB
        r0 = 1195.0
        v_c = np.sqrt(G * (MA + MB) / r0)
        n   = np.sqrt(G * (MA + MB) / r0**3)
        sim = f2bp.Simulation()
        p = f2bp.Body("P"); p.shape = f2bp.EllipsoidShape(aA,bA,cA); p.density = rhoA
        s = f2bp.Body("S"); s.shape = f2bp.EllipsoidShape(aB,bB,cB); s.density = rhoB
        sim.add(p); sim.add(s)
        sim.set_state([r0,0,0],[0,v_c,0],[0,0,n],[0,0,n])
        return sim

    def test_nOut_zero_raises(self):
        with pytest.raises(ValueError, match="nOut"):
            self._sim().integrate(100.0, integrator=RK4(dt=10.0), nOut=0)

    def test_nOut_negative_raises(self):
        with pytest.raises(ValueError, match="nOut"):
            self._sim().integrate(100.0, integrator=RK4(dt=10.0), nOut=-1)

    def test_nOut_with_rk87_raises(self):
        with pytest.raises(ValueError, match="RK87"):
            self._sim().integrate(100.0, integrator=RK87(), nOut=5)

    def test_nOut_1_valid(self):
        results = self._sim().integrate(100.0, integrator=RK4(dt=10.0), nOut=1)
        assert results.n_steps >= 2

    def test_lgvi_with_perturbation_raises(self):
        sim = self._sim()
        sim.add_perturbation(FlybyPerturbation(1e22, -1e9, 1.5, 0, 0, 0, 0))
        with pytest.raises(ValueError, match="LGVI"):
            sim.integrate(100.0, integrator=LGVI(dt=10.0))
