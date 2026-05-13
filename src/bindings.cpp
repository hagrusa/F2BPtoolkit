// bindings.cpp — pybind11 bindings for F2BPtoolkit
//
// Exposes SimConfig, SimResult, and run_simulation() as the Python
// extension module `_core`.
//
// Python layer (simulation.py) works entirely in SI units (m, kg, s).
// This layer merely passes values through; the C++ side expects km/kg/s.
// All unit conversion is done in simulation.py before calling run_simulation.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // std::vector, std::string, std::array
#include "gubas_core.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "F2BPtoolkit C++ extension — Full Two-Body Problem integrator";

    // ── SimConfig ─────────────────────────────────────────────────────────────
    py::class_<SimConfig>(m, "SimConfig",
        "Input configuration for a F2BP simulation.\n\n"
        "All length quantities in km, masses in kg, times in seconds.\n"
        "The Python Simulation class is responsible for unit conversion.")
        .def(py::init<>())

        // Gravity / expansion orders
        .def_readwrite("G",             &SimConfig::G)
        .def_readwrite("gravity_order", &SimConfig::gravity_order)
        .def_readwrite("order_a",       &SimConfig::order_a)
        .def_readwrite("order_b",       &SimConfig::order_b)

        // Primary shape
        .def_readwrite("a_shape",    &SimConfig::a_shape)
        .def_readwrite("aA",         &SimConfig::aA)
        .def_readwrite("bA",         &SimConfig::bA)
        .def_readwrite("cA",         &SimConfig::cA)
        .def_readwrite("rhoA",       &SimConfig::rhoA)
        .def_readwrite("tet_fileA",  &SimConfig::tet_fileA)
        .def_readwrite("vert_fileA", &SimConfig::vert_fileA)

        // Secondary shape
        .def_readwrite("b_shape",    &SimConfig::b_shape)
        .def_readwrite("aB",         &SimConfig::aB)
        .def_readwrite("bB",         &SimConfig::bB)
        .def_readwrite("cB",         &SimConfig::cB)
        .def_readwrite("rhoB",       &SimConfig::rhoB)
        .def_readwrite("tet_fileB",  &SimConfig::tet_fileB)
        .def_readwrite("vert_fileB", &SimConfig::vert_fileB)

        // Initial state: list/array of 30 doubles
        // pybind11/stl.h converts between Python list and std::array<double,30>
        .def_readwrite("x0", &SimConfig::x0)

        // Integration settings
        .def_readwrite("integ_flag", &SimConfig::integ_flag)
        .def_readwrite("t0",         &SimConfig::t0)
        .def_readwrite("tf",         &SimConfig::tf)
        .def_readwrite("dt",         &SimConfig::dt)
        .def_readwrite("tol",        &SimConfig::tol)
        .def_readwrite("out_freq",   &SimConfig::out_freq)

        // Perturbation toggles
        .def_readwrite("flyby_toggle", &SimConfig::flyby_toggle)
        .def_readwrite("helio_toggle", &SimConfig::helio_toggle)
        .def_readwrite("sg_toggle",    &SimConfig::sg_toggle)
        .def_readwrite("tt_toggle",    &SimConfig::tt_toggle)

        // Flyby perturber parameters
        .def_readwrite("Mplanet",  &SimConfig::Mplanet)
        .def_readwrite("a_hyp",    &SimConfig::a_hyp)
        .def_readwrite("e_hyp",    &SimConfig::e_hyp)
        .def_readwrite("i_hyp",    &SimConfig::i_hyp)
        .def_readwrite("RAAN_hyp", &SimConfig::RAAN_hyp)
        .def_readwrite("om_hyp",   &SimConfig::om_hyp)
        .def_readwrite("tau_hyp",  &SimConfig::tau_hyp)

        // Heliocentric orbit parameters
        .def_readwrite("Msolar",     &SimConfig::Msolar)
        .def_readwrite("a_helio",    &SimConfig::a_helio)
        .def_readwrite("e_helio",    &SimConfig::e_helio)
        .def_readwrite("i_helio",    &SimConfig::i_helio)
        .def_readwrite("RAAN_helio", &SimConfig::RAAN_helio)
        .def_readwrite("om_helio",   &SimConfig::om_helio)
        .def_readwrite("tau_helio",  &SimConfig::tau_helio)

        // Solar radiation / Hill gravity
        .def_readwrite("sol_rad", &SimConfig::sol_rad)
        .def_readwrite("au_def",  &SimConfig::au_def)

        // Tidal torque parameters
        .def_readwrite("love1",   &SimConfig::love1)
        .def_readwrite("love2",   &SimConfig::love2)
        .def_readwrite("refrad1", &SimConfig::refrad1)
        .def_readwrite("refrad2", &SimConfig::refrad2)
        .def_readwrite("eps1",    &SimConfig::eps1)
        .def_readwrite("eps2",    &SimConfig::eps2)
        .def_readwrite("Msun",    &SimConfig::Msun)

        .def("__repr__", [](const SimConfig& c) {
            return "<SimConfig G=" + std::to_string(c.G) +
                   " order=" + std::to_string(c.gravity_order) +
                   " tf=" + std::to_string(c.tf) + ">";
        });

    // ── SimResult ─────────────────────────────────────────────────────────────
    py::class_<SimResult>(m, "SimResult",
        "Output from run_simulation().\n\n"
        "Lengths in km, velocities in km/s, angular velocities in rad/s.\n"
        "Masses in kg, inertia in kg·km².\n"
        "The Python SimulationResults class converts km → m.")
        .def(py::init<>())

        .def_readwrite("status",            &SimResult::status)
        .def_readwrite("times",             &SimResult::times)
        .def_readwrite("states",            &SimResult::states)
        .def_readwrite("hyp_states",        &SimResult::hyp_states)
        .def_readwrite("solar_states",      &SimResult::solar_states)
        .def_readwrite("mass_primary",      &SimResult::mass_primary)
        .def_readwrite("mass_secondary",    &SimResult::mass_secondary)
        .def_readwrite("inertia_primary",   &SimResult::inertia_primary)
        .def_readwrite("inertia_secondary", &SimResult::inertia_secondary)

        .def("__repr__", [](const SimResult& r) {
            return "<SimResult status='" + r.status +
                   "' n_steps=" + std::to_string(r.times.size()) + ">";
        });

    // ── run_simulation ────────────────────────────────────────────────────────
    m.def("run_simulation", &run_simulation,
          py::arg("cfg"),
          py::call_guard<py::gil_scoped_release>(),
          "Run a Full Two-Body Problem simulation.\n\n"
          "Parameters\n"
          "----------\n"
          "cfg : SimConfig\n"
          "    Fully populated configuration struct.\n\n"
          "Returns\n"
          "-------\n"
          "SimResult\n"
          "    Times (s), states (N×30, km/km·s/rad·s), masses (kg),\n"
          "    inertia tensors (kg·km²), and optional perturber states.");
}
