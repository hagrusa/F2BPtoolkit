#pragma once
// gubas_core.h — F2BPtoolkit C++ library interface
//
// Physics based on Hou 2016 (Full Two-Body Problem formulation).
// Adapted from GUBAS (General Use Binary Asteroid Simulator).
//
// Units throughout: km, kg, seconds, radians.
// The Python layer is responsible for converting SI (m, kg, s) inputs.

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <armadillo>

using namespace arma;

// ─────────────────────────────────────────────────────────────────────────────
// Internal parameter struct — holds pointers to all computed system parameters.
// Populated by run_simulation() before calling an integrator.
// ─────────────────────────────────────────────────────────────────────────────
struct parameters {
    double*      G;
    cube*        TA;       // primary inertia integrals
    cube*        TB;       // secondary inertia integrals
    cube*        TBp;      // TB rotated from B to A frame
    cube*        TS;       // point-mass inertia integral for 3rd body
    cube*        Tsun;     // point-mass inertia integral for sun
    field<cube>* dT;       // partials of TBp w.r.t. elements of C
    mat*         IA;       // primary principal moments of inertia
    mat*         IB;       // secondary principal moments of inertia
    mat*         IdA;      // modified MOI for LGVI (primary)
    mat*         IdB;      // modified MOI for LGVI (secondary)
    double*      m;        // reduced mass Mc*Ms/(Mc+Ms)
    double*      nu;       // mass ratio Ms/(Ms+Mc)
    mat*         tk;       // tk expansion coefficients
    sp_mat*      a;        // a expansion coefficients
    sp_mat*      b;        // b expansion coefficients
    int*         n;        // mutual potential truncation order
    int*         flyby_toggle;
    int*         helio_toggle;
    int*         sg_toggle;
    int*         tt_toggle;
    double*      Mplanet;
    double*      a_hyp;
    double*      e_hyp;
    double*      i_hyp;
    double*      RAAN_hyp;
    double*      om_hyp;
    double*      tau_hyp;
    double*      n_hyp;
    double*      Msolar;
    double*      a_helio;
    double*      e_helio;
    double*      i_helio;
    double*      RAAN_helio;
    double*      om_helio;
    double*      tau_helio;
    double*      n_helio;
    double*      sol_rad;
    double*      au_def;
    double*      love1;
    double*      love2;
    double*      refrad1;
    double*      refrad2;
    double*      rhoA;
    double*      rhoB;
    double*      eps1;
    double*      eps2;
    double*      mean_motion;
    mat*         sg_acc;
    mat*         acc_3BP;
    mat*         acc_solar;
    mat*         tt_1;
    mat*         tt_2;
    mat*         tt_orbit;
};

// ─────────────────────────────────────────────────────────────────────────────
// SimConfig — user-facing input struct (replaces ic_input.txt + initialization)
// ─────────────────────────────────────────────────────────────────────────────
struct SimConfig {
    // Gravity constant (km³ kg⁻¹ s⁻²)
    double G = 6.674e-20;

    // Mutual potential and inertia integral expansion orders
    int gravity_order = 2;
    int order_a       = 2;   // primary
    int order_b       = 2;   // secondary

    // Primary body
    int    a_shape  = 1;     // 0=sphere, 1=ellipsoid, 2=polyhedron
    double aA = 0.0, bA = 0.0, cA = 0.0;  // semi-axes in km
    double rhoA = 0.0;                      // density in kg/km³
    std::string tet_fileA;
    std::string vert_fileA;

    // Secondary body
    int    b_shape  = 1;
    double aB = 0.0, bB = 0.0, cB = 0.0;
    double rhoB = 0.0;
    std::string tet_fileB;
    std::string vert_fileB;

    // Initial state: [r(3), v(3), wc(3), ws(3), Cc(9), C(9)] in km/km·s⁻¹/rad·s⁻¹
    std::array<double, 30> x0 = {};

    // Integration
    int    integ_flag = 1;   // 1=RK4, 2=LGVI, 3=RK87, 4=ABM
    double t0  = 0.0;
    double tf  = 86400.0;
    double dt  = 1.0;        // fixed time step (s)
    double tol = 1e-10;      // adaptive tolerance
    double out_freq = 0.0;   // output cadence (s); 0 = every step

    // Perturbation toggles and parameters
    int flyby_toggle = 0;
    int helio_toggle = 0;
    int sg_toggle    = 0;
    int tt_toggle    = 0;

    double Mplanet  = 0.0;
    double a_hyp    = -1.0e6;
    double e_hyp    = 1.5;
    double i_hyp    = 0.0;
    double RAAN_hyp = 0.0;
    double om_hyp   = 0.0;
    double tau_hyp  = 0.0;

    double Msolar    = 1.989e30;
    double a_helio   = 2.243e8;   // 1.5 AU in km
    double e_helio   = 0.0;
    double i_helio   = 0.0;
    double RAAN_helio = 0.0;
    double om_helio  = 0.0;
    double tau_helio = 0.0;

    double sol_rad = 1.0;
    double au_def  = 1.496e8;   // km

    double love1   = 0.0;
    double love2   = 0.0;
    double refrad1 = 1.0;
    double refrad2 = 1.0;
    double eps1    = 0.0;
    double eps2    = 0.0;
    double Msun    = 1.989e30;
};

// ─────────────────────────────────────────────────────────────────────────────
// SimResult — simulation output
// ─────────────────────────────────────────────────────────────────────────────
struct SimResult {
    std::string status = "success";

    // Output time series
    std::vector<double> times;           // s
    std::vector<double> states;          // N×30, row-major, km/km·s⁻¹/rad·s⁻¹
    std::vector<double> hyp_states;      // N×6, flyby perturber (km, km/s)
    std::vector<double> solar_states;    // N×6, heliocentric body (km, km/s)

    // Body properties (from inertia integral computation)
    double mass_primary   = 0.0;   // kg
    double mass_secondary = 0.0;   // kg
    std::vector<double> inertia_primary;    // (3,) kg·km²
    std::vector<double> inertia_secondary;  // (3,) kg·km²
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

// Run a full simulation from config → results
SimResult run_simulation(const SimConfig& cfg);

// ─────────────────────────────────────────────────────────────────────────────
// Internal physics function declarations (implemented in gubas_core.cpp)
// ─────────────────────────────────────────────────────────────────────────────
void   tk_calc(int m, mat* t);
void   a_calc(int n, sp_mat* a);
void   b_calc(int n, sp_mat* b);
int    t_ind(int a, int b, int c, int d, int e, int f, int g, int dim);
double Q_ijk(double i, double j, double k);
double tet_sums(double l, double m, double n,
                double x1, double x2, double x3,
                double y1, double y2, double y3,
                double z1, double z2, double z3);
void   poly_inertia_met(int q, double rho,
                        std::string tet_file, std::string vert_file, cube* T);
void   inertia_rot(mat C, int q, cube* T, cube* Tp);
void   poly_moi_met(double rho,
                    std::string tet_file, std::string vert_file, mat* I);
void   ell_mass_params_met(double order, double order_body,
                           double rho, double a, double b, double c,
                           mat* I, cube* T);
double u_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
               mat e, cube* TA, cube* TBp);
double du_dx_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
                   mat e, double R, int dx, cube* TA, cube* TBp);
double de_dx(mat e, double R, int de, int dx);
void   dT_dc(int i, int j, mat C, int q, cube* TA, cube* dT);
double du_dc_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
                   mat e, cube* TA, cube* dT);
double du_x(double G, int m, mat* t, sp_mat* a, sp_mat* b,
            mat e, double R, int dx, cube* TA, cube* TBp);
double du_c(double G, int m, mat* t, sp_mat* a, sp_mat* b,
            mat e, double R, cube* TA, cube* dT);
double potential(double G, int m, mat* t, sp_mat* a, sp_mat* b,
                 mat e, double R, cube* TA, cube* TBp);
mat    hou_ode(mat x, mat t, parameters inputs);
mat    tilde_op(double x, double y, double z);
mat    tilde_opv(mat A);
double factorial(double x);
vec    kepler2cart(double* a_hyp, double* e_hyp, double* i_hyp,
                   double* RAAN_hyp, double* om_hyp, double f0_hyp,
                   double* G, double* Mplanet);
double kepler(double* n_hyp, double t, double* e_hyp, double* tau_hyp);
void   grav_3BP(vec R_s, mat* NA, mat* pos, double* nu,
                double* G, double* Mplanet, mat* acc_3BP);
void   solar_accel(vec R_s, mat* NA, mat* pos, double* nu,
                   double* G, double* Msun, mat* acc_solar);
void   hill_solar_grav(mat* NA, mat* pos, mat* vel, double* n, mat* acc);
void   md_tidal_torque(mat* pos, mat* vel, mat* w1, mat* w2,
                       mat* NA, mat* AB, parameters inputs);
void   map_potential_partials(mat* C, mat* r, parameters inputs,
                              mat* du_dr, mat* M);
void   hamiltonian_map(double h, parameters inputs,
                       mat* x, mat* x_out,
                       mat* fab, mat* fna, mat* Fab, mat* Fna,
                       mat* g_map, mat* G_map, mat* grad_G,
                       mat* du_dr_n, mat* M_n, mat* x0);
void   F_cayley_calc(double h, mat* g, mat* I, mat* f,
                     mat* F, mat* G, mat* grad_G);
void   F_exp_calc(double h, mat* g, mat* I, mat* f,
                  mat* F, mat* G, mat* grad_G);
