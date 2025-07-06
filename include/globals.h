// Global constants and settings for the SIR modeland the controller tuning parameters
// Author: Sarasij Banerjee
// Date: July 5, 2025
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <vector>

// Simulation horizon settings
constexpr int ts = 1; // Sampling time in days
constexpr int Tstop = 100; // Simulation time
constexpr int p = 10; // Prediction horizon
constexpr int m = 8; // Move horizon
constexpr int ant_len = Tstop + p; // Anticipation length


// Controller Model structure. Specific to the SIR model
constexpr int nx = 2;
constexpr int ny = 1;
constexpr int nu = 2;
constexpr int nu_m = 1;
constexpr int nu_d = nu - nu_m;

// System Parameters. Specific to the SIR model
constexpr double Betai = 0.0005f;
constexpr double Gammai = 0.29f;
constexpr double kvi = 0.0f;
constexpr double mu = 0.08f;
constexpr int Br = 500;

// Derived parameters. Specific to the SIR model
constexpr double Si = (mu + Gammai) / Betai;
constexpr double Ii = Br / (mu + Gammai) - (mu + kvi) / Betai;

// controller Initialization. Specific to the SIR model
const Eigen::Matrix<double, ny, 1> y_init = (Eigen::Matrix<double, ny, 1>() << Ii).finished();
const Eigen::Matrix<double, nu_m, 1> u_init = (Eigen::Matrix<double, nu_m, 1>() << Betai).finished();
const Eigen::Matrix<double, nu_d, 1> dm_init = (Eigen::Matrix<double, nu_d, 1>() << kvi).finished();
const Eigen::Matrix<double, ny, 1> du_init = (Eigen::Matrix<double, ny, 1>() << Gammai).finished();

// Controller limits
const Eigen::Matrix<double, ny, 1> y_min = (Eigen::Matrix<double, ny, 1>() << 0).finished(); // Minimum Infected Population
const Eigen::Matrix<double, ny, 1> y_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished(); // Infinity

const Eigen::Matrix<double, ny, 1> u_min = (Eigen::Matrix<double, ny, 1>() << 0).finished(); // Social Contact Limit
const Eigen::Matrix<double, ny, 1> u_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished(); // Infinity

const Eigen::Matrix<double, ny, 1> delu_min = (Eigen::Matrix<double, ny, 1>() << -1e10f).finished(); // Move Size Constraints
const Eigen::Matrix<double, ny, 1> delu_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished();


// MPC Weights
const Eigen::Matrix<double, ny, 1> wy = (Eigen::Matrix<double, ny, 1>() << 1).finished(); // Wy. Can be used for multiple outputs
const Eigen::Matrix<double, nu_m, 1> wdu = (Eigen::Matrix<double, nu_m, 1>() << 0.1f).finished(); // Wdu. Can be used for multiple inputs

// Time constants
const Eigen::Matrix<double, ny, 1> tau_r = (Eigen::Matrix<double, ny, 1>() << 5).finished(); // tau_r. Time constant for the reference signal
const Eigen::Matrix<double, nu_d, 1> tau_d = (Eigen::Matrix<double, nu_d, 1>() << 3).finished(); // tau_d. Time constant for the measured disturbance signal
const Eigen::Matrix<double, ny, 1> tau_u = (Eigen::Matrix<double, ny, 1>() << 1).finished(); // tau_u. Time constant for the unmeasured disturbance signal
constexpr int noise_choice = 1; // Unmeasured disturbance model choice: 1 for single integrating noise, 2 for double integrating. Unmeasured disturbance considered on the output.

// Signals settings
constexpr int ref_time = 10; // Reference signal setpoint change time
constexpr int dm_time = 50; // Measured disturbance step change time
constexpr int du_time = 80; // Unmeasured disturbance step change time

// Signal step changes. Specific to the SIR model
const Eigen::Matrix<double, ny, 1> del_sp = (Eigen::Matrix<double, ny, 1>() << -0.15f * Ii).finished(); // Magnitude of the reference signal step change
const Eigen::Matrix<double, nu_d, 1> del_dm = (Eigen::Matrix<double, nu_d, 1>() << 0.1f).finished(); // Magnitude of the measured disturbance step change
const Eigen::Matrix<double, nu_d, 1> del_du = (Eigen::Matrix<double, nu_d, 1>() << 0.05f).finished();// Magnitude of the unmeasured disturbance step change

// Identity Matrix
template <int Rows> // using a non-type template
Eigen::Matrix<double, Rows, Rows> Identity() {
    return Eigen::Matrix<double, Rows, Rows>::Identity();
}

// Zero Matrix
template <int Rows, int Cols > // using a non-type template
Eigen::Matrix<double, Rows, Cols> Zero_Mat() {
    return Eigen::Matrix<double, Rows, Cols>::Zero();
}


// solver constants for SIR dynamic model
const double dt_init = 0.1f;
const double abs_tol = 1e-6f;
const double rel_tol = 1e-6f;

constexpr int nx_ode = 2; // Number of states in the ODE system. For the true plant

// Conversion to std::vector for plotting Eigen vectors in the debugger
template<typename Derived>
inline std::vector<double> toStdVec(const Eigen::MatrixBase<Derived>& vec) {
    return std::vector<double>(vec.derived().data(), vec.derived().data() + vec.derived().size());
}