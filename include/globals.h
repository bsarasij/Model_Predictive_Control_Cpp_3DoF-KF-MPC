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

constexpr int u_len_m = nu_m * m;
constexpr int y_len_p = ny * p;
constexpr int Cu_len = 4 * u_len_m + 2 * y_len_p; // To be used in controller lims


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
inline const Eigen::Matrix<double, ny, 1> y_init = (Eigen::Matrix<double, ny, 1>() << Ii).finished();
inline const Eigen::Matrix<double, nu_m, 1> u_init = (Eigen::Matrix<double, nu_m, 1>() << Betai).finished();
inline const Eigen::Matrix<double, nu_d, 1> dm_init = (Eigen::Matrix<double, nu_d, 1>() << kvi).finished();
inline const Eigen::Matrix<double, ny, 1> du_init = (Eigen::Matrix<double, ny, 1>() << Gammai).finished();

// Controller limits
inline const Eigen::Matrix<double, ny, 1> y_min = (Eigen::Matrix<double, ny, 1>() << 0).finished(); // Minimum Infected Population
inline const Eigen::Matrix<double, ny, 1> y_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished(); // Infinity

inline const Eigen::Matrix<double, ny, 1> u_min = (Eigen::Matrix<double, ny, 1>() << 0).finished(); // Social Contact Limit
inline const Eigen::Matrix<double, ny, 1> u_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished(); // Infinity

inline const Eigen::Matrix<double, ny, 1> delu_min = (Eigen::Matrix<double, ny, 1>() << -1e10f).finished(); // Move Size Constraints
inline const Eigen::Matrix<double, ny, 1> delu_max = (Eigen::Matrix<double, ny, 1>() << 1e10f).finished();


// MPC Weights
inline const Eigen::Matrix<double, ny, 1> wy = (Eigen::Matrix<double, ny, 1>() << 1).finished(); // Wy. Can be used for multiple outputs
inline const Eigen::Matrix<double, nu_m, 1> wdu = (Eigen::Matrix<double, nu_m, 1>() << 0.1f).finished(); // Wdu. Can be used for multiple inputs

// Time constants
inline const Eigen::Matrix<double, ny, 1> tau_r = (Eigen::Matrix<double, ny, 1>() << 5).finished(); // tau_r. Time constant for the reference signal
inline const Eigen::Matrix<double, nu_d, 1> tau_d = (Eigen::Matrix<double, nu_d, 1>() << 3).finished(); // tau_d. Time constant for the measured disturbance signal
inline const Eigen::Matrix<double, ny, 1> tau_u = (Eigen::Matrix<double, ny, 1>() << 1).finished(); // tau_u. Time constant for the unmeasured disturbance signal
inline constexpr int noise_choice = 1; // Unmeasured disturbance model choice: 1 for single integrating noise, 2 for double integrating. Unmeasured disturbance considered on the output.

// Signals settings
constexpr int ref_time = 10; // Reference signal setpoint change time
constexpr int dm_time = 50; // Measured disturbance step change time
constexpr int du_time = 80; // Unmeasured disturbance step change time

// Signal step changes. Specific to the SIR model
inline const Eigen::Matrix<double, ny, 1> del_sp = (Eigen::Matrix<double, ny, 1>() << -0.15f * Ii).finished(); // Magnitude of the reference signal step change
inline const Eigen::Matrix<double, nu_d, 1> del_dm = (Eigen::Matrix<double, nu_d, 1>() << 0.1f).finished(); // Magnitude of the measured disturbance step change
inline const Eigen::Matrix<double, nu_d, 1> del_du = (Eigen::Matrix<double, nu_d, 1>() << 0.05f).finished();// Magnitude of the unmeasured disturbance step change

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
inline const double dt_init = 0.1f;
inline const double abs_tol = 1e-6f;
inline const double rel_tol = 1e-6f;

constexpr int nx_ode = 2; // Number of states in the ODE system. For the true plant

// Conversion to std::vector for plotting Eigen vectors in the debugger
template<typename Derived>
inline std::vector<double> toStdVec(const Eigen::MatrixBase<Derived>& vec) {
    return std::vector<double>(vec.derived().data(), vec.derived().data() + vec.derived().size());
}