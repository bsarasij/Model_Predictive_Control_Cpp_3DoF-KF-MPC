//Header1.h
// Defines all the necessary includes, constants, and classes for the SIR model and the controller matrices and signals
// Author: Sarasij Banerjee
// Date: July 5, 2025

#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>
#include <boost/numeric/odeint.hpp>
#include "globals.h"
#include <osqp.h>
#include <OsqpEigen/OsqpEigen.h>
#include "gnuplot-iostream.h"

using namespace Eigen;
using namespace boost::numeric::odeint;



// The Matrices class encapsulates all the key matrices and related computations required for the controller's predictive and noise model.
//
// Structures:
// - SS_Mat: Contains the discrete-time state-space matrices (Ad, Bd_m, Bd_d, Cd) for the SIR model, where Ad is the state matrix, 
//   Bd_m and Bd_d are the input matrices for manipulated and disturbance inputs, and Cd is the output matrix. A, B, C are the continous-time 
// state-space matrices which get discretized to give (Ad, Bd_m, Bd_d, Cd). Replace (A, B, C, D) with your problem-specific matrices. 
// Right now I am using linearized version of SIR ode to generate my matrices. It can be replaced with any data-driven model as well.
// 
// - Ctrl_Mat: Holds the controller prediction matrices (Sx, Su, Sd, Ip) used in model predictive control (MPC). Sx maps the  
//   state to output space, Su and Sd map future manipulated and disturbance inputs to outputs, and Ip is a block identity matrix 
//   for output stacking.
// 
// - Noise_Model_Mat: Defines the noise model matrices (Aw, Cw) for unmeasured disturbances, supporting both single and double 
//   integrating noise models. This allows for offset-free MPPC design.
// 
// - Aug_SS_Mat: Contains the augmented state-space matrices (A_aug, Bm_aug, Bd_aug, C_aug) that combine the plant and noise models 
//   for robust control design. These matrices are used to propagate the augmented state, noise, and output vectors inside the controller.
//
// Members:
// - ss_mat: Instance of SS_Mat, initialized with the discretized SIR model matrices.
// - noise_mat: Instance of Noise_Model_Mat, initialized based on the selected noise model type.
// - aug_mat: Instance of Aug_SS_Mat, constructed by augmenting the plant and noise models.
// - ctrl_mat: Instance of Ctrl_Mat, constructed for use in MPC optimization.
//
// Key Functions:
// - Matrices(): Constructor that initializes all matrix structures by calling the relevant computation functions.
// - static Noise_Model_Mat eval_noise_model(int ch): Returns the noise model matrices for the specified type (single/double integrating).
// - SS_Mat compute_mat(): Computes and discretizes the continuous-time SIR model matrices.
// - Aug_SS_Mat eval_aug_matrices(SS_Mat&, Noise_Model_Mat&): Builds the augmented state-space matrices from the plant and noise models.
// - Ctrl_Mat controller_mat(Aug_SS_Mat): Constructs the prediction matrices (Sx, Su, Sd, Ip) for MPC based on the augmented model.
// - auto discretize(Eigen::Matrix<double, nx, nx>&, const Eigen::Matrix<double, nx, nu>&): Discretizes the continuous-time system 
//   matrices using matrix exponentials and computes the discrete input matrices.
//
// The Matrices class is intended to be instantiated once, providing all necessary matrices for controller design, simulation, and 
// optimization in a structured and reusable manner.


class Matrices {
public:
    struct SS_Mat {
        Eigen::Matrix<double, nx, nx> Ad;
        Eigen::Matrix<double, nx, nu_m> Bd_m;
        Eigen::Matrix<double, nx, nu_d> Bd_d;
        Eigen::Matrix<double, ny, nx> Cd;
    };

    struct Ctrl_Mat {
        Eigen::Matrix<double, ny* p, nx+ 2*ny> Sx;
        Eigen::Matrix<double, ny* p, nu_m* m> Su;
        Eigen::Matrix<double, ny* p, nu_d* m> Sd;
        Eigen::Matrix<double, ny* p, ny> Ip;
    };

    struct Noise_Model_Mat {
        Eigen::Matrix<double, ny, ny> Aw;
        Eigen::Matrix<double, ny, ny> Cw;
    };

    struct Aug_SS_Mat {
        Eigen::Matrix<double, nx+2*ny, nx + 2 * ny> A_aug;
        Eigen::Matrix<double, nx + 2 * ny, nu_m> Bm_aug;
        Eigen::Matrix<double, nx + 2 * ny, nu_d> Bd_aug;
        Eigen::Matrix<double, ny, nx + 2 * ny> C_aug;

    };

    SS_Mat ss_mat;
    Noise_Model_Mat noise_mat;
    Aug_SS_Mat aug_mat;
    Ctrl_Mat ctrl_mat;

    Matrices() {
        ss_mat = compute_mat();
        noise_mat = eval_noise_model(noise_choice);
        aug_mat = eval_aug_matrices(ss_mat, noise_mat);
        ctrl_mat = controller_mat(aug_mat);
    }

    static Noise_Model_Mat eval_noise_model(int ch) {
        Eigen::Matrix<double, ny, ny> Aw;
        Eigen::Matrix<double, ny, ny> Cw;

        switch (ch) {
        case 1: //single integrating
            Aw = Zero_Mat<ny, ny>();
            break;
        case 2: //double integrating
            Aw = Identity<ny>();
            break;
        default:
            throw std::invalid_argument("Unsupported Unmeasured Disturbance Type");
        }

        Cw = Identity<ny>();
        return Noise_Model_Mat{ Aw, Cw };
    }

private:
    Aug_SS_Mat eval_aug_matrices(SS_Mat& ss_mat, Noise_Model_Mat& noise_mat) {
        Eigen::Matrix<double, nx + 2 * ny, nx + 2 * ny> A_aug;
        Eigen::Matrix<double, nx + 2 * ny, nu_m> Bm_aug;
        Eigen::Matrix<double, nx + 2 * ny, nu_d> Bd_aug;
        Eigen::Matrix<double, ny, nx + 2 * ny> C_aug;
        
        A_aug << ss_mat.Ad, Zero_Mat<nx, ny>(), Zero_Mat<nx, ny>(),
            Zero_Mat<ny, nx>(), noise_mat.Aw, Zero_Mat<ny, ny>(),
            ss_mat.Cd* ss_mat.Ad, noise_mat.Aw, Identity<ny>();

        Bm_aug << ss_mat.Bd_m,
            Zero_Mat<ny, nu_m>(),
            ss_mat.Cd* ss_mat.Bd_m;

        Bd_aug << ss_mat.Bd_d,
            Zero_Mat<ny, nu_d>(),
            ss_mat.Cd* ss_mat.Bd_d;

        C_aug << Zero_Mat<ny, nx>(), Zero_Mat<ny, ny>(), Identity<ny>();
        return Aug_SS_Mat{ A_aug, Bm_aug, Bd_aug, C_aug };
    } 

    Ctrl_Mat controller_mat(Aug_SS_Mat aug_mat) {
        Eigen::Matrix<double, ny* p, nx + 2 * ny> Sx;
        Eigen::Matrix<double, ny* p, nu_m* m> Su;
        Eigen::Matrix<double, ny* p, nu_d* m> Sd;

        Sx.setZero();
        Su.setZero();
        Sd.setZero();


        Eigen::Matrix<double, nx + 2*ny , nx + 2*ny> A_power = aug_mat.A_aug;

        Sx.block(0, 0, ny, nx + 2*ny) = aug_mat.C_aug * aug_mat.A_aug;
        for (int i = 1; i < p; i += 1) {
            Sx.block(i * ny, 0, ny, nx+2*ny) = aug_mat.C_aug * A_power;
            A_power *= aug_mat.A_aug;
        }

        Matrix<double, nx + 2 * ny, nx + 2 * ny> Apow = Identity<nx + 2 * ny>();
        std::array<Eigen::Matrix<double, ny, nu_m>, p> CABm_blocks;
        std::array<Eigen::Matrix<double, ny, nu_d>, p> CABd_blocks;

        for (int i = 0; i < p; ++i) {
            Apow *= aug_mat.A_aug;
            CABm_blocks[i] = aug_mat.C_aug * Apow * aug_mat.Bm_aug;
            CABd_blocks[i] = aug_mat.C_aug * Apow * aug_mat.Bd_aug;
        }

        for (int row = 0; row < p; ++row) {
            for (int col = 0; col < m; ++col) {
                if (row >= col) {
                    Su.block<ny, nu_m>(row * ny, col * nu_m) = CABm_blocks[row - col];
                    Sd.block<ny, nu_d>(row * ny, col * nu_d) = CABd_blocks[row - col];

                }
            }
        }



        auto I_y = Identity<ny>();
        auto ones = Matrix<double, p, 1>::Ones();
        auto Ip = Eigen::kroneckerProduct(ones, I_y);
        return Ctrl_Mat{ Sx, Su, Sd, Ip };
    }

    auto discretize(Eigen::Matrix<double, nx, nx>& A, const Eigen::Matrix<double, nx, nu>& B) {
        Eigen::Matrix<double, nx, nx> Ad = (A * ts).exp();
        auto I = Identity<nx>();
        Eigen::Matrix<double, nx, nu> Bd = A.inverse() * (Ad - I) * B;
        Eigen::Matrix<double, nx, nu_m> Bd_m = Bd.block(0, 0, nx, nu_m);
        Eigen::Matrix<double, nx, nu_d> Bd_d = Bd.block(0, nu_m, nx, nu_d);

        return std::make_tuple(Ad, Bd_m, Bd_d);
    }


    SS_Mat compute_mat() {
        Eigen::Matrix<double, nx, nx> A;
        Eigen::Matrix<double, nx, nu> B;
        Eigen::Matrix<double, ny, nx> C;

        A << -Betai * Ii - mu - kvi, -Betai * Si,
            Betai* Ii, Betai* Si - Gammai - mu;
        B << -Si * Ii, -Si,
            Si* Ii, 0;
        C << 0, 1;

        auto mat = discretize(A, B);
        auto& Ad = std::get<0>(mat);
        auto& Bd_m = std::get<1>(mat);
        auto& Bd_d = std::get<2>(mat);

        return SS_Mat{ Ad, Bd_m, Bd_d, C };
    }

    
};


// The three_dof_tuning class encapsulates the computation and storage of tuning parameters and filter gains
// for a three-degree-of-freedom (3-DOF) controller structure. This class is responsible for generating the
// filter coefficients and observer gains used for reference tracking, disturbance rejection, and control effort shaping.
//
// Members:
// - alpha_r: Filter coefficient vector for the reference signal, computed from the reference filter time constant (tau_r).
// - alpha_d: Filter coefficient vector for the disturbance signal, computed from the disturbance filter time constant (tau_d).
// - fa: Filter coefficient vector for the control effort, computed from the control filter time constant (tau_u).
// - Kf: Observer/filter gain matrix (nx + 2*ny x ny), computed based on the filter coefficients and the noise model. Of the form: [0 Fb^T Fa^T]^T (Lee et al. 1994)
//
// Constructor:
// - three_dof_tuning(): Initializes all filter coefficients and the observer gain matrix Kf using the provided time constants and
//   the selected noise model.
//
// Key Functions:
// - compute_alpha(): Template function that computes the filter coefficient vector (alpha) for a given time constant (tau).
//   The 'choice' parameter determines if the function returns the direct coefficient (for reference/disturbance) or its complement (for control).
// - compute_Kf(): Computes the observer/filter gain matrix Kf using the filter coefficients (fa) and the noise model matrix (Aw).
//   This gain is used in the controller to filter the measured output and estimate the disturbance.
//
// The three_dof_tuning class is intended to be instantiated once per simulation or controller setup, providing all necessary
// filter and observer gains for robust reference tracking and disturbance rejection in the 3-DOF control architecture.


class three_dof_tuning {
public:
    Eigen::Matrix<double, ny, 1> alpha_r;
    Eigen::Matrix<double, ny, 1> alpha_d;
    Eigen::Matrix<double, ny, 1> fa;
    Eigen::Matrix<double, nx + 2 * ny, ny> Kf;


    three_dof_tuning()
        : alpha_r(compute_alpha(tau_r, true)),
        alpha_d(compute_alpha(tau_d, true)),
        fa(compute_alpha(tau_u, false))
    {
        auto noise = Matrices::eval_noise_model(noise_choice);
        Kf = compute_Kf(fa, noise.Aw);
    }


private:
    template <typename T_alpha>
    Eigen::Matrix<double, ny, 1> compute_alpha(const T_alpha& tau, bool choice) {
        Eigen::Matrix<double, ny, 1> alpha;
        if constexpr (std::is_same_v<T_alpha, double>) {
            alpha = Eigen::Matrix<double, ny, 1>::Constant(std::exp(- ts / tau));
        }
        else {
            alpha = (-ts/tau.array()).exp();
        }
        return choice ? alpha : 1.0 - alpha.array();
    }

    Eigen::Matrix<double, nx + 2 * ny, ny> compute_Kf(const Eigen::Matrix<double, ny, 1>& fa, const Eigen::Matrix<double, ny, ny>& Aw) {
        Eigen::Matrix<double,ny, ny> Fa = fa.asDiagonal();
        Eigen::Matrix<double, ny, ny> Fb = Fa.pow(2) * (Identity<ny>() + Aw - Aw * Fa).inverse();
        Eigen::Matrix<double, nx, ny> Zero_x = Zero_Mat<nx, ny>();

        Eigen::Matrix<double, nx + 2 * ny, ny> Kf;
        Kf << Zero_x,
            Fb,
            Fa;
        return Kf;
    }

};


// The Signals class generates and manages all reference, measured and unmeasured disturbance variable signals required for simulation and control.
// It also provides filtered versions of these signals using type-1 filters, which are essential for reference and disturbance tracking
// in the controller design. One can also design a type-2 filter for the disturbance signal if needed (presence of integrator).
//
// All signals, inputs and outputs can be made multivariable. Cururently I am doing SISO for the SIR problem.
// 
// Members:
// - ref: Reference trajectory matrix for the output(s) over the simulation horizon (ant_len x ny).
// - dm: Disturbance trajectory matrix for the disturbance input(s) over the simulation horizon (ant_len x nu_d).
// - du: Manipulated variable trajectory matrix for the manipulated input(s) over the simulation horizon (Tstop x ny).
// - filt_ref: Filtered reference trajectory, computed using a type-1 filter and the reference filter tuning parameters.
// - filt_dm: Filtered disturbance trajectory, computed using a type-1 filter and the disturbance filter tuning parameters.
//
// Constructor:
// - Signals(): Initializes the reference, disturbance, and manipulated variable signals using make_ref(), make_dm(), and make_du().
//   It also computes the filtered reference and disturbance signals using the type1_filt_func and the filter parameters from
//   three_dof_tuning.
//
// Key Functions:
// - make_ref(): Generates the reference trajectory for the output(s), switching from the initial value to the setpoint at ref_time.
// - make_dm(): Generates the disturbance trajectory, switching from the initial value to a new value at dm_time.
// - make_du(): Generates the manipulated variable trajectory, switching from the initial value to a new value at du_time.
// - type1_filt_func(): Template function that applies a type-1 filter to a given signal using the provided filter coefficients (alpha).
//   This is used for both reference and disturbance signals to shape their response for the controller.
//
// The Signals class is intended to be instantiated once per simulation, providing all required signal trajectories and their filtered
// versions for use in the control algorithm and performance evaluation.



class Signals {
public:

    Eigen::Matrix<double, ant_len, ny> ref;
    Eigen::Matrix<double, ant_len, nu_d> dm;
    Eigen::Matrix<double, Tstop, ny> du;
    Eigen::Matrix<double, ant_len, ny> filt_ref;
    Eigen::Matrix<double, ant_len, nu_d> filt_dm;
    
    Signals()
        : ref(make_ref()),
        dm(make_dm()),
        du(make_du()) {

        three_dof_tuning filt_tune;
        filt_ref = type1_filt_func(ref, filt_tune.alpha_r);
        filt_dm = type1_filt_func(dm, filt_tune.alpha_d);
    }

private:
    Eigen::Matrix<double, ant_len, ny> make_ref() {
        Eigen::Matrix<double, ant_len, ny> r = Eigen::Matrix<double, ant_len, ny>::Zero();
        r.block(0, 0, ref_time, ny) = y_init.replicate(ref_time, 1);
        r.block(ref_time, 0, ant_len - ref_time, ny) = (y_init + del_sp).replicate(ant_len - ref_time, 1);
        return r;
    }

    Eigen::Matrix<double, ant_len, nu_d> make_dm() {
        Eigen::Matrix<double, ant_len, nu_d> d = Eigen::Matrix<double, ant_len, nu_d>::Zero();
        d.block(0, 0, dm_time, nu_d) = dm_init.transpose().replicate(dm_time, 1);
        d.block(dm_time, 0, ant_len - dm_time, nu_d) = (dm_init + del_dm).transpose().replicate(ant_len - dm_time, 1);
        return d;
    }

    Eigen::Matrix<double, Tstop, ny> make_du() {
        Eigen::Matrix<double, Tstop, ny> u = Eigen::Matrix<double, Tstop, ny>::Zero();
        u.block(0, 0, du_time, ny) = (du_init).transpose().replicate(du_time, 1);
        u.block(du_time, 0, Tstop - du_time, ny) = (du_init + del_du).transpose().replicate(Tstop - du_time, 1);
        return u;
    }
    template<int sig_len, int nu_v>
    Eigen::Matrix<double, sig_len, nu_v> type1_filt_func(const Eigen::Matrix<double, sig_len, nu_v>& sig,
            const Eigen::Matrix<double, nu_v, 1>& alpha);
};

// Template function to apply a type-1 filter to a given signal using the provided filter coefficients (alpha).
template<int ant_len, int nu_v>
Eigen::Matrix<double, ant_len, nu_v>
Signals::type1_filt_func(const Eigen::Matrix<double, ant_len, nu_v>& sig,
    const Eigen::Matrix<double, nu_v, 1>& alpha)
{
    Eigen::Matrix<double, ant_len, nu_v> B;
    {
        Eigen::Matrix<double, nu_v, nu_v> D = alpha.asDiagonal();
        B = sig * (Eigen::Matrix<double, nu_v, nu_v>::Identity() - D);

        Eigen::Matrix<double, ant_len, 1> e1 = Eigen::Matrix<double, ant_len, 1>::Zero();
        e1(0) = 1.0f;

        Eigen::Matrix<double, 1, nu_v> tmp = alpha.transpose().cwiseProduct(sig.row(0));
        B.noalias() += e1 * tmp;
    }

    Eigen::Matrix<double, ant_len, nu_v> sig_filt;
    for (int j = 0; j < nu_v; ++j) {
        double aj = alpha(j);
        sig_filt(0, j) = B(0, j);
        for (int i = 1; i < ant_len; ++i) {
            sig_filt(i, j) = aj * sig_filt(i - 1, j) + B(i, j);
        }
    }
    return sig_filt;
}

// The Weights_Lims class defines and constructs the weighting matrices and constraint limits required for the quadratic programming (QP)
// problem in model predictive control (MPC). It centralizes the setup of cost function weights and all input/output constraints, ensuring
// a consistent and modular approach to controller configuration.
//
// Structures:
// - Weights: Contains the QP cost function matrices:
//   - H_u: The Hessian matrix for the manipulated variable increments (m*nu_m x m*nu_m), combining output tracking and input move suppression weights.
//   - Grad_coeff: The gradient coefficient matrix (m*nu_m x p*ny), used in the QP cost function for output tracking.
//
// - Lims: Contains all constraint matrices and vectors for the QP problem:
//   - Cu: The stacked constraint matrix for input, input increment, and output constraints (Cu_len x m*nu_m).
//   - y_max_lim, y_min_lim: Output upper and lower bounds over the prediction horizon (y_len_p x 1).
//   - u_max_lim, u_min_lim: Input upper and lower bounds over the control horizon (u_len_m x 1).
//   - delu_max_lim, delu_min_lim: Move size increment upper and lower bounds (u_len_m x 1).
//   - All dimensions are automatically determined from the problem size and horizon lengths.
//
// Members:
// - weight_mats: Instance of Weights, initialized with the appropriate cost function matrices for the current controller setup.
// - lim_mats: Instance of Lims, initialized with all constraint matrices and vectors for the QP problem.
//
// Constructor:
// - Weights_Lims(): Instantiates a Matrices object and uses its controller matrices to construct the weights and limits for the QP.
//
// Key Functions:
// - def_Weights(): Builds the Hessian and gradient matrices for the QP cost function using the output and input weights (Wy, Wdu).
// - def_Lims(): Constructs the stacked constraint matrix and all bound vectors for the QP, including input, input increment, and output constraints.
//
// The Weights_Lims class is intended to be instantiated once per controller setup, providing all cost and constraint data required
// for the QP-based MPC optimization step.


class Weights_Lims {
public:
    struct Weights {
        Eigen::Matrix<double, m* nu_m, m* nu_m> H_u;
        Eigen::Matrix<double, m* nu_m, p*ny> Grad_coeff;

    };
    struct Lims {
        static constexpr int u_len_m = nu_m * m;
        static constexpr int y_len_p = ny * p;
        static constexpr int Cu_len = 4 * u_len_m + 2 * y_len_p;

        Eigen::Matrix <double, Cu_len, nu_m*m> Cu;
        Eigen::Matrix<double, y_len_p, 1> y_max_lim;
        Eigen::Matrix<double, y_len_p, 1> y_min_lim;
        Eigen::Matrix<double, u_len_m, 1> u_max_lim;
        Eigen::Matrix<double, u_len_m, 1> u_min_lim;
        Eigen::Matrix<double, u_len_m, 1> delu_max_lim;
        Eigen::Matrix<double, u_len_m, 1> delu_min_lim;

    };

    Weights weight_mats;
    Lims lim_mats;

    Weights_Lims() {
        Matrices mats;
        weight_mats = def_Weights(mats.ctrl_mat.Su);
        lim_mats = def_Lims(mats.ctrl_mat.Su);
    }
private:
    Weights def_Weights(const Eigen::Matrix<double, ny* p, nu_m* m>& Su) {
       auto Wy_vec = wy.replicate(p, 1).eval();
       auto Wy = Wy_vec.asDiagonal();

       auto Wdu_vec = wdu.replicate(m, 1).eval();
       auto Wdu = Wdu_vec.asDiagonal();

       auto Wdu_squared = Wdu.diagonal().array().square().matrix().asDiagonal().toDenseMatrix().eval();
       
       auto H_u = Su.transpose() * Wy * Wy * Su + Wdu_squared;
       auto G_coeff = 2 * Su.transpose() * Wy * Wy;

        return Weights{ H_u.eval(), G_coeff.eval()};
    }
    Lims def_Lims(Eigen::Matrix<double,ny*p,nu_m*m> Su) {
        auto I = Identity<nu_m>();
        Eigen::Matrix<double, nu_m* m, nu_m* m> I_L;
        I_L.setZero();
        
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col <= row; ++col) {
                I_L.block(row * nu_m, col * nu_m, nu_m, nu_m) = I;
            }
        }
        
        auto I_u = Identity<nu_m*m>();
        Eigen::Matrix<double, 4 * nu_m * m + 2 * ny * p, m*nu_m> Cu;
        

        Cu << -I_L,
            I_L,
            -I_u,
            I_u,
            -Su,
            Su;

        Eigen::Matrix<double, ny* p, 1> y_max_lim = y_max.replicate(p, 1);
        Eigen::Matrix<double, ny* p, 1> y_min_lim = y_min.replicate(p, 1);
        Eigen::Matrix<double, nu_m* m, 1> u_max_lim = u_max.replicate(m, 1);
        Eigen::Matrix<double, nu_m* m, 1> u_min_lim = u_min.replicate(m, 1);
        Eigen::Matrix<double, nu_m* m, 1> delu_max_lim = delu_max.replicate(m, 1);
        Eigen::Matrix<double, nu_m* m, 1> delu_min_lim = delu_min.replicate(m, 1);        
        
        return Lims{ Cu,  y_max_lim , y_min_lim , u_max_lim , u_min_lim , delu_max_lim , delu_min_lim };
    }
    
};


// typedef for SIR dynamic model solver
typedef Eigen::Matrix<double, nx_ode, 1> state_type;
typedef Eigen::Matrix<double, nx, 1> state_type_controller;

typedef runge_kutta_cash_karp54<state_type> error_stepper_type;
typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;


// Function prototype for dynamic_model
state_type dynamic_model(const double& beta, const double& gamma, const double& kv,
    const state_type& x_km1, double t);

// Function for control algorithm

typedef Eigen::Matrix<double, nu_m, 1> u_type;
typedef Eigen::Matrix<double, ny, 1> y_type;
typedef Eigen::Matrix<double, ny, 1> dm_type;
typedef Eigen::Matrix<double, nx + 2 * ny, 1> x_aug_type;

const state_type_controller del_x_km1_init = state_type_controller::Zero();
const y_type del_xw_km1_init = y_type::Zero();

// Initialize the augmented state vector
inline x_aug_type get_x_aug_km1_init() {
    x_aug_type x_aug_km1_init = x_aug_type::Zero();
    x_aug_km1_init.block(0, 0, nx, 1) = del_x_km1_init;
    x_aug_km1_init.block(nx, 0, ny, 1) = del_xw_km1_init;
    x_aug_km1_init.block(nx + ny, 0, ny, 1) = y_init;
    return x_aug_km1_init;
}

// Function prototype for controller. 
u_type controller(y_type, dm_type, dm_type, int);

