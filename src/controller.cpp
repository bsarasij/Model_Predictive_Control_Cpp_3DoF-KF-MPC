// Module: MPC Implementation
//Author: Sarasij Banerjee
//Date: July 5, 2025
#include "Header1.h"

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
deltaColumns(const Eigen::MatrixBase<Derived>& M); // Function to compute the forward difference of each column in a matrix. Needed for Δd(k) calculations. Defined below.

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> // Function to conflate the columns of a matrix into a single column vector. Needed for multivariable inputs/disturbances.
conflateColumns(const Eigen::MatrixBase<Derived>& M);


u_type controller(y_type y_k, dm_type dm_km1, dm_type dm_km1_filt, int k) {
	
	static Matrices cont_matrices;
	static three_dof_tuning tuning_mat;
	static Signals signals;
	static Weights_Lims weights_lims;

	static const auto& A_aug = cont_matrices.aug_mat.A_aug; // Augmented state-space matrices for propagation of controller's internal model.
	static const auto& Bm_aug = cont_matrices.aug_mat.Bm_aug;
	static const auto& Bd_aug = cont_matrices.aug_mat.Bd_aug;
	static const auto& C_aug = cont_matrices.aug_mat.C_aug;
	static const auto& Kf = tuning_mat.Kf; // Kalman filter gain matrix for state estimation.

	static const auto& Sx = cont_matrices.ctrl_mat.Sx; // State feedback matrix for projecting the state estimate onto the output space.
	static const auto& Sd = cont_matrices.ctrl_mat.Sd;	// Disturbance feedback matrix for projecting the disturbance estimate onto the output space.

	static const auto& filt_ref_vec = signals.filt_ref; // Filtered reference signal controller. Filter Speed dictates speed of response. 1st Degree of Freedom
	static const auto& dm_vec = signals.dm;
	static const auto& dm_filt_vec = signals.filt_dm; // Filtered disturbance signal controller. Filter Speed dictates speed of response. 2nd Degree of Freedom

	

	static const auto& H_u = weights_lims.weight_mats.H_u; // Coefficient of the quadratic cost term in the QP
	static const auto& G_coeff = weights_lims.weight_mats.Grad_coeff; // Coefficient of the error projection for the linear cost term in the QP. G = G_coeff * error_proj;
	static const auto& Cu = weights_lims.lim_mats.Cu; // Coefficients of the linear constraints in the QP

	static const auto& u_max_lim = weights_lims.lim_mats.u_max_lim; // Time-domain constraints
	static const auto& u_min_lim = weights_lims.lim_mats.u_min_lim;

	static const auto& delu_max_lim = weights_lims.lim_mats.delu_max_lim;
	static const auto& delu_min_lim = weights_lims.lim_mats.delu_min_lim;

	static const auto& y_max_lim = weights_lims.lim_mats.y_max_lim;
	static const auto& y_min_lim = weights_lims.lim_mats.y_min_lim;

	static u_type u_km1 = u_init; 
	static u_type delta_uk_minus_1 = u_type::Zero();

	static dm_type dm_km2 = dm_init;
	static dm_type dm_km2_filt = dm_init;

	auto& ref_proj = filt_ref_vec.block(k, 0, ny * p, 1); // Filtered Reference over the prediction horizon.

	dm_type delta_dm_km1 = dm_km1 - dm_km2; // Differenced disturbance vector for state-space with embedded integrator
	dm_type delta_dm_km1_filt = dm_km1_filt - dm_km2_filt; // Filtered version

	auto delta_dm_sig = deltaColumns(dm_vec); // forecasted Δd(k) over the horizon
	auto delta_dm_filt_sig = deltaColumns(dm_filt_vec); // filtered version

	static const auto& filt_ref_vec_conf = conflateColumns(filt_ref_vec); // Creating single column vector of the form [y_1(1) y_2(1) y_1(2) y_2(2) y_1(3) y_(3)]^T and so on
	static const auto& delta_dm_sig_conf = conflateColumns(delta_dm_sig); // Similarly for disturbances
	static const auto& delta_dm_sig_vec_conf = conflateColumns(delta_dm_filt_sig); // Filtered disturbances
	
	auto  delta_D_forecast = delta_dm_sig_conf.block(k, 0, m * nu_m, 1); // Forecasted disturbance vector for the move horizon
	auto  delta_D_forecast_filt = delta_dm_sig_vec_conf.block(k, 0, m * nu_m, 1); // Filtered version

	static x_aug_type x_hat_km1_given_km1 = get_x_aug_km1_init(); // augmented state vector.
	static x_aug_type x_hat_km1_given_km1_filt = get_x_aug_km1_init(); // One filtered version and one unfiltered version (To be used for the 3rd DoF)




	auto x_hat_k_given_km1 = ((A_aug)*x_hat_km1_given_km1 + (Bm_aug)*delta_uk_minus_1 + (Bd_aug)*delta_dm_km1).eval(); // State prediction step
	auto x_hat_k_given_k = (x_hat_k_given_km1 + Kf * (y_k - C_aug * x_hat_k_given_km1)).eval(); // State correction step using Kalman filter gain

	auto x_hat_k_given_km1_filt = ((A_aug)*x_hat_km1_given_km1_filt + (Bm_aug)*delta_uk_minus_1 + (Bd_aug)*delta_dm_km1_filt).eval(); // State prediction step for filtered state
	auto x_hat_k_given_k_filt = (x_hat_k_given_km1_filt + Kf * (y_k - C_aug * x_hat_k_given_km1)).eval(); // Note that correction term uses unfiltered state prediction x_hat_k_given_km1. 
																										// This ensures that unmeasured disturbance estimate is not influenced by filtered measured disturbance signals
																										// This constitutes the 3rd DoF. Refer to Banerjee et al. I&ECR 2025	

	auto y_proj = (Sx * x_hat_k_given_k + Sd * delta_D_forecast).eval(); // Projected output free response over prediction horizon, used for constraints 
	auto y_proj_filt = (Sx * x_hat_k_given_k_filt + Sd * delta_D_forecast_filt).eval(); // Projected filtered output free response over prediction horizon, used for MPC cost function

	// Debug: Checking for inf or NaN in y_proj
	if (!y_proj.allFinite()) {
		std::cerr << "y_proj contains inf or NaN at step " << k << std::endl;
		std::cerr << "y_proj: " << y_proj.transpose() << std::endl;
	}

	auto error_proj = ref_proj - y_proj_filt; // Error projection for the MPC cost function. Used to compute the control action.

	auto G = G_coeff * error_proj; // Linear cost term in the QP problem. This is the gradient of the cost function with respect to the control action.




	static constexpr int u_len_m = nu_m * m;
	static constexpr int y_len_p = ny * p;
	static constexpr int C_len = 4 * u_len_m + 2 * y_len_p;

	Eigen::Matrix<double, C_len, 1> C; {
		auto uh_lim = u_km1.replicate(m, 1) - u_max_lim;
		auto ul_lim = u_min_lim - u_km1.replicate(m, 1);
		auto delu_h_lim = -delu_max_lim;
		auto delu_l_lim = delu_min_lim;
		auto yh_lim = y_proj - y_max_lim;
		auto yl_lim = -y_proj + y_min_lim;

		C << uh_lim, // RHS of Cu*del_U_k <= C. Linear constraints for the control action
			ul_lim,
			delu_h_lim,
			delu_l_lim,
			yh_lim,
			yl_lim;

		// Debug: Check for inf or NaN in C
		if (!C.allFinite()) {
			std::cerr << "C contains inf or NaN at step " << k << std::endl;
			std::cerr << "C: " << C.transpose() << std::endl;
		}
		std::cerr << "Full C: " << C.transpose() << std::endl << std::flush;
	}


	//Setting up the osqpEigen solver for the MPC problem
	Eigen::SparseMatrix<double> Q = (2.0 * H_u.cast<double>()).sparseView();
	Eigen::VectorXd F = (-G).cast<double>();
	
	Eigen::SparseMatrix<double> A = (-Cu.cast<double>()).sparseView();

	Eigen::VectorXd l = Eigen::VectorXd::Constant(C_len, -1e20); // Lower bounds for the constraints. Set to a large negative value.
	Eigen::VectorXd u = - C.cast<double>();                      

	OsqpEigen::Solver solver;
	solver.settings()->setWarmStart(true);
	solver.data()->setNumberOfVariables(u_len_m);
	solver.data()->setNumberOfConstraints(C_len);
	solver.data()->setHessianMatrix(Q);
	solver.data()->setGradient(F);
	solver.data()->setLinearConstraintsMatrix(A);
	solver.data()->setLowerBound(l);
	solver.data()->setUpperBound(u);

	solver.initSolver();

	solver.solve();


	Eigen::VectorXd sol = solver.getSolution();
	Eigen::Matrix<double, nu_m* m, 1> del_U_k = sol.cast<double>();

	auto& del_uk = (del_U_k.block(0, 0, nu_m, 1)).eval(); // Extracting the first nu_m elements of the solution as the control action increment Δu(k). Receding horizon control action.
	auto u_k = (u_km1 + del_uk).eval(); // Control action at time k is the previous control action plus the increment Δu(k).




	// Update state
	x_hat_km1_given_km1 = x_hat_k_given_k;
	x_hat_km1_given_km1_filt = x_hat_k_given_k_filt;
	u_km1 = u_k;
	delta_uk_minus_1 = del_uk;
	dm_km2 = dm_km1;
	dm_km2_filt = dm_km1_filt;

	/*auto del_U_k_deb = toStdVec(del_U_k);
	auto l_deb = toStdVec(l);
	auto u_deb = toStdVec(u);
	auto F_deb = toStdVec(F);
	auto C_deb = toStdVec(C);*/ //Uncomment this if you have Visual Debugger with plot window. This will plot the data for debugging purposes. Doesn't support EigenBase. So create std::vectors.

	return u_k; // Implement control action at time k.
}

// Function to compute the forward difference of each column in a matrix. Needed for Δd(k) calculations.
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
deltaColumns(const Eigen::MatrixBase<Derived>& M)
{
	using Scalar = typename Derived::Scalar;

	const Eigen::Index n = M.rows();
	const Eigen::Index s = M.cols();

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(n, s);

	if (n == 0)              // empty input
		return out;

	out.row(0).setZero();    // first row = 0
	if (n > 1)               // remaining rows = forward difference
		out.bottomRows(n - 1) = M.bottomRows(n - 1) - M.topRows(n - 1);

	return out;
}

// Function to conflate the columns of a matrix into a single column vector. Needed for multivariable inputs/disturbances.
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1>
conflateColumns(const Eigen::MatrixBase<Derived>& M)
{
	using Scalar = typename Derived::Scalar;

	if (M.cols() == 1)
		return M;

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(M.size());

	for (Eigen::Index row = 0; row < M.rows(); ++row)
		out.segment(row * M.cols(), M.cols()) = M.row(row).transpose();

	return out;
}