// Define the system of ODEs for the infection model. Can be replaced with any system of ODE's. Represents the true blackbox system.
//Author: Sarasij Banerjee
//Date: July 5, 2025

#include "Header1.h"
struct InfectionModel {
	double beta, // infection rate. Manipulated Variable
		gamma, // recovery rate. Unmeasured Disturbance
		kv; // vaccination rate. Measured Disturbance

	void operator() (const state_type& x, state_type& dxdt, double t) const { // Tried a functor as it supports holding state variables. Also functor needed for passing the ode system to odeint along with states.
		double S = x[0],
			I = x[1];
		dxdt[0] = Br - beta * S * I - mu * S - kv * S; // True infection spread model with vaccination, recovery, birth and death
		dxdt[1] = beta * S * I - gamma * I - mu * I;
	}
};

state_type dynamic_model(const double& beta, const double& gamma, const double& kv, const state_type& x_km1, double t) {
	InfectionModel infectModel;
	infectModel.beta = beta;
	infectModel.gamma = gamma;
	infectModel.kv = kv;
	state_type x;
	x = x_km1;

	const double dt_init = 0.1f, // Initial time step for the ODE solver with variable time step
		abs_tol = 1e-6f,
		rel_tol = 1e-6f;


	runge_kutta4<state_type> stepper; // Using Runge-Kutta 4th order method for ODE integration
	integrate_adaptive(stepper, infectModel, x, t, t + 1, dt_init); // Integrate the ODE system from t to t + 1 with the initial state x_km1

	return x; // Return the state after integration

}