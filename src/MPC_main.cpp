// MPC C++.cpp : This file contains the 'main' function. Program execution begins and ends here.
//Author: Sarasij Banerjee
//Date: July 5, 2025

#include "Header1.h"

int main()
{
    Signals sig;
	auto& ref = sig.ref; // Reference signal for the output
	auto& dm = sig.dm; // Measured Disturbance signal
	auto& du = sig.du; // Unmeasured Disturbance signal

	auto& filt_ref = sig.filt_ref; // Filtered reference signal
	auto& filt_dm = sig.filt_dm; // Filtered measured disturbance signal

    
	auto u_k = u_init; // Initial control input
	y_type y_k; // Output vector for the current time step
    
	state_type x_km1; // State vector for the previous time step
    x_km1 << Si, // Initial state vector for the SIR model (Susceptible, Infected)
		Ii; 

	state_type x_ode; // State vector for the current time step after ODE update

	Eigen::Matrix<double, Tstop, ny> y_hist; // History of outputs
	Eigen::Matrix<double, Tstop, nu_m> u_hist; // History of control inputs
    y_hist.setZero();
    u_hist.setZero();

    for (int t = 0; t < Tstop; t++) {
		x_ode = dynamic_model(u_k(0), du(t, 0), dm(t, 0), x_km1, t); // Dynamic model update for the SIR model. 1 Manipulated Var, 1 Measured Dist, 1 Unmeasured, 2 States
		y_k(0) = x_ode[1]; // Second state is Infection, which is the output we are controlling

		u_k = controller(y_k, dm_type(dm(t, 0)), dm_type(filt_dm(t, 0)), t); // Controller update using the current output and disturbance
        
		x_km1 = x_ode; // Update the previous state for the next iteration

		y_hist.row(t) = y_k.transpose(); // Store the output and input in the history
        u_hist.row(t) = u_k.transpose();

    }


    // Gnuplot plotting
    Gnuplot gp;
    std::vector<std::pair<double, double>> y_hist_vec, ref_vec, u_hist_vec;
    for (int t = 0; t < Tstop; ++t) {
        y_hist_vec.emplace_back(t, y_hist(t, 0));
        ref_vec.emplace_back(t, ref(t, 0));
        u_hist_vec.emplace_back(t, u_hist(t, 0));
    }
    gp << "set multiplot layout 2,1 title 'Simulation Results'\n";
    gp << "set title 'y_hist and ref'; set xlabel 'Time'; set ylabel 'y'\n";
    gp << "plot '-' with lines title 'y_hist', '-' with lines title 'ref'\n";
    gp.send1d(y_hist_vec);
    gp.send1d(ref_vec);
    gp << "set title 'u_hist'; set xlabel 'Time'; set ylabel 'u'\n";
    gp << "plot '-' with lines title 'u_hist'\n";
    gp.send1d(u_hist_vec);
    gp << "unset multiplot\n";
    std::cin.get(); 
    return 0;
}
