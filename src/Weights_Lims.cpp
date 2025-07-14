#include "Header1.h"


Weights_Lims::Weights Weights_Lims::def_Weights(const Eigen::Matrix<double, y_len_p, u_len_m>& Su) {
    Eigen::Matrix<double,y_len_p,1> Wy_vec = wy.replicate(p, 1).eval();
    Eigen::Matrix<double, y_len_p, y_len_p> Wy = Wy_vec.asDiagonal();

    Eigen::Matrix<double, u_len_m, 1> Wdu_vec = wdu.replicate(m, 1).eval();
    Eigen::Matrix<double, u_len_m, u_len_m> Wdu = Wdu_vec.asDiagonal();

    Eigen::Matrix<double, u_len_m, u_len_m> Wdu_squared = Wdu.diagonal().array().square().matrix().asDiagonal().toDenseMatrix().eval();

    Eigen::Matrix<double, u_len_m, u_len_m> H_u = Su.transpose() * Wy * Wy * Su + Wdu_squared;
    Eigen::Matrix<double, u_len_m, y_len_p> G_coeff = 2 * Su.transpose() * Wy * Wy;

    return Weights_Lims::Weights{ H_u.eval(), G_coeff.eval() };
}
Weights_Lims::Lims Weights_Lims::def_Lims(Eigen::Matrix<double, y_len_p, u_len_m> Su) {
    Eigen::Matrix<double, nu_m, nu_m> I = Identity<nu_m>();
    Eigen::Matrix<double, u_len_m, u_len_m> I_L;
    I_L.setZero();

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col <= row; ++col) {
            I_L.block(row * nu_m, col * nu_m, nu_m, nu_m) = I;
        }
    }

    Eigen::Matrix<double, u_len_m, u_len_m> I_u = Identity<nu_m * m>();
    Eigen::Matrix<double, 4 * u_len_m + 2 * y_len_p, u_len_m> Cu;


    Cu << -I_L,
        I_L,
        -I_u,
        I_u,
        -Su,
        Su;

    Eigen::Matrix<double, y_len_p, 1> y_max_lim = y_max.replicate(p, 1);
    Eigen::Matrix<double, y_len_p, 1> y_min_lim = y_min.replicate(p, 1);
    Eigen::Matrix<double, u_len_m, 1> u_max_lim = u_max.replicate(m, 1);
    Eigen::Matrix<double, u_len_m, 1> u_min_lim = u_min.replicate(m, 1);
    Eigen::Matrix<double, u_len_m, 1> delu_max_lim = delu_max.replicate(m, 1);
    Eigen::Matrix<double, u_len_m, 1> delu_min_lim = delu_min.replicate(m, 1);

    return Weights_Lims::Lims{ Cu,  y_max_lim , y_min_lim , u_max_lim , u_min_lim , delu_max_lim , delu_min_lim };
}
