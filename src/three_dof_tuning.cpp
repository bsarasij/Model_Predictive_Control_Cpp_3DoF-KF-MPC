#include "Header1.h"

Eigen::Matrix<double, nx + 2 * ny, ny> three_dof_tuning::compute_Kf(const Eigen::Matrix<double, ny, 1>& fa, const Eigen::Matrix<double, ny, ny>& Aw) {
    Eigen::Matrix<double, ny, ny> Fa = fa.asDiagonal();
    Eigen::Matrix<double, ny, ny> Fb = Fa.pow(2) * (Identity<ny>() + Aw - Aw * Fa).inverse();
    Eigen::Matrix<double, nx, ny> Zero_x = Zero_Mat<nx, ny>();

    Eigen::Matrix<double, nx + 2 * ny, ny> Kf;
    Kf << Zero_x,
        Fb,
        Fa;
    return Kf;
}

