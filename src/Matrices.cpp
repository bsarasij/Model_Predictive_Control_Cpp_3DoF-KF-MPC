#include "Header1.h"

Matrices::Noise_Model_Mat Matrices::eval_noise_model(int ch) {
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
    return Matrices::Noise_Model_Mat{ Aw, Cw };
}


Matrices::Aug_SS_Mat Matrices::eval_aug_matrices(SS_Mat& ss_mat, Noise_Model_Mat& noise_mat) {
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

Matrices::Ctrl_Mat Matrices::controller_mat(Aug_SS_Mat aug_mat) {
    Eigen::Matrix<double, ny* p, nx + 2 * ny> Sx;
    Eigen::Matrix<double, ny* p, u_len_m> Su;
    Eigen::Matrix<double, ny* p, nu_d* m> Sd;

    Sx.setZero();
    Su.setZero();
    Sd.setZero();


    Eigen::Matrix<double, nx + 2 * ny, nx + 2 * ny> A_power = aug_mat.A_aug;

    Sx.block(0, 0, ny, nx + 2 * ny) = aug_mat.C_aug * aug_mat.A_aug;
    for (int i = 1; i < p; i += 1) {
        Sx.block(i * ny, 0, ny, nx + 2 * ny) = aug_mat.C_aug * A_power;
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



    Eigen::Matrix<double,ny,ny> I_y = Identity<ny>();
    Eigen::Matrix<double, p, 1> ones = Matrix<double, p, 1>::Ones();
    Eigen::Matrix<double, ny*p, ny> Ip = Eigen::kroneckerProduct(ones, I_y);
    return Ctrl_Mat{ Sx, Su, Sd, Ip };
}

auto Matrices::discretize(Eigen::Matrix<double, nx, nx>& A, const Eigen::Matrix<double, nx, nu>& B) {
    Eigen::Matrix<double, nx, nx> Ad = (A * ts).exp();
    Eigen::Matrix<double, nx, nx> I = Identity<nx>();
    Eigen::Matrix<double, nx, nu> Bd = A.inverse() * (Ad - I) * B;
    Eigen::Matrix<double, nx, nu_m> Bd_m = Bd.block(0, 0, nx, nu_m);
    Eigen::Matrix<double, nx, nu_d> Bd_d = Bd.block(0, nu_m, nx, nu_d);

    return std::make_tuple(Ad, Bd_m, Bd_d);
}


Matrices::SS_Mat Matrices::compute_mat() {
    Eigen::Matrix<double, nx, nx> A;
    Eigen::Matrix<double, nx, nu> B;
    Eigen::Matrix<double, ny, nx> C;

    A << -Betai * Ii - mu - kvi, -Betai * Si,
        Betai* Ii, Betai* Si - Gammai - mu;
    B << -Si * Ii, -Si,
        Si* Ii, 0;
    C << 0, 1;

    auto mat = discretize(A, B);
    Eigen::Matrix<double,nx,nx>& Ad = std::get<0>(mat);
    Eigen::Matrix<double, nx, nu_m>& Bd_m = std::get<1>(mat);
    Eigen::Matrix<double, nx, nu_d>& Bd_d = std::get<2>(mat);

    return Matrices::SS_Mat{ Ad, Bd_m, Bd_d, C };
}
