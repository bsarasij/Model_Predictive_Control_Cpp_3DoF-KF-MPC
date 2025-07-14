#include "Header1.h"


Eigen::Matrix<double, ant_len, ny> Signals::make_ref() {
    Eigen::Matrix<double, ant_len, ny> r = Eigen::Matrix<double, ant_len, ny>::Zero();
    r.block(0, 0, ref_time, ny) = y_init.replicate(ref_time, 1);
    r.block(ref_time, 0, ant_len - ref_time, ny) = (y_init + del_sp).replicate(ant_len - ref_time, 1);
    return r;
}

Eigen::Matrix<double, ant_len, nu_d> Signals::make_dm() {
    Eigen::Matrix<double, ant_len, nu_d> d = Eigen::Matrix<double, ant_len, nu_d>::Zero();
    d.block(0, 0, dm_time, nu_d) = dm_init.transpose().replicate(dm_time, 1);
    d.block(dm_time, 0, ant_len - dm_time, nu_d) = (dm_init + del_dm).transpose().replicate(ant_len - dm_time, 1);
    return d;
}

Eigen::Matrix<double, Tstop, ny> Signals::make_du() {
    Eigen::Matrix<double, Tstop, ny> u = Eigen::Matrix<double, Tstop, ny>::Zero();
    u.block(0, 0, du_time, ny) = (du_init).transpose().replicate(du_time, 1);
    u.block(du_time, 0, Tstop - du_time, ny) = (du_init + del_du).transpose().replicate(Tstop - du_time, 1);
    return u;
}