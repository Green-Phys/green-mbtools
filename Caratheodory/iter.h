/*
Authors: Jiani Fei and Emanuel Gull
Affiliation: Department of Physics, University of Michigan, Ann Arbor
The code is part of the paper: Analytical Continuation of Matrix-Valued Functions: Carathéodory Formalism
Citation of the paper is encouraged.
*/

#ifndef ITER_H
#define ITER_H

#include "carath.h"


template <class T>
class Cara : precision_<T> {
private:
    using typename precision_<T>::ca_real;
    using typename precision_<T>::ca_complex;
    using typename precision_<T>::ca_complex_vector;
    using typename precision_<T>::ca_complex_matrix;
    using typename precision_<T>::ca_complex_matrix_vector;
public:
    Cara (int imag_num, int dim, std::string ifile);
    void evaluation (real_domain_data<T> & real);
private:
    int dim_;
    imag_domain_data <T> imag;
    ca_complex_matrix_vector Ws; //W_is
    ca_complex_matrix_vector Vs; //intermediate Vs (for calculating Psis)
    ca_complex_matrix_vector Fs; //intermediate Psis (Schur class functions)
    ca_complex_matrix_vector sqrt_one; //[1 - W_i * W_i^dagger]^0.5
    ca_complex_matrix_vector sqrt_two; //[1 - W_i^dagger * W_i]^-0.5
    //calculate Hermitian square root of matrix M
    ca_complex_matrix sqrt_m (const ca_complex_matrix & M) {
      Eigen::ComplexEigenSolver<ca_complex_matrix> ces;
      ces.compute(M);
      ca_complex_matrix evals = ces.eigenvalues();//(M.rows(), 1);
      for (int i = 0; i < M.rows(); i++) {
        evals(i, 0) = std::sqrt(ces.eigenvalues()(i, 0));
      }
      return ces.eigenvectors() * evals.asDiagonal() * ces.eigenvectors().inverse();
    }
    //calculate W_is
    void core();
    void check_correctness ();
};


template <class T>
Cara<T>::Cara (int imag_num, int dim, std::string ifile) :
                dim_(dim), imag(imag_num, dim, ifile) {

    //reshape intermediate vectors
    Ws.resize(imag_num);
    Vs.resize(imag_num);
    Fs.resize(imag_num);
    sqrt_one.resize(imag_num);
    sqrt_two.resize(imag_num);
    core();
}


template <class T>
void Cara<T>::core () {
    ca_complex One {1., 0.};
    ca_complex_matrix id (dim_, dim_);
    id.setIdentity();
    for (int i = 0; i < imag.N_imag(); i++)
        Ws[i] = imag.val()[i];
    for (int i = imag.N_imag() - 1; i > 0; i--) {
        ca_complex zi = imag.freq()[i];
        ca_complex_matrix Wi = Ws[i];
        ca_complex_matrix sqrt_one_i = sqrt_m(id - Wi * Wi.adjoint());
        ca_complex_matrix sqrt_two_i = sqrt_m(id - Wi.adjoint() * Wi);
        ca_complex_matrix sqrt_one_i_inv = sqrt_one_i.inverse();
        ca_complex_matrix sqrt_two_i_inv = sqrt_two_i.inverse();
        for (int j = i - 1; j >= 0; j--) {
            ca_complex zj = imag.freq()[j];
            ca_complex_matrix Wj = Ws[j];
            ca_complex y_ij = ca_complex{std::abs(zi), 0.} * (zi - zj) / zi / (One - std::conj(zi) * zj);
            Eigen::FullPivLU<ca_complex_matrix> lu(id - Wi.adjoint() * Wj);
            Ws[j] = sqrt_one_i_inv * (Wj - Wi) * lu.solve(sqrt_two_i) / y_ij;
        }
        sqrt_one[i] = sqrt_one_i;
        sqrt_two[i] = sqrt_two_i_inv;
    }
    sqrt_one[0] = sqrt_m(id - Ws[0] * Ws[0].adjoint());
    sqrt_two[0] = sqrt_m(id - Ws[0].adjoint() * Ws[0]).inverse();
}


template <class T>
void Cara<T>::evaluation (real_domain_data<T> & real) {
    ca_complex I {0., 1.};
    ca_complex One {1., 0.};
    ca_complex_matrix id (dim_, dim_);
    id.setIdentity();
    for (int i = 0; i < real.N_real(); i++) {
        ca_complex z = (real.freq()[i] - I) / (real.freq()[i] + I);
        ca_complex z0 = imag.freq()[0];
        ca_complex_matrix W0 = Ws[0];
        Vs[0] = ca_complex{std::abs(z0), 0.} * (z0 - z) / z0 / (One - std::conj(z0) * z) * id;
        Fs[0] = (id + Vs[0] * W0.adjoint()).inverse() * (Vs[0] + W0);
        for (int j = 1; j < imag.N_imag(); j++) {
            ca_complex zj = imag.freq()[j];
            ca_complex_matrix Wj = Ws[j];
            Vs[j] = ca_complex{std::abs(zj), 0.} * (zj - z) / zj / (One - std::conj(zj) * z) * 
                    sqrt_one[j] * Fs[j - 1] * sqrt_two[j];
            Eigen::FullPivLU<ca_complex_matrix>lu (id + Vs[j] * Wj.adjoint());
            Fs[j] = lu.solve(Vs[j] + Wj);
        }
        ca_complex_matrix real_i = Fs[imag.N_imag() - 1];
        Eigen::FullPivLU<ca_complex_matrix>lu2 (id + real_i);
        real.val()[i] = -I * lu2.solve(id - real_i);
    }
}


#endif
