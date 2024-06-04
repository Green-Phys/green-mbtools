/*
Authors: Jiani Fei and Emanuel Gull
Affiliation: Department of Physics, University of Michigan, Ann Arbor
The code is part of the paper: Analytical Continuation of Matrix-Valued Functions: Carathéodory Formalism
Citation of the paper is encouraged.
*/

#ifndef CARATH_H
#define CARATH_H

#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include "mpfr_float.h"

//precision class is used to define typenames
//template T can be any precision type, e.g. double or mpreal
template <class T>
class precision_ {
protected:
    using ca_real = T;
    using ca_complex = std::complex<T>;
    using ca_complex_vector = std::vector<ca_complex>;
    using ca_complex_matrix = Eigen::Matrix <ca_complex, Eigen::Dynamic, Eigen::Dynamic>;
    using ca_complex_matrix_vector = std::vector<ca_complex_matrix>;
};


//Matsubara data storage (unit disk frequencies, Schur class matrix values)
template <class T>
class imag_domain_data : precision_<T> {
private:
    using typename precision_<T>::ca_real;
    using typename precision_<T>::ca_complex;
    using typename precision_<T>::ca_complex_vector;
    using typename precision_<T>::ca_complex_matrix;
    using typename precision_<T>::ca_complex_matrix_vector;
public:
    imag_domain_data (int imag_num, int dim, std::string ifile) : N_imag_(imag_num), dim_(dim) {
        std::ifstream ifs(ifile);
        val_.resize(N_imag_);
        freq_.resize(N_imag_);
        ca_complex_matrix id (dim_, dim_);
        id.setIdentity();
        ca_complex I {0., 1.};
        ca_complex One {1., 0.};
        ca_real freq, re, im;
        for (int i = 0; i < N_imag_; i++) {
            ifs >> freq;
            freq_[i] = ca_complex{0., freq};
            freq_[i] = (freq_[i] - I) / (freq_[i] + I);
            val_[i] = ca_complex_matrix(dim_, dim_);
        }
        for (int i = 0; i < N_imag_; i++) {
            //real part
            for (int j = 0; j < dim_; j++) {
                for (int k = 0; k < dim_; k++) {
                    ifs >> re;
                    val_[i](j, k) = ca_complex{re, 0.};
                }
            }
            //imag part
            for (int j = 0; j < dim_; j++) {
                for (int k = 0; k < dim_; k++) {
                    ifs >> im;
                    val_[i](j, k) += ca_complex{0., im};
                }
            }
            val_[i] = (id - I * val_[i]) * (id + I * val_[i]).inverse();
        }
        std::reverse(freq_.begin(),freq_.end());
        std::reverse(val_.begin(), val_.end());
    }
    //number of Matsubara points
    int N_imag() const { return N_imag_; }
    //Matrix values (transformed to Schur class ||M|| < 1) 
    //At Matsubara points (transformed to unit disk |z| < 1)
    const ca_complex_matrix_vector &val() const { return val_; }
    //Unit disk frequencies
    const ca_complex_vector &freq() const { return freq_; }
private:
    int N_imag_;
    int dim_;
    ca_complex_matrix_vector val_;
    ca_complex_vector freq_;
};


//real frequency storage, at omega+i*eta
template <class T>
class real_domain_data : precision_<T> {
private:
    using typename precision_<T>::ca_real;
    using typename precision_<T>::ca_complex;
    using typename precision_<T>::ca_complex_vector;
    using typename precision_<T>::ca_complex_matrix;
    using typename precision_<T>::ca_complex_matrix_vector;
public:
    real_domain_data (int real_num, int dim, int use_real_grid, std::string rfile, double min,
                      double max, double eta): N_real_(real_num), dim_(dim) {
        freq_.resize(real_num);
        val_.resize(real_num);
        if (use_real_grid) {
            std::cout << "Using custom grid" << std::endl;
            std::ifstream rfs(rfile);
            ca_real freq;
            for (int i = 0; i < real_num; i++) {
                rfs >> freq;
                freq_[i] = ca_complex{freq, eta};
            }
        }
        else {
            std::cout << "Using default grid" << std::endl;
            ca_real inter = (max - min) / (real_num - 1);
            ca_real temp = min;
            freq_[0] = ca_complex{min, eta};
            for (int i = 1; i < real_num; i++) {
                temp += inter;
                freq_[i] = ca_complex{temp, eta};
            }
        }
        for (int i = 0; i < real_num; i++) {
            val_[i].resize(dim_, dim_);
        }
    }
    //number of real frequencies
    int N_real() const { return N_real_; }
    //Values at real frequencies
    ca_complex_matrix_vector &val() { return val_; }
    //real frequencies (omega + i*eta)
    const ca_complex_vector &freq() const { return freq_; }
    //write real frequencies and their matrix values to the output file
    //trace form
    void trace_write (std::string kfile) {
        std::ofstream ofs(kfile);
        for(int i = 0; i < N_real_; i++){
            ofs << std::fixed << std::setprecision(15);
            ofs << freq_[i].real() << " " 
                << - val_[i].trace().imag() / M_PI << std::endl;
        }
    }
    //compact form
    void compact_write (std::string cfile) {
        std::ofstream ofs(cfile);
        for(int i = 0; i < N_real_; i++){
            ofs << std::fixed << std::setprecision(15);
            ofs << freq_[i].real() << " ";
            for (int j = 0; j < dim_; j++)
                for (int k = 0; k < dim_; k++)
                    ofs << val_[i](j, k).real() << " " << val_[i](j, k).imag() << " ";
            ofs << std::endl;
        }
    }
private:
    std::ofstream ofs;
    int N_real_;
    int dim_;
    ca_complex_matrix_vector val_;
    ca_complex_vector freq_;
};


#endif
