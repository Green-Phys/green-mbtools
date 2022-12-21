#include <complex.h>
#include <fftw3.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <assert.h> //assert when file could not open
#include <vector>


template <typename T>
std::string to_string_p(const T a_value, const int n = 15){
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


double sobolev_main(
    std::string cofile, int n_params, int n_real, double w_min, double w_max,
    double eta, std::string param_file, std::string spectral_file, double lagrange=0.00001
) {
    fftw_complex *inp, *out;
    fftw_plan plan;

    // Open Nevanlinna coefficients file
    std::ifstream ifile_coeff(cofile);
    assert(ifile_coeff);
    // Open parameter file
    std::ifstream ifile_param(param_file);
    assert(ifile_param);
    // Output file for spectral function
    std::ofstream ofile_spectral(spectral_file);
    assert(ofile_spectral);

    inp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_real);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_real);
    plan = fftw_plan_dft_1d(n_real, inp, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Load parameters from ifile_param
    std::vector<std::complex<long double>> par1(n_params);
    std::vector<std::complex<long double>> par2(n_params);
    for (int i=0; i < n_params; i++) {
        long double re1, im1, re2, im2;
        ifile_param >> re1 >> im1 >> re2 >> im2;
        par1[i] = std::complex<long double> {re1, im1};
        par2[i] = std::complex<long double> {re2, im2};
    }

    std::vector<std::complex<long double>> zpow(n_params);
    std::complex<long double> One = std::complex<long double> {1, 0};
    std::complex<long double> Im = std::complex<long double> {0, 1};
    std::complex<long double> sqrt_pi = std::complex<long double> {std::sqrt(M_PI), 0};
    for (int i=0; i < n_real; i++) {
        // Read frequency and Nevanlinna coefficients for the ith real frequency w = freq
        long double freq, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im;
        ifile_coeff >> freq >> a_re >> a_im >> b_re >> b_im >> c_re >> c_im >> d_re >> d_im;
        std::complex<long double> z {freq, eta};
        std::complex<long double> a {a_re, a_im};
        std::complex<long double> b {b_re, b_im};
        std::complex<long double> c {c_re, c_im};
        std::complex<long double> d {d_re, d_im};
        std::complex<long double> z1 = (z - Im) / (z + Im);

        // Hardy polynomial for final Pade fraction
        std::complex<long double> poly1, poly2, poly;
        for (int j=0; j<n_params; j++) {
            zpow[j] = std::pow(z1, j);
            poly1 += par1[j] * (zpow[j] * (One - z1) / sqrt_pi);
            poly2 += par2[j] * std::conj(zpow[j] * (One - z1) / sqrt_pi);
        }
        poly = poly1 + poly2;

        // Nevanlinna continued function for the given Hardy polynomial
        std::complex<long double> theta = (a * poly + b) / (c * poly + d);
        long double real = 1 / M_PI * std::imag(Im * ((One + theta) / (One - theta)));
        inp[i][0] = real;
        inp[i][1] = 0;
        ofile_spectral << freq << " " << real << std::endl;
    }

    fftw_execute(plan);

    // Get (1 - integral(A(w) from w_min to w_max))
    long double sum_out1 = 0;
    for (int i=0; i<n_real; i++) {
        sum_out1 += inp[i][0];
    }
    sum_out1 = std::pow(1 - sum_out1 * (w_max - w_min) / double(n_real), 2);

    // Get the smoothness norm
    long double sum_out2 = 0;
    if (n_real % 2 == 0){
        for (int i=0; i<n_real/2; i++){
            sum_out2 += std::pow(2 * M_PI * i / (w_max - w_min), 4) * (
                std::pow(out[i][0], 2) + std::pow(out[i][1], 2)
            );
        }
        for (int i=n_real/2 + 1; i<n_real; i++) {
            sum_out2 += std::pow(2 * M_PI * (i - n_real) / (w_max - w_min), 4) * (
                std::pow(out[i][0], 2) + std::pow(out[i][1], 2)
            );
        }
    } else {
        for (int i=0; i<n_real; i++) {
            sum_out2 += std::pow(2 * M_PI * std::min(i, n_real - i) / (w_max - w_min), 4) * (
                std::pow(out[i][0], 2) + std::pow(out[i][1], 2)
            );
        }
    }
    sum_out2 = sum_out2 * (w_max - w_min) / double(n_real) / double(n_real);

    // Full Lagrange multiplier
    double sum_out = double(sum_out1 + lagrange * sum_out2);

    // Close files
    ifile_coeff.close();
    ifile_param.close();
    ofile_spectral.close();

    // Wrap up the fftw objects
    fftw_destroy_plan(plan);
    fftw_free(inp);
    fftw_free(out);

    return sum_out;
}
