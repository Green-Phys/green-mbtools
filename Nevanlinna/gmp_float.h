/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GREEN_AC_GMP_FLOAT_H
#define GREEN_AC_GMP_FLOAT_H

#include <gmpxx.h>

#include <Eigen/Core>
#include <complex>
#include <type_traits>

namespace green::ac {

  using gmp_float = mpf_class;

  namespace gmp::type_traits {
    template <typename T>
    using convertible_t = std::enable_if_t<!std::is_same_v<std::decay_t<T>, gmp_float> && std::is_arithmetic_v<T>, gmp_float>;
    template <typename T>
    constexpr bool is_scalar = /*std::is_same_v<std::decay_t<T>, mpfr_float> || */ std::is_arithmetic_v<T>;
    template <typename T>
    constexpr bool is_gmp_scalar = std::is_same_v<std::decay_t<T>, gmp_float> /* || std::is_arithmetic_v<T>*/;

  }  // namespace gmp::type_traits

  inline double to_double(const gmp_float& v) { return v.get_d(); }

}  // namespace green::ac

namespace std {
  // std::complex explicit template specialization for gmp_float to overcome libc++ limitations.
  template <>
  class complex<green::ac::gmp_float> {
    using real_t = green::ac::gmp_float;

  public:
             complex() : _real(real_t(0, 500)), _imag(real_t(0, 500)) {}
             complex(const real_t& real, const real_t& imag) : _real(real), _imag(imag) {}
             complex(real_t&& real, real_t&& imag) : _real(real), _imag(imag) {}
             complex(const real_t& real) : _real(real), _imag(real_t(0)) {}

             complex(const std::complex<real_t>& cplx) : _real(cplx.real()), _imag(cplx.imag()) {}
             complex(std::complex<real_t>&& cplx) noexcept : _real(std::move(cplx._real)), _imag(std::move(cplx._imag)) {}

    complex& operator=(const real_t& rhs) {
      _real = rhs;
      _imag = real_t(0);
      return *this;
    }
    complex& operator=(const complex& rhs) = default;
    complex& operator=(complex&& rhs)      = default;

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator-=(const std::complex<S>& rhs) {
      _real -= rhs.real();
      _imag -= rhs.imag();
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator+=(const std::complex<S>& rhs) {
      _real += rhs.real();
      _imag += rhs.imag();
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator*=(const std::complex<S>& rhs) {
      *this = *this * rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator/=(const std::complex<S>& rhs) {
      *this = *this / rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator-=(const S& rhs) {
      _real -= rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator+=(const S& rhs) {
      _real += rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator*=(const S& rhs) {
      _real *= rhs;
      _imag *= rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_gmp_scalar<S>, complex>& operator/=(const S& rhs) {
      _real /= rhs;
      _imag /= rhs;
      return *this;
    }

    template <typename S, std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
    complex(const S& real) : _real(real_t(real), 500), _imag(real_t(0), 500) {}

    template <typename S>
    complex(const std::complex<S>& cplx) :
        _real(green::ac::gmp::type_traits::convertible_t<S>(cplx.real(), 500)), _imag(real_t(cplx.imag(), 500)) {}

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator=(const S& rhs) {
      _real = real_t(rhs);
      _imag = real_t(0);
      return *this;
    }
    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator=(const std::complex<S>& rhs) {
      _real = real_t(rhs.real());
      _imag = real_t(rhs.imag());
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator-=(const complex<S>& rhs) {
      _real -= real_t(rhs.real());
      _imag -= real_t(rhs.imag());
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator+=(const complex<S>& rhs) {
      _real += real_t(rhs.real());
      _imag += real_t(rhs.imag());
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator*=(const complex<S>& rhs) {
      *this = *this * rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator/=(const complex<S>& rhs) {
      *this = *this / rhs;
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator-=(const S& rhs) {
      _real -= real_t(rhs);
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator+=(const S& rhs) {
      _real += real_t(rhs);
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator*=(const S& rhs) {
      _real *= real_t(rhs);
      _imag *= real_t(rhs);
      return *this;
    }

    template <typename S>
    std::enable_if_t<green::ac::gmp::type_traits::is_scalar<S>, complex>& operator/=(const S& rhs) {
      _real /= real_t(rhs);
      _imag /= real_t(rhs);
      return *this;
    }

    complex operator-() const { return complex{-_real, -_imag}; }

  private:
    real_t _real;
    real_t _imag;

  public:
    [[nodiscard]] const real_t& real() const { return _real; }
    [[nodiscard]] const real_t& imag() const { return _imag; }

    void                        real(const real_t& re) { _real = re; }
    void                        imag(const real_t& im) { _imag = im; }
  };

  template <>
  inline bool operator==(const complex<green::ac::gmp_float>& x, const complex<green::ac::gmp_float>& y) {
    return x.real() == y.real() && x.imag() == y.imag();
  }

  template <>
  inline bool operator==(const complex<green::ac::gmp_float>& x, const green::ac::gmp_float& y) {
    return x.real() == y && x.imag() == green::ac::gmp_float(0);
  }

  template <>
  inline bool operator==(const green::ac::gmp_float& x, const complex<green::ac::gmp_float>& y) {
    return x == y.real() && green::ac::gmp_float(0) == y.imag();
  }

  template <typename S>
  std::enable_if_t<is_arithmetic_v<S>, bool> operator==(const complex<green::ac::gmp_float>& x, const complex<S>& y) {
    return abs(x.real() - green::ac::gmp_float(y.real())) < 1e-12 && abs(x.imag() - green::ac::gmp_float(y.imag())) < 1e-12;
  }

  template <typename S>
  std::enable_if_t<is_arithmetic_v<S>, bool> operator==(const complex<green::ac::gmp_float>& x, S y) {
    return abs(x.real() - green::ac::gmp_float(y)) < 1e-12 && x.imag() == 0;
  }

  template <typename S>
  std::enable_if_t<is_arithmetic_v<S>, bool> operator==(const S& y, const complex<green::ac::gmp_float>& x) {
    return abs(x.real() - green::ac::gmp_float(y)) < 1e-12 && x.imag() == 0;
  }

  inline auto operator*(const complex<green::ac::gmp_float>& x, const complex<green::ac::gmp_float>& y) {
    const green::ac::gmp_float& a = x.real();
    const green::ac::gmp_float& b = x.imag();
    const green::ac::gmp_float& c = y.real();
    const green::ac::gmp_float& d = y.imag();
    return std::complex<green::ac::gmp_float>((a * c - b * d), (a * d + b * c));
  }

  inline auto operator/(const complex<green::ac::gmp_float>& x, const complex<green::ac::gmp_float>& y) {
    const green::ac::gmp_float& a     = x.real();
    const green::ac::gmp_float& b     = x.imag();
    const green::ac::gmp_float& c     = y.real();
    const green::ac::gmp_float& d     = y.imag();
    green::ac::gmp_float        denom = c;
    denom *= c;
    denom += d * d;
    return std::complex<green::ac::gmp_float>((a * c + b * d) / denom, (b * c - a * d) / denom);
  }

  inline auto operator+(const complex<green::ac::gmp_float>& x, const complex<green::ac::gmp_float>& y) {
    std::complex<green::ac::gmp_float> r(x);
    r += y;
    return r;
  }  // LCOV_EXCL_LINE

  inline auto operator-(const complex<green::ac::gmp_float>& x, const complex<green::ac::gmp_float>& y) {
    std::complex<green::ac::gmp_float> r(x);
    r -= y;
    return r;
  }  // LCOV_EXCL_LINE

  inline complex<green::ac::gmp_float> conj(const complex<green::ac::gmp_float>& x) {
    return complex<green::ac::gmp_float>(x.real(), -x.imag());
  }

  inline green::ac::gmp_float abs(const complex<green::ac::gmp_float>& x) {
    return sqrt(x.real() * x.real() + x.imag() * x.imag());
  }

  inline green::ac::gmp_float real(const complex<green::ac::gmp_float>& x) { return x.real(); }

  inline green::ac::gmp_float imag(const complex<green::ac::gmp_float>& x) { return x.imag(); }
}  // namespace std

namespace Eigen {

  template <>
  struct NumTraits<std::complex<green::ac::gmp_float>> :
      NumTraits<std::complex<double>>  // permits to get the epsilon, dummy_precision, lowest, highest functions
  {
    typedef green::ac::gmp_float Real;
    typedef green::ac::gmp_float NonInteger;
    typedef green::ac::gmp_float Nested;

    enum { IsComplex = 1, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 1, AddCost = 3, MulCost = 3 };

    
  };

}  // namespace Eigen

#endif  // GREEN_AC_GMP_FLOAT_H
