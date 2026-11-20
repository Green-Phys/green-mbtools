#!/usr/bin/env bash
# Keep bundled GMP/MPFR compatible with macOS 14 on the newer ARM runner.
set -euo pipefail
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
prefix=${1:?Pass the isolated dependency prefix}
build=${RUNNER_TEMP:?}/green-macos-wheel-sources
mkdir -p "$prefix" "$build"
cd "$build"

# Release checksums match the upstream releases recorded by Homebrew/core.
curl --fail --location --retry 3 -o gmp.tar.xz https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz
printf '%s  %s\n' a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898 gmp.tar.xz | shasum -a 256 -c -
curl --fail --location --retry 3 -o mpfr.tar.xz https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.2.tar.xz
printf '%s  %s\n' b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01 mpfr.tar.xz | shasum -a 256 -c -
tar -xf gmp.tar.xz
tar -xf mpfr.tar.xz

export MACOSX_DEPLOYMENT_TARGET=14.0
export CC=clang CXX=clang++
export CFLAGS='-O2 -mmacosx-version-min=14.0'
export CXXFLAGS="$CFLAGS"
export LDFLAGS='-mmacosx-version-min=14.0'
jobs=$(sysctl -n hw.logicalcpu)
(
    cd gmp-6.3.0
    ./configure --prefix="$prefix" --enable-cxx --enable-shared --disable-static
    make -j "$jobs"
    make install
)
(
    cd mpfr-4.2.2
    ./configure --prefix="$prefix" --with-gmp="$prefix" --enable-shared --disable-static
    make -j "$jobs"
    make install
)
for library in libgmp.10.dylib libgmpxx.4.dylib libmpfr.6.dylib; do
    xcrun vtool -show-build "$prefix/lib/$library"
done
