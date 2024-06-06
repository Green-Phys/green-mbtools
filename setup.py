from setuptools import setup, Extension

# Build Nevanlinna extension
nevanlinna = Extension(
    'green_mbtools.pesto.nevanlinna',
    sources=['Nevanlinna/nevanlinna.cpp', ],
    include_dirs=['/usr/include/eigen3', ],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=["-std=c++17"]
)

# Build Caratheodory extension
caratheodory = Extension(
    'green_mbtools.pesto.caratheodory',
    sources=['Caratheodory/caratheodory.cpp', ],
    include_dirs=['/usr/include/eigen3', ],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=["-std=c++17"]
)

setup(
   ext_modules=[nevanlinna, caratheodory]
)
