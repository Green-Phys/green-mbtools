from numpy.distutils.core import setup, Extension


# Build Nevanlinna extension
nevanlinna = Extension(
    'mbanalysis.nevanlinna',
    sources=['Nevanlinna/nevanlinna.cpp', ],
    include_dirs=[
        '/usr/local/include',
        '/usr/local/include/eigen3',
    ],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=["-std=c++11"]
)

# Build Caratheodory extension
caratheodory = Extension(
    'mbanalysis.caratheodory',
    sources=['Caratheodory/caratheodory.cpp', ],
    include_dirs=[
        '/usr/local/include',
        '/usr/local/include/eigen3',
    ],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=["-std=c++11"]
)

setup(
   name='mbanalysis',
   version='1.3.0',
   description="A package for post processing of finite-temperature \
       Green's function and self-energy data",
   packages=['mbanalysis', ],
   ext_modules=[nevanlinna, caratheodory]
)
