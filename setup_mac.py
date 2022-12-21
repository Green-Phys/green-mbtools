from numpy.distutils.core import setup, Extension


# Build Nevanlinna extension
nevanlinna = Extension(
    'mbanalysis.nevanlinna',
    sources=['Nevanlinna/nevanlinna.cpp', ],
    include_dirs=[
        '/usr/local/include',
        '/usr/local/include/eigen3',
    ],
    libraries=['gmp', 'gmpxx', 'mpfr']
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
)

# Build Sobolev extension for Hardy optimization
sobolev = Extension(
    'mbanalysis.sobolev',
    sources=['Nevanlinna/sobolev.cpp', ],
    include_dirs=['/usr/local/include', ],
    library_dirs=['/usr/local/lib', ],
    libraries=['fftw3', 'm']
)

setup(
   name='mbanalysis',
   version='1.2.2',
   description="A package for post processing of finite-temperature \
       Green's function and self-energy data",
   packages=['mbanalysis', 'mbanalysis.src'],
   ext_modules=[nevanlinna, caratheodory, sobolev]
)
