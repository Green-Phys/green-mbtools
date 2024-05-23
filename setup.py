from setuptools import setup, Extension


# Build Nevanlinna extension
nevanlinna = Extension(
    'mbanalysis.nevanlinna',
    sources=['Nevanlinna/nevanlinna.cpp', ],
    include_dirs=['/usr/include/eigen3', ],
    libraries=['gmp', 'gmpxx', 'mpfr']
)

# Build Caratheodory extension
caratheodory = Extension(
    'mbanalysis.caratheodory',
    sources=['Caratheodory/caratheodory.cpp', ],
    include_dirs=['/usr/include/eigen3', ],
    libraries=['gmp', 'gmpxx', 'mpfr'],
)

setup(
   name='mbanalysis',
   version='1.3.0',
   description="A package for post processing of finite-temperature \
       Green's function and self-energy data",
   packages=['mbanalysis', ],
   ext_modules=[nevanlinna, caratheodory]
)
