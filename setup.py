from numpy.distutils.core import setup, Extension


# Build Nevanlinna extension
src = 'Nevanlinna'
files = ['nevanlinna.cpp']
sources = []
for fi in files:
    sources.append(src + '/' + fi)

nevanlinna = Extension(
    'mbanalysis.nevanlinna',
    sources=sources,
    include_dirs=['/usr/include/eigen3', ],
    libraries=['gmp', 'gmpxx']
)

setup(
   name='mbanalysis',
   version='1.1',
   description="A package for post processing of finite-temperature \
       Green's function and self-energy data",
   packages=['mbanalysis', 'mbanalysis.src'],
   ext_modules=[nevanlinna, ]
)
