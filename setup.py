from setuptools import setup, Extension
import sys

include_dirs = ['/usr/local/include', '/opt/homebrew/include', '/usr/local/include/eigen3', '/opt/homebrew/include/eigen3' ] if sys.platform=='darwin' else ['/usr/include/eigen3']
library_dirs = ['/opt/homebrew/lib','/usr/local/lib' ] if sys.platform=='darwin' else []

# Build Nevanlinna extension
nevanlinna = Extension(
    'green_mbtools.pesto.nevanlinna',
    sources=['Nevanlinna/nevanlinna.cpp' ],
    include_dirs=include_dirs + ['Nevanlinna',],
    library_dirs=library_dirs,
    depends=['Nevanlinna/schur.h','Nevanlinna/nevanlinna.h','Nevanlinna/gmp_float.h'],
    libraries=['gmp', 'gmpxx'],
    extra_compile_args=["-std=c++17"]
)

# Build Caratheodory extension
caratheodory = Extension(
    'green_mbtools.pesto.caratheodory',
    sources=['Caratheodory/caratheodory.cpp'],
    include_dirs=include_dirs + ['Caratheodory',],
    library_dirs=library_dirs,
    depends=['Caratheodory/carath.h','Caratheodory/iter.h','Caratheodory/mpfr_float.h'],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=["-std=c++17"]
)

setup(
   ext_modules=[nevanlinna, caratheodory]
)
