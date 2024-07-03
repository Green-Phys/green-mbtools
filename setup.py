from setuptools import setup, Extension
import sys
import platform

include_dirs    = ['/usr/local/include', '/opt/homebrew/include', '/opt/local/include','/usr/local/include/eigen3', '/opt/homebrew/include/eigen3', '/opt/local/include/eigen3'] if sys.platform=='darwin' else ['/usr/include/eigen3']
library_dirs    = ['/opt/homebrew/lib','/opt/local/lib', '/usr/local/lib' ] if sys.platform=='darwin' else []
compile_flags   = ["-std=c++17"]
extra_link_args = []
if sys.platform=='darwin' and int(platform.mac_ver()[0].split('.')[0]) >= 14 :
    compile_flags+=['-arch', 'x86_64', '-arch', 'arm64']
    extra_link_args+=["-undefined", "dynamic_lookup"]

# Build Caratheodory extension
caratheodory = Extension(
    'green_mbtools.pesto.caratheodory',
    sources=['Caratheodory/caratheodory.cpp'],
    include_dirs=include_dirs + ['Caratheodory',],
    library_dirs=library_dirs,
    depends=['Caratheodory/carath.h','Caratheodory/iter.h','Caratheodory/mpfr_float.h'],
    libraries=['gmp', 'gmpxx', 'mpfr'],
    extra_compile_args=compile_flags,
    extra_link_args=extra_link_args
)

setup(
   ext_modules=[caratheodory]
)
