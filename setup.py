from numpy.distutils.core import setup


setup(
   name='mbanalysis',
   version='1.1',
   description="A package for post processing of finite-temperature \
       Green's function and self-energy data",
   packages=['mbanalysis', 'mbanalysis.src'],
   package_data={
       "mbanalysis.data": ["*.h5", "*.npy"]
   }
)
