from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("numpy_shuffle_row_wise.pyx")
)
