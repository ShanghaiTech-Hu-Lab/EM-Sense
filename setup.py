from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("gaussian", sources=["gaussian.pyx"],
        include_dirs=[numpy.get_include()]),
]
setup(
    name="gaussian",
    ext_modules=cythonize(extensions),
)