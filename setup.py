import glob
import os
import platform
import setuptools
from distutils.core import Extension


# This script is used to build and/or install the trainer into as Python extension.
# To build the extention, run: python python/extension/setup.py build
# To install the extention (after build), run: python python/extension/setup.py install

# Please only build and install the extension using the same installation of Python
# as the one intended for interacting with the library

# Force distutil to use the g++ compiler
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# Standard Build Configuration
OPTIMIZATION = ['-O3']
STD = ['-std=c++11']
INCLUDES = ['-I', 'include']

# Platform Specific Build Configuration
if platform.system() == "Darwin":
    STDLIB = ['-stdlib=libc++']
    TBB_LIBS = ['-ltbb', '-ltbbmalloc']
    CL_LIBS = ['-framework', 'OpenCL']
    GMP_LIBS = ['-lgmp']
elif platform.system() == "Linux":
    STDLIB = []
    TBB_LIBS = ['-ltbb', '-ltbbmalloc']
    CL_LIBS = ['-lOpenCL']
    GMP_LIBS = ['-lgmp']

COMPILE_ARGS = OPTIMIZATION + STD + INCLUDES + STDLIB
LINK_ARGS = OPTIMIZATION + STD + INCLUDES + STDLIB + TBB_LIBS + CL_LIBS + GMP_LIBS

module = Extension(
    name="gosdt",
    # sources=['src/python_extension.cpp'],
    sources=[obj for obj in glob.glob('src/*.cpp')],
    language='c++',
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
    extra_objects=[obj for obj in glob.glob('src/*.o')]
)

setuptools.setup(
    name="gosdt-deprecated",
    version="0.0.1",
    author="Jimmy Lin, Chudi Zhong, and others",
    author_email="jimmy.projects.lin@gmail.com",
    description="C++ implementation of Generalized Optimal Sparse Decision Trees",
    ext_modules=[module],
    url="https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees",
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
)
