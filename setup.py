from distutils.core import setup, Extension
import glob
import sys
import os
import platform

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
    name='gosdt',
    # sources=['src/python_extension.cpp'],
    sources=[obj for obj in glob.glob('src/*.cpp')],
    language='c++',
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
    extra_objects=[obj for obj in glob.glob('src/*.o')]
)

setup(
    name='gosdt',
    version='0.1.1',
    ext_modules=[module]
)
