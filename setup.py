from distutils.core import setup, Extension
import glob
import sys
import os
import platform
import subprocess
from datetime import datetime
import platform

# This script is used to build and/or install the trainer into as Python extension.
# To build the extention, run: python python/extension/setup.py build
# To install the extention (after build), run: python python/extension/setup.py install

# Please only build and install the extension using the same installation of Python
# as the one intended for interacting with the library

# Force distutil to use the g++ compiler
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if "GOSDT_BUILD_OPT_FLAGS" in os.environ :
    OPTIMIZATION = os.environ['GOSDT_BUILD_OPT_FLAGS'].split(";")
else :
    OPTIMIZATION = ['-O3', "-march=native"]

# get the git version
label = subprocess.check_output(["git", "describe", "--always", "--tags", '--dirty=-dirty']).strip()
gitrev = label.decode('utf-8')
hostname = platform.node()
# Wed Sep 15 05:08:56 PM PDT 2021   # strftime("%a %b %d %r %Z %Y")
date = datetime.now().strftime("%Y%m%d-%H%M%S")


# Standard Build Configuration
STD = ['-std=gnu++11']
INCLUDES = ['-I', 'include']
DEFINES = [
    f'-DBUILD_GIT_REV=\"{gitrev}\"',
    f'-DBUILD_HOST=\"{hostname}\"',
    f'-DBUILD_DATE=\"{date}\"',
]

# Platform Specific Build Configuration
if platform.system() == "Darwin":
    STDLIB = ['-stdlib=libc++']
    TBB_LIBS = ['-ltbb', '-ltbbmalloc']
    GMP_LIBS = ['-lgmp']
elif platform.system() == "Linux":
    STDLIB = []
    TBB_LIBS = ['-ltbb', '-ltbbmalloc']
    GMP_LIBS = ['-lgmp']

COMPILE_ARGS = OPTIMIZATION + STD + INCLUDES + STDLIB + DEFINES
LINK_ARGS = OPTIMIZATION + STD + INCLUDES + STDLIB + TBB_LIBS + GMP_LIBS

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
