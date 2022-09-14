import platform
import os
import pathlib
import sys
import distro

from setuptools import find_packages
from skbuild import setup

cmake_args = []

if platform.system() == "Windows" or (platform.system() == "Linux" and distro.id() == "centos"):
    assert "VCPKG_INSTALLATION_ROOT" in os.environ, \
        "The environment variable \"VCPKG_INSTALLATION_ROOT\" must be set before running this script."
    toolchain_path = pathlib.Path(os.getenv("VCPKG_INSTALLATION_ROOT")) / "scripts/buildsystems/vcpkg.cmake"
    cmake_args.append("-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain_path))

print("Additional CMake Arguments = {}".format(cmake_args))

# Fetch the current Python version
# Set the environment variable so that CMake can find the header search path of the current Python installation
version_info = sys.version_info
version = "{}.{}".format(version_info.major, version_info.minor)
os.environ["PYTHON3_VERSION"] = version
assert "PYTHON3_VERSION" in os.environ
print("The current Python version is {}.".format(version))

setup(
    name="gosdt",
    version="1.0.5",
    description="Implementation of General Optimal Sparse Decision Tree",
    author="UBC Systopia Research Lab",
    license="BSD 3-Clause",
    packages=find_packages(where='.'),
    cmake_install_dir="gosdt",
    cmake_args=cmake_args,
    python_requires=">=3.6",
)