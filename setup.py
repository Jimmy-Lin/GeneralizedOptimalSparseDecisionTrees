import platform
import os
import pathlib
from setuptools import find_packages
from skbuild import setup

cmake_args = []

if platform.system() == "Windows":
    assert "VCPKG_INSTALLATION_ROOT" in os.environ, \
        "The environment variable \"VCPKG_INSTALLATION_ROOT\" must be set before running this script."
    toolchain_path = pathlib.Path(os.getenv("VCPKG_INSTALLATION_ROOT")) / "scripts/buildsystems/vcpkg.cmake"
    cmake_args.append("-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain_path))

print("Additional CMake Arguments = {}".format(cmake_args))

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