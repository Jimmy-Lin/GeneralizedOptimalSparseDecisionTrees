import platform
import sys
import os
import pathlib
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

cmake_args = []

if platform.system() == "Windows":
    assert "VCPKG" in os.environ, "The environment variable \"VCPKG\" must be set before running this script."
    toolchain_path = pathlib.Path(os.getenv("VCPKG")) / "scripts/buildsystems/vcpkg.cmake"
    cmake_args.append("-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain_path))

print("Additional CMake Arguments = {}".format(cmake_args))

setup(
    name="gosdt",
    version="9.9.9",
    description="Implementation of General Optimal Sparse Decision Tree",
    author="TODO",
    license="BSD 3-Clause",
    packages=find_packages(where='.'),
    cmake_install_dir="gosdt",
    cmake_args=cmake_args,
    python_requires=">=3.6",
)