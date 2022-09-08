import sys
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

print("Packages:")
print(find_packages(where="."))

setup(
    name="gosdt",
    version="9.9.9",
    description="Implementation of General Optimal Sparse Decision Tree",
    author="TODO",
    license="BSD 3-Clause",
    packages=find_packages(where='.'),
    cmake_install_dir="gosdt",
    python_requires=">=3.6",
)