## How to build the project

This page documents how you can build the project and the Python extension module manually.

### Step 1: Install required development tools

GOSDT uses `CMake` as its default cross-platform build system and `Ninja` as the default generator for parallel builds.  
GOSDT relies on `scikit-build` to compile the Python extension and build the redistributable wheel file.  
GOSDT exploits `delocate`, `auditwheel` and `delvewheel` to copy all required 3rd-party dynamic libraries into the wheel file on macOS, Ubuntu and Windows respectively.

**macOS:**

```bash
brew install cmake
brew install ninja
brew install pkg-config
pip3 install --upgrade scikit-build
pip3 install --upgrade delocate
```

**Ubuntu:**

```bash
sudo apt install -y cmake
sudo apt install -y ninja-build
sudo apt install -y pkg-config
pip3 install --upgrade scikit-build
pip3 install --upgrade auditwheel
sudo apt install -y patchelf # Required by auditwheel
```

**Windows:**

**Step 1.1:** 

Please make sure that you launch the Powershell as Admin.

```ps1
winget install Kitware.CMake
Invoke-WebRequest -Uri "https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip" -OutFile "C:\ninja-win.zip"
Expand-Archive "C:\ninja-win.zip" -DestinationPath "C:\Windows\"
New-Item -Path "C:\Windows\ninja-build.exe" -ItemType SymbolicLink -Value "C:\Windows\ninja.exe"
Remove-Item C:\ninja-win.zip
pip3 install --upgrade scikit-build
pip3 install --upgrade delvewheel
```

**Step 1.2:**

Additionally, please follow the [guide](https://vcpkg.io/en/getting-started.html) to install the C++ package manager `vcpkg` on Windows.  
Once you have installed `vcpkg`, for example, to `C:\vcpkg`, you need to...
- Update your `PATH` variable to include `C:\vcpkg`.
- Add a new environment variable `VCPKG_INSTALLATION_ROOT` with a value of `C:\vcpkg`.

You can verify whether the new variable `VCPKG_INSTALLATION_ROOT` is set properly by typing the following command in Powershell:

```ps1
$ENV:VCPKG_INSTALLATION_ROOT
```

**Step 1.3:**

GMP does not come with a CMake module file, so `pkgconfig` is needed to find the GMP library on Windows, Ubuntu and macOS.  
Please follow the guide on [StackOverflow](https://stackoverflow.com/a/22363820).  
In short, download all those three zip files, extract them to, for example, `C:\pkgconfig`, and update the PATH variable on Windows.

### Step 2: Install required 3rd-party libraries

GOSDT relies on `IntelTBB` and `GMP`.
Header-only libraries are already included in this repository.

**macOS:**

```bash
brew install tbb
brew install gmp
```

**Ubuntu:**

```bash
sudo apt install -y libtbb-dev
sudo apt install -y libgmp-dev
```

**Windows:**

```ps1
vcpkg install tbb:x64-windows
vcpkg install gmp:x64-windows
```

### Step 3: Build the project

You can build the GOSDT project by...
- Using the automatic build script `build.py` (GOSDT CLI + Python Wheel)
- Using `scikit-build` manually (GOSDT CLI + Python Wheel)
- Using `cmake` manually (GOSDT CLI Only)

#### Method 1:

This repository ships a `build.py` script that builds the library and the Python wheel automatically.

```bash
python3 build.py
```

#### Method 2:

**Step 3.1:** Build the C++ library and the Python wheel

Please adjust the number of threads `-j8` accordingly.

```bash
# Debug Build
python3 setup.py bdist_wheel --build-type=Debug -G Ninja -- -- -j8

# Release Build
python3 setup.py bdist_wheel --build-type=Release -G Ninja -- -- -j8
```

You can find the command line tools in `_skbuild/<platform-specific>/cmake-build/` and the wheel file in `dist/`.

**Step 3.2:** Add all required 3rd-party libraries to the wheel file

Please adjust the name of your wheel file accordingly.

**macOS:**

```bash
delocate-wheel -w dist -v dist/gosdt-1.0.5-cp310-cp310-macosx_12_0_x86_64.whl
```

**Ubuntu:**

```bash
auditwheel repair -w dist --plat linux_x86_64 dist/gosdt-1.0.5-cp310-cp310-linux_x86_64.whl
```

**Windows:**

```ps1
python3 -m delvewheel repair --no-mangle-all --add-path "$ENV:VCPKG_INSTALLATION_ROOT\installed\x64-windows\bin" dist/gosdt-1.0.5-cp310-cp310-win_amd64.whl -w dist
```

You will find the fixed wheel file in `dist`.

#### Method 3:

If you build the project on Ubuntu or macOS, you need to remove the `-DCMAKE_TOOLCHAIN_FILE=...` option.  
If you build the project on Windows, you must run the following commands in a **Developer Powershell for Visual Studio**.  
Please adjust the number of threads `--parallel 8` accordingly.

**Debug Build:**

```bash
# Create the build directory
mkdir build

# Generate all necessary files for the low-level build system
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ENV:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug 

# Compile the project from scratch with 8 threads
cmake --build build --config Debug --clean-first --parallel 8

# Launch the GOSDT executable and its tests
.\build\Debug\gosdt.exe
.\build\Debug\gosdt_tests.exe
```

**Release Build:**

```bash
# Create the build directory
mkdir build

# Generate all necessary files for the low-level build system
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ENV:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# Compile the project from scratch with 8 threads
cmake --build build --config Release --clean-first --parallel 8

# Launch the GOSDT executable and its tests
.\build\Release\gosdt.exe
.\build\Release\gosdt_tests.exe
```

### Step 4: Build the manylinux wheel (Optional)

```bash
# Change the working directory to this repository
cd path/to/gosdt-guesses-internal

# Run the CentOS docker provided by manylinux
# The repository is mapped at /source
sudo docker run -i -t --mount type=bind,source=`pwd`,target=/source quay.io/pypa/manylinux2014_x86_64

# CentOS Shell
# Install required development tools
yum install -y zip
yum install -y cmake
yum install -y ninja-build
yum install pkgconfig
python3.7 -m pip install --upgrade scikit-build
python3.7 -m pip install --upgrade auditwheel
yum install -y patchelf

# Install the VCPKG package manager
git clone https://github.com/Microsoft/vcpkg.git 
./vcpkg/bootstrap-vcpkg.sh
export PATH=/vcpkg:$PATH
export VCPKG_INSTALLATION_ROOT=/vcpkg

# Install required 3rd-party libraries
vcpkg install tbb
vcpkg install gmp

# Change the working directory to the repository
cd /source

# Build the project and the wheel file
python3.7 build.py

# Quit the CentOS shell
exit
```

You can find the manylinux wheel in `dist`.