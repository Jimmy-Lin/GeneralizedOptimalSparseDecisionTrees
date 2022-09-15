cmake_minimum_required(VERSION 3.10)

# Project `Name` and `Language`
project(GOSDT)

# Set the language standard to `c++11`
set(CMAKE_CXX_STANDARD 11)

# Set the compiler flags
if (MSVC)
    set(CMAKE_C_FLAGS                   "${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG             "${CMAKE_C_FLAGS_DEBUG} /DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE           "${CMAKE_C_FLAGS_RELEASE}")

    # `#define NOMINMAX` prevents expansion of min and max macros on Windows,
    # otherwise `std::numeric_limits<T>::max()/min()` leads to MSVC compiler errors.
    # Reference: https://stackoverflow.com/questions/27442885/syntax-error-with-stdnumeric-limitsmax
    set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} /bigobj /w /DNOMINMAX")
    set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE}")
else()
    set(CMAKE_C_FLAGS                   "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG             "${CMAKE_C_FLAGS_DEBUG} -O0 -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE           "${CMAKE_C_FLAGS_RELEASE}")

    set(CMAKE_CXX_FLAGS                 "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

#
# MARK: - Targets
#

# Target Definition
set(TARGET_CLI "gosdt")
set(TARGET_TESTS "gosdt_tests")
set(TARGET_PYTHON "libgosdt")

# Target: GOSDT CLI
file(GLOB_RECURSE SOURCE_FILES_CLI src/*.cpp)
list(REMOVE_ITEM SOURCE_FILES_CLI "${CMAKE_SOURCE_DIR}/src/python_extension.cpp")
add_executable(${TARGET_CLI} ${SOURCE_FILES_CLI})
target_include_directories(${TARGET_CLI} PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Target: GOSDT Tests
file(GLOB_RECURSE SOURCE_FILES_TESTS src/*.cpp)
list(REMOVE_ITEM SOURCE_FILES_TESTS "${CMAKE_SOURCE_DIR}/src/python_extension.cpp")
list(REMOVE_ITEM SOURCE_FILES_TESTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(INSERT SOURCE_FILES_TESTS 0 "${CMAKE_SOURCE_DIR}/test/test.cpp")
add_executable(${TARGET_TESTS} ${SOURCE_FILES_TESTS})
target_include_directories(${TARGET_TESTS} PUBLIC ${CMAKE_SOURCE_DIR}/include)

#
# MARK: - Dependencies
#

# Dependencies: Intel TBB
find_package(TBB REQUIRED)
target_link_libraries(${TARGET_CLI} PRIVATE TBB::tbb)
target_link_libraries(${TARGET_CLI} PRIVATE TBB::tbbmalloc)
target_link_libraries(${TARGET_TESTS} PRIVATE TBB::tbb)
target_link_libraries(${TARGET_TESTS} PRIVATE TBB::tbbmalloc)

# Dependencies: GMP
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
target_link_libraries(${TARGET_CLI} PRIVATE PkgConfig::GMP)
target_include_directories(${TARGET_CLI} PRIVATE ${GMP_INCLUDE_DIRS})
target_link_libraries(${TARGET_TESTS} PRIVATE PkgConfig::GMP)
target_include_directories(${TARGET_TESTS} PRIVATE ${GMP_INCLUDE_DIRS})

# Dependencies: Threads (pthread on macOS and Ubuntu, win32 thread on Windows)
# This is needed because the CentOS docker provided by manylinux reports linker errors
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${TARGET_CLI} PRIVATE Threads::Threads)
target_link_libraries(${TARGET_TESTS} PRIVATE Threads::Threads)

# Target: GOSDT Python Module
if (SKBUILD)
    message(STATUS "GOSDT is built using scikit-build. Will build the Python module.")
    # Find the Python 3 development environment
    if (NOT DEFINED Python3_INCLUDE_DIR)
        message(FATAL_ERROR "The CMake variable Python3_INCLUDE_DIR should have been defined by scikit-build.")
    endif()
    # Create the list of source files needed to build the Python extension
    file(GLOB_RECURSE SOURCE_FILES_PY src/*.cpp)
    list(REMOVE_ITEM SOURCE_FILES_PY "${CMAKE_SOURCE_DIR}/src/main.cpp")
    # Define the CMake target for the Python extension
    add_library(${TARGET_PYTHON} MODULE ${SOURCE_FILES_PY})
    target_include_directories(${TARGET_PYTHON} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Python3_INCLUDE_DIR} ${GMP_INCLUDE_DIRS})
    target_link_libraries(${TARGET_PYTHON} TBB::tbb TBB::tbbmalloc PkgConfig::GMP Threads::Threads)
    # Set up the Python extension
    find_package(PythonExtensions REQUIRED)
    ## Use the suffix `.abi3.so` or `.pyd` so that Python 3 on other platforms can find the dylib and import it properly
    message(STATUS "The current Python extension suffix is \"${PYTHON_EXTENSION_MODULE_SUFFIX}\".")
    if (WIN32)
        set(PYTHON_EXTENSION_MODULE_SUFFIX ".pyd")
    else()
        set(PYTHON_EXTENSION_MODULE_SUFFIX ".abi3.so")
    endif()
    message(STATUS "The new Python extension suffix is \"${PYTHON_EXTENSION_MODULE_SUFFIX}\".")
    ## Define the Python extension module target
    python_extension_module(${TARGET_PYTHON})
    # Install `libgosdt` to the root directory of the Python extension package
    install(TARGETS ${TARGET_PYTHON} LIBRARY DESTINATION .)
endif()
