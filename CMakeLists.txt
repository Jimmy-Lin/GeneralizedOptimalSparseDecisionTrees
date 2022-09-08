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
set(TARGET_CLI "gosdt_cli")
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

# Target: GOSDT Python Module
if (SKBUILD)
    message(STATUS "GOSDT is built using scikit-build. Will build the Python module.")
    find_package(Python3 COMPONENTS Development)
    file(GLOB_RECURSE SOURCE_FILES_PY src/*.cpp)
    list(REMOVE_ITEM SOURCE_FILES_PY "${CMAKE_SOURCE_DIR}/src/main.cpp")
    message(STATUS ${SOURCE_FILES_PY})
    add_library(${TARGET_PYTHON} MODULE ${SOURCE_FILES_PY})
    target_include_directories(${TARGET_PYTHON} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Python3_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})
    target_link_libraries(${TARGET_PYTHON} TBB::tbb TBB::tbbmalloc PkgConfig::GMP)
    find_package(PythonExtensions REQUIRED)
    python_extension_module(${TARGET_PYTHON})
    install(TARGETS ${TARGET_PYTHON} LIBRARY DESTINATION .)
endif()
