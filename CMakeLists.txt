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
set(TARGET "gosdt")
set(TARGET_TESTS "${TARGET}_tests")

# Target: GOSDT
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/python_extension.cpp")
add_executable(${TARGET} ${SOURCE_FILES})
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Target: GOSDT Tests
set(SOURCE_FILES_TESTS ${SOURCE_FILES})
list(REMOVE_ITEM SOURCE_FILES_TESTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(INSERT SOURCE_FILES_TESTS 0 "${CMAKE_SOURCE_DIR}/test/test.cpp")
add_executable(${TARGET_TESTS} ${SOURCE_FILES_TESTS})
target_include_directories(${TARGET_TESTS} PUBLIC ${CMAKE_SOURCE_DIR}/include)

message(STATUS "SRC MAIN: ${SOURCE_FILES}")
message(STATUS "SRC TEST: ${SOURCE_FILES_TESTS}")

#
# MARK: - Dependencies
#

# Dependencies: Intel TBB
find_package(TBB REQUIRED)
target_link_libraries(${TARGET} PRIVATE TBB::tbb)
target_link_libraries(${TARGET} PRIVATE TBB::tbbmalloc)
target_link_libraries(${TARGET_TESTS} PRIVATE TBB::tbb)
target_link_libraries(${TARGET_TESTS} PRIVATE TBB::tbbmalloc)

# Dependencies: GMP
find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
target_link_libraries(${TARGET} PRIVATE PkgConfig::GMP)
target_include_directories(${TARGET} PRIVATE ${GMP_INCLUDE_DIRS})
target_link_libraries(${TARGET_TESTS} PRIVATE PkgConfig::GMP)
target_include_directories(${TARGET_TESTS} PRIVATE ${GMP_INCLUDE_DIRS})
