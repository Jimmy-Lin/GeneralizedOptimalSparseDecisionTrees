AC_PREREQ([2.69])
AC_INIT([gosdt],[0.1.0])
: ${CXXFLAGS=-O3} # for production release
# : ${CXXFLAGS=-O0 -g} # for debugging
AC_CONFIG_MACRO_DIR([auto])
AC_CONFIG_AUX_DIR([build])
m4_include([auto/ax_cxx_compile_stdcxx.m4])
m4_include([auto/acx_pthread.m4])
m4_include([auto/boost.m4])
m4_include([auto/ax_check_cl.m4])

AC_CANONICAL_HOST
build_linux=no
build_windows=no
build_mac=no
# Detect the target system
case "${host_os}" in
    linux*)
        build_linux=yes
        ;;
    cygwin*|mingw*)
        build_windows=yes
        ;;
    darwin*)
        build_mac=yes
        ;;
    *)
        AC_MSG_ERROR(["OS $host_os is not supported"])
        ;;
esac
AM_CONDITIONAL([LINUX], [test "$build_linux" = "yes"])
AM_CONDITIONAL([WINDOWS], [test "$build_windows" = "yes"])
AM_CONDITIONAL([OSX], [test "$build_mac" = "yes"])

AC_ARG_ENABLE(
    [tests],
    AS_HELP_STRING([--enable-tests], [Include test suite in build process])
    [echo "Including tests in build process"])
AM_CONDITIONAL([INCLUDE_TESTS], [test "x$enable_tests" = "xyes"])

AC_ARG_ENABLE(
    [opencl],
    AS_HELP_STRING([--enable-opencl], [Include OpenCL in build process])
    [echo "Including OpenCL in build process"])
AM_CONDITIONAL([INCLUDE_OPENCL], [test "x$enable_opencl" = "xyes"])

# LT_INIT([shared disable-static]) # Configure for shared library instead of static library

# Checks for programs.
AC_PROG_CXX([g++]) # Check for C++ compiler (restricted to clang) and set it to the variable CXX
AX_CXX_COMPILE_STDCXX([11]) # Macro for checking the existence of stdlib for c++11

AC_LANG(C++) # Set test language to C++

# Check for a standard header
AC_CHECK_HEADERS([iostream], [], [AC_MSG_WARN([IO Stream not found.])])

# Check for GMP Library
AC_CHECK_LIB(gmp, __gmpz_init, ,
  [AC_MSG_ERROR([GNU MP is missing on this system. Please install GNU MP and try again.])])

# Check for Boost Library
BOOST_REQUIRE([1.55])

# Check for Intel TBB Library
SAVED_LDFLAGS=$LDFLAGS
LDFLAGS="$LDFLAGS -l tbb"
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM(
        [#include <tbb/concurrent_hash_map.h>], 
        []
    )],
    [TEST_LIBS="$TEST_LIBS -l tbb"] [TBB_INSTALLED=1], 
    [AC_MSG_ERROR([Intel TBB is missing on this system. Please install Intel TBB and try again.])]
)
LDFLAGS=$SAVED_LDFLAGS

# Check for OpenCL Library
if test "x$enable_opencl" = "xyes"; then
    echo "CHECKING FOR OPENCL"
    AX_CHECK_CL
    if test "X$no_cl" = "Xyes"; then
        AC_MSG_ERROR([OpenCL is missing on this system. Please install OpenCL and try again.])
    fi
fi

# # Checks for typedefs, structures, and compiler characteristics.
# AC_CHECK_HEADER_STDBOOL
# AC_C_INLINE
# AC_TYPE_INT16_T
# AC_TYPE_INT32_T
# AC_TYPE_INT64_T
# AC_TYPE_INT8_T
# AC_TYPE_MODE_T
# AC_TYPE_PID_T
# AC_TYPE_SIZE_T
# AC_TYPE_SSIZE_T
# AC_STRUCT_ST_BLOCKS
# AC_CHECK_MEMBERS([struct stat.st_rdev])
# AC_TYPE_UINT16_T
# AC_TYPE_UINT32_T
# AC_TYPE_UINT64_T
# AC_TYPE_UINT8_T
# AC_CHECK_TYPES([ptrdiff_t])

# # Checks for library functions.
# AC_FUNC_MALLOC
# AC_FUNC_REALLOC
# AC_FUNC_STRTOD
# AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localeconv memmove memset modf pow sqrt strerror strtoull])

# AC_CONFIG_SUBDIRS([autoconf])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
AC_CONFIG_FILES([Makefile])

AC_OUTPUT
