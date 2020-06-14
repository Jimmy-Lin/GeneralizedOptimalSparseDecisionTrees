ACLOCAL_AMFLAGS = -I m4

test: gosdt_test
	./gosdt_test
	rm gosdt_test

# Extra files bundled in distribution
EXTRA_DIST = README.md \
	doc/dependencies.md \
	doc/development.md \
	doc/performance.md \
	doc/usage.md \
	doc/license.md \
	include \
	autobuild \
	setup.py

CPP_FLAGS = -I include -msse4.1
LD_FLAGS = -Wall -I include -pthread
LD_ADD = -ltbb -ltbbmalloc
CL_LD_FLAGS = @CL_CFLAGS@
CL_ADD = @CL_LIBS@


if OSX
LD_FLAGS += -stdlib=libc++
if INCLUDE_OPENCL
CPP_FLAGS += -DINCLUDE_OPENCL
LD_FLAGS += $(CL_LD_FLAGS)
LD_ADD += $(CL_ADD)
endif
endif

bin_PROGRAMS = gosdt
gosdt_CPPFLAGS = $(CPP_FLAGS)
gosdt_LDFLAGS = $(LD_FLAGS) 
gosdt_LDADD = $(LD_ADD)

gosdt_SOURCES = \
	src/kernels/utility.hpp \
	src/bitmask.cpp \
	src/bitmask.hpp \
	src/key.cpp \
	src/key.hpp \
	src/key_pair.cpp \
	src/key_pair.hpp \
	src/task.cpp \
	src/task.hpp \
	src/graph.cpp \
	src/graph.hpp \
	src/queue.cpp \
	src/queue.hpp \
	src/model.cpp \
	src/model.hpp \
	src/encoder.cpp \
	src/encoder.hpp \
	src/index.cpp \
	src/index.hpp \
	src/dataset.cpp \
	src/dataset.hpp \
	src/optimizer.cpp \
	src/optimizer.hpp \
	src/gosdt.cpp \
	src/gosdt.hpp \
	src/main.cpp \
	src/main.hpp

if INCLUDE_TESTS
bin_PROGRAMS += gosdt_test
gosdt_test_CPPFLAGS = $(CPP_FLAGS)
gosdt_test_LDFLAGS = $(LD_FLAGS)
gosdt_test_LDADD = $(LD_ADD)

gosdt_test_SOURCES = \
	src/kernels/utility.hpp \
	src/bitmask.cpp \
	src/bitmask.hpp \
	src/key.cpp \
	src/key.hpp \
	src/key_pair.cpp \
	src/key_pair.hpp \
	src/task.cpp \
	src/task.hpp \
	src/graph.cpp \
	src/graph.hpp \
	src/queue.cpp \
	src/queue.hpp \
	src/model.cpp \
	src/model.hpp \
	src/encoder.cpp \
	src/encoder.hpp \
	src/index.cpp \
	src/index.hpp \
	src/dataset.cpp \
	src/dataset.hpp \
	src/optimizer.cpp \
	src/optimizer.hpp \
	src/gosdt.cpp \
	src/gosdt.hpp \
	test/test_bitmask.hpp \
	test/test_key.hpp \
	test/test_minimization.hpp \
	test/test_summation.hpp \
	test/test_alternative.hpp \
	test/test_summand.hpp \
	test/test_graph.hpp \
	test/test_queue.hpp \
	test/test_dataset.hpp \
	test/test.cpp \
	test/test.hpp

endif
