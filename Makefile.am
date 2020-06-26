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
LD_ADD = -ltbb -ltbbmalloc -lgmp
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
	src/types.hpp \
	src/integrity_violation.hpp \
	src/optimizer/diagnosis/false_convergence.hpp \
	src/optimizer/diagnosis/non_convergence.hpp \
	src/optimizer/diagnosis/trace.hpp \
	src/optimizer/diagnosis/tree.hpp \
	src/optimizer/dispatch/dispatch.hpp \
	src/optimizer/extraction/models.hpp \
	src/bitmask.cpp \
	src/bitmask.hpp \
	src/configuration.cpp \
	src/configuration.hpp \
	src/task.cpp \
	src/task.hpp \
	src/message.cpp \
	src/message.hpp \
	src/tile.cpp \
	src/tile.hpp \
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
	src/state.cpp \
	src/state.hpp \
	src/local_state.cpp \
	src/local_state.hpp \
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
	src/types.hpp \
	src/integrity_violation.hpp \
	src/optimizer/diagnosis/false_convergence.hpp \
	src/optimizer/diagnosis/non_convergence.hpp \
	src/optimizer/diagnosis/trace.hpp \
	src/optimizer/diagnosis/tree.hpp \
	src/optimizer/dispatch/dispatch.hpp \
	src/optimizer/extraction/models.hpp \
	src/bitmask.cpp \
	src/bitmask.hpp \
	src/configuration.cpp \
	src/configuration.hpp \
	src/task.cpp \
	src/task.hpp \
	src/message.cpp \
	src/message.hpp \
	src/tile.cpp \
	src/tile.hpp \
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
	src/state.cpp \
	src/state.hpp \
	src/local_state.cpp \
	src/local_state.hpp \
	src/gosdt.cpp \
	src/gosdt.hpp \
	test/test_bitmask.hpp \
	test/test_index.hpp \
	test/test_queue.hpp \
	test/test_consistency.hpp \
	test/test.cpp \
	test/test.hpp

endif
