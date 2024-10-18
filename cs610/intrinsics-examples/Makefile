# Check vectorization support on your CPU with 'lscpu | egrep "mmx|sse[2|3]?|sse4_[1|2]?|avx2?"'

# Read assembly: Compile with -S to avoid assembling, or use "objdump -CdSs".
# Check whether arithmetic instructions contain the "ps" or "pd" suffix. For
# AVX*, check whether %ymm registers are being used.

CC=gcc
CXX=g++

# Preprocessing flags, not C++ flags
CPPFLAGS=

CFLAGS=-g std=c17 -masm=att -march=native

# List optimizations enabled with different optimization-level flags: "gcc -Q
# -O2 --help=optimizers". Earlier GCC enabled -ftree-loop-vectorize and
# -ftree-slp-vectorize at -O3. Now GCC enables these optimizations at -O2 from
# v12.1+.

# GCC uses a profitability model to decide whether to vectorize. It is
# controlled by "-fvect-cost-model=model", where the options are "very-cheap",
# "cheap", "dynamic", and "unlimited". -O2 uses the "cheap" cost model, which
# permits loop vectorization if the trip count of a scalar vectorizable loop is
# a multiple of the hardware vector length, and with no observable code size
# increasing. The default cost model for loop vectorization at -O3 uses a
# "dynamic" model with more checks to determine whether the vectorized code will
# achieve performance gains.

# -O Same as -O1
# -mavx Prefer AVX instructions compared to SSE
# -march=native	Autodetect the architecture of the build computer, works on GNU/Linux only
# -masm=att Use ATnT dialect
# -ftree-vectorize Perform vectorization, enabled from -O2
# -fopt-info-vec-missed Print information about missed optimization opportunities
# -fdump-tree-vect-all
# -fverbose-asm Put extra comments in the generated assembly to make it more readable

# Reduce noise in the generated assembly for ease of reading
# https://stackoverflow.com/questions/38552116/how-to-remove-noise-from-gcc-clang-assembly-output
# -fno-asynchronous-unwind-tables
# -fno-exceptions
# -fno-rtti

CXXFLAGS=-g -std=c++17 -masm=att -march=native

DEFS=
LIBS=

.DEFAULT_TARGET: all
# First target is the default target
#all: diff-variants struct unaligned-cost-gcc unaligned-intrinsic-cost check-cpu-capability inclusive-prefix-sum-gcc-sse4
all: unaligned-intrinsic-cost check-cpu-capability inclusive-prefix-sum-gcc-sse4

# $@ is a variable that contains the target name.
#diff-variants: diff-variants.cpp
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O -mavx -mavx2 -ftree-loop-vectorize -fno-tree-slp-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed -fdump-tree-vect-all -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -S -fverbose-asm $<
#	$(CXX) $< -o $(patsubst %.cpp,%.out,$<)

#struct: struct.cpp
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

#unaligned-cost-gcc: unaligned-cost-gcc.cpp
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

unaligned-intrinsic-cost: unaligned-intrinsic-cost.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

check-cpu-capability: check-cpu-capability.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O2 $< -o $(patsubst %.cpp,%.out,$<)

inclusive-prefix-sum-gcc-sse4: inclusive-prefix-sum-gcc-sse4.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O2 -msse4 -ftree-loop-vectorize -fno-tree-slp-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed -fdump-tree-vect-all -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti -fverbose-asm $< -o $(patsubst %.cpp,%.out,$<)

# The target is not a filename, always run the rule even if there is a file
# called "clean".
.PHONY: clean
clean:
	rm -f -- *.out *.s *.vect compile_commands.json output*
