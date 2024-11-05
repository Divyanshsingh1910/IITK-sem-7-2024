CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_70 -lineinfo -res-usage -src-in-ptx

# First target is the default target
all: problem1-loop-unroll-and-pinned problem1-loop-unroll-and-uvm problem1-loop-unroll problem1-naive-and-opt

# $@ is a variable that contains the target name.
problem1-loop-unroll-and-pinned: problem1-loop-unroll-and-pinned.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-loop-unroll-and-uvm: problem1-loop-unroll-and-uvm.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-loop-unroll: problem1-loop-unroll.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem1-naive-and-opt: problem1-naive-and-opt.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
