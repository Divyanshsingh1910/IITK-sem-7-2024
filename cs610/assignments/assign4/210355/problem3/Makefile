CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -arch=sm_86 -lineinfo -res-usage -src-in-ptx -ccbin=g++-9

# First target is the default target
all: problem3-naive problem3-opt problem3-opt-and-uvm

# $@ is a variable that contains the target name.
problem3-naive: problem3-naive.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-opt: problem3-opt.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem3-opt-and-uvm: problem3-opt-and-uvm.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
