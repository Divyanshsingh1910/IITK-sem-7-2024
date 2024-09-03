CC=gcc
CXX=g++
CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
CPPFLAGS=
LIBS=-pthread

# First target is the default target
all: problem1.out problem1-opt.out

# $@ is a variable that contains the target name.
problem1.out: problem1.cpp
	$(CXX) $(CXXFLAGS) problem1.cpp -o problem1.out $(LIBS)

problem1-opt.out: problem1-opt.cpp
	$(CXX) $(CXXFLAGS) problem1-opt.cpp -o problem1-opt.out $(LIBS)
    
# The target is not a filename
clean:
	rm -f -- *.out compile_commands.json
