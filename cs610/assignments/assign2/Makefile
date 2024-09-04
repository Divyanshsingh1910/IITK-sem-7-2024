CC=gcc
CXX=g++
CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
CPPFLAGS=
LIBS=-pthread

# First target is the default target
all: problem1

# $@ is a variable that contains the target name.
problem1: problem1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<) $(LIBS)

# The target is not a filename
clean:
	rm -f -- *.out compile_commands.json
