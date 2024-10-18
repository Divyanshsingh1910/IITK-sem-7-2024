CC=gcc
CXX=g++

# Preprocessing flags, not C++ flags
CPPFLAGS=

# Reduce noise in the generated assembly for ease of reading
CFLAGS=-g -std=c17 -masm=att -msse4 -mavx2 -march=native -O2 -fopenmp -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions

CXXFLAGS=-g -std=c++17 -masm=att -march=native -O2 -fopenmp -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti

DEFS=
LIBS=-pthread

.DEFAULT_TARGET: all
# First target is the default target
all: problem1 problem2 problem3 problem4-v0

# $@ is a variable that contains the target name.
problem1: problem1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

problem2: problem2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

problem3: problem3.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<) $(LIBS)

problem4-v0: problem4-v0.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $(patsubst %.c,%.out,$<)

# The target is not a filename, always run the rule even if there is a file
# called "clean".
.PHONY: clean
clean:
	rm -f -- *.out *.s *.vect compile_commands.json output*
