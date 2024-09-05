CC=gcc
CXX=g++
CFLAGS=-O3 -std=c17 -mavx -mavx2
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
CPPFLAGS=
LIBS=-pthread

# First target is the default target
all: problem1.out problem1-wo-ts.out problem1-wo-fs.out problem1-wo-ts_n_fs.out

# $@ is a variable that contains the target name.
problem1.out: problem1.cpp
	$(CXX) $(CXXFLAGS) problem1.cpp -o problem1.out $(LIBS)

problem1-wo-ts_n_fs.out: problem1-wo-ts_n_fs.cpp
	$(CXX) $(CXXFLAGS) problem1-wo-ts_n_fs.cpp -o problem1-wo-ts_n_fs.out $(LIBS)

problem1-wo-ts.out: problem1-wo-ts.cpp
	$(CXX) $(CXXFLAGS) problem1-wo-ts.cpp -o problem1-wo-ts.out $(LIBS)

problem1-wo-fs.out: problem1-wo-fs.cpp
	$(CXX) $(CXXFLAGS) problem1-wo-fs.cpp -o problem1-wo-fs.out $(LIBS)


# The target is not a filename
clean:
	rm -f -- *.out compile_commands.json
