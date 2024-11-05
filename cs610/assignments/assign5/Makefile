CXX=g++
CPPFLAGS=
CXXFLAGS=-O3 -std=c++17

# First target is the default target
all: problem1

# $@ is a variable that contains the target name.
problem1: problem1.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

problem2: problem2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
