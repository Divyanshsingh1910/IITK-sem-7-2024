CXX=g++
CPPFLAGS=
CXXFLAGS=-O3 -std=c++17

# First target is the default target
all: problem2

problem2: problem2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp $< -o $(patsubst %.cpp,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
