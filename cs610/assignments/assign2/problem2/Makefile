CXX=g++
CXXFLAGS=-O3 -std=c++17 -mavx -mavx2
LIBS=-pthread

# First target is the default target
all: problem2.out 

# $@ is a variable that contains the target name.
problem2.out: problem2.cpp
	$(CXX) $(CXXFLAGS) problem2.cpp -o problem2.out $(LIBS)
    
# The target is not a filename
clean:
	rm -f -- *.out compile_commands.json
