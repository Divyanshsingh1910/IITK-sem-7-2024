//library imports


#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <unistd.h>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;


void print_usage(char *prog_name) {
    cerr << "usage: " << prog_name <<
   " <input file> <producer count> <lines per thread> <buffer size> <output file>\n";
    exit(EXIT_FAILURE);
}

int thread_count = 0;

int main(int argc, char* argv[])
{
    if(argc != 5){
       print_usage(argv[0]);
    }

    thread_count = strtol(argv[2], NULL, 10);

    std::string input = argv[1];


}
