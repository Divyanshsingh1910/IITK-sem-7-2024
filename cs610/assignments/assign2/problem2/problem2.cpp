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

/*  data structures */
std::queue<string> buffer;          //memory buffer

/*  variables   */
uint64_t current_buffer_size;
uint64_t MAX_BUFFER_SIZE;
uint64_t THREAD_COUNT;


/* locks    */
//reads from the file
pthread_mutex_t file_mutext = PTHREAD_MUTEX_INITIALIZER;

//undefined args error
void print_usage(char *prog_name) {
    cerr << "usage: " << prog_name <<
   " <input file> <producer count> <lines per thread> <buffer size> <output file>\n";
    exit(EXIT_FAILURE);
}

/*  thread routine  */
void *thread_runner(void *);

int main(int argc, chadr* argv[])
{
    if(argc != 5){
       print_usage(argv[0]);
    }

    //args read
    THREAD_COUNT = strtol(argv[2], NULL, 10);
    std::string input_file = argv[1];

    //thread spawning
    pthread_t threads_worker[THREAD_COUNT];

    struct t_data *thread_args = (struct t_data *)malloc(
            sizeof(struct t_data) * THREAD_COUNT);

    for(int i = 0; i < THREAD_COUNT; i++){
            thread_args[i].tid = i;
            pthread_create(&threads_worker[i], nullptr,
                    thread_runner, (void*)&thread_args[i]);
    }



    //barrier sync
    for(int i=0; i < THREAD_COUNT; i++)
            pthread_join(threads_worker[i], NULL);


    return EXIT_SUCCESS;

}


void *thread_runner( void* th_args){
    struct t_data *args = (struct t_data *)th_args;
    uint32_t thread_id = args->tid;

}







