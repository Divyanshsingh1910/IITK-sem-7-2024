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
#include <atomic>

using std::cerr;
using std::cout;
using std::endl;
using std::ios;

struct t_data {
    uint32_t tid;
};


/*  data structures */
std::queue<std::string> buffer;          //memory buffer


/*  variables   */
uint64_t current_buffer_size;
uint64_t MAX_BUFFER_SIZE;
uint64_t THREAD_COUNT;
uint64_t LINES_PER_THREAD;

// total lines read from the file
std::atomic< uint64_t > TOT_LINES_READ = 0;
// reader status
std::atomic<bool> READING = true;


/* global files */
std::fstream input_file;
std::fstream output_file;


/* locks    */
pthread_mutex_t input_file_lock     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mem_buffer_lock     = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t thread_wise_lock    = PTHREAD_MUTEX_INITIALIZER;


/*  cond-vars   */
pthread_cond_t cond         = PTHREAD_COND_INITIALIZER;
pthread_cond_t ack_cond     = PTHREAD_COND_INITIALIZER;
std::atomic <bool> run_consumer = false;

//undefined args error
void print_usage(char *prog_name) {
    cerr << "usage: " << prog_name <<
   " <input file> <producer count> <lines per thread> <buffer size> <output file>\n";
    exit(EXIT_FAILURE);
}


/*  thread routine  */
void *thread_runner(void *);
void *write_to_file(void *);

bool debug = false;

/*  memory buffer management    */
void write_to_buffer(std::queue<std::string>& thread_buff, uint32_t thread_id){
    bool not_done = true;

    while(not_done){
        pthread_mutex_lock(&mem_buffer_lock);
        if(debug)
                cout <<"thread #" + std::to_string(thread_id)
                + " took buffer_lock to read" << endl;

        if(buffer.size() == MAX_BUFFER_SIZE){
            if(debug)
                cout <<  "buffer-full so releasing buffer lock" << endl;
            //signal consumer
            run_consumer = true;
            if( pthread_cond_signal(&cond) != 0){
                cerr << "pthread_cond_signal() error" << endl;
                exit(4);
            }

            if(debug)
                cout <<  "Signal sent to consumer" <<
                " + waiting for consumer to clean" << endl;
            while(run_consumer)
                pthread_cond_wait(&ack_cond, &mem_buffer_lock);

           // pthread_mutex_unlock(&mem_buffer_lock);
            //continue;
        }
        else{
            while(buffer.size() < MAX_BUFFER_SIZE
                && !thread_buff.empty()){

                buffer.push(thread_buff.front());
                thread_buff.pop();
            }

            if(thread_buff.empty())
                not_done = false;
        }

        if(debug) cout << "thread going to release buffer lock" << endl;
        pthread_mutex_unlock(&mem_buffer_lock);
    }


}

int main(int argc, char* argv[])
{
    if(argc != 6){
       print_usage(argv[0]);
    }

    //args read
    std::string in_file_path    = argv[1];
    THREAD_COUNT                = strtol(argv[2], NULL, 10);
    LINES_PER_THREAD            = strtol(argv[3], NULL, 10);
    MAX_BUFFER_SIZE             = strtol(argv[4], NULL, 10);
    std::string out_file_path   = argv[5];


    //thread structs
    //additional writer thread
    pthread_t threads_worker[THREAD_COUNT + 1];

    struct t_data *thread_args =
        (struct t_data *)malloc(sizeof(struct t_data) * (THREAD_COUNT + 1));


    //opening files
    input_file.open(in_file_path.c_str(), ios::in);
    output_file.open(out_file_path.c_str(), ios::out );

    if(!input_file.is_open()){
        cerr << "Can't open the input file" << endl;
        return 1;
    }
    if(!output_file.is_open()){
        cerr << "Can't open the output file" << endl;
        return 1;
    }


    //writer thread spawning
    uint32_t idx = THREAD_COUNT;
    thread_args[idx].tid = idx; // just for sanity purpose
    pthread_create(&threads_worker[idx], nullptr, write_to_file,
            (void*)&thread_args[idx]);

    //reader thread spawning
    for(int i = 0; i < THREAD_COUNT; i++){
        thread_args[i].tid = i;
        pthread_create(&threads_worker[i], nullptr,
                thread_runner, (void*)&thread_args[i]);
    }



    //barrier for producer threads
    for(int i=0; i < THREAD_COUNT; i++)
        pthread_join(threads_worker[i], NULL);

    READING = false;    //atomic
    if(debug)
        cout <<  "Producer threads are done with their work" << endl;

    /*  It might be possible that consumer is still
        waiting for cond_var
    */
    run_consumer = true;
    pthread_cond_signal(&cond);
    //wait for consumer thread to finish
    pthread_join(threads_worker[THREAD_COUNT], NULL);


    //closing I/O
    input_file.close();
    output_file.close();

    return EXIT_SUCCESS;

}

//Reader threads
void *thread_runner( void* th_args){
    struct t_data *args = (struct t_data *)th_args;
    uint32_t thread_id = args->tid;
    std::string line;
    std::queue<std::string> thread_buff;

    //take to lock to read from file
    pthread_mutex_lock(&input_file_lock);

    //sanity check
    line = "THEAD #" + std::to_string(thread_id) + " is writing:\n";
    thread_buff.push(line);

    if(TOT_LINES_READ < LINES_PER_THREAD * (THREAD_COUNT - 1)) //atomic
        //read only L lines
    {
        uint64_t line_count = 0;

        while (std::getline( input_file, line)){
            //buffer.push(line);
            thread_buff.push(line);

            line_count++; //read lines

            if(line_count == LINES_PER_THREAD) break;
        }

        TOT_LINES_READ += LINES_PER_THREAD;     //atomic

    }
    else{
        while(!input_file.eof() && std::getline(input_file, line)){
            //buffer.push(line);
            thread_buff.push(line);
        }

    }

    pthread_mutex_unlock(&input_file_lock);

    //write thread's content into the buffer
    //other thread can't write in between
    pthread_mutex_lock(&thread_wise_lock);
    if(debug)
        cout << "Thread #" + std::to_string(thread_id) +
            "took thread_wise_lock" << endl;
    write_to_buffer(thread_buff, thread_id);
    if(debug)
        cout << "Thread #" + std::to_string(thread_id) +
            "going to release thread_wise_lock" << endl;
    pthread_mutex_unlock(&thread_wise_lock);

    pthread_exit(nullptr);
}



//Writer threads
void *write_to_file( void* th_args ){


    while(READING)
    {
        if(debug)
 cout <<  "Write to file spawned:" << endl;
        pthread_mutex_lock(&mem_buffer_lock);

        if(debug)
            cout << "consumer took buffer_lock to waite & clean" << endl;

        run_consumer = false;
        pthread_cond_signal(&ack_cond);

        while(!run_consumer){
            pthread_cond_wait(&cond, &mem_buffer_lock);
        }

        if(debug)
 cout <<  "consumer signalled to clean buffer" << endl;

        while(!buffer.empty()){
            output_file << buffer.front() << endl;
            buffer.pop();
        }

       // run_consumer = false;

        //pthread_cond_signal(&ack_cond);

        if(debug)
            cout <<  "consumer going to release the buffer lock" << endl;
        pthread_mutex_unlock(&mem_buffer_lock);
    }

    if(debug)
 cout <<  "Just reading the last of lines " << endl;
    while(!buffer.empty()){
        output_file << buffer.front() << endl;
        buffer.pop();
    }


    pthread_exit(nullptr);
}

