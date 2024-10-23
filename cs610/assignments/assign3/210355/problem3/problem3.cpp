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
#include <omp.h>

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

std::atomic< uint64_t > TOT_LINES_READ      = 0; //total lines read from the file
std::atomic< uint64_t > TOT_THREADS_DONE    = 0; //total lines read from the file
std::atomic<bool> READING = true;           //reader status


/*  cond-vars   */
pthread_cond_t cond             = PTHREAD_COND_INITIALIZER;
pthread_cond_t ack_cond         = PTHREAD_COND_INITIALIZER;
std::atomic <bool> run_consumer = false; //variable for cond-var

/* memory-buffer lock */
pthread_mutex_t buffer_lock = PTHREAD_MUTEX_INITIALIZER;

/* global files */
std::fstream input_file;
std::fstream output_file;


//undefined args error
void print_usage(char *prog_name) {
    cerr << "usage: " << prog_name <<
   " <input file> <producer count> <lines per thread> <buffer size> <output file>\n";
    exit(EXIT_FAILURE);
}



bool debug = false; 

void do_task();

int main(int argc, char* argv[])
{

	if(argc != 6){
		print_usage(argv[0]);
	}

    std::map<std::string, std::string> arg_map;
	std::string arg, key, value;
    for(int i = 1; i < argc; i++){
        arg = argv[i];
        size_t pos = arg.find('=');
        if (pos == std::string::npos){
			print_usage(argv[0]);
		}
		key = arg.substr(0, pos);
		value = arg.substr(pos + 1);
		arg_map[key] = value;
    }

    std::string in_file_path    = arg_map["-inp"];
    THREAD_COUNT                = std::stoi(arg_map["-thr"]);
    LINES_PER_THREAD            = std::stoi(arg_map["-lns"]);
    MAX_BUFFER_SIZE             = std::stoi(arg_map["-buf"]);
    std::string out_file_path   = arg_map["-out"];


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

    //thread-jobs
    do_task();


    //closing I/O
    input_file.close();
    output_file.close();

    return EXIT_SUCCESS;

}


void do_task(){
    std::string line;
    std::queue<std::string> thread_buff;

#pragma omp parallel num_threads(THREAD_COUNT+1) private(line, thread_buff)
{
    int tid = omp_get_thread_num();
    if(tid < THREAD_COUNT){ //Reader thread
      /************************************************
                  PRODUCER THREAD START
      ************************************************/

          /* read file one-by-one by making section critical */
#pragma omp critical (read_from_file)
        {

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

        } //critical-section ends here --> thread_buff mein line read krli

        /* Write to buffer one-by-one -> another critical section*/
#pragma omp critical (write_to_buffer)
        {

            while(!thread_buff.empty())
            {
                /* Now write to buffer taking buffer lock */
                pthread_mutex_lock(&buffer_lock);

                if(buffer.size() == MAX_BUFFER_SIZE){
                    //signal consumer
                    run_consumer = true;
                    if( pthread_cond_signal(&cond) != 0){
                        cerr << "pthread_cond_signal() error" << endl;
                        exit(4);
                    }
                    //released buffer_lock -> signalled consumer


                    while(run_consumer) //waiting for consumer's signal
                        pthread_cond_wait(&ack_cond, &buffer_lock);

                }
                else{
                    while(buffer.size() < MAX_BUFFER_SIZE
                        && !thread_buff.empty()){

                        buffer.push(thread_buff.front());
                        thread_buff.pop();
                    }
                }

                pthread_mutex_unlock(&buffer_lock);
            }


            TOT_THREADS_DONE += 1;
            if(TOT_THREADS_DONE == THREAD_COUNT)
                READING = false;

            //maybe consumer is waiting
            run_consumer = true;
            pthread_cond_signal(&cond);
        }//critical section ends here


    /************************************************
                PRODUCER THREAD END
     ************************************************/
    }
    else{ //Write thread
    /************************************************
                CONSUMER THREAD START
     ************************************************/
        while(READING)
        {
            pthread_mutex_lock(&buffer_lock);
            if(debug)
                cout << "thread-id: " << tid << ", consumer acq buffer_lock" << endl;

            /*
                The reason behind putting the `ack_cond` variable above the
                `cond` variable is because when the producer receives signal
                to continue its execution I want the consumer thread to be in
                the state of acquired `buffer_lock`
            */
            run_consumer = false;
            pthread_cond_signal(&ack_cond);

            while(!run_consumer){
                pthread_cond_wait(&cond, &buffer_lock);
            }


            while(!buffer.empty()){
                output_file << buffer.front() << endl;
                buffer.pop();
            }


            pthread_mutex_unlock(&buffer_lock);
        }

        while(!buffer.empty()){
            output_file << buffer.front() << endl;
            buffer.pop();
        }

    /************************************************
                CONSUMER THREAD END
     ************************************************/
    }
} //paralle-region ends here

}
