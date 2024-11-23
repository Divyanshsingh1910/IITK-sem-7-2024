#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
//my includes 
#include<atomic>
#include<omp.h>

using std::cout;
using std::endl;
using std::string;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::filesystem::path;

static constexpr uint64_t RANDOM_SEED = 42;
static const uint32_t bucket_count = 1000;
static constexpr uint64_t MAX_OPERATIONS = 1e+15;
static const uint32_t SENTINEL_KEY = 0;
static const uint32_t SENTINEL_VALUE = 0;
static const uint32_t PROBING_RETRIES = (1 << 20);
static const uint32_t TOMBSTONE_KEY = UINT32_MAX;

const int32_t NUM_THREADS = 4;


// Function to unpack a 64-bit integer into two 32-bit integers
inline void unpackKeyValue(uint64_t value, uint32_t& key, uint32_t& val) {
    key = static_cast<uint32_t>(value >> 32);
    val = static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void create_file(path pth, const uint32_t* data, uint64_t size) {
    FILE* fptr = NULL;
    fptr = fopen(pth.string().c_str(), "wb+");
    fwrite(data, sizeof(uint32_t), size, fptr);
    fclose(fptr);
}

/** Read n integer data from file given by pth and fill in the output variable
    data */
void read_data(path pth, uint64_t n, uint32_t* data) {
    FILE* fptr = fopen(pth.string().c_str(), "rb");
    string fname = pth.string();
    if (!fptr) {
        string error_msg = "Unable to open file: " + fname;
        perror(error_msg.c_str());
    }
    int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
    if (freadStatus == 0) {
        string error_string = "Unable to read the file " + fname;
        perror(error_string.c_str());
    }
    fclose(fptr);
}

// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1e7;
/** percentage of insert queries */
uint64_t NUM_POPS = 30;
/** percentage of delete queries */
uint64_t NUM_PUSH = 70;
/** number of iterations */
uint64_t runs = 1;


// List of valid flags and description
void validFlagsDescription() {
    cout << "ops: specify total number of operations\n";
    cout << "rns: the number of iterations\n";
    cout << "add: percentage of push queries\n";
    cout << "rem: percentage of pop queries\n";
}

// Code snippet to parse command line flags and initialize the variables
int parse_args(char* arg) {
    string s = string(arg);
    string s1;
    uint64_t val;

    try {
        s1 = s.substr(0, 4);
        string s2 = s.substr(5);
        val = stol(s2);
    } catch (...) {
        cout << "Supported: " << std::endl;
        cout << "-*=[], where * is:" << std::endl;
        validFlagsDescription();
        return 1;
    }

    if (s1 == "-ops") {
        NUM_OPS = val;
    } else if (s1 == "-rns") {
        runs = val;
    } else if (s1 == "-add") {
        NUM_PUSH = val;
    } else if (s1 == "-rem") {
        NUM_POPS = val;
    } else {
        std::cout << "Unsupported flag:" << s1 << "\n";
        std::cout << "Use the below list flags:\n";
        validFlagsDescription();
        return 1;
    }
    return 0;
}


// Node structure
class Node {
public:
    uint64_t value;
    Node* next_node;
    Node(uint32_t v) : value(v), next_node(nullptr) {}
};

//stack things 
std::atomic <uint64_t> head_node ;
std::atomic <uint64_t> opr_count = 0;

__inline__ uint64_t put_count_bits(Node* _ptr){
    uint64_t ptr = (uint64_t)_ptr;
    //free the old counter bits
    ptr = (((uint64_t)ptr >> 4)<<4);

    //put the new counter bits
    return ptr | (opr_count.load() & ((1<<4)-1));
}

__inline__ Node* remove_count_bits(uint64_t _ptr){
    uint64_t ptr = _ptr;
    ptr = (ptr >> 4) << 4;
    return (Node*) ptr;
}

void push(uint32_t v) {
    Node* temp = new (std::align_val_t(16)) Node(v);
    if (!temp) {
        throw std::bad_alloc();
    }
  
    uint64_t old_head = head_node.load();
    uint64_t new_head= put_count_bits(temp);

    temp->next_node = remove_count_bits(old_head);
    
    while(!head_node.compare_exchange_strong(old_head, new_head)){
        old_head = head_node.load();
        temp->next_node = remove_count_bits(old_head);
    }
}

int pop(){

    uint64_t old_head = head_node.load();
    if(remove_count_bits(old_head) == nullptr) return -1;

    uint64_t new_head = put_count_bits(remove_count_bits(old_head)->next_node);

    while(!head_node.compare_exchange_strong(old_head, new_head)){
        old_head = head_node.load();
        new_head = put_count_bits(remove_count_bits(old_head)->next_node);
    }
    opr_count.fetch_add(1);
    uint32_t poped_value = remove_count_bits(old_head)->value;
    return poped_value;
}

void print_stack(){
    Node* start = remove_count_bits(head_node.load());
    while(start){
        cout << start->value << " ";
        start = start->next_node;
    }

    cout << endl;
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        int error = parse_args(argv[i]);
        if (error == 1) {
            cout << "Argument error, terminating run.\n";
            exit(EXIT_FAILURE);
        }
    }

    uint64_t ADD = NUM_OPS * (NUM_PUSH / 100.0);
    uint64_t REM = NUM_OPS * (NUM_POPS / 100.0);

    cout << "NUM OPS: " << NUM_OPS << " NUM_PUSH: " << ADD << " NUM_POPS: " << REM
    <<  "\n";

    path cwd = std::filesystem::current_path();
    path path_insert_values = cwd / "random_values_insert.bin";

    assert(std::filesystem::exists(path_insert_values));

    auto* values_to_insert = new uint32_t[ADD];

    read_data(path_insert_values, ADD, values_to_insert);

    std::mt19937 gen(RANDOM_SEED);
    std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

    float time = 0.0F;

    HRTimer start, end;
    omp_set_num_threads(NUM_THREADS);
    for (int j = 0; j < runs; j++) {
        start = HR::now();
        //do our job NUM_OPS times 
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static, NUM_OPS/NUM_THREADS)
        for(int i=0; i<NUM_OPS; i++){
            if(j==0 && i==0) cout << "tot_threads: " << omp_get_num_threads() << endl;
            int32_t thread_id = omp_get_thread_num();
            int32_t insert_value = 0;
            int32_t popped_value = 0;

            if(i >= ADD){
   
                popped_value = pop();
            }
            else{
                insert_value = values_to_insert[i];
       
                push(insert_value);
            }
 
        }
        end = HR::now();
        float iter_time = duration_cast<milliseconds>(end - start).count();
     
        time += iter_time;
    }

    cout << "Time taken (ms): " << time / runs
       << "\n";
    /*print_stack();*/

    return EXIT_SUCCESS;
}
