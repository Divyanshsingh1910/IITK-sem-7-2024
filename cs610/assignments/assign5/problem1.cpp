#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

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

typedef struct {
  uint32_t key;
  uint32_t value;
} KeyValue;

// Pack key-value into a 64-bit integer
inline uint64_t packKeyValue(uint32_t key, uint32_t val) {
  return (static_cast<uint64_t>(key) << 32) |
         (static_cast<uint32_t>(val) & 0xFFFFFFFF);
}

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
uint64_t NUM_OPS = 1e8;
/** percentage of insert queries */
uint64_t INSERT = 100;
/** percentage of delete queries */
uint64_t DELETE = 0;
/** number of iterations */
uint64_t runs = 2;

// List of valid flags and description
void validFlagsDescription() {
  cout << "ops: specify total number of operations\n";
  cout << "rns: the number of iterations\n";
  cout << "add: percentage of insert queries\n";
  cout << "rem: percentage of delete queries\n";
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
    INSERT = val;
  } else if (s1 == "-rem") {
    DELETE = val;
  } else {
    std::cout << "Unsupported flag:" << s1 << "\n";
    std::cout << "Use the below list flags:\n";
    validFlagsDescription();
    return 1;
  }
  return 0;
}

void batch_insert() {}

void batch_delete() {}

void batch_search() {}

int main(int argc, char* argv[]) {
  for (int i = 1; i < argc; i++) {
    int error = parse_args(argv[i]);
    if (error == 1) {
      cout << "Argument error, terminating run.\n";
      exit(EXIT_FAILURE);
    }
  }

  uint64_t ADD = NUM_OPS * (INSERT / 100.0);
  uint64_t REM = NUM_OPS * (DELETE / 100.0);
  uint64_t FIND = NUM_OPS - (ADD + REM);

  cout << "NUM OPS: " << NUM_OPS << " ADD: " << ADD << " REM: " << REM
       << " FIND: " << FIND << "\n";

  auto* h_kvs_insert = new KeyValue[ADD];
  memset(h_kvs_insert, 0, sizeof(KeyValue) * ADD);
  auto* h_keys_del = new uint32_t[REM];
  memset(h_keys_del, 0, sizeof(uint32_t) * REM);
  auto* h_keys_lookup = new uint32_t[FIND];
  memset(h_keys_lookup, 0, sizeof(uint32_t) * FIND);

  // Use shared files filled with random numbers
  path cwd = std::filesystem::current_path();
  path path_insert_keys = cwd / "random_keys_insert.bin";
  path path_insert_values = cwd / "random_values_insert.bin";
  path path_delete = cwd / "random_keys_delete.bin";
  path path_search = cwd / "random_keys_search.bin";

  assert(std::filesystem::exists(path_insert_keys));
  assert(std::filesystem::exists(path_insert_values));
  assert(std::filesystem::exists(path_delete));
  assert(std::filesystem::exists(path_search));

  // Read data from file
  auto* tmp_keys_insert = new uint32_t[ADD];
  read_data(path_insert_keys, ADD, tmp_keys_insert);
  auto* tmp_values_insert = new uint32_t[ADD];
  read_data(path_insert_values, ADD, tmp_values_insert);
  for (int i = 0; i < ADD; i++) {
    h_kvs_insert[i].key = tmp_keys_insert[i];
    h_kvs_insert[i].value = tmp_values_insert[i];
  }
  delete[] tmp_keys_insert;
  delete[] tmp_values_insert;

  auto* tmp_keys_delete = new uint32_t[REM];
  read_data(path_delete, REM, tmp_keys_delete);
  for (int i = 0; i < REM; i++) {
    h_keys_del[i] = tmp_keys_delete[i];
  }
  delete[] tmp_keys_delete;

  auto* tmp_keys_search = new uint32_t[FIND];
  read_data(path_search, FIND, tmp_keys_search);
  for (int i = 0; i < FIND; i++) {
    h_keys_lookup[i] = tmp_keys_search[i];
  }
  delete[] tmp_keys_search;

  // Max limit of the uint32_t: 4,294,967,295
  std::mt19937 gen(RANDOM_SEED);
  std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

  float total_insert_time = 0.0F;
  float total_delete_time = 0.0F;
  float total_search_time = 0.0F;

  HRTimer start, end;
  for (int i = 0; i < runs; i++) {
    start = HR::now();
    batch_insert();
    end = HR::now();
    float iter_insert_time = duration_cast<milliseconds>(end - start).count();

    start = HR::now();
    batch_delete();
    end = HR::now();
    float iter_delete_time = duration_cast<milliseconds>(end - start).count();

    start = HR::now();
    batch_search();
    end = HR::now();
    float iter_search_time = duration_cast<milliseconds>(end - start).count();

    total_insert_time += iter_insert_time;
    total_delete_time += iter_delete_time;
    total_search_time += iter_search_time;
  }

  cout << "Time taken by insert kernel (ms): " << total_insert_time / runs
       << "\nTime taken by delete kernel (ms): " << total_delete_time / runs
       << "\nTime taken by search kernel (ms): " << total_search_time / runs
       << "\n";

  return EXIT_SUCCESS;
}
