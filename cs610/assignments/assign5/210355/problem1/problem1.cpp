#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

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

constexpr uint32_t TABLE_SIZE = 1000000;  

constexpr double MAX_LOAD = 0.8;          
constexpr uint32_t THREAD_COUNT = 8;      
constexpr uint32_t PRIME_MOD = 7 + 1e9;

struct Entry {
    std::atomic<uint32_t> key;
    std::atomic<uint32_t> value;
    std::mutex lock;

    Entry() : key(UINT32_MAX), value(0) {}
};

class ParallelHashTable {
protected:  
    std::vector<Entry> table;
    std::atomic<size_t> element_count;
    size_t capacity;

    size_t primary_hash(uint32_t key) const {
        return key % capacity;
    }

    //all the hash function with override this
    virtual size_t probe_offset(uint32_t key) const = 0;

    bool find_slot(uint32_t key, size_t& pos) const {
        size_t start_pos = primary_hash(key);
        size_t current = start_pos;
        size_t step = probe_offset(key);
        
        do {
            uint32_t existing = table[current].key.load();
            if (existing == key || existing == UINT32_MAX) {
                pos = current;
                return true;
            }
            current = (current + step) % capacity;
        } while (current != start_pos);
        
        return false;
    }

public:
    ParallelHashTable(size_t initial_size = TABLE_SIZE) 
        : table(initial_size), element_count(0), capacity(initial_size) {}

    virtual ~ParallelHashTable() = default;

    void insert(uint32_t key, uint32_t val, bool* success) {  
        if (static_cast<double>(element_count) / capacity >= MAX_LOAD) {
            *success = false;  
            return;
        }

        size_t pos;
        if (!find_slot(key, pos)) {
            *success = false;  
            return;
        }

        std::lock_guard<std::mutex> guard(table[pos].lock);
        
        if (table[pos].key.load() == UINT32_MAX) {
            table[pos].value.store(val);
            table[pos].key.store(key);
            element_count++;
            *success = true;  
        } else if (table[pos].key.load() == key) {
            *success = false;  
        } else {
            *success = false;  
        }
    }

    void remove(uint32_t key, bool* success) {  
        size_t pos;
        if (!find_slot(key, pos)) {
            *success = false;  
            return;
        }

        std::lock_guard<std::mutex> guard(table[pos].lock);
        
        if (table[pos].key.load() == key) {
            table[pos].key.store(UINT32_MAX);
            element_count--;
            *success = true;  
        } else {
            *success = false;  
        }
    }

    void lookup(uint32_t key, int64_t* result) {  
        size_t pos;
        if (!find_slot(key, pos)) {
            *result = -1;  
            return;
        }

        std::lock_guard<std::mutex> guard(table[pos].lock);
        
        if (table[pos].key.load() == key) {
            *result = table[pos].value.load();  
        } else {
            *result = -1;  
        }
    }

    void clear() {
        for (auto& entry : table) {
            std::lock_guard<std::mutex> guard(entry.lock);
            entry.key.store(UINT32_MAX);
            entry.value.store(0);
        }
        element_count.store(0);
    }

    size_t size() const {
        return element_count.load();
    }
};

class LinearProbingHashTable : public ParallelHashTable {
protected:
    size_t probe_offset(uint32_t) const override {
        return 1;
    }
public:
    using ParallelHashTable::ParallelHashTable;
};

class QuadraticProbingHashTable : public ParallelHashTable {
protected:
    size_t probe_offset(uint32_t key) const override {
        return (key * key + 1) % capacity;  
    }
public:
    using ParallelHashTable::ParallelHashTable;
};

class PrimeStepHashTable : public ParallelHashTable {
protected:
    size_t probe_offset(uint32_t key) const override {
        return PRIME_MOD - (key % PRIME_MOD);
    }
public:
    using ParallelHashTable::ParallelHashTable;
};

template<typename T>
void batch_insert(T& table, size_t count, const std::vector<std::pair<uint32_t, uint32_t>>& pairs, 
                 std::vector<bool>& results) {
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for (size_t i = 0; i < count; i++) {
        bool success;  
        table.insert(pairs[i].first, pairs[i].second, &success);  
        results[i] = success;  
    }
}

template<typename T>
void batch_delete(T& table, size_t count, const std::vector<uint32_t>& keys,
                 std::vector<bool>& results) {
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for (size_t i = 0; i < count; i++) {
        bool success;  
        table.remove(keys[i], &success);  
        results[i] = success;  
    }
}

template<typename T>
void batch_lookup(T& table, size_t count, const std::vector<uint32_t>& keys,
                 std::vector<int64_t>& results) {
    #pragma omp parallel for num_threads(THREAD_COUNT)
    for (size_t i = 0; i < count; i++) {
        int64_t result;  
        table.lookup(keys[i], &result);  
        results[i] = result;  
    }
}

std::vector<uint32_t> read_binary_file(const std::filesystem::path& path, size_t count) {
    std::vector<uint32_t> data(count);
    std::ifstream file;  
    file.open(path, std::ios::binary);  
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint32_t));
    return data;
}

void run_performance_test(size_t op_count = 1000000, double insert_ratio = 0.6,
                        double delete_ratio = 0.35) {
    size_t insert_count = op_count * insert_ratio;
    size_t delete_count = op_count * delete_ratio;
    size_t lookup_count = op_count - (insert_count + delete_count);

    auto insert_keys = read_binary_file("random_keys_insert.bin", insert_count);
    auto insert_vals = read_binary_file("random_values_insert.bin", insert_count);
    auto delete_keys = read_binary_file("random_keys_delete.bin", delete_count);
    auto lookup_keys = read_binary_file("random_keys_search.bin", lookup_count);

    std::vector<std::pair<uint32_t, uint32_t>> insert_pairs(insert_count);
    for (size_t i = 0; i < insert_count; i++) {
        insert_pairs[i] = {insert_keys[i], insert_vals[i]};
    }

    
    std::vector<std::unique_ptr<ParallelHashTable>> tables;
    tables.emplace_back(new LinearProbingHashTable());
    tables.emplace_back(new QuadraticProbingHashTable());
    tables.emplace_back(new PrimeStepHashTable());

    const char* strategy_names[] = {
        "Linear Probing",
        "Quadratic Probing",
        "Prime Step Probing"
    };

    for (size_t i = 0; i < tables.size(); i++) {
        std::cout << "\nTesting " << strategy_names[i] << std::endl;
        auto& table = *tables[i];
        
        std::vector<bool> insert_results(insert_count);
        std::vector<bool> delete_results(delete_count);
        std::vector<int64_t> lookup_results(lookup_count);

        HRTimer start, end;
        start = HR::now();
        batch_insert(table, insert_count, insert_pairs, insert_results);
        end = HR::now();
        float insert_time = duration_cast<milliseconds>(end- start).count();

        start = HR::now();
        batch_delete(table, delete_count, delete_keys, delete_results);
        end = HR::now();
        float delete_time = duration_cast<milliseconds>(end- start).count();

        start = HR::now();
        batch_lookup(table, lookup_count, lookup_keys, lookup_results);
        end = HR::now();
        float lookup_time = duration_cast<milliseconds>(end- start).count();

        std::cout << "Insert time: " << insert_time << "ms\n"
                  << "Delete time: " << delete_time << "ms\n"
                  << "Lookup time: " << lookup_time << "ms\n";

        table.clear();
    }
}



void print_usage() {
    std::cout << "Usage: ./program [-ops=N] [-add=X] [-rem=Y] [-rns=Z]\n"
              << "  -ops=N : Total number of operations (default: 1e6)\n"
              << "  -add=X : Percentage of insert operations (default: 60)\n"
              << "  -rem=Y : Percentage of delete operations (default: 35)\n"
              << "  -rns=Z : Number of test runs (default: 2)\n";
}

int parse_argument(const std::string& arg, uint64_t& target) {
    try {
        size_t pos = arg.find('=');
        if (pos == std::string::npos) return 1;
        target = std::stoull(arg.substr(pos + 1));
        return 0;
    } catch (...) {
        return 1;
    }
}
// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1e8;
/** percentage of insert queries */
uint64_t INSERT = 60;
/** percentage of delete queries */
uint64_t DELETE = 30;
/** number of iterations */
uint64_t runs = 2;

int main(int argc, char* argv[]) {
    uint64_t num_ops =  NUM_OPS;    
    uint64_t insert_pct = INSERT;      
    uint64_t delete_pct = DELETE;      
    uint64_t num_runs = 2;         

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        
        int error = 0;
        if (arg.substr(0, 5) == "-ops=") {
            error = parse_argument(arg, num_ops);
        } else if (arg.substr(0, 5) == "-add=") {
            error = parse_argument(arg, insert_pct);
        } else if (arg.substr(0, 5) == "-rem=") {
            error = parse_argument(arg, delete_pct);
        } else if (arg.substr(0, 5) == "-rns=") {
            error = parse_argument(arg, num_runs);
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            print_usage();
            return 1;
        }

        if (error) {
            std::cout << "Error parsing argument: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    if (insert_pct + delete_pct > 100) {
        std::cout << "Error: Insert and delete percentages sum to more than 100%" << std::endl;
        return 1;
    }

    std::cout << "Running tests with configuration:\n"
              << "  Total operations: " << num_ops << "\n"
              << "  Insert operations: " << insert_pct << "%\n"
              << "  Delete operations: " << delete_pct << "%\n"
              << "  Lookup operations: " << (100 - insert_pct - delete_pct) << "%\n"
              << "  Number of runs: " << num_runs << "\n\n";

    try {
        run_performance_test(num_ops, insert_pct/100.0, delete_pct/100.0);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
