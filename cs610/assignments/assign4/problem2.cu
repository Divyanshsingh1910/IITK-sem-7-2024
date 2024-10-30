#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (1 << 24);

// TODO  
__host__ void thrust_sum(const uint32_t* input, uint32_t* output) {}

// TODO
__global__ void cuda_sum() {}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                           const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between the two arrays.\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}

int main() {
  auto* h_input = new uint32_t[N];
  std::fill_n(h_input, N, 1);

  // Use Thrust code as reference
  auto* h_thrust_ref = new uint32_t[N];
  std::fill_n(h_thrust_ref, N, 0);
  // TODO: Time your code
  thrust_sum(h_input, h_thrust_ref);

  // TODO: Use a CUDA kernel, time your code
  delete[] h_thrust_ref;
  delete[] h_input;

  return EXIT_SUCCESS;
}
