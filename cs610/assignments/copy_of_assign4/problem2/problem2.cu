#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

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

const uint64_t N = (1 << 9);

// TODO  
__host__ void thrust_sum(const uint32_t* input, uint32_t* output) {
  //copy to device and take thrust-prefix sum 
  uint32_t* d_input;
  cudaCheckError( cudaMalloc(&d_input, N*sizeof(uint32_t)));
  cudaCheckError( cudaMemcpy(d_input, input, N*sizeof(uint32_t), cudaMemcpyHostToDevice));

  thrust::device_ptr<uint32_t> device_input(d_input);
  thrust::exclusive_scan(device_input, device_input+N, device_input);

  //copy back to host 
  cudaCheckError( cudaMemcpy(output, d_input, N*sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__global__ void prefix_kernel_old(uint32_t* input, uint32_t* output, uint32_t N){
  __shared__ uint32_t shared_mem[1024];

  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if(thread_id>=N){
    return;
  }

  int index_out = 0, index_in = 1;

  shared_mem[index_out*N + thread_id] = (thread_id>0)?input[thread_id-1]:0;

  __syncthreads();

  for(int jump = 1; jump < N; jump *= 2){
    index_out = 1 - index_out;
    index_in = 1 - index_out;

    if(thread_id >= jump){
      shared_mem[index_out*N + thread_id] += shared_mem[index_in*N + thread_id - jump];
    }
    else{
      shared_mem[index_out*N + thread_id] = shared_mem[index_in*N + thread_id];
    }

    __syncthreads();
  }

  output[thread_id] = shared_mem[index_out*N + thread_id];
}
__global__ void prefix_kernel(uint32_t* input, uint32_t* output, uint32_t N){
  __shared__ uint32_t shared_mem[1024];

  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if(thread_id>=N){
    return;
  }

  int index_out = 0, index_in = 1;

  shared_mem[threadIdx.x] = (thread_id>0)?input[thread_id-1]:0;

  __syncthreads();

  for(int jump = 1; jump <= blockDim.x; jump *= 2){
    index_out = 1 - index_out;
    index_in = 1 - index_out;

    if(threadIdx.x >= jump){
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x - jump];
    }
    else{
      shared_mem[threadIdx.x] = shared_mem[threadIdx.x];
    }

    __syncthreads();
  }

  output[thread_id] = shared_mem[threadIdx.x];
}

// TODO
__host__ void cuda_sum(const uint32_t* input, uint32_t* output, uint32_t N) {

  //allocate memory on cuda 
  uint32_t* d_input, *d_output;
  uint32_t first_value = d_input[0];

  cudaCheckError( cudaMalloc(&d_input, N*sizeof(uint32_t)));
  cudaCheckError( cudaMalloc(&d_output, N*sizeof(uint32_t)));
  cudaCheckError( cudaMemcpy(d_input, input, N*sizeof(uint32_t), cudaMemcpyHostToDevice));

  dim3 grids((N/1024) + 1,1,1);
  dim3 blocks(1024,1,1);

  prefix_kernel<<<grids, blocks>>>(d_input, d_output, N);

  cudaCheckError( cudaMemcpy(output, d_output, N*sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

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
  auto* h_thrust_ref = new uint32_t[N];
  std::fill_n(h_thrust_ref, N, 0);
  auto* h_cuda = new uint32_t[N];
  std::fill_n(h_cuda, N, 0);

  // Use Thrust code as reference
  // TODO: Time your code
  cudaEvent_t start, end;
  cudaCheckError( cudaEventCreate(&start) );
  cudaCheckError( cudaEventCreate(&end) );
  cudaCheckError( cudaEventRecord(start) );

  thrust_sum(h_input, h_thrust_ref);

  cudaCheckError( cudaEventRecord(end) );
  cudaCheckError( cudaEventSynchronize(end) );
  float thrust_time;
  cudaCheckError( cudaEventElapsedTime(&thrust_time, start, end) );

  // TODO: Use a CUDA kernel, time your code
  cudaCheckError( cudaEventCreate(&start) );
  cudaCheckError( cudaEventCreate(&end) );
  cudaCheckError( cudaEventRecord(start) );

  cuda_sum(h_input, h_cuda, N);

  cudaCheckError( cudaEventRecord(end) );
  cudaCheckError( cudaEventSynchronize(end) );
  float my_time;
  cudaCheckError( cudaEventElapsedTime(&my_time, start, end) );

  cout << "Thrust prefix sum time(ms): " << thrust_time*10 << "\n";
  cout << "my prefix sum time(ms): " << my_time*10 << "\n";
  check_result(h_thrust_ref, h_cuda, N);
  // Free memory
  //delete[] h_thrust_ref;
  //delete[] h_input;
  //delete[] h_cuda;

  return EXIT_SUCCESS;
}
