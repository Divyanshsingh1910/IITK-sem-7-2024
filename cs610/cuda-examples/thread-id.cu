#include <cuda.h>
#include <iostream>
#include <stdlib.h>

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(uint32_t) * (ARRAY_SIZE))

using std::cout;

__global__ void thread_details(uint32_t* const block, uint32_t* const thread,
                               uint32_t* const warp,
                               uint32_t* const calc_thread) {
  const uint32_t thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  block[thread_idx] = blockIdx.x;
  thread[thread_idx] = threadIdx.x;
  warp[thread_idx] = threadIdx.x / warpSize;
  calc_thread[thread_idx] = thread_idx;
}

int main() {
  uint32_t cpu_block[ARRAY_SIZE];
  uint32_t cpu_thread[ARRAY_SIZE];
  uint32_t cpu_warp[ARRAY_SIZE];
  uint32_t cpu_calc_thread[ARRAY_SIZE];

  uint32_t* gpu_block = nullptr;
  uint32_t* gpu_thread = nullptr;
  uint32_t* gpu_warp = nullptr;
  uint32_t* gpu_calc_thread = nullptr;

  cudaMalloc((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

  const uint32_t num_blocks = 2;
  const uint32_t num_threads = 64;
  thread_details<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp,
                                              gpu_calc_thread);

  // cudaMemcpy() is a blocking call, so we do not need cudaDeviceSynchronize()
  cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,
             cudaMemcpyDeviceToHost);

  cudaFree(gpu_block);
  cudaFree(gpu_thread);
  cudaFree(gpu_warp);
  cudaFree(gpu_calc_thread);

  for (uint16_t i = 0; i < ARRAY_SIZE; i++) {
    cout << "Block Index: " << cpu_block[i] << ", Warp Num: " << cpu_warp[i]
         << ", Thread Index: " << cpu_thread[i]
         << " <---> Calculated Thread Id: " << cpu_calc_thread[i] << "\n";
  }

  return EXIT_SUCCESS;
}
