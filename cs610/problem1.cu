#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

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

const uint64_t N = (1 << 8);
const uint64_t SIZE_IN_BYTES = N * N * N * sizeof(double);

#define TILE_WIDTH 8
#define BLOCK_WIDTH (TILE_WIDTH + 2)

// check if i, j, k are in [L, R)
__device__ bool is_valid(const int i, const int j, const int k, const int L, const int R){
	return L <= i && i < R && L <= j && j < R && L <= k && k < R;
}

__global__ void kernel1(const double *in, double *out, const int N) {
		int i = blockIdx.z * blockDim.z + threadIdx.z; // Depth index
		int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index
		int k = blockIdx.x * blockDim.x + threadIdx.x; // Column index

		if(i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1){
				out[i * N * N + j * N + k]  = 0.8 * (  in[(i - 1) * N * N + j * N + k]
																						 + in[(i + 1) * N * N + j * N + k]
																						 + in[i * N * N + (j - 1) * N + k]
																						 + in[i * N * N + (j + 1) * N + k]
																						 + in[i * N * N + j * N + (k - 1)]
																						 + in[i * N * N + j * N + (k + 1)]
																						);
		}
}

__global__ void kernel2(const double *in, double *out, const int N){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int dep_o = blockIdx.z * TILE_WIDTH + tz; // Output depth
	int row_o = blockIdx.y * TILE_WIDTH + ty; // Output row
	int col_o = blockIdx.x * TILE_WIDTH + tx; // Output column

	int dep_i = dep_o - 1; // Input depth
	int row_i = row_o - 1; // Input row
	int col_i = col_o - 1; // Input column

	__shared__ double temp[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

	if(is_valid(dep_i, row_i, col_i, 0, N)){
			temp[tz][ty][tx] = in[dep_i * N * N + row_i * N + col_i];
	}
	__syncthreads();

	if(is_valid(tz, ty, tx, 0, TILE_WIDTH) && is_valid(dep_o, row_o, col_o, 1, N - 1)){
			out[dep_o * N * N + row_o * N + col_o] = 0.8 * ( temp[tz + 1][ty][tx + 1]
																										 + temp[tz + 1][ty + 1][tx]
																										 + temp[tz + 1][ty + 2][tx + 1]
																										 + temp[tz + 1][ty + 1][tx + 2]
																										 + temp[tz][ty + 1][tx + 1]
																										 + temp[tz + 2][ty + 1][tx + 1]
																										);
	}
}

__global__ void kernel3(const double *in, double *out, const int N){}

__host__ void stencil(const double *in, double *out, const int N) {
		for(int i = 1; i < N - 1; i++){
				for(int j = 1; j < N - 1; j++){
						for(int k = 1; k < N - 1; k++){
								out[i * N * N + j * N + k]  = 0.8 * (  in[(i - 1) * N * N + j * N + k]
																										 + in[(i + 1) * N * N + j * N + k]
																										 + in[i * N * N + (j - 1) * N + k]
																										 + in[i * N * N + (j + 1) * N + k]
																										 + in[i * N * N + j * N + (k - 1)]
																										 + in[i * N * N + j * N + (k + 1)]
																										);
						}
				}
		}
}

__host__ void check_result(const double *w_ref, const double *w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            // w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
            w_ref[i * N * N + j * N + k] - w_opt[i * N * N + j * N + k];
        if (std::fabs(this_diff) > THRESHOLD) {
          // printf("Diff found at: i = %lu, j = %lu, k = %lu\n", i, j, k);
					numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(const double *A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix(double *in){
	for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
					for(int k = 0; k < N; k++){
							// in[i * N * N + j * N + k] = (float)rand();
							in[i * N * N + j * N + k] = 1.0;
					}
			}
	}
}

void naive(double *in, double *out_ref){
	// Allocating memory on host (CPU)
	double *out = (double*)malloc(SIZE_IN_BYTES);

	// Allocating memory on device (GPU)
	double *d_in = NULL, *d_out = NULL;
	cudaCheckError(cudaMalloc(&d_in, SIZE_IN_BYTES));
	cudaCheckError(cudaMalloc(&d_out, SIZE_IN_BYTES));

	// Copying data from host (CPU) to device (GPU)
	cudaCheckError(cudaMemcpy(d_in, in, SIZE_IN_BYTES, cudaMemcpyHostToDevice));

	// Setting execution environment
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y, N / threadsPerBlock.z);

  cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// Launching kernel
	kernel1<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0); // cudaEventRecord is asynchronous
	// Copying data from device (GPU) back to host (CPU)
	cudaCheckError(cudaMemcpy(out, d_out, SIZE_IN_BYTES, cudaMemcpyDeviceToHost));
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	check_result(out_ref, out, N);
  std::cout << "Naive version time (ms): " << kernel_time << "\n";

	// Freeing memory on host (CPU)
	free(out);

	// Freeing memory on device (GPU)
	cudaFree(d_in);
	cudaFree(d_out);
}

void shared_memory(double *in, double *out_ref){
	// Allocating memory on host (CPU)
	double *out = (double*)malloc(SIZE_IN_BYTES);

	// Allocating memory on device (GPU)
	double *d_in = NULL, *d_out = NULL;
	cudaCheckError(cudaMalloc(&d_in, SIZE_IN_BYTES));
	cudaCheckError(cudaMalloc(&d_out, SIZE_IN_BYTES));

	// Copying data from host (CPU) to device (GPU)
	cudaCheckError(cudaMemcpy(d_in, in, SIZE_IN_BYTES, cudaMemcpyHostToDevice));

	// Setting execution environment
	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N / TILE_WIDTH, N / TILE_WIDTH);

  cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// Launching kernel
	kernel2<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0); // cudaEventRecord is asynchronous
	// Copying data from device (GPU) back to host (CPU)
	cudaCheckError(cudaMemcpy(out, d_out, SIZE_IN_BYTES, cudaMemcpyDeviceToHost));
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	check_result(out_ref, out, N);
  std::cout << "Shared memory time (ms): " << kernel_time << "\n";

	// Freeing memory on host (CPU)
	free(out);

	// Freeing memory on device (GPU)
	cudaFree(d_in);
	cudaFree(d_out);
}

void pinned_memory(){
	double *in = NULL, *out_ref = NULL, *out = NULL;

	// Allocating memory
	cudaHostAlloc(&in, SIZE_IN_BYTES, cudaHostAllocDefault);
	cudaHostAlloc(&out_ref, SIZE_IN_BYTES, cudaHostAllocDefault);
	cudaHostAlloc(&out, SIZE_IN_BYTES, cudaHostAllocDefault);

	init_matrix(in);

	double clkbegin = rtclock();
  stencil(in, out_ref, N);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

	double *d_in = NULL, *d_out = NULL;

	cudaMalloc(&d_in, SIZE_IN_BYTES);
	cudaMalloc(&d_out, SIZE_IN_BYTES);

	cudaMemcpy(d_in, in, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	// Setting execution environment
	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N / TILE_WIDTH, N / TILE_WIDTH);

  cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// Launching kernel
	kernel2<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0); // cudaEventRecord is asynchronous
	cudaMemcpy(out, d_out, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	check_result(out_ref, out, N);
  std::cout << "Pinned memory (ms): " << kernel_time << "\n";

	// Freeing memory
	cudaFree(in);
	cudaFree(out_ref);
	cudaFree(out);
	cudaFree(d_in);
	cudaFree(d_out);
}

void unified_virtual_memory(){
	double *in = NULL, *out_ref = NULL, *out = NULL;

	// Allocating memory
	cudaMallocManaged(&in, SIZE_IN_BYTES);
	out_ref = (double*)malloc(SIZE_IN_BYTES);
	cudaMallocManaged(&out, SIZE_IN_BYTES);

	init_matrix(in);

	double clkbegin = rtclock();
  stencil(in, out_ref, N);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

	// Setting execution environment
	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N / TILE_WIDTH, N / TILE_WIDTH);

  cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	// Launching kernel
	kernel2<<<numBlocks, threadsPerBlock>>>(in, out, N);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0); // cudaEventRecord is asynchronous
	cudaDeviceSynchronize();
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	check_result(out_ref, out, N);
  std::cout << "Unified virtual memory (ms): " << kernel_time << "\n";

	// Freeing memory
	free(out_ref);
	cudaFree(in);
	cudaFree(out);
}

int main() {
  srand(time(NULL)); // Seed the random number generator with the current time

	// Allocating memory on host (CPU)
	double *in      = (double*)malloc(SIZE_IN_BYTES);
	double *out_ref = (double*)malloc(SIZE_IN_BYTES);

	// Initializing array with random values
	init_matrix(in);

	double clkbegin = rtclock();
  stencil(in, out_ref, N);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

	naive(in, out_ref);
	shared_memory(in, out_ref);
 	//unified_virtual_memory();
	//pinned_memory();

	return EXIT_SUCCESS;
}
