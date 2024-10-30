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

const uint64_t N = (64);
const uint32_t THREADS_PER_BLOCK = 32;

// TODO: Edit the function definition as required
__global__ void kernel1(const double* d_in, double* d_out) {
  
  int x = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
  int y = blockIdx.y*THREADS_PER_BLOCK + threadIdx.y;
  int z = blockIdx.z*THREADS_PER_BLOCK + threadIdx.z;

  uint32_t zWidth = (THREADS_PER_BLOCK * blockDim.x) * (THREADS_PER_BLOCK * blockDim.y);
  uint32_t ywidth = THREADS_PER_BLOCK * blockDim.x;

  d_out[z*zWidth + y*ywidth + x] = 0.8 * (d_in[(z-1)*zWidth + y*ywidth + x] + 
      d_in[(z+1)*zWidth + y*ywidth + x] + d_in[z*zWidth + (y-1)*ywidth + x] + 
      d_in[z*zWidth + (y+1)*ywidth + x] + d_in[z*zWidth + y*ywidth + x+1] + 
      d_in[z*zWidth + y*ywidth + x-1]); 
}

// TODO: Edit the function definition as required
__global__ void kernel2() {}



// TODO: Edit the function definition as required
__host__ void stencil(int N, double *in, double *out) {
  //this one is the default one which runs on cpu 
  for(int i=1; i<N-1; i++) {
    for(int j=1; j<N-1; j++) {
      for(int k=1; k<N-1; k++) {
        out[i*N*N + j*N + k] = 0.8 * (
                in[(i-1)*N*N + j*N     + k    ] +
                in[(i+1)*N*N + j*N     + k    ] +
                in[i*N*N     + (j-1)*N + k    ] +
                in[i*N*N     + (j+1)*N + k    ] +
                in[i*N*N     + j*N     + (k-1)] +
                in[i*N*N     + j*N     + (k+1)] );
      }
    }
  }
}



__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
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



void print_mat(const double* A) {
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

int main() {
  uint64_t SIZE = N * N * N;
  uint64_t MEMSIZE = SIZE * sizeof(double);
  double* h_in  = (double*) malloc(MEMSIZE); 
  double* h_out  = (double*) malloc(MEMSIZE); 

  //random values in h_in 
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      for(int k=0; k<N; k++){
        h_in[i*N*N + j*N + k] = (i+j+k)*(0.1);
      }
    }
  }

  double clkbegin = rtclock();
  stencil(N, h_in, h_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;


  /////////////////// Kernel 1 ////////////////////////////////
  //cudaError_t status;
  cudaEvent_t start, end;
  
  // TODO: Fill in kernel1
  double *d_in, *d_out; //device data 
  //cuda memory allocation
  cudaCheckError( cudaMalloc(&d_in, MEMSIZE));
  cudaCheckError( cudaMalloc(&d_out, MEMSIZE));

  //copy the input memory 
  cudaCheckError( cudaMemcpy(d_in, h_in, MEMSIZE, cudaMemcpyHostToDevice) );

  //Invokding the kernel 1 
  cudaCheckError( cudaEventCreate(&start) );
  cudaCheckError( cudaEventCreate(&end) );
  cudaCheckError( cudaEventRecord(start, 0) );
  // dimension defintions 
  uint32_t grid  = N/THREADS_PER_BLOCK;
  dim3 dimGrid(grid, grid, grid);
  dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  //CUDA Kernel cal
  kernel1<<<dimGrid, dimBlock>>>(d_in, d_out);

  cudaCheckError( cudaEventRecord(end, 0) );
  cudaCheckError( cudaEventSynchronize(end) );

  float kernel_time;
  cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );
  
  // TODO: Adapt check_result() and invoke
  double *kernel1_out = (double*) malloc(MEMSIZE);
  cudaCheckError( cudaMemcpy(kernel1_out, d_out, MEMSIZE, cudaMemcpyDeviceToHost));

  check_result(h_out, kernel1_out, N);

  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
  cudaCheckError( cudaEventDestroy(start) );
  cudaCheckError( cudaEventDestroy(end) );


  return EXIT_SUCCESS;
  
  /////////////////// Kernel 2 ////////////////////////////////
  // TODO: Fill in kernel2
  // TODO: Adapt check_result() and invoke
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  // TODO: Free memory

  return EXIT_SUCCESS;
}
