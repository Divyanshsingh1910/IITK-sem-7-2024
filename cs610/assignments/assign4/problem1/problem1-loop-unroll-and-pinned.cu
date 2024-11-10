#include <cassert>
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

const uint64_t N = (128); //64
const uint32_t THREADS_PER_BLOCKX = 4;
const uint32_t THREADS_PER_BLOCKY = 4;
const uint32_t THREADS_PER_BLOCKZ = 4;


/*
  Each thread-block will process a 32x8x8 cube
  Each thread block will have 32x2x2 threads
  Hence each thread process a 1x4x4 cube block
*/

const uint32_t BLOCK_JUMPS = 4;
const uint32_t TILE_DIMX = 32;
const uint32_t TILE_DIMY = 8;
const uint32_t TILE_DIMZ = 8;

const uint32_t low = 0;
const uint32_t high = 4;

__global__ void print_mat_host(const double* A) {
  // printf("Hellow World\n");
  for (int i = low; i < high; ++i) {
    for (int j = low; j < high; ++j) {
      for (int k = low; k < high; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }

}

// TODO: Edit the function definition as required
__global__ void kernel2(double* d_in, double* d_out) {

  __shared__ double tile[(TILE_DIMX+2)*(TILE_DIMY+2)*(TILE_DIMZ+2)];
  uint32_t PER_THREAD_CNT = BLOCK_JUMPS; // 4

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;

  uint32_t zWidth = (blockDim.x*gridDim.x)*(blockDim.y*gridDim.y*(PER_THREAD_CNT));
  uint32_t ywidth = blockDim.x*gridDim.x;

  uint32_t z_bar = z;
  uint32_t y_bar = y;

  //load to shared memory
  for(int i=0; i<PER_THREAD_CNT; i++){
    for(int j=0; j<PER_THREAD_CNT; j++){
      z = z_bar*(PER_THREAD_CNT) + i;
      y = y_bar*(PER_THREAD_CNT) + j;

      if(!(x==0 || x==N-1 || y==0 || y==N-1 || z==0 || z==N-1)){
         tile[(1+threadIdx.z*PER_THREAD_CNT-1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
              = d_in[(z-1)*zWidth + y*ywidth + x];
         tile[(1+threadIdx.z*PER_THREAD_CNT+1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
            = d_in[(z+1)*zWidth + y*ywidth + x];
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMY+2)*(TILE_DIMX+2) +
              (1+threadIdx.y*PER_THREAD_CNT-1)*(TILE_DIMX+2) + (1+threadIdx.x)]
            = d_in[z*zWidth + (y-1)*ywidth + x];
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT+1)*(TILE_DIMX+2) + (1+threadIdx.x)]
            = d_in[z*zWidth + (y+1)*ywidth + x];
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x-1)]
            = d_in[z*zWidth + y*ywidth + x-1];
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x+1)]
            = d_in[z*zWidth + y*ywidth + x+1];
      }
    }
  }

  //synchronize
  __syncthreads();

  //use shared memory now
  for(int i=0; i<PER_THREAD_CNT; i++){
    for(int j=0; j<PER_THREAD_CNT; j += 2){
      z = z_bar*(PER_THREAD_CNT) + i;
      y = y_bar*(PER_THREAD_CNT) + j;

      if(!(x==0 || x==N-1 || y==0 || y==N-1 || z==0 || z==N-1)){
         d_out[z*zWidth + y*ywidth + x] = 0.8 * (
         tile[(1+threadIdx.z*PER_THREAD_CNT-1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT+1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMY+2)*(TILE_DIMX+2) +
              (1+threadIdx.y*PER_THREAD_CNT-1)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT+1)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x-1)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x+1)]);
      }
      y = y_bar*(PER_THREAD_CNT) + j+1;

      if(!(x==0 || x==N-1 || y==0 || y==N-1 || z==0 || z==N-1)){
         d_out[z*zWidth + y*ywidth + x] = 0.8 * (
         tile[(1+threadIdx.z*PER_THREAD_CNT-1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT+1)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMY+2)*(TILE_DIMX+2) +
              (1+threadIdx.y*PER_THREAD_CNT-1)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT+1)*(TILE_DIMX+2) + (1+threadIdx.x)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x-1)]
           +
         tile[(1+threadIdx.z*PER_THREAD_CNT)*(TILE_DIMX+2)*(TILE_DIMY+2) +
              (1+threadIdx.y*PER_THREAD_CNT)*(TILE_DIMX+2) + (1+threadIdx.x+1)]);
      }
    }
  }
  //ohh my god! please make it work fine!!!
}


// TODO: Edit the function definition as required
__host__ void stencil(int N, double *in, double *out) {
  //this one is the default one which runs on cpu
  for(int i=1; i<N-1; i++) {
    for(int j=1; j<N-1; j++) {
      for(int k=1; k<N-1; k++) {
        out[i*N*N + j*N + k] = 0.8 * (
          in[(i-1)*N*N + j*N + k] +
          in[(i+1)*N*N + j*N + k] +
          in[i*N*N + (j-1)*N + k] +
          in[i*N*N + (j+1)*N + k] +
          in[i*N*N + j*N + (k-1)] +
          in[i*N*N + j*N + (k+1)]
        );
      }
    }
  }
}



__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  uint64_t nonZeroCnt = 0;
  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        if(w_ref[i + N * j + N * N * k] == 0.0)
          nonZeroCnt += 1;

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
  for (int i = low; i < high; ++i) {
    for (int j = low; j < high; ++j) {
      for (int k = low; k < high; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

void print_mat_full(const double* A) {
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

bool debug = false;
int main() {
  uint64_t SIZE = N * N * N;
  uint64_t MEMSIZE = SIZE * sizeof(double);
  //double* h_in  = (double*) malloc(MEMSIZE);
  double* h_in;
  cudaCheckError( cudaMallocHost(&h_in, MEMSIZE));
  double* h_out  = (double*) malloc(MEMSIZE);

  //random values in h_in
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      for(int k=0; k<N; k++){
        h_in[i*N*N + j*N + k] = 0.1;
      }
    }
  }

 if(debug){
   cout << "Host input data: \n";
   print_mat(h_in);
 }

 double clkbegin = rtclock();
 stencil(N, h_in, h_out);
 double clkend = rtclock();
 double cpu_time = clkend - clkbegin;
 cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

 if(debug){
   cout << "Host output data: \n";
   print_mat(h_out);
 }
 /////////////////// Kernel 2 ////////////////////////////////
 // TODO: Fill in kernel2
 double *d_k2_in, *d_k2_out;

 cudaCheckError( cudaMalloc(&d_k2_in, MEMSIZE));
 cudaCheckError( cudaMalloc(&d_k2_out, MEMSIZE));

 // dimension defintions
 dim3 dimGrid2(N/TILE_DIMX, N/TILE_DIMY, N/TILE_DIMZ);
 dim3 dimBlock2(TILE_DIMX, TILE_DIMY/BLOCK_JUMPS, TILE_DIMZ/BLOCK_JUMPS);
 //Invokding the kernel 1

 cudaEvent_t start2, end2;
 cudaCheckError( cudaEventCreate(&start2) );
 cudaCheckError( cudaEventCreate(&end2) );
 cudaCheckError( cudaEventRecord(start2) );
 cudaCheckError( cudaMemcpy(d_k2_in, h_in, MEMSIZE, cudaMemcpyHostToDevice) );

 //CUDA Kernel cal
 kernel2<<<dimGrid2, dimBlock2>>>(d_k2_in, d_k2_out);

 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess)
     printf("<kernel2> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventRecord(end2) );
 cudaCheckError( cudaEventSynchronize(end2) );
 // TODO: Adapt check_result() and invoke
 double *kernel2_out = (double*) malloc(MEMSIZE);
 cudaCheckError( cudaMemcpy(kernel2_out, d_k2_out, MEMSIZE, cudaMemcpyDeviceToHost));


 float kernel2_time;
 cudaCheckError( cudaEventElapsedTime(&kernel2_time, start2, end2) );

 if(debug){
   cout << "k2:Device Output data: \n";
   print_mat(kernel2_out);
 }
 check_result(h_out, kernel2_out, N);

 std::cout << "Kernel(Loop-unrolling + pinned) time (ms): " << kernel2_time << "\n";
 cudaCheckError( cudaEventDestroy(start2) );
 cudaCheckError( cudaEventDestroy(end2) );

 // TODO: Free memory
 cudaCheckError( cudaFree(d_k2_out));
 cudaCheckError( cudaFree(d_k2_in));
 cudaFreeHost(h_in);
 free(h_out);
 free(kernel2_out);
 return EXIT_SUCCESS;
}
