#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

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

const uint64_t N = (1 << 6);
const uint32_t KERNEL_DIM = 3;
const uint32_t low = 3;
const uint32_t high = 7;
const uint32_t THREADS_PER_BLOCKX = 32;
const uint32_t THREADS_PER_BLOCKY = 32;
const uint32_t TILE_DIM = 32;

__global__ void print_mat_host2D(const float* A){
  // printf("Hellow World\n");
  for (int i = low; i < high; ++i) {
    for (int j = low; j < high; ++j) {
      printf("%f,", A[i * N + j]);
    }
    printf("      ");
  }
  printf("\n");
  
}
// TODO: Edit the function definition as required
__global__ void kernel2D(float* inp, float* out, float* kernel) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  float sum = 0.0;
  int32_t d = KERNEL_DIM/2;

  for(int i=0; i<KERNEL_DIM; i++){
    for(int j=0; j<KERNEL_DIM; j++){
      sum += kernel[i*KERNEL_DIM + j] * 
        ((y-d+i<0 || y-d+i>=N || x-d+j<0 || x-d+j>=N)? 0:inp[(y-d+i)*N + (x-d+j)]);
    }
  } 

  out[y*N + x] = sum;
}


__global__ void kernel2D_opt(float* inp, float* out, float* kernel) {

  const int32_t d = KERNEL_DIM/2;
  __shared__ float tile[(TILE_DIM+2*d)*(TILE_DIM+2*d)];
  const int32_t TILE_N = TILE_DIM+2*d;

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  for(int i=0; i<KERNEL_DIM; i++){
    for(int j=0; j<KERNEL_DIM; j++){
      tile[(threadIdx.y+i)*TILE_N + threadIdx.x+j] = 
      ((y-d+i<0 || y-d+i>=N || x-d+j<0 || x-d+j>=N)? 0:inp[(y-d+i)*N + (x-d+j)]);
    }
  }  
  //tile[(threadIdx.y+d)*TILE_N + threadIdx.x+d] = inp[y*N + x];
  __syncthreads();

  float sum = 0.0;

  for(int i=0; i<KERNEL_DIM; i++){
    for(int j=0; j<KERNEL_DIM; j++){
      sum += kernel[i*N + j]* tile[(threadIdx.y+i)*TILE_N + threadIdx.x+j];
    }
  } 

  out[y*N + x] = sum;
}


// TODO: Edit the function definition as required
__global__ void kernel3D() {}


__host__ void Conv2D_host(int32_t N, float* inp, float* out, float* kernel) {

  float sum;
  int32_t d = KERNEL_DIM/2;

  for(int y=0; y<N; y++){
    for(int x=0; x<N; x++){
      //kernel computation around element x,y
      sum = 0.0;
      for(int i=0; i<KERNEL_DIM; i++){
        for(int j=0; j<KERNEL_DIM; j++){
          sum += kernel[i*KERNEL_DIM + j] * 
            ((y-d+i<0 || y-d+i>=N || x-d+j<0 || x-d+j>=N)? 0:inp[(y-d+i)*N + (x-d+j)]);
        }
      } 
      out[y*N + x] = sum;
    }
  }

}


__host__ void Conv3D_host() {}

__host__ void check_result_3D(const float*** w_ref, const float*** w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
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
         << "; 3DMax Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ void check_result_2D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      double this_diff =
          w_ref[i + N * j] - w_opt[i + N * j];
      if (std::fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff) {
          maxdiff = this_diff;
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; 2DMax Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print2D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "n";
  }
}

void print_mat3D(const float* A) {
  for (int i = low; i < high; ++i) {
    for (int j = low; j < high; ++j) {
      for (int k = low; k < high; ++k) {
        printf("%f,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

void print_mat2D(const float *A) {
  for (int i = low; i < high; ++i) {
    for (int j = low; j < high; ++j) {
        printf("%f,", A[i * N + j]);
    }
    printf("      ");
  }
  printf("\n");
}

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "n";
    }
    cout << "n";
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
  uint64_t SIZE;
  uint64_t MEMSIZE;
  uint64_t KERNEL_MEMSIZE;
 ///////////////////////////////////// 2*D /////////////////////////////////////
 SIZE = N*N;
 MEMSIZE = SIZE * sizeof(float);
 KERNEL_MEMSIZE = KERNEL_DIM*KERNEL_DIM*sizeof(float);
 
 // Host device calculation

 float* h2d_in = (float*) malloc(MEMSIZE);
 float* h2d_out = (float*) malloc(MEMSIZE);
 float* kernel_mat2d = (float*) malloc(KERNEL_MEMSIZE);

 //initialization of data 
 for(int i=0; i<N; i++){
   for(int j=0; j<N; j++){
     h2d_in[i*N + j] = (i+j)*1.0;
   }
 }

 //initialization of kernel  
 for(int i=0; i<KERNEL_DIM; i++){
   for(int j=0; j<KERNEL_DIM; j++){
     kernel_mat2d[i*N + j] = (i+j)*1.0;
   }
 }

 if(debug){
   cout << "Host input data: \n";
   print_mat2D(h2d_in);
 }

 double clkbegin = rtclock();
 Conv2D_host(N, h2d_in, h2d_out,kernel_mat2d);
 double clkend = rtclock();
 double cpu_time = clkend - clkbegin;
 cout << "2D Convolution time on CPU: " << cpu_time * 1000 << " msec" << endl;

 if(debug){
   cout << "Host output data: \n";
   print_mat2D(h2d_out);
 }

 // TODO: Fill in Kernel2D
 float *d2d_in, *d2d_out,*d2d_kernel;
 cudaCheckError( cudaMalloc(&d2d_in, MEMSIZE));
 cudaCheckError( cudaMalloc(&d2d_out, MEMSIZE));
 cudaCheckError( cudaMalloc(&d2d_kernel,KERNEL_MEMSIZE));

 cudaCheckError( cudaMemcpy(d2d_in, h2d_in, MEMSIZE, cudaMemcpyHostToDevice));
 cudaCheckError( cudaMemcpy(d2d_kernel, kernel_mat2d,KERNEL_MEMSIZE, cudaMemcpyHostToDevice));


 if(debug){
   cout << "Device input data: \n";
   print_mat_host2D<<<1,1>>>(d2d_in);
 }
 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) 
     printf("<print_mat_host> Error: %s\n", cudaGetErrorString(err));

 cudaEvent_t start, end;
 cudaCheckError( cudaEventCreate(&start) );
 cudaCheckError( cudaEventCreate(&end) );
 cudaCheckError( cudaEventRecord(start) );
 // dimension defintions 
 dim3 dimGrid(N/THREADS_PER_BLOCKX, N/THREADS_PER_BLOCKY,1);
 dim3 dimBlock(THREADS_PER_BLOCKX, THREADS_PER_BLOCKY, 1);

 //CUDA Kernel cal
 kernel2D<<<dimGrid, dimBlock>>>(d2d_in, d2d_out, d2d_kernel);

 err = cudaGetLastError();
 if (err != cudaSuccess) 
     printf("<kernel2D> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventRecord(end) );
 // TODO: Adapt check_result() and invoke
 float*kernel12d_out = (float*) malloc(MEMSIZE);

 cudaCheckError( cudaMemcpy(kernel12d_out, d2d_out, MEMSIZE, cudaMemcpyDeviceToHost));

 cudaCheckError( cudaEventSynchronize(end) );

 float kernel_time;
 cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );

 if(debug){
   cout << "Device Output data: \n";
   print_mat2D(kernel12d_out);
 }
 check_result_2D(h2d_out, kernel12d_out);
 std::cout << "Kernel2D time (ms): " << kernel_time << "\n";

 //cudaCheckError( cudaEventDestroy(start));
 //cudaCheckError( cudaEventDestroy(end));


 ///////// Optimized 2D conv ////////////// 
 cudaCheckError( cudaEventRecord(start));

 kernel2D_opt<<<dimGrid, dimBlock>>>(d2d_in, d2d_out, d2d_kernel);
 err = cudaGetLastError();
 if (err != cudaSuccess) 
     printf("<kernel2D_opt> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventRecord(end) );
 // TODO: Adapt check_result() and invoke
 //optimzed wala 
 cudaCheckError( cudaMemcpy(kernel12d_out, d2d_out, MEMSIZE, cudaMemcpyDeviceToHost));

 cudaCheckError( cudaEventSynchronize(end) );
 cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );

 if(debug){
   cout << "Optimized Device Output data: \n";
   print_mat2D(kernel12d_out);
 }
 check_result_2D(h2d_out, kernel12d_out);
 std::cout << "kernel2D_opt time (ms): " << kernel_time << "\n";

 cudaCheckError( cudaEventDestroy(start));
 cudaCheckError( cudaEventDestroy(end));

 return EXIT_SUCCESS;
 ///////////////////////////////////// 3*D /////////////////////////////////////
 // TODO: Fill in kernel3D
 // TODO: Adapt check_result() and invoke
 std::cout << "Kernel3D time (ms): " << kernel_time << "\n";

 // TODO: Free memory

 return EXIT_SUCCESS;
}
