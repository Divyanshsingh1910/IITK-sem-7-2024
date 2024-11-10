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

const uint64_t N = (1 << 9);
const uint32_t KERNEL_DIM = 3;
const uint32_t low = 5;
const uint32_t high = 8;
const uint32_t THREADS_PER_BLOCKX = 32;
const uint32_t THREADS_PER_BLOCKY = 32;
const uint32_t THREADS_PER_BLOCKZ = 4;
const uint32_t TILE_DIM = 32;
const uint32_t TILE_DIMX = 32;
const uint32_t TILE_DIMY = 16; //16
const uint32_t TILE_DIMZ = 16; //16
const uint32_t ELEMENTS_PER_THREAD = 1; // 4

__global__ void print_mat_host3D(const float* A){
  // printf("Hellow World\n");
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
  const int32_t NUM_Y = TILE_DIM/blockDim.y;

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  for(int y_dir=0; y_dir<NUM_Y; y_dir++){
    for(int i=0; i<KERNEL_DIM; i++){
      for(int j=0; j<KERNEL_DIM; j++){
        tile[(threadIdx.y+y_dir+i)*TILE_N + threadIdx.x+j] =
        ((y+y_dir-d+i<0 || y+y_dir-d+i>=N || x-d+j<0 || x-d+j>=N)? 0:inp[(y+y_dir-d+i)*N + (x-d+j)]);
      }
    }
  }
  //tile[(threadIdx.y+d)*TILE_N + threadIdx.x+d] = inp[y*N + x];
  __syncthreads();

  float sum = 0.0;

  for(int y_dir=0; y_dir<NUM_Y; y_dir++){
    for(int i=0; i<KERNEL_DIM; i++){
      for(int j=0; j<KERNEL_DIM; j++){
        sum += kernel[i*N + j]* tile[(threadIdx.y+y_dir+i)*TILE_N + threadIdx.x+j];
      }
    }
    out[(y+y_dir)*N + x] = sum;
  }
}


// TODO: Edit the function definition as required
__global__ void kernel3D(float* inp, float* out, float* kernel) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;

  float sum = 0.0;
  int32_t d = KERNEL_DIM/2;

  for(int k=0; k<KERNEL_DIM; k++){
    for(int i=0; i<KERNEL_DIM; i++){
      for(int j=0; j<KERNEL_DIM; j++){
        sum += kernel[k*KERNEL_DIM*KERNEL_DIM + i*KERNEL_DIM + j] *
            ((z-d+k<0 || z-d+k>=N || y-d+i<0 || y-d+i>=N || x-d+j<0 || x-d+j>=N)?
             0:inp[(z-d+k)*N*N + (y-d+i)*N + (x-d+j)]);
      }
    }
  }
  out[z*N*N + y*N + x] = sum;
}

__global__ void kernel3D_opt(float* inp, float* out, float* kernel) {

  const int32_t d = KERNEL_DIM/2;
  __shared__ float tile[(TILE_DIMX+2*d)*(TILE_DIMY+2*d)*(TILE_DIMZ+2*d)];
  const int32_t TILE_NX = TILE_DIMX+2*d;
  const int32_t TILE_NY = TILE_DIMY+2*d;
  const int32_t NUM_Y = TILE_DIMY/blockDim.y;
  const int32_t NUM_Z = TILE_DIMZ/blockDim.z;

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;


      for(int k=0; k<KERNEL_DIM; k++){
        for(int i=0; i<KERNEL_DIM; i++){
          for(int j=0; j<KERNEL_DIM; j++){
            tile[(threadIdx.z*(NUM_Z)+k)*TILE_NX*TILE_NY + (threadIdx.y*(NUM_Y)+i)*TILE_NX + threadIdx.x+j] =
                ((z*(NUM_Z)-d+k<0 || z*(NUM_Z)-d+k>=N || y*(NUM_Y)-d+i<0 || y*(NUM_Y)-d+i>=N || x-d+j<0 || x-d+j>=N)?
                 0:inp[(z*(NUM_Z)-d+k)*N*N + (y*(NUM_Y)-d+i)*N + (x-d+j)]);
          }
        }
      }

  //tile[(threadIdx.y+d)*TILE_N + threadIdx.x+d] = inp[y*N + x];
  __syncthreads();

  float sum = 0.0;
  //if(!(TILE_DIMZ/blockDim.z == 4)) printf("Yeh toh kuch galat ho chuka hai\n");


      sum = 0.0;
      for(int k=0; k<KERNEL_DIM; k++){
        for(int i=0; i<KERNEL_DIM; i++){
          for(int j=0; j<KERNEL_DIM; j++){
            sum += kernel[k*KERNEL_DIM*KERNEL_DIM + i*KERNEL_DIM + j] *
              tile[(threadIdx.z*(NUM_Z)+k)*TILE_NX*TILE_NY + (threadIdx.y*(NUM_Y)+i)*TILE_NX + threadIdx.x+j] ;
          }
        }
      }
      out[(z*(NUM_Z))*N*N + (y*(NUM_Y))*N + x] = sum;

  //printf("(%d,%d,%d)\n",x,y,z);

}

__global__ void kernel3D_opt_old(float* inp, float* out, float* kernel) {

  const int32_t d = KERNEL_DIM/2;
  __shared__ float tile[(TILE_DIMX+2*d)*(TILE_DIMY+2*d)*(TILE_DIMZ+2*d)];
  const int32_t TILE_NX = TILE_DIMX+2*d;
  const int32_t TILE_NY = TILE_DIMY+2*d;
  const int32_t NUM_Y = TILE_DIMY/blockDim.y;
  const int32_t NUM_Z = TILE_DIMZ/blockDim.z;

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int z = blockIdx.z*blockDim.z + threadIdx.z;

  for(int z_dir=0; z_dir<NUM_Z; z_dir++){
    for(int y_dir=0; y_dir<NUM_Y; y_dir++){

      for(int k=0; k<KERNEL_DIM; k++){
        for(int i=0; i<KERNEL_DIM; i++){
          for(int j=0; j<KERNEL_DIM; j++){
            tile[(threadIdx.z*(NUM_Z)+z_dir+k)*TILE_NX*TILE_NY + (threadIdx.y*(NUM_Y)+y_dir+i)*TILE_NX + threadIdx.x+j] =
                ((z*(NUM_Z)+z_dir-d+k<0 || z*(NUM_Z)+z_dir-d+k>=N || y*(NUM_Y)+y_dir-d+i<0 || y*(NUM_Y)+y_dir-d+i>=N || x-d+j<0 || x-d+j>=N)?
                 0:inp[(z*(NUM_Z)+z_dir-d+k)*N*N + (y*(NUM_Y)+y_dir-d+i)*N + (x-d+j)]);
          }
        }
      }

    }
  }
  //tile[(threadIdx.y+d)*TILE_N + threadIdx.x+d] = inp[y*N + x];
  __syncthreads();

  float sum = 0.0;
  //if(!(TILE_DIMZ/blockDim.z == 4)) printf("Yeh toh kuch galat ho chuka hai\n");

  for(int z_dir=0; z_dir<NUM_Z; z_dir++){
    for(int y_dir=0; y_dir<NUM_Y; y_dir++){

      sum = 0.0;
      for(int k=0; k<KERNEL_DIM; k++){
        for(int i=0; i<KERNEL_DIM; i++){
          for(int j=0; j<KERNEL_DIM; j++){
            sum += kernel[k*KERNEL_DIM*KERNEL_DIM + i*KERNEL_DIM + j] *
              tile[(threadIdx.z*(NUM_Z)+z_dir+k)*TILE_NX*TILE_NY + (threadIdx.y*(NUM_Y)+y_dir+i)*TILE_NX + threadIdx.x+j] ;
          }
        }
      }
      out[(z*(NUM_Z)+z_dir)*N*N + (y*(NUM_Y)+y_dir)*N + x] = sum;

    }
  }
  //printf("(%d,%d,%d)\n",x,y,z);

}

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


__host__ void Conv3D_host(int32_t N, float* inp, float* out, float* kernel) {

  float sum;
  int32_t d = KERNEL_DIM/2;

  for(int z=0; z<N; z++){
    for(int y=0; y<N; y++){
      for(int x=0; x<N; x++){
        //kernel computation around element x,y
        sum = 0.0;
        for(int k=0; k<KERNEL_DIM; k++){
          for(int i=0; i<KERNEL_DIM; i++){
            for(int j=0; j<KERNEL_DIM; j++){
              sum += kernel[k*KERNEL_DIM*KERNEL_DIM + i*KERNEL_DIM + j] *
                ((z-d+k<0 || z-d+k>=N || y-d+i<0 || y-d+i>=N || x-d+j<0 || x-d+j>=N)?
                 0:inp[(z-d+k)*N*N + (y-d+i)*N + (x-d+j)]);
            }
          }
        }
        out[z*N*N + y*N + x] = sum;
      }
    }
  }
}

__host__ void check_result_3D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;
  int32_t nonzero_cnt = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        if(w_ref[i + N * j + N * N * k] )
          nonzero_cnt += 1;
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

bool debug =false;
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

 dim3 dimGrid_opt(N/TILE_DIM, N/(TILE_DIM),1);
 dim3 dimBlock_opt(TILE_DIM/1, TILE_DIM/4, 1); //1 tha y mein
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
 std::cout << "kernel2D(Optimized) time (ms): " << kernel_time << "\n";

 cudaCheckError( cudaEventDestroy(start));
 cudaCheckError( cudaEventDestroy(end));

 //return EXIT_SUCCESS;
 ///////////////////////////////////// 3*D /////////////////////////////////////
 printf("\n===================\n\n");
 SIZE = N*N*N;
 MEMSIZE = SIZE * sizeof(float);
 KERNEL_MEMSIZE = KERNEL_DIM*KERNEL_DIM*KERNEL_DIM*sizeof(float);


 float* h3d_in = (float*) malloc(MEMSIZE);
 float* h3d_out = (float*) malloc(MEMSIZE);
 float* kernel_mat3d = (float*) malloc(KERNEL_MEMSIZE);

 //initialization of data
 for(int k=0; k<N; k++){
   for(int i=0; i<N; i++){
     for(int j=0; j<N; j++){
       h3d_in[k*N*N + i*N + j] = (i+j+k)*1.0;
     }
   }
 }

 //initialization of kernel
 for(int k=0; k<KERNEL_DIM; k++){
   for(int i=0; i<KERNEL_DIM; i++){
     for(int j=0; j<KERNEL_DIM; j++){
       kernel_mat3d[k*KERNEL_DIM*KERNEL_DIM + i*KERNEL_DIM + j] = (i+j+k)*1.0;
     }
   }
 }

 if(debug){
   cout << "Host input data: \n";
   print_mat3D(h3d_in);
 }

 if(debug){
   cout << "kernel mat input data: \n";
   print_mat3D(kernel_mat3d);
 }
 clkbegin = rtclock();
 Conv3D_host(N, h3d_in, h3d_out,kernel_mat3d);
 clkend = rtclock();
 cpu_time = clkend - clkbegin;
 cout << "3D Convolution time on CPU: " << cpu_time * 1000 << " msec" << endl;

 if(debug){
   cout << "Host output data: \n";
   print_mat3D(h3d_out);
 }
 // TODO: Fill in kernel3D

 float *d3d_in, *d3d_out,*d3d_kernel;
 cudaCheckError( cudaMalloc(&d3d_in, MEMSIZE));
 cudaCheckError( cudaMalloc(&d3d_out, MEMSIZE));
 cudaCheckError( cudaMalloc(&d3d_kernel,KERNEL_MEMSIZE));

 cudaCheckError( cudaMemcpy(d3d_in, h3d_in, MEMSIZE, cudaMemcpyHostToDevice));
 cudaCheckError( cudaMemcpy(d3d_kernel, kernel_mat3d,KERNEL_MEMSIZE, cudaMemcpyHostToDevice));


 if(debug){
   cout << "Device input data: \n";
   print_mat_host3D<<<1,1>>>(d3d_in);
 }
 cudaCheckError( cudaDeviceSynchronize());
 err = cudaGetLastError();
 if (err != cudaSuccess)
     printf("<print_mat_host> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventCreate(&start) );
 cudaCheckError( cudaEventCreate(&end) );
 cudaCheckError( cudaEventRecord(start) );
 // dimension defintions
 dim3 dimGrid3d(N/(THREADS_PER_BLOCKX/8), N/(THREADS_PER_BLOCKY/8),N/(THREADS_PER_BLOCKZ));
 dim3 dimBlock3d(THREADS_PER_BLOCKX/8, THREADS_PER_BLOCKY/8, THREADS_PER_BLOCKZ);

 //CUDA Kernel cal
 kernel3D<<<dimGrid3d, dimBlock3d>>>(d3d_in, d3d_out, d3d_kernel);

 err = cudaGetLastError();
 if (err != cudaSuccess)
     printf("<kernel3D> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventRecord(end) );
 // TODO: Adapt check_result() and invoke
 float*kernel13d_out = (float*) malloc(MEMSIZE);

 cudaCheckError( cudaMemcpy(kernel13d_out, d3d_out, MEMSIZE, cudaMemcpyDeviceToHost));

 cudaCheckError( cudaEventSynchronize(end) );

 cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );

 if(debug){
   cout << "Device Output data: \n";
   print_mat3D(kernel13d_out);
 }
 check_result_3D(h3d_out, kernel13d_out);
 std::cout << "Kernel3D time (ms): " << kernel_time << "\n";

 /////////////// Optimized 3D Conv /////////////////////
 cudaCheckError( cudaEventRecord(start));

 dim3 dimGrid3d_opt(N/THREADS_PER_BLOCKX, N/TILE_DIMY,N/TILE_DIMZ);
 dim3 dimBlock3d_opt(THREADS_PER_BLOCKX, TILE_DIMY/16, TILE_DIMZ/16);

 kernel3D_opt<<<dimGrid3d_opt, dimBlock3d_opt>>>(d3d_in, d3d_out, d3d_kernel);
 cudaDeviceSynchronize();
 err = cudaGetLastError();
 if (err != cudaSuccess)
     printf("<kernel3D_opt> Error: %s\n", cudaGetErrorString(err));

 cudaCheckError( cudaEventRecord(end) );
 // TODO: Adapt check_result() and invoke
 //optimzed wala
 cudaCheckError( cudaMemcpy(kernel13d_out, d3d_out, MEMSIZE, cudaMemcpyDeviceToHost));

 cudaCheckError( cudaEventSynchronize(end) );
 cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );

 if(debug){
   cout << "Optimized Device Output data: \n";
   print_mat3D(kernel13d_out);
 }
 check_result_3D(h3d_out, kernel13d_out);
 std::cout << "kernel3D(Optimized) time (ms): " << kernel_time << "\n";

 cudaCheckError( cudaEventDestroy(start));
 cudaCheckError( cudaEventDestroy(end));
 // TODO: Free memory

 return EXIT_SUCCESS;
}
