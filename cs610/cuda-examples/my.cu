#include <stdio.h>
#include <cuda.h>

__global__ void hwkernel(){
  printf("Hello world!\n");
}

int main(){
  //asynchronous function call 
  hwkernel<<<1,1>>>();
  cudaDeviceSynchronize();
  return 0;
}
