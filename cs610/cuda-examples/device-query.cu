// https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/

#include <cuda.h>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

// https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
    case 2: { // Fermi
      if (devProp.minor == 1) {
        cores = mp * 48;
      } else {
        cores = mp * 32;
      }
      break;
    }
    case 3: { // Kepler
      cores = mp * 192;
      break;
    }
    case 5: { // Maxwell
      cores = mp * 128;
      break;
    }
    case 6: { // Pascal
      if (devProp.minor == 1) {
        cores = mp * 128;
      } else if (devProp.minor == 0) {
        cores = mp * 64;
      } else {
        cout << "Unknown device type\n";
      }
      break;
    }
    case 7: { // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) {
        cores = mp * 64;
      } else {
        cout << "Unknown device type\n";
      }
      break;
    }
    default: {
      cout << "Unknown device type\n";
      break;
    }
  }
  return cores;
}

int main() {
  cudaError_t err;

  int count;
  err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    cerr << cudaGetErrorString(err) << " " << __FILE__ << " " << __LINE__
         << endl;
  }
  cout << "Number of CUDA-enabled devices: " << count << "\n";

  cudaDeviceProp Props;
  for (int i = 0; i < count; i++) {
    err = cudaGetDeviceProperties(&Props, i);
    if (err != cudaSuccess) {
      cerr << cudaGetErrorString(err) << " " << __FILE__ << " " << __LINE__
           << endl;
    }

    cout << "Device number: " << i << "\n"
         << "Device name: " << Props.name << "\n" // GeForce GTX 1080 Ti
         << "\tIntegrated or discrete GPU? "
         << ((Props.integrated == 1) ? "integrated" : "discrete") << "\n"
         << "\tClock rate: " << Props.clockRate / 1024 << " MHz\n"
         << "\tCompute capability: " << Props.major << "." << Props.minor
         << "\n"
         // Can overlap memory copy and kernel execution
         << "\tNumber of asynchronous engines: " << Props.asyncEngineCount
         << "\n"
         << "\tConcurrent kernel execution: " << Props.concurrentKernels << "\n"
         << "\tConcurrent copy and execution: " << Props.deviceOverlap << "\n\n"
         << "\tCan map host memory: " << Props.canMapHostMemory << "\n\n"

         << "\tNumber of SMs: " << Props.multiProcessorCount << "\n"
         << "\tTotal number of CUDA cores: " << getSPcores(Props) << "\n"
         << "\tMax threads per SM: " << Props.maxThreadsPerMultiProcessor
         << "\n"
         << "\tMax threads per block: " << Props.maxThreadsPerBlock
         << "\n" // 1024
         << "\tWarp size: " << Props.warpSize
         << "\n"
         // [2^31, 2^16, 2^16] for compute_3 onward
         << "\tMax grid size (i.e., max number of blocks): ["
         << Props.maxGridSize[0] << "," << Props.maxGridSize[1] << ","
         << Props.maxGridSize[2] << "]\n"
         << "\tMax block dimension: [" << Props.maxThreadsDim[0] << ","
         << Props.maxThreadsDim[1] << "," << Props.maxThreadsDim[2] << "]\n\n"

         << "\tTotal global memory: " << Props.totalGlobalMem / (1024 * 1024)
         << " MB\n"
         << "\tShared memory per SM: "
         << Props.sharedMemPerMultiprocessor / 1024 << " KB\n"
         << "\t32-bit registers per SM: " << Props.regsPerMultiprocessor << "\n"
         << "\tShared mem per block: " << Props.sharedMemPerBlock / 1024
         << " KB\n" // 48K
         << "\tRegisters per block: " << Props.regsPerBlock << "\n"
         << "\tTotal const mem: " << Props.totalConstMem / 1024 << " KB\n"
         << "\tL2 cache size: " << Props.l2CacheSize / 1024 << " KB\n\n";
  }

  return EXIT_SUCCESS;
}
