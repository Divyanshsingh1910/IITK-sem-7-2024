#include <cassert>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <papi.h>


using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)
#define BLOCKi (32)
#define BLOCKj (32)
#define BLOCKk (32)

void matmul_ijk(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
    uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {

    for(int ii = 0; ii<SIZE; ii += BLOCKi){

        for(int jj = 0; jj < SIZE; jj += BLOCKj){

            for(int kk=0; kk < SIZE; kk += BLOCKk){
                for (int i = ii; i < min(SIZE, ii+BLOCKi); i++) {
                    for (int j = jj; j < min(SIZE, jj + BLOCKj); j++) {
                        uint32_t sum = 0.0;
                        for (int k = kk; k < min(SIZE, kk + BLOCKk); k++) {

                            sum += A[i * SIZE + k] * B[k * SIZE + j];

                        }
                        C[i * SIZE + j] += sum;
                    }
                }
            }
        }
    }
}

void init(uint32_t *mat, const int SIZE) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            mat[i * SIZE + j] = 1;
        }
    }
}

void handle_error(int retval) {
  cout << "PAPI error: " << retval << ": " << PAPI_strerror(retval) << "\n";
  exit(EXIT_FAILURE);
}

void print_matrix(const uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << mat[i * SIZE + j] << "\t";
    }
    cout << "\n";
  }
}

void check_result(const uint32_t *ref, const uint32_t *opt, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (ref[i * SIZE + j] != opt[i * SIZE + j]) {
        assert(false && "Diff found between sequential and blocked versions!\n");
      }
    }
  }
}

int main() {
  uint32_t *A = new uint32_t[N * N];
  uint32_t *B = new uint32_t[N * N];
  uint32_t *C_blk = new uint32_t[N * N];
 
  init(A, N);
  init(B, N);
  init(C_blk, N);

  HRTimer start = HR::now();
  HRTimer end = HR::now();

  start = HR::now();

  int retval = PAPI_hl_region_begin("PAPI-HL");
  if (retval != PAPI_OK)
    handle_error(retval);


//   matmul_ijk_blocking(A, B, C_blk, N);
    matmul_ijk(A, B, C_blk, N);

  retval = PAPI_hl_region_end("PAPI-HL");
  if (retval != PAPI_OK)
    handle_error(retval);

  cout << "\nPASSED\n";


  end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Time with blocking (us): " << duration << "\n";

  return EXIT_SUCCESS;
}
