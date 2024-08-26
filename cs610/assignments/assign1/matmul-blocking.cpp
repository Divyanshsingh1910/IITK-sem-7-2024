#include <cassert>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)
// #define BLOCK (16)
#define BLOCKi (4)
#define BLOCKj (4)
#define BLOCKk (4)

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
  uint32_t *C_seq = new uint32_t[N * N];

  init(A, N);
  init(B, N);
  init(C_seq, N);

  HRTimer start = HR::now();
  matmul_ijk(A, B, C_seq, N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Time without blocking (us): " << duration << "\n";

  uint32_t *C_blk = new uint32_t[N * N];
  init(C_blk, N);

  start = HR::now();
  matmul_ijk_blocking(A, B, C_blk, N);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Time with blocking (us): " << duration << "\n";

  check_result(C_seq, C_blk, N);

  return EXIT_SUCCESS;
}
