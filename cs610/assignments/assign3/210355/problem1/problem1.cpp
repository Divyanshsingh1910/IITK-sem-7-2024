#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include<x86intrin.h>

//custom lib
//#include "mylib.h"
bool debug = 0;

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (1024)

void matmul_seq(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}


void matmul_sse4(float** A, float** B, float** C) {
    int stride = 4;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            /*
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
            */
            __m128 row = _mm_load_ps1(&A[i][k]); //all a[i,k]
            for(int j = 0; j<N; j += stride){
                __m128 col = _mm_load_ps(&B[k][j]);  //j,j+1,j+2,j+3

                __m128 temp = _mm_load_ps(&C[i][j]);
                temp = _mm_add_ps(temp, _mm_mul_ps(row, col));
                _mm_store_ps(&C[i][j],temp);
            }
        }
    }
}


void matmul_sse4_unaligned(float** A, float** B, float** C) {
    int stride = 4;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            /*
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
            */
            __m128 row = _mm_load_ps1(&A[i][k]); //all a[i,k]
            for(int j = 0; j<N; j += stride){
                __m128 col = _mm_loadu_ps(&B[k][j]);  //j,j+1,j+2,j+3

                __m128 temp = _mm_loadu_ps(&C[i][j]);
                temp = _mm_add_ps(temp, _mm_mul_ps(row, col));
                _mm_storeu_ps(&C[i][j],temp);
            }
        }
    }
}

void matmul_avx2(float** A, float** B, float** C) {
    int stride = 8;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            /*
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
            */
            __m256 row = _mm256_set1_ps(A[i][k]); //8 copies of a[i,k]
            for(int j = 0; j<N; j += stride){
                //cout << "j: " << j << endl;
                __m256 col = _mm256_load_ps(&B[k][j]);  //j,j+1,j+2,..,j+7
                //cout << "col loaded" << endl;

                __m256 temp = _mm256_load_ps(&C[i][j]);
                //cout << "temp loaded" << endl;
                temp = _mm256_add_ps(temp, _mm256_mul_ps(row, col));
                _mm256_store_ps(&C[i][j],temp);
            }
        }
    }
}

void matmul_avx2_unaligned(float** A, float** B, float** C) {
    int stride = 8;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            /*
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
            */
            __m256 row = _mm256_set1_ps(A[i][k]); //8 copies of a[i,k]
            for(int j = 0; j<N; j += stride){
                //cout << "j: " << j << endl;
                __m256 col = _mm256_loadu_ps(&B[k][j]);  //j,j+1,j+2,..,j+7
                //cout << "col loaded" << endl;

                __m256 temp = _mm256_loadu_ps(&C[i][j]);
                //cout << "temp loaded" << endl;
                temp = _mm256_add_ps(temp, _mm256_mul_ps(row, col));
                _mm256_storeu_ps(&C[i][j],temp);
            }
        }
    }
}

void matmul_sse4_old(float** A, float** B, float** C) {
  int stride = 4;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
       /*
          sum = 0;
          for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
          }
          C[i][j] = sum;
       */

        __m128 sum = _mm_setzero_ps();
        // sum: [0,0,0,0]

        for(int k = 0; k<N; k = k+stride){

              //__m128 row = _mm_load_ps(const_cast<float*>(&A[i][k]));
              __m128 row = _mm_load_ps(&A[i][k]);
              __m128 col = _mm_set_ps(B[k+3][j],B[k+2][j],B[k+1][j],B[k][j]);

              sum = _mm_fmadd_ps(row, col, sum); //row[i]*col[i] + sum[i]
              //sum = _mm_add_ps(sum, _mm_mul_ps(row,col));
        }

        //sum:[s1,s2,s3,s4] & we need (s1+s2+s3+s4) at C[i][j]
        //horizontal sum of `sum`
        __m128 x128 = sum;

        // x64: x0+x2, x1+x3, - , -
        __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));

        // x32: x0+x2 + x1+x3, - , - , -
        __m128 x32 = _mm_add_ps(x64, _mm_shuffle_ps(x64, x64, _MM_SHUFFLE(0,3,2,1)));

        // cast x32 to float
        C[i][j] = _mm_cvtss_f32(x32);
        C[i][j] = ((float*)&x32)[0];

    }
  }
}

void matmul_avx2_old(float** A, float** B, float** C) {
  int stride = 8;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      /*
          sum = 0;
          for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
          }
          C[i][j] = sum;
       */

        __m256 sum = _mm256_setzero_ps();
        // sum: [0,0,0,0]

        for(int k = 0; k<N; k = k+stride){

              __m256 row = _mm256_load_ps(const_cast<float*>(&A[i][k]));
              __m256 col = _mm256_set_ps(B[k+7][i], B[k+6][j], B[k+5][j],
                                         B[k+4][j], B[k+3][j], B[k+2][j],
                                         B[k+1][j], B[k][j]);

              sum = _mm256_fmadd_ps(row, col, sum); //row[i]*col[i] + sum[i]
        }

        //sum:[s1,s2,s3,s4,...] & we need (s1+s2+s3+s4+...) at C[i][j]
        //horizontal sum of sum

        // x128: x0+x4, x1+x5, x2+x6, x3+x7
        __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(sum,0), _mm256_extractf128_ps(sum,1));

        // x64: x0+x4+x2+x6, x1+x5+x3+x7, - , -
        __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));

        // x32: x0+x4+x2+x6 + x1+x5+x3+x7, - , - , -
        __m128 x32 = _mm_add_ps(x64, _mm_shuffle_ps(x64, x64, _MM_SHUFFLE(0,3,2,1)));

        // cast x32 to float
        C[i][j] = _mm_cvtss_f32(x32);

    }
  }
}

void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      //float this_diff = w_ref[i][j] - w_opt[i][j];
      double this_diff =(double) w_ref[i][j] -(double) w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
         // cout << "diff: " <<(double) w_ref[i][j] << ", " << w_opt[i][j] << endl;
         // cout << this_diff << endl;
         // exit(0);
          numdiffs++;
          if (this_diff > maxdiff)
              maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  auto** A = new float*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new float[N]();
  }
  auto** B = new float*[N];
  for (int i = 0; i < N; i++) {
    B[i] = new float[N]();
  }

  auto** A_aligned = (float**)aligned_alloc(32, sizeof(float*) * N);
  auto** B_aligned = (float**)aligned_alloc(32, sizeof(float*) * N);

  auto** C_seq = new float*[N];
  auto** C_sse4 = new float*[N];
  auto** C_avx2 = new float*[N];
  auto** C_sse4_aligned = (float**)aligned_alloc(32, sizeof(float*) * N);
  auto** C_avx2_aligned = (float**)aligned_alloc(32, sizeof(float*) * N);
  for (int i = 0; i < N; i++) {
    C_seq[i] = new float[N]();
    C_sse4[i] = new float[N]();
    C_avx2[i] = new float[N]();

    A_aligned[i] = (float*)aligned_alloc(32, sizeof(float) * N);
    B_aligned[i] = (float*)aligned_alloc(32, sizeof(float) * N);
    C_sse4_aligned[i] = (float*)aligned_alloc(32, sizeof(float) * N);
    C_avx2_aligned[i] = (float*)aligned_alloc(32, sizeof(float) * N);
  }

  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0.1F;
      B[i][j] = 0.2F;
      A_aligned[i][j] = 0.1F;
      B_aligned[i][j] = 0.2F;

      C_seq[i][j] = 0.0F;
      C_sse4[i][j] = 0.0F;
      C_avx2[i][j] = 0.0F;
      C_sse4_aligned[i][j] = 0.0F;
      C_avx2_aligned[i][j] = 0.0F;
    }
  }

  HRTimer start = HR::now();
  matmul_seq(A, B, C_seq);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4_unaligned(A, B, C_sse4);
  end = HR::now();
  check_result(C_seq, C_sse4);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 (unaligned) time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_unaligned(A, B, C_avx2);
  end = HR::now();
  check_result(C_seq, C_avx2);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 (unaligned) time: " << duration << " ms" << endl;


  start = HR::now();
  matmul_sse4(A_aligned, B_aligned, C_sse4_aligned);
  end = HR::now();
  check_result(C_seq, C_sse4_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2(A_aligned, B_aligned, C_avx2_aligned);
  end = HR::now();
  check_result(C_seq, C_avx2_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 time: " << duration << " ms" << endl;


  return EXIT_SUCCESS;
}
