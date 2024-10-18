#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 24)
#define ALIGN (64)

float gA[N] __attribute__((aligned(ALIGN)));
float gB[N] __attribute__((aligned(ALIGN)));
float gC[N] __attribute__((aligned(ALIGN)));

float *gD = nullptr;
float *gE = nullptr;
float *gF = nullptr;

void test1(float *__restrict__ pa, float *__restrict__ pb,
           float *__restrict__ pc) {
  __m128 rA, rB, rC;
  for (int i = 0; i < N; i += 4) {
    rA = _mm_load_ps(&pa[i]);
    rB = _mm_load_ps(&pb[i]);
    rC = _mm_add_ps(rA, rB);
    _mm_store_ps(&pc[i], rC);
  }
}

// Using unaligned functions
void test2(float *__restrict__ pa, float *__restrict__ pb,
           float *__restrict__ pc) {
  __m128 rA, rB, rC;
  for (int i = 0; i < N; i += 4) {
    rA = _mm_loadu_ps(&pa[i]);
    rB = _mm_loadu_ps(&pb[i]);
    rC = _mm_add_ps(rA, rB);
    _mm_storeu_ps(&pc[i], rC);
  }
}

// Using unaligned functions starting from an odd iteration
void test3(float *__restrict__ pa, float *__restrict__ pb,
           float *__restrict__ pc) {
  __m128 rA, rB, rC;
  for (int i = 1; i < N - 3; i += 4) {
    rA = _mm_loadu_ps(&pa[i]);
    rB = _mm_loadu_ps(&pb[i]);
    rC = _mm_add_ps(rA, rB);
    _mm_storeu_ps(&pc[i], rC);
  }
}

int main() {
  for (size_t i = 0; i < N; i++) {
    gA[i] = 1;
    gB[i] = static_cast<float>(i);
    gC[i] = 0;
  }

  HRTimer start = HR::now();
  test1(gA, gB, gC);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "static allocation: test1 time (us): " << duration << endl;

  start = HR::now();
  test2(gA, gB, gC);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "static allocation: test2 time (us): " << duration << endl;

  start = HR::now();
  test3(gA, gB, gC);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "static allocation: test3 time (us): " << duration << endl;

  // 64-byte aligned
  gD = static_cast<float *>(_mm_malloc(N * sizeof(int), ALIGN));
  gE = static_cast<float *>(_mm_malloc(N * sizeof(int), ALIGN));
  // C++17 feature
  gF = static_cast<float *>(aligned_alloc(ALIGN, N * sizeof(int)));

  for (size_t i = 0; i < N; i++) {
    gD[i] = 1;
    gE[i] = static_cast<float>(i);
    gF[i] = 0;
  }

  start = HR::now();
  test1(gD, gE, gF);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "dynamic allocation: test1 time (us): " << duration << endl;

  start = HR::now();
  test2(gD, gE, gF);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "dynamic allocation: test2 time (us): " << duration << endl;

  start = HR::now();
  test3(gD, gE, gF);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "dynamic allocation: test3 time (us): " << duration << endl;

  std::free(gF);
  _mm_free(gE);
  _mm_free(gD);

  return EXIT_SUCCESS;
}
