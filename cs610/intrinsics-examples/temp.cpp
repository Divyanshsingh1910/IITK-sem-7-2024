#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

//#define N (1 << 10)
#define N (1 << 4)
#define SSE_WIDTH_BITS (128)
#define ALIGN (32)

void print_array(const int *array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i *)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i *)val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}


// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds, one broadcast
int sse4_version(int *__restrict__ source, int *__restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  // Return vector of type __m128i with all elements set to zero, to be added as
  // previous sum for the first four elements.
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be
    // aligned on a 16-byte boundary to be safe.
    __m128i x = _mm_load_si128((__m128i *)&source[i]);
    // Let the numbers in x be [d,c,b,a], where a is at source[i].
    __m128i tmp0 = _mm_slli_si128(x, 4);
    // Shift x left by 4 bytes while shifting in zeros. tmp0 becomes [c,b,a,0].
    __m128i tmp1 =
        _mm_add_epi32(x, tmp0); // Add packed 32-bit integers in x and tmp0.
    // tmp1 becomes [c+d,b+c,a+b,a].
    // Shift tmp1 left by 8 bytes while shifting in zeros.
    __m128i tmp2 = _mm_slli_si128(tmp1, 8); // tmp2 becomes [a+b,a,0,0].
    // Add packed 32-bit integers in tmp2 and tmp1.
    __m128i out = _mm_add_epi32(tmp2, tmp1);
    // out contains [a+b+c+d,a+b+c,a+b,a].
    out = _mm_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by
    // offset.

    cout << "out: ";
    print128i_u32(out, 0);

    // Store 128-bits of integer data from out into memory. dest_addr must be
    // aligned on a 16-byte boundary to be safe.
    _mm_store_si128((__m128i *)&dest[i], out);
    // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the
    // formula (z << 6) | (y << 4) | (x << 2) | w.
    int mask = _MM_SHUFFLE(3, 3, 3, 3);
    // Bits [7:0] of mask are 11111111 to pick the third integer (11) from out
    // (i.e., a+b+c+d).

    // Shuffle 32-bit integers in out using the control in mask.
    offset = _mm_shuffle_epi32(out, mask);
    // offset now contains 4 copies of a+b+c+d.
  }
  return dest[N - 1];
}

void print_ps(__m128 var){
    float a[4];
    _mm_store_ps(&a[0], var);
    cout << "var[0:4]: " << a[0] << " " << a[1] << " " << a[2] << " " << a[3]<<endl;
}

__attribute__((optimize("no-tree-vectorize"))) int main() {
//  int *array = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
//  std::fill(array, array + N, 1);
//
//  int *sse_res = static_cast<int *>(aligned_alloc(ALIGN, N * sizeof(int)));
//  std::fill(sse_res, sse_res + N, 0);
//  HRTimer start = HR::now();
//  int val_sse = sse4_version(array, sse_res);
//  HRTimer end = HR::now();
//  auto duration = duration_cast<microseconds>(end - start).count();
//  delete[] sse_res;
//
//  return EXIT_SUCCESS;
    __m128 a = _mm_set_ps(4.5, 3.3, 2.8, 1.1);
    print_ps(a);
    a = (__m128)_mm_slli_si128((__m128i)a, 4);
    print_ps(a);
    return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "make -j4 "
// End:
