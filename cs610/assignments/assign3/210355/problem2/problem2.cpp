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

#define N (1 << 30)
#define SSE_WIDTH_BITS (128)
#define ALIGN (32)

/** Helper methods for debugging */

void print_array(const int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print256i_u32(__m256i var) {
  alignas(ALIGN) uint32_t val[8];
  _mm256_store_si256((__m256i*)val, var);

  cout << "-->: ";
  for(int i=0; i<8; i++){
      cout << val[i] << " ";
  }
  cout << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

__attribute__((optimize("no-tree-vectorize"))) int
ref_version(int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp)
  for (int i = 0; i < N; i++) {
    tmp += source[i];
#pragma omp scan inclusive(tmp)
    dest[i] = tmp;
  }
  return tmp;
}

// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds,
// one broadcast
int sse4_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  // Return vector of type __m128i with all elements set to zero, to be added as
  // previous sum for the first four elements.
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be
    // aligned on a 16-byte boundary to be safe.
    __m128i x = _mm_load_si128((__m128i*)&source[i]);
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
    // Store 128-bits of integer data from out into memory. dest_addr must be
    // aligned on a 16-byte boundary to be safe.
    _mm_store_si128((__m128i*)&dest[i], out);
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

inline __m256i  _mm256_sll(__m256i var, int x){
    uint8_t mask = 0b11111111;
    mask = mask << x;
    __m256i zero = _mm256_setzero_si256();

    __m256i shift_mask;

    switch(x){
        case 4: shift_mask = _mm256_set_epi32(3,2,1,0,0,0,0,0);
                break;

        case 2: shift_mask = _mm256_set_epi32(5,4,3,2,1,0,0,0);
                break;

        default: shift_mask = _mm256_set_epi32(6,5,4,3,2,1,0,0);

    }

    __m256i temp = _mm256_permutevar8x32_epi32(var,shift_mask);

    return _mm256_blend_epi32(zero, temp, mask);
}

int avx2_version(const int* __restrict__ source, int* __restrict__ dest){
    __builtin_assume_aligned(source, ALIGN*2);
    __builtin_assume_aligned(dest, ALIGN*2);

    //running prefix sum
    __m256i offset = _mm256_setzero_si256();

    const int stride = 8;

    for(int i=0; i<N; i += stride){
        //take 8 floats into a ymm
        // x: a7, a6, a5, ...a1, a0
        __m256i x = _mm256_load_si256((__m256i*)&source[i]);
        // cout << "x"; print256i_u32(x);

        /*Now, I want to have prefix sum of x in x*/

        //temp0: a6, a5, ... , a0, 0
        __m256i temp0 = _mm256_sll(x,1);

        //temp1: a6+a7, a5+a6, a4+a5, a3+a4, a2+a3, a1+a2, a0+a1, a0
        __m256i temp1 = _mm256_add_epi32(x, temp0);

        //temp2: a4+a5, a3+a4, a2+a3, a1+a2, a0+a1, a0, 0, 0
        __m256i temp2 = _mm256_sll(temp1,2);

        //temp3: a4+a5+a6+a7, a3+a4+a5+a6, a2+a3+a4+a5, a1+a2+a3+a4,
        //       a0+a1+a2+a3, a0+a1+a2   , a0+a1,       a0
        __m256i temp3 = _mm256_add_epi32(temp1, temp2);

        //temp4: a0+a1+a2+a3, a0+a1+a2, a0+a1, a0, 0, 0, 0, 0
        __m256i temp4 = _mm256_sll(temp3, 4);

        //out: temp3 + temp4 <--- this is the prefix
        __m256i out = _mm256_add_epi32(temp3, temp4);

        out = _mm256_add_epi32(out, offset);
        //out[7]: has the prefix sum till source[i+7]
        _mm256_store_si256((__m256i*)&dest[i], out);

        //offset should have 8 copies of out[7]
        //first extract the 7th element--> broadcast over offset
        offset = _mm256_set1_epi32(_mm256_extract_epi32(out, 7));

    }

    return dest[N-1];
}

__attribute__((optimize("no-tree-vectorize"))) int main() {
    int* array = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
    std::fill(array, array + N, 1);

    int* ref_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
    std::fill(ref_res, ref_res + N, 0);
    HRTimer start = HR::now();
    int val_ser = ref_version(array, ref_res);
    HRTimer end = HR::now();
    auto duration = duration_cast<microseconds>(end - start).count();
     cout << "Serial version: " << val_ser << " time: " << duration << endl;

    int* omp_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
    std::fill(omp_res, omp_res + N, 0);
    start = HR::now();
    int val_omp = omp_version(array, omp_res);
    end = HR::now();
    duration = duration_cast<microseconds>(end - start).count();
    assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
     cout << "OMP version: " << val_omp << " time: " << duration << endl;
    free(omp_res);

    int* sse_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
    std::fill(sse_res, sse_res + N, 0);
    start = HR::now();
    int val_sse = sse4_version(array, sse_res);
    end = HR::now();
    duration = duration_cast<microseconds>(end - start).count();
    assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
     cout << "SSE version: " << val_sse << " time: " << duration << endl;
    free(sse_res);

    int* avx2_res = static_cast<int*>(aligned_alloc(ALIGN*2, N * sizeof(int)));
    std::fill(avx2_res, avx2_res + N, 0);
    start = HR::now();
    int val_avx2 = avx2_version(array, avx2_res);
    end = HR::now();
    duration = duration_cast<microseconds>(end - start).count();
    assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
     cout << "AVX2 version: " << val_avx2 << " time: " << duration << endl;

  return EXIT_SUCCESS;
}
