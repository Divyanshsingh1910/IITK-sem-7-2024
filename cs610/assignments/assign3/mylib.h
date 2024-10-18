#include <iostream>
#include <x86intrin.h>

using namespace std;
#define ALIGN 16

void print_array(const int *array, int N) {
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
  cout << "->: " << val[0] << " " << val[1] << "\n";
}

void print128_f32(__m128 var){
  alignas(ALIGN) float val[4];
  _mm_store_ps(val, var);
  cout << "->: " << val[0] << " " << val[1] <<
      " " << val[2] << " " << val[3] << "\n";
}

/*
void print256_f32(__m256 var){
  alignas(ALIGN) float val[8];
  _mm256_store_ps(val, var);
  cout << "->: ";

  for(auto v: val)
      cout << v << " ";
  cout << "\n";
}
*/
