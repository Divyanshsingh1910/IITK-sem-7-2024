#include <cstdlib>
#include <immintrin.h>
#include <iostream>

using std::cout;
using std::endl;

// We can use Boost Predef for checking the compiler
int main() {
  // GCC Specific
  // https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html

  // Clang and Intel support GNU extensions
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) &&         \
    !defined(__INTEL_COMPILER)
  if (__builtin_cpu_supports("mmx")) {
    cout << "MMX is supported!\n";
  } else {
    cout << "MMX is not supported!\n";
  }

  if (__builtin_cpu_supports("sse2")) {
    cout << "SSE2 is supported!\n";
  } else {
    cout << "SSE2 is not supported!\n";
  }

  if (__builtin_cpu_supports("sse4.1")) {
    cout << "SSE4.1 is supported!\n";
  } else {
    cout << "SSE4.1 is not supported!\n";
  }

  if (__builtin_cpu_supports("avx2")) {
    cout << "AVX2 is supported!\n";
  } else {
    cout << "AVX2 is not supported!\n";
  }
#endif

  // Intel Intrinsics API
  // https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/may-i-use-cpu-feature.html

#if defined(__INTEL_COMPILER)
  if (_may_i_use_cpu_feature(_FEATURE_SSE4_1 | _FEATURE_SSE4_2)) {
    cout << "SSE4.x is supported!\n";
  } else {
    cout << "SSE4.x is not supported!\n";
  }

  if (_may_i_use_cpu_feature(_FEATURE_AVX2)) {
    cout << "AVX2 is supported!\n";
  } else {
    cout << "AVX2 is not supported!\n";
  }

  if (_may_i_use_cpu_feature(_FEATURE_AVX512)) {
    cout << "AVX512 is supported!\n";
  } else {
    cout << "AVX512 is not supported!\n";
  }

  // Allows the compiler to generate the necessary code to use the AVX and SSE2
  // features in the processor.
  // _allow_cpu_features(_FEATURE_SSE2 | _FEATURE_AVX);
#endif

  return EXIT_SUCCESS;
}
