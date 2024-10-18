#include <iostream>

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

int main(int argc, char **argv)
{
    __m128 a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    __m128 b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);

    __m128 c = _mm_add_ps(a, b);

    float d[4];
    _mmu_store_ps(d, c);

    std::cout << "result equals " << d[0] << "," << d[1]
              << "," << d[2] << "," << d[3] << std::endl;

    return 0;
}
