
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <x86intrin.h>
#include "mylib.h"

using std::cout;
using std::endl;

__attribute__((optimize("no-tree-vectorize"))) int main() {

    float a[4];
    for(int i=0; i<4; i++)
        a[i] = i * 1.0;

    __m128 A = _mm_load_ps(a);
    cout << "A"; print128_f32(A);

/*
    float b[8];
    for(int i=0; i<8; i++)
        b[i] = i * 1.0;

    __m256 B = _mm256_load_ps(b);
    cout << "B"; print256_f32(B);
*/
    return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "make -j4 "
// End:
