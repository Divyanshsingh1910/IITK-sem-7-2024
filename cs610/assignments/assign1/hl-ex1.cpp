// export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"

#include <cstdlib>
#include <iostream>
#include <papi.h>

using std::cerr;
using std::cout;
using std::endl;
using std::exit;

#define T (1024 * 1024)

#define N 32 // Fits in 32KB L1 cache
double A[N][N];

void handle_error(int retval) {
  cout << "PAPI error: " << retval << ": " << PAPI_strerror(retval) << "\n";
  exit(EXIT_FAILURE);
}

int main() {
  // Initialize PAPI library
  int retval;

      // Initialize PAPI
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        cout << "PAPI library init error!" << endl;
        exit(EXIT_FAILURE);
    }
  
  retval = PAPI_hl_region_begin("PAPI-HL");
  if (retval != PAPI_OK)
    handle_error(retval);

  /* computation */
  for (int it = 0; it < T; it++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        A[i][j] += 1;
      }
    }
  }

  retval = PAPI_hl_region_end("PAPI-HL");
  if (retval != PAPI_OK)
    handle_error(retval);

  cout << "\nPASSED\n";
  exit(EXIT_SUCCESS);
}
