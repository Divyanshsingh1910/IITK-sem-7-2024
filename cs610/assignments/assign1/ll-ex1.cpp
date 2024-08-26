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

int main() {
  /* Initialize the PAPI library */
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  // Positive return code other than PAPI_VER_CURRENT indicates a library
  // version mismatch and a negative return code indicates an initialization
  // error.
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    cerr << "PAPI library version mismatch: " << retval << " != " << PAPI_VER_CURRENT << "\n";
    exit(EXIT_FAILURE);
  } else if (retval < 0) {
    cerr << "PAPI library initialization error: " << retval << " != " << PAPI_VER_CURRENT << "\n";
    exit(EXIT_FAILURE);
  }

  int eventset = PAPI_NULL;
  retval = PAPI_create_eventset(&eventset);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_create_eventset()" << endl;
    exit(EXIT_FAILURE);
  }

  if (PAPI_add_event(eventset, PAPI_TOT_INS) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_TOT_INS!\n";
    exit(EXIT_FAILURE);
  }
  if (PAPI_add_event(eventset, PAPI_TOT_CYC) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_TOT_CYC!\n";
    exit(EXIT_FAILURE);
  }

  retval = PAPI_start(eventset);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_start()" << endl;
    exit(EXIT_FAILURE);
  }

  /* computation */
  for (int it = 0; it < T; it++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < N; i++) {
        A[i][j] += 1;
      }
    }
  }

  long long int values[2];
  retval = PAPI_stop(eventset, values);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_stop()" << endl;
    exit(EXIT_FAILURE);
  }

  PAPI_cleanup_eventset(eventset);
  PAPI_destroy_eventset(&eventset);
  PAPI_shutdown();

  cout << "TOT_INS: " << values[0] << "\nTOT_CYC: " << values[1] << endl;
  return EXIT_SUCCESS;
}
