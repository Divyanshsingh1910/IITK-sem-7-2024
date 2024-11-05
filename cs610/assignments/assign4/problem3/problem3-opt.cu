#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())
typedef unsigned long long int ull;
using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define NSEC_SEC_MUL (1.0e9)

struct timespec begin_grid, end_main;
// to store values of disp.txt
double a[120];
// to store values of grid.txt
double b[30];
const int32_t LOOP_SIZE = 13;
const int32_t NUM_THREADS = LOOP_SIZE*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE;

struct VALUES{
  double* values = nullptr;
  VALUES* next = nullptr;
};

__device__ void allocate_n_assign(VALUES** out, int thread_id, double* in){
  VALUES* temp = new VALUES();
  temp->values = in;

  VALUES* head = out[thread_id];
  if(!head){
    head = temp;
    return;
  }
  while(head->next){
    head = head->next;
  }

  head->next = temp;
}

__constant__ double device_E_vals[10];
__constant__ int device_S_vals[10];


__global__ void gridloopsearch_counting_kernel(unsigned long long int* per_thread_cnt,
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6,
    double dd7, double dd8, double dd9, double dd10, double dd11, double dd12,
    double dd13, double dd14, double dd15, double dd16, double dd17,
    double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27,
    double dd28, double dd29, double dd30, double c11, double c12, double c13,
    double c14, double c15, double c16, double c17, double c18, double c19,
    double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29,
    double c210, double d2, double ey2, double c31, double c32, double c33,
    double c34, double c35, double c36, double c37, double c38, double c39,
    double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49,
    double c410, double d4, double ey4, double c51, double c52, double c53,
    double c54, double c55, double c56, double c57, double c58, double c59,
    double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69,
    double c610, double d6, double ey6, double c71, double c72, double c73,
    double c74, double c75, double c76, double c77, double c78, double c79,
    double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89,
    double c810, double d8, double ey8, double c91, double c92, double c93,
    double c94, double c95, double c96, double c97, double c98, double c99,
    double c910, double d9, double ey9, double c101, double c102, double c103,
    double c104, double c105, double c106, double c107, double c108,
    double c109, double c1010, double d10, double ey10, double kk) {
  // results values
  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // grid search starts
  int r1 = blockIdx.z;
  int r2 = blockIdx.y;
  int r3 = blockIdx.x;
  int r4 = threadIdx.x;
  int r5 = threadIdx.y;
  //printf("(%d,%d,%d)\n",r1,r2,r3);

  int thread_id = r1*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE+ r2*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE
    + r3*LOOP_SIZE*LOOP_SIZE + r4*LOOP_SIZE + r5;
  if(thread_id>= NUM_THREADS)
    return;
  int64_t local_cnt = 0;
  //for (int r1 = 0; r1 < s1; ++r1) {
  x1 = dd1 + r1 * dd3;

  //for (int r2 = 0; r2 < s2; ++r2) {
  x2 = dd4 + r2 * dd6;

  //for (int r3 = 0; r3 < s3; ++r3) {
  x3 = dd7 + r3 * dd9;
  
  //for (int r4 = 0; r4 < s4; ++r4) {
  x4 = dd10 + r4 * dd12;

  //for (int r5 = 0; r5 < s5; ++r5) {
  x5 = dd13 + r5 * dd15;
  per_thread_cnt[thread_id] = 0;

            for (int r6 = 0; r6 < device_S_vals[5]; ++r6) {
              x6 = dd16 + r6 * dd18;

              for (int r7 = 0; r7 < device_S_vals[6]; ++r7) {
                x7 = dd19 + r7 * dd21;

                for (int r8 = 0; r8 < device_S_vals[7]; ++r8) {
                  x8 = dd22 + r8 * dd24;

                  for (int r9 = 0; r9 < device_S_vals[8]; ++r9) {
                    x9 = dd25 + r9 * dd27;

                    for (int r10 = 0; r10 < device_S_vals[9]; ++r10) {
                      x10 = dd28 + r10 * dd30;

                      // constraints

                      q1 = fabs(c11 * x1 + c12 * x2 + c13 * x3 + c14 * x4 +
                                c15 * x5 + c16 * x6 + c17 * x7 + c18 * x8 +
                                c19 * x9 + c110 * x10 - d1);

                      q2 = fabs(c21 * x1 + c22 * x2 + c23 * x3 + c24 * x4 +
                                c25 * x5 + c26 * x6 + c27 * x7 + c28 * x8 +
                                c29 * x9 + c210 * x10 - d2);

                      q3 = fabs(c31 * x1 + c32 * x2 + c33 * x3 + c34 * x4 +
                                c35 * x5 + c36 * x6 + c37 * x7 + c38 * x8 +
                                c39 * x9 + c310 * x10 - d3);

                      q4 = fabs(c41 * x1 + c42 * x2 + c43 * x3 + c44 * x4 +
                                c45 * x5 + c46 * x6 + c47 * x7 + c48 * x8 +
                                c49 * x9 + c410 * x10 - d4);
                      q5 = fabs(c51 * x1 + c52 * x2 + c53 * x3 + c54 * x4 +
                                c55 * x5 + c56 * x6 + c57 * x7 + c58 * x8 +
                                c59 * x9 + c510 * x10 - d5);

                      q6 = fabs(c61 * x1 + c62 * x2 + c63 * x3 + c64 * x4 +
                                c65 * x5 + c66 * x6 + c67 * x7 + c68 * x8 +
                                c69 * x9 + c610 * x10 - d6);

                      q7 = fabs(c71 * x1 + c72 * x2 + c73 * x3 + c74 * x4 +
                                c75 * x5 + c76 * x6 + c77 * x7 + c78 * x8 +
                                c79 * x9 + c710 * x10 - d7);

                      q8 = fabs(c81 * x1 + c82 * x2 + c83 * x3 + c84 * x4 +
                                c85 * x5 + c86 * x6 + c87 * x7 + c88 * x8 +
                                c89 * x9 + c810 * x10 - d8);

                      q9 = fabs(c91 * x1 + c92 * x2 + c93 * x3 + c94 * x4 +
                                c95 * x5 + c96 * x6 + c97 * x7 + c98 * x8 +
                                c99 * x9 + c910 * x10 - d9);

                      q10 = fabs(c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 +
                                 c105 * x5 + c106 * x6 + c107 * x7 + c108 * x8 +
                                 c109 * x9 + c1010 * x10 - d10);

                      if ((q1 <= device_E_vals[0]) && (q2 <= device_E_vals[1]) && (q3 <= device_E_vals[2]) &&
                          (q4 <= device_E_vals[3]) && (q5 <= device_E_vals[4]) && (q6 <= device_E_vals[5]) &&
                          (q7 <= device_E_vals[6]) && (q8 <= device_E_vals[7]) && (q9 <= device_E_vals[8]) &&
                          (q10 <= device_E_vals[9])) {
                        local_cnt = local_cnt + 1;
                        //printf("yes\n");

                        // xi's which satisfy the constraints to be written in file
                        // vector temp = {x1,x2,x3,x4,x5,x6,x7,x8,x9}
                        // result[r1*LOOP_SIZE*LOOP_SIZE + r2*LOOP_SIZE + r3].push_back(temp);
                        //double ttemp[] = {x1,x2,x3,x4,x5,x6,x7,x8,x9,x10};
                        
                      }
                    }
                  }
                }
              }
            }
  /*
  }
  }
  }
  }
  }
  */
  per_thread_cnt[thread_id] = local_cnt;
  // end function gridloopsearch
  //printf("We had total %ld pnts\n",pnts);
}

__global__ void gridloopsearch_kernel(double* buffer, ull* per_thread_cnt,
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6,
    double dd7, double dd8, double dd9, double dd10, double dd11, double dd12,
    double dd13, double dd14, double dd15, double dd16, double dd17,
    double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27,
    double dd28, double dd29, double dd30, double c11, double c12, double c13,
    double c14, double c15, double c16, double c17, double c18, double c19,
    double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29,
    double c210, double d2, double ey2, double c31, double c32, double c33,
    double c34, double c35, double c36, double c37, double c38, double c39,
    double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49,
    double c410, double d4, double ey4, double c51, double c52, double c53,
    double c54, double c55, double c56, double c57, double c58, double c59,
    double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69,
    double c610, double d6, double ey6, double c71, double c72, double c73,
    double c74, double c75, double c76, double c77, double c78, double c79,
    double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89,
    double c810, double d8, double ey8, double c91, double c92, double c93,
    double c94, double c95, double c96, double c97, double c98, double c99,
    double c910, double d9, double ey9, double c101, double c102, double c103,
    double c104, double c105, double c106, double c107, double c108,
    double c109, double c1010, double d10, double ey10, double kk) {
  // results values
  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // grid search starts
  int r1 = blockIdx.z;
  int r2 = blockIdx.y;
  int r3 = blockIdx.x;
  int r4 = threadIdx.x;
  int r5 = threadIdx.y;
  //printf("(%d,%d,%d)\n",r1,r2,r3);

  int thread_id = r1*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE+ r2*LOOP_SIZE*LOOP_SIZE*LOOP_SIZE
    + r3*LOOP_SIZE*LOOP_SIZE + r4*LOOP_SIZE + r5;
  if(thread_id>= NUM_THREADS)
    return;
  int64_t local_cnt = 0;
  //for (int r1 = 0; r1 < s1; ++r1) {
  x1 = dd1 + r1 * dd3;

  //for (int r2 = 0; r2 < s2; ++r2) {
  x2 = dd4 + r2 * dd6;

  //for (int r3 = 0; r3 < s3; ++r3) {
  x3 = dd7 + r3 * dd9;
  
  //for (int r4 = 0; r4 < s4; ++r4) {
  x4 = dd10 + r4 * dd12;

  //for (int r5 = 0; r5 < s5; ++r5) {
  x5 = dd13 + r5 * dd15;

            for (int r6 = 0; r6 < device_S_vals[5]; ++r6) {
              x6 = dd16 + r6 * dd18;

              for (int r7 = 0; r7 < device_S_vals[6]; ++r7) {
                x7 = dd19 + r7 * dd21;

                for (int r8 = 0; r8 < device_S_vals[7]; ++r8) {
                  x8 = dd22 + r8 * dd24;

                  for (int r9 = 0; r9 < device_S_vals[8]; ++r9) {
                    x9 = dd25 + r9 * dd27;

                    for (int r10 = 0; r10 < device_S_vals[9]; ++r10) {
                      x10 = dd28 + r10 * dd30;

                      // constraints

                      q1 = fabs(c11 * x1 + c12 * x2 + c13 * x3 + c14 * x4 +
                                c15 * x5 + c16 * x6 + c17 * x7 + c18 * x8 +
                                c19 * x9 + c110 * x10 - d1);

                      q2 = fabs(c21 * x1 + c22 * x2 + c23 * x3 + c24 * x4 +
                                c25 * x5 + c26 * x6 + c27 * x7 + c28 * x8 +
                                c29 * x9 + c210 * x10 - d2);

                      q3 = fabs(c31 * x1 + c32 * x2 + c33 * x3 + c34 * x4 +
                                c35 * x5 + c36 * x6 + c37 * x7 + c38 * x8 +
                                c39 * x9 + c310 * x10 - d3);

                      q4 = fabs(c41 * x1 + c42 * x2 + c43 * x3 + c44 * x4 +
                                c45 * x5 + c46 * x6 + c47 * x7 + c48 * x8 +
                                c49 * x9 + c410 * x10 - d4);
                      q5 = fabs(c51 * x1 + c52 * x2 + c53 * x3 + c54 * x4 +
                                c55 * x5 + c56 * x6 + c57 * x7 + c58 * x8 +
                                c59 * x9 + c510 * x10 - d5);

                      q6 = fabs(c61 * x1 + c62 * x2 + c63 * x3 + c64 * x4 +
                                c65 * x5 + c66 * x6 + c67 * x7 + c68 * x8 +
                                c69 * x9 + c610 * x10 - d6);

                      q7 = fabs(c71 * x1 + c72 * x2 + c73 * x3 + c74 * x4 +
                                c75 * x5 + c76 * x6 + c77 * x7 + c78 * x8 +
                                c79 * x9 + c710 * x10 - d7);

                      q8 = fabs(c81 * x1 + c82 * x2 + c83 * x3 + c84 * x4 +
                                c85 * x5 + c86 * x6 + c87 * x7 + c88 * x8 +
                                c89 * x9 + c810 * x10 - d8);

                      q9 = fabs(c91 * x1 + c92 * x2 + c93 * x3 + c94 * x4 +
                                c95 * x5 + c96 * x6 + c97 * x7 + c98 * x8 +
                                c99 * x9 + c910 * x10 - d9);

                      q10 = fabs(c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 +
                                 c105 * x5 + c106 * x6 + c107 * x7 + c108 * x8 +
                                 c109 * x9 + c1010 * x10 - d10);

                      if ((q1 <= device_E_vals[0]) && (q2 <= device_E_vals[1]) && (q3 <= device_E_vals[2]) &&
                          (q4 <= device_E_vals[3]) && (q5 <= device_E_vals[4]) && (q6 <= device_E_vals[5]) &&
                          (q7 <= device_E_vals[6]) && (q8 <= device_E_vals[7]) && (q9 <= device_E_vals[8]) &&
                          (q10 <= device_E_vals[9])) {
                      //put values into buffer starting from buffer[per_thread_cnt[thread_id]*10 + local_cnt*10]
                        double* temp = &buffer[(per_thread_cnt[thread_id]+local_cnt)*10];
                        temp[0] = x1;
                        temp[1] = x2;
                        temp[2] = x3;
                        temp[3] = x4;
                        temp[4] = x5;
                        temp[5] = x6;
                        temp[6] = x7;
                        temp[7] = x8;
                        temp[8] = x9;
                        temp[9] = x10;
                        local_cnt = local_cnt + 1;
                      }
                    }
                  }
                }
              }
            }
  /*
  }
  }
  }
  }
  }
  */
  // end function gridloopsearch
  //printf("We had total %ld pnts\n",pnts);
}

bool debug =false;
int main() {

  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  dim3 dimGrid(LOOP_SIZE, LOOP_SIZE,LOOP_SIZE);
  dim3 dimBlock((LOOP_SIZE), (LOOP_SIZE), 1);
  
  cudaEvent_t start, end;
  cudaCheckError( cudaEventCreate(&start) );
  cudaCheckError( cudaEventCreate(&end) );
  cudaCheckError( cudaEventRecord(start) );

  ////////////////// PRE-COMPUTATION ON CPU ///////////////////////// 

  double* E_vals = (double*)malloc(10*sizeof(double));
  int* S_vals = (int*)malloc(10*sizeof(int));
  E_vals[0] = kk * a[12*0 + 11];
  E_vals[1] = kk * a[12*1 + 11];
  E_vals[2] = kk * a[12*2 + 11];
  E_vals[3] = kk * a[12*3 + 11];
  E_vals[4] = kk * a[12*4 + 11];
  E_vals[5] = kk * a[12*5 + 11];
  E_vals[6] = kk * a[12*6 + 11];
  E_vals[7] = kk * a[12*7 + 11];
  E_vals[8] = kk * a[12*8 + 11];
  E_vals[9] = kk * a[12*9 + 11];

  S_vals[0] = floor((b[0*3 + 1] - b[0*3 + 0]) / b[0*3 + 2]);
  S_vals[1] = floor((b[1*3 + 1] - b[1*3 + 0]) / b[1*3 + 2]);
  S_vals[2] = floor((b[2*3 + 1] - b[2*3 + 0]) / b[2*3 + 2]);
  S_vals[3] = floor((b[3*3 + 1] - b[3*3 + 0]) / b[3*3 + 2]);
  S_vals[4] = floor((b[4*3 + 1] - b[4*3 + 0]) / b[4*3 + 2]);
  S_vals[5] = floor((b[5*3 + 1] - b[5*3 + 0]) / b[5*3 + 2]);
  S_vals[6] = floor((b[6*3 + 1] - b[6*3 + 0]) / b[6*3 + 2]);
  S_vals[7] = floor((b[7*3 + 1] - b[7*3 + 0]) / b[7*3 + 2]);
  S_vals[8] = floor((b[8*3 + 1] - b[8*3 + 0]) / b[8*3 + 2]);
  S_vals[9] = floor((b[9*3 + 1] - b[9*3 + 0]) / b[9*3 + 2]);

  cudaMemcpyToSymbol(device_E_vals, E_vals, 10*sizeof(double));
  cudaMemcpyToSymbol(device_S_vals, S_vals, 10*sizeof(int));
   
  ///////////////// INVOKING KERNEL INSIDE CPU LOOPS ////////////////
  ull* d_pnts;
  cudaCheckError( cudaMalloc(&d_pnts,8));
  
  //// COUNTING KERNEL LAUNCH ///////
  ull* d_per_thread_cnt;
  cudaCheckError( cudaMalloc(&d_per_thread_cnt, NUM_THREADS*sizeof(ull)));
  ull* h_per_thread_cnt = (ull*)malloc(NUM_THREADS*sizeof(ull)); 

  gridloopsearch_counting_kernel<<<dimGrid, dimBlock>>>(d_per_thread_cnt,
       b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
       b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21],
       b[22], b[23], b[24], b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2],
       a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
       a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
       a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33],
       a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
       a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53],
       a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63],
       a[64], a[65], a[66], a[67], a[68], a[69], a[70], a[71], a[72], a[73],
       a[74], a[75], a[76], a[77], a[78], a[79], a[80], a[81], a[82], a[83],
       a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91], a[92], a[93],
       a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102], a[103],
       a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
       a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  
  ull last_value;
  cudaCheckError( cudaMemcpy(&last_value, d_per_thread_cnt+NUM_THREADS-1, 8, cudaMemcpyDeviceToHost));

  thrust::device_ptr<ull> per_thread_thrust(d_per_thread_cnt);
  thrust::exclusive_scan(per_thread_thrust, per_thread_thrust + NUM_THREADS, per_thread_thrust);

  //// COMPUTING KERNEL LAUNCH ///////
  ull TOT_POINTS = last_value;
  cudaCheckError( cudaMemcpy(&last_value, d_per_thread_cnt+NUM_THREADS-1, 8, cudaMemcpyDeviceToHost));
  TOT_POINTS += last_value;
  //assert(TOT_POINTS==11608);

  double* d_buffer;
  //printf(" Total points: %llu, Memory assigned to buffer: (%llu)\n",TOT_POINTS, TOT_POINTS*10*sizeof(double));
  cudaCheckError( cudaMalloc(&d_buffer, TOT_POINTS*10*sizeof(double)));

  gridloopsearch_kernel<<<dimGrid, dimBlock>>>(d_buffer, d_per_thread_cnt,
       b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
       b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21],
       b[22], b[23], b[24], b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2],
       a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
       a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
       a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33],
       a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
       a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53],
       a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63],
       a[64], a[65], a[66], a[67], a[68], a[69], a[70], a[71], a[72], a[73],
       a[74], a[75], a[76], a[77], a[78], a[79], a[80], a[81], a[82], a[83],
       a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91], a[92], a[93],
       a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102], a[103],
       a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
       a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);

  double* h_buffer = (double*) malloc(TOT_POINTS*10*sizeof(double));
  cudaCheckError( cudaMemcpy(h_buffer, d_buffer, TOT_POINTS*10*sizeof(double), cudaMemcpyDeviceToHost));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("<kernel> Error: %s\n", cudaGetErrorString(err));
 

  //printing to file
  
  FILE* fptr = fopen("./results-v0.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }
  for(int i=0; i<TOT_POINTS*10; i += 10){
    double* output = &h_buffer[i];
    fprintf(fptr, "%lf\t", output[0]);
    fprintf(fptr, "%lf\t", output[1]);
    fprintf(fptr, "%lf\t", output[2]);
    fprintf(fptr, "%lf\t", output[3]);
    fprintf(fptr, "%lf\t", output[4]);
    fprintf(fptr, "%lf\t", output[5]);
    fprintf(fptr, "%lf\t", output[6]);
    fprintf(fptr, "%lf\t", output[7]);
    fprintf(fptr, "%lf\t", output[8]);
    fprintf(fptr, "%lf\n", output[9]);
  }
  fclose(fptr);
  
  printf("results pnts: %llu\n", TOT_POINTS);
  cudaCheckError( cudaEventRecord(end) );
  cudaCheckError( cudaEventSynchronize(end) );
  float kernel_time;
  cudaCheckError( cudaEventElapsedTime(&kernel_time, start, end) );
  std::cout << "Kernel(Optimized) time (ms): " << kernel_time << "\n";


  return EXIT_SUCCESS;
}
