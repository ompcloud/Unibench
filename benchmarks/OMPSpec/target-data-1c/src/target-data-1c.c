// ----------------------------------------------------------------------------------------
// Implementation of Example target.3c (Section 52.3, page 196) from Openmp
// 4.0.2 Examples
// on the document http://openmp.org/mp-documents/openmp-examples-4.0.2.pdf
//
//
//
//
// ----------------------------------------------------------------------------------------

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

/* Problem size */
#define N 8192

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE *A, DATA_TYPE *B) {
  int i;

  for (i = 0; i < N; i++) {
    A[i] = i / 2.0;
    B[i] = ((N - 1) - i) / 3.0;
  }

  return;
}

void vec_mult(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i;

  for (i = 0; i < N; i++)
    C[i] = A[i] * B[i];
}

void vec_mult_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i;

#pragma omp target data map(to : A[ : N], B[ : N]) map(from : C[ : N])         \
                                device(DEVICE_ID)
  {
#pragma target
#pragma omp parallel for
    for (i = 0; i < N; i++)
      C[i] = A[i] * B[i];
  }
}

int compareResults(DATA_TYPE *B, DATA_TYPE *B_GPU) {
  int i, fail;
  fail = 0;

  // Compare B and B_GPU
  for (i = 0; i < N; i++) {
    if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);

  return fail;
}

int main(int argc, char *argv[]) {
  double t_start, t_end, t_start_OMP, t_end_OMP;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *C_OMP;

  A = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  C_OMP = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two vector multiplication <<\n");

  // initialize the arrays
  init(A, B);

  t_start_OMP = rtclock();
  vec_mult_OMP(A, B, C_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP); //);

#ifdef RUN_TEST
  t_start = rtclock();
  vec_mult(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); //);

  fail = compareResults(C, C_OMP);
#endif

  free(A);
  free(B);
  free(C);
  free(C_OMP);

  return fail;
}
