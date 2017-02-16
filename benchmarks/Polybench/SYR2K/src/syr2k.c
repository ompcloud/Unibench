/**
 * syr2k.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "BenchmarksUtil.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.10

/* Problem size */
#ifdef RUN_TEST
#define SIZE 1100
#elif RUN_BENCHMARK
#define SIZE 9600
#else
#define SIZE 1000
#endif

#define N SIZE
#define M SIZE

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] = ((DATA_TYPE)i * j + 2) / N;
    }

    for (j = 0; j < M; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
      B[i * N + j] = ((DATA_TYPE)i * j + 1) / N;
    }
  }
}

void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i * N + j] *= BETA;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

void syr2k_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *Cinit) {

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Cinit[i * N + j] *= BETA;
    }
  }
  DATA_TYPE sum;

//#pragma omp target map(to: A[:N*M], B[:N*M]) map(tofrom: C[:N*N]) device
//(DEVICE_ID)
/*#pragma omp parallel for //collapse(2)
for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
      {
        //sum = 0;
        for (k = 0; k < M; k++)
          {
            C[i*N + j] += ALPHA * A[i*M + k] * B[j*M + k];
            C[i*N + j] += ALPHA * B[i*M + k] * A[j*M + k];
          }
        //C[i*N + j] += sum;
      }
  }*/

#pragma omp target map(to : A[ : N *M],                                        \
                               B[ : N *M], Cinit[ : N *N])                     \
                                   map(from : C[ : N *N]) device(DEVICE_ID)
#pragma omp parallel for // collapse(2)
  for (int i = 0; i < N; i++) {

#pragma omp target data map(from : Cinit[i *N : (i + 1) * N])                  \
                                             map(to : C[i *N : (i + 1) * N])
    for (int j = 0; j < N; j++) {
      C[i * N + j] = Cinit[i * N + j];
      for (int k = 0; k < M; k++) {
        C[i * N + j] += ALPHA * A[i * M + k] * B[j * M + k];
        C[i * N + j] += ALPHA * B[i * M + k] * A[j * M + k];
      }
    }
  }
}

int compareResults(DATA_TYPE *C, DATA_TYPE *C_Gpu) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (percentDiff(C[i * N + j], C_Gpu[i * N + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
  return fail;
}

int main() {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *C;
  DATA_TYPE *Cinit;
  DATA_TYPE *C_Gpu;

  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  Cinit = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C_Gpu = (DATA_TYPE *)calloc(N * M, sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-2k operations >>\n");

  init_arrays(A, B, Cinit);

  t_start = rtclock();
  syr2k_OMP(A, B, C_Gpu, Cinit);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
  init_arrays(A, B, C);

  t_start = rtclock();
  syr2k(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(C, C_Gpu);
#endif

  free(A);
  free(B);
  free(C);
  free(C_Gpu);

  return fail;
}
