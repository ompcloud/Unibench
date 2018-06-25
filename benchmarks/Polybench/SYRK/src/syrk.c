/**
 * syrk.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
*/

#include "BenchmarksUtil.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

// define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#ifdef RUN_TEST
#define SIZE 1100
#elif RUN_BENCHMARK
#define SIZE 9600
#else
#define SIZE 1000
#endif

/* Problem size */
#define N SIZE
#define M SIZE

/* Declared constant values for alpha and beta */
/* (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE *A, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      A[i * M + j] = ((DATA_TYPE)i * j) / N;
    }
    for (j = 0; j < M; j++) {
      C[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
      D[i * M + j] = ((DATA_TYPE)i * j + 2) / N;
    }
  }
}

int compareResults(DATA_TYPE *C, DATA_TYPE *D) {
  int i, j, fail;
  fail = 0;

  // Compare C with D
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      if (percentDiff(C[i * M + j], D[i * M + j]) > ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         ERROR_THRESHOLD, fail);

  return fail;
}

void syrk(DATA_TYPE *A, DATA_TYPE *C) {
  int i, j, k;

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      C[i * M + j] *= beta;
    }
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      for (k = 0; k < M; k++) {
        C[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
      }
    }
  }
}

void syrkGPU(DATA_TYPE *A, DATA_TYPE *Dinit, DATA_TYPE *D1, DATA_TYPE *D2) {
  double t_start, t_end;

  t_start = rtclock();

  #pragma omp target teams map(to : A[ : N *M], Dinit[ : N *M]) map(tofrom : D1[ : N *M], D2[ : N *M]) device(DEVICE_ID)
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        D1[i * M + j] = Dinit[i * M + j] * beta;
      }
    }
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        D2[i * N + j] = D1[i * N + j];
        for (int k = 0; k < M; k++) {
          D2[i * N + j] += alpha * A[i * M + k] * A[j * M + k];
        }
      }
    }
  }

  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}

int main() {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *A;
  DATA_TYPE *C;
  DATA_TYPE *Dinit;
  DATA_TYPE *D1;
  DATA_TYPE *D2;

  A = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  C = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  Dinit = (DATA_TYPE *)malloc(N * M * sizeof(DATA_TYPE));
  D1 = (DATA_TYPE *)calloc(N * M, sizeof(DATA_TYPE));
  D2 = (DATA_TYPE *)calloc(N * M, sizeof(DATA_TYPE));

  fprintf(stdout, "<< Symmetric rank-k operations >>\n");

  init_arrays(A, C, Dinit);
  syrkGPU(A, Dinit, D1, D2);

#ifdef RUN_TEST
  t_start = rtclock();
  syrk(A, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(C, D2);
#endif

  free(A);
  free(C);
  free(Dinit);
  return fail;
}
