/**
 * 2mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define ERROR_THRESHOLD 0.05

/* Problem size. */
#ifdef RUN_TEST
#define SIZE 1100
#elif RUN_BENCHMARK
#define SIZE 9600
#else
#define SIZE 1000
#endif

#define NI SIZE
#define NJ SIZE
#define NK SIZE
#define NL SIZE

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D) {
  int i, j;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NK; j++) {
      A[i * NI + j] = ((DATA_TYPE)i * j) / NI;
    }
  }

  for (i = 0; i < NK; i++) {
    for (j = 0; j < NJ; j++) {
      B[i * NK + j] = ((DATA_TYPE)i * (j + 1)) / NJ;
    }
  }

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NJ; j++) {
      // C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      D[i * NL + j] = ((DATA_TYPE)i * (j + 2)) / NK;
    }
  }
}

int compareResults(DATA_TYPE *E, DATA_TYPE *E_GPU) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < NL; i++) {
    for (j = 0; j < NI; j++) {
      if (percentDiff(E[i * NI + j], E_GPU[i * NI + j]) > ERROR_THRESHOLD) {
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

void mm2_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D,
             DATA_TYPE *E) {
  int i, j, k;

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] = 0.0;
      for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
      }
    }
  }

  for (i = 0; i < NI; i++) {
    for (j = 0; j < NL; j++) {
      E[i * NL + j] = 0.0;
      for (k = 0; k < NJ; ++k) {
        E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
      }
    }
  }
}

void mm2_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E) {

#pragma omp target teams map(from: E[:NI*NL], C[:NI*NJ]) map(to: A[:NI*NK], B[:NK*NJ], D[:NJ*NL]) device(DEVICE_ID) 
  {
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NJ; j++) {
        C[i * NJ + j] = 0.0;
        for (int k = 0; k < NK; ++k) {
          C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
      }
    }

    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < NI; i++) {
      for (int j = 0; j < NL; j++) {
        E[i * NL + j] = 0.0;
        for (int k = 0; k < NJ; ++k) {
          E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  double t_start, t_end, t_start_GPU, t_end_GPU;

  int fail = 0;

  DATA_TYPE *C;
  DATA_TYPE *C_GPU;
  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *D;
  DATA_TYPE *E;
  DATA_TYPE *E_GPU;

  C = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  C_GPU = (DATA_TYPE *)calloc(NI * NJ, sizeof(DATA_TYPE));
  A = (DATA_TYPE *)malloc(NI * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NK * NJ * sizeof(DATA_TYPE));
  D = (DATA_TYPE *)malloc(NJ * NL * sizeof(DATA_TYPE));
  E = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));
  E_GPU = (DATA_TYPE *)calloc(NI * NL, sizeof(DATA_TYPE));

  fprintf(stdout,
          "<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D) >>\n");

  init_array(A, B, C, D);

  t_start_GPU = rtclock();
  mm2_OMP(A, B, C_GPU, D, E_GPU);
  t_end_GPU = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_GPU - t_start_GPU);

#ifdef RUN_TEST
  t_start = rtclock();
  mm2_cpu(A, B, C, D, E);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail += compareResults(C, C_GPU);
  fail += compareResults(E, E_GPU);
#endif

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_GPU);

  return fail;
}
