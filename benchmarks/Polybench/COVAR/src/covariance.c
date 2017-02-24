/**
 * covariance.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#ifdef RUN_TEST
#define SIZE 1100
#elif RUN_BENCHMARK
#define SIZE 16000
#else
#define SIZE 1000
#endif

/* Problem size */
#define M SIZE
#define N SIZE

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#ifndef SPARSE
#define SPARSE 0
#endif

void init_arrays(DATA_TYPE *data) {
  int i, j;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if ((i != j || (i%2==0)) && SPARSE) {
        data[i * (N + 1) + j] = 0;
      } else {
        data[i * (N + 1) + j] = ((DATA_TYPE)i * j) / M;
      }
    }
  }
}

int compareResults(DATA_TYPE *symmat, DATA_TYPE *symmat_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  for (i = 1; i < (M + 1); i++) {
    for (j = 1; j < (N + 1); j++) {
      if (percentDiff(symmat[i * (N + 1) + j],
                      symmat_outputFromGpu[i * (N + 1) + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
  return fail;
}

void covariance(DATA_TYPE *data, DATA_TYPE *symmat, DATA_TYPE *mean) {
  int i, j, j1, j2;

  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j < (M + 1); j++) {
    mean[j] = 0.0;
    for (i = 1; i < (N + 1); i++) {
      mean[j] += data[i * (M + 1) + j];
    }
    mean[j] /= FLOAT_N;
  }

  /* Center the column vectors. */
  for (i = 1; i < (N + 1); i++) {
    for (j = 1; j < (M + 1); j++) {
      data[i * (M + 1) + j] -= mean[j];
    }
  }

  /* Calculate the m * m covariance matrix. */
  for (j1 = 1; j1 < (M + 1); j1++) {
    for (j2 = j1; j2 < (M + 1); j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;
      for (i = 1; i < N + 1; i++) {
        symmat[j1 * (M + 1) + j2] +=
            data[i * (M + 1) + j1] * data[i * (M + 1) + j2];
      }
      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }
}

void covariance_OMP(DATA_TYPE *data, DATA_TYPE *data2, DATA_TYPE *symmat,
                    DATA_TYPE *mean) {

/* Determine mean of column vectors of input data matrix */

#pragma omp target map(to : data[ : (M + 1) * (N + 1)]) map(                   \
    tofrom : mean[ : (M + 1)],                                                 \
                   data2[ : (M + 1) * (N + 1)]) map(from : symmat[ : (         \
                       M + 1) * (N + 1)]) device(DEVICE_ID)
  {
#pragma omp parallel for
    for (int j = 1; j < (M + 1); j++) {
      mean[j] = 0.0;
      for (int i = 1; i < (N + 1); i++) {
        mean[j] += data[i * (M + 1) + j];
      }
      mean[j] /= FLOAT_N;
    }

/* Center the column vectors. */
#pragma omp parallel for // collapse(2)
    for (int i = 1; i < (N + 1); i++) {
#pragma omp target data map(to : data[i *(M + 1) : (i + 1) * (M + 1)])                      \
                                       map(tofrom : data2[i *(M + 1) : (i + 1) * (M + 1)])
      for (int j = 1; j < (M + 1); j++) {
        data2[i * (M + 1) + j] = data[i * (M + 1) + j] - mean[j];
      }
    }

/* Calculate the m * m covariance matrix. */
#pragma omp parallel for // collapse(2) schedule(dynamic,8)
    for (int j1 = 1; j1 < (M + 1); j1++) {
      for (int j2 = j1; j2 < (M + 1); j2++) {
        symmat[j1 * (M + 1) + j2] = 0.0;
        for (int i = 1; i < N + 1; i++) {
          symmat[j1 * (M + 1) + j2] +=
              data2[i * (M + 1) + j1] * data2[i * (M + 1) + j2];
        }
        symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
      }
    }
  }
}

int main() {
  double t_start, t_end;
  int fail = 0;

  DATA_TYPE *data;
  DATA_TYPE *data_GPU;
  DATA_TYPE *data2_GPU;
  DATA_TYPE *symmat;
  DATA_TYPE *mean;
  DATA_TYPE *mean_GPU;
  DATA_TYPE *symmat_outputFromGpu;

  data = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  data2_GPU = (DATA_TYPE *)calloc((M + 1) * (N + 1), sizeof(DATA_TYPE));
  symmat = (DATA_TYPE *)calloc((M + 1) * (M + 1), sizeof(DATA_TYPE));
  mean = (DATA_TYPE *)calloc((M + 1), sizeof(DATA_TYPE));
  symmat_outputFromGpu =
      (DATA_TYPE *)calloc((M + 1) * (M + 1), sizeof(DATA_TYPE));
  mean_GPU = (DATA_TYPE *)calloc((M + 1), sizeof(DATA_TYPE));

  fprintf(stdout, "<< Covariance Computation >>\n");

  init_arrays(data);

  t_start = rtclock();
  covariance_OMP(data, data2_GPU, symmat_outputFromGpu, mean_GPU);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST

  t_start = rtclock();
  covariance(data, symmat, mean);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(symmat, symmat_outputFromGpu);
#endif

  free(data);
  free(symmat);
  free(mean);
  free(symmat_outputFromGpu);

  return fail;
}
