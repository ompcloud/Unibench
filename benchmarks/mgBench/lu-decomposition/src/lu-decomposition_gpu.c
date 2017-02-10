/*
    This program makes the decomposition of matrices.
    It receives an input array and returns two triangular matrices in the same
   array b.
    This program create a csv file with the time execution results for each
   function(CPU,GPU) in this format: size of matrix, cpu time, gpu time.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 1.0

    Run:
    folder_ipmacc/ipmacc folder_archive/LU_decomposition.c
    ./a.out
*/

#include "../../common/mgbenchUtilFunctions.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef RUN_TEST
#define SIZE 500
#define points 250
#elif RUN_BENCHMARK
#define SIZE 500
#define points 250
#else
#define SIZE 500
#define points 250
#endif

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define var SIZE / points

// Initialize matrices.
void init(int s, float *a, float *b) {
  int i, j, q;
  q = s * s;
  for (i = 0; i < s; ++i) {
    for (j = 0; j < s; ++j) {
      a[i * s + j] = (float)(q - (10 * i + 5 * j));
      b[i * s + j] = 0.0f;
    }
  }
}

/// Crout algorithm GPU
/// s = size of matrix
void Crout_GPU(int s, float *a, float *b) {
#pragma omp target map(to : a[0 : SIZE *SIZE]) map(tofrom : b[0 : SIZE *SIZE]) \
    device(DEVICE_ID)
  {
#pragma omp parallel for
    for (int k = 0; k < s; ++k) {
      float sum;
      for (int j = k; j < s; ++j) {
        sum = 0.0;
        for (int i = 0; i < k; ++i) {
          sum += b[j * s + i] * b[i * s + k];
        }
        b[j * s + k] = (a[j * s + k] - sum); // not dividing by diagonals
      }
      for (int i = k + 1; i < s; ++i) {
        sum = 0.0;
        for (int j = 0; j < k; ++j) {
          sum += b[k * s + j] * b[i * s + i];
        }
        b[k * s + i] = (a[k * s + i] - sum) / b[k * s + k];
      }
    }
  }
}

void Crout_CPU(int s, float *a, float *b) {
  int k, j, i;
  float sum;

  for (k = 0; k < s; ++k) {
    for (j = k; j < s; ++j) {
      sum = 0.0;
      for (i = 0; i < k; ++i) {
        sum += b[j * s + i] * b[i * s + k];
      }
      b[j * s + k] = (a[j * s + k] - sum); // not dividing by diagonals
    }
    for (i = k + 1; i < s; ++i) {
      sum = 0.0;
      for (j = 0; j < k; ++j) {
        sum += b[k * s + j] * b[i * s + i];
      }
      b[k * s + i] = (a[k * s + i] - sum) / b[k * s + k];
    }
  }
}

int compareResults(float *b_cpu, float *b_gpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(b_cpu[i * SIZE + j], b_gpu[i * SIZE + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);

  return fail;
}

int main(int argc, char *argv[]) {
  double t_start, t_end;
  int fail = 0;
  int i;

  float *a, *b_cpu, *b_gpu;
  a = (float *)malloc(sizeof(float) * SIZE * SIZE);
  b_cpu = (float *)malloc(sizeof(float) * SIZE * SIZE);
  b_gpu = (float *)malloc(sizeof(float) * SIZE * SIZE);

  fprintf(stdout, "<< LU decomposition GPU >>\n");

  t_start = rtclock();
  for (i = 2; i < SIZE; i += var) {
    init(i, a, b_gpu);
    Crout_GPU(i, a, b_gpu);
  }
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
  t_start = rtclock();
  for (i = 2; i < SIZE; i += var) {
    init(i, a, b_cpu);
    Crout_CPU(i, a, b_cpu);
  }
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(b_cpu, b_gpu);
#endif

  free(a);
  free(b_cpu);
  free(b_gpu);

  return fail;
}
