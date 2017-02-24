/*
   This program performs matrix multiplication on the GPU with
   dynamically allocated matrices.

    Author: Gleison Souza Diniz Mendonça
    Date: 04-01-2015
    version 2.0

    Run:
    ipmacc mat-mul_gpu.c -o mat
    ./mat matrix-size
*/

#include "BenchmarksUtil.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef RUN_TEST
#define SIZE 1234
#elif RUN_BENCHMARK
#define SIZE 16000
#else
#define SIZE 4000
#endif

typedef float DATA_TYPE;

#define PERCENT_DIFF_ERROR_THRESHOLD 0.01

// Initialize matrices.
void init(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c_cpu, DATA_TYPE *c_gpu) {
  int i, j;
  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      if ((i != j || (i%2==0)) && SPARSE) {
        a[i * SIZE + j] = 0;
        b[i * SIZE + j] = 0;
      } else {
        a[i * SIZE + j] = ((DATA_TYPE)i * j) / SIZE;
        b[i * SIZE + j] = ((DATA_TYPE)i * (j + 1)) / SIZE;
      }
    }
  }
}

/// matrix multiplication algorithm GPU
/// s = size of matrix
void mul_GPU(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c) {

#pragma omp target map(to : a[ : SIZE *SIZE], b[0 : SIZE *SIZE])               \
                           map(from : c[ : SIZE *SIZE]) device(DEVICE_ID)
  {
#pragma omp parallel for // collapse(1)
    for (int i = 0; i < SIZE; ++i) {
#pragma omp target data map(to : a[i *SIZE : (i + 1) * SIZE])                  \
                                       map(from : c[i *SIZE : (i + 1) * SIZE])
      for (int j = 0; j < SIZE; ++j) {
        // float sum = 0.0;
        for (int k = 0; k < SIZE; ++k) {
          // sum += a[i * SIZE + k] * b[k * SIZE + j];
          c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];
        }
        // c[i * SIZE + j] = sum;
      }
    }
  }
}

void mul_CPU(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c) {

  int i, j, k;
  DATA_TYPE sum = 0.0;

  for (i = 0; i < SIZE; ++i) {
    for (j = 0; j < SIZE; ++j) {
      sum = 0.0;
      for (k = 0; k < SIZE; ++k) {
        sum = sum + a[i * SIZE + k] * b[k * SIZE + j];
      }
      c[i * SIZE + j] = sum;
    }
  }
}

int compareResults(DATA_TYPE *b_cpu, DATA_TYPE *b_gpu) {
  int i, j, fail;
  fail = 0;

  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      if (percentDiff(b_cpu[i * SIZE + j], b_gpu[i * SIZE + j]) >
          PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
        if (i < 10)
          fprintf(stdout, "%f != %f \n", b_cpu[i * SIZE + j],
                  b_gpu[i * SIZE + j]);
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
  DATA_TYPE *a, *b, *c_cpu, *c_gpu;

  a = (float *)malloc(sizeof(DATA_TYPE) * SIZE * SIZE);
  b = (float *)malloc(sizeof(DATA_TYPE) * SIZE * SIZE);
  c_cpu = (float *)calloc(sizeof(DATA_TYPE), SIZE * SIZE);
  c_gpu = (float *)calloc(sizeof(DATA_TYPE), SIZE * SIZE);

  init(a, b, c_cpu, c_gpu);

  fprintf(stdout, "<< Matrix Multiplication >>\n");

  t_start = rtclock();
  mul_GPU(a, b, c_gpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
  t_start = rtclock();
  mul_CPU(a, b, c_cpu);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(c_cpu, c_gpu);
#endif

  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);

  return fail;
}
