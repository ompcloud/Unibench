/*
    This program checks the collinearity of points.
    It receives an input a vector with points and returns the mathematical
   functions that pass these points. It have a list to store answers.
    This program create a csv file with the time execution results for each
   function(CPU,GPU) in this format: size of vector, cpu with list time, gpu
   with list time.

    Author: Gleison Souza Diniz Mendon?a
    Date: 04-05-2015
    version 2.0

    Run:
    folder_ipmacc/ipmacc folder_archive/colinear_v2.c
    ./a.out
*/
#include "BenchmarksUtil.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef RUN_TEST
#define SIZE 1024
#elif RUN_BENCHMARK
#define SIZE 1024 * 16 * 2
#else
#define SIZE 1024
#endif

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

typedef struct point {
  int x;
  int y;
} point;

point *points;

void generate_points() {
  int i;
  for (i = 0; i < SIZE; i++) {
    points[i].x = (i * 777) % 11;
    points[i].y = (i * 777) % 13;
  }
}

/// colinear list algorithm GPU
/// N = size of vector
int colinear_list_points_GPU() {
  int val = 0;
  int *parallel_lines = (int *)malloc(sizeof(int) * SIZE);

  for (int i = 0; i < SIZE; i++) {
    parallel_lines[i] = 0;
  }

#pragma omp target map(to : points[ : SIZE])                                   \
    map(tofrom : parallel_lines[ : SIZE]) device(DEVICE_ID)
  {
#pragma omp parallel for collapse(1)
    for (int i = 0; i < SIZE; ++i) {
      for (int j = 0; j < SIZE; j++) {
        for (int k = 0; k < SIZE; k++) {
          /// to understand if is colinear points
          int slope_coefficient, linear_coefficient;
          int ret;
          ret = 0;
          slope_coefficient = points[j].y - points[i].y;
          if ((points[j].x - points[i].x) != 0) {
            slope_coefficient = slope_coefficient / (points[j].x - points[i].x);
            linear_coefficient =
                points[i].y - (points[i].x * slope_coefficient);
            if (slope_coefficient != 0 && linear_coefficient != 0 &&
                points[k].y ==
                    (points[k].x * slope_coefficient) + linear_coefficient) {
              ret = 1;
            }
          }
          if (ret == 1) {
            parallel_lines[(i & (SIZE - 1))] = 1;
          }
        }
      }
    }
  }

  val = 0;
  for (int i = 0; i < SIZE; i++) {
    if (parallel_lines[i] == 1) {
      val = 1;
      break;
    }
  }

  free(parallel_lines);

  return val;
}

int colinear_list_points_CPU() {

  int i, j, k, val;
  val = 0;

  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      for (k = 0; k < SIZE; k++) {
        /// to understand if is colinear points
        int slope_coefficient, linear_coefficient;
        int ret;
        ret = 0;
        slope_coefficient = points[j].y - points[i].y;

        if ((points[j].x - points[i].x) != 0) {
          slope_coefficient = slope_coefficient / (points[j].x - points[i].x);
          linear_coefficient = points[i].y - (points[i].x * slope_coefficient);

          if (slope_coefficient != 0 && linear_coefficient != 0 &&
              points[k].y ==
                  (points[k].x * slope_coefficient) + linear_coefficient) {
            ret = 1;
          }
        }
        /// to list add
        if (ret == 1) {
          val = 1;
        }
      }
    }
  }

  return val;
}

int compareResults(int A, int A_outputFromGpu) {
  int i, j, fail;
  fail = 0;
  if (A != A_outputFromGpu)
    fail = 1;

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
  return fail;
}

int main(int argc, char *argv[]) {
  double t_start, t_end;
  int fail = 0;

  int result_CPU, result_GPU;

  fprintf(stdout, "<< Collinear List >>\n");

  points = (point *)malloc(sizeof(points) * SIZE);
  generate_points();

  t_start = rtclock();
  result_GPU = colinear_list_points_GPU();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

#ifdef RUN_TEST
  t_start = rtclock();
  result_CPU = colinear_list_points_CPU();
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  fail = compareResults(result_GPU, result_CPU);
#endif

  free(points);

  return fail;
}
