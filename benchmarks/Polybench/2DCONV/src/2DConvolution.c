/**
 * 2DConvolution.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
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

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
  c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
  c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

  for (int i = 1; i < NI - 1; ++i) // 0
    {
      for (int j = 1; j < NJ - 1; ++j) // 1
	{
	  B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
	    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
	    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
	}
    }
}

void conv2D_OMP(DATA_TYPE* A, DATA_TYPE* B)
{
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
  c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
  c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

  #pragma omp target map(to: A[:NI*NJ]) map(from: B[:NI*NJ]) device(DEVICE_ID)
  #pragma omp parallel for
  for (int i = 1; i < NI - 1; ++i)
    {
      for (int j = 1; j < NJ - 1; ++j)
	{
	  B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
	    + c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
	    + c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
	}
    }
}

void init(DATA_TYPE* A)
{
  int i, j;

  for (i = 0; i < NI; ++i)
    {
      for (j = 0; j < NJ; ++j)
	{
	  A[i*NJ + j] = (float)rand()/RAND_MAX;
	}
    }
}

int compareResults(DATA_TYPE* B, DATA_TYPE* B_GPU)
{
  int i, j, fail;
  fail = 0;
	
  // Compare B and B_GPU
  for (i=1; i < (NI-1); i++) 
    {
      for (j=1; j < (NJ-1); j++) 
	{
	  if (percentDiff(B[i*NJ + j], B_GPU[i*NJ + j]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
	
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);

  return fail;
}

int main(int argc, char *argv[])
{
  double t_start, t_end, t_start_OMP, t_end_OMP;
  int fail = 0;

  DATA_TYPE* A;
  DATA_TYPE* B;  
  DATA_TYPE* B_OMP;
	
  A = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
  B_OMP = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two dimensional (2D) convolution <<\n");

  //initialize the arrays
  init(A);

  t_start_OMP = rtclock();
  conv2D_OMP(A, B_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP);//);

#ifdef RUN_TEST
  t_start = rtclock();
  conv2D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);//);
	
  compareResults(B, B_OMP);
#endif

  free(A);
  free(B);
  free(B_OMP);
	
  return fail;
}

