/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05



/* Problem size */
#define N 8192

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;
	
  for (i = 0; i < N; i++)
    {
      tmp[i] = 0;
      y[i] = 0;
      for (j = 0; j < N; j++)
	{
	  tmp[i] = A[i*N + j] * x[j] + tmp[i];
	  y[i] = B[i*N + j] * x[j] + y[i];
	}
		
      y[i] = ALPHA * tmp[i] + BETA * y[i];
    }
}

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
  int i, j;
	
  #pragma omp target device (DEVICE_ID)
  #pragma omp target map(to: A[:N*N], B[:N*N], x[:N], tmp[:N]) map(tofrom: y[:N])
  #pragma omp parallel for
  for (i = 0; i < N; i++)
    {
      tmp[i] = 0;
      y[i] = 0;
      for (j = 0; j < N; j++)
	{
	  tmp[i] = A[i*N + j] * x[j] + tmp[i];
	  y[i] = B[i*N + j] * x[j] + y[i];
	}
      
      y[i] = ALPHA * tmp[i] + BETA * y[i];
    }
}

void init(DATA_TYPE* A, DATA_TYPE* x)
{
  int i, j;

  for (i = 0; i < N; i++)
    {
      x[i] = ((DATA_TYPE) i) / N;
      	
      for (j = 0; j < N; j++) 
	{
	  A[i*N + j] = ((DATA_TYPE) i*j) / N;
	}
    }
}

void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
  int i, fail;
  fail = 0;
	
  for (i=0; i<(N); i++) 
    {
      if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
	{
	  fail++;
	}
    }
  
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;  
  DATA_TYPE* x;  
  DATA_TYPE* y;
  DATA_TYPE* y_outputFromGpu;
  DATA_TYPE* tmp;
	
  A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
  x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
  y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  tmp = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  init(A, x);
	
  t_start = rtclock();
  gesummv_OMP(A, B, x, y_outputFromGpu, tmp);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
  t_start = rtclock();
  gesummv(A, B, x, y, tmp);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
  compareResults(y, y_outputFromGpu);

  free(A);
  free(B);  
  free(x);  
  free(y);
  free(y_outputFromGpu);
  free(tmp);

  return 0;
}

