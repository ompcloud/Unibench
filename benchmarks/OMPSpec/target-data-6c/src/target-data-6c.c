// ----------------------------------------------------------------------------------------
// Implementation of Example target.3c (Section 52.3, page 196) from Openmp 4.0.2 Examples 
// on the document http://openmp.org/mp-documents/openmp-examples-4.0.2.pdf
//
// 
//
//
// ----------------------------------------------------------------------------------------

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#define GPU_DEVICE 1

/* Problem size */
#define N 8192
#define THRESHOLD 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init(DATA_TYPE * A, DATA_TYPE * B) {
	int i;

	for(i=0;i<N;i++) {
		A[i] = i/2.0;
		B[i] = ((N-1)-i)/3.0;
	}

	return;
}

void init_again(DATA_TYPE * A, DATA_TYPE * B) {
	int i;

	for(i=0;i<N;i++) {
		A[i] = i;
		B[i] = ((N-1)-i);
	}

	return;
}

void vec_mult(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i;

	for (i=0; i<N; i++)
		C[i] = A[i] * B[i];

	init_again(A, B);

	for (i=0; i<N; i++)
		C[i] += A[i] * B[i];
}

void vec_mult_OMP(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C) {
	int i;

	#pragma omp target data if(N>THRESHOLD) map(from: C[:N])
	{
		#pragma omp target if(N>THRESHOLD) map(to: A[:N], B[:N])
		#pragma omp parallel for
		for (i=0; i<N; i++)
			C[i] = A[i] * B[i];

		init_again(A, B);

		#pragma omp target if (N>THRESHOLD) map(to: A[:N], B[:N])
		#pragma omp parallel for
		for (i=0; i<N; i++)
			C[i] += A[i] * B[i];
	}
}

void compareResults(DATA_TYPE* B, DATA_TYPE* B_GPU)
{
  int i, fail;
  fail = 0;
	
  // Compare B and B_GPU
  for (i=0; i < N; i++) 
    {
		if(B[i] != B_GPU[i]) printf("DIFF @ %d![%f, %f]\n", i, B[i], B_GPU[i]);
	  if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
	
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
	
}

int main(int argc, char *argv[])
{
  double t_start, t_end, t_start_OMP, t_end_OMP;

  DATA_TYPE* A;
  DATA_TYPE* B;  
  DATA_TYPE* C;  
  DATA_TYPE* C_OMP;
	
  A = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  C_OMP = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

  fprintf(stdout, ">> Two vector multiplication <<\n");

  //initialize the arrays
  init(A, B);

  t_start_OMP = rtclock();
  vec_mult_OMP(A, B, C_OMP);
  t_end_OMP = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end_OMP - t_start_OMP);//);
	

  //initialize the arrays
  init(A, B);

  t_start = rtclock();
  vec_mult(A, B, C);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);//);
	
  compareResults(C, C_OMP);

  free(A);
  free(B);
  free(C);
  free(C_OMP);
	
  return 0;
}
