/*
   
    This program calculates result of vector product.
    It generates a vector with the results of two vectors multiplication.
    This program create a csv file with the time execution results for each function(CPU,GPU) in this format: size of vector,cpu time,gpu time.
    
    Author: Kezia Andrade
    Date: 04-06-2015
    version 1.0
    
    Run:
    folder_ipmacc/ipmacc folder_archive/mat-mul-sun-acc.c
    ./a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#ifdef RUN_TEST
#define SIZE 1100
#elif RUN_BENCHMARK
#define SIZE 9600
#else
#define SIZE 1000
#endif

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
	
// Initialize matrices.
void init_arrays(float *A, float *B) 
{
    int i;
    for (i = 0; i < SIZE; ++i)
    {
       	A[i] = (float)i + 3*i;
       	B[i] = (float)i + 2*i;
    }
}

void product_GPU(float *A, float *B, float *C)
{
    int i;
    
    #pragma omp target map(to: A[0:SIZE], B[0:SIZE]) map(from: C[0:SIZE]) device(DEVICE_ID)
    {
	#pragma omp parallel for
        for (i = 0; i < SIZE; ++i)
	{
            C[i] = A[i] * B[i];
	}
    }
	
}

void product_CPU(float *A, float *B, float *C)
{
    int i;
     
    for (i = 0; i < SIZE; ++i)
    {
        C[i] = A[i] * B[i];
    }
}


int compareResults(float *A, float *A_outputFromGpu)
{
    int i, j, fail;
    fail = 0;

    for (i=0; i < SIZE; i++) 
    {
        if (percentDiff(A[i], A_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
	{				
	    fail++;
	    //printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
	}
    }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

  return fail;
}


int main(int argc, char *argv[]) {

    double t_start, t_end;	
    float *A, *B, *C_CPU, *C_GPU;
    int fail = 0;
 
    A = (float *) malloc(sizeof(float) * SIZE);
    B = (float *) malloc(sizeof(float) * SIZE);
    C_CPU = (float *) malloc(sizeof(float) * SIZE);
    C_GPU = (float *) malloc(sizeof(float) * SIZE);


    fprintf(stdout, "<< Vector Product >>\n");
    init_arrays(A, B);

    t_start = rtclock();
    product_GPU(A, B, C_GPU);
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

#ifdef RUN_TEST
    t_start = rtclock();
    product_CPU(A, B, C_CPU);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	

    fail = compareResults(C_CPU, C_GPU);
#endif

    free(A);
    free(B);
    free(C_CPU);
    free(C_GPU);

    return fail;
}

