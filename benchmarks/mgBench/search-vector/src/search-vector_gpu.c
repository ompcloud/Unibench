/*
    This program searche a values in unordered vector and returns if find or not
    This program create a csv file with the time execution results for each function(CPU,GPU) in this format: size of vector,cpu time,gpu time.
    
    Author: Kezia Andrade
    Date: 04-07-2015
    version 1.0
    
    Run:
    folder_ipmacc/ipmacc folder_archive/search_in_vector.c
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

#define PERCENT_DIFF_ERROR_THRESHOLD 0.01


void init(float *a) 
{
	int i;
	for (i = 0; i < SIZE; ++i)
	{
        	a[i] = 2*i+7;
	}
}

int search_GPU(float *a, float c)
{
    int i;
    int find = -1;
	int *find2;

	find2 = &find;

    #pragma omp target map(to: a[:SIZE]) map(from: find2[:1]) device(DEVICE_ID)
    {
        #pragma omp parallel for
        for (i = 0; i < SIZE; ++i)
        {
	    if(a[i] == c)
	    {
	        find2[0] = i;
	        i=SIZE;
	    }
        }
    }

    return find;
}

int search_CPU(float *a, float c)
{
	int i;
    	int find = -1;

	for (i = 0; i < SIZE; ++i)
	{
        	if(a[i] == c)
        	{
            	    find = i;
	    	    i=SIZE;
        	}
	}

	return find;
}

int main(int argc, char *argv[]) {
    double t_start, t_end;
    float *a, c;
    int find_cpu, find_gpu;

    a = (float *) malloc(sizeof(float) * SIZE);
    c = (float) SIZE-5;

    init(a);

    fprintf(stdout, "<< Search Vector >>\n");

    t_start = rtclock();
    find_gpu = search_GPU(a, c);
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

    t_start = rtclock();
    find_cpu = search_CPU(a, c);
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	

    if(find_gpu == find_cpu)
        printf("Working %d=%d\n", find_gpu, find_cpu);
    else
	printf("Error %d != %d\n", find_gpu, find_cpu);

    free(a);

    return 0;
}

