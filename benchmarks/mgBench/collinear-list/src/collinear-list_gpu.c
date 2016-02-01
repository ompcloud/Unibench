/*
    This program checks the collinearity of points.
    It receives an input a vector with points and returns the mathematical functions that pass these points. It have a list to store answers.
    This program create a csv file with the time execution results for each function(CPU,GPU) in this format: size of vector, cpu with list time, gpu with list time.
    
    Author: Gleison Souza Diniz Mendon?a 
    Date: 04-05-2015
    version 2.0
    
    Run:
    folder_ipmacc/ipmacc folder_archive/colinear_v2.c
    ./a.out
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

#define SIZE 1024
#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

typedef struct point
{
	int x;
	int y;
} point;

point *points;

void generate_points()
{
    int i;
    for(i=0;i<SIZE;i++)
    {
        points[i].x = (i*777)%11;
        points[i].y = (i*777)%13;
    }
}

/// colinear list algorithm GPU
/// N = size of vector
int colinear_list_points_GPU()
{
    int i,j,k,p,val;
    val = 0;
    p = 10000;
	
    int *parallel_lines;
    parallel_lines = (int *) malloc(sizeof(int)*p);
    for(i=0;i<p;i++)
    {
        parallel_lines[i] = 0;	
    }

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: points[0:SIZE]) map(tofrom: parallel_lines[0:p])	
    {
        #pragma omp parallel for collapse(3)
	for(i = 0; i < SIZE; i++)
	{
	    for(j = 0; j < SIZE; j++)
	    {
	        for(k = 0; k < SIZE; k++)
		{
		    /// to understand if is colinear points
		    int slope_coefficient,linear_coefficient;
		    int ret;
		    ret = 0;
		    slope_coefficient = points[j].y - points[i].y;
		    if((points[j].x - points[i].x)!=0)
		    {
		        slope_coefficient = slope_coefficient / (points[j].x - points[i].x);
			linear_coefficient = points[i].y - (points[i].x * slope_coefficient);
		        if(slope_coefficient!=0 &&linear_coefficient!=0
                            &&points[k].y == (points[k].x * slope_coefficient) + linear_coefficient)
			{
        		    ret = 1;
			}
		    }
		    if(ret==1)
		    {
		        parallel_lines[(i%p)] = 1;
		    }
		}
	    }
    	}
    }

    val = 0;
    for(i=0;i<p;i++)
    {
        if(parallel_lines[i]==1)
	{
	    val = 1;
	    break;    
	}
    }
   
    free(parallel_lines);
    
    return val;
}

int colinear_list_points_CPU()
{
	
    int i,j,k,val;
    val = 0;
	
    for(i = 0; i < SIZE; i++)
    {
	for(j = 0; j < SIZE; j++)
	{
    	    for(k = 0; k < SIZE; k++)
	    {
		/// to understand if is colinear points
		int slope_coefficient,linear_coefficient;
		int ret;
		ret = 0;
		slope_coefficient = points[j].y - points[i].y;
				
		if((points[j].x - points[i].x)!=0)
		{
	    	    slope_coefficient = slope_coefficient / (points[j].x - points[i].x);
		    linear_coefficient = points[i].y - (points[i].x * slope_coefficient);
		    
                    if(slope_coefficient!=0
		       &&linear_coefficient!=0
		       &&points[k].y == (points[k].x * slope_coefficient) + linear_coefficient)
		    {
			ret = 1;
    		    }
		}
		/// to list add
		if(ret==1)
		{
		    val = 1;
		}
	    }
        }
    }

    return val;
}

void compareResults(int A, int A_outputFromGpu)
{
    int i, j, fail;
    fail = 0;
    if(A != A_outputFromGpu) fail = 1;

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


int main(int argc, char *argv[])
{
    double t_start, t_end;
    int result_CPU, result_GPU;   
 
    fprintf(stdout, "<< Collinear List >>\n");

    points = (point *) malloc(sizeof(points)*SIZE);
    generate_points();

    t_start = rtclock();
    result_GPU = colinear_list_points_GPU();
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	


    t_start = rtclock();
    result_CPU = colinear_list_points_CPU();
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	

    compareResults(result_GPU, result_CPU);

    free(points);
    
    return 0;
}
