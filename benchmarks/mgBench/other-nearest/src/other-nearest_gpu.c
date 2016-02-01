//<libmptogpu> Error executing kernel. Global Work Size is NULL or exceeded valid range.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include "../../common/mgbenchUtilFunctions.h"

typedef struct point
{
    int x;
    int y;
}point;

typedef struct sel_points
{
    int position;
    float value;
}sel_points;

#define SIZE 500
#define points 250
#define var SIZE/points
#define default_v 100000.00
#define GPU_DEVICE 0
#define ERROR_THRESHOLD 0.01

void init(int s, point *vector, sel_points *selected)
{
    int i,j;
    for(i=0;i<s;i++)
    {
        vector[i].x = i;
        vector[i].y = i*2;
    }
    for(i=0;i<s;i++)
    {
        for(j=0;j<s;j++)
        {
            selected[i*s+j].position = 0;
            selected[i*s+j].value = default_v;
        }
    }
}

void k_nearest_gpu(int s, point *vector, sel_points *selected)
{
    int i,j,m,q;
    q = s*s;

    #pragma omp target device (GPU_DEVICE)
    #pragma omp target map(to: vector[0: s]) map(tofrom: selected[0:q])
    {
	#pragma omp parallel for collapse(2) 
        for(i=0;i<s;i++)
        {
            for(j=i+1;j<s;j++)
            {
                float distance,x,y;
                x = vector[i].x - vector[j].x;
                y = vector[i].y - vector[j].y;
                x = x * x;
                y = y * y;
                
                distance = x + y;
                distance = sqrt(distance);
                
                selected[i*s+j].value = distance;
                selected[i*s+j].position = j;
                
                selected[j*s+i].value = distance;
                selected[j*s+i].position = i;
            }
        }
        
        /// for each line in matrix
        /// order values
	#pragma omp parallel for collapse(1)        
	for(i=0;i<s;i++)
        {
            for(j=0;j<s;j++)
            {
                for(m=j+1;m<s;m++)
                {
                    if(selected[i*s+j].value>selected[i*s+m].value)
                    {
                        sel_points aux;
                        aux = selected[i*s+j];
                        selected[i*s+j] = selected[i*s+m];
                        selected[i*s+m] = aux;
                    }
                } 
               
            }
        }
    }
}


void k_nearest_cpu(int s, point *vector, sel_points *selected)
{
    int i,j;
    for(i=0;i<s;i++)
    {
        for(j=i+1;j<s;j++)
        {
            float distance,x,y;
            x = vector[i].x - vector[j].x;
            y = vector[i].y - vector[j].y;
            x = x * x;
            y = y * y;
            
            distance = x + y;
            distance = sqrt(distance);
            
            selected[i*s+j].value = distance;
            selected[i*s+j].position = j;
            
            selected[j*s+i].value = distance;
            selected[j*s+i].position = i;
        }
    }
}

void order_points(int s, point *vector, sel_points *selected)
{
    int i;
    for(i=0;i<s;i++)
    {
        /// for each line in matrix
        /// order values
        int j;
        for(j=0;j<s;j++)
        {
            int m;
            for(m=j+1;m<s;m++)
            {
                if(selected[i*s+j].value>selected[i*s+m].value)
                {
                    sel_points aux;
                    aux = selected[i*s+j];
                    selected[i*s+j] = selected[i*s+m];
                    selected[i*s+m] = aux;
                }
            } 
        }
    }
}


void compareResults(sel_points* B, sel_points* B_GPU)
{
  int i, j, fail;
  fail = 0;
	
  // Compare B and B_GPU
  for (i=0; i < SIZE; i++) 
    {
      for (j=0; j < SIZE; j++) 
	{
	  //Value
	  if (percentDiff(B[i*SIZE + j].value, B_GPU[i*SIZE + j].value) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	  //Position
	  if (percentDiff(B[i*SIZE + j].position, B_GPU[i*SIZE + j].position) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
	
}

int main(int argc, char *argv[])
{
    double t_start, t_end;
    point *vector;
    sel_points *selected_cpu, *selected_gpu;

    vector = (point *) malloc(sizeof(point) * SIZE);
    selected_cpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);   
    selected_gpu = (sel_points *)malloc(sizeof(sel_points) * SIZE * SIZE);
     
    int i;
    
    fprintf(stdout, "<< Nearest >>\n");
    
    t_start = rtclock();
    for(i=(var-1);i<SIZE;i+=var)
    {
        init(i, vector, selected_cpu);
        k_nearest_cpu(i, vector, selected_cpu);
        order_points(i, vector, selected_cpu);
    }
    t_end = rtclock();
    fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);	


    t_start = rtclock();
    for(i=(var-1);i<SIZE;i+=var)
    {
        init(i, vector, selected_gpu);
        k_nearest_gpu(i, vector, selected_gpu);
    }
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

    compareResults(selected_cpu, selected_gpu);

    free(selected_cpu);
    free(selected_gpu);
    free(vector);
    return 0;
}

