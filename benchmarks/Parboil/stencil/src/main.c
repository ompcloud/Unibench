
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include "../../common/parboil.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "file.h"
#include "common.h"
#include "kernels.h"

#include "../../common/polybenchUtilFuncts.h"

#define ERROR_THRESHOLD 0.05

double t_start, t_end, t_start_GPU, t_end_GPU;

float *h_Anext_GPU, *h_Anext_CPU;
int NX, NY, NZ;

typedef float DATA_TYPE;
struct pb_Parameters *parameters;

int compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU)
{
  int i, j, k, fail=0;

  for (k=0; k < NZ; k++) {
	for (j=0; j < NY; j++) {
		for (i=0; i < NX; i++) {
			if (percentDiff(A[Index3D (NX, NY, i, j, k)], A_GPU[Index3D (NX, NY, i, j, k)]) > ERROR_THRESHOLD) {
			 fail++;
		    }
		}
	}	
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);

  return fail;
}

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
	int s=0;
        int i, j, k;
	for(i=0;i<nz;i++)
	{
		for(j=0;j<ny;j++)
		{
			for(k=0;k<nx;k++)
			{
                                fread(A0+s,sizeof(float),1,fp);
				s++;
			}
		}
	}
	return 0;
}

double stencilGPU(int argc, char** argv) {
//	struct pb_TimerSet timers;
//	struct pb_Parameters *parameters;
	
//	printf("CPU-based 7 points stencil codes****\n");
//	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
//	printf("This version maintained by Chris Rodrigues  ***********\n");
	

//	pb_InitializeTimerSet(&timers);
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
    int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	//host data
	float *h_A0;
	float *h_Anext;

	size=nx*ny*nz;

	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
	read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  memcpy (h_Anext,h_A0 ,sizeof(float)*size);

  int t;
	t_start_GPU = rtclock();
	for(t=0;t<iteration;t++)
	{
		cpu_stencilGPU(c0,c1, h_A0, h_Anext, nx, ny,  nz);
	    float *temp=h_A0;
	    h_A0 = h_Anext;
	    h_Anext = temp;
	}
	t_end_GPU = rtclock();
  float *temp=h_A0;
  h_A0 = h_Anext;
  h_Anext_GPU = temp;
	NX = nx;
	NY = ny;
	NZ = nz;
/*	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}*/
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
//	free (h_Anext);
//	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//	pb_PrintTimerSet(&timers);
//	pb_FreeParameters(parameters);
	return t_end_GPU - t_start_GPU;
}

double stencilCPU(int argc, char** argv) {
//	struct pb_TimerSet timers;
//	struct pb_Parameters *parameters;
	

//	printf("CPU-based 7 points stencil codes****\n");
//	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
//	printf("This version maintained by Chris Rodrigues  ***********\n");
//	parameters = pb_ReadParameters(&argc, argv);

//	pb_InitializeTimerSet(&timers);
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
    int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	//host data
	float *h_A0;
	float *h_Anext;

	size=nx*ny*nz;

	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
  	FILE *fp = fopen(parameters->inpFiles[0], "rb");
	read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  memcpy (h_Anext,h_A0 ,sizeof(float)*size);

  int t;
	t_start = rtclock();
	for(t=0;t<iteration;t++)
	{
		cpu_stencilCPU(c0,c1, h_A0, h_Anext, nx, ny,  nz);
	    float *temp=h_A0;
	    h_A0 = h_Anext;
	    h_Anext = temp;
	}
	t_end = rtclock();

  float *temp=h_A0;
  h_A0 = h_Anext;
  h_Anext_CPU = temp;
 	NX = nx;
	NY = ny;
	NZ = nz;
/*	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}*/
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
//	free (h_Anext);
//	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//	pb_PrintTimerSet(&timers);
	
	return t_end - t_start;
}

int main(int argc, char** argv) {
  double t_GPU, t_CPU;
  int fail = 0;

	parameters = pb_ReadParameters(&argc, argv);

  t_GPU = stencilGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

#ifdef RUN_TEST
  t_CPU = stencilCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  fail = compareResults(h_Anext_GPU, h_Anext_CPU);
#endif

	pb_FreeParameters(parameters);
	free (h_Anext_GPU);
	free (h_Anext_CPU);

        return fail;

}
