/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include "../../common/parboil.h"
#include <iostream>
#include "sgemm_kernel.cc"

#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1
double t_start, t_end, t_start_GPU, t_end_GPU;

float *matC_GPU, *matC_CPU;
int NX, NY;

typedef float DATA_TYPE;

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU)
{
  int i, j, fail=0;

  for (i=0; i < NX; i++)
    {
	  for (j=0; j < NY; j++)
		{
		  if (percentDiff(A[i*NY + j], A_GPU[i*NY + j]) > ERROR_THRESHOLD) 
		    {
			 fail++;
		    }
		}
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

double sgemmGPU(int argc, char *argv[]) {

  struct pb_Parameters *params;
//  struct pb_TimerSet timers;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

//  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
//  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

//  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  // allocate space for C
//  std::vector<float> matC(matArow*matBcol);
	matC_GPU = (float *)calloc(matArow*matBcol, sizeof(float));

  t_start_GPU = rtclock();
  // Use standard sgemm interface
  basicSgemmGPU('N', 'T', matArow, matBcol, matAcol, 1.0f,
      &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, matC_GPU,
      matArow);
  t_end_GPU = rtclock();

//	matC_GPU = &matC.front();
	NX = matArow;
	NY = matBcol;

//  if (params->outFile) {
    /* Write C to file */
//    pb_SwitchToTimer(&timers, pb_TimerID_IO);
//    writeColMajorMatrixFile(params->outFile, matArow, matBcol, matC); 
//  }

//  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//  double CPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_COMPUTE]));
//  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/CPUtime/1e9 << std::endl;
//  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return t_end_GPU - t_start_GPU;
}

double sgemmCPU(int argc, char *argv[]) {

  struct pb_Parameters *params;
//  struct pb_TimerSet timers;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

//  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
//  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

//  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  // allocate space for C
//  std::vector<float> matC(matArow*matBcol);
	matC_CPU = (float *)calloc(matArow*matBcol, sizeof(float));

  t_start = rtclock();
  // Use standard sgemm interface
  basicSgemmCPU('N', 'T', matArow, matBcol, matAcol, 1.0f,
      &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, matC_CPU,
      matArow);
  t_end = rtclock();

//	matC_GPU = &matC.front();
	NX = matArow;
	NY = matBcol;

//  if (params->outFile) {
    /* Write C to file */
//    pb_SwitchToTimer(&timers, pb_TimerID_IO);
//    writeColMajorMatrixFile(params->outFile, matArow, matBcol, matC); 
//  }

//  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//  double CPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_COMPUTE]));
//  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/CPUtime/1e9 << std::endl;
//  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return t_end - t_start;
}

int
main (int argc, char *argv[]) {
  double t_GPU, t_CPU;

  t_GPU = sgemmGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = sgemmCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(matC_GPU, matC_CPU);

  return 0;
}
