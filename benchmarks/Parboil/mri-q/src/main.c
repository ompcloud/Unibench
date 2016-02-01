/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>

#include "../../common/parboil.h"

#include "file.h"
#include "computeQ.cc"

#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.1
#define GPU_DEVICE 1
double t_start, t_end, t_start_GPU, t_end_GPU;

float *Qr_GPU, *Qi_GPU;		/* Q signal (complex) */
float *Qr_CPU, *Qi_CPU;		/* Q signal (complex) */
int N;

typedef float DATA_TYPE;

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU, DATA_TYPE *B, DATA_TYPE *B_GPU)
{
  int i,fail=0;

  for (i=0; i < N; i++)
    {
	  if (percentDiff(A[i], A_GPU[i]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
    }

  for (i=0; i < N; i++)
    {
	  if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

double mriqGPU(int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  struct kValues* kVals;

  struct pb_Parameters *params;
//  struct pb_TimerSet timers;

//  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

//  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
//         numX, original_numK, numK);

//  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr_GPU, &Qi_GPU);

  ComputePhiMagCPU(numK, phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  #pragma omp parallel for
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  t_start_GPU = rtclock();
  ComputeQGPU(numK, numX, kVals, x, y, z, Qr_GPU, Qi_GPU);
  t_end_GPU = rtclock();

//  if (params->outFile)
//    {
      /* Write Q to file */
//      pb_SwitchToTimer(&timers, pb_TimerID_IO);
//      outputData(params->outFile, Qr_GPU, Qi_GPU, numX);
//      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
//    }

  N = numX;

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);

  return t_end_GPU - t_start_GPU;
/*  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);*/
}

double mriqCPU(int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  struct kValues* kVals;

  struct pb_Parameters *params;
//  struct pb_TimerSet timers;

//  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

//  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
//         numX, original_numK, numK);

//  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr_CPU, &Qi_CPU);

  ComputePhiMagCPU(numK, phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  #pragma omp parallel for
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  t_start = rtclock();
  ComputeQCPU(numK, numX, kVals, x, y, z, Qr_CPU, Qi_CPU);
  t_end = rtclock();

//  if (params->outFile)
//    {
      /* Write Q to file */
//      pb_SwitchToTimer(&timers, pb_TimerID_IO);
//      outputData(params->outFile, Qr_CPU, Qi_CPU, numX);
//      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
//    }

  N = numX;

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);

  return t_end - t_start;
/*  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);*/
}

int
main (int argc, char *argv[]) {
  double t_GPU, t_CPU;

  t_GPU = mriqGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = mriqCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(Qr_CPU, Qr_GPU, Qi_CPU, Qi_GPU);

	free(Qr_GPU);
	free(Qi_GPU);
	free(Qr_CPU);
	free(Qi_CPU);

  return 0;
}
