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
#include <sys/time.h>

#include "file.h"
#include "convert_dataset.h"

#include "../../common/polybenchUtilFuncts.h"


#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1
double t_start, t_end, t_start_GPU, t_end_GPU;

float *h_Ax_vector_GPU, *h_Ax_vector_CPU;
int N;

typedef float DATA_TYPE;

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU)
{
  int i, fail=0;

  for (i=0; i < N; i++)
    {
	  if (percentDiff(A[i], A_GPU[i]) > ERROR_THRESHOLD) 
	    {
		 fail++;
	    }
    }
	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);	
  int i;
	for(i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}

/*
void jdsmv(int height, int len, float* value, int* perm, int* jds_ptr, int* col_index, float* vector,
        float* result){
        int i;
        int col,row;
        int row_index =0;
        int prem_indicator=0;
        for (i=0; i<len; i++){
                if (i>=jds_ptr[prem_indicator+1]){
                        prem_indicator++;
                        row_index=0;
                }
                if (row_index<height){
                col = col_index[i];
                row = perm[row_index];
                result[row]+=value[i]*vector[col];
                }

                row_index++;
        }
        return;
}
*/

double spmvGPU(int argc, char** argv) {
//	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
//	printf("CPU-based sparse matrix vector multiplication****\n");
//	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
//	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }

	
//	pb_InitializeTimerSet(&timers);
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	int len;
	int depth;
	int dim;
	int pad=1;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
    float *h_x_vector;
	
	
    //load matrix from files
//	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);

 

	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		0, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);		

  h_Ax_vector=(float*)malloc(sizeof(float)*dim);
  h_x_vector=(float*)malloc(sizeof(float)*dim);
//  generate_vector(h_x_vector, dim);
  input_vec( parameters->inpFiles[1],h_x_vector,dim);

	
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	
  int p, i;
	t_start_GPU = rtclock();
	//main execution
	#pragma omp target device(1)
	#pragma omp target map(to: h_nzcnt[:nzcnt_len], h_ptr[:col_count], h_indices[:len], h_data[:len], h_perm[:col_count], h_x_vector[:dim]) map(from: h_Ax_vector[:dim])
	for(p=0;p<50;p++)
	{
    #pragma omp parallel for
		for (i = 0; i < dim; i++) {
      int k;
		  float sum = 0.0f;
		  //int  bound = h_nzcnt[i / 32];
		  int  bound = h_nzcnt[i];
		  for(k=0;k<bound;k++ ) {
			int j = h_ptr[k] + i;
			int in = h_indices[j];

			float d = h_data[j];
			float t = h_x_vector[in];

			sum += d*t;
		  }
    //  #pragma omp critical 
		  h_Ax_vector[h_perm[i]] = sum;
		}
	}
	t_end_GPU = rtclock();

	h_Ax_vector_GPU = h_Ax_vector;
	N = dim;

//	if (parameters->outFile) {
//		pb_SwitchToTimer(&timers, pb_TimerID_IO);
//		outputData(parameters->outFile,h_Ax_vector,dim);
		
//	}
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_x_vector);
//	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);
  return t_end_GPU - t_start_GPU;
}

double spmvCPU(int argc, char** argv) {
//	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
//	printf("CPU-based sparse matrix vector multiplication****\n");
//	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
//	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }

	
//	pb_InitializeTimerSet(&timers);
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	int len;
	int depth;
	int dim;
	int pad=1;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
    float *h_x_vector;
	
	
    //load matrix from files
//	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);

 

	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		0, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);		

  h_Ax_vector=(float*)malloc(sizeof(float)*dim);
  h_x_vector=(float*)malloc(sizeof(float)*dim);
//  generate_vector(h_x_vector, dim);
  input_vec( parameters->inpFiles[1],h_x_vector,dim);

	
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	
  int p, i;
	//main execution
	t_start = rtclock();
	for(p=0;p<50;p++)
	{
		for (i = 0; i < dim; i++) {
      int k;
		  float sum = 0.0f;
		  //int  bound = h_nzcnt[i / 32];
		  int  bound = h_nzcnt[i];
		  for(k=0;k<bound;k++ ) {
			int j = h_ptr[k] + i;
			int in = h_indices[j];

			float d = h_data[j];
			float t = h_x_vector[in];

			sum += d*t;
		  }
    //  #pragma omp critical 
		  h_Ax_vector[h_perm[i]] = sum;
		}
	}
	t_end = rtclock();

	h_Ax_vector_CPU = h_Ax_vector;
	N = dim;

//	if (parameters->outFile) {
//		pb_SwitchToTimer(&timers, pb_TimerID_IO);
//		outputData(parameters->outFile,h_Ax_vector,dim);
		
//	}
//	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_x_vector);
//	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

//	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);
  return t_end - t_start;
}

int main(int argc, char** argv) {
  double t_GPU, t_CPU;

  t_GPU = spmvGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = spmvCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(h_Ax_vector_GPU, h_Ax_vector_CPU);

	free (h_Ax_vector_GPU);
	free (h_Ax_vector_CPU);

	return 0;

}
