/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Base C implementation of MM
 */

#include <iostream>

void basicSgemmGPU( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

	int mm, nn, i;
	#pragma omp target device(1)
	#pragma omp target map(to: A[:m*k], B[:n*k]) map(tofrom: C[:m*n])
  #pragma omp parallel for
  for (mm = 0; mm < m; ++mm) {
    for (nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}

void basicSgemmCPU( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

	int mm, nn, i;
  for (mm = 0; mm < m; ++mm) {
    for (nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
