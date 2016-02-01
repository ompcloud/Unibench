/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/



void cpu_stencilGPU(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz);
void cpu_stencilCPU(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz);
