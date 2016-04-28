/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include "../../common/BOTSCommonUtils.h"
#include "sparselu.h"

int size = 50, subsize = 100;

/***********************************************************************
 * checkmat:
 **********************************************************************/
int checkmat (float *M, float *N)
{
   int i, j;
   float r_err;

   for (i = 0; i < subsize; i++)
   {
      for (j = 0; j < subsize; j++)
      {
         r_err = M[i*subsize+j] - N[i*subsize+j];
         if ( r_err == 0.0 ) continue;

         if (r_err < 0.0 ) r_err = -r_err;

         if ( M[i*subsize+j] == 0 )
         {
           fprintf(stdout, "Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n",
                    i,j, M[i*subsize+j], i,j, N[i*subsize+j]);
           return 0;
         }
         r_err = r_err / M[i*subsize+j];
         if(r_err > EPSILON)
         {
            fprintf(stdout, "Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
                    i,j, M[i*subsize+j], i,j, N[i*subsize+j], r_err);
            return 0;
         }
      }
   }
   return 1;
}
/***********************************************************************
 * genmat:
 **********************************************************************/
void genmat (float *M[])
{
   int null_entry, init_val, i, j, ii, jj;
   float *p;

   init_val = 1325;

   /* generating the structure */
   for (ii=0; ii < size; ii++)
   {
      for (jj=0; jj < size; jj++)
      {
         /* computing null entries */
         null_entry=0;
         if ((ii<jj) && (ii%3 !=0)) null_entry = 1;
         if ((ii>jj) && (jj%3 !=0)) null_entry = 1;
	 if (ii%2==1) null_entry = 1;
	 if (jj%2==1) null_entry = 1;
	 if (ii==jj) null_entry = 0;
	 if (ii==jj-1) null_entry = 0;
         if (ii-1 == jj) null_entry = 0;
         /* allocating matrix */
         if (null_entry == 0){
            M[ii*size+jj] = (float *) malloc(subsize*subsize*sizeof(float));
	    if ((M[ii*size+jj] == NULL))
            {
               fprintf(stdout, "Error: Out of memory\n");
               exit(101);
            }
            /* initializing matrix */
            p = M[ii*size+jj];
            for (i = 0; i < subsize; i++)
            {
               for (j = 0; j < subsize; j++)
               {
	            init_val = (3125 * init_val) % 65536;
      	            (*p) = (float)((init_val - 32768.0) / 16384.0);
                    p++;
               }
            }
         }
         else
         {
            M[ii*size+jj] = NULL;
         }
      }
   }
}
/***********************************************************************
 * print_structure:
 **********************************************************************/
void print_structure(char *name, float *M[])
{
   int ii, jj;
   fprintf(stdout, "Structure for matrix %s @ 0x%p\n",name, M);
   for (ii = 0; ii < size; ii++) {
     for (jj = 0; jj < size; jj++) {
        if (M[ii*size+jj]!=NULL) {fprintf(stdout, "x");}
        else fprintf(stdout, " ");
     }
     fprintf(stdout, "\n");
   }
   fprintf(stdout, "\n");
}
/***********************************************************************
 * allocate_clean_block:
 **********************************************************************/
float * allocate_clean_block()
{
  int i,j;
  float *p, *q;

  p = (float *) malloc(subsize*subsize*sizeof(float));
  q=p;
  if (p!=NULL){
     for (i = 0; i < subsize; i++)
        for (j = 0; j < subsize; j++){(*p)=0.0; p++;}

  }
  else
  {
      fprintf(stdout, "Error: Out of memory\n");
      exit (101);
  }
  return (q);
}

/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float *diag)
{
   int i, j, k;

   for (k=0; k<subsize; k++)
      for (i=k+1; i<subsize; i++)
      {
         diag[i*subsize+k] = diag[i*subsize+k] / diag[k*subsize+k];
         for (j=k+1; j<subsize; j++)
            diag[i*subsize+j] = diag[i*subsize+j] - diag[i*subsize+k] * diag[k*subsize+j];
      }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float *diag, float *row)
{
   int i, j, k;
   for (i=0; i<subsize; i++)
      for (k=0; k<subsize; k++)
      {
         row[i*subsize+k] = row[i*subsize+k] / diag[k*subsize+k];
         for (j=k+1; j<subsize; j++)
            row[i*subsize+j] = row[i*subsize+j] - row[i*subsize+k]*diag[k*subsize+j];
      }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float *row, float *col, float *inner)
{
   int i, j, k;
   for (i=0; i<subsize; i++)
      for (j=0; j<subsize; j++)
         for (k=0; k<subsize; k++)
            inner[i*subsize+j] = inner[i*subsize+j] - row[i*subsize+k]*col[k*subsize+j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float *diag, float *col)
{
   int i, j, k;
   for (j=0; j<subsize; j++)
      for (k=0; k<subsize; k++)
         for (i=k+1; i<subsize; i++)
            col[i*subsize+j] = col[i*subsize+j] - diag[i*subsize+k]*col[k*subsize+j];
}


void sparselu_init (float ***pBENCH, char *pass)
{
   *pBENCH = (float **) malloc(size*size*sizeof(float *));
   genmat(*pBENCH);
   print_structure(pass, *pBENCH);
}

void sparselu_par_call(float **BENCH)
{
   int ii, jj, kk;

   fprintf(stdout, "Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
           size,size,subsize,subsize);
#pragma omp parallel
#pragma omp single nowait
#pragma omp task untied
   for (kk=0; kk<size; kk++)
   {
      lu0(BENCH[kk*size+kk]);
      for (jj=kk+1; jj<size; jj++)
         if (BENCH[kk*size+jj] != NULL)
            #pragma omp task untied firstprivate(kk, jj) shared(BENCH)
         {
            fwd(BENCH[kk*size+kk], BENCH[kk*size+jj]);
         }
      for (ii=kk+1; ii<size; ii++)
         if (BENCH[ii*size+kk] != NULL)
            #pragma omp task untied firstprivate(kk, ii) shared(BENCH)
         {
            bdiv (BENCH[kk*size+kk], BENCH[ii*size+kk]);
         }

      #pragma omp taskwait

      for (ii=kk+1; ii<size; ii++)
         if (BENCH[ii*size+kk] != NULL)
            for (jj=kk+1; jj<size; jj++)
               if (BENCH[kk*size+jj] != NULL)
               #pragma omp task untied firstprivate(kk, jj, ii) shared(BENCH)
               {
                     if (BENCH[ii*size+jj]==NULL) BENCH[ii*size+jj] = allocate_clean_block();
                     bmod(BENCH[ii*size+kk], BENCH[kk*size+jj], BENCH[ii*size+jj]);
               }

      #pragma omp taskwait
   }
   fprintf(stdout, " completed!\n");
}


void sparselu_seq_call(float **BENCH)
{
   int ii, jj, kk;

   for (kk=0; kk<size; kk++)
   {
      lu0(BENCH[kk*size+kk]);
      for (jj=kk+1; jj<size; jj++)
         if (BENCH[kk*size+jj] != NULL)
         {
            fwd(BENCH[kk*size+kk], BENCH[kk*size+jj]);
         }
      for (ii=kk+1; ii<size; ii++)
         if (BENCH[ii*size+kk] != NULL)
         {
            bdiv (BENCH[kk*size+kk], BENCH[ii*size+kk]);
         }
      for (ii=kk+1; ii<size; ii++)
         if (BENCH[ii*size+kk] != NULL)
            for (jj=kk+1; jj<size; jj++)
               if (BENCH[kk*size+jj] != NULL)
               {
                     if (BENCH[ii*size+jj]==NULL) BENCH[ii*size+jj] = allocate_clean_block();
                     bmod(BENCH[ii*size+kk], BENCH[kk*size+jj], BENCH[ii*size+jj]);
               }

   }
}

void sparselu_fini (float **BENCH, char *pass)
{
   print_structure(pass, BENCH);
}

int sparselu_check(float **SEQ, float **BENCH)
{
   int ii,jj,ok=1;

   for (ii=0; ((ii<size) && ok); ii++)
   {
      for (jj=0; ((jj<size) && ok); jj++)
      {
         if ((SEQ[ii*size+jj] == NULL) && (BENCH[ii*size+jj] != NULL)) ok = 0;
         if ((SEQ[ii*size+jj] != NULL) && (BENCH[ii*size+jj] == NULL)) ok = 0;
         if ((SEQ[ii*size+jj] != NULL) && (BENCH[ii*size+jj] != NULL))
            ok = checkmat(SEQ[ii*size+jj], BENCH[ii*size+jj]);
      }
   }
   return ok;
}

void print_usage() {

   fprintf(stderr, "\n");
   fprintf(stderr, "Usage: %s -[options]\n", "SparseLU");
   fprintf(stderr, "\n");
   fprintf(stderr, "Where options are:\n");
   fprintf(stderr, "  -n <number>  :  Set matrix size\n");
   fprintf(stderr, "  -m <number>  :  Set blocks size\n");
   fprintf(stderr, "  -h         : Print program's usage (this help).\n");

}

int main(int argc, char* argv[]) {
  float **BENCH, **SEQ;
  double t_start, t_end;
  int i;

  for (i=1; i<argc; i++) {
        if (argv[i][0] == '-') {
          switch (argv[i][1]) {
            case 'n': /* read argument size 0 */
                   argv[i][1] = '*';
                   i++;
                   if (argc == i) { "Erro\n"; exit(100); }
                   size = atoi(argv[i]);
                   break;
            case 'm': /* read argument size 0 */
                   argv[i][1] = '*';
                   i++;
                   if (argc == i) { "Erro\n"; exit(100); }
                   subsize = atoi(argv[i]);
                   break;
                 case 'h': /* print usage */
                   argv[i][1] = '*';
                   print_usage();
                   exit (100);
                   break;
             }
        }
  }


  sparselu_init(&BENCH,"benchmark");
  t_start = rtclock();
  sparselu_par_call(BENCH);
  t_end = rtclock();
  sparselu_fini(BENCH, "benchmark");
  fprintf(stdout, "Parallel Runtime: %0.6lfs\n", t_end - t_start);

  sparselu_init(&SEQ,"benchmark");
  t_start = rtclock();
  sparselu_seq_call(SEQ);
  t_end = rtclock();
  sparselu_fini(SEQ, "benchmark");
  fprintf(stdout, "Sequential Runtime: %0.6lfs\n", t_end - t_start);

  if (sparselu_check(SEQ, BENCH)) {
    fprintf(stdout, "Result: Successful\n");
  } else {
    fprintf(stdout, "Result: Unsuccessful\n");
  }

  return 0;
}