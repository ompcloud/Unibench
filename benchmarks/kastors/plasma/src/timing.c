/**
 *
 * @file timing.c
 *
 *  PLASMA auxiliary routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Mathieu Faverge
 * @author Dulceneia Becker
 * @date 2010-11-15
 *
 **/

 #define _FMULS FMULS_GEQRF( M, N )
#define _FADDS FADDS_GEQRF( M, N )

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PLASMA_EZTRACE
#include <eztrace.h>
#endif

#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#include "plasma.h"
#include "core_blas.h"
#include "flops.h"
#include "timing.h"
#include "auxiliary.h"
#include "main.h"
#include "workspace.h"
#include "Utils.h"
#define EPSILON 1.0E-9

static double RunTest(double *t_, struct user_parameters*);

static double
RunTest_dpotrf(real_Double_t *t_, struct user_parameters* params)
{
    double  t;
    int64_t N     = params->matrix_size;
    int64_t NB    = params->blocksize;
    int check     = params->check;
    int uplo = PlasmaUpper;
    double check_res = 0;

    /* Allocate Data */
    PLASMA_desc *descA = NULL;
    double* ptr = malloc(N * N * sizeof(double));
    PLASMA_Desc_Create(&descA, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, N, 0, 0, N, N);

#pragma omp parallel
#pragma omp master
    plasma_pdplgsy_quark( (double)N, *descA, 51 );

    /* Save A for check */
    double *A = NULL;
    if(check) {
        A = (double*)malloc(N * N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descA, (void*)A, N);
    }

    double t_start, t_end;

    //START_TIMING();
    t_start = rtclock();
#pragma omp parallel
#pragma omp master
    plasma_pdpotrf_quark(uplo, *descA);
    //STOP_TIMING();
    t_end = rtclock();

    *t_ = t_end - t_start;

    /* Check the solution */
    if ( check )
    {
        PLASMA_desc *descB = NULL;
        double* ptr = (double*)malloc(N * sizeof(double));
        PLASMA_Desc_Create(&descB, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, 1, 0, 0, N, 1);

        plasma_pdpltmg_seq(* descB, 7672 );
        double* B = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)B, N);

        PLASMA_dpotrs_Tile( uplo, descA, descB );

        double* X = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)X, N);

        check_res = d_check_solution(N, N, 1, A, N, B, X, N);

        PASTE_CODE_FREE_MATRIX( descB );
        free( A );
        free( B );
        free( X );
    }

    PASTE_CODE_FREE_MATRIX( descA );

    return check_res;
}

static double
RunTest_dgetrf(real_Double_t *t_, struct user_parameters* params)
{
    double  t;
    int64_t N     = params->matrix_size;
    int64_t NB    = params->blocksize;
    int check     = params->check;
    double check_res = 0;

    /* Allocate Data */
    PLASMA_desc *descA = NULL;
    double* ptr = malloc(N * N * sizeof(double));
    PLASMA_Desc_Create(&descA, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, N, 0, 0, N, N);

    int* piv = (int*)malloc(N * sizeof(double));

#pragma omp parallel
#pragma omp master
    plasma_pdpltmg_quark(*descA, 3456);

    /* Save AT in lapack layout for check */
    double *A = NULL;
    if(check) {
        A = (double*)malloc(N * N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descA, (void*)A, N);
    }

    double t_start, t_end;

    //START_TIMING();
    t_start = rtclock();
#pragma omp parallel
#pragma omp master
    plasma_pdgetrf_rectil_quark(*descA, piv);
    //STOP_TIMING();
    t_end = rtclock();

    *t_ = t_end - t_start;

    /* Check the solution */
    if ( check )
    {
        PLASMA_desc *descB = NULL;
        double* ptr = (double*)malloc(N * sizeof(double));
        PLASMA_Desc_Create(&descB, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, 1, 0, 0, N, 1);

        plasma_pdpltmg_seq(*descB, 7732 );
        double* b = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)b, N);

        PLASMA_dgetrs_Tile( PlasmaNoTrans, descA, piv, descB );

        double* x = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)x, N);

        check_res = d_check_solution(N, N, 1, A, N, b, x, N);

        PASTE_CODE_FREE_MATRIX( descB );
        free(A); free(b); free(x);
    }

    PASTE_CODE_FREE_MATRIX( descA );
    free( piv );

    return check_res;
}

static double
RunTest_dgeqrf(real_Double_t *t_, struct user_parameters* params)
{
    double t;
    PLASMA_desc *descT;
    int64_t N     = params->matrix_size;
    int64_t IB    = params->iblocksize;
    int64_t NB    = params->blocksize;
    int check     = params->check;
    double check_res = 0;

    /* Allocate Data */
    PLASMA_desc *descA = NULL;
    double *ptr = (double*)malloc(N * N * sizeof(double));
    PLASMA_Desc_Create(&descA, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, N, 0, 0, N, N);

#pragma omp parallel
#pragma omp master
    plasma_pdpltmg_quark(*descA, 5373 );

    /* Save A for check */
    double *A = NULL;
    if ( check ) {
        A = (double*)malloc(N * N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descA, (void*)A, N);
    }

    /* Allocate Workspace */
    plasma_alloc_ibnb_tile(N, N, PlasmaRealDouble, &descT, IB, NB);


    double t_start, t_end;

    //START_TIMING();
    t_start = rtclock();
#pragma omp parallel
#pragma omp master
    plasma_pdgeqrf_quark( *descA, *descT , IB);
    //STOP_TIMING();
    t_end = rtclock();

    *t_ = t_end - t_start;

    /* Check the solution */
    if ( check )
    {
        /* Allocate B for check */
        PLASMA_desc *descB = NULL;
        double* ptr = (double*)malloc(N * sizeof(double));
        PLASMA_Desc_Create(&descB, ptr, PlasmaRealDouble, NB, NB, NB*NB, N, 1, 0, 0, N, 1);

        /* Initialize and save B */
        plasma_pdpltmg_seq(*descB, 2264 );
        double *B = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)B, N);

        /* Compute the solution */
        PLASMA_dgeqrs_Tile( descA, descT, descB , IB);

        /* Copy solution to X */
        double *X = (double*)malloc(N * sizeof(double));
        plasma_pdtile_to_lapack_quark(*descB, (void*)X, N);

        check_res = d_check_solution(N, N, 1, A, N, B, X, N);

        /* Free checking structures */
        PASTE_CODE_FREE_MATRIX( descB );
        free( A );
        free( B );
        free( X );
    }

    /* Free data */
    PLASMA_Dealloc_Handle_Tile(&descT);
    PASTE_CODE_FREE_MATRIX( descA );

    return check_res;
}

int ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */


double run(struct user_parameters* params)
{
    double   t;
    double   fmuls, fadds;
    double   flops;

    params->succeed = 1;
    int type = params->type;
    if (params->matrix_size <= 0) {
        params->matrix_size = 2048;
    }

    if (params->blocksize <= 0) {
        params->blocksize = 128;
    }
//ifdef
    if (params->iblocksize <= 0) {
        params->iblocksize = params->blocksize;
    }

    if (params->type <= 0) {
        params->type = 1;
	type = params->type;
    }


    int64_t N    = params->matrix_size;
    int64_t M    = params->matrix_size;
    fadds = (double)(_FADDS);
    fmuls = (double)(_FMULS);
    flops = 1e-9 * (fmuls + fadds);
    //mod aqui

    if (type == 1) {
      if (RunTest_dgeqrf(&t, params) > EPSILON && params->check)
          params->succeed = 0;
    } else if (type == 2) {
      if (RunTest_dgetrf(&t, params) > EPSILON && params->check)
          params->succeed = 0;
    } else if (type == 3) {
      if (RunTest_dpotrf(&t, params) > EPSILON && params->check)
          params->succeed = 0;
    } 

    // return gflops
    return flops / t;
}
