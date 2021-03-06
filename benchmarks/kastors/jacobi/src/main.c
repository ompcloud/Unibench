#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "main.h"

#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)

void parse(int argc, char* argv[], struct user_parameters* params)
{
    int i;
    for(i=1; i<argc; i++) {
        if(!strcmp(argv[i], "-c"))
            params->check = 1;
        else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("----------------------------------------------\n");
            printf("-                KaStORS                     -\n");
            printf("-   Kaapi Starpu OpenMP Runtime task Suite   -\n");
            printf("----------------------------------------------\n");
            printf("-h, --help : Show help information\n");
            printf("-c : Ask to check result\n");
            printf("-i : Number of iterations\n");
            printf("-r : Number ot timestep iteration\n");
            printf("-n : Matrix size\n");
            printf("-b : Block size\n");
            printf("-t : Choose algorithm (leavng blank will run type task)\n(Options for type) 1 - task, 2 - task with depend, 3 - block-for, 4 - block with task, 5 - block with task and depend\n");
            exit(EXIT_SUCCESS);
        } else if(!strcmp(argv[i], "-i")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-i requires a number\n");
                exit(EXIT_FAILURE);
            }

        } else if(!strcmp(argv[i], "-r")) {
            if (++i < argc)
                params->titer = atoi(argv[i]);
            else {
                fprintf(stderr, "-r requires a number\n");
                exit(EXIT_FAILURE);
            }


        } else if(!strcmp(argv[i], "-n")) {
            if (++i < argc)
                params->matrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-n requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-b")) {
            if (++i < argc)
                params->blocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-b requires a number\n");
                exit(EXIT_FAILURE);
            }

        } else if(!strcmp(argv[i], "-t")) {
            if (++i < argc)
                params->type = atoi(argv[i]);
            else {
                fprintf(stderr, "-t requires a number\n");
                exit(EXIT_FAILURE);
            }

        } else
            fprintf(stderr, "Unknown parameter : %s\n", argv[i]);
    }
}

int comp (const void * elem1, const void * elem2)
{
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    struct user_parameters params;
    memset(&params, 0, sizeof(params));

    /* default value */
    params.niter = 1;

    parse(argc, argv, &params);

// get Number of thread if OpenMP is activated
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads();
#endif

    // warmup
    printf("Running Parallel code\n");
    run(&params);

    double mean = 0.0;
    double meansqr = 0.0;
    double min_ = DBL_MAX;
    double max_ = -1;
    double* all_times = (double*)malloc(sizeof(double) * params.niter);

    for (int i=0; i<params.niter; ++i)
    {
      double cur_time = run(&params);
      all_times[i] = cur_time;
      mean += cur_time;
      min_ = min(min_, cur_time);
      max_ = max(max_, cur_time);
      meansqr += cur_time * cur_time;
      }
    mean /= params.niter;
    meansqr /= params.niter;
    double stddev = sqrt(meansqr - mean * mean);

    qsort(all_times, params.niter, sizeof(double), comp);
    double median = all_times[params.niter / 2];

    free(all_times);

    printf("Parallel Stats\n");
    printf("Program : %s\n", argv[0]);
    printf("Size : %d\n", params.matrix_size);
    printf("Blocksize : %d\n", params.blocksize);
    printf("Iteration time : %d\n", params.titer);

    printf("Iterations : %d\n", params.niter);
#ifdef GFLOPS
    printf("Gflops:: ");
#else
    printf("Time(sec):: ");
#endif
    printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
           mean, stddev, min_, max_, median);
    if(params.check)
        printf("Check : %s\n", (params.succeed)?
                ((params.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params.string2display !=0)
      printf("%s", params.string2display);
    printf("\n");

    /* Rodar aqui o codigo sequencial run_seq*/
    printf("Running Sequential code\n");
    struct user_parameters params_seq;
    memset(&params_seq, 0, sizeof(params_seq));

    /* default value */
    params_seq.niter = params.niter;
    params_seq.matrix_size = params.matrix_size;
    params_seq.blocksize = params.blocksize;
    params_seq.titer = params.titer;
    params_seq.check = params.check;
    params_seq.type = 6;
    //strcpy(params_seq.string2display, params.string2display);
    params_seq.string2display = params.string2display;
    //parse(argc, argv, &params_seq);

    // warmup
    run(&params_seq);

    double mean_seq = 0.0;
    double meansqr_seq = 0.0;
    double min_seq = DBL_MAX;
    double max_seq = -1;
    double* all_times_seq = (double*)malloc(sizeof(double) * params_seq.niter);

    for (int i=0; i<params_seq.niter; ++i)
    {
      double cur_time = run(&params_seq);
      all_times_seq[i] = cur_time;
      mean_seq += cur_time;
      min_seq = min(min_, cur_time);
      max_seq = max(max_, cur_time);
      meansqr_seq += cur_time * cur_time;
      }
    mean_seq /= params_seq.niter;
    meansqr_seq /= params_seq.niter;
    double stddev_seq = sqrt(meansqr_seq - mean_seq * mean_seq);

    qsort(all_times_seq, params_seq.niter, sizeof(double), comp);
    double median_seq = all_times_seq[params_seq.niter / 2];

    free(all_times_seq);


    printf("Sequential Stats\n");
    printf("Program : %s\n", argv[0]);
    printf("Size : %d\n", params_seq.matrix_size);
    printf("Blocksize : %d\n", params_seq.blocksize);

    printf("Iteration time : %d\n", params_seq.titer);

    printf("Iterations : %d\n", params_seq.niter);
#ifdef GFLOPS
    printf("Gflops:: ");
#else
    printf("Time(sec):: ");
#endif
    printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
           mean_seq, stddev_seq, min_seq, max_seq, median_seq);
    if(params_seq.check)
        printf("Check : %s\n", (params_seq.succeed)?
                ((params_seq.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params_seq.string2display !=0)
      printf("%s", params_seq.string2display);
    printf("\n");

    return 0;
}
