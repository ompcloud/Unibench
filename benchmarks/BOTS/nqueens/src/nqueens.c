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

/*
 * Original code from the Cilk project (by Keith Randall)
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <alloca.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include "../../common/BOTSCommonUtils.h"


/* Checking information */

static int solutions[] = {
        1,
        0,
        0,
        2,
        10, /* 5 */
        4,
        40,
        92,
        352,
        724, /* 10 */
        2680,
        14200,
        73712,
        365596,
};
#define MAX_SOLUTIONS sizeof(solutions)/sizeof(int)

//#ifdef FORCE_TIED_TASKS
//int mycount=0;
//#pragma omp threadprivate(mycount)


int total_count, total_count_seq;

int if_cutoff, final_cutoff, manual_cutoff;

int cutoff_value =3;


/*
 * <a> contains array of <n> queen positions.  Returns 1
 * if none of the queens conflict, and returns 0 otherwise.
 */
int ok(int n, char *a)
{
     int i, j;
     char p, q;

     for (i = 0; i < n; i++) {
	  p = a[i];

	  for (j = i + 1; j < n; j++) {
	       q = a[j];
	       if (q == p || q == p - (j - i) || q == p + (j - i))
		    return 0;
	  }
     }
     return 1;
}

void nqueens_ser (int n, int j, char *a, int *solutions)
{
	int res;
	int i;

	if (n == j) {
		/* good solution, count it */
                	*solutions = 1;
                	return;
	}

	*solutions = 0;


     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
		{
	  		/* allocate a temporary array and copy <a> into it */
	  		a[j] = (char) i;
	  		if (ok(j + 1, a)) {
	       			nqueens_ser(n, j + 1, a,&res);
				*solutions += res;
			}
		}
	}
}

void nqueens_if(int n, int j, char *a, int *solutions, int depth)
{
	int *csols;
	int i;

	if (n == j) {
		/* good solution, count it */
		*solutions = 1;
		return;
	}
	*solutions = 0;
	csols = alloca(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
 		#pragma omp task untied if(depth < cutoff_value)
		{
	  		/* allocate a temporary array and copy <a> into it */
	  		char * b = alloca(n * sizeof(char));
	  		memcpy(b, a, j * sizeof(char));
	  		b[j] = (char) i;
	  		if (ok(j + 1, b))
       			nqueens_if(n, j + 1, b,&csols[i],depth+1);
		}
	}

	#pragma omp taskwait
	for ( i = 0; i < n; i++) *solutions += csols[i];
}


void nqueens_final(int n, int j, char *a, int *solutions, int depth)
{
	int *csols;
	int i;


	if (n == j) {
		/* good solution, count it */
		*solutions += 1;
		return;
	}


        char final = omp_in_final();
        if ( !final ) {
	  *solutions = 0;
	  csols = alloca(n*sizeof(int));
	  memset(csols,0,n*sizeof(int));
        }

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
 		#pragma omp task untied final(depth+1 >= cutoff_value) mergeable
		{
                        char *b;
                        int *sol;
			if ( omp_in_final() && depth+1 > cutoff_value ) {
		           b = a;
                           sol = solutions;
                        } else {
	  		/* allocate a temporary array and copy <a> into it */
	  		   b = alloca(n * sizeof(char));
	  		   memcpy(b, a, j * sizeof(char));
                           sol = &csols[i];
                        }
	  		b[j] = i;
	  		if (ok(j + 1, b))
       			nqueens_final(n, j + 1, b,sol,depth+1);
		}
	}

	#pragma omp taskwait
       if ( !final ) {
	for ( i = 0; i < n; i++) *solutions += csols[i];
       }
}

void nqueens_manual(int n, int j, char *a, int *solutions, int depth)
{
	int *csols;
	int i;


	if (n == j) {
		/* good solution, count it */
		*solutions = 1;
		return;
	}
	*solutions = 0;
	csols = alloca(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
		if ( depth < cutoff_value ) {
 			#pragma omp task untied
			{
	  			/* allocate a temporary array and copy <a> into it */
	  			char * b = alloca(n * sizeof(char));
	  			memcpy(b, a, j * sizeof(char));
	  			b[j] = (char) i;
	  			if (ok(j + 1, b))
       				nqueens_manual(n, j + 1, b,&csols[i],depth+1);
			}
		} else {
  			a[j] = (char) i;
  			if (ok(j + 1, a))
                		nqueens_ser(n, j + 1, a,&csols[i]);
		}
	}

	#pragma omp taskwait
	for ( i = 0; i < n; i++) *solutions += csols[i];
}

void nqueens(int n, int j, char *a, int *solutions, int depth)
{
	int *csols;
	int i;


	if (n == j) {
		/* good solution, count it */
		*solutions = 1;
		return;
	}

	*solutions = 0;
	csols = alloca(n*sizeof(int));
	memset(csols,0,n*sizeof(int));

     	/* try each possible position for queen <j> */
	for (i = 0; i < n; i++) {
 		#pragma omp task untied
		{
	  		/* allocate a temporary array and copy <a> into it */
	  		char * b = alloca(n * sizeof(char));
	  		memcpy(b, a, j * sizeof(char));
	  		b[j] = (char) i;
	  		if (ok(j + 1, b))
       				nqueens(n, j + 1, b,&csols[i],depth); //FIXME: depth or depth+1 ???
		}
	}

	#pragma omp taskwait
	for ( i = 0; i < n; i++) *solutions += csols[i];
}

void find_queens_seq (int size)
{
    total_count_seq = 0;

    fprintf(stdout,"Computing N-Queens algorithm (n=%d) ", size);

    char *a;
    a = alloca(size * sizeof(char));

    nqueens_ser(size, 0, a, &total_count_seq);

    fprintf(stdout," completed!\n");
}

void find_queens (int size)
{
	total_count=0;

            fprintf(stdout,"Computing N-Queens algorithm (n=%d) ", size);

            if (if_cutoff) {
                    #pragma omp parallel
                    {
                        #pragma omp single
                        {
                            char *a;

                            a = alloca(size * sizeof(char));
                            nqueens_if(size, 0, a, &total_count,0);
                        }
                    }
            }
            else if (manual_cutoff) {
                    #pragma omp parallel
                    {
                        #pragma omp single
                        {
                            char *a;

                            a = alloca(size * sizeof(char));
                            nqueens_manual(size, 0, a, &total_count,0);
                        }
                    }
            }
            else if (final_cutoff) {
                    #pragma omp parallel
                    {
                        #pragma omp single
                        {
                            char *a;

                            a = alloca(size * sizeof(char));
                            nqueens_final(size, 0, a, &total_count,0);
                        }
                    }
            }
            else {
                    #pragma omp parallel
                    {
                        #pragma omp single
                        {
                            char *a;

                            a = alloca(size * sizeof(char));
                            nqueens(size, 0, a, &total_count,0);
                        }
                    }
            }

	fprintf(stdout," completed!\n");
}


int verify_queens (int size)
{
	if ( size > MAX_SOLUTIONS ) return -1;
	if ( total_count == solutions[size-1] && total_count == total_count_seq) return 1;
	return 0;
}

void print_usage() {

   fprintf(stderr, "\n");
   fprintf(stderr, "Usage: %s -[options]\n", "N-Queens");
   fprintf(stderr, "\n");
   fprintf(stderr, "Where options are:\n");
   fprintf(stderr, "  -n <number>  : Board size\n");
   fprintf(stderr, "  -a <flag> : Set if-cutoff on\n");
   fprintf(stderr, "  -b <flag> : Set manual-cutoff on\n");
   fprintf(stderr, "  -c <flag> : Set final-cutoff on (choose one or none)\n");
   fprintf(stderr, "  -h         : Print program's usage (this help).\n");
   fprintf(stderr, "\n");

}


int main(int argc, char* argv[]) {

    int size = 14, i;

      for (i=1; i<argc; i++) {
        if (argv[i][0] == '-') {
          switch (argv[i][1]) {
              case 'n': /* read argument size 0 */
                     argv[i][1] = '*';
                     i++;
                     if (argc == i) { "Erro\n"; exit(100); }
                     size = atoi(argv[i]);
                     break;
              case 'a': /* read argument size 0 */
                     argv[i][1] = '*';
                     //i++;
                     //if (argc == i) { "Erro\n"; exit(100); }
                     if_cutoff = 1;
                     manual_cutoff = 0;
                     final_cutoff = 0;
                     break;
              case 'b': /* read argument size 0 */
                     argv[i][1] = '*';
                     //i++;
                     //if (argc == i) { "Erro\n"; exit(100); }
                     manual_cutoff = 1;
                     if_cutoff = 0;
                     final_cutoff = 0;
                     break;
              case 'c': /* read argument size 0 */
                     argv[i][1] = '*';
                     //i++;
                     //if (argc == i) { "Erro\n"; exit(100); }
                     final_cutoff = 1;
                     if_cutoff = 0;
                     manual_cutoff = 0;
                     break;
               case 'h': /* print usage */
                     argv[i][1] = '*';
                     print_usage();
                     exit (100);
                     break;
             }
        }
  }

    double t_start, t_end;

    t_start = rtclock();
    find_queens(size);
    t_end = rtclock();
    fprintf(stdout, "Parallel Runtime: %0.6lfs\n", t_end - t_start);

    t_start = rtclock();
    find_queens_seq(size);
    t_end = rtclock();
    fprintf(stdout, "Sequential Runtime: %0.6lfs\n", t_end - t_start);

  if (verify_queens(size) == 1) {
    fprintf(stdout, "Result: Successful\n");
  } else {
    fprintf(stdout, "Result: Unsuccessful\n");
  }

    return 0;
}